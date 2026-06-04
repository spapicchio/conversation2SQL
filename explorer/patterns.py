"""Deterministic (no-LLM) tool-interaction anti-pattern detection for result traces.

Operates on the normalized tool-event sequence extracted from each record's
``messages``. All aggregates are sample-average: averaged per instance first, then
across instances, mirroring ``loader.pass_at_1`` and ``metrics.reliability_metrics``.

This module must not import from ``explorer.loader`` (would be circular); it carries
its own small message-text helper instead.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field

import pandas as pd


def _message_text(content: object) -> str:
    """Best-effort text of a tool message's content (dict or str)."""
    if isinstance(content, dict):
        for key in ("content", "message", "error"):
            value = content.get(key)
            if isinstance(value, str):
                return value
        return json.dumps(content, ensure_ascii=False)
    if isinstance(content, str):
        return content
    return ""


@dataclass(frozen=True)
class ToolEvent:
    message_index: int   # index into record["messages"] of the tool-result message
    tool_name: str
    arguments: dict
    status: str          # "success" or an error status
    is_error: bool       # status != "success"
    result_text: str     # tool message content as text


def extract_tool_events(record: dict) -> list[ToolEvent]:
    """Walk messages, pairing each ai tool_call with its following tool result(s).

    When an ai turn has K tool_calls, the next K tool messages are its results.
    Arguments are matched to a result by tool name, in order (handles multi-call
    turns and back-to-back tool messages).
    """
    events: list[ToolEvent] = []
    pending: list[tuple[str, dict]] = []  # (tool_name, arguments) awaiting a result
    for idx, msg in enumerate(record.get("messages", [])):
        role = msg.get("role")
        if role == "ai":
            pending = [
                (tc.get("tool_name", ""), tc.get("arguments", {}) or {})
                for tc in (msg.get("tool_calls") or [])
            ]
        elif role == "tool":
            name = msg.get("tool_name", "")
            args: dict = {}
            for j, (pname, pargs) in enumerate(pending):
                if pname == name:
                    args = pargs
                    pending.pop(j)
                    break
            status = msg.get("status", "")
            events.append(
                ToolEvent(
                    message_index=idx,
                    tool_name=name,
                    arguments=args,
                    status=status,
                    is_error=status != "success",
                    result_text=_message_text(msg.get("content")),
                )
            )
    return events


@dataclass(frozen=True)
class PatternHit:
    name: str                    # "blind_submit"
    label: str                   # "Blind submit"
    detail: str                  # human evidence string
    message_indices: list[int] = field(default_factory=list)


ERROR_LOOP_MIN = 3  # consecutive execute_sql errors that constitute an unrecovered loop


def _norm_sql(sql: object) -> str:
    return " ".join(sql.split()) if isinstance(sql, str) else ""


def _norm_args(arguments: dict) -> str:
    a = dict(arguments or {})
    if isinstance(a.get("sql"), str):
        a["sql"] = _norm_sql(a["sql"])
    return json.dumps(a, sort_keys=True, ensure_ascii=False)


def _detect_blind_submit(events: list[ToolEvent], record: dict) -> PatternHit | None:
    seen_execute = False
    for e in events:
        if e.tool_name == "execute_sql":
            seen_execute = True
        elif e.tool_name == "submit_sql":
            if not seen_execute:
                return PatternHit(
                    "blind_submit", "Blind submit",
                    "submitted without any prior execute_sql", [e.message_index],
                )
            return None  # first submission was validated
    return None


def _detect_repeated_identical_call(events: list[ToolEvent], record: dict) -> PatternHit | None:
    seen: dict[tuple[str, str], list[int]] = {}
    for e in events:
        seen.setdefault((e.tool_name, _norm_args(e.arguments)), []).append(e.message_index)
    for (name, _), idxs in seen.items():
        if len(idxs) >= 2:
            return PatternHit(
                "repeated_identical_call", "Repeated identical call",
                f"`{name}` called {len(idxs)}× with identical arguments", idxs,
            )
    return None


def _detect_submit_after_error(events: list[ToolEvent], record: dict) -> PatternHit | None:
    errored: dict[str, int] = {}
    for e in events:
        if e.tool_name == "execute_sql" and e.is_error:
            errored[_norm_sql(e.arguments.get("sql"))] = e.message_index
        elif e.tool_name == "submit_sql":
            key = _norm_sql(e.arguments.get("sql"))
            if key and key in errored:
                return PatternHit(
                    "submit_after_error", "Submit after error",
                    "submitted SQL is identical to an execute_sql that errored",
                    [errored[key], e.message_index],
                )
    return None


def _detect_unrecovered_error_loop(events: list[ToolEvent], record: dict) -> PatternHit | None:
    streak: list[int] = []
    for e in events:
        if e.tool_name != "execute_sql":
            continue
        if e.is_error:
            streak.append(e.message_index)
            if len(streak) >= ERROR_LOOP_MIN:
                return PatternHit(
                    "unrecovered_error_loop", "Unrecovered error loop",
                    f"{len(streak)} consecutive execute_sql errors", list(streak),
                )
        else:
            streak = []
    return None


def _kb_needed(record: dict) -> bool:
    """Whether the task genuinely requires external knowledge.

    True iff ``gt_knowledge_base`` (the KB entries the gold solution depends on)
    is non-empty. This — not the full browsable ``masked_agent_kb`` — is what
    makes the KB-blind pattern *applicable*: a task whose answer needs no KB
    entry cannot be "blind" to one.
    """
    return bool(record.get("gt_knowledge_base"))


def _detect_kb_blind(events: list[ToolEvent], record: dict) -> PatternHit | None:
    if not _kb_needed(record):
        return None
    if not any(e.tool_name == "get_knowledge_definition" for e in events):
        n = len(record.get("gt_knowledge_base") or {})
        return PatternHit(
            "kb_blind", "KB-blind",
            f"answer needs {n} KB entr{'y' if n == 1 else 'ies'} but "
            "get_knowledge_definition never called",
        )
    return None


def _detect_budget_death(events: list[ToolEvent], record: dict) -> PatternHit | None:
    if any(e.tool_name == "submit_sql" and not e.is_error for e in events):
        return None
    budget = record.get("updated_user_patience")
    exhausted = any("budget exhausted" in e.result_text.lower() for e in events) or (
        isinstance(budget, (int, float)) and budget <= 0
    )
    if exhausted:
        return PatternHit(
            "budget_death", "Budget death",
            "budget exhausted before a successful submit_sql",
        )
    return None


def _detect_no_submission(events: list[ToolEvent], record: dict) -> PatternHit | None:
    if not any(e.tool_name == "submit_sql" for e in events):
        return PatternHit(
            "no_submission", "No submission",
            "conversation ended with no submit_sql call",
        )
    return None


Detector = Callable[[list[ToolEvent], dict], "PatternHit | None"]

ANTI_PATTERNS: list[Detector] = [
    _detect_blind_submit,
    _detect_repeated_identical_call,
    _detect_submit_after_error,
    _detect_unrecovered_error_loop,
    _detect_kb_blind,
    _detect_budget_death,
    _detect_no_submission,
]

# (name, label) for display + enumeration independent of whether a detector fires.
PATTERN_CATALOG: list[tuple[str, str]] = [
    ("blind_submit", "Blind submit"),
    ("repeated_identical_call", "Repeated identical call"),
    ("submit_after_error", "Submit after error"),
    ("unrecovered_error_loop", "Unrecovered error loop"),
    ("kb_blind", "KB-blind"),
    ("budget_death", "Budget death"),
    ("no_submission", "No submission"),
]
PATTERN_NAMES: list[str] = [name for name, _ in PATTERN_CATALOG]

# Per-pattern "applicable" predicate: the samples on which a pattern *could* fire,
# i.e. its denominator. Patterns absent here are applicable to every sample. Only
# patterns with a restricted denominator need an entry. ``kb_blind`` reuses the
# same predicate as its detector so the two cannot drift.
PATTERN_APPLICABLE: dict[str, Callable[[dict], bool]] = {
    "kb_blind": _kb_needed,
}


def _is_applicable(name: str, record: dict) -> bool:
    pred = PATTERN_APPLICABLE.get(name)
    return pred(record) if pred is not None else True


def detect_patterns(record: dict) -> list[PatternHit]:
    events = extract_tool_events(record)
    return [hit for d in ANTI_PATTERNS if (hit := d(events, record)) is not None]


def _record_hits(record: dict) -> list[PatternHit]:
    """Use cached hits from loader if present, else compute."""
    cached = record.get("_pattern_hits")
    return cached if cached is not None else detect_patterns(record)


@dataclass(frozen=True)
class PatternStat:
    """A pattern's rate and the denominator it was computed over.

    ``rate`` is the sample-average among *applicable* samples (mean over instances
    of each instance's applicable-hit fraction); ``applicable_n`` is the total
    number of applicable samples backing it. For patterns applicable to every
    sample, ``rate`` matches the old frequency and ``applicable_n`` is all samples.
    Reporting both makes patterns with different denominators comparable — a 38%
    KB-blind rate over 120 KB-needing samples reads honestly next to a 12% rate
    over all 412 samples.
    """
    rate: float
    applicable_n: int


def aggregate_patterns(groups: dict[str, list[dict]]) -> dict[str, PatternStat]:
    """name -> PatternStat over the pattern's applicable samples.

    The rate is the sample-average (mean over instances of the fraction of that
    instance's *applicable* samples hitting the pattern); instances with no
    applicable sample are excluded from the mean. Per record a pattern counts at
    most once (presence, not occurrence count). ``applicable_n`` is the total
    applicable samples across all instances — the denominator behind the rate.
    """
    per_instance: dict[str, list[float]] = {n: [] for n in PATTERN_NAMES}
    applicable_n: dict[str, int] = {n: 0 for n in PATTERN_NAMES}
    for samples in groups.values():
        if not samples:
            continue
        present = [{h.name for h in _record_hits(r)} for r in samples]
        for name in PATTERN_NAMES:
            applicable = [i for i, r in enumerate(samples) if _is_applicable(name, r)]
            applicable_n[name] += len(applicable)
            if applicable:
                per_instance[name].append(
                    sum(name in present[i] for i in applicable) / len(applicable)
                )
    return {
        name: PatternStat(
            rate=(sum(v) / len(v) if v else 0.0),
            applicable_n=applicable_n[name],
        )
        for name, v in per_instance.items()
    }


def clean_fraction(groups: dict[str, list[dict]]) -> float:
    """Sample-average fraction of samples with zero anti-pattern hits."""
    per_instance: list[float] = []
    for samples in groups.values():
        if not samples:
            continue
        clean = sum(1 for r in samples if not _record_hits(r))
        per_instance.append(clean / len(samples))
    return sum(per_instance) / len(per_instance) if per_instance else 0.0


def positional_tool_distribution(
    groups: dict[str, list[dict]], max_pos: int = 8
) -> pd.DataFrame:
    """Long-form frame [position, tool, share] for a 100%-stacked positional plot.

    Columns 1..max_pos-1 are individual call positions; the last column ">=max_pos"
    shows the tool at position max_pos for still-active conversations. Each sample
    contributes exactly one category per position (a tool name, or "(no call)" once
    the sequence has ended), so shares per position sum to 1. Aggregated sample-average:
    per-instance share averaged over ALL instances (missing categories imply 0).
    """
    positions = [str(p) for p in range(1, max_pos)] + [f"≥{max_pos}"]
    n_inst = sum(1 for s in groups.values() if s)
    if n_inst == 0:
        return pd.DataFrame(columns=["position", "tool", "share"])

    accum: dict[str, dict[str, float]] = {pos: {} for pos in positions}
    for samples in groups.values():
        if not samples:
            continue
        counts: dict[str, Counter] = {pos: Counter() for pos in positions}
        for r in samples:
            seq = [e.tool_name for e in extract_tool_events(r)]
            for i, pos in enumerate(positions):
                cat = seq[i] if i < len(seq) else "(no call)"
                counts[pos][cat] += 1
        n = len(samples)
        for pos in positions:
            for cat, c in counts[pos].items():
                accum[pos][cat] = accum[pos].get(cat, 0.0) + (c / n)

    rows = [
        {"position": pos, "tool": cat, "share": total / n_inst}
        for pos in positions
        for cat, total in accum[pos].items()
    ]
    return pd.DataFrame(rows)
