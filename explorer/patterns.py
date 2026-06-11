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


# Verbatim prefix of the ToolMessage the budget middleware emits when it blocks
# a tool call (`tool_wrapper_patience_and_submit` in agent_callback.py). The
# message is *named after the blocked tool* with success status, but that tool
# never ran — events matching this prefix are pseudo-events, not real calls.
BUDGET_BLOCKED_PREFIX = "Budget exhausted"


@dataclass(frozen=True)
class ToolEvent:
    message_index: int   # index into record["messages"] of the tool-result message
    tool_name: str
    arguments: dict
    status: str          # "success" or an error status
    is_error: bool       # status != "success"
    result_text: str     # tool message content as text
    blocked: bool = False  # budget middleware blocked the call; the tool never ran


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
            result_text = _message_text(msg.get("content"))
            events.append(
                ToolEvent(
                    message_index=idx,
                    tool_name=name,
                    arguments=args,
                    status=status,
                    is_error=status != "success",
                    result_text=result_text,
                    blocked=result_text.startswith(BUDGET_BLOCKED_PREFIX),
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


def _validates_query(event: ToolEvent) -> bool:
    """True if this tool call ran the query against the DB (i.e. validated it).

    ``execute_sql`` always runs SQL. Under the ``psql_console`` ablation the query
    is run in the terminal instead, so a ``psql_console`` call validates too —
    *unless* the command is a backslash meta-command (``\\dt``, ``\\d`` …), which
    only inspects the schema and never runs the query. Detection mirrors
    ``_is_psql_meta_command`` in the tool: the first non-blank char being a
    backslash makes it a meta-command. A budget-blocked call never ran anything.
    """
    if event.blocked:
        return False
    if event.tool_name == "execute_sql":
        return True
    if event.tool_name == "psql_console":
        command = event.arguments.get("command")
        return isinstance(command, str) and not command.lstrip().startswith("\\")
    return False


def _detect_blind_submit(events: list[ToolEvent], record: dict) -> PatternHit | None:
    seen_validation = False
    for e in events:
        if _validates_query(e):
            seen_validation = True
        elif e.tool_name == "submit_sql":
            if not seen_validation:
                return PatternHit(
                    "blind_submit", "Blind submit",
                    "submitted without running the query against the DB first",
                    [e.message_index],
                )
            return None  # first submission was validated
    return None


def _detect_repeated_identical_call(events: list[ToolEvent], record: dict) -> PatternHit | None:
    seen: dict[tuple[str, str], list[int]] = {}
    for e in events:
        if e.blocked:  # the re-issue never executed
            continue
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
        # A budget-blocked pseudo-event neither extends nor resets the streak.
        if e.tool_name != "execute_sql" or e.blocked:
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
    # "Successful" means the answer passed, not that the submit tool ran without
    # erroring: a forced out-of-budget submit whose SQL fails has success status
    # but is still a budget death.
    if record.get("execution_accuracy"):
        return None
    budget = record.get("updated_user_patience")
    exhausted = any(e.blocked for e in events) or (
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


def _middleware_event_types(record: dict) -> set[str]:
    """The set of built-in-middleware activation types recorded for this run.

    Sourced from the ``middleware_events`` field the agent pipeline writes
    (``extract_middleware_events``). Records from runs predating that field carry
    no key — for the two *limit* patterns the trace fallback below still fires;
    ``context_editing`` has no trace footprint and so cannot fire on them.
    """
    return {e.get("type") for e in record.get("middleware_events") or []}


# Content prefixes the built-in limit middlewares emit into the message trace
# (mirrors TOOL_CALL_LIMIT_PREFIX / MODEL_CALL_LIMIT_PREFIX in the eval package's
# tools/__init__.py, which in turn mirror LangChain). Kept as a local copy — like
# colors.TOOL_COLORS mirrors TOOL_COSTS — so this analysis module stays free of the
# heavy eval-package import chain (langchain, psycopg). Must match those strings.
_TOOL_CALL_LIMIT_TEXT = "Tool call limit exceeded."
_MODEL_CALL_LIMIT_TEXT = "Model call limits exceeded:"


def _ai_indices_starting_with(record: dict, prefix: str) -> list[int]:
    """Trace indices of AIMessages whose (string) content starts with ``prefix``."""
    out = []
    for idx, msg in enumerate(record.get("messages", [])):
        if msg.get("role") == "ai":
            content = msg.get("content")
            if isinstance(content, str) and content.startswith(prefix):
                out.append(idx)
    return out


def _detect_tool_call_limit(events: list[ToolEvent], record: dict) -> PatternHit | None:
    # Trace is the primary source (always present); the field is a cross-check that
    # also covers any future variant whose text we don't match.
    idxs = [e.message_index for e in events if e.result_text.startswith(_TOOL_CALL_LIMIT_TEXT)]
    if idxs or "tool_call_limit" in _middleware_event_types(record):
        return PatternHit(
            "tool_call_limit", "Tool-call limit",
            "ToolCallLimitMiddleware blocked a tool — the agent burned past the "
            "safety-net tool-call cap (well above the patience budget)",
            idxs,
        )
    return None


def _detect_model_call_limit(events: list[ToolEvent], record: dict) -> PatternHit | None:
    idxs = _ai_indices_starting_with(record, _MODEL_CALL_LIMIT_TEXT)
    if idxs or "model_call_limit" in _middleware_event_types(record):
        return PatternHit(
            "model_call_limit", "Model-call limit",
            "ModelCallLimitMiddleware ended the run — the agent hit the safety-net "
            "model-call cap before finishing on its own",
            idxs,
        )
    return None


def _detect_context_editing(events: list[ToolEvent], record: dict) -> PatternHit | None:
    # No trace footprint (the serialized record drops response_metadata), so this
    # pattern relies solely on the middleware_events field and cannot fire on runs
    # produced before that field was added.
    if "context_editing" in _middleware_event_types(record):
        return PatternHit(
            "context_editing", "Context editing",
            "ContextEditingMiddleware cleared old tool outputs — the conversation "
            "grew large enough to trip the context-clearing trigger",
        )
    return None


def _detect_truncated_generation(events: list[ToolEvent], record: dict) -> PatternHit | None:
    # An AIMessage with finish_reason == "length" was cut off by the max-model-len
    # cap. No error is raised (we no longer send max_tokens on the local vLLM path),
    # so this is the only signal the generation — possibly a tool call — was
    # incomplete. Sourced from the per-message field, so old runs are covered.
    idxs = [
        idx
        for idx, msg in enumerate(record.get("messages", []))
        if msg.get("role") == "ai" and msg.get("finish_reason") == "length"
    ]
    if idxs:
        return PatternHit(
            "truncated_generation", "Truncated generation",
            f"{len(idxs)} model call(s) cut off by the max-model-len cap "
            "(finish_reason='length') — the generation was incomplete",
            idxs,
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
    _detect_tool_call_limit,
    _detect_model_call_limit,
    _detect_context_editing,
    _detect_truncated_generation,
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
    ("tool_call_limit", "Tool-call limit"),
    ("model_call_limit", "Model-call limit"),
    ("context_editing", "Context editing"),
    ("truncated_generation", "Truncated generation"),
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
    """A pattern's rate and the denominators it was computed over.

    ``rate`` is the sample-average among *applicable* samples (mean over instances
    of each instance's applicable-hit fraction). It is averaged over **instances**,
    so the honest denominator is ``applicable_instances`` (instances with ≥1
    applicable sample) — this is the ``N`` to show. ``applicable_n`` is the total
    applicable *samples* (instance × iteration) backing it; for a multi-iteration
    run ``applicable_n`` exceeds ``applicable_instances`` and is *not* the rate's
    denominator, so it should not be presented as ``n=`` next to the rate.

    The effective flagged-instance count consistent with the bar is
    ``rate * applicable_instances`` (an integer for single-iteration runs, possibly
    fractional otherwise). Reporting instances keeps patterns with different
    denominators comparable — a 38% KB-blind rate over 120 KB-needing instances
    reads honestly next to a 12% rate over all 412 instances.
    """
    rate: float
    applicable_n: int            # total applicable samples (instance × iteration)
    applicable_instances: int = 0  # instances with ≥1 applicable sample — the rate's N


def aggregate_patterns(groups: dict[str, list[dict]]) -> dict[str, PatternStat]:
    """name -> PatternStat over the pattern's applicable samples.

    The rate is the sample-average (mean over instances of the fraction of that
    instance's *applicable* samples hitting the pattern); instances with no
    applicable sample are excluded from the mean. Per record a pattern counts at
    most once (presence, not occurrence count). ``applicable_instances`` (the count
    of instances contributing to the mean) is the rate's true denominator;
    ``applicable_n`` is the total applicable samples behind those instances.
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
            applicable_instances=len(v),  # one entry per instance with applicable samples
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


def per_iteration_pattern_rates(records: list[dict]) -> pd.DataFrame:
    """Per-iteration anti-pattern rate, long-form ``[iteration, pattern, rate, n]``.

    Records are bucketed by their ``iteration`` field (one record per instance per
    iteration). Within each bucket, ``rate`` is the fraction of *applicable* records
    hitting the pattern and ``n`` is that applicable count. ``pattern`` carries the
    human label (matching ``PATTERN_CATALOG``).

    Exposes run-to-run variance the single sample-average bar hides; for a complete
    run (every instance present in every iteration) the mean of ``rate`` over
    iterations equals the per-instance sample-average from ``aggregate_patterns``.
    """
    labels = dict(PATTERN_CATALOG)
    by_iter: dict[int, list[dict]] = {}
    for r in records:
        by_iter.setdefault(int(r.get("iteration", 0) or 0), []).append(r)
    rows = []
    for it in sorted(by_iter):
        recs = by_iter[it]
        present = [{h.name for h in _record_hits(r)} for r in recs]
        for name in PATTERN_NAMES:
            applicable = [i for i, r in enumerate(recs) if _is_applicable(name, r)]
            n = len(applicable)
            rate = (sum(name in present[i] for i in applicable) / n) if n else 0.0
            rows.append({"iteration": it, "pattern": labels[name], "rate": rate, "n": n})
    return pd.DataFrame(rows)


def pattern_cooccurrence(records: list[dict]) -> pd.DataFrame:
    """Pattern×pattern co-occurrence across records — long-form ``[given, pattern, count, n_given, conditional]``.

    For each record the set of distinct anti-pattern names present is collected
    (presence, not occurrence count). ``count`` is the number of records where
    **both** ``given`` and ``pattern`` fire; ``n_given`` the number where ``given``
    fires; ``conditional = count / n_given`` is P(pattern | given) — of the records
    hitting ``given``, the fraction that also hit ``pattern``. The diagonal is
    ``given`` against itself (``count = n_given``, ``conditional = 1``).

    Only patterns that fire at least once appear on either axis, so a run with no
    occurrences of a pattern carries no empty row/column. ``given``/``pattern`` use
    the human labels (matching ``PATTERN_CATALOG``). Empty frame (with the columns)
    when nothing fired.

    Surfaces that anti-patterns are **not mutually exclusive**: a bright off-diagonal
    cell means the two failure modes tend to strike the same conversation.
    """
    labels = dict(PATTERN_CATALOG)
    totals: Counter = Counter()
    pair: Counter = Counter()
    for r in records:
        names = {h.name for h in _record_hits(r)}
        for a in names:
            totals[a] += 1
            for b in names:
                pair[(a, b)] += 1
    active = [name for name in PATTERN_NAMES if totals[name] > 0]
    rows = [
        {
            "given": labels[a],
            "pattern": labels[b],
            "count": pair.get((a, b), 0),
            "n_given": totals[a],
            "conditional": pair.get((a, b), 0) / totals[a],
        }
        for a in active
        for b in active
    ]
    if not rows:
        return pd.DataFrame(columns=["given", "pattern", "count", "n_given", "conditional"])
    return pd.DataFrame(rows)


def repeated_identical_tool_counts(records: list[dict]) -> pd.DataFrame:
    """Total surplus repeats per tool across the given records.

    For each record, identical calls are grouped by (tool, normalized-args); a
    group occurring ``k >= 2`` times contributes ``k - 1`` surplus repeats (the
    re-issues beyond the first). Surplus repeats are summed per tool name across
    all records. Returns a long-form frame ``[tool, repeats]`` sorted descending;
    empty (with those columns) when nothing was repeated.

    Intended for the ``repeated_identical_call`` drill-down: pass the records the
    pattern flagged, then plot ``repeats`` by ``tool``.
    """
    counts: Counter = Counter()
    for record in records:
        grouped: dict[tuple[str, str], int] = {}
        for e in extract_tool_events(record):
            if e.blocked:  # never executed, so not a real re-issue
                continue
            key = (e.tool_name, _norm_args(e.arguments))
            grouped[key] = grouped.get(key, 0) + 1
        for (tool, _), k in grouped.items():
            if k >= 2:
                counts[tool] += k - 1
    rows = [{"tool": tool, "repeats": c} for tool, c in counts.items()]
    if not rows:
        return pd.DataFrame(columns=["tool", "repeats"])
    return pd.DataFrame(rows).sort_values("repeats", ascending=False)


def _quintile_labels(n_bins: int) -> list[str]:
    """Percentage-range bucket labels, e.g. n_bins=5 → ['0-20%', …, '80-100%']."""
    step = 100 // n_bins
    return [f"{i * step}-{(i + 1) * step}%" for i in range(n_bins)]


def _progress_bin(frac: float, n_bins: int) -> int:
    """Map a fraction in [0, 1] to a bucket index 0..n_bins-1 (1.0 clamps to last)."""
    return min(int(frac * n_bins), n_bins - 1)


def first_submit_accuracy_by_quintile(
    records: list[dict], n_bins: int = 5
) -> pd.DataFrame:
    """Accuracy bucketed by *when* the agent first answered — long-form [quintile, accuracy, n].

    For each conversation, progress = (index of the first ``submit_sql`` call) /
    (total tool calls); it is dropped into one of ``n_bins`` percentage buckets
    (0-20% = answered almost immediately, 80-100% = answered only at the very end).
    Conversations that never submit (no answer attempt) and those with no tool
    calls are excluded. ``accuracy`` is the pooled pass-rate within a bucket — each
    (instance, iteration) sample counts once — making the row a conditional accuracy
    "of conversations that first answered in this quintile, what fraction passed";
    ``n`` is that bucket's sample count. Every bucket is returned (``accuracy`` NaN
    when empty) so the columns of a heatmap stay stable across runs.

    Surfaces the "patience pays off" trend: accuracy typically climbs as the first
    answer attempt moves later in the conversation.
    """
    labels = _quintile_labels(n_bins)
    buckets: list[list[bool]] = [[] for _ in range(n_bins)]
    for r in records:
        events = [e for e in extract_tool_events(r) if not e.blocked]
        if not events:
            continue
        first_submit = next(
            (i for i, e in enumerate(events) if e.tool_name == "submit_sql"), None
        )
        if first_submit is None:
            continue
        b = _progress_bin(first_submit / len(events), n_bins)
        buckets[b].append(bool(r.get("execution_accuracy")))
    rows = [
        {
            "quintile": labels[i],
            "accuracy": (sum(vals) / len(vals)) if vals else float("nan"),
            "n": len(vals),
        }
        for i, vals in enumerate(buckets)
    ]
    return pd.DataFrame(rows)


def tool_position_accuracy(records: list[dict], n_bins: int = 5) -> pd.DataFrame:
    """Accuracy cross-tabbed by tool × normalized call position — long-form [tool, quintile, accuracy, n].

    For each conversation and each tool call at index ``i``, position = ``i / total
    calls`` selects a percentage bucket. A conversation contributes (once) to
    ``cell(tool, quintile)`` if it called that tool anywhere in that quintile — so a
    conversation legitimately lands in several cells (that is the cross-tab). The
    cell ``accuracy`` is the pooled pass-rate of the contributing conversations and
    ``n`` their count. Only populated cells are returned.

    Detects whether *where* a tool is used (early vs. late) correlates with success;
    it is the accuracy-coloured companion to ``positional_tool_distribution``. The
    correlation is exploratory: a good-looking cell may reflect the tool, the timing,
    or a confound between them.
    """
    labels = _quintile_labels(n_bins)
    cells: dict[tuple[str, str], list[bool]] = {}
    for r in records:
        events = [e for e in extract_tool_events(r) if not e.blocked]
        if not events:
            continue
        passed = bool(r.get("execution_accuracy"))
        seen: set[tuple[str, str]] = {
            (e.tool_name, labels[_progress_bin(i / len(events), n_bins)])
            for i, e in enumerate(events)
        }
        for key in seen:
            cells.setdefault(key, []).append(passed)
    rows = [
        {"tool": tool, "quintile": q, "accuracy": sum(v) / len(v), "n": len(v)}
        for (tool, q), v in cells.items()
    ]
    if not rows:
        return pd.DataFrame(columns=["tool", "quintile", "accuracy", "n"])
    return pd.DataFrame(rows)


def positional_tool_distribution(
    groups: dict[str, list[dict]], n_bins: int = 5
) -> pd.DataFrame:
    """Long-form frame [quintile, tool, share] for a 100%-stacked positional plot.

    Each conversation's tool sequence is mapped onto ``n_bins`` normalized position
    buckets (``call index / total calls`` → 0-20% … 80-100%), matching the quintile
    axis of the accuracy heatmaps. Within a bucket a conversation splits its weight
    across the tools it called there (share = tool count / calls in the bucket), so
    it contributes exactly 1.0 to every bucket it reaches. ``share`` is that split
    averaged over the conversations that have a call in the bucket, so shares sum to
    1 per quintile. A short conversation may not reach the later quintiles (it then
    contributes to none of them); empty quintiles return no rows.

    Using fractions of the conversation rather than absolute call indices means a
    long run of one tool no longer "spills" into a trailing ``≥8`` bucket — it stays
    inside the quintiles it actually spans.
    """
    labels = _quintile_labels(n_bins)
    accum: dict[str, dict[str, float]] = {q: {} for q in labels}
    reached: dict[str, int] = {q: 0 for q in labels}  # instances with a call in the bin
    for samples in groups.values():
        if not samples:
            continue
        # Average the instance's iterations into one per-bucket tool distribution.
        inst_share: dict[str, dict[str, float]] = {q: {} for q in labels}
        inst_reached: dict[str, int] = {q: 0 for q in labels}
        for r in samples:
            seq = [e.tool_name for e in extract_tool_events(r) if not e.blocked]
            if not seq:
                continue
            binned: dict[str, Counter] = {q: Counter() for q in labels}
            for i, name in enumerate(seq):
                binned[labels[_progress_bin(i / len(seq), n_bins)]][name] += 1
            for q in labels:
                total = sum(binned[q].values())
                if total == 0:
                    continue
                inst_reached[q] += 1
                for name, c in binned[q].items():
                    inst_share[q][name] = inst_share[q].get(name, 0.0) + c / total
        for q in labels:
            if inst_reached[q] == 0:
                continue
            reached[q] += 1
            for name, s in inst_share[q].items():
                accum[q][name] = accum[q].get(name, 0.0) + s / inst_reached[q]

    rows = [
        {"quintile": q, "tool": name, "share": total / reached[q]}
        for q in labels
        for name, total in accum[q].items()
        if reached[q] > 0
    ]
    if not rows:
        return pd.DataFrame(columns=["quintile", "tool", "share"])
    return pd.DataFrame(rows)
