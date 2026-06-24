from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import yaml

try:  # pragma: no cover - bare import only when Streamlit runs from explorer/
    from metrics import ReliabilityStats, reliability_metrics
except ModuleNotFoundError:
    from explorer.metrics import ReliabilityStats, reliability_metrics

try:  # pragma: no cover - bare import only when Streamlit runs from explorer/
    from patterns import PatternStat, aggregate_patterns, clean_fraction, detect_patterns
except ModuleNotFoundError:
    from explorer.patterns import (
        PatternStat,
        aggregate_patterns,
        clean_fraction,
        detect_patterns,
    )


@dataclass
class RunStats:
    n_total: int
    n_passed: int
    n_instances: int       # unique instance_ids (records collapsed across iterations)
    pass_at_1: float       # mean over instances of (passes / samples); == accuracy for single-iter
    avg_input_tokens: float   # per LLM call (mean over model calls)
    avg_output_tokens: float  # per LLM call (mean over model calls)
    avg_total_input_tokens: float   # cumulative per conversation (prompt re-sent each turn)
    avg_total_output_tokens: float  # cumulative per conversation
    avg_model_calls: float    # LLM calls per conversation
    avg_cost: float
    avg_budget_remaining: float  # from the last trace budget note (state is a sentinel; see _remaining_budget)
    accuracy_by_database: dict[str, float]  # database -> pass rate
    error_distribution: Counter[str]  # error_class -> count
    tool_usage: Counter[str]  # tool_name -> total calls
    pattern_stats: dict[str, PatternStat] = field(default_factory=dict)  # name -> rate + applicable N
    clean_fraction: float = 0.0  # sample-avg fraction of samples with zero hits
    reliability: ReliabilityStats | None = None
    n_truncated: int = 0  # records with >=1 model call cut off by max-model-len
    truncated_fraction: float = 0.0  # n_truncated / n_total
    n_errors: int = 0  # crashed samples from results_error.jsonl (not in records)
    run_error_distribution: Counter[str] = field(default_factory=Counter)  # error_class -> count


@dataclass
class RunData:
    records: list[dict]
    config: dict
    stats: RunStats
    malformed_count: int = 0
    source_file: str = "results.jsonl"
    groups: dict[str, list[dict]] = field(default_factory=dict)
    n_iterations: int = 0
    duplicate_count: int = 0  # dropped repeats of an (instance_id, iteration) pair
    errors: list[dict] = field(default_factory=list)  # results_error.jsonl, tagged with _error_class


def classify_submit_error(record: dict) -> str:
    """Return a human-readable error class for the record's last submit_sql result."""
    if record.get("execution_accuracy"):
        return "Passed"
    last_msg = None
    for msg in record.get("messages", []):
        if msg.get("role") == "tool" and "submit" in (msg.get("tool_name") or ""):
            last_msg = msg
    if last_msg is None:
        return "No Submission"
    content = last_msg.get("content", {})
    message = content.get("message", "") if isinstance(content, dict) else str(content)
    message_lower = message.lower()
    # Target-side failures are dataset problems — classify them first so a DB
    # error that happens to mention e.g. an empty query isn't mislabeled.
    if "[TARGET ERROR]" in message:
        return "Target Error"
    if "empty query" in message_lower:
        return "Empty Query"
    if re.search(r"syntax error", message_lower):
        return "Syntax Error"
    if re.search(r"column .+ does not exist|does not exist", message_lower):
        return "Column/Relation Not Found"
    if "your sql is not correct" in message_lower:
        return "Wrong SQL"
    if "databaseerror" in message_lower:
        return "DB Error"
    return "Other"


def classify_run_error(error: object) -> str:
    """Return a human-readable class for a results_error.jsonl 'error' string.

    These are mid-run crashes (the sample never produced a completed record),
    not submit_sql outcomes — see classify_submit_error for the latter. Check
    the more specific provider classes (context-window, bad-request) before the
    generic ones so a context-window error isn't mislabeled as a bad request.
    """
    text = error if isinstance(error, str) else str(error)
    if "ContextWindowExceededError" in text:
        return "Context Window Exceeded"
    if "BadRequestError" in text:
        return "Bad Request"
    if "InternalServerError" in text:
        return "Internal Server Error"
    if "Timeout" in text:
        return "Timeout"
    if "updated_user_patience" in text:
        return "Patience State Error"
    return "Other"


def _tool_message_text(content: object) -> str:
    if isinstance(content, dict):
        for key in ("content", "message", "error"):
            value = content.get(key)
            if isinstance(value, str):
                return value
        return ""
    if isinstance(content, str):
        return content
    return ""


def _remaining_budget(record: dict) -> float | None:
    """Best-effort remaining patience budget at the end of a conversation.

    The record-level ``updated_user_patience`` cannot be averaged: the agent
    middleware overwrites it with terminal sentinels (-2 after a terminal
    submit_sql, -1 when a tool was budget-blocked), so it reflects *how* the
    episode ended, not how much budget was left. The budget the agent actually
    saw lives in the ``[SYSTEM NOTE: Remaining budget: x/y]`` annotations,
    parsed into ``remaining_budget`` on tool messages at serialization time —
    take the last one. Records without a note fall back to the non-sentinel
    state value, then to the initial budget (nothing was ever deducted).
    Returns None when no budget information exists (e.g. no_tool baseline).
    """
    remaining = None
    for msg in record.get("messages", []):
        if msg.get("role") == "tool" and msg.get("remaining_budget") is not None:
            remaining = msg["remaining_budget"]
    if remaining is not None:
        return float(remaining)
    updated = record.get("updated_user_patience")
    if isinstance(updated, (int, float)) and updated >= 0:
        return float(updated)
    initial = record.get("initial_user_patience")
    if isinstance(initial, (int, float)):
        return float(initial)
    return None


# Conversation-length metrics surfaced in the length boxplot (app.py / compare.py).
LENGTH_METRICS = ("Model calls", "Tool calls", "Budget spent")


def conversation_length(record: dict, metric: str) -> float | None:
    """Return one length metric for a single conversation record.

    - ``"Model calls"`` — ``num_model_calls`` (the agent's LLM turns), 0 if absent.
    - ``"Tool calls"``  — number of entries in ``tool_calls_in_order``.
    - ``"Budget spent"`` — ``initial_user_patience - remaining`` coins, where
      ``remaining`` comes from :func:`_remaining_budget`. Returns ``None`` when no
      budget information exists (e.g. a no-tool baseline) so callers can drop it;
      the other two metrics are always defined.
    """
    if metric == "Model calls":
        return float(record.get("num_model_calls") or 0)
    if metric == "Tool calls":
        return float(len(record.get("tool_calls_in_order") or []))
    if metric == "Budget spent":
        remaining = _remaining_budget(record)
        initial = record.get("initial_user_patience")
        if remaining is None or not isinstance(initial, (int, float)):
            return None
        return float(initial) - remaining
    raise ValueError(f"unknown length metric: {metric!r}")


def conversation_length_split(
    records: list[dict], metric: str, label: str = ""
) -> dict[str, list[float]]:
    """Partition conversation-length values by execution accuracy.

    Returns a dict with two keys. Without a label: ``"Passed"`` and
    ``"Failed"``. With a label: ``"{label} ✓"`` and ``"{label} ✗"``.
    Records where :func:`conversation_length` returns ``None`` (e.g.
    no-tool baselines for "Budget spent") are dropped from both lists.
    """
    pass_key = f"{label} ✓" if label else "Passed"
    fail_key = f"{label} ✗" if label else "Failed"
    passed: list[float] = []
    failed: list[float] = []
    for r in records:
        v = conversation_length(r, metric)
        if v is None:
            continue
        (passed if r.get("execution_accuracy") else failed).append(v)
    return {pass_key: passed, fail_key: failed}


def _compute_stats(
    records: list[dict],
    groups: dict[str, list[dict]] | None = None,
    errors: list[dict] = (),
) -> RunStats:
    n_total = len(records)
    n_passed = sum(1 for r in records if r.get("execution_accuracy", False))

    # pass@1: average the per-instance pass rate so multiple iterations count as
    # repeated samples of one instance, not as n× more samples.
    inst_groups = groups
    if inst_groups is None:
        inst_groups = {}
        for rec in records:
            inst_groups.setdefault(rec.get("instance_id", ""), []).append(rec)
    n_instances = len(inst_groups)
    pass_at_1 = (
        sum(
            sum(1 for r in g if r.get("execution_accuracy", False)) / len(g)
            for g in inst_groups.values()
            if g
        )
        / max(n_instances, 1)
    )

    avg_input = sum(r.get("mean_prompt_tokens") or 0 for r in records) / max(n_total, 1)
    avg_output = sum(r.get("mean_completion_tokens") or 0 for r in records) / max(
        n_total, 1
    )
    # Cumulative tokens per conversation: the agent re-sends the whole prompt
    # each turn, so these sums are the real billed token counts (much larger
    # than the per-call means for tool-using baselines).
    avg_total_input = sum(r.get("total_prompt_tokens") or 0 for r in records) / max(
        n_total, 1
    )
    avg_total_output = sum(
        r.get("total_completion_tokens") or 0 for r in records
    ) / max(n_total, 1)
    avg_model_calls = sum(r.get("num_model_calls") or 0 for r in records) / max(
        n_total, 1
    )
    avg_cost = sum(r.get("total_cost") or 0 for r in records) / max(n_total, 1)
    # Mean over the records that carry budget information (see _remaining_budget).
    budget_values = [
        b for b in (_remaining_budget(r) for r in records) if b is not None
    ]
    avg_budget = sum(budget_values) / len(budget_values) if budget_values else 0.0

    db_totals: dict[str, int] = {}
    db_passed: dict[str, int] = {}
    for r in records:
        db = r.get("selected_database", "Unknown")
        db_totals[db] = db_totals.get(db, 0) + 1
        if r.get("execution_accuracy", False):
            db_passed[db] = db_passed.get(db, 0) + 1
    accuracy_by_database = {
        db: db_passed.get(db, 0) / total for db, total in db_totals.items()
    }

    error_distribution: Counter[str] = Counter(
        classify_submit_error(r) for r in records
    )

    tool_usage: Counter[str] = Counter()
    for r in records:
        for tc in r.get("tool_calls_in_order", []):
            if isinstance(tc, dict):
                tool_usage[tc.get("tool_name", "unknown")] += 1
            elif isinstance(tc, str):
                tool_usage[tc] += 1

    budget_exhausted = 0
    for r in records:
        for msg in r.get("messages", []):
            if msg.get("role") != "tool":
                continue
            content_text = _tool_message_text(msg.get("content"))
            if "Budget exhausted" in content_text:
                budget_exhausted += 1
    if budget_exhausted:
        tool_usage["budget_exhausted"] += budget_exhausted

    # Truncation: an AI message with finish_reason == "length" was cut off by the
    # max-model-len cap (no error is raised). Derived from the per-message field
    # so old runs — which predate the record-level was_truncated flag — are
    # covered without rewriting any historical JSONL.
    n_truncated = sum(
        1
        for r in records
        if any(
            msg.get("role") == "ai" and msg.get("finish_reason") == "length"
            for msg in r.get("messages", [])
        )
    )
    truncated_fraction = n_truncated / n_total if n_total else 0.0

    g = groups or {}
    pattern_stats = aggregate_patterns(g)
    clean = clean_fraction(g)

    n_errors = len(errors)
    run_error_distribution: Counter[str] = Counter(
        e.get("_error_class", "Other") for e in errors
    )

    return RunStats(
        n_total=n_total,
        n_passed=n_passed,
        n_instances=n_instances,
        pass_at_1=pass_at_1,
        avg_input_tokens=avg_input,
        avg_output_tokens=avg_output,
        avg_total_input_tokens=avg_total_input,
        avg_total_output_tokens=avg_total_output,
        avg_model_calls=avg_model_calls,
        avg_cost=avg_cost,
        avg_budget_remaining=avg_budget,
        accuracy_by_database=accuracy_by_database,
        error_distribution=error_distribution,
        tool_usage=tool_usage,
        pattern_stats=pattern_stats,
        clean_fraction=clean,
        reliability=reliability_metrics(groups or {}),
        n_truncated=n_truncated,
        truncated_fraction=truncated_fraction,
        n_errors=n_errors,
        run_error_distribution=run_error_distribution,
    )


def _has_results(d: Path) -> bool:
    return (
        bool(list(d.glob("results_iter*.jsonl")))
        or (d / "results.jsonl").exists()
        or (d / "results_smaller.jsonl").exists()
    )


def list_runs(results_root: Path) -> dict[str, list[str]]:
    """Return {date: [run_key, ...]} newest-date-first.

    run_key is either:
      - "HH-MM-SS/slug"  (new layout: time dir contains slug subdirs)
      - "HH-MM-SS__slug" (old layout: slug dir sits directly under date)
    """
    runs: dict[str, list[str]] = {}
    if not results_root.exists():
        return runs
    for date_dir in sorted(results_root.iterdir(), reverse=True):
        if not date_dir.is_dir():
            continue
        run_keys: list[str] = []
        for subdir in sorted(date_dir.iterdir(), reverse=True):
            if not subdir.is_dir():
                continue
            if _has_results(subdir):
                # Old flat layout: results.jsonl sits directly in this dir.
                run_keys.append(subdir.name)
            else:
                # New layout: subdir is a time dir; look one level deeper for slugs.
                for slug_dir in sorted(subdir.iterdir(), reverse=True):
                    if slug_dir.is_dir() and _has_results(slug_dir):
                        run_keys.append(f"{subdir.name}/{slug_dir.name}")
        if run_keys:
            runs[date_dir.name] = run_keys
    return runs


def _iter_num(p: Path) -> int:
    """Extract the iteration index from a results_iter{N}.jsonl filename."""
    m = re.search(r"results_iter(\d+)", p.name)
    return int(m.group(1)) if m else 0


def _read_jsonl(source: Path) -> tuple[list[dict], int]:
    records: list[dict] = []
    malformed = 0
    with source.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                malformed += 1
    return records, malformed


def load_run(path: Path) -> RunData:
    """Load records and config from results/<baseline>/<date>/<run>/.

    Prefers per-iteration files (results_iter*.jsonl). Falls back to the legacy
    single file (results_smaller.jsonl, then results.jsonl), treated as iteration 0.
    """
    iter_files = sorted(path.glob("results_iter*.jsonl"), key=_iter_num)
    records: list[dict] = []
    malformed = 0
    if iter_files:
        for f in iter_files:
            recs, m = _read_jsonl(f)
            records.extend(recs)
            malformed += m
        source_file = f"results_iter*.jsonl ({len(iter_files)} files)"
    else:
        smaller = path / "results_smaller.jsonl"
        full = path / "results.jsonl"
        source = smaller if smaller.exists() else full
        if source.exists():
            records, malformed = _read_jsonl(source)
        source_file = source.name

    # A resume/recover double-write would otherwise count as an extra sample
    # and silently inflate pass@1 / pass@k / reliability: keep the first record
    # of each (instance_id, iteration) pair and count the dropped repeats.
    seen_pairs: set[tuple[str, object]] = set()
    deduped: list[dict] = []
    duplicate_count = 0
    for r in records:
        iid = r.get("instance_id")
        key = (iid, r.get("iteration", 0))
        if iid and key in seen_pairs:
            duplicate_count += 1
            continue
        if iid:
            seen_pairs.add(key)
        deduped.append(r)
    records = deduped

    for r in records:
        r["_error_class"] = classify_submit_error(r)
        r["_pattern_hits"] = detect_patterns(r)
        r.setdefault("iteration", 0)

    groups: dict[str, list[dict]] = {}
    for r in records:
        groups.setdefault(r.get("instance_id", ""), []).append(r)
    for g in groups.values():
        g.sort(key=lambda r: r.get("iteration", 0))
    n_iterations = len({r.get("iteration", 0) for r in records}) if records else 0

    errors: list[dict] = []
    error_file = path / "results_error.jsonl"
    if error_file.exists():
        raw_errors, _ = _read_jsonl(error_file)
        seen_err: set[tuple[object, object]] = set()
        for e in raw_errors:
            key = (e.get("instance_id"), e.get("iteration", 0))
            if e.get("instance_id") and key in seen_err:
                continue
            if e.get("instance_id"):
                seen_err.add(key)
            e["_error_class"] = classify_run_error(e.get("error", ""))
            errors.append(e)

    config: dict = {}
    config_path = path / "config.yaml"
    if config_path.exists():
        with config_path.open(encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

    return RunData(
        records=records,
        config=config,
        stats=_compute_stats(records, groups, errors),
        malformed_count=malformed,
        source_file=source_file,
        groups=groups,
        n_iterations=n_iterations,
        duplicate_count=duplicate_count,
        errors=errors,
    )


def join_runs(runs: dict[str, RunData]) -> "pd.DataFrame":
    """Merge N RunData objects on instance_id into a comparison DataFrame.

    Columns: instance_id, database, Question, then one column per run label with
    values "c/n" (c passes out of n samples for that instance) or "—" (task
    absent in that run).
    """
    all_ids: dict[str, dict] = {}
    for run_data in runs.values():
        for r in run_data.records:
            iid = r.get("instance_id", "")
            if iid not in all_ids:
                q = r.get("amb_user_query") or r.get("not_ambiguos_query", "")
                all_ids[iid] = {
                    "instance_id": iid,
                    "database": r.get("selected_database", ""),
                    "Question": (q[:80] + "…") if len(q) > 80 else q,
                }

    rows = []
    for iid, base in all_ids.items():
        row = dict(base)
        for label, run_data in runs.items():
            group = run_data.groups.get(iid)
            if not group:
                row[label] = "—"
            else:
                c = sum(1 for r in group if r.get("execution_accuracy"))
                row[label] = f"{c}/{len(group)}"
        rows.append(row)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    # For testing
    run_data = list_runs(Path("results"))
    print(run_data)
