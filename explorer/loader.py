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


@dataclass
class RunStats:
    n_total: int
    n_passed: int
    n_instances: int       # unique instance_ids (records collapsed across iterations)
    pass_at_1: float       # mean over instances of (passes / samples); == accuracy for single-iter
    avg_input_tokens: float
    avg_output_tokens: float
    avg_cost: float
    avg_budget_remaining: float
    accuracy_by_database: dict[str, float]  # database -> pass rate
    error_distribution: Counter[str]  # error_class -> count
    tool_usage: Counter[str]  # tool_name -> total calls
    reliability: ReliabilityStats | None = None


@dataclass
class RunData:
    records: list[dict]
    config: dict
    stats: RunStats
    malformed_count: int = 0
    source_file: str = "results.jsonl"
    groups: dict[str, list[dict]] = field(default_factory=dict)
    n_iterations: int = 0


def classify_submit_error(record: dict) -> str:
    """Return a human-readable error class for the record's last submit_sql result."""
    if record.get("execution_accuracy"):
        return "Passed"
    last_msg = None
    for msg in record.get("messages", []):
        if msg.get("role") == "tool" and "submit" in msg.get("tool_name", ""):
            last_msg = msg
    if last_msg is None:
        return "No Submission"
    content = last_msg.get("content", {})
    message = content.get("message", "") if isinstance(content, dict) else str(content)
    message_lower = message.lower()
    if (
        "empty query" in message_lower
        or "empty" in message_lower
        and "query" in message_lower
    ):
        return "Empty Query"
    if "[TARGET ERROR]" in message:
        return "Target Error"
        
    if re.search(r"syntax error", message_lower):
        return "Syntax Error"
    if re.search(r"column .+ does not exist|does not exist", message_lower):
        return "Column/Relation Not Found"
    if "your sql is not correct" in message_lower:
        return "Wrong SQL"
    if "databaseerror" in message_lower:
        return "DB Error"
    return "Other"


def _compute_stats(records: list[dict], groups: dict[str, list[dict]] | None = None) -> RunStats:
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
    avg_cost = sum(r.get("total_cost") or 0 for r in records) / max(n_total, 1)
    avg_budget = sum(r.get("updated_user_patience") or 0 for r in records) / max(
        n_total, 1
    )

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

    return RunStats(
        n_total=n_total,
        n_passed=n_passed,
        n_instances=n_instances,
        pass_at_1=pass_at_1,
        avg_input_tokens=avg_input,
        avg_output_tokens=avg_output,
        avg_cost=avg_cost,
        avg_budget_remaining=avg_budget,
        accuracy_by_database=accuracy_by_database,
        error_distribution=error_distribution,
        tool_usage=tool_usage,
        reliability=reliability_metrics(groups or {}),
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

    for r in records:
        r["_error_class"] = classify_submit_error(r)
        r.setdefault("iteration", 0)

    groups: dict[str, list[dict]] = {}
    for r in records:
        groups.setdefault(r.get("instance_id", ""), []).append(r)
    for g in groups.values():
        g.sort(key=lambda r: r.get("iteration", 0))
    n_iterations = len({r.get("iteration", 0) for r in records}) if records else 0

    config: dict = {}
    config_path = path / "config.yaml"
    if config_path.exists():
        with config_path.open(encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

    return RunData(
        records=records,
        config=config,
        stats=_compute_stats(records, groups),
        malformed_count=malformed,
        source_file=source_file,
        groups=groups,
        n_iterations=n_iterations,
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
