from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml


@dataclass
class RunStats:
    n_total: int
    n_passed: int
    avg_input_tokens: float
    avg_output_tokens: float
    avg_cost: float
    avg_budget_remaining: float
    accuracy_by_database: dict[str, float]  # database -> pass rate
    error_distribution: Counter[str]  # error_class -> count
    tool_usage: Counter[str]  # tool_name -> total calls


@dataclass
class RunData:
    records: list[dict]
    config: dict
    stats: RunStats
    malformed_count: int = 0
    source_file: str = "results_smaller.jsonl"


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
    if re.search(r"syntax error", message_lower):
        return "Syntax Error"
    if re.search(r"column .+ does not exist|does not exist", message_lower):
        return "Column/Relation Not Found"
    if "your sql is not correct" in message_lower:
        return "Wrong SQL"
    if "databaseerror" in message_lower:
        return "DB Error"
    return "Other"


def _compute_stats(records: list[dict]) -> RunStats:
    n_total = len(records)
    n_passed = sum(1 for r in records if r.get("execution_accuracy", False))
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
        avg_input_tokens=avg_input,
        avg_output_tokens=avg_output,
        avg_cost=avg_cost,
        avg_budget_remaining=avg_budget,
        accuracy_by_database=accuracy_by_database,
        error_distribution=error_distribution,
        tool_usage=tool_usage,
    )


def list_runs(results_root: Path) -> dict[str, list[str]]:
    """Return {date: [time, ...]} newest-first, scanning results/<date>/<time>/."""
    runs: dict[str, list[str]] = {}
    if not results_root.exists():
        return runs
    for date_dir in sorted(results_root.iterdir(), reverse=True):
        if not date_dir.is_dir():
            continue
        times = sorted(
            [d.name for d in date_dir.iterdir() if d.is_dir()],
            reverse=True,
        )
        if times:
            runs[date_dir.name] = times
    return runs


def load_run(path: Path) -> RunData:
    """Load records and config from results/<baseline>/<date>/<time>/."""
    source = path / "results.jsonl"

    records: list[dict] = []
    malformed = 0
    if source.exists():
        with source.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    malformed += 1

    for r in records:
        r["_error_class"] = classify_submit_error(r)

    config: dict = {}
    config_path = path / "config.yaml"
    if config_path.exists():
        with config_path.open(encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

    return RunData(
        records=records,
        config=config,
        stats=_compute_stats(records),
        malformed_count=malformed,
        source_file=source.name,
    )


def join_runs(runs: dict[str, RunData]) -> "pd.DataFrame":
    """Merge N RunData objects on instance_id into a comparison DataFrame.

    Columns: instance_id, database, Question, then one column per run label
    with values ✓ (passed), ✗ (failed), or — (task absent in that run).
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

    id_to_records: dict[str, dict[str, dict]] = {}
    for label, run_data in runs.items():
        for r in run_data.records:
            iid = r.get("instance_id", "")
            id_to_records.setdefault(iid, {})[label] = r

    rows = []
    for iid, base in all_ids.items():
        row = dict(base)
        for label in runs:
            record = id_to_records.get(iid, {}).get(label)
            if record is None:
                row[label] = "—"
            elif record.get("execution_accuracy"):
                row[label] = "✓"
            else:
                row[label] = "✗"
        rows.append(row)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    # For testing
    run_data = list_runs(Path("results"))
    print(run_data)