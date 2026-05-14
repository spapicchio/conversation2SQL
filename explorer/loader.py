from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class RunStats:
    n_total: int
    n_passed: int
    avg_tokens: float
    avg_cost: float
    avg_budget_remaining: float
    accuracy_by_category: dict[str, float]  # category -> pass rate
    tool_usage: Counter[str]  # tool_name -> total calls


@dataclass
class RunData:
    records: list[dict]
    config: dict
    stats: RunStats
    malformed_count: int = 0
    source_file: str = "results_smaller.jsonl"


def _compute_stats(records: list[dict]) -> RunStats:
    n_total = len(records)
    n_passed = sum(1 for r in records if r.get("execution_accuracy", False))
    avg_tokens = sum(r.get("total_tokens", 0) for r in records) / max(n_total, 1)
    avg_cost = sum(r.get("total_cost", 0) for r in records) / max(n_total, 1)
    avg_budget = sum(r.get("updated_user_patience", 0) for r in records) / max(n_total, 1)

    cat_totals: dict[str, int] = {}
    cat_passed: dict[str, int] = {}
    for r in records:
        cat = r.get("category", "Unknown")
        cat_totals[cat] = cat_totals.get(cat, 0) + 1
        if r.get("execution_accuracy", False):
            cat_passed[cat] = cat_passed.get(cat, 0) + 1
    accuracy_by_category = {
        cat: cat_passed.get(cat, 0) / total
        for cat, total in cat_totals.items()
    }

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
        avg_tokens=avg_tokens,
        avg_cost=avg_cost,
        avg_budget_remaining=avg_budget,
        accuracy_by_category=accuracy_by_category,
        tool_usage=tool_usage,
    )


def list_runs(results_root: Path) -> dict[str, dict[str, list[str]]]:
    """Return {baseline: {date: [time, ...]}} newest-first within each date."""
    runs: dict[str, dict[str, list[str]]] = {}
    if not results_root.exists():
        return runs
    for baseline_dir in sorted(results_root.iterdir()):
        if not baseline_dir.is_dir():
            continue
        dates: dict[str, list[str]] = {}
        for date_dir in sorted(baseline_dir.iterdir(), reverse=True):
            if not date_dir.is_dir():
                continue
            times = sorted(
                [d.name for d in date_dir.iterdir() if d.is_dir()],
                reverse=True,
            )
            if times:
                dates[date_dir.name] = times
        if dates:
            runs[baseline_dir.name] = dates
    return runs


def load_run(path: Path) -> RunData:
    """Load records and config from results/<baseline>/<date>/<time>/."""
    raise NotImplementedError
