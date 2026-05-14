import json
from collections import Counter
from pathlib import Path

import pytest

from explorer.loader import RunData, RunStats, _compute_stats, list_runs, load_run


# ── helpers ────────────────────────────────────────────────────────────────────

def make_record(
    execution_accuracy=False,
    total_tokens=100,
    total_cost=0.001,
    updated_user_patience=3,
    category="Query",
    tool_calls_in_order=None,
    **extra,
):
    return {
        "execution_accuracy": execution_accuracy,
        "total_tokens": total_tokens,
        "total_cost": total_cost,
        "updated_user_patience": updated_user_patience,
        "category": category,
        "tool_calls_in_order": tool_calls_in_order or [],
        "amb_user_query": "What is the count?",
        "not_ambiguos_query": "What is the count?",
        "instance_id": "test_001",
        "selected_database": "mydb",
        "sol_sql": ["SELECT 1"],
        "predicted_sql": "SELECT 1",
        "messages": [],
        **extra,
    }


# ── _compute_stats ─────────────────────────────────────────────────────────────

class TestComputeStats:
    def test_empty_records(self):
        stats = _compute_stats([])
        assert stats.n_total == 0
        assert stats.n_passed == 0
        assert stats.avg_tokens == 0.0
        assert stats.avg_cost == 0.0
        assert stats.avg_budget_remaining == 0.0
        assert stats.accuracy_by_category == {}
        assert stats.tool_usage == Counter()

    def test_single_passed(self):
        records = [make_record(execution_accuracy=True, total_tokens=200, total_cost=0.01, updated_user_patience=4)]
        stats = _compute_stats(records)
        assert stats.n_total == 1
        assert stats.n_passed == 1
        assert stats.avg_tokens == 200.0
        assert stats.avg_cost == pytest.approx(0.01)
        assert stats.avg_budget_remaining == 4.0

    def test_accuracy_by_category(self):
        records = [
            make_record(execution_accuracy=True, category="Query"),
            make_record(execution_accuracy=False, category="Query"),
            make_record(execution_accuracy=True, category="Management"),
        ]
        stats = _compute_stats(records)
        assert stats.accuracy_by_category["Query"] == pytest.approx(0.5)
        assert stats.accuracy_by_category["Management"] == pytest.approx(1.0)

    def test_tool_usage_dict_format(self):
        records = [
            make_record(tool_calls_in_order=[
                {"tool_name": "execute_sql", "arguments": {}, "tool_cost": 1},
                {"tool_name": "submit_sql", "arguments": {}, "tool_cost": 0},
            ]),
            make_record(tool_calls_in_order=[
                {"tool_name": "execute_sql", "arguments": {}, "tool_cost": 1},
            ]),
        ]
        stats = _compute_stats(records)
        assert stats.tool_usage["execute_sql"] == 2
        assert stats.tool_usage["submit_sql"] == 1

    def test_missing_fields_default_to_zero(self):
        stats = _compute_stats([{"execution_accuracy": False}])
        assert stats.avg_tokens == 0.0
        assert stats.avg_cost == 0.0


# ── list_runs ──────────────────────────────────────────────────────────────────

class TestListRuns:
    def test_empty_root(self, tmp_path):
        assert list_runs(tmp_path) == {}

    def test_nonexistent_root(self, tmp_path):
        assert list_runs(tmp_path / "nonexistent") == {}

    def test_single_run(self, tmp_path):
        (tmp_path / "no_tool" / "2026_05_14" / "09_17_54").mkdir(parents=True)
        assert list_runs(tmp_path) == {"no_tool": {"2026_05_14": ["09_17_54"]}}

    def test_multiple_times_newest_first(self, tmp_path):
        for t in ["09_00_00", "10_00_00", "08_00_00"]:
            (tmp_path / "no_tool" / "2026_05_14" / t).mkdir(parents=True)
        times = list_runs(tmp_path)["no_tool"]["2026_05_14"]
        assert times == ["10_00_00", "09_00_00", "08_00_00"]

    def test_dates_newest_first(self, tmp_path):
        for d in ["2026_05_12", "2026_05_14", "2026_05_13"]:
            (tmp_path / "no_tool" / d / "09_00_00").mkdir(parents=True)
        dates = list(list_runs(tmp_path)["no_tool"].keys())
        assert dates == ["2026_05_14", "2026_05_13", "2026_05_12"]

    def test_skips_files_in_baseline_dir(self, tmp_path):
        (tmp_path / "no_tool" / "2026_05_14" / "09_00_00").mkdir(parents=True)
        (tmp_path / "no_tool" / "some_file.txt").write_text("noise")
        result = list_runs(tmp_path)
        assert "some_file.txt" not in result.get("no_tool", {})
