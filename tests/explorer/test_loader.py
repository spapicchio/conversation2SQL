import json
from collections import Counter
from pathlib import Path

import pandas as pd
import pytest

from explorer.loader import RunData, RunStats, _compute_stats, list_runs, load_run, join_runs


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


# ── load_run ───────────────────────────────────────────────────────────────────

class TestLoadRun:
    def _write_jsonl(self, path: Path, records: list[dict]) -> None:
        path.write_text(
            "\n".join(json.dumps(r) for r in records) + "\n",
            encoding="utf-8",
        )

    def test_loads_smaller_jsonl(self, tmp_path):
        self._write_jsonl(tmp_path / "results_smaller.jsonl", [make_record()])
        run = load_run(tmp_path)
        assert len(run.records) == 1
        assert run.source_file == "results_smaller.jsonl"

    def test_falls_back_to_full_jsonl(self, tmp_path):
        self._write_jsonl(tmp_path / "results.jsonl", [make_record()])
        run = load_run(tmp_path)
        assert len(run.records) == 1
        assert run.source_file == "results.jsonl"

    def test_prefers_smaller_over_full(self, tmp_path):
        self._write_jsonl(tmp_path / "results_smaller.jsonl", [make_record(total_tokens=111)])
        self._write_jsonl(tmp_path / "results.jsonl", [make_record(total_tokens=999)])
        run = load_run(tmp_path)
        assert run.records[0]["total_tokens"] == 111

    def test_loads_config_yaml(self, tmp_path):
        self._write_jsonl(tmp_path / "results_smaller.jsonl", [make_record()])
        (tmp_path / "config.yaml").write_text("pipeline:\n  baseline: no_tool\n")
        run = load_run(tmp_path)
        assert run.config["pipeline"]["baseline"] == "no_tool"

    def test_missing_config_yaml_returns_empty_dict(self, tmp_path):
        self._write_jsonl(tmp_path / "results_smaller.jsonl", [make_record()])
        run = load_run(tmp_path)
        assert run.config == {}

    def test_malformed_lines_counted(self, tmp_path):
        content = (
            json.dumps(make_record()) + "\n"
            + "NOT JSON\n"
            + json.dumps(make_record()) + "\n"
        )
        (tmp_path / "results_smaller.jsonl").write_text(content, encoding="utf-8")
        run = load_run(tmp_path)
        assert len(run.records) == 2
        assert run.malformed_count == 1

    def test_empty_run_folder_returns_empty(self, tmp_path):
        run = load_run(tmp_path)
        assert run.records == []
        assert run.malformed_count == 0
        assert run.config == {}

    def test_stats_computed(self, tmp_path):
        records = [make_record(execution_accuracy=True), make_record(execution_accuracy=False)]
        self._write_jsonl(tmp_path / "results_smaller.jsonl", records)
        run = load_run(tmp_path)
        assert run.stats.n_total == 2
        assert run.stats.n_passed == 1


# ── join_runs ──────────────────────────────────────────────────────────────────

def _make_run_data(records: list[dict]) -> RunData:
    from collections import Counter
    return RunData(
        records=records,
        config={},
        stats=RunStats(
            n_total=len(records),
            n_passed=sum(1 for r in records if r.get("execution_accuracy")),
            avg_input_tokens=0.0,
            avg_output_tokens=0.0,
            avg_cost=0.0,
            avg_budget_remaining=0.0,
            accuracy_by_database={},
            error_distribution=Counter(),
            tool_usage=Counter(),
        ),
        malformed_count=0,
    )


class TestJoinRuns:
    def test_two_runs_same_task_pass_fail(self):
        r_a = make_record(instance_id="t1", execution_accuracy=True, selected_database="db1")
        r_b = make_record(instance_id="t1", execution_accuracy=False, selected_database="db1")
        runs = {"run_a": _make_run_data([r_a]), "run_b": _make_run_data([r_b])}
        df = join_runs(runs)
        assert len(df) == 1
        assert df.iloc[0]["run_a"] == "✓"
        assert df.iloc[0]["run_b"] == "✗"

    def test_task_absent_in_one_run_shows_dash(self):
        r_a = make_record(instance_id="t1", execution_accuracy=True, selected_database="db1")
        r_b = make_record(instance_id="t2", execution_accuracy=False, selected_database="db1")
        runs = {"run_a": _make_run_data([r_a]), "run_b": _make_run_data([r_b])}
        df = join_runs(runs)
        assert len(df) == 2
        t1 = df[df["instance_id"] == "t1"].iloc[0]
        t2 = df[df["instance_id"] == "t2"].iloc[0]
        assert t1["run_a"] == "✓"
        assert t1["run_b"] == "—"
        assert t2["run_a"] == "—"
        assert t2["run_b"] == "✗"

    def test_question_truncated_at_80_chars(self):
        long_q = "A" * 100
        r = make_record(instance_id="t1", amb_user_query=long_q, selected_database="db1")
        runs = {"run_a": _make_run_data([r])}
        df = join_runs(runs)
        q = df.iloc[0]["Question"]
        assert q.endswith("…")
        assert len(q) == 81  # 80 chars + ellipsis

    def test_question_not_truncated_when_short(self):
        r = make_record(instance_id="t1", amb_user_query="Short", selected_database="db1")
        runs = {"run_a": _make_run_data([r])}
        df = join_runs(runs)
        assert df.iloc[0]["Question"] == "Short"

    def test_not_ambiguous_query_fallback(self):
        r = make_record(instance_id="t1", selected_database="db1")
        r["amb_user_query"] = ""
        r["not_ambiguos_query"] = "Fallback question"
        runs = {"run_a": _make_run_data([r])}
        df = join_runs(runs)
        assert df.iloc[0]["Question"] == "Fallback question"

    def test_database_column_from_first_run(self):
        r = make_record(instance_id="t1", execution_accuracy=True, selected_database="my_db")
        runs = {"run_a": _make_run_data([r])}
        df = join_runs(runs)
        assert df.iloc[0]["database"] == "my_db"

    def test_empty_runs_returns_empty_dataframe(self):
        runs = {"run_a": _make_run_data([]), "run_b": _make_run_data([])}
        df = join_runs(runs)
        assert len(df) == 0
        assert isinstance(df, pd.DataFrame)

    def test_columns_include_all_run_labels(self):
        r = make_record(instance_id="t1", execution_accuracy=True, selected_database="db1")
        runs = {"alpha": _make_run_data([r]), "beta": _make_run_data([r]), "gamma": _make_run_data([r])}
        df = join_runs(runs)
        assert "alpha" in df.columns
        assert "beta" in df.columns
        assert "gamma" in df.columns
