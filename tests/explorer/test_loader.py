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
        assert stats.avg_input_tokens == 0.0
        assert stats.avg_output_tokens == 0.0
        assert stats.avg_cost == 0.0
        assert stats.avg_budget_remaining == 0.0
        assert stats.accuracy_by_database == {}
        assert stats.tool_usage == Counter()

    def test_single_passed(self):
        records = [make_record(
            execution_accuracy=True,
            mean_prompt_tokens=150,
            mean_completion_tokens=50,
            total_cost=0.01,
            updated_user_patience=4,
        )]
        stats = _compute_stats(records)
        assert stats.n_total == 1
        assert stats.n_passed == 1
        assert stats.avg_input_tokens == 150.0
        assert stats.avg_output_tokens == 50.0
        assert stats.avg_cost == pytest.approx(0.01)
        assert stats.avg_budget_remaining == 4.0

    def test_pass_at_1_collapses_iterations(self):
        # Two instances, three iterations each. Iterations are samples of the
        # same instance, so pass@1 averages per-instance rates, not all 6 records.
        records = [
            make_record(instance_id="t1", execution_accuracy=True),
            make_record(instance_id="t1", execution_accuracy=True),
            make_record(instance_id="t1", execution_accuracy=False),  # t1: 2/3
            make_record(instance_id="t2", execution_accuracy=False),
            make_record(instance_id="t2", execution_accuracy=False),
            make_record(instance_id="t2", execution_accuracy=False),  # t2: 0/3
        ]
        groups: dict[str, list[dict]] = {}
        for r in records:
            groups.setdefault(r["instance_id"], []).append(r)
        stats = _compute_stats(records, groups)
        assert stats.n_total == 6
        assert stats.n_instances == 2
        assert stats.pass_at_1 == pytest.approx((2 / 3 + 0.0) / 2)

    def test_pass_at_1_equals_accuracy_single_iteration(self):
        records = [
            make_record(instance_id="t1", execution_accuracy=True),
            make_record(instance_id="t2", execution_accuracy=False),
        ]
        stats = _compute_stats(records)
        assert stats.n_instances == 2
        assert stats.pass_at_1 == pytest.approx(0.5)

    def test_accuracy_by_database(self):
        records = [
            make_record(execution_accuracy=True, selected_database="db_a"),
            make_record(execution_accuracy=False, selected_database="db_a"),
            make_record(execution_accuracy=True, selected_database="db_b"),
        ]
        stats = _compute_stats(records)
        assert stats.accuracy_by_database["db_a"] == pytest.approx(0.5)
        assert stats.accuracy_by_database["db_b"] == pytest.approx(1.0)

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

    def test_budget_exhausted_counted_in_tool_usage(self):
        records = [
            make_record(
                tool_calls_in_order=[
                    {"tool_name": "execute_sql", "arguments": {}, "tool_cost": 1},
                ],
                messages=[
                    {
                        "role": "tool",
                        "tool_name": "execute_sql",
                        "content": {
                            "content": "Budget exhausted (0.0 remaining). You MUST call submit_sql now with your best SQL.",
                            "parse_error": "Expecting value: line 1 column 1 (char 0)",
                        },
                    }
                ],
            )
        ]
        stats = _compute_stats(records)
        assert stats.tool_usage["execute_sql"] == 1
        assert stats.tool_usage["budget_exhausted"] == 1

    def test_missing_fields_default_to_zero(self):
        stats = _compute_stats([{"execution_accuracy": False}])
        assert stats.avg_input_tokens == 0.0
        assert stats.avg_output_tokens == 0.0
        assert stats.avg_cost == 0.0


# ── list_runs ──────────────────────────────────────────────────────────────────

class TestListRuns:
    def test_empty_root(self, tmp_path):
        assert list_runs(tmp_path) == {}

    def test_nonexistent_root(self, tmp_path):
        assert list_runs(tmp_path / "nonexistent") == {}

    def test_single_run(self, tmp_path):
        run_dir = tmp_path / "2026_05_14" / "09_17_54__qwen-ddl"
        run_dir.mkdir(parents=True)
        (run_dir / "results.jsonl").write_text("{}\n", encoding="utf-8")
        assert list_runs(tmp_path) == {"2026_05_14": ["09_17_54__qwen-ddl"]}

    def test_multiple_times_newest_first(self, tmp_path):
        for t in ["09_00_00__a", "10_00_00__b", "08_00_00__c"]:
            run_dir = tmp_path / "2026_05_14" / t
            run_dir.mkdir(parents=True)
            (run_dir / "results.jsonl").write_text("{}\n", encoding="utf-8")
        times = list_runs(tmp_path)["2026_05_14"]
        assert times == ["10_00_00__b", "09_00_00__a", "08_00_00__c"]

    def test_dates_newest_first(self, tmp_path):
        for d in ["2026_05_12", "2026_05_14", "2026_05_13"]:
            run_dir = tmp_path / d / "09_00_00__slug"
            run_dir.mkdir(parents=True)
            (run_dir / "results.jsonl").write_text("{}\n", encoding="utf-8")
        dates = list(list_runs(tmp_path).keys())
        assert dates == ["2026_05_14", "2026_05_13", "2026_05_12"]

    def test_skips_files_in_date_dir(self, tmp_path):
        (tmp_path / "2026_05_14" / "09_00_00__slug").mkdir(parents=True)
        (tmp_path / "2026_05_14" / "some_file.txt").write_text("noise")
        result = list_runs(tmp_path)
        assert "some_file.txt" not in result.get("2026_05_14", [])


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
    groups: dict[str, list[dict]] = {}
    for r in records:
        groups.setdefault(r.get("instance_id", ""), []).append(r)
    return RunData(
        records=records,
        config={},
        stats=RunStats(
            n_total=len(records),
            n_passed=sum(1 for r in records if r.get("execution_accuracy")),
            n_instances=len(groups),
            pass_at_1=0.0,
            avg_input_tokens=0.0,
            avg_output_tokens=0.0,
            avg_cost=0.0,
            avg_budget_remaining=0.0,
            accuracy_by_database={},
            error_distribution=Counter(),
            tool_usage=Counter(),
        ),
        malformed_count=0,
        groups=groups,
        n_iterations=1,
    )


class TestJoinRuns:
    def test_two_runs_same_task_pass_fail(self):
        r_a = make_record(instance_id="t1", execution_accuracy=True, selected_database="db1")
        r_b = make_record(instance_id="t1", execution_accuracy=False, selected_database="db1")
        runs = {"run_a": _make_run_data([r_a]), "run_b": _make_run_data([r_b])}
        df = join_runs(runs)
        assert len(df) == 1
        assert df.iloc[0]["run_a"] == "1/1"
        assert df.iloc[0]["run_b"] == "0/1"

    def test_task_absent_in_one_run_shows_dash(self):
        r_a = make_record(instance_id="t1", execution_accuracy=True, selected_database="db1")
        r_b = make_record(instance_id="t2", execution_accuracy=False, selected_database="db1")
        runs = {"run_a": _make_run_data([r_a]), "run_b": _make_run_data([r_b])}
        df = join_runs(runs)
        assert len(df) == 2
        t1 = df[df["instance_id"] == "t1"].iloc[0]
        t2 = df[df["instance_id"] == "t2"].iloc[0]
        assert t1["run_a"] == "1/1"
        assert t1["run_b"] == "—"
        assert t2["run_a"] == "—"
        assert t2["run_b"] == "0/1"

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

    def test_cell_shows_pass_count_over_samples(self):
        g = [
            make_record(instance_id="t1", execution_accuracy=True, selected_database="db1"),
            make_record(instance_id="t1", execution_accuracy=False, selected_database="db1"),
        ]
        runs = {"run_a": _make_run_data(g)}
        df = join_runs(runs)
        assert df.iloc[0]["run_a"] == "1/2"


def test_compute_stats_pattern_frequency_sample_average():
    # One instance, 2 samples: one blind-submit (no execute), one validated.
    blind = make_record(
        instance_id="i1",
        messages=[
            {"role": "ai", "tool_calls": [{"tool_name": "submit_sql", "arguments": {"sql": "x"}}]},
            {"role": "tool", "tool_name": "submit_sql", "status": "success", "content": "ok"},
        ],
    )
    validated = make_record(
        instance_id="i1",
        messages=[
            {"role": "ai", "tool_calls": [{"tool_name": "execute_sql", "arguments": {"sql": "x"}}]},
            {"role": "tool", "tool_name": "execute_sql", "status": "success", "content": "ok"},
            {"role": "ai", "tool_calls": [{"tool_name": "submit_sql", "arguments": {"sql": "x"}}]},
            {"role": "tool", "tool_name": "submit_sql", "status": "success", "content": "ok"},
        ],
    )
    groups = {"i1": [blind, validated]}
    stats = _compute_stats([blind, validated], groups)
    assert stats.pattern_stats["blind_submit"].rate == 0.5
    assert stats.pattern_stats["blind_submit"].applicable_n == 2
    assert stats.clean_fraction == 0.5
