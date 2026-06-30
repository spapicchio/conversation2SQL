import json
from collections import Counter
from pathlib import Path

import pandas as pd
import pytest

from explorer.loader import (
    LENGTH_METRICS,
    RunData,
    RunStats,
    _compute_stats,
    classify_run_error,
    classify_submit_error,
    conversation_length,
    conversation_length_split,
    list_runs,
    load_run,
    join_runs,
)


class TestClassifyRunError:
    def test_timeout(self):
        assert classify_run_error("litellm.Timeout: APITimeoutError - timed out") == "Timeout"

    def test_context_window(self):
        msg = "litellm.ContextWindowExceededError: ContextWindowExceededError - maximum context length is 64000"
        assert classify_run_error(msg) == "Context Window Exceeded"

    def test_internal_server_error(self):
        assert classify_run_error("litellm.InternalServerError: boom") == "Internal Server Error"

    def test_bad_request(self):
        assert classify_run_error("litellm.BadRequestError: bad params") == "Bad Request"

    def test_patience_state_error(self):
        assert classify_run_error("At key 'updated_user_patience': value is invalid") == "Patience State Error"

    def test_other_fallback(self):
        assert classify_run_error("some unrecognized failure") == "Other"

    def test_non_string_input(self):
        assert classify_run_error({"weird": 1}) == "Other"


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


# ── conversation_length ──────────────────────────────────────────────────────────

class TestConversationLength:
    def test_metric_registry(self):
        assert LENGTH_METRICS == ("Model calls", "Tool calls", "Budget spent")

    def test_model_calls(self):
        rec = make_record(num_model_calls=7)
        assert conversation_length(rec, "Model calls") == 7

    def test_model_calls_missing_is_zero(self):
        rec = make_record()
        assert conversation_length(rec, "Model calls") == 0

    def test_tool_calls_counts_entries(self):
        rec = make_record(
            tool_calls_in_order=[
                {"tool_name": "execute_sql"},
                {"tool_name": "ask_user"},
                {"tool_name": "submit_sql"},
            ]
        )
        assert conversation_length(rec, "Tool calls") == 3

    def test_tool_calls_empty(self):
        rec = make_record()
        assert conversation_length(rec, "Tool calls") == 0

    def test_budget_spent_is_initial_minus_remaining(self):
        # Remaining budget is read from the [SYSTEM NOTE] tool-message annotation.
        rec = make_record(
            initial_user_patience=10,
            messages=[{"role": "tool", "remaining_budget": 4}],
        )
        assert conversation_length(rec, "Budget spent") == 6

    def test_budget_spent_none_without_budget_info(self):
        # No remaining-budget note, terminal sentinel state, and no initial budget
        # (a no-tool baseline) -> undefined, so the caller can drop it.
        rec = make_record(updated_user_patience=-2)
        rec.pop("initial_user_patience", None)
        assert conversation_length(rec, "Budget spent") is None


# ── conversation_length_split ─────────────────────────────────────────────────────

class TestConversationLengthSplit:
    def test_splits_by_execution_accuracy(self):
        records = [
            make_record(execution_accuracy=True,  num_model_calls=5),
            make_record(execution_accuracy=False, num_model_calls=2),
            make_record(execution_accuracy=True,  num_model_calls=8),
        ]
        result = conversation_length_split(records, "Model calls")
        assert result["Passed"] == [5.0, 8.0]
        assert result["Failed"] == [2.0]

    def test_custom_label_uses_checkmark_suffix(self):
        records = [
            make_record(execution_accuracy=True,  num_model_calls=3),
            make_record(execution_accuracy=False, num_model_calls=1),
        ]
        result = conversation_length_split(records, "Model calls", label="runA")
        assert "runA ✓" in result
        assert "runA ✗" in result
        assert result["runA ✓"] == [3.0]
        assert result["runA ✗"] == [1.0]

    def test_drops_none_values(self):
        # Budget spent returns None when no budget info exists.
        records = [
            make_record(execution_accuracy=True),   # no initial_user_patience → None
            make_record(execution_accuracy=False),
        ]
        result = conversation_length_split(records, "Budget spent")
        assert result["Passed"] == []
        assert result["Failed"] == []

    def test_all_passed(self):
        records = [make_record(execution_accuracy=True, num_model_calls=4)]
        result = conversation_length_split(records, "Model calls")
        assert result["Passed"] == [4.0]
        assert result["Failed"] == []

    def test_all_failed(self):
        records = [make_record(execution_accuracy=False, num_model_calls=1)]
        result = conversation_length_split(records, "Model calls")
        assert result["Passed"] == []
        assert result["Failed"] == [1.0]


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

    def test_total_tokens_reflect_cumulative_per_conversation(self):
        # The full prompt is re-sent each turn, so total_*_tokens (summed across
        # LLM calls) is the real billed token count — averaged across records.
        records = [
            make_record(
                total_prompt_tokens=1000,
                total_completion_tokens=120,
                num_model_calls=4,
            ),
            make_record(
                total_prompt_tokens=3000,
                total_completion_tokens=280,
                num_model_calls=6,
            ),
        ]
        stats = _compute_stats(records)
        assert stats.avg_total_input_tokens == 2000.0
        assert stats.avg_total_output_tokens == 200.0
        assert stats.avg_model_calls == 5.0

    def test_total_tokens_default_to_zero_when_missing(self):
        stats = _compute_stats([{"execution_accuracy": False}])
        assert stats.avg_total_input_tokens == 0.0
        assert stats.avg_total_output_tokens == 0.0
        assert stats.avg_model_calls == 0.0

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

    def test_avg_budget_remaining_reads_last_trace_note_not_sentinel(self):
        # The record-level updated_user_patience is overwritten with terminal
        # sentinels by the agent middleware (-2 after a terminal submit, -1 on
        # a budget block), so the metric must come from the last [SYSTEM NOTE]
        # budget annotation parsed into the tool messages.
        records = [make_record(
            updated_user_patience=-2,
            messages=[
                {"role": "tool", "tool_name": "execute_sql", "status": "success",
                 "content": {}, "remaining_budget": 9.0, "total_budget": 12.0},
                {"role": "tool", "tool_name": "execute_sql", "status": "success",
                 "content": {}, "remaining_budget": 5.0, "total_budget": 12.0},
                {"role": "tool", "tool_name": "submit_sql", "status": "success",
                 "content": {"passed": True, "message": "ok"}},
            ],
        )]
        stats = _compute_stats(records)
        assert stats.avg_budget_remaining == 5.0

    def test_avg_budget_remaining_falls_back_to_initial_budget(self):
        # Sentinel state and no budget note anywhere (terminal on the very
        # first tool call): nothing was ever deducted → the initial budget.
        records = [make_record(updated_user_patience=-2, initial_user_patience=12)]
        stats = _compute_stats(records)
        assert stats.avg_budget_remaining == 12.0

    def test_avg_budget_remaining_excludes_records_without_budget_info(self):
        # no_tool baseline records carry updated_user_patience=None and no
        # trace notes — they must not drag the average toward 0.
        records = [
            make_record(
                updated_user_patience=-2,
                messages=[
                    {"role": "tool", "tool_name": "execute_sql", "status": "success",
                     "content": {}, "remaining_budget": 4.0, "total_budget": 12.0},
                ],
            ),
            make_record(updated_user_patience=None),
        ]
        stats = _compute_stats(records)
        assert stats.avg_budget_remaining == 4.0

    def test_missing_fields_default_to_zero(self):
        stats = _compute_stats([{"execution_accuracy": False}])
        assert stats.avg_input_tokens == 0.0
        assert stats.avg_output_tokens == 0.0
        assert stats.avg_cost == 0.0

    def test_truncated_records_counted_from_message_finish_reason(self):
        """Conversations with an AI message cut off by the max-model-len cap
        (finish_reason='length') must be counted — derived on-read from the
        per-message field, so old runs without record-level flags are covered."""
        records = [
            make_record(
                messages=[
                    {"role": "ai", "finish_reason": "stop"},
                    {"role": "ai", "finish_reason": "length"},
                ]
            ),
            make_record(
                messages=[{"role": "ai", "finish_reason": "tool_calls"}]
            ),
        ]
        stats = _compute_stats(records)
        assert stats.n_truncated == 1
        assert stats.truncated_fraction == pytest.approx(0.5)

    def test_no_truncation_defaults_to_zero(self):
        stats = _compute_stats([make_record()])
        assert stats.n_truncated == 0
        assert stats.truncated_fraction == 0.0


# ── classify_submit_error ──────────────────────────────────────────────────────

class TestClassifySubmitError:
    def _rec(self, message, passed=False):
        return {
            "execution_accuracy": passed,
            "messages": [
                {"role": "tool", "tool_name": "submit_sql", "status": "success",
                 "content": {"passed": passed, "message": message}},
            ],
        }

    def test_target_error_wins_over_empty_query_words(self):
        # A target-side failure is a dataset problem even when the underlying
        # DB error happens to mention an empty query.
        rec = self._rec(
            "[TARGET ERROR] DatabaseError executing submitted SQL: "
            "can't execute an empty query"
        )
        assert classify_submit_error(rec) == "Target Error"

    def test_prediction_empty_query(self):
        rec = self._rec(
            "[PREDICTION ERROR] DatabaseError executing submitted SQL: "
            "can't execute an empty query"
        )
        assert classify_submit_error(rec) == "Empty Query"

    def test_empty_and_query_words_apart_do_not_mean_empty_query(self):
        rec = self._rec("Your SQL is not correct. The query returned an empty result set.")
        assert classify_submit_error(rec) == "Wrong SQL"


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
            json.dumps(make_record(instance_id="t1")) + "\n"
            + "NOT JSON\n"
            + json.dumps(make_record(instance_id="t2")) + "\n"
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
        records = [
            make_record(execution_accuracy=True, instance_id="t1"),
            make_record(execution_accuracy=False, instance_id="t2"),
        ]
        self._write_jsonl(tmp_path / "results_smaller.jsonl", records)
        run = load_run(tmp_path)
        assert run.stats.n_total == 2
        assert run.stats.n_passed == 1

    def test_duplicate_instance_iteration_pairs_deduped(self, tmp_path):
        # A resume/recover double-write must not silently count as an extra
        # sample (it would inflate pass@1 / pass@k / reliability).
        rec = make_record(iteration=0)
        self._write_jsonl(tmp_path / "results_iter0.jsonl", [rec, rec])
        run = load_run(tmp_path)
        assert len(run.records) == 1
        assert run.duplicate_count == 1
        assert run.stats.n_total == 1
        assert len(run.groups["test_001"]) == 1

    def test_same_instance_across_iterations_not_deduped(self, tmp_path):
        self._write_jsonl(tmp_path / "results_iter0.jsonl", [make_record(iteration=0)])
        self._write_jsonl(tmp_path / "results_iter1.jsonl", [make_record(iteration=1)])
        run = load_run(tmp_path)
        assert len(run.records) == 2
        assert run.duplicate_count == 0

    def test_loads_errors(self, tmp_path):
        self._write_jsonl(tmp_path / "results_iter0.jsonl", [make_record(instance_id="t1")])
        self._write_jsonl(
            tmp_path / "results_error.jsonl",
            [
                {"instance_id": "e1", "iteration": 0, "error": "litellm.Timeout: x"},
                {"instance_id": "e2", "iteration": 0, "error": "litellm.BadRequestError: y"},
            ],
        )
        run = load_run(tmp_path)
        assert run.stats.n_errors == 2
        assert run.stats.run_error_distribution == Counter({"Timeout": 1, "Bad Request": 1})
        assert {e["instance_id"] for e in run.errors} == {"e1", "e2"}
        assert all("_error_class" in e for e in run.errors)
        # purely diagnostic: completed records / accuracy untouched
        assert len(run.records) == 1
        assert run.stats.n_instances == 1

    def test_no_error_file(self, tmp_path):
        self._write_jsonl(tmp_path / "results_iter0.jsonl", [make_record(instance_id="t1")])
        run = load_run(tmp_path)
        assert run.errors == []
        assert run.stats.n_errors == 0
        assert run.stats.run_error_distribution == Counter()

    def test_errors_deduped(self, tmp_path):
        self._write_jsonl(tmp_path / "results_iter0.jsonl", [make_record(instance_id="t1")])
        self._write_jsonl(
            tmp_path / "results_error.jsonl",
            [
                {"instance_id": "e1", "iteration": 0, "error": "litellm.Timeout: x"},
                {"instance_id": "e1", "iteration": 0, "error": "litellm.Timeout: x"},
            ],
        )
        run = load_run(tmp_path)
        assert run.stats.n_errors == 1


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
            avg_total_input_tokens=0.0,
            avg_total_output_tokens=0.0,
            avg_model_calls=0.0,
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
