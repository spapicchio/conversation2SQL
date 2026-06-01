import json
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from conversation2sql.eval_framework.main_pipe_workflow import (
    _resolve_baseline_settings,
    _resolve_iterations,
    workflow_evaluation_pipeline,
)
from conversation2sql.config_input import (
    ConfigPipeline, ConfigReader, ConfigPredictor, ConfigUserSimulator,
)


class TestResolveBaselineSettings:
    @pytest.mark.parametrize("baseline,expected_amb,expected_user_sim", [
        ("no_tool", False, False),
        ("tools_only", False, False),
        ("tools_user", False, True),
        ("bird_full", True, True),
    ])
    def test_resolves(self, baseline, expected_amb, expected_user_sim):
        amb, runner, needs = _resolve_baseline_settings(baseline)
        assert amb is expected_amb
        assert needs is expected_user_sim
        assert callable(runner)

    def test_unknown_baseline_raises(self):
        with pytest.raises(ValueError):
            _resolve_baseline_settings("nonsense")


class TestResolveIterations:
    def test_passthrough_when_temperature_positive(self):
        assert _resolve_iterations(5, 0.7) == 5

    def test_collapses_to_one_when_temperature_zero(self):
        assert _resolve_iterations(5, 0.0) == 1

    def test_single_iteration_stays_one_when_temperature_zero(self):
        assert _resolve_iterations(1, 0.0) == 1

    def test_collapses_when_temperature_negative(self):
        assert _resolve_iterations(3, -0.1) == 1


@pytest.fixture
def configs(tmp_path):
    return (
        ConfigPipeline(debug=True, output_folder=str(tmp_path)),
        ConfigReader(),
        ConfigPredictor(),
        ConfigUserSimulator(),
    )


def _fake_task():
    t = MagicMock()
    t.instance_id = "task_1"
    t.model_dump.return_value = {
        "instance_id": "task_1",
        "selected_database": "db1",
        "amb_user_query": "q?",
        "sol_sql": "SELECT 1",
        "sql_query_conditions": {},
        "not_ambiguos_query": "q",
        "gt_knowledge_base": [],
        "category": "easy",
        "ddl_database_schema": "CREATE TABLE t (id INT);",
        "user_query_ambiguity": {},
    }
    return t


def _stub_response():
    return {
        "messages": [], "initial_user_patience": 0, "updated_user_patience": 0,
        "tool_called_patience": [], "total_cost": 0, "total_tokens": 0,
        "total_prompt_tokens": 0, "total_completion_tokens": 0,
        "mean_prompt_tokens": 0, "mean_completion_tokens": 0,
        "tool_calls_in_order": [], "execution_accuracy": False,
        "predicted_sql": "",
    }


@patch("conversation2sql.eval_framework.main_pipe_workflow.run_agent_bird_baseline")
@patch("conversation2sql.eval_framework.main_pipe_workflow.run_baseline_no_tool")
@patch("conversation2sql.eval_framework.main_pipe_workflow.load_bird_interact_as_tasks")
@patch("conversation2sql.eval_framework.main_pipe_workflow.utils_create_model")
class TestDispatch:
    def test_no_tool_dispatches_to_run_baseline_no_tool(
        self, mock_create, mock_load, mock_no_tool, mock_agent, configs,
    ):
        cp, cr, cpred, cu = configs
        cp.baseline = "no_tool"
        mock_load.return_value = [_fake_task()]
        mock_no_tool.return_value = _stub_response()
        mock_create.return_value = MagicMock()

        workflow_evaluation_pipeline(cp, cr, cpred, cu)

        mock_no_tool.assert_called_once()
        mock_agent.assert_not_called()
        assert mock_load.call_args.kwargs["make_data_ambiguous"] is False

    def test_tools_user_dispatches_to_agent_with_ask_user_true(
        self, mock_create, mock_load, mock_no_tool, mock_agent, configs,
    ):
        cp, cr, cpred, cu = configs
        cp.baseline = "tools_user"
        mock_load.return_value = [_fake_task()]
        mock_agent.return_value = _stub_response()
        mock_create.return_value = MagicMock()

        workflow_evaluation_pipeline(cp, cr, cpred, cu)

        mock_agent.assert_called_once()
        assert mock_agent.call_args.kwargs["enable_ask_user"] is True
        assert mock_create.call_count == 3

    def test_tools_only_skips_user_sim_construction(
        self, mock_create, mock_load, mock_no_tool, mock_agent, configs,
    ):
        cp, cr, cpred, cu = configs
        cp.baseline = "tools_only"
        mock_load.return_value = [_fake_task()]
        mock_agent.return_value = _stub_response()
        mock_create.return_value = MagicMock()

        workflow_evaluation_pipeline(cp, cr, cpred, cu)

        assert mock_agent.call_args.kwargs["enable_ask_user"] is False
        assert mock_create.call_count == 1

    def test_results_written_to_output_folder(
        self, mock_create, mock_load, mock_no_tool, mock_agent, configs, tmp_path,
    ):
        # Python writes results_iter{i}.jsonl directly into the given output_folder.
        cp, cr, cpred, cu = configs
        cp.baseline = "tools_only"
        mock_load.return_value = [_fake_task()]
        mock_agent.return_value = _stub_response()
        mock_create.return_value = MagicMock()

        workflow_evaluation_pipeline(cp, cr, cpred, cu)

        assert (tmp_path / "results_iter0.jsonl").exists()


class TestConcurrencyConfig:
    def test_concurrency_defaults_to_1(self):
        assert ConfigPipeline().concurrency == 1

    def test_num_iterations_defaults_to_1(self):
        assert ConfigPipeline().num_iterations == 3

    def test_num_iterations_rejects_zero(self):
        with pytest.raises(ValidationError):
            ConfigPipeline(num_iterations=0)

    def test_resume_defaults_to_false(self):
        assert ConfigPipeline().resume is False


@patch("conversation2sql.eval_framework.main_pipe_workflow.run_agent_bird_baseline")
@patch("conversation2sql.eval_framework.main_pipe_workflow.run_baseline_no_tool")
@patch("conversation2sql.eval_framework.main_pipe_workflow.load_bird_interact_as_tasks")
@patch("conversation2sql.eval_framework.main_pipe_workflow.utils_create_model")
class TestConcurrency:
    def test_all_tasks_processed_with_concurrency(
        self, mock_create, mock_load, mock_no_tool, mock_agent, configs
    ):
        cp, cr, cpred, cu = configs
        cp = cp.model_copy(update={"baseline": "no_tool", "concurrency": 4})
        mock_load.return_value = [_fake_task() for _ in range(4)]
        mock_no_tool.return_value = _stub_response()
        mock_create.return_value = MagicMock()

        workflow_evaluation_pipeline(cp, cr, cpred, cu)

        assert mock_no_tool.call_count == 4
        lines = (Path(cp.output_folder) / "results_iter0.jsonl").read_text().splitlines()
        assert len(lines) == 4

    def test_concurrent_tasks_run_in_parallel(
        self, mock_create, mock_load, mock_no_tool, mock_agent, configs
    ):
        cp, cr, cpred, cu = configs
        cp = cp.model_copy(update={"baseline": "no_tool", "concurrency": 4})

        concurrent_count = 0
        max_concurrent = 0
        count_lock = threading.Lock()

        def slow_runner(task, model_agent):
            nonlocal concurrent_count, max_concurrent
            with count_lock:
                concurrent_count += 1
                if concurrent_count > max_concurrent:
                    max_concurrent = concurrent_count
            time.sleep(0.05)
            with count_lock:
                concurrent_count -= 1
            return _stub_response()

        mock_load.return_value = [_fake_task() for _ in range(4)]
        mock_no_tool.side_effect = slow_runner
        mock_create.return_value = MagicMock()

        workflow_evaluation_pipeline(cp, cr, cpred, cu)

        assert max_concurrent > 1


@patch("conversation2sql.eval_framework.main_pipe_workflow.run_agent_bird_baseline")
@patch("conversation2sql.eval_framework.main_pipe_workflow.run_baseline_no_tool")
@patch("conversation2sql.eval_framework.main_pipe_workflow.load_bird_interact_as_tasks")
@patch("conversation2sql.eval_framework.main_pipe_workflow.utils_create_model")
class TestIterations:
    def test_n_iterations_write_one_file_each_when_temperature_positive(
        self, mock_create, mock_load, mock_no_tool, mock_agent, configs
    ):
        cp, cr, cpred, cu = configs
        cp = cp.model_copy(update={"baseline": "no_tool", "num_iterations": 3})
        cpred = cpred.model_copy(update={"temperature": 0.7})
        mock_load.return_value = [_fake_task() for _ in range(2)]
        mock_no_tool.return_value = _stub_response()
        mock_create.return_value = MagicMock()

        workflow_evaluation_pipeline(cp, cr, cpred, cu)

        assert mock_no_tool.call_count == 6
        out = Path(cp.output_folder)
        for i in range(3):
            lines = (out / f"results_iter{i}.jsonl").read_text().splitlines()
            assert len(lines) == 2
            record = json.loads(lines[0])
            assert record["iteration"] == i

    def test_temperature_zero_collapses_to_single_iteration(
        self, mock_create, mock_load, mock_no_tool, mock_agent, configs
    ):
        cp, cr, cpred, cu = configs
        cp = cp.model_copy(update={"baseline": "no_tool", "num_iterations": 3})
        # cpred.temperature defaults to 0.0
        mock_load.return_value = [_fake_task() for _ in range(2)]
        mock_no_tool.return_value = _stub_response()
        mock_create.return_value = MagicMock()

        workflow_evaluation_pipeline(cp, cr, cpred, cu)

        assert mock_no_tool.call_count == 2
        out = Path(cp.output_folder)
        assert (out / "results_iter0.jsonl").exists()
        assert not (out / "results_iter1.jsonl").exists()


def test_load_completed_pairs_reads_all_iterations(tmp_path):
    from conversation2sql.eval_framework.main_pipe_workflow import _load_completed_pairs

    (tmp_path / "results_iter0.jsonl").write_text(
        json.dumps({"instance_id": "a", "iteration": 0}) + "\n"
        + json.dumps({"instance_id": "b", "iteration": 0}) + "\n"
        + "{ this is a truncated line\n"  # crash can leave a partial trailing line
    )
    (tmp_path / "results_iter1.jsonl").write_text(
        json.dumps({"instance_id": "a", "iteration": 1}) + "\n"
    )

    pairs = _load_completed_pairs(tmp_path, num_iterations=2)
    assert pairs == {("a", 0), ("b", 0), ("a", 1)}


def test_load_completed_pairs_missing_files_return_empty(tmp_path):
    from conversation2sql.eval_framework.main_pipe_workflow import _load_completed_pairs

    assert _load_completed_pairs(tmp_path, num_iterations=3) == set()


def test_saved_snapshot_roundtrips_through_parser(tmp_path):
    from conversation2sql.eval_framework.main_pipe_workflow import _save_configs_as_yaml
    from conversation2sql.cli_parser import PydanticParser

    cp = ConfigPipeline(output_folder=str(tmp_path), baseline="tools_user")
    cr = ConfigReader()
    cpred = ConfigPredictor(model_name="some/model")
    cu = ConfigUserSimulator(model_name="user/model", model_provider="openai")
    _save_configs_as_yaml(tmp_path, cp, cr, cpred, cu)

    parser = PydanticParser(
        [ConfigPipeline, ConfigReader, ConfigPredictor, ConfigUserSimulator]
    )
    rp, rr, rpred, ruser = parser.parse_args_and_config(
        ["--config", str(tmp_path / "config.yaml")]
    )
    # The user-simulator section must survive the round-trip.
    assert ruser.model_name == "user/model"
    assert ruser.model_provider == "openai"
    assert rpred.model_name == "some/model"
    assert rp.baseline == "tools_user"
