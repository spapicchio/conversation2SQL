from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from conversation2sql.eval_framework.main_pipe_workflow import (
    _resolve_baseline_settings,
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
    }
    return t


def _stub_response():
    return {
        "messages": [], "initial_user_patience": 0, "updated_user_patience": 0,
        "tool_called_patience": [], "total_cost": 0, "total_tokens": 0,
        "total_prompt_tokens": 0, "total_completion_tokens": 0,
        "mean_prompt_tokens": 0, "mean_completion_tokens": 0,
        "tool_calls_in_order": [], "execution_accuracy": False,
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

    def test_output_path_includes_baseline(
        self, mock_create, mock_load, mock_no_tool, mock_agent, configs, tmp_path,
    ):
        cp, cr, cpred, cu = configs
        cp.baseline = "tools_only"
        mock_load.return_value = [_fake_task()]
        mock_agent.return_value = _stub_response()
        mock_create.return_value = MagicMock()

        workflow_evaluation_pipeline(cp, cr, cpred, cu)

        matches = list(Path(tmp_path).glob("tools_only/**/results.jsonl"))
        assert len(matches) == 1
