"""End-to-end smoke tests for the four baselines.

Gated: run with `uv run pytest tests/eval_framework/integration/ -m integration`.
Requires:
- Postgres containers from `.devcontainer/docker-compose.yml` running
- API keys for the configured provider in `.env`
- The bird-interact dataset on disk
"""
import pytest

from conversation2sql.config_input import (
    ConfigPipeline, ConfigReader, ConfigPredictor, ConfigUserSimulator,
)
from conversation2sql.eval_framework.main_pipe_workflow import (
    workflow_evaluation_pipeline,
)


REQUIRED_KEYS = {
    "execution_accuracy", "total_cost", "total_tokens", "tool_calls_in_order",
    "messages", "config_pipeline",
}


@pytest.mark.integration
@pytest.mark.parametrize("baseline", ["no_tool", "tools_only", "tools_user", "bird_full"])
def test_baseline_runs_one_task_and_emits_expected_shape(baseline, tmp_path):
    cp = ConfigPipeline(debug=True, baseline=baseline, output_folder=str(tmp_path))
    cr = ConfigReader()  # _resolve_baseline_settings overrides make_data_ambiguous
    cpred = ConfigPredictor()
    cu = ConfigUserSimulator()

    results = workflow_evaluation_pipeline(cp, cr, cpred, cu)

    assert len(results) == 1
    record = results[0]
    missing = REQUIRED_KEYS - record.keys()
    assert not missing, f"Missing keys for {baseline}: {missing}"
    assert isinstance(record["execution_accuracy"], bool)
