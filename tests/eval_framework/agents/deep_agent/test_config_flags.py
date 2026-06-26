from conversation2sql.config_input import ConfigPipeline, ConfigReader
from conversation2sql.eval_framework.state import TaskData


def test_config_reader_deep_flags_default_false():
    cfg = ConfigReader()
    assert cfg.deep_enable_todos is False
    assert cfg.deep_enable_subagents is False
    assert cfg.deep_enable_summarization is False
    assert cfg.deep_enable_fs_write is False


def test_pipeline_accepts_deep_agent_baseline():
    cfg = ConfigPipeline(baseline="deep_agent")
    assert cfg.baseline == "deep_agent"


def test_taskdata_carries_deep_flags(make_minimal_task_kwargs):
    task = TaskData(**make_minimal_task_kwargs())
    assert task.deep_enable_todos is False
    assert task.deep_enable_fs_write is False
