from conversation2sql.config_input import ConfigPipeline, ConfigReader
from conversation2sql.eval_framework.state import TaskData


def test_config_reader_drops_removed_deep_flags():
    cfg = ConfigReader()
    assert not hasattr(cfg, "deep_enable_subagents")
    assert not hasattr(cfg, "deep_enable_todos")
    assert not hasattr(cfg, "deep_enable_fs_write")


def test_pipeline_accepts_deep_agent_baseline():
    cfg = ConfigPipeline(baseline="deep_agent")
    assert cfg.baseline == "deep_agent"


def test_config_reader_deep_catalog_root_default_is_lite_catalog():
    cfg = ConfigReader()
    assert cfg.deep_catalog_root == "data/bird_interact/catalog_bird_interact_lite"


def test_taskdata_carries_deep_catalog_root(make_minimal_task_kwargs):
    task = TaskData(**make_minimal_task_kwargs(deep_catalog_root="/some/root"))
    assert task.deep_catalog_root == "/some/root"
