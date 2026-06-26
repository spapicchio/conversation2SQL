from conversation2sql.eval_framework.agents.deep_agent.agent_code_state import (
    DeepAgentCustomState,
)
from conversation2sql.eval_framework.agents.deep_agent.filesystem_seed import (
    FS_TOOL_COSTS,
    deep_tool_costs,
)


def test_merged_state_has_files_and_patience_keys():
    keys = DeepAgentCustomState.__annotations__.keys()
    assert "files" in keys
    assert "updated_user_patience" in keys
    assert "tool_called_patience" in keys


def test_read_only_costs_exclude_write_tools():
    costs = deep_tool_costs(enable_fs_write=False)
    assert "read_file" in costs and "ls" in costs
    assert "write_file" not in costs and "edit_file" not in costs
    # reused tools are accounted for
    assert "execute_sql" in costs and "ask_user" in costs and "submit_sql" in costs


def test_fs_write_costs_include_write_tools():
    costs = deep_tool_costs(enable_fs_write=True)
    assert "write_file" in costs and "edit_file" in costs


def test_fs_tool_costs_are_positive():
    assert all(v > 0 for v in FS_TOOL_COSTS.values())
