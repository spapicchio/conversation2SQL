"""Single source of truth: per-tool cost + prompt summary for maintenance_agent."""
from conversation2sql.eval_framework.agents.deep_agent.tools.bash_tool import BASH_TOOL_SPECS
from conversation2sql.eval_framework.agents.maintenance_agent.tools.maintenance_tools import (
    MAINTENANCE_TOOL_SPECS,
    return_tool_comment_on_issue,
    return_tool_run_tests,
    return_tool_write_query,
    run_tests_impl,
    submit,
)
from conversation2sql.eval_framework.agents.tool_specs import ToolSpec

MA_TOOL_SPECS: dict[str, ToolSpec] = {**BASH_TOOL_SPECS, **MAINTENANCE_TOOL_SPECS}
MA_TOOL_COSTS: dict[str, float] = {name: spec.cost for name, spec in MA_TOOL_SPECS.items()}

__all__ = [
    "MA_TOOL_SPECS",
    "MA_TOOL_COSTS",
    "return_tool_write_query",
    "return_tool_run_tests",
    "return_tool_comment_on_issue",
    "run_tests_impl",
    "submit",
]
