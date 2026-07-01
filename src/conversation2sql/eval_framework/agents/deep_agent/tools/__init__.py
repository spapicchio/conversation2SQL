# Single source of truth: per-tool cost + prompt summary. TOOL_COSTS is derived
# for the legacy callers (middleware, metrics) that only need the number.
from conversation2sql.eval_framework.agents.bird_baseline.tools import USER_TOOL_SPECS
from conversation2sql.eval_framework.agents.deep_agent.tools.bash_tool import BASH_TOOL_SPECS
from conversation2sql.eval_framework.agents.tool_specs import ToolSpec

DA_TOOL_SPECS: dict[str, ToolSpec] = {**BASH_TOOL_SPECS, **USER_TOOL_SPECS}
DA_TOOL_COSTS: dict[str, float] = {name: spec.cost for name, spec in DA_TOOL_SPECS.items()}