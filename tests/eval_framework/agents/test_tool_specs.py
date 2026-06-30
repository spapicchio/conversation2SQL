"""Tests for the single-source ToolSpec registry.

Each tool's cost and short description live once in a ToolSpec; the agent
prompt's tool list and each tool's schema description (the LLM-facing
docstring) are both derived from it, so they cannot drift.
"""
from types import SimpleNamespace

from conversation2sql.eval_framework.agents.bird_baseline.tools import (
    TOOL_SPECS,
    TOOL_COSTS,
    execute_sql,
    psql_console,
    create_python_udf,
    get_schema,
    get_table_names,
    get_table_schema,
    get_all_column_meanings,
    get_column_meaning,
    get_all_external_knowledge_names,
    get_knowledge_definition,
    get_all_knowledge_definitions,
    submit_sql,
    return_tool_ask_user,
)
from conversation2sql.eval_framework.agents.bird_baseline.tools.tool_specs import (
    ToolSpec,
    format_cost,
    stamp_cost_in_descriptions,
)
from conversation2sql.eval_framework.agents.bird_baseline.prompts import (
    build_bird_interact_agent_messages,
)


_MODULE_TOOLS = [
    execute_sql,
    psql_console,
    create_python_udf,
    get_schema,
    get_table_names,
    get_table_schema,
    get_all_column_meanings,
    get_column_meaning,
    get_all_external_knowledge_names,
    get_knowledge_definition,
    get_all_knowledge_definitions,
    submit_sql,
]


# --------------------------------------------------------------------------
# Registry shape / back-compat
# --------------------------------------------------------------------------
def test_tool_costs_derived_from_specs():
    assert TOOL_COSTS == {name: spec.cost for name, spec in TOOL_SPECS.items()}
    # Back-compat: the numbers every downstream consumer relies on are unchanged.
    assert TOOL_COSTS["execute_sql"] == 1.0
    assert TOOL_COSTS["get_table_names"] == 0.5
    assert TOOL_COSTS["ask_user"] == 2.0
    assert TOOL_COSTS["submit_sql"] == 3.0


def test_every_tool_has_a_spec():
    names = {t.name for t in _MODULE_TOOLS}
    names.add("ask_user")  # built lazily by the factory
    assert names == set(TOOL_SPECS)


# --------------------------------------------------------------------------
# format_cost
# --------------------------------------------------------------------------
def test_format_cost_strips_trailing_zero():
    assert format_cost(1.0) == "1"
    assert format_cost(3.0) == "3"
    assert format_cost(0.5) == "0.5"


# --------------------------------------------------------------------------
# stamp mechanism (single-sourced cost on the tool schema description)
# --------------------------------------------------------------------------
def test_stamp_appends_cost_from_spec():
    tool = SimpleNamespace(name="x", description="do a thing.")
    stamp_cost_in_descriptions([tool], {"x": ToolSpec("x", 2.0, "do a thing")})
    assert "Cost: 2 bird-coins." in tool.description


def test_stamp_is_idempotent():
    tool = SimpleNamespace(name="x", description="do a thing.")
    specs = {"x": ToolSpec("x", 2.0, "do a thing")}
    stamp_cost_in_descriptions([tool], specs)
    stamp_cost_in_descriptions([tool], specs)
    assert tool.description.count("Cost:") == 1


# --------------------------------------------------------------------------
# Real tools: cost present in the schema description exactly once, from spec
# --------------------------------------------------------------------------
def test_module_tools_have_single_cost_line_from_spec():
    for tool in _MODULE_TOOLS:
        cost = format_cost(TOOL_SPECS[tool.name].cost)
        assert tool.description.count("Cost:") == 1, tool.name
        assert f"Cost: {cost} bird-coins." in tool.description, tool.name


def test_ask_user_description_carries_cost():
    tool = return_tool_ask_user(None, None)
    assert tool.description.count("Cost:") == 1
    assert "Cost: 2 bird-coins." in tool.description


# --------------------------------------------------------------------------
# Prompt renders the tool list straight from the specs
# --------------------------------------------------------------------------
def _system_prompt(**params) -> str:
    base = {
        "total_budget": 20,
        "amb_user_query": "q",
        "enable_ask_user": True,
        "enable_table_schema_tools": True,
        "enable_psql_console": False,
    }
    base.update(params)
    return build_bird_interact_agent_messages(base)[0]["content"]


def test_prompt_lists_each_tool_summary_and_cost_from_specs():
    text = _system_prompt()
    for name in [
        "execute_sql",
        "get_schema",
        "get_table_names",
        "get_column_meaning",
        "get_knowledge_definition",
        "ask_user",
        "submit_sql",
    ]:
        spec = TOOL_SPECS[name]
        line = f"{name}: {spec.summary}. Cost: {format_cost(spec.cost)}"
        assert line in text, name


def test_prompt_psql_mode_uses_psql_spec_summary():
    text = _system_prompt(enable_psql_console=True)
    spec = TOOL_SPECS["psql_console"]
    assert f"psql_console: {spec.summary}. Cost: {format_cost(spec.cost)}" in text
    assert "execute_sql" not in text
