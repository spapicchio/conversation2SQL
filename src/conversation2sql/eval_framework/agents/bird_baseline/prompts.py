"""Agent prompt templates — inline Jinja2 strings with typed Pydantic params.

Each prompt family exposes:
  - A ``*Params`` Pydantic model with the required template variables.
  - A ``build_*_messages()`` function that renders and returns ``list[BaseMessage]``.

Template variables use ``{{ jinja2 }}`` delimiters.
"""

from __future__ import annotations

from conversation2sql.eval_framework.agents.bird_baseline.tools import TOOL_SPECS
from conversation2sql.eval_framework.agents.bird_baseline.tools.tool_specs import (
    format_cost,
)
from conversation2sql.eval_framework.agents.utils import utils_build_messages

# =============================================================================
# Bird-Interact Agent
# =============================================================================

_BIRD_AGENT_SYSTEM = """
You are a helpful PostgreSQL agent that interacts with a {{ "user and a " if enable_ask_user else "" }}database to solve the user's question.

Task description:
Your goal is to understand the user's {{ "ambiguous " if enable_ask_user else "" }}question{{ " involving external knowledge retrieval" if enable_ask_user else "" }} and generate the correct SQL query to solve it.
You can:
{% if enable_ask_user %}1. Interact with the user to ask clarifying questions or submit the SQL query.
2. Interact with the database environment to explore the database and retrieve relevant information.{% else %}1. Interact with the database environment to explore the database and retrieve relevant information.
2. Submit the SQL query when ready.{% endif %}

The interaction ends when you submit the correct SQL query or the budget runs out.
Each action costs bird-coins, so you should be efficient.

Available tools and costs:
{% if enable_psql_console %}- psql_console: {{ tool_specs['psql_console'].summary }}. Cost: {{ tool_specs['psql_console'].cost }}
{% else %}- execute_sql: {{ tool_specs['execute_sql'].summary }}. Cost: {{ tool_specs['execute_sql'].cost }}
- get_schema: {{ tool_specs['get_schema'].summary }}. Cost: {{ tool_specs['get_schema'].cost }}
{% if enable_table_schema_tools %}- get_table_names: {{ tool_specs['get_table_names'].summary }}. Cost: {{ tool_specs['get_table_names'].cost }}
- get_table_schema: {{ tool_specs['get_table_schema'].summary }}. Cost: {{ tool_specs['get_table_schema'].cost }}
{% endif %}{% endif %}{% if enable_python_udf %}- create_python_udf: {{ tool_specs['create_python_udf'].summary }}. Cost: {{ tool_specs['create_python_udf'].cost }}
{% endif %}- get_all_column_meanings: {{ tool_specs['get_all_column_meanings'].summary }}. Cost: {{ tool_specs['get_all_column_meanings'].cost }}
- get_column_meaning: {{ tool_specs['get_column_meaning'].summary }}. Cost: {{ tool_specs['get_column_meaning'].cost }}
- get_all_external_knowledge_names: {{ tool_specs['get_all_external_knowledge_names'].summary }}. Cost: {{ tool_specs['get_all_external_knowledge_names'].cost }}
- get_knowledge_definition: {{ tool_specs['get_knowledge_definition'].summary }}. Cost: {{ tool_specs['get_knowledge_definition'].cost }}
- get_all_knowledge_definitions: {{ tool_specs['get_all_knowledge_definitions'].summary }}. Cost: {{ tool_specs['get_all_knowledge_definitions'].cost }}
{% if enable_ask_user %}- ask_user: {{ tool_specs['ask_user'].summary }}. Cost: {{ tool_specs['ask_user'].cost }}
{% endif %}- submit_sql: {{ tool_specs['submit_sql'].summary }}. Cost: {{ tool_specs['submit_sql'].cost }}

Important strategy tips:
{% if enable_psql_console %}- First explore the database with psql_console: use \\dt to list tables and \\d+ <table> to inspect a table's columns, foreign keys with descriptions, then check column meanings and relevant external knowledge to understand the task.
{% else %}- First explore the database schema, column meanings, and relevant external knowledge to understand the task.
{% endif %}{% if enable_ask_user %}- If the user's intent is ambiguous, ask clarifying questions to figure out the real intent before committing to SQL.
- Ask one clarification question at a time.
{% endif %}- Be efficient with your actions to conserve budget.
- Make sure the submitted SQL is valid and addresses all aspects of the question.
- Keep track of the remaining budget and prioritize actions accordingly.
- Be careful with broad retrieval tools such as get_all_column_meanings and get_all_knowledge_definitions because they may return a long context.
- Test SQL with {{ "psql_console" if enable_psql_console else "execute_sql" }} before submit_sql when useful.
- If a submission fails and budget remains, debug and try again.
{% if enable_python_udf %}- Use create_python_udf to implement complex formulas as Python functions, then call the returned name in execute_sql or psql_console.
{% endif %}{% if enable_ask_user %}- After a successful phase-1 submission, you may receive a follow-up question for phase 2.
{% endif %}"""

_BIRD_AGENT_USER = """
User's Question: 
{{ amb_user_query }}

[SYSTEM NOTE: You have a total action budget of {{ total_budget }} units. Each action consumes budget. If the budget runs out, you must submit.]
"""


def build_bird_interact_agent_messages(
        params: dict,
) -> list[dict]:
    # Render the tool list straight from TOOL_SPECS (the single source of truth
    # for each tool's cost + summary) instead of hardcoding the wording or the
    # numbers in the template, where they could drift out of sync.
    params = {
        **params,
        "tool_specs": {
            name: {"summary": spec.summary, "cost": format_cost(spec.cost)}
            for name, spec in TOOL_SPECS.items()
        },
    }
    return utils_build_messages(_BIRD_AGENT_SYSTEM, _BIRD_AGENT_USER, params)
