"""Agent prompt templates — inline Jinja2 strings with typed Pydantic params.

Each prompt family exposes:
  - A ``*Params`` Pydantic model with the required template variables.
  - A ``build_*_messages()`` function that renders and returns ``list[BaseMessage]``.

Template variables use ``{{ jinja2 }}`` delimiters.
"""

from __future__ import annotations

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
{% if enable_psql_console %}- psql_console: run ONE PostgreSQL statement or one read-only psql meta-command (\\dt list tables, \\d <table> describe, \\l list databases, \\df list functions). Read-only session. Cost: 1
{% else %}- execute_sql: execute a PostgreSQL query. Cost: 1
- get_schema: get the full database schema. Cost: 1
{% if enable_table_schema_tools %}- get_table_names: list all table names in the database. Cost: 0.5
- get_table_schema: get one table's schema (CREATE TABLE, sample rows, and foreign keys to joinable tables). Cost: 0.5
{% endif %}{% endif %}- get_all_column_meanings: get all column meanings. Cost: 1
- get_column_meaning: get the meaning of one column. Cost: 0.5
- get_all_external_knowledge_names: get all external knowledge names. Cost: 0.5
- get_knowledge_definition: get one external knowledge definition along with the knowledge it depends on (its prerequisites). Cost: 0.5
- get_all_knowledge_definitions: get all external knowledge definitions. Cost: 1
{% if enable_ask_user %}- ask_user: ask the user a clarification question. Cost: 2
{% endif %}- submit_sql: submit the SQL for evaluation. Cost: 3

Important strategy tips:
- First explore the database schema, column meanings, and relevant external knowledge to understand the task.
{% if enable_ask_user %}- If the user's intent is ambiguous, ask clarifying questions to figure out the real intent before committing to SQL.
- Ask one clarification question at a time.
{% endif %}- Be efficient with your actions to conserve budget.
- Make sure the submitted SQL is valid and addresses all aspects of the question.
- Keep track of the remaining budget and prioritize actions accordingly.
- Be careful with broad retrieval tools such as get_all_column_meanings and get_all_knowledge_definitions because they may return a long context.
- Test SQL with {{ "psql_console" if enable_psql_console else "execute_sql" }} before submit_sql when useful.
- If a submission fails and budget remains, debug and try again.
{% if enable_ask_user %}- After a successful phase-1 submission, you may receive a follow-up question for phase 2.
{% endif %}"""

_BIRD_AGENT_USER = """
User's Question: 
{{ amb_user_query }}

[SYSTEM NOTE: You have a total action budget of {{ total_budget }} units. Each action consumes budget. If the budget runs out, you must submit.]
"""


def build_bird_interact_agent_messages(
        params: dict,
) -> list[dict]:
    return utils_build_messages(_BIRD_AGENT_SYSTEM, _BIRD_AGENT_USER, params)
