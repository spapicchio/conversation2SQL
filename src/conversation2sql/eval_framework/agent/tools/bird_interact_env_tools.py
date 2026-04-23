"""
Bird-Interact environment tools.

These tools correspond to the `Environment` interaction object defined in the
Bird-Interact agent prompt.  Each tool maps one-to-one to an action from the
original text-based action space, with its patience cost documented in the
tool description so the LLM sees it via normal tool-schema injection.

All tools receive a ``ToolRuntime[UserContext, AgentState]`` injected by
LangGraph.  The runtime carries:
  - ``runtime.context``  → ``UserContext`` (db_dsn, knowledge, column meanings)
  - ``runtime.state``    → ``AgentState``  (user_patience counter)

Cost summary (mirrors the original prompt):
    execute_sql                      → 1 patience
    get_schema                       → 1 patience
    get_all_column_meanings          → 1 patience
    get_column_meaning               → 0.5 patience
    get_all_external_knowledge_names → 0.5 patience
    get_knowledge_definition         → 0.5 patience
    get_all_knowledge_definitions    → 1 patience
"""

import json
import re

import psycopg2
from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime
from pydantic import BaseModel

from conversation2sql.eval_framework.agent.agent_code_state import CustomAgentState
from conversation2sql.eval_framework.agent.tools.utils_db_execute import _execute_query, _format_result
from conversation2sql.eval_framework.state import TaskData

MAX_RESULT_LENGTH = 500

DB_TOOL_COSTS: dict[str, float] = {
    "execute_sql": 1.0,
    "get_schema": 1.0,
    "get_all_column_meanings": 1.0,
    "get_column_meaning": 0.5,
    "get_all_external_knowledge_names": 0.5,
    "get_knowledge_definition": 0.5,
    "get_all_knowledge_definitions": 1.0,
}

KNOWLEDGE_VISIBLE_FIELDS = ["id", "knowledge", "description", "definition"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
class ExecuteSQLResponse(BaseModel):
    result: str
    success: bool
    error: str | None = None


# ---------------------------------------------------------------------------
# Database execution tools
# ---------------------------------------------------------------------------


@tool
def execute_sql(sql: str, runtime: ToolRuntime[TaskData, CustomAgentState]) -> str:
    """Execute a SQL query against the PostgreSQL database and return the results.
    Use this to explore the database, test queries, or verify your SQL before submitting.
    Cost: 1 bird-coin.

    Args:
        sql: The PostgreSQL SQL query to execute.

    Returns:
        The query results formatted as a table, or an error message.
    """
    db_dsn = runtime.context.db_dsn

    sql_cleaned = re.sub(r'--.*$', '', sql, flags=re.MULTILINE)
    sql_cleaned = re.sub(r'/\*.*?\*/', '', sql_cleaned, flags=re.DOTALL)
    sql_upper = sql_cleaned.strip().upper()
    if not sql_upper.startswith(("SELECT", "WITH", "EXPLAIN")):
        execute_sql_res = ExecuteSQLResponse(result="", success=False,
                                             error="Only SELECT queries allowed in execute_sql")
        return json.dumps(execute_sql_res.model_dump_json(), indent=2)

    try:
        result, desc = _execute_query(query=sql, db_dsn=db_dsn)
        format_result = _format_result(result, desc)
        execute_sql_res = ExecuteSQLResponse(result=format_result[:MAX_RESULT_LENGTH], success=True)

    except psycopg2.errors.QueryCanceled:
        execute_sql_res = ExecuteSQLResponse(result="", success=False, error="SQL execution timed out")
    except psycopg2.DatabaseError as e:
        execute_sql_res = ExecuteSQLResponse(result="", success=False, error=f"DatabaseError SQL error: {str(e)}")

    return json.dumps(execute_sql_res.model_dump_json(), indent=2)


@tool
def get_schema(runtime: ToolRuntime[TaskData, CustomAgentState]) -> str:
    """Get the full database schema (CREATE TABLE statements) for the current task's database.
    Cost: 1 bird-coin.

    Returns:
        The database schema as text.
    """
    schema = runtime.context.ddl_database_schema
    return json.dumps({'schema': schema}, indent=2)


# ---------------------------------------------------------------------------
# Column meaning tools
# ---------------------------------------------------------------------------
@tool
def get_all_column_meanings(runtime: ToolRuntime[TaskData, CustomAgentState]) -> str:
    """Get the meanings/descriptions of all columns in the database.
    Cost: 1 bird-coin.

    Returns:
        JSON string with column meanings for all tables.
    """
    column_meanings = runtime.context.column_meanings
    #  key = f"{db_name}|{req.table_name.lower()}|{req.column_name.lower()}"
    output = {k: v.model_dump(exclude_none=True) for k, v in column_meanings.items()}
    return {"column_meanings": json.dumps(output, indent=2)}


@tool
def get_column_meaning(
        table_name: str, column_name: str, runtime: ToolRuntime[TaskData, CustomAgentState]
) -> str:
    """Get the meaning/description of a specific column in a table.
    Cost: 0.5 bird-coins.

    Args:
        table_name: Name of the table.
        column_name: Name of the column.

    Returns:
        The column meaning/description.
    """
    db_name = runtime.context.selected_database
    key = f"{db_name}|{table_name.lower()}|{column_name.lower()}"
    meaning = runtime.context.column_meanings.get(key, "Column meaning not found")
    return json.dumps({
        "meaning": meaning
        if isinstance(meaning, str) else meaning.model_dump(exclude_none=True)
    }, indent=2)


# ---------------------------------------------------------------------------
# External knowledge tools
# ---------------------------------------------------------------------------

@tool
def get_all_external_knowledge_names(
        runtime: ToolRuntime[TaskData, CustomAgentState],
) -> str:
    """Get the names of all available external knowledge entries for this database.
    Use this to discover what domain knowledge is available.
    Cost: 0.5 bird-coins.

    Returns:
        JSON list of knowledge entry names.
    """
    masked_agent_kb = runtime.context.masked_agent_kb
    output = {
        "names": list(masked_agent_kb.keys())
    }
    return json.dumps(output, indent=2)


@tool
def get_knowledge_definition(
        knowledge_name: str,
        runtime: ToolRuntime[TaskData, CustomAgentState],
) -> str:
    """Get the definition/details of a specific external knowledge entry.
    Cost: 0.5 bird-coins.

    Args:
        knowledge_name: The name of the knowledge entry to look up.

    Returns:
        JSON string with the knowledge definition.
    """
    # Note that for Ambiguous query with KB ambiguity this is masked
    if knowledge_name in runtime.context.masked_agent_kb:
        # https://github.com/bird-bench/BIRD-Interact/blob/451fe2c3518ee1cf908d8139e2913483bd519381/BIRD-Interact-ADK/db_environment/server.py#L29
        kb_entry = runtime.context.masked_agent_kb[knowledge_name].model_dump(include=set(KNOWLEDGE_VISIBLE_FIELDS))
        return json.dumps({"knowledge": kb_entry}, indent=2)

    return json.dumps({"knowledge": "Knowledge not found."}, indent=2)


@tool
def get_all_knowledge_definitions(
        runtime: ToolRuntime[TaskData, CustomAgentState],
) -> str:
    """Return all external knowledge with definitions (cost: 1 patience)."""
    # https://github.com/bird-bench/BIRD-Interact/blob/451fe2c3518ee1cf908d8139e2913483bd519381/BIRD-Interact-ADK/db_environment/server.py#L29
    dump_kb = []
    for knowledge_name in runtime.context.masked_agent_kb:
        kb_entry = runtime.context.masked_agent_kb[knowledge_name].model_dump(include=set(KNOWLEDGE_VISIBLE_FIELDS))
        dump_kb.append(kb_entry)
    return json.dumps({"knowledge": dump_kb}, indent=2)
