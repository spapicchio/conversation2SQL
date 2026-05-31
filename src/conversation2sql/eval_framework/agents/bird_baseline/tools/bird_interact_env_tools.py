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

Each langchain ``@tool`` is a thin wrapper that pulls the relevant fields out
of ``runtime.context`` and delegates to a plain Python ``*_impl`` function.
The ``*_impl`` functions hold all the real logic and are unit-tested directly
in ``tests/eval_framework/tools/`` without needing the LangGraph runtime.

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

import psycopg2
from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime
from pydantic import BaseModel

from conversation2sql.eval_framework.agents.bird_baseline.agent_code_state import CustomAgentState
from conversation2sql.eval_framework.agents.bird_baseline.tools.utils import remove_comments
from conversation2sql.eval_framework.agents.bird_baseline.tools.utils_db_execute import (
    _execute_query,
    _format_result,
)
from conversation2sql.eval_framework.agents.utils_kb_linearize import (
    format_entry_line,
    linearize_kb,
)
from conversation2sql.eval_framework.state import (
    ColumnMeaningEntry,
    ExternalKnowledgeEntry,
    TaskData,
)

MAX_RESULT_LENGTH = 500

DB_TOOL_COSTS: dict[str, float] = {
    "execute_sql": 2.0,
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
# Pure implementation functions (testable without LangGraph runtime)
# ---------------------------------------------------------------------------
def execute_sql_impl(sql: str, db_dsn: str) -> ExecuteSQLResponse:
    sql_upper = remove_comments(sql).upper()
    if not sql_upper.startswith(("SELECT", "WITH", "EXPLAIN")):
        return ExecuteSQLResponse(
            result="",
            success=False,
            error="Only SELECT queries allowed in execute_sql",
        )

    try:
        result, desc = _execute_query(query=sql, db_dsn=db_dsn)
        format_result = _format_result(result, desc)
        return ExecuteSQLResponse(
            result=f"{format_result[:MAX_RESULT_LENGTH]}",
            success=True,
        )

    except psycopg2.extensions.QueryCanceledError:
        return ExecuteSQLResponse(
            result="Error: SQL execution timed out.",
            success=False,
            error="SQL execution timed out",
        )

    except psycopg2.DatabaseError as e:
        return ExecuteSQLResponse(
            result=f"Error: Database error occurred: {str(e)}.",
            success=False,
            error=str(e),
        )


def get_schema_impl(ddl_database_schema: str) -> dict:
    return {"schema": ddl_database_schema}


def get_all_column_meanings_impl(
        column_meanings: dict[str, ColumnMeaningEntry],
) -> dict:
    output = {
        k: v.model_dump_json(exclude_none=True) for k, v in column_meanings.items()
    }
    return {"column_meanings": output}


def get_column_meaning_impl(
        table_name: str,
        column_name: str,
        db_name: str,
        column_meanings: dict[str, ColumnMeaningEntry],
) -> dict:
    key = f"{db_name}|{table_name.lower()}|{column_name.lower()}"
    meaning = column_meanings.get(key, "Column meaning not found")
    return {
        "meaning": meaning
        if isinstance(meaning, str)
        else meaning.model_dump_json(exclude_none=True)
    }


def get_all_external_knowledge_names_impl(
        masked_agent_kb: dict[str, ExternalKnowledgeEntry],
) -> dict:
    return {"names": list(masked_agent_kb.keys())}


def get_knowledge_definition_impl(
        knowledge_name: str,
        masked_agent_kb: dict[str, ExternalKnowledgeEntry],
) -> dict:
    if knowledge_name in masked_agent_kb:
        kb_entry = masked_agent_kb[knowledge_name].model_dump_json(
            include=set(KNOWLEDGE_VISIBLE_FIELDS)
        )
        return {"knowledge": kb_entry}
    return {"knowledge": "Knowledge not found."}


def get_all_knowledge_definitions_impl(
        masked_agent_kb: dict[str, ExternalKnowledgeEntry],
) -> dict:
    dump_kb = []
    for knowledge_name in masked_agent_kb:
        kb_entry = masked_agent_kb[knowledge_name].model_dump_json(
            include=set(KNOWLEDGE_VISIBLE_FIELDS)
        )
        dump_kb.append(kb_entry)
    return {"knowledge": dump_kb}


# ---------------------------------------------------------------------------
# Database execution tools (langchain wrappers)
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
    response = execute_sql_impl(sql=sql, db_dsn=runtime.context.db_dsn)
    return response.model_dump_json()


@tool
def get_schema(runtime: ToolRuntime[TaskData, CustomAgentState]) -> str:
    """Get the full database schema (CREATE TABLE statements) for the current task's database.
    Cost: 1 bird-coin.

    Returns:
        The database schema as text.
    """
    return json.dumps(
        get_schema_impl(ddl_database_schema=runtime.context.ddl_database_schema),
        indent=2,
    )


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
    return json.dumps(
        get_all_column_meanings_impl(column_meanings=runtime.context.column_meanings),
        indent=2,
    )


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
    return json.dumps(
        get_column_meaning_impl(
            table_name=table_name,
            column_name=column_name,
            db_name=runtime.context.selected_database,
            column_meanings=runtime.context.column_meanings,
        ),
        indent=2,
    )


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
    return json.dumps(
        get_all_external_knowledge_names_impl(masked_agent_kb=runtime.context.masked_agent_kb),
        indent=2,
    )


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
    if runtime.context.is_kb_linearized:
        line = format_entry_line(knowledge_name, runtime.context.masked_agent_kb)
        return json.dumps({"knowledge": line}, indent=2)
    return json.dumps(
        get_knowledge_definition_impl(
            knowledge_name=knowledge_name,
            masked_agent_kb=runtime.context.masked_agent_kb,
        ),
        indent=2,
    )


@tool
def get_all_knowledge_definitions(
        runtime: ToolRuntime[TaskData, CustomAgentState],
) -> str:
    """Return all external knowledge with definitions (cost: 1 patience)."""
    if runtime.context.is_kb_linearized:
        flat = linearize_kb(runtime.context.masked_agent_kb)
        return json.dumps({"knowledge": flat}, indent=2)
    return json.dumps(
        get_all_knowledge_definitions_impl(
            masked_agent_kb=runtime.context.masked_agent_kb,
        ),
        indent=2,
    )
