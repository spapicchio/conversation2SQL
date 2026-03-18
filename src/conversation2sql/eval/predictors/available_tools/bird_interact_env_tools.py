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
from typing import Any

import psycopg2
import psycopg2.extras
from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime

from conversation2sql.eval.interfaces import ToolUserContext
from conversation2sql.eval.predictors.available_tools._patience_utils import deduct_and_note
from conversation2sql.eval.predictors.langchain_agent_factory import CustomAgentState
from conversation2sql.eval.registry import tool_registry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _connect(db_dsn: str) -> psycopg2.extensions.connection:
    conn = psycopg2.connect(db_dsn, cursor_factory=psycopg2.extras.RealDictCursor)
    conn.set_session(readonly=True, autocommit=True)
    return conn


def _require_db(context: ToolUserContext) -> str:
    if not context.db_dsn:
        raise ValueError("UserContext.db_dsn is not set — cannot execute database tools.")
    return context.db_dsn


# ---------------------------------------------------------------------------
# Database execution tools
# ---------------------------------------------------------------------------

@tool_registry.register(name="execute_sql")
@tool(
    description=(
        "Execute one or more SQL commands against the database and return the fetched rows. "
        "Multiple statements may be separated by semicolons. "
        "Cost: 1 patience."
    )
)
def execute_sql(sql: str, runtime: ToolRuntime[ToolUserContext, CustomAgentState]) -> str:
    """Execute SQL and return results (cost: 1 patience)."""
    note = deduct_and_note(runtime, cost=1)
    db_dsn = _require_db(runtime.context)
    try:
        conn = _connect(db_dsn)
        cursor = conn.cursor()
        results: list[Any] = []
        for statement in sql.split(";"):
            statement = statement.strip()
            if not statement:
                continue
            cursor.execute(statement)
            rows = cursor.fetchall()
            if rows:
                results.append([dict(r) for r in rows])
        conn.close()
        output = json.dumps(results, ensure_ascii=False, default=str) if results else "(no rows returned)"
    except psycopg2.Error as exc:
        output = f"[SQL ERROR] {exc}"
    return output + note


@tool_registry.register(name="get_schema")
@tool(
    description=(
        "Return the database schema in DDL format (CREATE TABLE statements) "
        "together with a few sample rows per table so the agent can understand "
        "available tables and columns at a glance. "
        "Cost: 1 patience."
    )
)
def get_schema(runtime: ToolRuntime[ToolUserContext, CustomAgentState]) -> str:
    """Return the DDL schema with sample rows (cost: 1 patience)."""
    note = deduct_and_note(runtime, cost=1)
    db_dsn = _require_db(runtime.context)
    try:
        conn = _connect(db_dsn)
        cursor = conn.cursor()
        # Fetch all user tables in the public schema
        cursor.execute("""
            SELECT table_name, table_type
            FROM information_schema.tables
            WHERE table_schema = 'public'
              AND table_type IN ('BASE TABLE', 'VIEW')
            ORDER BY table_name
        """)
        tables = cursor.fetchall()
        parts: list[str] = []
        for row in tables:
            table_name: str = row["table_name"]
            table_type: str = row["table_type"]
            # Build a CREATE TABLE / VIEW DDL from column metadata
            cursor.execute("""
                SELECT column_name, data_type, is_nullable, column_default
                FROM information_schema.columns
                WHERE table_schema = 'public' AND table_name = %s
                ORDER BY ordinal_position
            """, (table_name,))
            columns = cursor.fetchall()
            col_defs = []
            for col in columns:
                nullable = "" if col["is_nullable"] == "YES" else " NOT NULL"
                default = f" DEFAULT {col['column_default']}" if col["column_default"] else ""
                col_defs.append(f"    {col['column_name']} {col['data_type']}{nullable}{default}")
            create_kw = "VIEW" if table_type == "VIEW" else "TABLE"
            ddl = f"CREATE {create_kw} {table_name} (\n" + ",\n".join(col_defs) + "\n);"
            parts.append(ddl)
            try:
                cursor.execute(f"SELECT * FROM {table_name} LIMIT 3")  # noqa: S608
                sample_rows = [dict(r) for r in cursor.fetchall()]
                parts.append(f"-- Sample data: {json.dumps(sample_rows, ensure_ascii=False, default=str)}")
            except psycopg2.Error:
                pass
            parts.append("")
        conn.close()
        output = "\n".join(parts).strip() or "(no tables found)"
    except psycopg2.Error as exc:
        output = f"[SCHEMA ERROR] {exc}"
    return output + note


# ---------------------------------------------------------------------------
# Column meaning tools
# ---------------------------------------------------------------------------

@tool_registry.register(name="get_all_column_meanings")
@tool(
    description=(
        "Return the meaning / description of every column in every table of the database. "
        "This can produce a long response — prefer get_column_meaning when you only need "
        "a single column. "
        "Cost: 1 patience."
    )
)
def get_all_column_meanings(runtime: ToolRuntime[ToolUserContext, CustomAgentState]) -> str:
    """Return all column meanings from the dataset metadata (cost: 1 patience)."""
    note = deduct_and_note(runtime, cost=1)
    column_meanings = runtime.context.column_meanings
    if not column_meanings:
        output = "(no column meanings available)"
    else:
        output = json.dumps(column_meanings, indent=2, ensure_ascii=False)
    return output + note


@tool_registry.register(name="get_column_meaning")
@tool(
    description=(
        "Return the meaning / description of a single column in the specified table. "
        "Cost: 0.5 patience."
    )
)
def get_column_meaning(table_name: str, column_name: str, runtime: ToolRuntime[ToolUserContext, CustomAgentState]) -> str:
    """Return the meaning for one column (cost: 0.5 patience)."""
    note = deduct_and_note(runtime, cost=0.5)
    table = runtime.context.column_meanings.get(table_name)
    if table is None:
        return f"(no column meanings found for table '{table_name}')" + note
    meaning = table.get(column_name)
    if meaning is None:
        return f"(no meaning found for column '{table_name}.{column_name}')" + note
    return meaning + note


# ---------------------------------------------------------------------------
# External knowledge tools
# ---------------------------------------------------------------------------

@tool_registry.register(name="get_all_external_knowledge_names")
@tool(
    description=(
        "Return the names of all pieces of external knowledge available for this question. "
        "Use get_knowledge_definition to retrieve the full definition of a specific item. "
        "Cost: 0.5 patience."
    )
)
def get_all_external_knowledge_names(runtime: ToolRuntime[ToolUserContext, CustomAgentState]) -> str:
    """Return a list of all external knowledge names (cost: 0.5 patience)."""
    note = deduct_and_note(runtime, cost=0.5)
    names = [entry.get("knowledge", "") for entry in runtime.context.external_knowledge]
    output = json.dumps(names, ensure_ascii=False) if names else "(no external knowledge available)"
    return output + note


@tool_registry.register(name="get_knowledge_definition")
@tool(
    description=(
        "Return the full definition of a specific piece of external knowledge by its name. "
        "Cost: 0.5 patience."
    )
)
def get_knowledge_definition(knowledge_name: str, runtime: ToolRuntime[ToolUserContext, CustomAgentState]) -> str:
    """Return a single knowledge definition (cost: 0.5 patience)."""
    note = deduct_and_note(runtime, cost=0.5)
    for entry in runtime.context.external_knowledge:
        if entry.get("knowledge") == knowledge_name:
            return json.dumps(entry, indent=2, ensure_ascii=False) + note
    return f"(no knowledge found with name '{knowledge_name}')" + note


@tool_registry.register(name="get_all_knowledge_definitions")
@tool(
    description=(
        "Return all external knowledge names together with their full definitions. "
        "This can produce a long response — prefer get_knowledge_definition when you only "
        "need one item. "
        "Cost: 1 patience."
    )
)
def get_all_knowledge_definitions(runtime: ToolRuntime[ToolUserContext, CustomAgentState]) -> str:
    """Return all external knowledge with definitions (cost: 1 patience)."""
    note = deduct_and_note(runtime, cost=1)
    if not runtime.context.external_knowledge:
        return "(no external knowledge available)" + note
    return json.dumps(runtime.context.external_knowledge, indent=2, ensure_ascii=False) + note
