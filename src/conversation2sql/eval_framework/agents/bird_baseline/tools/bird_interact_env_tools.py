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
    get_table_names                  → 0.5 patience
    get_table_schema                 → 0.5 patience
    get_all_column_meanings          → 1 patience
    get_column_meaning               → 0.5 patience
    get_all_external_knowledge_names → 0.5 patience
    get_knowledge_definition         → 0.5 patience
    get_all_knowledge_definitions    → 1 patience
"""

import json
import os
import re
import subprocess

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
    linearize_kb,
    linearize_prerequisites,
)
from conversation2sql.eval_framework.state import (
    ColumnMeaningEntry,
    ExternalKnowledgeEntry,
    TaskData,
)

MAX_RESULT_LENGTH = 500

# Appended only when the formatted result actually exceeds MAX_RESULT_LENGTH.
# Without this, the cut is silent and the agent mistakes a successful query for
# a broken one (it sees a row sliced mid-value and re-runs / second-guesses,
# wasting bird-coins). The note states the query succeeded and how to see more.
TRUNCATION_NOTICE = (
    f"\n... [output truncated to {MAX_RESULT_LENGTH} characters; "
    "the query ran successfully — add a LIMIT or select fewer columns to see more]"
)

DB_TOOL_COSTS: dict[str, float] = {
    # Cost 1 to match the value advertised in the system prompt and the tool's
    # own docstring ("Cost: 1 bird-coin"); the agent budgets against that figure,
    # so charging 2 here desynced its budget planning and forced premature submits.
    "execute_sql": 1.0,
    "get_schema": 1.0,
    "get_table_names": 0.5,
    "get_table_schema": 0.5,
    "get_all_column_meanings": 1.0,
    "get_column_meaning": 0.5,
    "get_all_external_knowledge_names": 0.5,
    "get_knowledge_definition": 0.5,
    "get_all_knowledge_definitions": 1.0,
    # Single read-only psql terminal tool (ablation). Replaces the four DB
    # tools above when enable_psql_console is set; flat cost like execute_sql.
    "psql_console": 1.0,
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
# psql_console (single read-only terminal tool — ablation)
# ---------------------------------------------------------------------------
PSQL_TIMEOUT_S = 60

# Read-only protects the DATABASE, not the dev container. psql's `\!` runs host
# shell commands, and `\o`/`\copy`/`\i`/`\e`/`\w`/`\s` (and `\g`/`\gx` WITH an
# argument) read or write the host filesystem / pipe to a shell. These are
# refused before psql is ever spawned. Bare `\g`/`\gx` (no argument) just re-run
# the query buffer and are allowed; read-only inspection commands (\dt \d \l
# \df …) are allowed.
_PSQL_HOST_REACHING = frozenset(
    {"!", "o", "out", "copy", "i", "ir", "e", "ef", "ev", "w", "s"}
)
_PSQL_META_RE = re.compile(r"\\(!|[A-Za-z]+)")

PSQL_GUARDRAIL_REFUSAL = (
    "Refused: psql meta-commands that reach the host shell or filesystem "
    "(\\!, \\o, \\copy, \\i, \\e, \\w, \\s, and \\g/\\gx with a file or pipe) "
    "are not allowed. Use plain SQL or read-only inspection commands such as "
    "\\dt, \\d <table>, \\l, \\df."
)


def _violates_psql_guardrail(command: str) -> bool:
    """True if ``command`` uses a host-reaching psql meta-command.

    Conservative by design: a SQL string literal that happens to contain e.g.
    ``\\copy`` is also refused. That is an acceptable false-positive for a
    read-only inspection tool.
    """
    for match in _PSQL_META_RE.finditer(command):
        name = match.group(1)
        if name in _PSQL_HOST_REACHING:
            return True
        if name in ("g", "gx"):
            # Dangerous only with a trailing argument (\g <file> or \g |cmd) on
            # the same line; bare \g just re-runs the buffer.
            tail = command[match.end():].split("\n", 1)[0].strip()
            if tail:
                return True
    return False


def psql_console_impl(command: str, db_dsn: str) -> str:
    if _violates_psql_guardrail(command):
        return PSQL_GUARDRAIL_REFUSAL

    env = {
        **os.environ,
        "PGOPTIONS": "-c default_transaction_read_only=on -c statement_timeout=60s",
    }
    try:
        proc = subprocess.run(
            ["psql", db_dsn, "-X", "-c", command],
            env=env,
            capture_output=True,
            text=True,
            timeout=PSQL_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        return f"Error: psql timed out after {PSQL_TIMEOUT_S}s."

    output = proc.stdout if proc.returncode == 0 else (proc.stderr or proc.stdout)
    if len(output) > MAX_RESULT_LENGTH:
        output = output[:MAX_RESULT_LENGTH] + TRUNCATION_NOTICE
    return output


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
        if len(format_result) > MAX_RESULT_LENGTH:
            format_result = format_result[:MAX_RESULT_LENGTH] + TRUNCATION_NOTICE
        return ExecuteSQLResponse(
            result=format_result,
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


# Quoted-or-bare identifier patterns used to dissect the DDL blob. The dump uses
# double-quoted identifiers, but we accept bare ones too for robustness.
_CREATE_TABLE_RE = re.compile(r'CREATE TABLE\s+"?([A-Za-z_]\w*)"?', re.IGNORECASE)
_ALTER_LINE_RE = re.compile(r"^\s*ALTER TABLE\b", re.IGNORECASE | re.MULTILINE)
_ALTER_TARGET_RE = re.compile(r'ALTER TABLE\s+"?([A-Za-z_]\w*)"?', re.IGNORECASE)
_REFERENCES_RE = re.compile(r'REFERENCES\s+"?([A-Za-z_]\w*)"?', re.IGNORECASE)


def _parse_ddl(ddl_database_schema: str) -> tuple[dict[str, str], list[str]]:
    """Split a ``{db}_ddl.txt`` blob into per-table blocks and FK statements.

    The dump is a sequence of ``CREATE TABLE`` blocks (each followed by its
    ``First 3 rows`` sample) and then a trailing run of
    ``ALTER TABLE … FOREIGN KEY …`` statements. Returns:
      - ``tables``: ``{name: block}`` where each block spans from its
        ``CREATE TABLE`` line up to the next ``CREATE TABLE``, the first
        ``ALTER TABLE``, or EOF — so the sample-rows block is preserved.
      - ``alters``: the trailing ``ALTER TABLE`` statements, one stripped line
        each, so they can be re-attached per table.
    """
    first_alter = _ALTER_LINE_RE.search(ddl_database_schema)
    tables_region = (
        ddl_database_schema[: first_alter.start()]
        if first_alter
        else ddl_database_schema
    )

    alters: list[str] = []
    if first_alter:
        alters = [
            line.strip()
            for line in ddl_database_schema[first_alter.start():].splitlines()
            if line.strip().upper().startswith("ALTER TABLE")
        ]

    tables: dict[str, str] = {}
    matches = list(_CREATE_TABLE_RE.finditer(tables_region))
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(tables_region)
        tables[match.group(1)] = tables_region[start:end].rstrip()
    return tables, alters


def _alter_mentions_table(alter: str, table_name: str) -> bool:
    """True if ``table_name`` is the ALTER target (FK child) or the referenced
    table (FK parent) — so joinability surfaces from both sides of the FK."""
    target = _ALTER_TARGET_RE.search(alter)
    parent = _REFERENCES_RE.search(alter)
    return table_name in {m.group(1) for m in (target, parent) if m}


def get_table_names_impl(ddl_database_schema: str) -> dict:
    tables, _ = _parse_ddl(ddl_database_schema)
    return {"names": list(tables.keys())}


def get_table_schema_impl(table_name: str, ddl_database_schema: str) -> dict:
    tables, alters = _parse_ddl(ddl_database_schema)
    if table_name not in tables:
        return {"schema": "Table not found."}
    related = [a for a in alters if _alter_mentions_table(a, table_name)]
    return {"schema": "\n".join([tables[table_name], *related])}


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
        The query results as a markdown table on success, or the error message
        on failure.
    """
    response = execute_sql_impl(sql=sql, db_dsn=runtime.context.db_dsn)
    if response.success:
        return response.result
    # On failure ``error`` always holds the message; fall back to ``result``
    # (which carries the "Error: ..." prefix) only if it were ever unset.
    return response.error or response.result


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


@tool
def get_table_names(runtime: ToolRuntime[TaskData, CustomAgentState]) -> str:
    """List the names of all tables in the current task's database.
    Use this to discover which tables exist before fetching a specific one's
    schema with get_table_schema. Cost: 0.5 bird-coins.

    Returns:
        JSON list of table names.
    """
    return json.dumps(
        get_table_names_impl(ddl_database_schema=runtime.context.ddl_database_schema),
        indent=2,
    )


@tool
def get_table_schema(
        table_name: str, runtime: ToolRuntime[TaskData, CustomAgentState]
) -> str:
    """Get the schema of a single table: its CREATE TABLE statement, a few
    sample rows, and the foreign-key constraints linking it to other tables
    (both the ones it references and the ones referencing it, so you can see
    which tables are joinable). Cost: 0.5 bird-coins.

    Args:
        table_name: Name of the table to fetch.

    Returns:
        JSON string with the table schema, or "Table not found." if absent.
    """
    return json.dumps(
        get_table_schema_impl(
            table_name=table_name,
            ddl_database_schema=runtime.context.ddl_database_schema,
        ),
        indent=2,
    )


@tool
def psql_console(command: str, runtime: ToolRuntime[TaskData, CustomAgentState]) -> str:
    """Run ONE read-only PostgreSQL command in a `psql` terminal and return its text output.

    This single tool covers both querying data and inspecting the schema, so you
    do not need separate schema/query tools. Pass exactly one of:

    1. A SQL query — SELECT / WITH / EXPLAIN. The session is READ-ONLY, so
       INSERT / UPDATE / DELETE / CREATE / DROP and other writes are rejected by
       the server. Use this to test and verify a query before submit_sql.
    2. A single psql backslash meta-command for schema exploration:
         \\dt           list all tables
         \\d <table>    describe one table: columns, types, indexes, foreign keys
         \\l            list databases
         \\df           list functions
         \\dn           list schemas
       (broader \\d+ / \\d <pattern> forms also work).

    Typical flow: \\dt to see tables → \\d <table> to learn a table's columns and
    keys → a SELECT to inspect real values → submit_sql. Host shell / filesystem
    meta-commands (\\!, \\copy, \\o, \\i, \\e, \\w, \\s) are blocked. Cost: 1 bird-coin.

    Args:
        command: One SQL statement OR one psql backslash meta-command.

    Returns:
        The psql output on success, or the error message on failure.
    """
    return psql_console_impl(command=command, db_dsn=runtime.context.db_dsn)


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
    When the KB is linearized, this also returns the entry's transitive
    prerequisites (the knowledge it depends on) and their dependency edges.
    Cost: 0.5 bird-coins.

    Args:
        knowledge_name: The name of the knowledge entry to look up.

    Returns:
        JSON string with the knowledge definition.
    """
    if runtime.context.is_kb_linearized:
        section = linearize_prerequisites(
            knowledge_name, runtime.context.masked_agent_kb
        )
        return json.dumps({"knowledge": section}, indent=2)
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
