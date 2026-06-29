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
    psql_console                     → 1 patience
    create_python_udf                → 1 patience
"""

from conversation2sql.logger import get_logger
import json
import os
import re
import subprocess


import psycopg2
import psycopg2.sql as pgsql
from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime
from pydantic import BaseModel

from conversation2sql.eval_framework.agents.bird_baseline.agent_code_state import (
    CustomAgentState,
)
from conversation2sql.eval_framework.agents.bird_baseline.tools.tool_specs import (
    ToolSpec,
)
from conversation2sql.eval_framework.agents.bird_baseline.tools.utils import (
    remove_comments,
)
from conversation2sql.eval_framework.agents.bird_baseline.tools.utils_db_execute import (
    MAX_CELL_CHARS,
    MAX_RESULT_ROWS,
    _execute_query,
    _format_result,
    _more_rows_note,
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

_pg_type_re = re.compile(r"^[A-Za-z0-9_ ()\[\],]+$")
_pg_name_re = re.compile(r"^[a-z_][a-z0-9_]*$")

# Single source of truth for each DB tool's cost + prompt summary. The cost is
# stamped onto the tool's schema description (via stamp_cost_in_descriptions) and
# the summary is rendered in the agent prompt's tool list, so neither can drift.
# DB_TOOL_COSTS is derived below for the legacy callers (middleware, metrics).
DB_TOOL_SPECS: dict[str, ToolSpec] = {
    # Cost 1 to match the value the agent budgets against; charging 2 here
    # desynced its budget planning and forced premature submits.
    "execute_sql": ToolSpec("execute_sql", 1.0, "execute a PostgreSQL query"),
    "get_schema": ToolSpec("get_schema", 1.0, "get the full database schema"),
    "get_table_names": ToolSpec(
        "get_table_names", 0.5, "list all table names in the database"
    ),
    "get_table_schema": ToolSpec(
        "get_table_schema",
        0.5,
        "get one table's schema (CREATE TABLE, sample rows, and foreign keys to joinable tables)",
    ),
    "get_all_column_meanings": ToolSpec(
        "get_all_column_meanings", 1.0, "get all column meanings"
    ),
    "get_column_meaning": ToolSpec(
        "get_column_meaning", 0.5, "get the meaning of one column"
    ),
    "get_all_external_knowledge_names": ToolSpec(
        "get_all_external_knowledge_names", 0.5, "get all external knowledge names"
    ),
    "get_knowledge_definition": ToolSpec(
        "get_knowledge_definition",
        0.5,
        "get one external knowledge definition along with the knowledge it depends on (its prerequisites)",
    ),
    "get_all_knowledge_definitions": ToolSpec(
        "get_all_knowledge_definitions", 1.0, "get all external knowledge definitions"
    ),
    # Single read-only psql terminal tool (ablation). Replaces the four DB
    # tools above when enable_psql_console is set; flat cost like execute_sql.
    "psql_console": ToolSpec(
        "psql_console",
        1.0,
        "run ONE PostgreSQL statement or one read-only psql meta-command. Read-only session",
    ),
    "create_python_udf": ToolSpec(
        "create_python_udf",
        1.0,
        "create a PostgreSQL Python UDF (plpython3u) for complex formulas; returns the callable function name",
    ),
}

DB_TOOL_COSTS: dict[str, float] = {
    name: spec.cost for name, spec in DB_TOOL_SPECS.items()
}

KNOWLEDGE_VISIBLE_FIELDS = ["id", "knowledge", "description", "definition"]


def _safe_instance_prefix(instance_id: str, max_len: int = 30) -> str:
    """Return a SQL-safe prefix derived from instance_id for UDF namespacing.

    Lowercases and replaces any character outside [a-z0-9] with '_', then
    truncates to max_len. max_len=30 leaves ≥32 chars for '_<function_name>'
    within PostgreSQL's 63-char identifier limit.
    """
    safe = re.sub(r"[^a-z0-9]", "_", instance_id.lower())
    return safe[:max_len]


logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
class ExecuteSQLResponse(BaseModel):
    result: str
    success: bool
    error: str | None = None


class UDFParameter(BaseModel):
    """One parameter for a Python UDF: a name and its PostgreSQL type."""

    name: str
    pg_type: str


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
            tail = command[match.end() :].split("\n", 1)[0].strip()
            if tail:
                return True
    return False


def _is_psql_meta_command(command: str) -> bool:
    """True if ``command`` is a psql backslash meta-command (``\\dt``, ``\\d``,
    ``\\l``, ``\\df`` …) rather than a SQL query.

    Schema-inspection meta-commands list table/column/function *names*, so row
    truncation would silently drop names off the end of the listing (the agent
    then can't see tables it needs). Only SQL queries — which can return
    arbitrarily many data rows — are truncated. Detection mirrors how psql
    dispatches: the first non-blank character being a backslash makes it a
    meta-command (a ``SELECT … \\g`` still starts with SQL and stays truncatable).
    """
    return command.lstrip().startswith("\\")


# psql's aligned output ends with a "(N rows)" footer; we read N for the note.
_PSQL_ROWCOUNT_RE = re.compile(r"^\((\d+) rows?\)", re.MULTILINE)

# Total-output char cap for psql SQL results. psql output is a flat aligned-text
# blob with no per-cell addressability, so it gets a *total* cap rather than the
# per-cell cap execute_sql uses; it mirrors the same failure mode (few rows, huge
# cells — e.g. JSON aggregates — that the row cap never catches). Sized as
# MAX_CELL_CHARS * MAX_RESULT_ROWS so a worst-case row-capped psql result stays in
# the same ballpark as an execute_sql result whose one big cell is per-cell capped.
# Only SQL output is capped (callers skip _truncate_psql_output for meta-commands),
# so schema listings keep every name.
MAX_PSQL_OUTPUT_CHARS = MAX_CELL_CHARS * MAX_RESULT_ROWS


def _cap_psql_output(output: str, max_chars: int = MAX_PSQL_OUTPUT_CHARS) -> str:
    """Clip psql SQL output to ``max_chars`` total (marker included). Returns
    ``output`` unchanged when it fits."""
    if len(output) <= max_chars:
        return output
    marker = f"\n…[output truncated to {max_chars} chars; was {len(output)}]"
    return output[: max_chars - len(marker)] + marker


def _truncate_psql_output(output: str, max_rows: int = MAX_RESULT_ROWS) -> str:
    """Row- and char-truncate psql's aligned SQL output, mirroring
    ``_format_result`` + the per-cell cap.

    Keeps the 2 header lines (column header + ``---+---`` separator) plus the
    first ``max_rows`` data rows, then appends the shared "more rows" note with
    the true total parsed from psql's ``(N rows)`` footer. The result is then
    clipped to ``MAX_PSQL_OUTPUT_CHARS`` so a few rows carrying giant cells can't
    flood the agent even when the row cap doesn't fire. If the footer is absent
    (an error, EXPLAIN-less output, …) or the result already fits the row cap,
    only the char cap is applied.
    """
    match = _PSQL_ROWCOUNT_RE.search(output)
    if match is None:
        return _cap_psql_output(output)
    total = int(match.group(1))
    if total <= max_rows:
        return _cap_psql_output(output)
    lines = output.split("\n")
    kept = lines[: 2 + max_rows]
    return _cap_psql_output("\n".join(kept) + _more_rows_note(str(total)))


# ---------------------------------------------------------------------------
# Strict inspection mode (enable_psql_strict_inspection ablation)
# ---------------------------------------------------------------------------
# In strict mode psql_console is an allowlist, not a denylist: only SQL queries,
# SQL-syntax help (\h), the help index (\?), and the read-only Informational
# meta-commands (the \d-family plus \l, \sf, \sv, \z) are allowed. Every other
# meta-command is refused. The rationale: each call is a fresh `psql -X -c <one
# command>` with no session persistence, so formatting/buffer/variable/
# conditional commands are inert and connection/OS commands are out of scope —
# none of them help write SQL.
#
# The non-\d informational/help commands. \s (history) is host-reaching and
# deliberately excluded, so only the two-letter \sf/\sv pass; the whole \d-family
# is matched by prefix below.
_PSQL_INFO_EXTRA = frozenset({"l", "sf", "sv", "z", "h"})

PSQL_STRICT_REFUSAL = (
    "Refused: this console is restricted to read-only inspection. Allowed: a "
    "SELECT/WITH/EXPLAIN query, \\h for SQL syntax, or an informational "
    "meta-command (the \\d-family, \\l, \\sf, \\sv, \\z). Run \\? to list them."
)

PSQL_STRICT_HELP_BANNER = (
    "This is a READ-ONLY inspection console. You may run:\n"
    "  - a SQL query: SELECT / WITH / EXPLAIN\n"
    "  - \\h [NAME] for SQL syntax help\n"
    "  - one of the informational meta-commands listed below.\n"
    "All other psql meta-commands are disabled.\n\n"
)


def _is_informational_meta(name: str) -> bool:
    """True if ``name`` (the letters right after the leading backslash) is a
    read-only Informational meta-command. The entire ``\\d`` family qualifies;
    plus ``\\l``, ``\\sf``, ``\\sv``, ``\\z``, and the ``\\h`` SQL-syntax help."""
    return name.startswith("d") or name in _PSQL_INFO_EXTRA


def _strict_psql_refusal(command: str) -> str | None:
    """Return a refusal message if ``command`` is not allowed under strict
    inspection mode, else ``None`` (allowed — caller runs it).

    A command that does not start with a backslash is a SQL query: it keeps the
    host-reaching guardrail so a trailing ``\\!``/``\\g | sh`` injection is still
    caught. A backslash command is allowed only when it is ``\\?`` (handled as
    filtered help by the caller) or an informational meta-command.
    """
    stripped = command.lstrip()
    if not stripped.startswith("\\"):
        return PSQL_GUARDRAIL_REFUSAL if _violates_psql_guardrail(command) else None
    if stripped.startswith("\\?"):
        return None
    match = re.match(r"\\([A-Za-z]+)", stripped)
    if match and _is_informational_meta(match.group(1)):
        return None
    return PSQL_STRICT_REFUSAL


def _extract_informational_help(full_help: str) -> str:
    """Slice the ``Informational`` section out of psql's ``\\?`` output.

    psql's help is a sequence of column-0 section headers (``General``,
    ``Informational``, ``Formatting`` …) each followed by indented command
    lines. We keep the ``Informational`` header and its body, dropping every
    other section so strict mode doesn't advertise blocked commands.
    """
    out: list[str] = []
    capturing = False
    for line in full_help.split("\n"):
        is_header = bool(line) and not line[0].isspace() and not line.startswith("\\")
        if is_header:
            capturing = line.strip() == "Informational"
            if capturing:
                out.append(line)
            continue
        if capturing:
            out.append(line)
    return "\n".join(out).rstrip()


def _run_psql(command: str, db_dsn: str) -> str:
    """Run one ``psql -X -c <command>`` under a read-only, timeout-bounded env
    and return its text output (stdout on success, else stderr/stdout)."""
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
    return proc.stdout if proc.returncode == 0 else (proc.stderr or proc.stdout)


def _filtered_psql_help(db_dsn: str) -> str:
    """Run ``\\?`` and return only its Informational section, prefixed with a
    banner reminding the agent that SQL queries and ``\\h`` are also available."""
    return PSQL_STRICT_HELP_BANNER + _extract_informational_help(
        _run_psql("\\?", db_dsn)
    )


def psql_console_impl(command: str, db_dsn: str, strict: bool = False) -> str:
    if strict:
        refusal = _strict_psql_refusal(command)
        if refusal is not None:
            return refusal
        if command.lstrip().startswith("\\?"):
            return _filtered_psql_help(db_dsn)
    elif _violates_psql_guardrail(command):
        return PSQL_GUARDRAIL_REFUSAL

    output = _run_psql(command, db_dsn)
    if not _is_psql_meta_command(command):
        output = _truncate_psql_output(output)
    return output


def create_python_udf_impl(
    function_name: str,
    parameters: list[UDFParameter],
    return_type: str,
    python_body: str,
    db_dsn: str,
    instance_id: str,
) -> str:
    """Create a plpython3u UDF in PostgreSQL and return its qualified name.

    The function is registered under ``<safe_prefix>_<sanitised_name>`` where
    ``safe_prefix`` is derived from ``instance_id`` via ``_safe_instance_prefix``,
    ensuring parallel conversations on the same database do not collide.

    Returns the qualified name on success, or the error message on failure.
    """
    safe_name = re.sub(r"[^a-z0-9]", "_", function_name.lower())
    prefix = _safe_instance_prefix(instance_id)
    qualified = f"{prefix}_{safe_name}"[:63]

    if not _pg_type_re.match(return_type):
        return f"Error: invalid return_type {return_type!r}"
    for p in parameters:
        if not _pg_name_re.match(p.name):
            return f"Error: invalid parameter name {p.name!r}"
        if not _pg_type_re.match(p.pg_type):
            return f"Error: invalid parameter type {p.pg_type!r}"

    param_str = ", ".join(f"{p.name} {p.pg_type}" for p in parameters)
    ddl = (
        f"CREATE OR REPLACE FUNCTION {qualified}({param_str}) "
        f"RETURNS {return_type} AS $plpy$ {python_body} $plpy$ LANGUAGE plpython3u"
    )

    conn = psycopg2.connect(db_dsn)
    try:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS plpython3u")
            cur.execute(ddl)
        conn.commit()
        return qualified
    except psycopg2.DatabaseError as exc:
        conn.rollback()
        return f"Error: {exc}"
    finally:
        conn.close()


def cleanup_python_udfs_impl(db_dsn: str, prefix: str) -> None:
    """Drop all plpython3u UDFs whose name starts with ``prefix_``.

    Called unconditionally after each agent run when enable_python_udf=True.
    Uses ``pg_get_function_identity_arguments`` to build the exact DROP
    signature so overloaded functions are handled correctly.
    No-op when no matching functions exist.
    """
    conn = psycopg2.connect(db_dsn)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT proname, pg_get_function_identity_arguments(oid) "
                "FROM pg_proc WHERE proname LIKE %s",
                (f"{prefix}_%",),
            )
            for proname, args in cur.fetchall():
                cur.execute(f"DROP FUNCTION IF EXISTS {proname}({args})")
        conn.commit()
    finally:
        conn.close()


def apply_column_comments_impl(
    db_dsn: str,
    column_meanings: dict[str, "ColumnMeaningEntry"],
) -> None:
    """Write COMMENT ON COLUMN for every entry in column_meanings to the DB.

    Called once per database at eval startup when enable_psql_console=True so
    that \\d+ shows column descriptions, closing the information gap with
    get_table_schema (which embeds meanings as DDL inline comments).

    The key format is ``{db_name}|{table}|{column}`` (from the column-meaning
    JSON); entries that do not match this form are skipped silently.
    Idempotent: PostgreSQL overwrites an existing comment with the same value.
    """
    conn = psycopg2.connect(db_dsn)
    try:
        with conn.cursor() as cur:
            for key, entry in column_meanings.items():
                parts = key.split("|")
                if len(parts) != 3:
                    continue
                _, table, column = parts
                try:
                    cur.execute("SAVEPOINT sp")
                    cur.execute(
                        pgsql.SQL("COMMENT ON COLUMN {}.{} IS %s").format(
                            pgsql.Identifier(table),
                            pgsql.Identifier(column),
                        ),
                        (entry.column_meaning,),
                    )
                    cur.execute("RELEASE SAVEPOINT sp")
                except psycopg2.Error as exc:
                    cur.execute("ROLLBACK TO SAVEPOINT sp")
                    cur.execute("RELEASE SAVEPOINT sp")
                    logger.warning(
                        f"Failed to apply comment for {table}.{column}: {exc}"
                    )
        conn.commit()
    finally:
        conn.close()


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
        # _format_result handles row-based truncation + the "more rows" note.
        format_result = _format_result(result, desc)
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
            for line in ddl_database_schema[first_alter.start() :].splitlines()
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
    schema with get_table_schema.
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
    which tables are joinable).

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
    2. You can inspect the schema with psql's backslash meta-commands. Run \\? to
       list the available inspection commands, then use them.
    You can also run \\d+ <table> to get the description of the columns with their comments (meanings).
    Args:
        command: One SQL statement OR one psql backslash meta-command.

    Returns:
        The psql output on success, or the error message on failure.
    """
    #     2. A single psql backslash meta-command for schema exploration:
    #      \\dt           list all tables
    #      \\d <table>    describe one table: columns, types, indexes, foreign keys
    #      \\l            list databases
    #      \\df           list functions
    #      \\dn           list schemas
    #    (broader \\d+ / \\d <pattern> forms also work).

    # Typical flow: \\dt to see tables → \\d <table> to learn a table's columns and
    # keys → a SELECT to inspect real values → submit_sql. Host shell / filesystem
    # meta-commands (\\!, \\copy, \\o, \\i, \\e, \\w, \\s) are blocked. Under the
    # enable_psql_strict_inspection ablation only SQL + \\h + the informational
    # \\d-family are allowed and \\? lists just those.

    return psql_console_impl(
        command=command,
        db_dsn=runtime.context.db_dsn,
        strict=runtime.context.enable_psql_strict_inspection,
    )


@tool
def create_python_udf(
    function_name: str,
    parameters: list[UDFParameter],
    return_type: str,
    python_body: str,
    runtime: ToolRuntime[TaskData, CustomAgentState],
) -> str:
    """Create a PostgreSQL Python (plpython3u) user-defined function and return
    its callable name. Use it to implement formulas that are hard to express in
    pure SQL, then call the returned name inside execute_sql or psql_console.

    Args:
        function_name: Desired function name (auto-prefixed for isolation).
        parameters: List of {name, pg_type} parameter definitions.
        return_type: PostgreSQL return type of the function.
        python_body: Python code body ending with `return <value>`.
            The body runs inside PostgreSQL via plpython3u.
            Import standard library modules with normal `import` statements.

    PostgreSQL type → Python type mapping for parameters and return type:

        PostgreSQL type              Python type inside body
        ----------------------------------------------------
        INTEGER / SMALLINT           int
        BIGINT                       int
        REAL / FLOAT4                float
        DOUBLE PRECISION / FLOAT8    float
        NUMERIC / DECIMAL            decimal.Decimal  (import decimal)
        TEXT / VARCHAR / CHAR        str
        BOOLEAN                      bool
        DATE                         datetime.date    (import datetime)
        TIMESTAMP                    datetime.datetime
        TIMESTAMPTZ                  datetime.datetime (timezone-aware)
        BYTEA                        bytes
        any[] (array type)           list
        composite type               dict  (keys = column names)
        NULL (any nullable type)     None

    Returns:
        The qualified function name to use in SELECT queries, e.g.
        ``instance_42_my_formula``. On error returns the error message.
    """
    return create_python_udf_impl(
        function_name=function_name,
        parameters=parameters,
        return_type=return_type,
        python_body=python_body,
        db_dsn=runtime.context.db_dsn,
        instance_id=runtime.context.instance_id,
    )


# ---------------------------------------------------------------------------
# Column meaning tools
# ---------------------------------------------------------------------------
@tool
def get_all_column_meanings(runtime: ToolRuntime[TaskData, CustomAgentState]) -> str:
    """Get the meanings/descriptions of all columns in the database.
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
    Returns:
        JSON list of knowledge entry names.
    """
    return json.dumps(
        get_all_external_knowledge_names_impl(
            masked_agent_kb=runtime.context.masked_agent_kb
        ),
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
    """Return all external knowledge with definitions.
    Returns:
        JSON string with all knowledge entries and their definitions.
    """
    if runtime.context.is_kb_linearized:
        flat = linearize_kb(runtime.context.masked_agent_kb)
        return json.dumps({"knowledge": flat}, indent=2)
    return json.dumps(
        get_all_knowledge_definitions_impl(
            masked_agent_kb=runtime.context.masked_agent_kb,
        ),
        indent=2,
    )
