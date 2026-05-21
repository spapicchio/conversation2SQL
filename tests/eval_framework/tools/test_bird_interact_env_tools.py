"""Unit tests for the env-side Bird-Interact tools.

Each test exercises the pure ``*_impl`` function directly so the behaviour is
verified without needing a LangGraph runtime, the LangChain ``@tool``
decorator, or a real PostgreSQL server.
"""
from __future__ import annotations

from conversation2sql.eval_framework.agents.bird_baseline.tools import _execute_query

import json
from unittest.mock import patch

import psycopg2

from conversation2sql.eval_framework.agents.bird_baseline.tools import bird_interact_env_tools as env_tools
from conversation2sql.eval_framework.agents.bird_baseline.tools import (
    KNOWLEDGE_VISIBLE_FIELDS,
    ExecuteSQLResponse,
    execute_sql_impl,
    get_all_column_meanings_impl,
    get_all_external_knowledge_names_impl,
    get_all_knowledge_definitions_impl,
    get_column_meaning_impl,
    get_knowledge_definition_impl,
    get_schema_impl,
)


# ---------------------------------------------------------------------------
# execute_sql_impl
# ---------------------------------------------------------------------------

def test_long_run_timeout_real_db():
    sql = "SELECT pg_sleep(3600);"
    db_dsn = 'postgresql://root:123123@localhost:5433/solar_panel'
    output = execute_sql_impl(
        sql,
        db_dsn
    )
    assert output.success is False
    assert output.error == "SQL execution timed out"


def test_simple_query_real_db():
    sql = "SELECT sitekey, sitelabel FROM plants LIMIT 10;"
    db_dsn = 'postgresql://root:123123@localhost:5433/solar_panel'
    result, desc = _execute_query(
        sql,
        db_dsn
    )
    print(result)
    assert len(result) == 10
    assert desc[0][0] == "sitekey"
    assert desc[1][0] == "sitelabel"
    response = execute_sql_impl(sql, db_dsn)
    assert response.success is True
    assert response.error is None




class TestExecuteSqlImpl:
    """Behavioural contract for ``execute_sql_impl``.

    ``execute_sql_impl`` is the read-only SQL gateway that the agent uses to
    inspect the target database. The expectations exercised here are:

    * only read-only statements (SELECT / WITH / EXPLAIN) actually reach the
      database;
    * any database failure is captured in a structured response rather than
      bubbling up as an exception (timeouts get a friendly message);
    * results are bounded in size so a runaway query cannot blow up the
      agent's context window.
    """

    def test_select_query_returns_formatted_result(self):
        """Happy path: a plain SELECT is forwarded to the executor and the
        formatted rows come back inside a successful ``ExecuteSQLResponse``.
        Also pins the executor argument names so future refactors notice."""
        rows = [{"id": 1}, {"id": 2}]
        desc = [("id",)]
        with patch.object(
            env_tools, "_execute_query", return_value=(rows, desc)
        ) as exec_mock:
            response = execute_sql_impl("SELECT id FROM users;", db_dsn="dsn")

        assert isinstance(response, ExecuteSQLResponse)
        assert response.success is True
        assert response.error is None
        assert "id" in response.result
        # Pin the kwarg contract — ``_execute_query`` is the seam every test
        # in this class patches, so a rename here would silently break them.
        exec_mock.assert_called_once_with(query="SELECT id FROM users;", db_dsn="dsn")

    def test_with_query_is_allowed(self):
        """CTEs are read-only and must be accepted by the SELECT-only filter."""
        with patch.object(env_tools, "_execute_query", return_value=([], [])):
            response = execute_sql_impl(
                "WITH t AS (SELECT 1) SELECT * FROM t;", db_dsn="dsn"
            )
        assert response.success is True

    def test_explain_query_is_allowed(self):
        """EXPLAIN is read-only — useful for the agent to inspect plans, so it
        must pass the gate alongside SELECT/WITH."""
        with patch.object(env_tools, "_execute_query", return_value=([], [])):
            response = execute_sql_impl("EXPLAIN SELECT 1;", db_dsn="dsn")
        assert response.success is True

    def test_non_select_query_is_rejected_without_db_call(self):
        """Safety guarantee: a write statement must be rejected *before* the
        executor is touched, so a misbehaving agent cannot mutate the database
        even if the executor itself were permissive."""
        with patch.object(env_tools, "_execute_query") as exec_mock:
            response = execute_sql_impl("DELETE FROM users;", db_dsn="dsn")
        # The executor must never see the query when validation fails.
        exec_mock.assert_not_called()
        assert response.success is False
        assert response.error == "Only SELECT queries allowed in execute_sql"
        assert response.result == ""

    def test_inline_comments_are_stripped_before_validation(self):
        """Leading line/block comments would otherwise hide the keyword and
        defeat the SELECT-only check; the implementation must strip them
        first so a legitimate annotated query still runs."""
        with patch.object(
            env_tools, "_execute_query", return_value=([], [])
        ) as exec_mock:
            response = execute_sql_impl(
                "-- comment\n/* block */ SELECT 1;",
                db_dsn="dsn",
            )
        assert response.success is True
        exec_mock.assert_called_once()

    def test_query_canceled_error_is_translated(self):
        """Statement-timeout cancellations bubble out of psycopg2 as
        ``QueryCanceledError``. The tool should translate that into a stable,
        human-readable message — the agent uses the string to decide whether
        to retry with a different query."""
        with patch.object(
            env_tools,
            "_execute_query",
            side_effect=psycopg2.extensions.QueryCanceledError("boom"),
        ):
            response = execute_sql_impl("SELECT 1;", db_dsn="dsn")
        assert response.success is False
        assert response.error == "SQL execution timed out"

    def test_database_error_is_wrapped(self):
        """Generic database errors (syntax, missing column, etc.) must be
        captured as a failed response carrying the original message so the
        agent can self-correct rather than crash the run."""
        with patch.object(
            env_tools,
            "_execute_query",
            side_effect=psycopg2.DatabaseError("syntax bad"),
        ):
            response = execute_sql_impl("SELECT 1;", db_dsn="dsn")
        assert response.success is False
        assert response.error is not None
        assert "syntax bad" in response.error

    def test_long_result_is_truncated_to_max_length(self):
        """Result truncation is enforced so the agent's context cannot be
        flooded by huge result sets. Exact equality (not <=) catches a
        regression where truncation accidentally becomes a no-op."""
        # Pick a length strictly larger than the cap so we can detect a
        # missing truncation step (it would leave the surplus 100 chars).
        long_text = "x" * (env_tools.MAX_RESULT_LENGTH + 100)
        with (
            patch.object(env_tools, "_execute_query", return_value=("ignored", None)),
            patch.object(env_tools, "_format_result", return_value=long_text),
        ):
            response = execute_sql_impl("SELECT 1;", db_dsn="dsn")
        assert response.success is True
        assert len(response.result) == env_tools.MAX_RESULT_LENGTH



# ---------------------------------------------------------------------------
# get_schema_impl
# ---------------------------------------------------------------------------
def test_get_schema_impl_returns_schema_under_key():
    """``get_schema_impl`` is a thin wrapper that exposes the cached DDL
    under a stable ``"schema"`` key — locking the key prevents a silent rename
    from breaking every prompt template that consumes it."""
    result = get_schema_impl("CREATE TABLE t (id INT);")
    assert result == {"schema": "CREATE TABLE t (id INT);"}


# ---------------------------------------------------------------------------
# get_all_column_meanings_impl
# ---------------------------------------------------------------------------
def test_get_all_column_meanings_impl_serializes_each_entry(column_meanings):
    """Every entry in the column-meanings map must be serialised individually
    (one JSON blob per column) so the agent can pull a single column without
    parsing the whole catalogue. The keyset must round-trip unchanged."""
    result = get_all_column_meanings_impl(column_meanings)

    # Keys are preserved verbatim — they are ``db|table|column`` triples and
    # downstream lookups depend on exact strings.
    assert set(result["column_meanings"].keys()) == set(column_meanings.keys())
    for raw_key, raw in result["column_meanings"].items():
        decoded = json.loads(raw)
        assert decoded["column_meaning"] == column_meanings[raw_key].column_meaning


def test_get_all_column_meanings_impl_empty_dict():
    """Edge case: an empty catalogue must return the canonical empty shape
    (``{"column_meanings": {}}``) rather than raising or returning ``None``."""
    assert get_all_column_meanings_impl({}) == {"column_meanings": {}}


# ---------------------------------------------------------------------------
# get_column_meaning_impl
# ---------------------------------------------------------------------------
class TestGetColumnMeaningImpl:
    """``get_column_meaning_impl`` resolves a ``(db, table, column)`` triple
    against the catalogue. The behaviours pinned here are: case-insensitive
    matching (so the agent doesn't have to track exact casing), a stable
    not-found marker, and namespacing by database (different DBs may share
    table/column names without colliding)."""

    def test_existing_column_returns_serialized_entry(self, column_meanings):
        """Happy path: a known column resolves to its JSON-serialised entry."""
        result = get_column_meaning_impl(
            table_name="users",
            column_name="id",
            db_name="mydb",
            column_meanings=column_meanings,
        )
        decoded = json.loads(result["meaning"])
        assert decoded["column_meaning"] == "user primary key"

    def test_lookup_is_case_insensitive(self, column_meanings):
        """Case-insensitivity matters because LLM-generated SQL (and the
        catalogue) routinely disagree on identifier casing."""
        result = get_column_meaning_impl(
            table_name="USERS",
            column_name="ID",
            db_name="mydb",
            column_meanings=column_meanings,
        )
        decoded = json.loads(result["meaning"])
        assert decoded["column_meaning"] == "user primary key"

    def test_missing_column_returns_string_marker(self, column_meanings):
        """A miss returns a stable string marker instead of raising — the
        agent treats the marker as a signal to look elsewhere."""
        result = get_column_meaning_impl(
            table_name="users",
            column_name="ghost",
            db_name="mydb",
            column_meanings=column_meanings,
        )
        assert result == {"meaning": "Column meaning not found"}

    def test_missing_db_namespace_returns_marker(self, column_meanings):
        """Same column name in a different database must NOT match — the
        ``db_name`` qualifier is part of the catalogue key."""
        result = get_column_meaning_impl(
            table_name="users",
            column_name="id",
            db_name="other_db",
            column_meanings=column_meanings,
        )
        assert result == {"meaning": "Column meaning not found"}


# ---------------------------------------------------------------------------
# get_all_external_knowledge_names_impl
# ---------------------------------------------------------------------------
def test_get_all_external_knowledge_names_impl_lists_all_names(masked_agent_kb):
    """The agent uses this to discover what knowledge entries it can ask
    about — so every key in the KB must appear in the returned list."""
    result = get_all_external_knowledge_names_impl(masked_agent_kb)
    assert sorted(result["names"]) == sorted(masked_agent_kb.keys())


def test_get_all_external_knowledge_names_impl_empty():
    """Empty KB → empty list under the canonical key. Ensures the shape
    stays consistent so the agent can iterate without a None-check."""
    assert get_all_external_knowledge_names_impl({}) == {"names": []}


# ---------------------------------------------------------------------------
# get_knowledge_definition_impl
# ---------------------------------------------------------------------------
class TestGetKnowledgeDefinitionImpl:
    """``get_knowledge_definition_impl`` enforces a *masking* contract: only
    the fields in ``KNOWLEDGE_VISIBLE_FIELDS`` are exposed to the agent.
    Internal bookkeeping (``type``, ``children_knowledge``, etc.) must stay
    hidden — leaking them would give the agent oracle information it cannot
    have at inference time."""

    def test_existing_entry_returns_visible_fields_only(self, masked_agent_kb):
        """Hits return *only* the whitelisted fields — verifying both that
        the visible ones are present and that hidden ones are absent."""
        result = get_knowledge_definition_impl("active_user", masked_agent_kb)
        decoded = json.loads(result["knowledge"])
        assert set(decoded.keys()) == set(KNOWLEDGE_VISIBLE_FIELDS)
        assert decoded["knowledge"] == "active_user"
        # Negative assertions are deliberate — these fields exist on the
        # source object but must be stripped before returning.
        assert "type" not in decoded
        assert "children_knowledge" not in decoded

    def test_missing_entry_returns_marker(self, masked_agent_kb):
        """Unknown name → stable marker string, never an exception."""
        result = get_knowledge_definition_impl("does_not_exist", masked_agent_kb)
        assert result == {"knowledge": "Knowledge not found."}


# ---------------------------------------------------------------------------
# get_all_knowledge_definitions_impl
# ---------------------------------------------------------------------------
def test_get_all_knowledge_definitions_impl_returns_one_per_entry(masked_agent_kb):
    """Bulk variant: one serialised entry per KB row, all of them honouring
    the same field-masking contract as the single-entry version."""
    result = get_all_knowledge_definitions_impl(masked_agent_kb)
    assert len(result["knowledge"]) == len(masked_agent_kb)
    for raw in result["knowledge"]:
        decoded = json.loads(raw)
        # Re-checked here so a divergence between single/bulk paths is caught.
        assert set(decoded.keys()) == set(KNOWLEDGE_VISIBLE_FIELDS)


def test_get_all_knowledge_definitions_impl_empty():
    """Empty KB → empty list, matching the agreed shape."""
    assert get_all_knowledge_definitions_impl({}) == {"knowledge": []}


# ---------------------------------------------------------------------------
# Wrappers — sanity check the langchain @tool wires through to the impls.
#
# The ``*_impl`` tests above cover semantics; the wrapper tests below verify
# that the LangChain ``@tool`` glue (a) pulls the right fields out of the
# runtime context, and (b) JSON-serialises the result so a model-facing
# message can carry it. They're intentionally shallow — one assertion is
# enough to detect a wiring regression.
# ---------------------------------------------------------------------------
class _Runtime:
    """Stub LangGraph runtime that only exposes ``context``.

    The real runtime carries much more state, but the env tools only ever
    read ``runtime.context``, so the stub can be this small.
    """

    def __init__(self, context):
        self.context = context


def _invoke_tool(tool_obj, *, runtime, **kwargs):
    """Call a langchain ``@tool`` while injecting the runtime by hand.

    LangChain normally injects the runtime through its tool-calling machinery;
    bypassing that lets us drive the function directly with a stub.
    """
    return tool_obj.func(runtime=runtime, **kwargs)


def test_get_schema_wrapper_delegates_to_impl(task_data):
    """Wiring check: the wrapper must read ``ddl_database_schema`` from the
    task context and return the same dict the impl produces, JSON-encoded."""
    raw = _invoke_tool(env_tools.get_schema, runtime=_Runtime(task_data))
    assert json.loads(raw) == {"schema": task_data.ddl_database_schema}


def test_execute_sql_wrapper_returns_serialized_response(task_data):
    """Wiring check: the wrapper must serialise the ``ExecuteSQLResponse``
    to JSON so it can be embedded in a tool message back to the model."""
    with patch.object(env_tools, "_execute_query", return_value=([], [])):
        raw = _invoke_tool(
            env_tools.execute_sql, sql="SELECT 1;", runtime=_Runtime(task_data)
        )

    decoded = json.loads(raw)
    assert decoded["success"] is True
    assert decoded["error"] is None


def test_get_column_meaning_wrapper_uses_selected_database(task_data):
    """Wiring check: the wrapper must derive ``db_name`` from
    ``task.selected_database`` rather than expecting the caller to pass it,
    so the column lookup is namespaced to the active task's DB."""
    raw = _invoke_tool(
        env_tools.get_column_meaning,
        table_name="users",
        column_name="id",
        runtime=_Runtime(task_data),
    )
    assert "user primary key" in raw


def test_get_knowledge_definition_wrapper_returns_marker_for_missing(task_data):
    """Wiring check: a miss in the wrapper must propagate the same
    ``Knowledge not found.`` marker the impl returns — confirms the
    not-found path is not lost in serialisation."""
    raw = _invoke_tool(
        env_tools.get_knowledge_definition,
        knowledge_name="ghost",
        runtime=_Runtime(task_data),
    )
    assert json.loads(raw) == {"knowledge": "Knowledge not found."}


# ---------------------------------------------------------------------------
# Linearized branch — is_kb_linearized=True
# ---------------------------------------------------------------------------

def test_get_knowledge_definition_linearized_returns_single_line(task_data_linearized):
    """With is_kb_linearized=True, the tool returns one formatted line for the entry,
    not a JSON-dumped ExternalKnowledgeEntry."""
    raw = _invoke_tool(
        env_tools.get_knowledge_definition,
        knowledge_name="active_user",
        runtime=_Runtime(task_data_linearized),
    )
    decoded = json.loads(raw)
    assert "knowledge" in decoded
    assert isinstance(decoded["knowledge"], str)
    assert "[active_user]" in decoded["knowledge"]
    assert "revenue" not in decoded["knowledge"].lower()


def test_get_knowledge_definition_linearized_missing_returns_sentinel(task_data_linearized):
    """Missing name under is_kb_linearized=True -> same not-found sentinel."""
    raw = _invoke_tool(
        env_tools.get_knowledge_definition,
        knowledge_name="ghost",
        runtime=_Runtime(task_data_linearized),
    )
    assert json.loads(raw) == {"knowledge": "Knowledge not found."}


def test_get_all_knowledge_definitions_linearized_returns_flat_string(task_data_linearized):
    """With is_kb_linearized=True, the tool returns a single flat string,
    not a list of per-entry JSON strings."""
    raw = _invoke_tool(
        env_tools.get_all_knowledge_definitions,
        runtime=_Runtime(task_data_linearized),
    )
    decoded = json.loads(raw)
    assert "knowledge" in decoded
    assert isinstance(decoded["knowledge"], str)
    assert "# Definitions" in decoded["knowledge"]
    assert "active_user" in decoded["knowledge"]


def test_get_all_external_knowledge_names_same_regardless_of_linearized_flag(task_data, task_data_linearized):
    """Names are identical whether is_kb_linearized=True or False."""
    names_false = json.loads(_invoke_tool(env_tools.get_all_external_knowledge_names, runtime=_Runtime(task_data)))
    names_true  = json.loads(_invoke_tool(env_tools.get_all_external_knowledge_names, runtime=_Runtime(task_data_linearized)))
    assert sorted(names_false["names"]) == sorted(names_true["names"])

