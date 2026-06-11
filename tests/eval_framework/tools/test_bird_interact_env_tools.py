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
from conversation2sql.eval_framework.agents.bird_baseline.tools import utils_db_execute
from conversation2sql.eval_framework.agents.bird_baseline.tools import (
    KNOWLEDGE_VISIBLE_FIELDS,
    ExecuteSQLResponse,
    execute_sql_impl,
    psql_console_impl,
    get_all_column_meanings_impl,
    get_all_external_knowledge_names_impl,
    get_all_knowledge_definitions_impl,
    get_column_meaning_impl,
    get_knowledge_definition_impl,
    get_schema_impl,
    get_table_names_impl,
    get_table_schema_impl,
)
from conversation2sql.eval_framework.agents.bird_baseline.tools.bird_interact_env_tools import (
    PSQL_GUARDRAIL_REFUSAL,
    apply_column_comments_impl,
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

    def test_result_is_returned_verbatim_from_format_result(self):
        """Row-aware truncation now lives entirely inside ``_format_result``;
        ``execute_sql_impl`` must pass that string through untouched (no extra
        character-level cut layered on top)."""
        formatted = "| id |\n| --- |\n| 1 |"
        with (
            patch.object(env_tools, "_execute_query", return_value=("ignored", None)),
            patch.object(env_tools, "_format_result", return_value=formatted),
        ):
            response = execute_sql_impl("SELECT 1;", db_dsn="dsn")
        assert response.success is True
        assert response.result == formatted


# ---------------------------------------------------------------------------
# _format_result / _format_cell
# ---------------------------------------------------------------------------
class TestFormatResult:
    """``_format_result`` renders RealDictRow rows as the GitHub-flavored
    markdown table the agent reads. These tests pin the structural and clarity
    properties we rely on: a header + ``| --- |`` separator + pipe-wrapped data
    rows, and compact full-precision JSON for container cells."""

    def test_markdown_table_header_separator_and_rows(self):
        """Output is a GFM table: header row, ``| --- |`` separator with one
        ``---`` per column, then pipe-wrapped data rows."""
        result = [{"sitekey": "SP9227", "sitelabel": "Solar Plant West"}]
        desc = (("sitekey",), ("sitelabel",))
        out = utils_db_execute._format_result(result, desc)
        lines = out.split("\n")
        assert lines[0] == "| sitekey | sitelabel |"
        assert lines[1] == "| --- | --- |"
        assert lines[2] == "| SP9227 | Solar Plant West |"

    def test_container_cell_is_compact_json_not_python_repr(self):
        """JSON/array columns come back as dict/list; they must render as
        compact double-quoted JSON (deterministic key order, full precision),
        never Python ``repr`` with single quotes or rounded floats."""
        result = [{"stations": [{"station": "Observatory", "aoi": 0.0146324}]}]
        desc = (("stations",),)
        out = utils_db_execute._format_result(result, desc)
        # Line 0 = header, line 1 = separator, line 2 = first data row.
        cell = out.split("\n")[2].strip("| ")
        # Compact separators, sorted keys, no precision loss, valid JSON.
        assert cell == '[{"aoi":0.0146324,"station":"Observatory"}]'
        assert "'" not in cell
        assert json.loads(cell) == [{"station": "Observatory", "aoi": 0.0146324}]

    def test_cells_are_not_truncated(self):
        """Width is intentionally unbounded so ``execute_sql`` and
        ``psql_console`` output stay comparable: a long text/JSON cell renders
        in full rather than being clipped to a per-cell cap."""
        result = [{"blob": {"k": "y" * 500}}]
        desc = (("blob",),)
        out = utils_db_execute._format_result(result, desc)
        cell = out.split("\n")[2].removeprefix("| ").removesuffix(" |")
        assert cell == '{"k":"' + "y" * 500 + '"}'

    def test_result_over_max_rows_is_row_truncated_with_note(self):
        """Beyond ``MAX_RESULT_ROWS`` the table keeps exactly that many data
        rows and appends a note stating the true total, so the agent sees whole
        rows (not a mid-row character cut) and knows more rows existed."""
        result = [{"id": i} for i in range(5)]
        desc = (("id",),)
        out = utils_db_execute._format_result(result, desc)
        lines = out.split("\n")
        # header + separator + MAX_RESULT_ROWS data rows, then the note.
        assert lines[:2] == ["| id |", "| --- |"]
        data_rows = [ln for ln in lines if ln.startswith("| ") and "---" not in ln and "id" not in ln]
        assert len(data_rows) == utils_db_execute.MAX_RESULT_ROWS
        assert f"showing first {utils_db_execute.MAX_RESULT_ROWS} of 5 rows" in out

    def test_result_at_max_rows_has_no_note(self):
        """A result at or below the row cap renders verbatim with no note."""
        result = [{"id": i} for i in range(utils_db_execute.MAX_RESULT_ROWS)]
        desc = (("id",),)
        out = utils_db_execute._format_result(result, desc)
        assert "showing first" not in out

    def test_fetch_limit_total_is_reported_with_plus(self):
        """When the row count hits the fetch cap the true total is unknown, so
        the note reports it as ``<limit>+`` rather than an exact (capped) count."""
        result = [{"id": i} for i in range(utils_db_execute.RESULT_FETCH_LIMIT)]
        desc = (("id",),)
        out = utils_db_execute._format_result(result, desc)
        assert f"of {utils_db_execute.RESULT_FETCH_LIMIT}+ rows" in out

    def test_none_and_empty_results_have_dedicated_messages(self):
        assert utils_db_execute._format_result(None, ()) == "Query executed successfully."
        assert utils_db_execute._format_result([], ()) == "Query executed, empty result set."


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


def test_execute_sql_wrapper_returns_table_on_success(task_data):
    """On success the wrapper returns the bare formatted result (the markdown
    table), not the serialised ``ExecuteSQLResponse`` envelope."""
    rows = [{"id": 1}]
    desc = [("id",)]
    with patch.object(env_tools, "_execute_query", return_value=(rows, desc)):
        raw = _invoke_tool(
            env_tools.execute_sql, sql="SELECT id FROM t;", runtime=_Runtime(task_data)
        )

    assert raw == "| id |\n| --- |\n| 1 |"


def test_execute_sql_wrapper_returns_error_message_on_failure(task_data):
    """On failure the wrapper returns only the error string, no JSON envelope."""
    raw = _invoke_tool(
        env_tools.execute_sql, sql="DELETE FROM t;", runtime=_Runtime(task_data)
    )

    assert raw == "Only SELECT queries allowed in execute_sql"


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

def test_get_knowledge_definition_linearized_returns_prerequisite_section(task_data_linearized):
    """With is_kb_linearized=True, the tool returns a linearized section (definitions
    block) for the entry, not a JSON-dumped ExternalKnowledgeEntry. The conftest entry
    has no prerequisites, so only its own definition appears."""
    raw = _invoke_tool(
        env_tools.get_knowledge_definition,
        knowledge_name="active_user",
        runtime=_Runtime(task_data_linearized),
    )
    decoded = json.loads(raw)
    assert "knowledge" in decoded
    assert isinstance(decoded["knowledge"], str)
    assert "# Definitions" in decoded["knowledge"]
    assert "[active_user]" in decoded["knowledge"]
    # active_user has no prerequisites, so the unrelated entry must not appear.
    assert "revenue" not in decoded["knowledge"].lower()


def test_get_knowledge_definition_linearized_includes_transitive_prerequisites(task_data):
    """With is_kb_linearized=True, looking up an entry also surfaces the knowledge it
    transitively depends on, with the dependency edges between them."""
    from conversation2sql.eval_framework.state import ExternalKnowledgeEntry

    chained_kb = {
        "base (BASE)": ExternalKnowledgeEntry(
            id=10, knowledge="base (BASE)", description="base value",
            definition="x", type="domain_knowledge", children_knowledge=[-1],
        ),
        "derived (DRV)": ExternalKnowledgeEntry(
            id=11, knowledge="derived (DRV)", description="uses base",
            definition="BASE * 2", type="domain_knowledge", children_knowledge=[10],
        ),
    }
    ctx = task_data.model_copy(
        update={"is_kb_linearized": True, "masked_agent_kb": chained_kb}
    )
    raw = _invoke_tool(
        env_tools.get_knowledge_definition,
        knowledge_name="derived (DRV)",
        runtime=_Runtime(ctx),
    )
    knowledge = json.loads(raw)["knowledge"]
    assert "(BASE, prerequisite_of, DRV)" in knowledge
    assert "[BASE]" in knowledge
    assert "[DRV]" in knowledge


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


# ---------------------------------------------------------------------------
# Granular table-schema tools (get_table_names / get_table_schema)
# ---------------------------------------------------------------------------

# A miniature but faithful DDL blob: two CREATE TABLE blocks each with a
# "First 3 rows" sample, followed by a trailing ALTER TABLE foreign-key block —
# exactly the shape of the real {db}_ddl.txt files. "orders" is the FK *child*
# and "customers" is the referenced *parent* of the single FK.
SAMPLE_DDL = '''-- PostgreSQL schema dump for schema: public

CREATE TABLE "orders" (
    "order_id" text NOT NULL PRIMARY KEY,
    "customer_id" text
);
First 3 rows:
order_id | customer_id
----------------------
O1 | C1
O2 | C2
...

CREATE TABLE "customers" (
    "customer_id" text NOT NULL PRIMARY KEY,
    "name" text
);
First 3 rows:
customer_id | name
------------------
C1 | Alice
...

ALTER TABLE "orders" ADD CONSTRAINT "fk_orders_customer" FOREIGN KEY ("customer_id") REFERENCES "customers" ("customer_id") ON DELETE NO ACTION;
'''


def test_parse_ddl_splits_tables_and_captures_alters():
    """The parser must split the blob on CREATE TABLE boundaries (preserving
    each table's sample-rows block) and collect the trailing ALTER statements
    separately so they can be re-attached per table."""
    tables, alters = env_tools._parse_ddl(SAMPLE_DDL)

    assert list(tables.keys()) == ["orders", "customers"]
    # each block keeps its own CREATE TABLE + sample rows, and stops before
    # the next table / the ALTER block
    assert 'CREATE TABLE "orders"' in tables["orders"]
    assert "First 3 rows" in tables["orders"]
    assert "O1 | C1" in tables["orders"]
    assert 'CREATE TABLE "customers"' not in tables["orders"]
    assert "ALTER TABLE" not in tables["orders"]
    # the FK lives in the alters list, not inside any table block
    assert len(alters) == 1
    assert alters[0].startswith("ALTER TABLE")


def test_get_table_names_impl_lists_all_in_order():
    """``get_table_names_impl`` mirrors ``get_all_external_knowledge_names`` —
    a bare list of names, in DDL order, under a ``"names"`` key."""
    assert get_table_names_impl(SAMPLE_DDL) == {"names": ["orders", "customers"]}


def test_get_table_schema_impl_includes_create_block_and_sample_rows():
    """A single-table fetch returns the CREATE TABLE block *and* its sample
    rows — the user wants the rows available without spending execute_sql."""
    result = get_table_schema_impl("orders", SAMPLE_DDL)
    schema = result["schema"]
    assert 'CREATE TABLE "orders"' in schema
    assert "First 3 rows" in schema
    assert "O1 | C1" in schema
    # an unrelated table's definition must not leak in
    assert 'CREATE TABLE "customers"' not in schema


def test_get_table_schema_impl_includes_fk_when_table_is_child():
    """When the looked-up table is the FK child (the ALTER TABLE target), the
    FK statement must be attached so the agent sees the parent it can join to."""
    schema = get_table_schema_impl("orders", SAMPLE_DDL)["schema"]
    assert 'ALTER TABLE "orders" ADD CONSTRAINT' in schema
    assert 'REFERENCES "customers"' in schema


def test_get_table_schema_impl_includes_fk_when_table_is_parent():
    """When the looked-up table is the referenced parent, the same FK must be
    attached so joinability is visible from *both* sides of the relationship."""
    schema = get_table_schema_impl("customers", SAMPLE_DDL)["schema"]
    assert 'CREATE TABLE "customers"' in schema
    assert 'ALTER TABLE "orders" ADD CONSTRAINT' in schema


def test_get_table_schema_impl_unknown_table_returns_marker():
    """Unknown names return the ``"Table not found."`` sentinel, mirroring the
    KB tool's ``"Knowledge not found."`` contract."""
    assert get_table_schema_impl("does_not_exist", SAMPLE_DDL) == {
        "schema": "Table not found."
    }


def test_get_table_names_wrapper_delegates_to_impl(task_data):
    """Wiring check: the wrapper reads ``ddl_database_schema`` from context."""
    ctx = task_data.model_copy(update={"ddl_database_schema": SAMPLE_DDL})
    raw = _invoke_tool(env_tools.get_table_names, runtime=_Runtime(ctx))
    assert json.loads(raw) == {"names": ["orders", "customers"]}


def test_get_table_schema_wrapper_delegates_to_impl(task_data):
    """Wiring check: the wrapper passes the requested table name and context
    DDL through to the impl and JSON-encodes the result."""
    ctx = task_data.model_copy(update={"ddl_database_schema": SAMPLE_DDL})
    raw = _invoke_tool(
        env_tools.get_table_schema, table_name="orders", runtime=_Runtime(ctx)
    )
    assert json.loads(raw) == get_table_schema_impl("orders", SAMPLE_DDL)


# ---------------------------------------------------------------------------
# psql_console_impl
# ---------------------------------------------------------------------------
import subprocess
from types import SimpleNamespace


class TestPsqlGuardrail:
    """Host-reaching backslash meta-commands must be refused before psql spawns."""

    def test_shell_escape_is_refused_without_spawning(self):
        with patch.object(env_tools.subprocess, "run") as run_mock:
            out = psql_console_impl("\\! echo pwned", db_dsn="dsn")
        assert out == PSQL_GUARDRAIL_REFUSAL
        run_mock.assert_not_called()

    def test_copy_to_file_is_refused(self):
        with patch.object(env_tools.subprocess, "run") as run_mock:
            out = psql_console_impl("\\copy t TO '/tmp/x.csv'", db_dsn="dsn")
        assert out == PSQL_GUARDRAIL_REFUSAL
        run_mock.assert_not_called()

    def test_output_redirect_is_refused(self):
        with patch.object(env_tools.subprocess, "run") as run_mock:
            assert psql_console_impl("\\o /tmp/x", db_dsn="dsn") == PSQL_GUARDRAIL_REFUSAL
        run_mock.assert_not_called()

    def test_include_file_is_refused(self):
        with patch.object(env_tools.subprocess, "run") as run_mock:
            assert psql_console_impl("\\i /etc/passwd", db_dsn="dsn") == PSQL_GUARDRAIL_REFUSAL
        run_mock.assert_not_called()

    def test_g_with_pipe_argument_is_refused(self):
        with patch.object(env_tools.subprocess, "run") as run_mock:
            assert psql_console_impl("SELECT 1 \\g | sh", db_dsn="dsn") == PSQL_GUARDRAIL_REFUSAL
        run_mock.assert_not_called()

    def test_bare_g_is_allowed(self):
        # Bare \g just re-runs the buffer — harmless, must reach psql.
        with patch.object(
            env_tools.subprocess, "run",
            return_value=SimpleNamespace(returncode=0, stdout="ok", stderr=""),
        ):
            assert psql_console_impl("SELECT 1 \\g", db_dsn="dsn") == "ok"

    def test_dt_inspection_command_is_allowed(self):
        with patch.object(
            env_tools.subprocess, "run",
            return_value=SimpleNamespace(returncode=0, stdout="list", stderr=""),
        ):
            assert psql_console_impl("\\dt", db_dsn="dsn") == "list"


class TestPsqlConsoleImpl:
    def test_argv_and_readonly_env(self):
        with patch.object(
            env_tools.subprocess, "run",
            return_value=SimpleNamespace(returncode=0, stdout="rows", stderr=""),
        ) as run_mock:
            out = psql_console_impl("SELECT 1;", db_dsn="postgresql://x/y")
        assert out == "rows"
        args, kwargs = run_mock.call_args
        assert args[0] == ["psql", "postgresql://x/y", "-X", "-c", "SELECT 1;"]
        assert "default_transaction_read_only=on" in kwargs["env"]["PGOPTIONS"]
        assert "statement_timeout=60s" in kwargs["env"]["PGOPTIONS"]
        assert kwargs["timeout"] == env_tools.PSQL_TIMEOUT_S

    def test_nonzero_exit_returns_stderr(self):
        with patch.object(
            env_tools.subprocess, "run",
            return_value=SimpleNamespace(returncode=1, stdout="", stderr="ERROR: boom"),
        ):
            assert psql_console_impl("SELECT bad;", db_dsn="dsn") == "ERROR: boom"

    def test_timeout_is_translated(self):
        with patch.object(
            env_tools.subprocess, "run",
            side_effect=subprocess.TimeoutExpired(cmd="psql", timeout=60),
        ):
            out = psql_console_impl("SELECT pg_sleep(99);", db_dsn="dsn")
        assert "timed out" in out.lower()

    def test_select_output_over_max_rows_is_row_truncated(self):
        """Aligned SQL output is truncated by *rows*: the header, separator and
        first ``MAX_RESULT_ROWS`` data rows are kept (so column names AND real
        data survive), the rest is dropped, and the ``(N rows)`` footer count is
        echoed in the note."""
        aligned = (
            " id | name \n"
            "----+------\n"
            " 1  | a    \n"
            " 2  | b    \n"
            " 3  | c    \n"
            " 4  | d    \n"
            " 5  | e    \n"
            "(5 rows)\n"
        )
        with patch.object(
            env_tools.subprocess, "run",
            return_value=SimpleNamespace(returncode=0, stdout=aligned, stderr=""),
        ):
            out = psql_console_impl("SELECT 1;", db_dsn="dsn")
        lines = out.split("\n")
        assert lines[0] == " id | name "
        assert lines[1] == "----+------"
        kept_data = [ln for ln in lines if ln.startswith(" ") and "|" in ln and "name" not in ln]
        assert len(kept_data) == env_tools.MAX_RESULT_ROWS
        assert " 4  | d    " not in out and " 5  | e    " not in out
        assert f"showing first {env_tools.MAX_RESULT_ROWS} of 5 rows" in out

    def test_select_output_at_or_under_max_rows_is_untouched(self):
        """A result with no more than ``MAX_RESULT_ROWS`` rows is returned
        verbatim (footer and all) — no note."""
        aligned = (
            " id \n"
            "----\n"
            " 1  \n"
            " 2  \n"
            "(2 rows)\n"
        )
        with patch.object(
            env_tools.subprocess, "run",
            return_value=SimpleNamespace(returncode=0, stdout=aligned, stderr=""),
        ):
            out = psql_console_impl("SELECT 1;", db_dsn="dsn")
        assert out == aligned
        assert "showing first" not in out

    def test_meta_command_output_is_not_truncated(self):
        # \dt (and other backslash meta-commands) list table/schema *names* —
        # truncating would drop names off the end of the listing, so the full
        # output must come through untouched however many rows it lists.
        big = "\n".join(f" t{i} " for i in range(50)) + "\n(50 rows)\n"
        with patch.object(
            env_tools.subprocess, "run",
            return_value=SimpleNamespace(returncode=0, stdout=big, stderr=""),
        ):
            out = psql_console_impl("\\dt", db_dsn="dsn")
        assert out == big
        assert "showing first" not in out


def test_psql_console_select_real_db():
    db_dsn = "postgresql://root:123123@localhost:5433/solar_panel"
    out = psql_console_impl("SELECT sitekey FROM plants LIMIT 1;", db_dsn)
    assert "sitekey" in out


def test_psql_console_dt_real_db():
    # \dt is a meta-command, so its full table listing is returned untruncated.
    db_dsn = "postgresql://root:123123@localhost:5433/solar_panel"
    out = psql_console_impl("\\dt", db_dsn)
    assert "plants" in out


def test_psql_console_write_rejected_by_readonly_real_db():
    db_dsn = "postgresql://root:123123@localhost:5433/solar_panel"
    out = psql_console_impl("CREATE TABLE _should_not_exist (id int);", db_dsn)
    assert "read-only" in out.lower()


# ---------------------------------------------------------------------------
# apply_column_comments_impl
# ---------------------------------------------------------------------------
class TestApplyColumnCommentsImpl:
    """apply_column_comments_impl writes COMMENT ON COLUMN to the DB so that
    \\d+ shows column descriptions, closing the gap with get_table_schema."""

    def test_issues_comment_on_column_for_each_valid_key(self):
        """One COMMENT ON COLUMN execute call per valid db|table|column key,
        wrapped in a savepoint pair, then a single commit."""
        from conversation2sql.eval_framework.state import ColumnMeaningEntry
        from unittest.mock import MagicMock, patch

        column_meanings = {
            "mydb|users|id": ColumnMeaningEntry(column_meaning="primary key"),
            "mydb|orders|total": ColumnMeaningEntry(column_meaning="order total"),
        }

        mock_cur = MagicMock()
        mock_conn = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cur)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        with patch.object(env_tools.psycopg2, "connect", return_value=mock_conn):
            apply_column_comments_impl("dsn://x", column_meanings)

        # Each column: SAVEPOINT sp + COMMENT ON COLUMN + RELEASE SAVEPOINT = 3 calls.
        assert mock_cur.execute.call_count == 6
        # Extract only the parameterised COMMENT calls (those with a tuple second arg).
        comment_params = [
            c[0][1] for c in mock_cur.execute.call_args_list if len(c[0]) == 2
        ]
        assert ("primary key",) in comment_params
        assert ("order total",) in comment_params
        mock_conn.commit.assert_called_once()
        mock_conn.close.assert_called_once()

    def test_skips_keys_with_wrong_format(self):
        """Keys that are not in db|table|column form are ignored silently —
        malformed entries in the JSON must not crash the setup step."""
        from conversation2sql.eval_framework.state import ColumnMeaningEntry
        from unittest.mock import MagicMock, patch

        column_meanings = {
            "not_a_valid_key": ColumnMeaningEntry(column_meaning="ignored"),
            "only|two": ColumnMeaningEntry(column_meaning="also ignored"),
        }

        mock_cur = MagicMock()
        mock_conn = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cur)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        with patch.object(env_tools.psycopg2, "connect", return_value=mock_conn):
            apply_column_comments_impl("dsn://x", column_meanings)

        mock_cur.execute.assert_not_called()
        mock_conn.commit.assert_called_once()

    def test_empty_dict_commits_with_no_execute_calls(self):
        """Empty column_meanings: connect, do nothing, commit, close."""
        from unittest.mock import MagicMock, patch

        mock_cur = MagicMock()
        mock_conn = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cur)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        with patch.object(env_tools.psycopg2, "connect", return_value=mock_conn):
            apply_column_comments_impl("dsn://x", {})

        mock_cur.execute.assert_not_called()
        mock_conn.commit.assert_called_once()


def test_apply_column_comments_visible_in_psql_describe_real_db():
    """Integration: after applying a comment, \\d+ <table> shows the description."""
    from conversation2sql.eval_framework.state import ColumnMeaningEntry

    db_dsn = "postgresql://root:123123@localhost:5433/solar_panel"
    test_meanings = {
        "solar_panel|plants|sitekey": ColumnMeaningEntry(
            column_meaning="unique site identifier"
        )
    }
    apply_column_comments_impl(db_dsn, test_meanings)
    out = psql_console_impl("\\d+ plants", db_dsn)
    assert "unique site identifier" in out


# ---------------------------------------------------------------------------
# _safe_instance_prefix
# ---------------------------------------------------------------------------
from conversation2sql.eval_framework.agents.bird_baseline.tools.bird_interact_env_tools import (
    _safe_instance_prefix,
)

def test_safe_instance_prefix_simple():
    assert _safe_instance_prefix("solar_panel_m_5") == "solar_panel_m_5"

def test_safe_instance_prefix_hyphens_and_dots():
    assert _safe_instance_prefix("alien-db.task.1") == "alien_db_task_1"

def test_safe_instance_prefix_uppercase():
    assert _safe_instance_prefix("SolarPanel_M_5") == "solarpanel_m_5"

def test_safe_instance_prefix_truncation():
    long_id = "a" * 50
    result = _safe_instance_prefix(long_id)
    assert len(result) == 30
    assert result == "a" * 30

def test_safe_instance_prefix_default_max_len():
    # default max_len is 30
    result = _safe_instance_prefix("x" * 31)
    assert len(result) == 30

