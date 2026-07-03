from __future__ import annotations

import json
import sys
from pathlib import Path

import psycopg2
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from extract_ddl import Column, ForeignKey, Table, _render_table_ddl  # noqa: E402
from generate_catalog import (  # noqa: E402
    _enums_used_by_table,
    _generate_kb_for_db,
    fetch_referenced_by,
    generate_catalog_for_db,
    load_column_meanings,
    render_constraints_markdown,
    render_database_overview_markdown,
    render_table_markdown,
)

from conversation2sql.eval_framework.state import ExternalKnowledgeEntry  # noqa: E402


def _col(name: str, data_type: str) -> Column:
    return Column(
        name=name,
        data_type=data_type,
        nullable=True,
        default=None,
        is_pk=False,
        is_unique=False,
    )


def test_enums_used_by_table_filters_and_preserves_order():
    columns = [
        _col("id", "integer"),
        _col("status", "account_status"),
        _col("tags", "label_kind[]"),
    ]
    all_enums = [
        ("label_kind", ["a", "b"]),
        ("account_status", ["active", "closed"]),
        ("unused_enum", ["x"]),
    ]
    result = _enums_used_by_table(columns, all_enums)
    assert result == [
        ("label_kind", ["a", "b"]),
        ("account_status", ["active", "closed"]),
    ]


def test_enums_used_by_table_empty_when_none_used():
    columns = [_col("id", "integer")]
    all_enums = [("account_status", ["active", "closed"])]
    assert _enums_used_by_table(columns, all_enums) == []


def test_generate_kb_for_db_slugifies_filenames_with_spaces_and_parens(
    tmp_path, monkeypatch
):
    entry = ExternalKnowledgeEntry(
        id=1,
        knowledge="Coherent Information Pattern (CIP)",
        description="",
        definition="",
        type="domain_knowledge",
        children_knowledge=[],
    )
    monkeypatch.setattr(
        "generate_catalog._get_external_knowledge",
        lambda dataset_path, database: {"Coherent Information Pattern (CIP)": entry},
    )

    written = _generate_kb_for_db(tmp_path, "mydb", tmp_path)

    assert written == 1
    assert (tmp_path / "coherent_information_pattern_cip.md").exists()
    assert not (tmp_path / "Coherent Information Pattern (CIP).md").exists()


def test_render_database_overview_without_kb_has_no_kb_section():
    result = render_database_overview_markdown("mydb", ["users"])
    assert "## Knowledge Base" not in result


def test_render_database_overview_includes_kb_index():
    kb = {
        "Active User (AU)": ExternalKnowledgeEntry(
            id=1,
            knowledge="Active User (AU)",
            description="logged in recently",
            definition="",
            type="domain_knowledge",
            children_knowledge=[],
        ),
    }
    result = render_database_overview_markdown("mydb", ["users"], kb)
    assert "## Knowledge Base" in result
    assert "Active User (AU)" in result
    assert "knowledge_base/active_user_au.md" in result
    assert "logged in recently" in result


def _write_meaning_file(tmp_path: Path, db_name: str, payload: dict) -> Path:
    db_dir = tmp_path / db_name
    db_dir.mkdir(parents=True)
    (db_dir / f"{db_name}_column_meaning_base.json").write_text(json.dumps(payload))
    return tmp_path


def test_load_column_meanings_groups_by_table(tmp_path):
    dataset_path = _write_meaning_file(
        tmp_path,
        "mydb",
        {
            "mydb|customers|email": "Login email, unique",
            "mydb|customers|id": "Surrogate PK",
            "mydb|orders|customer_id": "FK to customers",
        },
    )
    result = load_column_meanings(dataset_path, "mydb")
    assert result == {
        "customers": {"email": "Login email, unique", "id": "Surrogate PK"},
        "orders": {"customer_id": "FK to customers"},
    }


def test_load_column_meanings_missing_file_returns_empty(tmp_path):
    assert load_column_meanings(tmp_path, "absent_db") == {}


_PROBE_DSN = "postgresql://root:123123@localhost:5432/postgres"


def _db_available() -> bool:
    try:
        conn = psycopg2.connect(_PROBE_DSN, connect_timeout=2)
        conn.close()
        return True
    except Exception:
        return False


_DB_AVAILABLE = _db_available()


@pytest.mark.skipif(not _DB_AVAILABLE, reason="postgres :5432 not reachable")
def test_fetch_referenced_by_against_live_db():
    conn = psycopg2.connect(_PROBE_DSN)
    conn.autocommit = True
    schema = "catalog_gen_test"
    try:
        with conn.cursor() as cur:
            cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
            cur.execute(f"CREATE SCHEMA {schema}")
            cur.execute(f"CREATE TABLE {schema}.parent (id integer PRIMARY KEY)")
            cur.execute(
                f"CREATE TABLE {schema}.child ("
                f"  id integer PRIMARY KEY,"
                f"  parent_id integer REFERENCES {schema}.parent(id))"
            )
        refs = fetch_referenced_by(conn, schema, "parent")
        assert refs == ["child(parent_id)"]
        assert fetch_referenced_by(conn, schema, "child") == []
    finally:
        with conn.cursor() as cur:
            cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
        conn.close()


def _make_table() -> Table:
    return Table(
        schema="public",
        examples_str="",
        name="customers",
        columns=[
            _col("id", "integer"),
            _col("email", "text"),
            _col("status", "account_status"),
            _col("region_id", "integer"),
        ],
        foreign_keys=[ForeignKey("region_id", "regions", "id", "NO ACTION")],
        indexes=[],
        checks=[],
        composite_pk=[],
    )


def test_render_table_markdown_full():
    md = render_table_markdown(
        table=_make_table(),
        enums=[("account_status", ["active", "closed"])],
        column_meanings={"email": "Login email, unique", "status": "active or closed"},
        referenced_by=["orders(customer_id)"],
        examples_by_col={"email": ["a@x.com", "b@y.com"], "region_id": ["1", "2"]},
    )
    assert "# table: customers" in md
    assert "## Description" in md
    assert "## DDL" in md
    assert "```sql" in md
    assert "CREATE TYPE \"account_status\" AS ENUM ('active', 'closed');" in md
    assert 'CREATE TABLE "customers" (' in md
    assert 'FOREIGN KEY ("region_id") REFERENCES "regions" ("id")' in md
    assert "## Columns" in md
    assert "| column | type | description | examples |" in md
    assert "| email | text | Login email, unique | a@x.com, b@y.com |" in md
    assert "| id | integer |  |  |" in md
    assert "## Foreign keys" in md
    assert "- region_id -> regions(id)" in md
    assert "- referenced by: orders(customer_id)" in md


def test_render_table_markdown_omits_empty_fk_section():
    table = _make_table()
    table.foreign_keys = []
    md = render_table_markdown(
        table, enums=[], column_meanings={}, referenced_by=[], examples_by_col={}
    )
    assert "## Foreign keys" not in md
    assert "CREATE TYPE" not in md
    assert 'CREATE TABLE "customers" (' in md
    assert "FOREIGN KEY" not in md


def _pk_col(name: str, data_type: str) -> Column:
    return Column(
        name=name,
        data_type=data_type,
        nullable=False,
        default=None,
        is_pk=True,
        is_unique=False,
    )


def test_render_constraints_markdown_lists_pks_and_fks():
    customers = Table(
        schema="public",
        examples_str="",
        name="customers",
        columns=[_pk_col("id", "integer"), _col("region_id", "integer")],
        foreign_keys=[ForeignKey("region_id", "regions", "id", "NO ACTION")],
        indexes=[],
        checks=[],
        composite_pk=[],
    )
    order_items = Table(
        schema="public",
        examples_str="",
        name="order_items",
        columns=[_col("order_id", "integer"), _col("product_id", "integer")],
        foreign_keys=[
            ForeignKey("order_id", "orders", "id", "CASCADE"),
            ForeignKey("product_id", "products", "id", "RESTRICT"),
        ],
        indexes=[],
        checks=[],
        composite_pk=["order_id", "product_id"],
    )
    md = render_constraints_markdown("mydb", [customers, order_items])
    assert "# constraints: mydb" in md
    assert "## Primary keys" in md
    assert (
        'ALTER TABLE "customers" ADD CONSTRAINT "pk_customers" '
        'PRIMARY KEY ("id");'
    ) in md
    assert (
        'ALTER TABLE "order_items" ADD CONSTRAINT "pk_order_items" '
        'PRIMARY KEY ("order_id", "product_id");'
    ) in md
    assert "## Foreign keys" in md
    assert (
        'ALTER TABLE "customers" ADD CONSTRAINT "fk_customers_region_id" '
        'FOREIGN KEY ("region_id") REFERENCES "regions" ("id");'
    ) in md
    assert (
        'ALTER TABLE "order_items" ADD CONSTRAINT "fk_order_items_order_id" '
        'FOREIGN KEY ("order_id") REFERENCES "orders" ("id");'
    ) in md
    assert (
        'ALTER TABLE "order_items" ADD CONSTRAINT "fk_order_items_product_id" '
        'FOREIGN KEY ("product_id") REFERENCES "products" ("id");'
    ) in md
    # ON DELETE is intentionally omitted from this file's FK statements.
    assert "ON DELETE" not in md


def test_render_constraints_markdown_skips_tables_without_pk_or_fk():
    no_key_table = Table(
        schema="public",
        examples_str="",
        name="lookup",
        columns=[_col("label", "text")],
        foreign_keys=[],
        indexes=[],
        checks=[],
        composite_pk=[],
    )
    md = render_constraints_markdown("mydb", [no_key_table])
    assert "## Primary keys" in md
    assert "## Foreign keys" in md
    assert "ALTER TABLE" not in md


@pytest.mark.skipif(not _DB_AVAILABLE, reason="postgres :5432 not reachable")
def test_generate_catalog_for_db_end_to_end(tmp_path):
    """Exercise the full load_table -> render -> write path against a live DB.

    Regression guard: load_table must not invoke the broken fetch_examples
    path (it would raise TypeError, not psycopg2.Error, and crash the run).
    """
    conn = psycopg2.connect(_PROBE_DSN)
    conn.autocommit = True
    schema = "catalog_gen_e2e"
    try:
        with conn.cursor() as cur:
            cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
            cur.execute(f"CREATE SCHEMA {schema}")
            cur.execute(f"CREATE TABLE {schema}.customer (id integer PRIMARY KEY)")
            cur.execute(
                f"CREATE TABLE {schema}.orders ("
                f"  id integer PRIMARY KEY,"
                f"  customer_id integer REFERENCES {schema}.customer(id))"
            )
        written = generate_catalog_for_db(
            database="postgres",
            output_dir=tmp_path,
            db_dsn_template="postgresql://root:123123@localhost:5432/{database}",
            dataset_path=tmp_path,  # no meaning file -> empty descriptions
            schema=schema,
            only_table="orders",
        )
        assert written == 1
        out_file = tmp_path / "postgres" / "orders.md"
        assert out_file.exists()
        md = out_file.read_text()
        assert "# table: orders" in md
        assert 'CREATE TABLE "orders" (' in md
        assert "## Foreign keys" in md
        assert "- customer_id ->" in md
        assert "customer(id)" in md
    finally:
        with conn.cursor() as cur:
            cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
        conn.close()


def test_render_table_ddl_includes_inline_fk_when_enabled():
    table = _make_table()
    ddl = _render_table_ddl(table, None, include_foreign_keys=True)
    assert 'FOREIGN KEY ("region_id") REFERENCES "regions" ("id")' in ddl
    # Still a single valid CREATE TABLE statement.
    assert ddl.count("CREATE TABLE") == 1
    assert ddl.rstrip().endswith(");")


def test_render_table_ddl_omits_fk_by_default():
    table = _make_table()
    ddl = _render_table_ddl(table, None)
    assert "FOREIGN KEY" not in ddl
