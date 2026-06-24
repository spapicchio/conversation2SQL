from __future__ import annotations

import json
import sys
from pathlib import Path

import psycopg2
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from extract_ddl import Column, ForeignKey, Table  # noqa: E402
from generate_catalog import (  # noqa: E402
    _enums_used_by_table,
    fetch_referenced_by,
    generate_catalog_for_db,
    load_column_meanings,
    render_table_markdown,
)


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
    )
    assert "# table: customers" in md
    assert "## Description" in md
    assert "## DDL" in md
    assert "```sql" in md
    assert "CREATE TYPE \"account_status\" AS ENUM ('active', 'closed');" in md
    assert 'CREATE TABLE "customers" (' in md
    assert "## Columns" in md
    assert "| column | type | description |" in md
    assert "| email | text | Login email, unique |" in md
    assert "| id | integer |  |" in md
    assert "## Foreign keys" in md
    assert "- region_id -> regions(id)" in md
    assert "- referenced by: orders(customer_id)" in md


def test_render_table_markdown_omits_empty_fk_section():
    table = _make_table()
    table.foreign_keys = []
    md = render_table_markdown(table, enums=[], column_meanings={}, referenced_by=[])
    assert "## Foreign keys" not in md
    # Enums absent -> no CREATE TYPE, but CREATE TABLE still present.
    assert "CREATE TYPE" not in md
    assert 'CREATE TABLE "customers" (' in md


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
