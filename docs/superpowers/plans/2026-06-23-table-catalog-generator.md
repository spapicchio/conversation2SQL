# Table Catalog Generator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate, for one Postgres database, a Markdown catalog file per table (placeholder description + live DDL with enums + a column/description table + foreign keys), driven by `--database` and `--output-dir`.

**Architecture:** A new script `scripts/generate_catalog.py` **reuses** the existing live-Postgres introspection in `scripts/extract_ddl.py` (`open_readonly`, `fetch_tables`, `load_table`/`Table`, `fetch_enums`, `_render_table_ddl`) and adds: a pure per-table enum filter, a column-meaning loader (over the dataset's `<db>_column_meaning_base.json`, via the reader's `_get_column_meanings`), a pure Markdown renderer, a referenced-by query, and an argparse CLI. The renderer and the two pure helpers are unit-tested without a DB; the one new DB query gets an integration test skipped when Postgres is unreachable.

**Tech Stack:** Python 3.12, `psycopg2`, `argparse`, `uv`, `pytest`.

> **Deviation from the approved spec (intentional, DRY):** The spec named new modules `introspect.py` and `render.py`. During planning we found `scripts/extract_ddl.py` already performs the full live introspection the spec described, so this plan reuses those functions instead of duplicating them. The external contract (live-Postgres source of truth, CLI parameters, per-table Markdown output, empty description placeholder) is unchanged. All new code lives in `scripts/generate_catalog.py`.

## Global Constraints

- Python 3.12; run everything through `uv run` so the project venv is used.
- New script lives in `scripts/`; tests live in `tests/scripts/` and import script modules via the existing `sys.path.insert(0, str(REPO_ROOT / "scripts"))` pattern (see `tests/scripts/test_extract_ddl.py`).
- Postgres DSN default: `postgresql://root:123123@localhost:5432/{database}` (lite); `:5433` for full. Credentials `root` / `123123`.
- Table description is an **empty placeholder** for now (a `## Description` header followed by a blank line). Do not generate descriptions.
- Source of truth: DDL/enums/constraints/FKs from **live Postgres**; column descriptions from the dataset's `<db>_column_meaning_base.json` (keys `db|table|column`, lowercased).
- After implementation, run `uv run pytest tests/` (repo rule).

---

### Task 1: Per-table enum filter (pure helper)

**Files:**
- Create: `scripts/generate_catalog.py`
- Test: `tests/scripts/test_generate_catalog.py`

**Interfaces:**
- Consumes: `Column` dataclass from `extract_ddl` (fields: `name`, `data_type`, `nullable`, `default`, `is_pk`, `is_unique`); `fetch_enums` returns `list[tuple[str, list[str]]]`.
- Produces: `_enums_used_by_table(columns: list[Column], all_enums: list[tuple[str, list[str]]]) -> list[tuple[str, list[str]]]` — the subset of `all_enums` whose type name is used by at least one column (matching `data_type`, ignoring a trailing `[]` for arrays and surrounding quotes), preserving `all_enums` order.

- [ ] **Step 1: Write the failing test**

Create `tests/scripts/test_generate_catalog.py`:

```python
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from extract_ddl import Column  # noqa: E402
from generate_catalog import _enums_used_by_table  # noqa: E402


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/scripts/test_generate_catalog.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'generate_catalog'` (file does not exist yet).

- [ ] **Step 3: Write minimal implementation**

Create `scripts/generate_catalog.py`:

```python
"""Generate a per-table Markdown schema catalog for one Postgres database.

Run with `uv run python scripts/generate_catalog.py --help` for CLI options.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from urllib.parse import urlparse

import psycopg2
from psycopg2.extensions import connection as PgConnection

from extract_ddl import (
    Table,
    Column,
    _render_table_ddl,
    fetch_enums,
    fetch_tables,
    load_table,
    open_readonly,
)

logger = logging.getLogger("generate_catalog")


def _enums_used_by_table(
    columns: list[Column],
    all_enums: list[tuple[str, list[str]]],
) -> list[tuple[str, list[str]]]:
    enum_names = {name for name, _ in all_enums}
    used: set[str] = set()
    for col in columns:
        base = col.data_type.replace("[]", "").strip().strip('"')
        if base in enum_names:
            used.add(base)
    return [(name, labels) for name, labels in all_enums if name in used]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/scripts/test_generate_catalog.py -v`
Expected: PASS (both tests).

- [ ] **Step 5: Commit**

```bash
git add scripts/generate_catalog.py tests/scripts/test_generate_catalog.py
git commit -m "feat: add per-table enum filter for catalog generator"
```

---

### Task 2: Column-meaning loader

**Files:**
- Modify: `scripts/generate_catalog.py`
- Test: `tests/scripts/test_generate_catalog.py`

**Interfaces:**
- Consumes: `_get_column_meanings(dataset_path: Path, db_name: str) -> dict[str, ColumnMeaningEntry]` from `conversation2sql.eval_framework.dataset_readers.bird_interact_reader` (keys are lowercased `db|table|column`; each value has a `.column_meaning: str`).
- Produces: `load_column_meanings(dataset_path: Path, db_name: str) -> dict[str, dict[str, str]]` — `{table_name_lower: {column_name_lower: meaning}}`. Returns `{}` (with a warning) when the meaning file is missing.

- [ ] **Step 1: Write the failing test**

Append to `tests/scripts/test_generate_catalog.py`:

```python
import json  # noqa: E402

from generate_catalog import load_column_meanings  # noqa: E402


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/scripts/test_generate_catalog.py -k column_meanings -v`
Expected: FAIL — `ImportError: cannot import name 'load_column_meanings'`.

- [ ] **Step 3: Write minimal implementation**

Add to `scripts/generate_catalog.py` (after the imports, add the reader import; then the function). Add this import next to the `extract_ddl` import block:

```python
from conversation2sql.eval_framework.dataset_readers.bird_interact_reader import (
    _get_column_meanings,
)
```

And add the function:

```python
def load_column_meanings(
    dataset_path: Path, db_name: str
) -> dict[str, dict[str, str]]:
    """Load `<db>_column_meaning_base.json` into {table: {column: meaning}}.

    Keys in the source file are lowercased `db|table|column`. Missing file
    yields an empty mapping (catalog still renders, with empty descriptions).
    """
    try:
        raw = _get_column_meanings(dataset_path, db_name)
    except FileNotFoundError:
        logger.warning(
            "no column-meaning file for %s under %s; descriptions will be empty",
            db_name,
            dataset_path,
        )
        return {}
    out: dict[str, dict[str, str]] = {}
    prefix = f"{db_name.lower()}|"
    for key, entry in raw.items():
        if not key.startswith(prefix):
            continue
        parts = key.split("|")
        if len(parts) != 3:
            continue
        _, table, column = parts
        out.setdefault(table, {})[column] = entry.column_meaning
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/scripts/test_generate_catalog.py -k column_meanings -v`
Expected: PASS (both new tests).

- [ ] **Step 5: Commit**

```bash
git add scripts/generate_catalog.py tests/scripts/test_generate_catalog.py
git commit -m "feat: add column-meaning loader for catalog generator"
```

---

### Task 3: Markdown renderer

**Files:**
- Modify: `scripts/generate_catalog.py`
- Test: `tests/scripts/test_generate_catalog.py`

**Interfaces:**
- Consumes: `Table` dataclass from `extract_ddl` (fields: `schema`, `examples_str`, `name`, `columns: list[Column]`, `foreign_keys: list[ForeignKey]`, `indexes`, `checks`, `composite_pk`); `ForeignKey` (fields: `column`, `ref_table`, `ref_column`, `on_delete`); `_render_table_ddl(table, descriptions=None) -> str` (renders only `CREATE TABLE`, no enums).
- Produces: `render_table_markdown(table: Table, enums: list[tuple[str, list[str]]], column_meanings: dict[str, str], referenced_by: list[str]) -> str`. `column_meanings` is keyed by lowercased column name. `referenced_by` is a list of `"table(column)"` strings. The Foreign keys section is omitted when there are neither forward FKs nor referenced-by entries.

- [ ] **Step 1: Write the failing test**

Append to `tests/scripts/test_generate_catalog.py`:

```python
from extract_ddl import ForeignKey, Table  # noqa: E402
from generate_catalog import render_table_markdown  # noqa: E402


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/scripts/test_generate_catalog.py -k render -v`
Expected: FAIL — `ImportError: cannot import name 'render_table_markdown'`.

- [ ] **Step 3: Write minimal implementation**

Add to `scripts/generate_catalog.py`:

```python
def _md_cell(text: str) -> str:
    """Sanitize a value for a Markdown table cell (escape pipes, flatten newlines)."""
    return text.replace("|", "\\|").replace("\r", " ").replace("\n", " ").strip()


def render_table_markdown(
    table: Table,
    enums: list[tuple[str, list[str]]],
    column_meanings: dict[str, str],
    referenced_by: list[str],
) -> str:
    lines: list[str] = []
    lines.append(f"# table: {table.name}")
    lines.append("")
    lines.append("## Description")
    lines.append("")  # intentional empty placeholder

    lines.append("## DDL")
    lines.append("```sql")
    for ename, labels in enums:
        quoted = ", ".join(f"'{lbl}'" for lbl in labels)
        lines.append(f'CREATE TYPE "{ename}" AS ENUM ({quoted});')
    if enums:
        lines.append("")
    lines.append(_render_table_ddl(table, None))
    lines.append("```")
    lines.append("")

    lines.append("## Columns")
    lines.append("| column | type | description |")
    lines.append("| --- | --- | --- |")
    for col in table.columns:
        desc = _md_cell(column_meanings.get(col.name.lower(), ""))
        lines.append(f"| {col.name} | {col.data_type} | {desc} |")

    fk_lines = [
        f"- {fk.column} -> {fk.ref_table}({fk.ref_column})"
        for fk in table.foreign_keys
    ]
    ref_lines = [f"- referenced by: {ref}" for ref in referenced_by]
    if fk_lines or ref_lines:
        lines.append("")
        lines.append("## Foreign keys")
        lines.extend(fk_lines)
        lines.extend(ref_lines)

    lines.append("")
    return "\n".join(lines)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/scripts/test_generate_catalog.py -k render -v`
Expected: PASS (both render tests).

- [ ] **Step 5: Commit**

```bash
git add scripts/generate_catalog.py tests/scripts/test_generate_catalog.py
git commit -m "feat: add Markdown renderer for catalog generator"
```

---

### Task 4: Referenced-by query + CLI + integration test

**Files:**
- Modify: `scripts/generate_catalog.py`
- Test: `tests/scripts/test_generate_catalog.py`

**Interfaces:**
- Consumes: `open_readonly(host, port, user, password, dbname) -> PgConnection`, `fetch_tables(conn, schema) -> list[str]`, `fetch_enums(conn, schema)`, `load_table(conn, schema, name, db_dsn) -> Table` (all from `extract_ddl`); `_enums_used_by_table`, `load_column_meanings`, `render_table_markdown` from Tasks 1–3.
- Produces:
  - `fetch_referenced_by(conn: PgConnection, schema: str, table: str) -> list[str]` — `"<referencing_table>(<referencing_column>)"` for every FK pointing at `table`.
  - `generate_catalog_for_db(database, output_dir, db_dsn_template, dataset_path, schema="public", only_table=None) -> int` — writes `<output_dir>/<database>/<table>.md`; returns the number of tables written.
  - `main() -> int` argparse entry point.

- [ ] **Step 1: Write the failing integration test**

Append to `tests/scripts/test_generate_catalog.py`:

```python
import psycopg2  # noqa: E402
import pytest  # noqa: E402

from generate_catalog import fetch_referenced_by  # noqa: E402

_PROBE_DSN = "postgresql://root:123123@localhost:5432/postgres"


def _db_available() -> bool:
    try:
        conn = psycopg2.connect(_PROBE_DSN, connect_timeout=2)
        conn.close()
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _db_available(), reason="postgres :5432 not reachable")
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/scripts/test_generate_catalog.py -k referenced_by -v`
Expected: FAIL — `ImportError: cannot import name 'fetch_referenced_by'` (or SKIPPED if no DB; if skipped, the import error still surfaces at collection time as an error — implement Step 3 to proceed).

- [ ] **Step 3: Write the implementation**

Add `fetch_referenced_by`, `generate_catalog_for_db`, `parse_args`, `main` to `scripts/generate_catalog.py`:

```python
def fetch_referenced_by(conn: PgConnection, schema: str, table: str) -> list[str]:
    """Tables/columns that hold a foreign key pointing AT ``table``."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT c2.relname AS referencing_table,
                   a.attname  AS referencing_column
            FROM pg_constraint con
            JOIN pg_class c1     ON c1.oid = con.confrelid
            JOIN pg_class c2     ON c2.oid = con.conrelid
            JOIN pg_namespace n  ON n.oid = c1.relnamespace
            JOIN pg_attribute a  ON a.attrelid = con.conrelid
                                AND a.attnum = ANY(con.conkey)
            WHERE con.contype = 'f'
              AND c1.relname = %s
              AND n.nspname = %s
            ORDER BY referencing_table, referencing_column
            """,
            (table, schema),
        )
        return [f"{row[0]}({row[1]})" for row in cur.fetchall()]


def generate_catalog_for_db(
    database: str,
    output_dir: Path,
    db_dsn_template: str,
    dataset_path: Path,
    schema: str = "public",
    only_table: str | None = None,
) -> int:
    dsn = db_dsn_template.format(database=database)
    parsed = urlparse(dsn)
    conn = open_readonly(
        host=parsed.hostname or "localhost",
        port=parsed.port or 5432,
        user=parsed.username or "root",
        password=parsed.password or "",
        dbname=database,
    )
    try:
        meanings = load_column_meanings(dataset_path, database)
        all_enums = fetch_enums(conn, schema)
        table_names = [only_table] if only_table else fetch_tables(conn, schema)

        db_out = output_dir / database
        db_out.mkdir(parents=True, exist_ok=True)

        written = 0
        for name in table_names:
            try:
                table = load_table(conn, schema, name, dsn)
                enums = _enums_used_by_table(table.columns, all_enums)
                referenced_by = fetch_referenced_by(conn, schema, name)
                per_table = meanings.get(name.lower(), {})
                md = render_table_markdown(table, enums, per_table, referenced_by)
            except psycopg2.Error as exc:
                logger.warning("skipping table %s: %s", name, exc)
                continue
            (db_out / f"{name}.md").write_text(md, encoding="utf-8")
            written += 1
        logger.info("wrote %d table file(s) under %s", written, db_out)
        return written
    finally:
        conn.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate a per-table Markdown schema catalog for one database."
    )
    p.add_argument("--database", required=True, help="Postgres database name")
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--table", default=None, help="single table (default: all tables)")
    p.add_argument(
        "--db-dsn-template",
        default="postgresql://root:123123@localhost:5432/{database}",
        help="DSN with a {database} placeholder; use :5433 for the full dataset",
    )
    p.add_argument(
        "--dataset-path",
        type=Path,
        default=Path("data/bird_interact/bird-interact-lite"),
        help="locates <db>_column_meaning_base.json for column descriptions",
    )
    p.add_argument("--schema", default="public")
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    try:
        generate_catalog_for_db(
            database=args.database,
            output_dir=args.output_dir,
            db_dsn_template=args.db_dsn_template,
            dataset_path=args.dataset_path,
            schema=args.schema,
            only_table=args.table,
        )
    except psycopg2.Error as exc:
        logger.error("connection/extraction failed for %s: %s", args.database, exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the integration test**

Run: `uv run pytest tests/scripts/test_generate_catalog.py -k referenced_by -v`
Expected: PASS when Postgres `:5432` is up; SKIPPED otherwise. Either is acceptable.

- [ ] **Step 5: Run the full script suite + whole test suite**

Run: `uv run pytest tests/scripts/test_generate_catalog.py -v`
Expected: all PASS (integration test PASS or SKIPPED).

Run: `uv run pytest tests/`
Expected: green (no regressions).

- [ ] **Step 6: Smoke-test the CLI (manual, requires DB)**

Run (replace `<db>` with a real lite DB name, e.g. a folder under `data/bird_interact/bird-interact-lite/`):

```bash
uv run python scripts/generate_catalog.py \
  --database <db> --output-dir /tmp/catalog -v
```

Expected: `/tmp/catalog/<db>/<table>.md` files exist, each with `# table:`, a `## Description` placeholder, a ```sql``` DDL block, a `## Columns` table, and (where applicable) `## Foreign keys`.

- [ ] **Step 7: Commit**

```bash
git add scripts/generate_catalog.py tests/scripts/test_generate_catalog.py
git commit -m "feat: add referenced-by query and CLI for catalog generator"
```

---

## Self-Review

**Spec coverage:**
- Live-Postgres DDL/enums/constraints/FKs → Tasks 1, 3, 4 (reusing `extract_ddl`). ✓
- Per-table enum filtering (only enums the table uses) → Task 1. ✓
- Column descriptions from `<db>_column_meaning_base.json` → Task 2. ✓
- Empty table-description placeholder → Task 3 renderer. ✓
- Markdown layout (Description / DDL / Columns / Foreign keys, FK section omitted when empty) → Task 3. ✓
- CLI params `--database`, `--output-dir`, `--table`, `--db-dsn-template`, `--dataset-path` → Task 4. ✓
- Output to `<output-dir>/<database>/<table>.md` → Task 4. ✓
- Error handling: connection failure non-zero exit; per-table skip-and-continue → Task 4 (`main`, `generate_catalog_for_db`). ✓
- Tests: pure unit tests for enum filter / loader / renderer; integration test skipped when DB down; full `uv run pytest tests/` → Tasks 1–4. ✓

**Placeholder scan:** No "TBD"/"TODO"/"add error handling"-style placeholders; every code step shows complete code. The only intentional "placeholder" is the empty `## Description` body, which is a product requirement.

**Type consistency:** `Column`/`Table`/`ForeignKey` field names match `extract_ddl` (verified against source). `_enums_used_by_table` returns `list[tuple[str, list[str]]]`, consumed unchanged by `render_table_markdown`'s `enums` param. `load_column_meanings` returns `{table: {column: meaning}}`; the CLI passes `meanings.get(name.lower(), {})` as the renderer's `column_meanings` (keyed by column). `fetch_referenced_by` returns `list[str]` consumed as the renderer's `referenced_by`. Consistent.
