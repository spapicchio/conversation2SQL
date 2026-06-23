# Table Catalog Generator — Design

Date: 2026-06-23
Status: Approved

## Purpose

Generate an offline, file-based **schema catalog** for a BIRD-Interact database so a
model (or human) can navigate the schema table-by-table. For a given database the
generator produces one Markdown file per table containing:

- a placeholder **table description** (left empty for now; filled later, e.g. LLM-generated),
- the table's **live DDL** reconstructed from Postgres (`CREATE TYPE … AS ENUM` for the
  enums it uses, followed by a reconstructed `CREATE TABLE` with PK/FK/UNIQUE/CHECK
  constraints),
- a **Columns** table joining each column's real Postgres type with its description from
  the dataset's `<db>_column_meaning_base.json`,
- a **Foreign keys** section listing forward FKs and "referenced-by" back-references.

This is an **offline artifact only**. No agent tools change in this work; consumption is
wired up later.

## Source of truth

- **DDL / types / constraints / enums / FKs:** the **live Postgres** database
  (`:5432` lite, `:5433` full). Extracted via `pg_catalog` (approach A below).
- **Column descriptions:** the dataset file `<db>_column_meaning_base.json`
  (keys `db|table|column`, loaded into `ColumnMeaningEntry`).
- **Table description:** empty placeholder for now.

## Approach (decision)

**A — psycopg2 + `pg_catalog`, assemble the DDL ourselves.** Single connection, no
external binary. Columns from `pg_attribute` + `format_type`; constraints rendered exactly
via `pg_get_constraintdef`; enum values from `pg_enum` (only enums actually used by the
table's columns); back-references from `pg_constraint` (`confrelid = table`).

Rejected alternatives:
- **B — shell out to `pg_dump -t`:** most authoritative `CREATE TABLE`, but it omits the
  `CREATE TYPE … AS ENUM` for the table's enums and gives no structured column list, so we
  would still need `pg_catalog` for enums, back-refs, and the Columns table — mixing two
  mechanisms and adding a `pg_dump`-version-vs-server dependency. Harder to test.
- **C — reuse the existing `_parse_ddl` on the dataset `.txt`:** not the live DB; rejected
  because live Postgres is the chosen source of truth.

## Architecture

Three isolated units plus a thin CLI:

### 1. `introspect.py` — pure DB reader
Input: a psycopg2 connection + table name. Output: a structured `TableCatalog` dataclass:

- `name: str`
- `columns: list[ColumnInfo]` where `ColumnInfo = (name, pg_type, not_null, default)`
  (`pg_type` via `format_type(atttypid, atttypmod)`)
- `constraints: list[str]` — exact DDL text from `pg_get_constraintdef` (PK, FK, UNIQUE, CHECK)
- `enums: list[EnumInfo]` where `EnumInfo = (type_name, values: list[str])` — only enum
  types used by this table's columns, values ordered by `enumsortorder`
- `foreign_keys: list[str]` — forward FK summaries (e.g. `region_id -> regions(id)`)
- `referenced_by: list[str]` — back-references (e.g. `orders(customer_id)`)

No file I/O, no formatting.

### 2. `render.py` — pure renderer
Pure function: `TableCatalog` + column-meaning dict → Markdown string. Layout:

```
# table: <name>

## Description
<empty placeholder>

## DDL
```sql
CREATE TYPE <enum> AS ENUM (...);   -- enums first, only if any
CREATE TABLE <name> (
  <col> <type> [NOT NULL] [DEFAULT ...],
  ...
  <constraint ddl text>,
  ...
);
```

## Columns
| column | type | description |
|--------|------|-------------|
| ...    | ...  | ...         |

## Foreign keys              # omitted if none
- <forward fk>
- referenced by: <back-ref>
```

No DB, no I/O.

### 3. `scripts/generate_catalog.py` — CLI glue
Connects, enumerates the DB's tables (or one if `--table`), runs introspect → render →
writes files.

## CLI parameters

- `--database` (required) — Postgres DB name.
- `--output-dir` (required) — root; files land in `<output-dir>/<database>/<table>.md`.
- `--table` (optional) — single table; default = all tables in the database.
- `--db-dsn-template` (default `postgresql://root:123123@localhost:5432/{database}`) —
  swap port to `:5433` for the full dataset.
- `--dataset-path` (default `data/bird_interact/bird-interact-lite`) — locates
  `<db>_column_meaning_base.json` for column descriptions.

## Data flow

1. CLI substitutes `--database` into `--db-dsn-template`, opens a connection.
2. Lists tables from `pg_catalog` (public schema), or uses `--table`.
3. Loads column meanings for the DB (reusing the reader's `_get_column_meanings`).
4. Per table: `introspect()` → `render()` → write `<output-dir>/<database>/<table>.md`.

## Column meanings

Reuse `_get_column_meanings` from
`src/conversation2sql/eval_framework/dataset_readers/bird_interact_reader.py`
(loads `<db>_column_meaning_base.json` into `ColumnMeaningEntry`, lowercased
`db|table|column` keys). If the file is missing: warn and emit the Columns table with
empty descriptions.

## Error handling

- Connection failure → exit non-zero with a clear message.
- A table that cannot be introspected → log a warning, skip it, continue with the rest.
- Empty enum list or no FKs → omit that section rather than emit an empty header. The
  `## Description` section is the intentional exception (empty placeholder).

## Testing

- Unit-test `render.py` against a hand-built `TableCatalog` fixture: enum-before-table
  ordering, column/meaning join, presence/omission of the Foreign keys section, empty
  Description placeholder.
- Unit-test the DDL-assembly helper: column-line formatting with NOT NULL / DEFAULT.
- Integration test for `introspect.py` against the live container, **skipped** when
  `:5432` is unreachable so the suite still passes offline.
- Run `uv run pytest tests/` after implementation (repo rule).

## Out of scope (YAGNI for now)

- LLM-generated table descriptions (placeholder only for now).
- Any agent tool / runtime integration that consumes the catalog.
- Batch generation across all databases in one invocation (driven one DB at a time via
  `--database`; loop externally if needed).
