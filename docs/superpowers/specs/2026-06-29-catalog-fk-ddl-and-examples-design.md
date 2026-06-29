# Catalog: inline FK DDL + per-column examples

Date: 2026-06-29

## Goal

Enrich the per-table Markdown catalog produced by `scripts/generate_catalog.py`
(e.g. `data/bird_interact/catalog_bird_interact_lite/alien/tables/observationalconditions.md`)
with two additions that help a text-to-SQL model:

1. **Foreign keys rendered as DDL** — inline `FOREIGN KEY (...) REFERENCES ...`
   constraints inside the `CREATE TABLE` statement in the `## DDL` block, so
   each file is one self-contained, valid statement.
2. **Per-column value examples** — a 4th `examples` column in the `## Columns`
   table, populated by the existing type-aware `select_examples()` logic.

The existing human-readable `## Foreign keys` section is **kept** as-is.

## Why these choices

- **Inline FK in DDL** keeps every per-table file a complete, runnable
  `CREATE TABLE`. (`render_classical_ddl` renders FKs as separate `ALTER TABLE`
  statements because it dumps the whole DB at once; the per-table catalog has no
  such ordering concern.)
- **Per-column examples (not raw sample rows).** What most often breaks
  generated SQL is wrong literal values. `select_examples()` (already used and
  tested in TOON mode) surfaces each column's value domain:
  - low-cardinality (≤8 distinct) → **all** distinct values (the `WHERE`-clause
    vocabulary for enum-like columns — the biggest win),
  - numeric → min/median/max, date/time → min/max,
  - FK columns → real sampled keys from the referenced table,
  - text → top-3 most frequent values, `NULL` prefix when nullable,
  - sensitive columns (password/hash/token/…) masked.
  Raw "first 3 rows" under-samples exactly the low-cardinality columns that
  matter most for filters, and `fetch_examples` is currently buggy.
- **As a 4th column** (not a separate section) co-locates name/type/description/
  examples per column on one line and reuses `_md_cell` escaping.

## Design

### 1. FK as inline DDL — `scripts/extract_ddl.py`

Add an opt-in parameter to `_render_table_ddl`:

```python
def _render_table_ddl(
    table: Table,
    descriptions: dict[str, str] | None = None,
    include_foreign_keys: bool = False,
) -> str:
```

When `include_foreign_keys` is true, append one row per FK to the existing
`rows` list (after composite PK and checks, so the existing trailing-comma
logic handles them):

```
    FOREIGN KEY ("<col>") REFERENCES "<ref_table>" ("<ref_column>")
```

Default `False` preserves `render_classical_ddl`'s current output (it renders
FKs separately and must not gain duplicates). `generate_catalog` passes `True`.
ON DELETE is omitted to match the agreed format.

### 2. Per-column examples — `scripts/generate_catalog.py`

- `render_table_markdown` gains an `examples_by_col: dict[str, list[str]]`
  parameter. The `## Columns` header/separator gain an `examples` column, and
  each row renders `_md_cell(", ".join(examples_by_col.get(col.name, [])))`.
- It calls `_render_table_ddl(table, None, include_foreign_keys=True)`.
- `_generate_tables_for_db` computes examples per table using the live `conn`:
  ```python
  fk_by_col = {fk.column: fk for fk in table.foreign_keys}
  examples_by_col = {
      col.name: select_examples(conn, schema, table, col, fk_by_col)
      for col in table.columns
  }
  ```
  `select_examples` already swallows per-column DB errors and returns `[]`, so a
  flaky column degrades to an empty cell rather than aborting the table (which
  is itself wrapped in the existing per-table try/except).
- Import `select_examples` from `extract_ddl`.

The db-level `_foreign_key_constraints.md` file is unchanged.

## Testing

`tests/scripts/test_generate_catalog.py` already exists. Add/adjust:
- `render_table_markdown` emits the `examples` column header and values, and the
  DDL block contains an inline `FOREIGN KEY (...) REFERENCES ...` line.
- `_render_table_ddl(..., include_foreign_keys=True)` includes FK lines;
  default call still omits them (guards `render_classical_ddl`).

Run `uv run pytest tests/` after the change (per CLAUDE.md).

## Out of scope

- No changes to TOON output or `render_classical_ddl` behavior.
- No new CLI flags (examples + FK DDL are always on for the catalog).
- `fetch_examples` is not fixed here (unused by the catalog path).
