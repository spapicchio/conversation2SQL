# Constraints file: render PK/FK as DDL

Date: 2026-07-01

## Goal

`render_constraints_markdown()` in `scripts/generate_catalog.py` writes
`_foreign_key_constraints.md`, the db-level file listing every table's
primary/foreign keys. Today both sections are plain Markdown tables of column
names. Switch both to fenced `sql` blocks of executable DDL statements, so the
file doubles as a copy-pasteable constraint script (matching the FK-as-DDL
direction already taken for the per-table catalog files in
[[2026-06-29-catalog-fk-ddl-and-examples-design]]).

## Design

Only `render_constraints_markdown()` changes. Heading structure
(`# constraints: <database>`, `## Primary keys`, `## Foreign keys`), iteration
order over `tables`, and skip-if-no-PK / skip-if-no-FK behavior are unchanged.

**Primary keys** — one line per table with a PK (composite or single-column,
via the existing `_primary_key_columns()` helper):

```sql
ALTER TABLE "<table>" ADD CONSTRAINT "pk_<table>" PRIMARY KEY (<col1>, <col2>, ...);
```

**Foreign keys** — one line per `ForeignKey` on each table:

```sql
ALTER TABLE "<table>" ADD CONSTRAINT "fk_<table>_<column>" FOREIGN KEY ("<column>") REFERENCES "<ref_table>" ("<ref_column>");
```

No `ON DELETE` clause (kept out on purpose, even though `fk.on_delete` is
available) to keep these statements shorter than `extract_ddl.py`'s
`render_classical_ddl`, whose FK section does include it.

Constraint naming (`pk_<table>`, `fk_<table>_<column>`) matches the
`fk_<table>_<column>` convention `render_classical_ddl` already uses, so
naming stays consistent across the two scripts.

Both sections are wrapped in a single ` ```sql ... ``` ` block per section
(not one block per statement), each statement on its own line.

## Testing

`tests/scripts/test_generate_catalog.py` already covers
`render_constraints_markdown`. Update its assertions to check for the new
`ALTER TABLE ... ADD CONSTRAINT ...` lines instead of the old table rows, for
both a composite-PK and a single-column-PK table, and for a table with no PK
and no FKs (both sections must still render their heading with no rows).

Run `uv run pytest tests/` after the change (per CLAUDE.md).

## Out of scope

- Per-table `.md` files are unchanged (already handled in
  [[2026-06-29-catalog-fk-ddl-and-examples-design]]).
- No `ON DELETE` clause on the FK statements.
- No changes to `extract_ddl.py`.
