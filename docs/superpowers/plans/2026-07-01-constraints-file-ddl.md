# Constraints File DDL Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `render_constraints_markdown()` in `scripts/generate_catalog.py` emit the Primary keys and Foreign keys sections of `_foreign_key_constraints.md` as executable `ALTER TABLE ... ADD CONSTRAINT ...` DDL statements instead of Markdown tables.

**Architecture:** Single-function change, no new files, no new parameters. `render_constraints_markdown(database, tables)` keeps its signature and heading structure; only the body under each `##` heading changes from a Markdown table to a fenced ` ```sql ` block of one `ALTER TABLE` statement per constraint.

**Tech Stack:** Python 3.12, pytest (`uv run pytest`).

## Global Constraints

- Constraint naming: `pk_<table>` for primary keys, `fk_<table>_<column>` for foreign keys (matches `render_classical_ddl` in `scripts/extract_ddl.py`).
- No `ON DELETE` clause on the FK statements (per spec `docs/superpowers/specs/2026-07-01-constraints-file-ddl-design.md`).
- Tables with no PK, or no FKs, are skipped for that statement — the heading and empty `sql` block still render (no change to this existing behavior).
- Per-table `.md` files, `extract_ddl.py`, and TOON output are unchanged — out of scope.
- Run `uv run pytest tests/` after the change (per CLAUDE.md).

---

### Task 1: Render PK/FK sections of `_foreign_key_constraints.md` as DDL

**Files:**
- Modify: `scripts/generate_catalog.py:430-460` (`render_constraints_markdown`)
- Test: `tests/scripts/test_generate_catalog.py:217-249` (`test_render_constraints_markdown_lists_pks_and_fks`)

**Interfaces:**
- Consumes: `Table`, `ForeignKey` dataclasses from `extract_ddl.py` (unchanged: `Table.name`, `Table.foreign_keys: list[ForeignKey]`, `Table.composite_pk`, `Table.columns[i].is_pk`); `_primary_key_columns(table) -> list[str]` (existing helper at `scripts/generate_catalog.py:423-427`, unchanged).
- Produces: `render_constraints_markdown(database: str, tables: list[Table]) -> str` — signature unchanged, only the returned string's content changes.

- [ ] **Step 1: Update the existing test to assert the new DDL output**

Replace the body of `test_render_constraints_markdown_lists_pks_and_fks` in `tests/scripts/test_generate_catalog.py` (currently lines 217-249) with:

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/scripts/test_generate_catalog.py::test_render_constraints_markdown_lists_pks_and_fks tests/scripts/test_generate_catalog.py::test_render_constraints_markdown_skips_tables_without_pk_or_fk -v`

Expected: `test_render_constraints_markdown_lists_pks_and_fks` FAILs (asserting `ALTER TABLE ...` strings not present in the current Markdown-table output); `test_render_constraints_markdown_skips_tables_without_pk_or_fk` PASSes trivially (no new behavior needed for that one — it's a new regression guard, written now so both tests move together).

- [ ] **Step 3: Implement the DDL rendering**

Replace `render_constraints_markdown` in `scripts/generate_catalog.py` (currently lines 430-460) with:

```python
def render_constraints_markdown(database: str, tables: list[Table]) -> str:
    """Render a single Markdown file listing every PK/FK constraint in the db.

    A ``## Primary keys`` fenced SQL block (one ``ALTER TABLE ... ADD
    CONSTRAINT ... PRIMARY KEY (...);`` statement per table that has a PK)
    followed by a ``## Foreign keys`` fenced SQL block (one ``ALTER TABLE ...
    ADD CONSTRAINT ... FOREIGN KEY (...) REFERENCES ...;`` statement per FK,
    no ``ON DELETE`` clause). Tables are listed in the order they were loaded.
    """
    lines: list[str] = []
    lines.append(f"# constraints: {database}")
    lines.append("")

    lines.append("## Primary keys")
    lines.append("```sql")
    for table in tables:
        pk_cols = _primary_key_columns(table)
        if not pk_cols:
            continue
        cols = ", ".join(f'"{c}"' for c in pk_cols)
        lines.append(
            f'ALTER TABLE "{table.name}" ADD CONSTRAINT "pk_{table.name}" '
            f"PRIMARY KEY ({cols});"
        )
    lines.append("```")
    lines.append("")

    lines.append("## Foreign keys")
    lines.append("```sql")
    for table in tables:
        for fk in table.foreign_keys:
            lines.append(
                f'ALTER TABLE "{table.name}" '
                f'ADD CONSTRAINT "fk_{table.name}_{fk.column}" '
                f'FOREIGN KEY ("{fk.column}") '
                f'REFERENCES "{fk.ref_table}" ("{fk.ref_column}");'
            )
    lines.append("```")
    lines.append("")
    return "\n".join(lines)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/scripts/test_generate_catalog.py -v`

Expected: all tests in the file PASS, including the two touched in this task.

- [ ] **Step 5: Run the full test suite**

Run: `uv run pytest tests/`

Expected: no new failures introduced by this change (pre-existing failures unrelated to `generate_catalog.py`, if any, are out of scope).

- [ ] **Step 6: Commit**

```bash
git add scripts/generate_catalog.py tests/scripts/test_generate_catalog.py
git commit -m "$(cat <<'EOF'
feat(catalog): render constraints file PK/FK as DDL

ALTER TABLE ... ADD CONSTRAINT ... statements replace the plain Markdown
tables in _foreign_key_constraints.md, making the file directly runnable
as SQL.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```
