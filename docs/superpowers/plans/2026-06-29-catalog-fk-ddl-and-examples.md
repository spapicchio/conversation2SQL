# Catalog: inline FK DDL + per-column examples — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enrich each per-table Markdown catalog file with inline `FOREIGN KEY` constraints in its `CREATE TABLE` DDL and a per-column `examples` column populated by type-aware value sampling.

**Architecture:** Add an opt-in `include_foreign_keys` flag to `_render_table_ddl` in `scripts/extract_ddl.py` (default off, so `render_classical_ddl` is untouched). In `scripts/generate_catalog.py`, widen the `## Columns` table with an `examples` column fed by the existing `select_examples()` (reused from `extract_ddl`), computed per column in `_generate_tables_for_db` using the live connection.

**Tech Stack:** Python 3.12, psycopg2, pytest (run via `uv run`).

## Global Constraints

- Always run Python/pytest through `uv run` (project venv).
- Existing human-readable `## Foreign keys` section and db-level `_foreign_key_constraints.md` stay unchanged.
- `render_classical_ddl` and TOON output must keep their current behavior (FK rendering there is separate; no duplicate FKs).
- ON DELETE is omitted from the inline FK DDL (agreed format).
- Pre-existing failures `test_render_constraints_markdown_lists_pks_and_fks` and `test_generate_catalog_for_db_end_to_end` are OUT OF SCOPE — do not fix them, but do not introduce any new failures.

---

### Task 1: Opt-in inline FK rendering in `_render_table_ddl`

**Files:**
- Modify: `scripts/extract_ddl.py` (`_render_table_ddl`, around lines 471-508)
- Test: `tests/scripts/test_generate_catalog.py` (already imports from `extract_ddl`)

**Interfaces:**
- Produces: `_render_table_ddl(table: Table, descriptions: dict[str, str] | None = None, include_foreign_keys: bool = False) -> str`. When `include_foreign_keys=True`, the returned `CREATE TABLE` body contains one `    FOREIGN KEY ("<col>") REFERENCES "<ref_table>" ("<ref_column>")` row per entry in `table.foreign_keys`, sharing the existing trailing-comma logic.

- [ ] **Step 1: Write the failing test**

Add to `tests/scripts/test_generate_catalog.py` (import `_render_table_ddl` in the existing `from extract_ddl import ...` line):

```python
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
```

Update the import at the top of the file:

```python
from extract_ddl import Column, ForeignKey, Table, _render_table_ddl  # noqa: E402
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/scripts/test_generate_catalog.py::test_render_table_ddl_includes_inline_fk_when_enabled -v`
Expected: FAIL — `include_foreign_keys` is an unexpected keyword argument.

- [ ] **Step 3: Write minimal implementation**

In `scripts/extract_ddl.py`, change the signature and append FK rows. Replace the `_render_table_ddl` signature line:

```python
def _render_table_ddl(
    table: Table,
    descriptions: dict[str, str] | None = None,
    include_foreign_keys: bool = False,
) -> str:
```

Then, immediately after the existing `for chk in table.checks:` block that appends check rows (just before `rendered: list[str] = []`), add:

```python
    if include_foreign_keys:
        for fk in table.foreign_keys:
            rows.append(
                (
                    f'    FOREIGN KEY ("{fk.column}") '
                    f'REFERENCES "{fk.ref_table}" ("{fk.ref_column}")',
                    None,
                )
            )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/scripts/test_generate_catalog.py::test_render_table_ddl_includes_inline_fk_when_enabled tests/scripts/test_generate_catalog.py::test_render_table_ddl_omits_fk_by_default -v`
Expected: PASS (both).

- [ ] **Step 5: Commit**

```bash
git add scripts/extract_ddl.py tests/scripts/test_generate_catalog.py
git commit -m "feat(catalog): opt-in inline FOREIGN KEY rendering in _render_table_ddl"
```

---

### Task 2: Examples column + inline FK DDL in the per-table catalog

**Files:**
- Modify: `scripts/generate_catalog.py` (`render_table_markdown` lines 121-166; `_generate_tables_for_db` lines 246-283; imports lines 62-70)
- Test: `tests/scripts/test_generate_catalog.py` (`test_render_table_markdown_full`, `test_render_table_markdown_omits_empty_fk_section`)

**Interfaces:**
- Consumes: `_render_table_ddl(..., include_foreign_keys=True)` from Task 1; `select_examples(conn, schema, table, col, fk_by_col) -> list[str]` from `extract_ddl`.
- Produces: `render_table_markdown(table, enums, column_meanings, referenced_by, examples_by_col: dict[str, list[str]]) -> str` — `## Columns` header is `| column | type | description | examples |`, each row appends `_md_cell(", ".join(examples_by_col.get(col.name, [])))`, and the `## DDL` block calls `_render_table_ddl(table, None, include_foreign_keys=True)`.

- [ ] **Step 1: Update the failing tests**

In `tests/scripts/test_generate_catalog.py`, update `test_render_table_markdown_full` to pass examples and assert the new column + inline FK DDL:

```python
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
```

Update `test_render_table_markdown_omits_empty_fk_section` to pass the new arg:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/scripts/test_generate_catalog.py::test_render_table_markdown_full -v`
Expected: FAIL — `render_table_markdown()` got an unexpected keyword argument `examples_by_col`.

- [ ] **Step 3: Implement `render_table_markdown` changes**

In `scripts/generate_catalog.py`, change the signature and body. Signature (lines 121-126):

```python
def render_table_markdown(
    table: Table,
    enums: list[tuple[str, list[str]]],
    column_meanings: dict[str, str],
    referenced_by: list[str],
    examples_by_col: dict[str, list[str]],
) -> str:
```

In the `## DDL` block, change the `_render_table_ddl(table, None)` call (line 142) to:

```python
    lines.append(_render_table_ddl(table, None, include_foreign_keys=True))
```

Replace the `## Columns` block (lines 146-153) with:

```python
    lines.append("## Columns")
    lines.append("| column | type | description | examples |")
    lines.append("| --- | --- | --- | --- |")
    for col in table.columns:
        name = _md_cell(col.name)
        data_type = _md_cell(col.data_type)
        desc = _md_cell(column_meanings.get(col.name.lower(), ""))
        examples = _md_cell(", ".join(examples_by_col.get(col.name, [])))
        lines.append(f"| {name} | {data_type} | {desc} | {examples} |")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/scripts/test_generate_catalog.py::test_render_table_markdown_full tests/scripts/test_generate_catalog.py::test_render_table_markdown_omits_empty_fk_section -v`
Expected: PASS (both).

- [ ] **Step 5: Wire `select_examples` into `_generate_tables_for_db`**

In `scripts/generate_catalog.py`, add `select_examples` to the `from extract_ddl import (...)` block (lines 62-70):

```python
from extract_ddl import (
    Column,
    Table,
    _render_table_ddl,
    fetch_enums,
    fetch_tables,
    load_table,
    open_readonly,
    select_examples,
)
```

In `_generate_tables_for_db`, inside the `try:` (after `per_table = meanings.get(name.lower(), {})`, before the `render_table_markdown` call, lines ~262-263), build the examples map and pass it:

```python
            per_table = meanings.get(name.lower(), {})
            fk_by_col = {fk.column: fk for fk in table.foreign_keys}
            examples_by_col = {
                col.name: select_examples(conn, schema, table, col, fk_by_col)
                for col in table.columns
            }
            md = render_table_markdown(
                table, enums, per_table, referenced_by, examples_by_col
            )
```

- [ ] **Step 6: Run the full suite — no NEW failures**

Run: `uv run pytest tests/ -q`
Expected: the only failures are the two pre-existing ones noted in Global Constraints (`test_render_constraints_markdown_lists_pks_and_fks`, `test_generate_catalog_for_db_end_to_end`). All other tests, including the four touched/added here, PASS.

- [ ] **Step 7: Type-check**

Run: `uv run pyrefly check`
Expected: no new errors introduced by `scripts/generate_catalog.py` or `scripts/extract_ddl.py`.

- [ ] **Step 8: Commit**

```bash
git add scripts/generate_catalog.py tests/scripts/test_generate_catalog.py
git commit -m "feat(catalog): add per-column examples and inline FK DDL to table catalog"
```

---

## Optional manual verification (live DB on :5432)

Regenerate the sample file and eyeball it:

```bash
uv run python scripts/generate_catalog.py \
    --database alien --table observationalconditions \
    --output-dir /tmp/claude-0/-workspaces-conversation2SQL/412e143a-684f-46c1-9ff8-28b2d6c87e6c/scratchpad/catalog_check
```

Confirm the generated `.md` has an `examples` column in `## Columns` and a `FOREIGN KEY ("signalref") REFERENCES "signals" ("signalregistry")` line inside `CREATE TABLE`.

## Self-review notes

- Spec coverage: inline FK DDL (Task 1 + Task 2 step 3), per-column examples as 4th column (Task 2), `## Foreign keys` section + constraints file untouched (no task modifies them). ✅
- Type consistency: `_render_table_ddl(..., include_foreign_keys=...)` and `render_table_markdown(..., examples_by_col=...)` signatures match between definition and call sites. ✅
- Pre-existing failures explicitly fenced off in Global Constraints and Task 2 step 6. ✅
