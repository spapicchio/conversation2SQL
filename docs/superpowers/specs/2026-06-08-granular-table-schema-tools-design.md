# Granular DB schema tools (mirror the KB pattern)

**Date:** 2026-06-08
**Status:** Approved — ready for implementation

## Motivation

In the `tools_only` baseline, the model never fetches the entire KB at once: it
calls `get_all_external_knowledge_names` to discover names, then pulls each
definition with `get_knowledge_definition`. The DB side has no such granular
path — only `get_schema`, which dumps the whole DDL blob (every `CREATE TABLE`
+ "First 3 rows" sample data) in one shot. This mirrors
`get_all_knowledge_definitions` with no per-item equivalent.

This work adds the two missing granular tools so the agent can explore the
database table-by-table the same way it explores the KB.

## DDL file structure (input)

`{db_name}_ddl.txt` (read once per DB, `@cache`d by the reader into
`TaskData.ddl_database_schema`) is a single text blob:

1. A header comment (`-- PostgreSQL schema dump for schema: public`).
2. A series of `CREATE TABLE "name" (...)` blocks, each followed by a
   `First 3 rows:` sample-data block and a `...` separator. `PRIMARY KEY` is
   inline in the `CREATE TABLE`.
3. **At the end of the file**, a block of
   `ALTER TABLE "child" ADD CONSTRAINT "..." FOREIGN KEY ("col") REFERENCES "parent" ("col") ...;`
   statements. There are no standalone `CREATE INDEX` statements in the corpus.

## New tools

Keep `get_schema` (full dump) unchanged. Add two tools in
`bird_baseline/tools/bird_interact_env_tools.py`:

| Tool | Cost | KB analogue | Returns |
|------|------|-------------|---------|
| `get_table_names()` | 0.5 | `get_all_external_knowledge_names` | JSON list of table names parsed from the DDL |
| `get_table_schema(table_name)` | 0.5 | `get_knowledge_definition` | The table's `CREATE TABLE` block + its "First 3 rows" sample + every related FK |
| `get_schema()` *(unchanged)* | 1.0 | `get_all_knowledge_definitions` | The whole DDL blob |

### `get_table_schema` content

For a given `table_name`, return:

- The table's `CREATE TABLE "table_name" (...)` block (carries inline
  `PRIMARY KEY`), **including** the trailing "First 3 rows" sample-data block.
- Every `ALTER TABLE … FOREIGN KEY …` statement where the table appears
  **either as the child** (`ALTER TABLE "table_name"` target) **or as the
  referenced parent** (`REFERENCES "table_name"`). Including both directions
  surfaces joinable tables regardless of which side of the FK the agent is
  inspecting.

Returns `"Table not found."` for an unknown name (mirrors KB's
`"Knowledge not found."`).

## Parsing

A pure helper `_parse_ddl(ddl)` splits the blob into:

- `tables: dict[str, str]` — `{table_name: block}`, where each block runs from
  its `CREATE TABLE "name"` line up to the next `CREATE TABLE`, the first
  `ALTER TABLE`, or EOF (so the sample-rows block is included).
- `alters: list[str]` — the trailing `ALTER TABLE` statements, one per entry.

Table names are matched via a regex on `CREATE TABLE "?name"?`. The two
`*_impl` functions wrap the helper:

- `get_table_names_impl(ddl) -> {"names": [...]}` — names in DDL order.
- `get_table_schema_impl(table_name, ddl) -> {"schema": "..."}` — block + the
  related `ALTER` lines joined with the block; `{"schema": "Table not found."}`
  when the name is absent.

These have no LangGraph dependency and are unit-tested directly, like every
other tool's `*_impl`.

## Wiring touchpoints

- `DB_TOOL_COSTS` += `get_table_names: 0.5`, `get_table_schema: 0.5`; update the
  top-of-file cost-summary docstring.
- `tools/__init__.py` — imports + `__all__` (both `@tool` wrappers and both
  `*_impl` functions).
- `agent_code.py` — add both tools to the `tools=[...]` list.
- `prompts.py` — add both tools to the "Available tools and costs" list.
- `tools/CLAUDE.md` + root `CLAUDE.md` — extend the cost tables.

## Testing

New unit tests in `tests/eval_framework/tools/`:

- `_parse_ddl` splits tables correctly and captures the trailing `ALTER` block.
- `get_table_names_impl` returns all names in order.
- `get_table_schema_impl` includes the `CREATE TABLE` block **and** the sample
  rows.
- `get_table_schema_impl` includes FK lines where the table is the child **and**
  where it is the referenced parent.
- `get_table_schema_impl` returns `"Table not found."` for an unknown name.

Run `uv run pytest tests/` after implementation.

## Out of scope

- No DB-side `information_schema`/`pg_indexes` querying — all data comes from the
  static DDL blob already in `TaskData`.
- No change to `get_schema`, KB tools, or the patience-budget middleware.
