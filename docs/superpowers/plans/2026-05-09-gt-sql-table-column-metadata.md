# TDD Plan: GT SQL table_in_gt_sql Metadata

Date: 2026-05-09
Approach: test-driven development (red -> green -> refactor)

## Objective
Implement always-on reader enrichment that adds `table_in_gt_sql` (table -> used columns) for each task, including alias resolution, CTE buckets, wildcard expansion, ambiguity fallback, and parse-failure traceability.

## Step 1 (RED): Add failing unit tests for metadata extraction helper
Files to create/modify:
- `tests/eval_framework/dataset_readers/test_gt_sql_table_columns.py` (new)

Tests to add first:
1. Basic join with aliases:
- Input SQL references aliases only.
- Expect real table-name buckets with deduped columns.

2. Wildcard expansion:
- `t.*` expands to all columns in table `t`.
- Bare `*` expands over all visible source tables in scope.

3. Ambiguous unqualified column:
- Column exists in multiple visible tables.
- Expect entry under `unknown` bucket.

4. Disambiguation success:
- Unqualified column exists in exactly one visible table.
- Expect assignment to that table bucket.

5. CTE coverage:
- Query with one or more CTEs.
- Expect:
  - `cte_<name>` bucket populated with CTE-output columns
  - base-table buckets populated for columns used inside CTE definitions

6. Parse error contract:
- Invalid SQL triggers graceful failure.
- Expect empty `table_in_gt_sql` plus non-empty parse error string.

Test style:
- Keep tests pure unit tests (no DB).
- Build schema catalog from synthetic `column_meanings`-style keys for deterministic cases.

## Step 2 (GREEN): Implement extraction helper module
Files to create/modify:
- `src/conversation2sql/eval_framework/dataset_readers/sql_usage_extractor.py` (new)

Implementation tasks:
1. Public function:
- `extract_table_in_gt_sql(sql: str, table_to_columns: dict[str, set[str]], dialect: str = "postgres") -> tuple[dict[str, list[str]], str | None]`

2. Internals:
- Parse SQL with `sqlglot` AST traversal.
- Resolve aliases to real table names.
- Track scope visibility for subqueries/CTEs.
- Build CTE buckets as `cte_<cte_name>`.
- Expand stars from `table_to_columns` and CTE projections.
- Attempt schema-based disambiguation for unqualified columns.
- Route unresolved ambiguous refs to `unknown`.
- Deduplicate + sort columns per bucket.

3. Failure handling:
- Catch parse/traversal exceptions.
- Return `({}, raw_error_message)`.

## Step 3 (RED): Reader integration tests
Files to create/modify:
- `tests/eval_framework/dataset_readers/test_bird_interact_reader_metadata_integration.py` (new)

Tests to add first:
1. `load_bird_interact_as_tasks` adds `table_in_gt_sql` for valid sample.
2. Parse-failure sample still loads and includes parse error marker.
3. Metadata exists regardless of `make_data_ambiguous` value.

Notes:
- Use temporary minimal dataset fixtures (JSONL + schema/column meaning/KB files).
- Keep test dataset tiny to minimize IO.

## Step 4 (GREEN): Wire helper into reader
Files to modify:
- `src/conversation2sql/eval_framework/dataset_readers/bird_interact_reader.py`

Changes:
1. Build `table_to_columns` map from `column_meanings` keys.
2. Extract GT SQL usage from primary GT SQL (`sol_sql[0]`), optionally merge across all `sol_sql` entries if needed (decide and codify in tests).
3. Set new TaskData extra fields:
- `table_in_gt_sql`
- `table_in_gt_sql_parse_error`
4. Log warning on parse failure with `instance_id` and compact error summary.

## Step 5 (GREEN/COMPAT): Ensure state model compatibility
Files to modify:
- `src/conversation2sql/eval_framework/state.py`

Changes:
1. Add explicit optional fields to `TaskData` (recommended for typing clarity):
- `table_in_gt_sql: dict[str, list[str]] = Field(default_factory=dict)`
- `table_in_gt_sql_parse_error: str | None = None`

Why:
- Keeps contract discoverable and typed while preserving `extra='allow'` behavior.

## Step 6 (REFACTOR): Reuse existing SQL utilities carefully
Files to evaluate/possibly modify:
- `src/conversation2sql/eval_framework/agents/bird_baseline/tools/utils.py`

Actions:
1. Keep `_segment_sql` untouched for existing tool behavior unless tests show necessary changes.
2. If shared parser helpers are beneficial, extract minimal common utility without coupling reader logic to tool runtime behavior.

## Step 7: Validate full suite and regressions
Commands to run (required by repo guidance):
1. `uv run pytest tests/eval_framework/dataset_readers/`
2. `uv run pytest tests/`

Optional:
3. `uv run pyrefly check`

## Step 8: Documentation and traceability
Files to modify:
- `src/conversation2sql/eval_framework/dataset_readers/README.md`

Additions:
1. Document new per-task fields.
2. Document parse-failure marker strategy for downstream preprocessing filters.

## Definition of Done
1. New and existing tests pass.
2. Every loaded task has `table_in_gt_sql`.
3. Parse failures are visible via `table_in_gt_sql_parse_error` and warnings.
4. CTE/alias/wildcard/ambiguity behavior matches tests and spec.
5. Reader documentation updated.
