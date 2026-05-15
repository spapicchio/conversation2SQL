# GT SQL Table/Column Metadata Design

Date: 2026-05-09
Status: Brainstormed and approved with implementation constraints
Owner: eval_framework dataset_readers

## Goal
Add a new per-task metadata field at dataset-reading time that captures table/column usage from the GT SQL query.

New field name (required):
- `table_in_gt_sql`

Field semantics:
- Type: `dict[str, list[str]]`
- Key: table name bucket
- Value: deduplicated list of used columns for that bucket

## Final Requirements (from brainstorming)
1. Compute metadata for every loaded task in `load_bird_interact_as_tasks` (always-on, no config flag).
2. Add metadata under `table_in_gt_sql`.
3. Resolve aliases to real table names and deduplicate columns.
4. Expand wildcard usage (`*`) to all columns available in the corresponding table schema.
5. For unqualified/ambiguous columns, attempt schema-based disambiguation; if still ambiguous, place them under an `unknown` bucket.
6. Include tables/columns from all query scopes (main query + subqueries + CTEs + expressions).
7. CTE behavior:
- Include CTE output columns and base-table columns used inside CTEs.
- Add dedicated CTE buckets named `cte_<cte_name>` to make CTE origin explicit.
8. Parser strategy:
- Start from the parser tooling already in codebase (`sqlglot`, currently used in tools utils).
- Prefer robust AST parsing over token segmentation-only heuristics.
9. Parse failures:
- Do not fail dataset loading.
- Log a warning.
- Store raw parse error text in sample data so downstream preprocessing can remove failed examples deterministically.
10. Keep backward compatibility for existing pipeline/task execution behavior.

## Data Contract
Mandatory new metadata key in each task:
- `table_in_gt_sql: dict[str, list[str]]`

Companion failure marker in each task (proposed):
- `table_in_gt_sql_parse_error: str | None`

Rationale:
- User requested raw parse error kept in sample for future filtering.
- Keeping a dedicated parse-error field is simpler and safer than overloading `table_in_gt_sql`.

## Proposed Normalization Rules
1. Table bucket names:
- Physical tables: normalized lowercase table names.
- CTE buckets: `cte_<cte_name_lower>`.
- Unknown bucket: `unknown`.

2. Column names:
- Store lowercase column names.
- Deduplicate per bucket.
- Sort alphabetically for deterministic output.

3. Wildcard handling:
- `table_alias.*` -> expand using resolved base table/CTE schema.
- Bare `*` -> expand against all visible source tables in current scope.

4. Ambiguous columns:
- If exactly one visible source table contains the column, assign there.
- If multiple candidates remain, add to `unknown`.

## Algorithm Outline
1. Parse GT SQL with `sqlglot` (`postgres` dialect).
2. Build scope-aware mapping:
- alias -> base table
- CTE name -> projected columns
- visible source tables for each select scope
3. Collect column references from all expressions/scopes.
4. Resolve each reference to bucket(s):
- real table bucket, cte bucket, or `unknown`
5. Expand stars (`*`) using schema/column catalog from `column_meanings` and/or CTE projections.
6. Materialize deterministic `table_in_gt_sql` dict.
7. On exceptions: set empty dict, set parse error field, log warning with `instance_id`.

## Non-Goals
1. SQL semantic validation or execution.
2. Guaranteeing full equivalence for every dialect feature beyond repo-supported GT SQL.
3. Changing any agent/tool behavior.

## Acceptance Criteria
1. Every `TaskData` produced by reader contains `table_in_gt_sql`.
2. Alias-only references resolve to real table buckets.
3. CTE-heavy query creates both:
- `cte_<name>` buckets
- base table buckets used inside CTE definitions
4. `*` correctly expands to all known columns.
5. Ambiguous unresolved columns appear in `unknown`.
6. Parse failures are visible via both logs and `table_in_gt_sql_parse_error`.
7. Existing tests pass; new reader tests cover happy path + edge cases.

## Risks and Mitigations
1. SQL AST complexity with nested CTEs/subqueries.
- Mitigation: centralize logic in a dedicated helper module with focused unit tests.
2. Performance impact during dataset load.
- Mitigation: keep per-query parse O(query size), avoid DB calls, and use deterministic in-memory schema maps.
3. Incomplete wildcard expansion for exotic constructs.
- Mitigation: conservative fallback to `unknown` and explicit parse-error markers where needed.

## Open Implementation Detail (resolved recommendation)
Use `column_meanings` keys (`db|table|column`) as authoritative table->columns catalog for wildcard expansion, with fallback parsing from DDL only if needed.
