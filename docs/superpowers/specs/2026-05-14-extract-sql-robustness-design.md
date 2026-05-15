---
name: extract-sql-robustness
description: Robust SQL extraction from LLM responses — handles unclosed fences, dialect tags, and raw SQL heuristic
metadata:
  type: project
---

# extract_sql_from_response — Robustness Design

**File**: `src/conversation2sql/eval_framework/agents/no_tool_baseline/baseline_model.py`

## Problem

The current `extract_sql_from_response` relies on a single closed-fence regex. It returns `None` for:
- Unclosed fences (model writes ` ```sql` but never closes with ` ``` `)
- Raw SQL output with no fence at all (model reasons at length then emits a bare `SELECT`)

Both cases are recorded as `error="no_sql_block_found"` and counted as failures, even though the SQL is present in the response.

## Scope

- Only `sql` and bare (unlabelled) fences are supported — no dialect tags (`sqlite`, `postgresql`, etc.)
- No changes to the public signature of `extract_sql_from_response`
- No changes to `run_baseline_no_tool` or any call site

## Architecture

`extract_sql_from_response` becomes a dispatcher over three private strategies, tried in priority order:

```
extract_sql_from_response(text)
  └─ _extract_closed_fence(text)    → str | None   priority 1 (existing logic)
  └─ _extract_unclosed_fence(text)  → str | None   priority 2 (new)
  └─ _extract_raw_sql(text)         → str | None   priority 3 (new)
```

First non-`None`, non-empty result is returned. All three functions live in `baseline_model.py`.

## Components

### `_extract_closed_fence(text)`
Moves the existing `_FENCED_SQL_RE` regex + loop verbatim. Returns the last non-empty fenced block (closed ` ``` ` required).

### `_extract_unclosed_fence(text)`
Detects an opening ` ```sql` or ` ``` ` with no matching closing ` ``` `. Captures the text after the fence opener, then:
- If `;` is present → return text up to and including the first `;`, stripped
- Otherwise → return everything to end-of-string, stripped
- Return `None` if no unclosed fence exists or captured content is empty

### `_extract_raw_sql(text)`
Scans for the **last** line starting (after optional whitespace) with a SQL keyword: `SELECT`, `WITH`, `INSERT`, `UPDATE`, `DELETE`, `CREATE`, `DROP`, `ALTER`, `EXPLAIN`. Matching is case-insensitive with a word-boundary check (regex `\b` after the keyword, e.g. `r"\bSELECT\b"`) to avoid matching `SELECTING`. Lines starting with `--` before the keyword are skipped. From the matched line forward:
- If `;` is present → return text up to and including the first `;`, stripped
- Otherwise → return everything to end-of-string, stripped
- Return `None` if no keyword line is found

## Error Handling

No exceptions are raised. All strategies return `None` on failure. The `None` contract with `run_baseline_no_tool` is preserved — `error="no_sql_block_found"` is still recorded when all three strategies fail.

## Test Plan

New cases added to `TestExtractSqlFromResponse` in `tests/eval_framework/agents/test_no_tool_baseline.py`:

| Scenario | Expected |
|---|---|
| Unclosed fence with semicolon | content up to and including `;` |
| Unclosed fence without semicolon | full content to end |
| Unclosed fence, semicolon mid-block | content up to first `;` only |
| Raw SQL — `SELECT` keyword | extracted from that line to `;` |
| Raw SQL — `WITH` keyword | extracted from that line to `;` |
| Raw SQL — last keyword wins (two SELECT lines) | second SELECT content |
| Raw SQL — no semicolon | everything from keyword line to end |
| Keyword inside a word (`SELECTING`) | `None` (not extracted) |
| Closed fence takes priority over unclosed | closed block content returned |
