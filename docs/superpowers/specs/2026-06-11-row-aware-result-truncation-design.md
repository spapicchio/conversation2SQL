# Row-aware result truncation for `execute_sql` and `psql_console`

**Date:** 2026-06-11
**Status:** Approved (design)

## Problem

Both DB-result paths truncate output with a blind `output[:500]` character cut:

- `execute_sql_impl` (`bird_interact_env_tools.py`) builds a markdown table via
  `_format_result` (up to 100 rows), then cuts the whole string to 500 chars.
- `psql_console_impl` runs real `psql` and cuts SQL output to 500 chars
  (backslash meta-commands like `\d` are already exempt).

A character cut is blind to row structure. For wide results — and especially for
`psql`, whose default **aligned** format pads every column to its widest value —
the header line plus the `---+---+---` separator can exceed 500 chars *before a
single data row*, so the agent sees only column names and mistakes a working
query for a broken one (wasting bird-coins re-running it).

## Goal

Truncate by **whole rows**, not characters: always show the agent a small number
of complete rows, and add an explicit note when more rows existed. Keep the two
tools' output **comparable** — neither caps cell/row width — so the only
difference between them is the underlying renderer.

## Decisions

- **`MAX_RESULT_ROWS = 3`.** Deliberately tiny: the agent issues many SQL calls
  per task, and larger row dumps flood its context (and cost).
- **No width capping anywhere.** The existing per-cell 100-char cap in
  `_format_cell` is **removed**, no per-line cap is added for psql, and there is
  no global char backstop. This keeps `execute_sql` and `psql_console` output
  directly comparable (both width-unbounded; they differ only in renderer and
  row count). Accepted tradeoff: a single very large text/JSON cell can be long.
- **`MAX_RESULT_LENGTH` / `TRUNCATION_NOTICE`** (the old 500-char machinery) are
  retired; a new row-count note replaces the notice.

## Design

### Constants (`bird_interact_env_tools.py`)

Add `MAX_RESULT_ROWS = 3`. Remove `MAX_RESULT_LENGTH` and the old
`TRUNCATION_NOTICE`; introduce a small note builder so both paths emit identical
wording, e.g.:

```
... [showing first 3 of {N} rows; query ran successfully — add a LIMIT or select fewer columns to see more]
```

### `execute_sql` (`_format_result` + `execute_sql_impl`)

- `_format_result` gains a `max_rows` parameter (default `MAX_RESULT_ROWS`) and
  renders header + separator + the first `max_rows` rows (replacing the current
  `result[:100]`).
- `_format_cell` no longer truncates: it serializes dict/list to compact JSON as
  today but returns the full string (drop `s[:max_characters]`). The
  `max_characters` parameter is removed from `_format_cell` and `_format_result`.
- When `len(result) > max_rows`, append the note. `_execute_query` fetches up to
  10 000 rows, so `len(result)` is the true total; when it equals 10 000 (fetch
  cap hit) the note reports `10000+`.
- Remove the `if len(format_result) > MAX_RESULT_LENGTH: ...[:500]` block in
  `execute_sql_impl`.

### `psql_console` (`psql_console_impl`)

- Meta-commands (`\d`, `\dt`, `\l`, `\df`, …) stay exempt and untruncated, as
  today (detected by `_is_psql_meta_command`).
- For SQL output: split into lines, keep the 2 header lines (column header +
  `---+---` separator) plus the first `MAX_RESULT_ROWS` data rows. Parse the
  trailing `(N rows)` footer for the true total; when `N > MAX_RESULT_ROWS`,
  append the same note. No per-line width capping.

## Testing

Extend `tests/eval_framework/tools/test_bird_interact_env_tools.py`:

- `execute_sql`: result with > 3 rows → exactly 3 rows rendered + note with the
  correct total; result with ≤ 3 rows → no note; a long text/JSON cell renders in
  full (no truncation); fetch-cap case reports `10000+`.
- `psql_console`: aligned SQL output with > 3 rows → header + 3 rows + note with
  the footer count; ≤ 3 rows → no note; a backslash meta-command is returned
  untouched.

## Out of scope

- Changing psql's output format (aligned → unaligned/CSV).
- Token-based budgeting of results.
- Width/column trimming of any kind.
