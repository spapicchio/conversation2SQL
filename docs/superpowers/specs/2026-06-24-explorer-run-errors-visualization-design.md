# Explorer: visualize run errors (`results_error.jsonl`)

**Date:** 2026-06-24
**Status:** Approved (design)

## Problem

Each evaluation run directory contains a `results_error.jsonl` alongside
`results_iter*.jsonl`. It holds the instances that **crashed mid-run** — they
never produced a completed record. Each error line is a small object:

```json
{"instance_id": "alien_11", "iteration": 1, "error": "litellm.ContextWindowExceededError: ..."}
```

The explorer's `load_run` only reads `results_iter*.jsonl`, so these crashed
samples are **silently invisible** in every page. A run that errored on 27
samples looks identical to one that completed them — there is no count, no
breakdown, and no way to read the error.

Across the results currently on disk there are 379 such records, dominated by
`litellm.Timeout` (162), `ContextWindowExceededError` (132),
`InternalServerError` (37), `BadRequestError` (33), and a patience-state
assertion `At key 'updated_user_patience'` (15).

## Goal

Surface run errors in the explorer across all three pages — the single-run app
(`app.py`), the anti-pattern page (`pages/patterns.py`), and the compare page
(`pages/compare.py`) — as **count + breakdown + drill-down**.

## Non-goals / invariants

- **Purely diagnostic.** Errors must NOT change `pass_at_1`, `n_instances`,
  `n_total`, `error_distribution` (submit-SQL classes), or any existing chart.
  Every current metric is computed exactly as today.
- No change to how completed records are loaded, deduped, or aggregated.
- Errored-only instances are kept in their own "Run errors" sections; they are
  not woven into the per-task / `join_runs` tables.

## Architecture

Approach: parse and classify errors **once in the loader**, expose them on
`RunData` / `RunStats`, and have each page render a self-contained "Run errors"
section using the existing chart/table idioms. This keeps a single source of
truth (mirroring how `classify_submit_error` already works) and makes the logic
unit-testable.

### Data layer — `explorer/loader.py`

**New function** `classify_run_error(error: str) -> str`, mirroring
`classify_submit_error`. Substring/regex match on the raw error string:

| Class | Match (case-insensitive where sensible) |
|---|---|
| `Timeout` | `litellm.Timeout` / `Timeout` |
| `Context Window Exceeded` | `ContextWindowExceededError` |
| `Internal Server Error` | `InternalServerError` |
| `Bad Request` | `BadRequestError` |
| `Patience State Error` | `updated_user_patience` |
| `Other` | fallback |

Order matters: check the more specific provider classes
(`ContextWindowExceededError`, `BadRequestError`) before generic ones so a
context-window error (which is itself a kind of bad request) is not mislabeled.
A non-string `error` value is coerced with `str(...)` before matching.

**`load_run` changes:**
- After loading records, read `results_error.jsonl` from the run dir (if it
  exists) via the existing `_read_jsonl`.
- Dedup by `(instance_id, iteration)` using the same resume/recover
  double-write guard already applied to records.
- Tag each error dict with `_error_class = classify_run_error(error)`.
- Store the list on `RunData.errors`.

**`RunData` addition:** `errors: list[dict] = field(default_factory=list)`.

**`RunStats` additions.** `load_run` parses + dedups the error list first, then
passes it into `_compute_stats(records, groups, errors)` (new optional
`errors` param defaulting to `()`), which computes:
- `n_errors: int` — number of (deduped) error records.
- `run_error_distribution: Counter[str]` — `_error_class` → count.

> Named `run_error_*` to avoid colliding with the existing
> `RunStats.error_distribution`, which classifies submit-SQL failures of
> *completed* records — a different concept.

Both default to `0` / empty so runs without a `results_error.jsonl` (and old
cached fixtures) behave correctly.

### Page 1 — single-run app (`explorer/app.py`)

- Extend the metric-card row to a 7th card **"Errors"** = `stats.n_errors`,
  help text: crashed samples excluded from accuracy.
- New section **"Run errors"**, rendered only when `stats.n_errors > 0`:
  - Bar chart of `run_error_distribution` (mirrors the existing "Error
    Distribution" altair bar: error class on x, count on y, value labels).
  - Drill-down `st.dataframe` (`instance_id`, `iteration`, `error_class`) with
    single-row selection; selecting a row shows the full raw error string via
    `st.code(...)`. Error records have no `messages`, so `render_conversation`
    is not used here.

### Page 2 — anti-pattern page (`explorer/pages/patterns.py`)

Patterns are derived from tool traces that errored samples lack, so errors get
a standalone section rather than being folded into the heatmaps.

- New section **"Run errors"** at the bottom (after the existing drill-down):
  count caption + the same bar chart + the same drill-down table with raw
  message. Rendered only when the run has errors.
- This section is **independent of** the page-wide execution-accuracy scope
  toggle (errors have no execution accuracy), and reads from `run.errors`
  (the full run), not the scoped `records`.

### Page 3 — compare page (`explorer/pages/compare.py`)

- New row **"Errored instances"** in the operational comparison table
  (`n_errors` per run; lower-is-better highlight via the existing
  `_highlight_best` direction mechanism).
- A row of **"Run errors"** bar charts, one column per run (mirrors the
  existing per-run "Error Distribution" `st.columns` row), shown only when at
  least one selected run has errors. A run with no errors shows an info note in
  its column for alignment.
- An **expander per run** listing its errored `(instance_id, iteration,
  error_class)` rows, so drill-down is still possible. Kept out of the
  `join_runs` per-task table.

## Testing

In `tests/` (mirroring existing explorer tests):

- `classify_run_error`: one assertion per class (`Timeout`,
  `Context Window Exceeded`, `Internal Server Error`, `Bad Request`,
  `Patience State Error`) plus an `Other` fallback and a non-string input.
- `load_run`:
  - A fixture run dir with `results_iter*.jsonl` + `results_error.jsonl`
    populates `RunData.errors`, `stats.n_errors`, and
    `stats.run_error_distribution` correctly, and leaves `pass_at_1` /
    `n_instances` identical to the same run without the error file.
  - A run dir **without** `results_error.jsonl` yields `errors == []`,
    `n_errors == 0`, empty `run_error_distribution`.
  - A duplicate `(instance_id, iteration)` error line is deduped.

## Files touched

- `explorer/loader.py` — classifier, `load_run`, `RunData`, `RunStats`,
  `_compute_stats`.
- `explorer/app.py` — metric card + "Run errors" section.
- `explorer/pages/patterns.py` — "Run errors" section.
- `explorer/pages/compare.py` — operational row + per-run charts + expanders.
- `tests/...` — new tests for the classifier and loader.
