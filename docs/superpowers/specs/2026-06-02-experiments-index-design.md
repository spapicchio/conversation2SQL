# Experiments Index — Design

**Date:** 2026-06-02
**Status:** Approved (brainstorming) — pending spec review

## Problem

Evaluation runs are launched via the `justfile` and land in `results/<date>/<time>/<slug>/`.
Each run dir already carries a full `config.yaml` snapshot (and the same config is
embedded in every `results_iter*.jsonl` record), so per-run config is **already
captured**. What is missing is a **single cross-run index** the user can scan at a
glance, sort/diff to spot ablations, and annotate with free-text notes.

## Decision summary

- **Format:** a CSV — greppable, opens in Excel, diffs cleanly in git. (SQLite is the
  future escape hatch if cross-run querying is ever needed; not now.)
- **Location:** `experiments.csv` at the repo root, **git-tracked**. `results/` is
  gitignored, so putting the index there would leave hand-typed Notes un-versioned and
  local-only. Tracking it in git versions/shares the Notes.
- **Population:** **Hybrid** — a stub row is appended at launch; a reconcile pass
  rescans `results/` to backfill existing runs, fill metrics, and repair status.
- **Metrics source of truth:** the reconcile pass calls `explorer.loader.load_run()`
  and flattens its `RunStats`, so the CSV can never disagree with the explorer app.
- **Key:** `run_dir` = path **relative to `results/`** (e.g.
  `2026-06-02/08-48-35/no_tool__Qwen3.5-9B__ddl__lin__gt-db__gt-kb__iter9`). Stable and
  machine-independent. Discovered by scanning the tree (robust to the existing quirk
  where `config.yaml:output_folder` omits the `__iterN` suffix).

## Columns

Dedicated columns (easy filtering):

| Column     | Source |
|------------|--------|
| `run_dir`  | path relative to `results/` (the key) |
| `date`     | first path segment |
| `time`     | second path segment |
| `status`   | `running` \| `done` \| `error` \| `partial` (derived; see below) |
| `baseline` | `config.pipeline.baseline` |
| `model`    | `config.predictor.model_name` |

Single text column (so ablations diff as one token):

| Column | Source |
|--------|--------|
| `args` | canonical flag string rendered from `config.yaml` (see below) |

Metrics columns (flattened from `explorer.loader` `RunStats`):

| Column                  | RunStats field |
|-------------------------|----------------|
| `iters_present`         | `RunData.n_iterations` |
| `n_instances`           | `n_instances` |
| `n_total`               | `n_total` |
| `accuracy`              | `pass_at_1` |
| `avg_cost`              | `avg_cost` |
| `avg_in_tok`            | `avg_input_tokens` |
| `avg_out_tok`           | `avg_output_tokens` |
| `avg_budget_remaining`  | `avg_budget_remaining` |
| `reliability`           | `reliability.reliability` |
| `aptitude`              | `reliability.aptitude` |
| `unreliability`         | `reliability.unreliability` |

Manual column:

| Column  | Source |
|---------|--------|
| `Notes` | hand-edited; preserved across reconcile by `run_dir` |

### The `args` string

Rendered deterministically from `config.yaml` in **both** the launch-stub and
reconcile paths, so the two always agree. Derived from config (not the raw
`conv2sql` CLI), because reconcile only has `config.yaml` to work from. Flag names are
descriptive mirrors of the config fields, not necessarily byte-identical to what was
typed on the command line. Canonical fixed order; boolean flags emitted **only when
true** so two ablations differing in one knob differ by exactly one token.

Order and mapping:

```
--schema-type <reader.database_schema_type>
[--kb-linearized]            (when reader.is_kb_linearized)
[--gt-db]                    (when reader.read_only_gt_tables)
[--gt-kb]                    (when reader.read_only_gt_kb)
[--ambiguous]                (when reader.make_data_ambiguous)
--num-iterations <pipeline.num_iterations>
--temperature <predictor.temperature>
--top-p <predictor.top_p>
[--thinking]                 (when predictor.enable_thinking)
--provider <predictor.model_provider>
--user-sim <user_simulator.model_name>
```

Example:
```
--schema-type ddl --kb-linearized --gt-db --gt-kb --num-iterations 9 \
--temperature 0.6 --top-p 0.95 --thinking --provider hosted_vllm \
--user-sim gpt-5.4-mini-2026-03-17
```

### `status` derivation (reconcile)

- `error`   — run dir name ends in `__error`, or a `results_error.jsonl` exists.
- `done`    — `iters_present == config.pipeline.num_iterations` (all expected
  iterations present on disk).
- `partial` — at least one `results_iter*.jsonl` present but fewer than expected.
- `running` — set by `append_stub` at launch; only present until the next reconcile
  re-derives one of the above (a stub with no jsonl yet stays `running`).

## Components

New module `explorer/index.py` (lives beside `loader.py`, which it imports):

- `render_args(config: dict) -> str` — the canonical flag string above. Pure function;
  unit-tested directly.
- `row_from_config(run_dir, config) -> dict` — config-only columns (used by the stub).
- `row_from_run(run_dir, RunData) -> dict` — full row incl. flattened metrics.
- `reconcile(results_root: Path, csv_path: Path) -> None`
  1. Read existing CSV (if any) into `{run_dir: {Notes, ...}}`.
  2. `list_runs(results_root)` → for each run, `load_run()` → build full row.
  3. **Carry over `Notes`** from the existing CSV by `run_dir`; new runs get `""`.
  4. Sort rows newest-first (by `date`, `time`).
  5. Write **atomically** (temp file in same dir + `os.replace`).
- `append_stub(run_dir: Path, csv_path: Path) -> None` — idempotent: if `run_dir`
  already a row, no-op; else append a config-only row with `status=running` and empty
  metrics/Notes. Creates the CSV with a header if absent.

CLI / wiring:

- `python -m explorer.index reconcile` and `python -m explorer.index append <run_dir>`
  entrypoints (argparse `__main__`).
- `just index` recipe → `uv run python -m explorer.index reconcile`.
- `bash_scripts/utils/utils_evaluate.sh` `run_suite`: after `run_dir` is created and
  the `config.yaml` snapshot is in place, call
  `uv run python -m explorer.index append "<run_dir>"` (best-effort; failure must not
  abort the eval — wrap so a non-zero exit is logged and ignored).

## Notes-preservation invariant

`reconcile` is non-destructive to `Notes`: existing values are matched by `run_dir` and
re-emitted verbatim; only runs absent from the old CSV get an empty `Notes`. This is the
one behavior guaranteed by a test.

## Testing (`tests/explorer/test_index.py`)

1. `render_args` — given a config dict, returns the expected canonical string; a bool
   knob toggled changes exactly one token; false bools are omitted.
2. `reconcile` on a small fixture results tree → expected rows, columns, and metric
   values (reuse / mirror existing `tests/explorer` fixtures).
3. **Notes preservation:** write CSV, set a `Notes` value on one row, re-run
   `reconcile`, assert the note survives and metrics still update.
4. `append_stub` — idempotent (second call adds no duplicate); creates a
   `status=running` row with config columns populated and metric columns blank before
   any reconcile.
5. `status` derivation — full vs partial vs error fixtures map to the right value.

## Out of scope (YAGNI)

- No auto-commit of `experiments.csv` (the user commits when they choose).
- No per-database or per-error-class columns (those remain in the explorer app).
- No SQLite backend.
- No backfill of `args` from raw shell history — config-derived only.
