# Results Explorer

Streamlit app for browsing and analysing evaluation runs produced by the conversation2SQL pipeline.

## Running

From the **project root**:

```bash
uv run streamlit run explorer/app.py
```

The app expects a `results/` directory in the working directory, laid out as:

```
results/
└── <YYYY_MM_DD>/
    └── <HH_MM_SS>__<slug>/
        ├── tmux_log               # directory with run info
        ├── <bash_id>.sh           # the bash file used to launch the exp
        ├── results_iter*.jsonl    # one file per iteration (or legacy results.jsonl)
        └── config.yaml            # optional run config
```

Runs with `num_iterations > 1` produce one `results_iter{i}.jsonl` per iteration; the
explorer groups them by `instance_id` to compute reliability metrics. Legacy single-file
runs (`results.jsonl` / `results_smaller.jsonl`) load as a single iteration.

The run folder name is generated automatically by the pipeline and encodes the key parameters that distinguish runs: model name, schema type (`ddl` or `toon`), and optional flags (`lin` = KB linearized, `gt-db` = GT tables only, `gt-kb` = GT KB only).

## Features

| Panel | Description |
|---|---|
| **Sidebar** | Cascading selectors (date → run). Shows model name and generation params (collapsible). |
| **Metrics** | Accuracy, avg input/output tokens, avg cost, avg budget remaining. |
| **Reliability** | For multi-iteration runs (`results_iter*.jsonl`): Average P̄, Aptitude A⁹⁰, Unreliability U₁₀⁹⁰, Reliability R, and a pass@k curve. Hidden for single-iteration runs. |
| **Accuracy by Database** | Pass rate per database in the selected run. |
| **Error Distribution** | Breakdown of why failed tasks failed (see below). |
| **Tool Usage** | Per-tool call counts (hidden for baselines with no tools). |
| **Task table** | Filterable by pass/fail, error class, and free-text question search. Click a row to expand the full conversation. |

## Error classes

Error classes are assigned by `classify_submit_error()` in `loader.py`. The function inspects the **last `submit_sql_offline` tool message** in the record's message history.

| Class | Trigger |
|---|---|
| `Passed` | `execution_accuracy` is truthy |
| `No Submission` | No `submit_sql*` tool message found |
| `Target Error` | Message contains `[TARGET ERROR]` (the gold SQL failed — a dataset problem; checked first) |
| `Empty Query` | Message contains *"empty query"* |
| `Syntax Error` | Message matches `syntax error` |
| `Column/Relation Not Found` | Message matches `does not exist` |
| `Wrong SQL` | Message is *"Your SQL is not correct."* |
| `DB Error` | Other `DatabaseError` |
| `Other` | None of the above |

To add a new class, add a branch in `classify_submit_error()` before the final `return "Other"` line.

## Compare page

Navigate to **Run Comparison** in the Streamlit sidebar to compare multiple runs side by side.

| Panel | Description |
|---|---|
| **Sidebar** | Multi-select any number of runs (date / run). Requires ≥ 2. |
| **Stats** | Per-run metric cards: accuracy, avg tokens, avg cost, avg budget remaining. |
| **Reliability** | Per-run P̄, Aptitude, Unreliability, Reliability, and pass@N (multi-iteration runs only). |
| **Charts** | Accuracy by database, error distribution, and tool usage — one chart per run, aligned in columns. |
| **Task table** | One row per `instance_id`, one column per run showing pass count `c/n` (or — when absent). Filter by agreement, and search by question text. |
| **Conversation viewer** | Click a row to view conversations side by side; an iteration selector appears per pane for multi-iteration runs. With > 2 runs, two dropdowns let you choose which pair to compare. |

## Experiment tracking (`experiments.csv`)

`experiments.csv` at the repo root is a git-tracked index of every run under
`results/` (which is itself gitignored). One row per run dir, keyed by the path
relative to `results/`. Dedicated columns are `run_dir`, `date`, `time`, `status`,
`baseline`, `model`; everything else about the config is collapsed into a single
`args` flag-string (e.g. `--schema-type ddl --gt-db --gt-kb --num-iterations 9 …`)
so ablations diff as a single token. Metrics (`accuracy`, `avg_cost`, token counts,
`reliability`/`aptitude`/`unreliability`, …) are computed by `explorer/loader.py`,
so the CSV always agrees with the explorer app.

It is maintained by `explorer/index.py`:

- A `status=running` stub row is appended automatically when a run launches (the
  pipeline calls `append_stub` right after writing its `config.yaml` snapshot).
- Metrics and the final `status` (`done` / `partial` / `error`) are filled by
  `reconcile`, which runs after each eval and on demand via **`just index`**
  (`uv run python -m explorer.index reconcile`).
- The **`Notes`** column is yours to edit; `reconcile` never overwrites it and
  preserves rows it can no longer rediscover (e.g. still-running stubs).

