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
        ├── results.jsonl          # the result file
        └── config.yaml            # optional run config
```

The run folder name is generated automatically by the pipeline and encodes the key parameters that distinguish runs: model name, schema type (`ddl` or `toon`), and optional flags (`lin` = KB linearized, `gt-db` = GT tables only, `gt-kb` = GT KB only).

## Features

| Panel | Description |
|---|---|
| **Sidebar** | Cascading selectors (date → run). Shows model name and generation params (collapsible). |
| **Metrics** | Accuracy, avg input/output tokens, avg cost, avg budget remaining. |
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
| `Empty Query` | Message contains *"empty query"* |
| `Syntax Error` | Message matches `syntax error` |
| `Column/Relation Not Found` | Message matches `does not exist` |
| `Wrong SQL` | Message is *"Your SQL is not correct."* |
| `DB Error` | Other `DatabaseError` |
| `[TARGET ERROR]` | Target error executions |
| `Other` | None of the above |

To add a new class, add a branch in `classify_submit_error()` before the final `return "Other"` line.

## Compare page

Navigate to **Run Comparison** in the Streamlit sidebar to compare multiple runs side by side.

| Panel | Description |
|---|---|
| **Sidebar** | Multi-select any number of runs (date / run). Requires ≥ 2. |
| **Stats** | Per-run metric cards: accuracy, avg tokens, avg cost, avg budget remaining. |
| **Charts** | Accuracy by database, error distribution, and tool usage — one chart per run, aligned in columns. |
| **Task table** | One row per `instance_id`, one column per run (✓ / ✗ / —). Filter by agreement, and search by question text. |
| **Conversation viewer** | Click a row to view conversations side by side. With > 2 runs, two dropdowns let you choose which pair to compare. |
