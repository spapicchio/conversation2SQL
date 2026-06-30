# Conversation-length comparison boxplot

## Goal

Visualize how "long" conversations are and compare that distribution across
evaluation runs, with a selector to switch between three length metrics:
**Model calls / Tool calls / Budget spent**. One boxplot at a time (the radio
switches the metric), one box per run.

## Components

### 1. Length extractor — `explorer/loader.py`

A metric registry and a pure extractor, placed next to `_remaining_budget`
(which the budget metric reuses):

```python
LENGTH_METRICS = ("Model calls", "Tool calls", "Budget spent")

def conversation_length(record: dict, metric: str) -> float | None:
    # "Model calls"  -> record.get("num_model_calls") or 0
    # "Tool calls"   -> len(record.get("tool_calls_in_order", []))
    # "Budget spent" -> initial_user_patience - _remaining_budget(record),
    #                   or None when budget info is absent (no-tool baseline)
```

- Model/tool counts are always defined; a no-tool baseline simply yields
  small/zero counts.
- Budget spent returns `None` when `_remaining_budget(record)` is `None` or
  `initial_user_patience` is missing. Callers drop `None`s.

### 2. Chart helper — `explorer/charts.py`

```python
def conversation_length_box(series: dict[str, list[float]], metric_label: str) -> go.Figure
```

One `go.Box` per key (run label), built from **raw values** so Plotly computes
standard Tukey quartiles/whiskers. This deliberately differs from
`aptitude_unreliability_box`, which feeds *precomputed* percentiles because its
data is binary — here the data is continuous counts, so raw boxes are correct.
Colors come from `stable_color(label, kind="run")`. Empty series (all `None`,
e.g. budget on a no-tool baseline) are skipped.

## Wiring

### Single-run page (`app.py`)

New section after the existing charts: a `st.radio` over `LENGTH_METRICS` plus
one boxplot with a single box (`{model_label: [lengths…]}`) — the distribution
across that run's conversations.

### Compare page (`pages/compare.py`)

Same radio + boxplot, one box per selected run, using `disambiguate_run_labels`
for short labels (consistent with the reliability box). Placed near the other
comparison charts.

## Testing (TDD)

- `tests/explorer/test_loader.py`: `conversation_length` for each metric,
  including the budget-`None` (no-tool) case and the budget = initial − remaining
  arithmetic.
- `tests/explorer/test_charts.py`: `conversation_length_box` returns a
  `go.Figure` with one trace per non-empty series and skips empty ones.

## Out of scope

Item 5 in `need_changes.md` (length-bucket vs pass/fail correlation) is a
different chart and is not included here.
