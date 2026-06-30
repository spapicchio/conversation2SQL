# Conversation-Length Boxplot Pass/Fail Split — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split the "Conversation length" boxplot in both `app.py` and `pages/compare.py` into Passed (green) and Failed (red) boxes so the distribution of model calls, tool calls, and budget spent can be compared between correct and incorrect conversations.

**Architecture:** Add `kind="outcome"` to `stable_color` (green/red based on ✓/✗ or "passed"/"failed" in the label), a `conversation_length_split` helper in `loader.py`, a `marker_kind` parameter to `conversation_length_box`, then wire both pages to use the split. No new files; four existing files modified.

**Tech Stack:** Python 3.12, Plotly (`go.Box`), Streamlit, `uv run pytest` for tests.

---

## File Map

| File | Change |
|---|---|
| `explorer/colors.py` | Add `kind="outcome"` branch to `stable_color` |
| `explorer/loader.py` | Add `conversation_length_split` helper |
| `explorer/charts.py` | Add `marker_kind: str = "run"` param to `conversation_length_box` |
| `explorer/app.py` | Replace flat `lengths` list with `conversation_length_split` |
| `explorer/pages/compare.py` | Replace flat `length_series` dict with per-run splits |
| `tests/explorer/test_charts.py` | Tests for outcome coloring + new `marker_kind` param |
| `tests/explorer/test_loader.py` | Tests for `conversation_length_split` |

---

### Task 1: Add `kind="outcome"` to `stable_color` and test it

**Files:**
- Modify: `explorer/colors.py`
- Test: `tests/explorer/test_charts.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/explorer/test_charts.py`:

```python
from explorer.colors import stable_color

def test_outcome_color_passed_label():
    assert stable_color("Passed", kind="outcome") == "#2ca02c"

def test_outcome_color_checkmark_label():
    # compare.py uses "runA ✓" — the ✓ suffix triggers green
    assert stable_color("runA ✓", kind="outcome") == "#2ca02c"

def test_outcome_color_failed_label():
    assert stable_color("Failed", kind="outcome") == "#d62728"

def test_outcome_color_cross_label():
    assert stable_color("runA ✗", kind="outcome") == "#d62728"

def test_outcome_color_unknown_falls_back_to_hash():
    # Any label that isn't pass/fail should still return *something* (not raise).
    color = stable_color("some other label", kind="outcome")
    assert color.startswith("#") and len(color) == 7
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /workspaces/conversation2SQL
uv run pytest tests/explorer/test_charts.py::test_outcome_color_passed_label -v
```

Expected: `FAILED` — `AssertionError` (wrong color returned).

- [ ] **Step 3: Implement `kind="outcome"` in `stable_color`**

In `explorer/colors.py`, replace the body of `stable_color` so the `kind == "outcome"` branch comes before the existing `pinned` lookup:

```python
def stable_color(value: object, kind: str = "") -> str:
    """Return the fixed hex color for a category value.

    ``kind`` selects a pinned lookup (``"tool"``/``"pattern"``); ``"iteration"``
    maps by the iteration's integer index; ``"outcome"`` returns green for
    passed/✓ labels and red for failed/✗ labels; anything unknown (e.g.
    ``"run"``) hashes the string. Unpinned values within a pinned kind also
    hash, so the color is always defined.
    """
    if kind == "iteration":
        idx = _iteration_index(value)
        if idx is not None:
            return _PALETTE[idx % len(_PALETTE)]
        return _hashed_color(str(value))
    if kind == "outcome":
        s = str(value)
        if "✓" in s or "passed" in s.lower():
            return "#2ca02c"  # green
        if "✗" in s or "failed" in s.lower():
            return "#d62728"  # red
        return _hashed_color(s)
    pinned = _PINNED.get(kind)
    if pinned and str(value) in pinned:
        return pinned[str(value)]
    return _hashed_color(str(value))
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/explorer/test_charts.py -v
```

Expected: all `test_outcome_color_*` tests PASS; existing tests still PASS.

- [ ] **Step 5: Commit**

```bash
git add explorer/colors.py tests/explorer/test_charts.py
git commit -m "feat: add kind=outcome to stable_color (green=passed, red=failed)"
```

---

### Task 2: Add `conversation_length_split` to `loader.py` and test it

**Files:**
- Modify: `explorer/loader.py`
- Test: `tests/explorer/test_loader.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/explorer/test_loader.py` (after existing imports, use the existing `make_record` helper):

```python
from explorer.loader import conversation_length_split

class TestConversationLengthSplit:
    def test_splits_by_execution_accuracy(self):
        records = [
            make_record(execution_accuracy=True,  num_model_calls=5),
            make_record(execution_accuracy=False, num_model_calls=2),
            make_record(execution_accuracy=True,  num_model_calls=8),
        ]
        result = conversation_length_split(records, "Model calls")
        assert result["Passed"] == [5.0, 8.0]
        assert result["Failed"] == [2.0]

    def test_custom_label_uses_checkmark_suffix(self):
        records = [
            make_record(execution_accuracy=True,  num_model_calls=3),
            make_record(execution_accuracy=False, num_model_calls=1),
        ]
        result = conversation_length_split(records, "Model calls", label="runA")
        assert "runA ✓" in result
        assert "runA ✗" in result
        assert result["runA ✓"] == [3.0]
        assert result["runA ✗"] == [1.0]

    def test_drops_none_values(self):
        # Budget spent returns None when no budget info exists.
        records = [
            make_record(execution_accuracy=True),   # no initial_user_patience → None
            make_record(execution_accuracy=False),
        ]
        result = conversation_length_split(records, "Budget spent")
        assert result["Passed"] == []
        assert result["Failed"] == []

    def test_all_passed(self):
        records = [make_record(execution_accuracy=True, num_model_calls=4)]
        result = conversation_length_split(records, "Model calls")
        assert result["Passed"] == [4.0]
        assert result["Failed"] == []

    def test_all_failed(self):
        records = [make_record(execution_accuracy=False, num_model_calls=1)]
        result = conversation_length_split(records, "Model calls")
        assert result["Passed"] == []
        assert result["Failed"] == [1.0]
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/explorer/test_loader.py::TestConversationLengthSplit -v
```

Expected: `ImportError` — `cannot import name 'conversation_length_split'`.

- [ ] **Step 3: Implement `conversation_length_split` in `loader.py`**

Add after `conversation_length` (around line 157), before `_compute_stats`:

```python
def conversation_length_split(
    records: list[dict], metric: str, label: str = ""
) -> dict[str, list[float]]:
    """Partition conversation-length values by execution accuracy.

    Returns a dict with two keys. Without a label: ``"Passed"`` and
    ``"Failed"``. With a label: ``"{label} ✓"`` and ``"{label} ✗"``.
    Records where :func:`conversation_length` returns ``None`` (e.g.
    no-tool baselines for "Budget spent") are dropped from both lists.
    """
    pass_key = f"{label} ✓" if label else "Passed"
    fail_key = f"{label} ✗" if label else "Failed"
    passed: list[float] = []
    failed: list[float] = []
    for r in records:
        v = conversation_length(r, metric)
        if v is None:
            continue
        (passed if r.get("execution_accuracy") else failed).append(v)
    return {pass_key: passed, fail_key: failed}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/explorer/test_loader.py -v
```

Expected: all tests PASS (new + existing).

- [ ] **Step 5: Commit**

```bash
git add explorer/loader.py tests/explorer/test_loader.py
git commit -m "feat: add conversation_length_split helper to loader"
```

---

### Task 3: Add `marker_kind` parameter to `conversation_length_box`

**Files:**
- Modify: `explorer/charts.py`
- Test: `tests/explorer/test_charts.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/explorer/test_charts.py`:

```python
def test_conversation_length_box_outcome_kind_uses_green_for_passed():
    fig = conversation_length_box(
        {"Passed": [3.0, 5.0], "Failed": [1.0, 2.0]},
        "Model calls",
        marker_kind="outcome",
    )
    assert len(fig.data) == 2
    colors = {t.name: t.marker.color for t in fig.data}
    assert colors["Passed"] == "#2ca02c"
    assert colors["Failed"] == "#d62728"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/explorer/test_charts.py::test_conversation_length_box_outcome_kind_uses_green_for_passed -v
```

Expected: `FAILED` — `TypeError: conversation_length_box() got an unexpected keyword argument 'marker_kind'`.

- [ ] **Step 3: Add `marker_kind` param to `conversation_length_box` in `charts.py`**

Replace the existing function signature and body:

```python
def conversation_length_box(
    series: dict[str, list[float]], metric_label: str, marker_kind: str = "run"
) -> go.Figure:
    """Box plot of conversation lengths, one box per run.

    Unlike :func:`aptitude_unreliability_box` (which is fed precomputed
    percentiles because its scores are binary), this is handed the *raw*
    per-conversation values so Plotly computes standard Tukey quartiles and
    whiskers — the right summary for continuous counts. Empty series (e.g.
    budget-spent on a no-tool baseline) are skipped so no run draws a bare axis.

    ``marker_kind`` is forwarded to :func:`~colors.stable_color`; pass
    ``"outcome"`` when the series keys are "Passed"/"Failed" (or "{label} ✓"/
    "✗") so the boxes get semantic green/red colours.
    """
    fig = go.Figure()
    for label, values in series.items():
        if not values:
            continue
        fig.add_trace(
            go.Box(
                name=label,
                y=list(values),
                marker_color=stable_color(label, kind=marker_kind),
                boxmean=True,
                width=0.5,
            )
        )
    fig.update_layout(height=360, showlegend=False, margin=dict(t=10, b=10, l=10, r=10))
    fig.update_yaxes(title_text=metric_label, rangemode="tozero")
    return fig
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/explorer/test_charts.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add explorer/charts.py tests/explorer/test_charts.py
git commit -m "feat: add marker_kind param to conversation_length_box"
```

---

### Task 4: Wire up split in `app.py` and `compare.py`

**Files:**
- Modify: `explorer/app.py`
- Modify: `explorer/pages/compare.py`

No new tests needed — the unit behaviour is already covered by Tasks 1–3; page-level wiring is tested by running the app.

- [ ] **Step 1: Update `app.py`**

Update the import at the top of `app.py` — add `conversation_length_split`:

```python
from loader import conversation_length_split
```

Replace the conversation-length section (lines ~239–249) in `app.py`:

```python
    length_series = conversation_length_split(run.records, length_metric)
    if any(length_series.values()):
        st.plotly_chart(
            conversation_length_box(length_series, length_metric, marker_kind="outcome"),
            use_container_width=True,
        )
    else:
        st.caption(f"No "{length_metric}" data for this run.")
```

(Remove the old `lengths = [...]` list comprehension and the `model_label` variable if it is no longer used elsewhere; `model_label` is still used for the reliability chart above, so keep that assignment.)

- [ ] **Step 2: Update `compare.py`**

Update the import at the top of `compare.py` — add `conversation_length_split`:

```python
from loader import conversation_length_split
```

Replace the conversation-length section (lines ~192–205) in `compare.py`:

```python
length_series: dict[str, list[float]] = {}
for label, run in runs.items():
    split = conversation_length_split(
        run.records, length_metric, label=_display_label[label]
    )
    length_series.update(split)
if any(length_series.values()):
    st.plotly_chart(
        conversation_length_box(length_series, length_metric, marker_kind="outcome"),
        use_container_width=True,
    )
else:
    st.caption(f"No "{length_metric}" data in the selected runs.")
```

- [ ] **Step 3: Run full test suite**

```bash
uv run pytest tests/
```

Expected: all tests PASS.

- [ ] **Step 4: Commit**

```bash
git add explorer/app.py explorer/pages/compare.py
git commit -m "feat: split conversation-length boxplot by pass/fail outcome"
```