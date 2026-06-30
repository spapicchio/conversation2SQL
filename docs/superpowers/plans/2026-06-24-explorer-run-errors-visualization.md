# Explorer Run-Error Visualization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Surface the crashed-sample records in each run's `results_error.jsonl` (count + error-type breakdown + drill-down to the raw message) across the explorer's single-run, anti-pattern, and compare pages.

**Architecture:** Parse and classify error records once in `explorer/loader.py`, expose them on `RunData.errors` / `RunStats.n_errors` / `RunStats.run_error_distribution`, then render a self-contained "Run errors" section on each page using the existing Altair-bar + `st.dataframe` idioms. Errors are purely diagnostic — no existing metric or chart changes.

**Tech Stack:** Python 3.12, Streamlit, Altair, pandas. Run everything through `uv run` (project venv). Tests via `uv run pytest`.

## Global Constraints

- Run Python/pytest only through `uv run` (e.g. `uv run pytest tests/explorer/test_loader.py -v`).
- **Purely diagnostic invariant:** `pass_at_1`, `n_instances`, `n_total`, the existing `error_distribution` (submit-SQL classes), and every existing chart MUST be computed exactly as today. Errors are additive only.
- New `RunStats` fields default to `0` / empty `Counter`; `RunData.errors` defaults to `[]`. A run dir with no `results_error.jsonl` must behave exactly as before.
- Name the new stats `n_errors` and `run_error_distribution` (do **not** reuse `error_distribution`, which means submit-SQL failures of completed records).
- Follow existing file patterns: error bar charts mirror the existing "Error Distribution" Altair bar; drill-down mirrors the existing `st.dataframe(..., on_select="rerun", selection_mode="single-row")` pattern.
- `tests/explorer/test_loader.py` already defines `make_record(...)` and `TestLoadRun._write_jsonl`; reuse them.

---

### Task 1: Error classifier + loader wiring (data layer)

**Files:**
- Modify: `explorer/loader.py` (add `classify_run_error`; extend `RunData`, `RunStats`, `_compute_stats`, `load_run`)
- Test: `tests/explorer/test_loader.py`

**Interfaces:**
- Consumes: existing `_read_jsonl(source: Path) -> tuple[list[dict], int]`, the dedup pattern in `load_run`.
- Produces:
  - `classify_run_error(error: str) -> str` returning one of `"Timeout"`, `"Context Window Exceeded"`, `"Internal Server Error"`, `"Bad Request"`, `"Patience State Error"`, `"Other"`.
  - `RunData.errors: list[dict]` — each error dict tagged with `_error_class`.
  - `RunStats.n_errors: int`, `RunStats.run_error_distribution: Counter[str]`.
  - `_compute_stats(records, groups=None, errors=())` — new optional `errors` param.

- [ ] **Step 1: Write the failing tests for the classifier**

Add to `tests/explorer/test_loader.py` (after the imports, add `classify_run_error` to the `from explorer.loader import (...)` block first):

```python
class TestClassifyRunError:
    def test_timeout(self):
        assert classify_run_error("litellm.Timeout: APITimeoutError - timed out") == "Timeout"

    def test_context_window(self):
        msg = "litellm.ContextWindowExceededError: ContextWindowExceededError - maximum context length is 64000"
        assert classify_run_error(msg) == "Context Window Exceeded"

    def test_internal_server_error(self):
        assert classify_run_error("litellm.InternalServerError: boom") == "Internal Server Error"

    def test_bad_request(self):
        assert classify_run_error("litellm.BadRequestError: bad params") == "Bad Request"

    def test_patience_state_error(self):
        assert classify_run_error("At key 'updated_user_patience': value is invalid") == "Patience State Error"

    def test_other_fallback(self):
        assert classify_run_error("some unrecognized failure") == "Other"

    def test_non_string_input(self):
        assert classify_run_error({"weird": 1}) == "Other"
```

- [ ] **Step 2: Run the classifier tests to verify they fail**

Run: `uv run pytest tests/explorer/test_loader.py::TestClassifyRunError -v`
Expected: FAIL — `ImportError: cannot import name 'classify_run_error'`.

- [ ] **Step 3: Implement `classify_run_error`**

In `explorer/loader.py`, add directly below `classify_submit_error` (around line 91):

```python
def classify_run_error(error: object) -> str:
    """Return a human-readable class for a results_error.jsonl 'error' string.

    These are mid-run crashes (the sample never produced a completed record),
    not submit_sql outcomes — see classify_submit_error for the latter. Check
    the more specific provider classes (context-window, bad-request) before the
    generic ones so a context-window error isn't mislabeled as a bad request.
    """
    text = error if isinstance(error, str) else str(error)
    if "ContextWindowExceededError" in text:
        return "Context Window Exceeded"
    if "BadRequestError" in text:
        return "Bad Request"
    if "InternalServerError" in text:
        return "Internal Server Error"
    if "Timeout" in text:
        return "Timeout"
    if "updated_user_patience" in text:
        return "Patience State Error"
    return "Other"
```

- [ ] **Step 4: Run the classifier tests to verify they pass**

Run: `uv run pytest tests/explorer/test_loader.py::TestClassifyRunError -v`
Expected: PASS (7 passed).

- [ ] **Step 5: Write the failing tests for loader error wiring**

Add to `tests/explorer/test_loader.py` inside `class TestLoadRun`:

```python
    def test_loads_errors(self, tmp_path):
        self._write_jsonl(tmp_path / "results_iter0.jsonl", [make_record(instance_id="t1")])
        self._write_jsonl(
            tmp_path / "results_error.jsonl",
            [
                {"instance_id": "e1", "iteration": 0, "error": "litellm.Timeout: x"},
                {"instance_id": "e2", "iteration": 0, "error": "litellm.BadRequestError: y"},
            ],
        )
        run = load_run(tmp_path)
        assert run.stats.n_errors == 2
        assert run.stats.run_error_distribution == Counter({"Timeout": 1, "Bad Request": 1})
        assert {e["instance_id"] for e in run.errors} == {"e1", "e2"}
        assert all("_error_class" in e for e in run.errors)
        # purely diagnostic: completed records / accuracy untouched
        assert len(run.records) == 1
        assert run.stats.n_instances == 1

    def test_no_error_file(self, tmp_path):
        self._write_jsonl(tmp_path / "results_iter0.jsonl", [make_record(instance_id="t1")])
        run = load_run(tmp_path)
        assert run.errors == []
        assert run.stats.n_errors == 0
        assert run.stats.run_error_distribution == Counter()

    def test_errors_deduped(self, tmp_path):
        self._write_jsonl(tmp_path / "results_iter0.jsonl", [make_record(instance_id="t1")])
        self._write_jsonl(
            tmp_path / "results_error.jsonl",
            [
                {"instance_id": "e1", "iteration": 0, "error": "litellm.Timeout: x"},
                {"instance_id": "e1", "iteration": 0, "error": "litellm.Timeout: x"},
            ],
        )
        run = load_run(tmp_path)
        assert run.stats.n_errors == 1
```

- [ ] **Step 6: Run the loader tests to verify they fail**

Run: `uv run pytest tests/explorer/test_loader.py::TestLoadRun::test_loads_errors tests/explorer/test_loader.py::TestLoadRun::test_no_error_file tests/explorer/test_loader.py::TestLoadRun::test_errors_deduped -v`
Expected: FAIL — `AttributeError` on `run.errors` / `run.stats.n_errors`.

- [ ] **Step 7: Add the dataclass fields**

In `explorer/loader.py`, add to `RunStats` (after `truncated_fraction`, line ~48):

```python
    n_errors: int = 0  # crashed samples from results_error.jsonl (not in records)
    run_error_distribution: Counter[str] = field(default_factory=Counter)  # error_class -> count
```

Add to `RunData` (after `duplicate_count`, line ~60):

```python
    errors: list[dict] = field(default_factory=list)  # results_error.jsonl, tagged with _error_class
```

- [ ] **Step 8: Thread errors through `_compute_stats`**

In `explorer/loader.py`, change the signature (line ~182):

```python
def _compute_stats(
    records: list[dict],
    groups: dict[str, list[dict]] | None = None,
    errors: list[dict] = (),
) -> RunStats:
```

Just before the `return RunStats(` block (after `clean = clean_fraction(g)`, line ~276), add:

```python
    n_errors = len(errors)
    run_error_distribution: Counter[str] = Counter(
        e.get("_error_class", "Other") for e in errors
    )
```

And add the two new keyword args to the `return RunStats(...)` call:

```python
        n_truncated=n_truncated,
        truncated_fraction=truncated_fraction,
        n_errors=n_errors,
        run_error_distribution=run_error_distribution,
    )
```

- [ ] **Step 9: Read + dedup the error file in `load_run`**

In `explorer/loader.py`, inside `load_run`, immediately before the `config: dict = {}` block (line ~412), add:

```python
    errors: list[dict] = []
    error_file = path / "results_error.jsonl"
    if error_file.exists():
        raw_errors, _ = _read_jsonl(error_file)
        seen_err: set[tuple[object, object]] = set()
        for e in raw_errors:
            key = (e.get("instance_id"), e.get("iteration", 0))
            if e.get("instance_id") and key in seen_err:
                continue
            if e.get("instance_id"):
                seen_err.add(key)
            e["_error_class"] = classify_run_error(e.get("error", ""))
            errors.append(e)
```

Then update the stats call and the `RunData(...)` return:

```python
        stats=_compute_stats(records, groups, errors),
```

```python
        n_iterations=n_iterations,
        duplicate_count=duplicate_count,
        errors=errors,
    )
```

- [ ] **Step 10: Run the loader tests to verify they pass**

Run: `uv run pytest tests/explorer/test_loader.py -v`
Expected: PASS (all existing + new tests).

- [ ] **Step 11: Commit**

```bash
git add explorer/loader.py tests/explorer/test_loader.py
git commit -m "feat(explorer): parse results_error.jsonl in loader

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: "Run errors" section on the single-run app

**Files:**
- Modify: `explorer/app.py`

**Interfaces:**
- Consumes: `RunData.errors`, `RunStats.n_errors`, `RunStats.run_error_distribution` from Task 1.
- Produces: nothing consumed by later tasks (UI only).

This is a Streamlit UI task; it has no unit test (the existing pages have none — see `tests/explorer/` which tests `loader`, `charts`, `patterns`, `metrics`, `render`, `index`, not the page scripts). Verification is by import + manual render.

- [ ] **Step 1: Add the "Errors" metric card**

In `explorer/app.py`, the metric row currently is `c1, c2, c3, c4, c5, c6 = st.columns(6)` (line ~99). Change to 7 columns:

```python
    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
```

After the `c6.metric("Truncated", ...)` block (ends ~line 144), add:

```python
    c7.metric(
        "Errors",
        f"{stats.n_errors}",
        help=(
            "Samples that crashed mid-run (from results_error.jsonl) and never "
            "produced a completed record. Excluded from accuracy and every other "
            "metric — shown here for diagnosis only."
        ),
    )
```

- [ ] **Step 2: Add the "Run errors" section**

In `explorer/app.py`, immediately before `st.divider()` that precedes `st.subheader("Tasks")` (line ~247), add:

```python
    if stats.n_errors:
        st.subheader("Run errors")
        st.caption(
            f"{stats.n_errors} sample(s) crashed mid-run (results_error.jsonl) — "
            "diagnostic only, not part of accuracy."
        )
        run_err_df = pd.DataFrame(
            [{"Error Class": k, "Count": v}
             for k, v in stats.run_error_distribution.items()]
        ).sort_values("Error Class")
        _rerr_base = alt.Chart(run_err_df).mark_bar().encode(
            x=alt.X("Error Class:N", sort=None),
            y=alt.Y("Count:Q"),
        )
        st.altair_chart(
            (_rerr_base + _rerr_base.mark_text(
                align="center", baseline="bottom", dy=-2
            ).encode(text=alt.Text("Count:Q"))).properties(height=300),
            use_container_width=True,
        )

        err_rows = [
            {
                "#": i + 1,
                "instance_id": e.get("instance_id", ""),
                "iteration": e.get("iteration", 0),
                "error_class": e.get("_error_class", "Other"),
            }
            for i, e in enumerate(run.errors)
        ]
        err_event = st.dataframe(
            pd.DataFrame(err_rows),
            use_container_width=True,
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
            key="run_errors_table",
        )
        err_sel = err_event.selection.rows
        if err_sel:
            st.code(str(run.errors[err_sel[0]].get("error", "")))
```

- [ ] **Step 3: Verify the module imports cleanly**

Run: `uv run python -c "import ast; ast.parse(open('explorer/app.py').read()); print('ok')"`
Expected: `ok`.

- [ ] **Step 4: Run the full test suite (no regressions)**

Run: `uv run pytest tests/explorer -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add explorer/app.py
git commit -m "feat(explorer): show run errors on single-run page

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: "Run errors" section on the anti-pattern page

**Files:**
- Modify: `explorer/pages/patterns.py`

**Interfaces:**
- Consumes: `RunData.errors`, `RunStats.n_errors`, `RunStats.run_error_distribution` from Task 1 (via `run`, which is the unscoped `RunData`).
- Produces: UI only.

Note: this page scopes most content to `records` (filtered by `acc_scope`). Errors have no execution accuracy, so this section reads `run.errors` / `run.stats`, **not** the scoped `records`.

- [ ] **Step 1: Add the "Run errors" section at the end of the page**

In `explorer/pages/patterns.py`, append at the very end of the file (after the final `render_conversation(record, pattern_hits=record["_pattern_hits"])`, line ~468):

```python

# ── Run errors (crashed samples; independent of the accuracy scope) ──────────────
if run.stats.n_errors:
    st.divider()
    st.subheader("Run errors")
    st.caption(
        f"{run.stats.n_errors} sample(s) crashed mid-run (results_error.jsonl). "
        "These have no tool trace or accuracy, so they are independent of the "
        "execution-accuracy scope above and of the anti-pattern analysis."
    )
    rerr_df = pd.DataFrame(
        [{"Error Class": k, "Count": v}
         for k, v in run.stats.run_error_distribution.items()]
    ).sort_values("Error Class")
    _rerr_base = alt.Chart(rerr_df).mark_bar().encode(
        x=alt.X("Error Class:N", sort=None),
        y=alt.Y("Count:Q"),
    )
    st.altair_chart(
        (_rerr_base + _rerr_base.mark_text(
            align="center", baseline="bottom", dy=-2
        ).encode(text=alt.Text("Count:Q"))).properties(height=300),
        use_container_width=True,
    )
    err_rows = [
        {
            "#": i + 1,
            "instance_id": e.get("instance_id", ""),
            "iteration": e.get("iteration", 0),
            "error_class": e.get("_error_class", "Other"),
        }
        for i, e in enumerate(run.errors)
    ]
    err_event = st.dataframe(
        pd.DataFrame(err_rows),
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key="patterns_run_errors_table",
    )
    err_sel = err_event.selection.rows
    if err_sel:
        st.code(str(run.errors[err_sel[0]].get("error", "")))
```

- [ ] **Step 2: Verify the module parses cleanly**

Run: `uv run python -c "import ast; ast.parse(open('explorer/pages/patterns.py').read()); print('ok')"`
Expected: `ok`.

- [ ] **Step 3: Run the full explorer test suite (no regressions)**

Run: `uv run pytest tests/explorer -q`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add explorer/pages/patterns.py
git commit -m "feat(explorer): show run errors on anti-pattern page

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: Errors on the compare page

**Files:**
- Modify: `explorer/pages/compare.py`

**Interfaces:**
- Consumes: `RunData.errors`, `RunStats.n_errors`, `RunStats.run_error_distribution` from Task 1.
- Produces: UI only.

- [ ] **Step 1: Add "Errored instances" to the operational table**

In `explorer/pages/compare.py`, add to `_OPERATIONAL_ROWS` (after the `"Truncated"` row, line ~117). Lower is better → `False`:

```python
    ("Errored instances", lambda v: f"{int(v)}", False),
```

Add the matching entry to the dict returned by `_operational_value` (after `"Truncated": stats.truncated_fraction,`, line ~133):

```python
        "Errored instances": stats.n_errors,
```

- [ ] **Step 2: Add the per-run "Run errors" chart row**

In `explorer/pages/compare.py`, after the error-distribution `err_cols` loop (ends ~line 250, before the `has_tools = ...` tool-usage block), add:

```python
# Run-level crash errors (results_error.jsonl), one column per run. Shown only
# when at least one selected run has crashes; runs without get an info note for
# column alignment.
if any(run.stats.n_errors for run in runs.values()):
    max_run_err = max(
        (sum(run.stats.run_error_distribution.values()) for run in runs.values()),
        default=1,
    )
    run_err_cols = st.columns(len(runs))
    for col, (_, run) in zip(run_err_cols, runs.items()):
        with col:
            st.subheader("Run errors")
            if run.stats.n_errors:
                rerr_df = pd.DataFrame(
                    [{"Error Class": k, "Count": v}
                     for k, v in run.stats.run_error_distribution.items()]
                ).sort_values("Error Class")
                _rerr_base = alt.Chart(rerr_df).mark_bar().encode(
                    x=alt.X("Error Class:N", sort=None),
                    y=alt.Y("Count:Q", scale=alt.Scale(domain=[0, max_run_err])),
                )
                st.altair_chart(
                    (_rerr_base + _rerr_base.mark_text(
                        align="center", baseline="bottom", dy=-2
                    ).encode(text=alt.Text("Count:Q"))).properties(height=300),
                    use_container_width=True,
                )
                with st.expander(f"List ({run.stats.n_errors})"):
                    st.dataframe(
                        pd.DataFrame(
                            [{"instance_id": e.get("instance_id", ""),
                              "iteration": e.get("iteration", 0),
                              "error_class": e.get("_error_class", "Other")}
                             for e in run.errors]
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )
            else:
                st.info("No run errors")
```

- [ ] **Step 3: Verify the module parses cleanly**

Run: `uv run python -c "import ast; ast.parse(open('explorer/pages/compare.py').read()); print('ok')"`
Expected: `ok`.

- [ ] **Step 4: Run the full explorer test suite (no regressions)**

Run: `uv run pytest tests/explorer -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add explorer/pages/compare.py
git commit -m "feat(explorer): show run errors on compare page

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: Full-suite verification

**Files:** none (verification only)

- [ ] **Step 1: Run the whole test suite**

Run: `uv run pytest tests/`
Expected: PASS (per CLAUDE.md "Apply changes" rule).

- [ ] **Step 2: Smoke-check the loader against a real run with errors**

Run:
```bash
uv run python -c "
from pathlib import Path
from explorer.loader import load_run
r = load_run(Path('results/2026-06-23/08-28-51/tools_only__Qwen3.5-9B__ddl__lin__iter9__psql__strictpsql__double_ctx_bdg_psql_strict__error'))
print('n_errors', r.stats.n_errors)
print('dist', r.stats.run_error_distribution)
print('pass@1 unchanged:', r.stats.pass_at_1, 'n_instances', r.stats.n_instances)
"
```
Expected: `n_errors` > 0, a populated distribution, and a sane `pass_at_1`.

- [ ] **Step 3: Final commit if anything was adjusted**

(Only if Step 1/2 surfaced a fix.)

```bash
git add -A
git commit -m "test(explorer): verify run-error visualization end to end

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage:**
- Classifier (6 classes) → Task 1, Steps 1-4. ✔
- `load_run` reads + dedups error file, tags `_error_class`, `RunData.errors` → Task 1, Steps 7-9. ✔
- `RunStats.n_errors` + `run_error_distribution`, accuracy untouched → Task 1, Steps 7-8 + invariant test in Step 5. ✔
- Single-app: metric card + section (chart + drill-down + raw message) → Task 2. ✔
- Anti-pattern: standalone section, independent of accuracy scope → Task 3. ✔
- Compare: operational row + per-run charts + per-run expander → Task 4. ✔
- Tests: classifier per-class + Other + non-string; load_run with/without error file + dedup → Task 1 Steps 1, 5. ✔
- Full-suite run per CLAUDE.md → Task 5. ✔

**Placeholder scan:** No TBD/TODO; every code step shows full code. ✔

**Type consistency:** `classify_run_error`, `RunData.errors`, `RunStats.n_errors`, `RunStats.run_error_distribution`, and `_compute_stats(records, groups, errors)` are named identically across Task 1 and the consuming Tasks 2-4. ✔
