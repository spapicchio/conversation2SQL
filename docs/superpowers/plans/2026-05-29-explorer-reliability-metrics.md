# Explorer Reliability Metrics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add pass@k, average, aptitude, unreliability, and reliability metrics to the
Streamlit explorer, and teach it to load/group the pipeline's per-iteration result files
(`results_iter*.jsonl`).

**Architecture:** A new framework-free `explorer/metrics.py` holds the metric math
(pure functions over grouped records). `explorer/loader.py` discovers and groups the
per-iteration files and calls into `metrics.py`. The main page (`app.py`) and compare page
(`pages/compare.py`) render the new metrics; both degrade gracefully to today's behavior for
single-iteration runs.

**Tech Stack:** Python 3.12, numpy, pandas, Streamlit, Altair, pytest (run via `uv run`).

---

## Background facts for the implementer

- Per-sample score is **binary**: `1.0` if a record's `execution_accuracy` is truthy, else `0.0`.
- The pipeline writes one file per iteration: `results_iter{i}.jsonl`, and every record carries an `iteration` int field. Old runs have only `results.jsonl` / `results_smaller.jsonl` and no `iteration` field.
- **Import contexts differ:** Streamlit runs `explorer/app.py` with `explorer/` on `sys.path`, so modules import each other bare (`from loader import ...`, `from metrics import ...`). Pytest imports them as a package (`from explorer.loader import ...`, `from explorer.metrics import ...`). `loader.py` must therefore import `metrics` with a try/except fallback (shown in Task 2). New tests import via `explorer.*`.
- Run all Python/pytest through `uv run` (project rule in `.claude/CLAUDE.md`).
- Metric formulas (per instruction with sample scores `g`, n = sample count):
  - Average `P̄` = mean over instances of `mean(g)`.
  - Aptitude `A⁹⁰` = mean over instances of `numpy.percentile(g, 90)`.
  - Unreliability `U₁₀⁹⁰` = mean over instances of `percentile(g,90) − percentile(g,10)`.
  - Reliability `R = 1 − U` (0–1 scale).
  - pass@k (unbiased, Chen 2021): for an instance with n samples, c passes,
    `1 − ∏_{i=0}^{k-1}(n−c−i)/(n−i)` (= 1 when `n−c < k`), averaged over instances with `n ≥ k`.

---

## Task 1: `explorer/metrics.py` — metric math (pure functions)

**Files:**
- Create: `explorer/metrics.py`
- Test: `tests/explorer/test_metrics.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/explorer/test_metrics.py`:

```python
import pytest

from explorer.metrics import ReliabilityStats, passk_curve, reliability_metrics


def _samples(passes: int, total: int) -> list[dict]:
    """A group of `total` records, `passes` of which passed."""
    return [{"execution_accuracy": i < passes} for i in range(total)]


class TestPasskCurve:
    def test_all_pass(self):
        curve = passk_curve({"a": _samples(3, 3)})
        assert curve == {1: pytest.approx(1.0), 2: pytest.approx(1.0), 3: pytest.approx(1.0)}

    def test_known_values_n10_c3(self):
        curve = passk_curve({"a": _samples(3, 10)})
        assert curve[1] == pytest.approx(0.3)
        assert curve[5] == pytest.approx(0.9166666, abs=1e-6)
        assert curve[10] == pytest.approx(1.0)

    def test_empty(self):
        assert passk_curve({}) == {}

    def test_k_capped_by_smallest_group(self):
        # one instance has 2 samples, one has 3 → k=3 averages only the 3-sample instance
        curve = passk_curve({"a": _samples(0, 2), "b": _samples(3, 3)})
        assert set(curve) == {1, 2, 3}
        assert curve[3] == pytest.approx(1.0)  # only "b" qualifies for k=3


class TestReliabilityMetrics:
    def test_empty(self):
        rel = reliability_metrics({})
        assert rel == ReliabilityStats(0.0, 0.0, 0.0, 0.0, {})

    def test_all_pass_single_instance(self):
        rel = reliability_metrics({"a": _samples(3, 3)})
        assert rel.avg_performance == pytest.approx(1.0)
        assert rel.aptitude == pytest.approx(1.0)
        assert rel.unreliability == pytest.approx(0.0)
        assert rel.reliability == pytest.approx(1.0)

    def test_flaky_mix(self):
        # A: [1,1] all pass; B: [1,0] half pass
        rel = reliability_metrics({"a": _samples(2, 2), "b": _samples(1, 2)})
        assert rel.avg_performance == pytest.approx(0.75)        # (1.0 + 0.5)/2
        assert rel.aptitude == pytest.approx(0.95)               # (1.0 + 0.9)/2
        assert rel.unreliability == pytest.approx(0.4)           # (0.0 + 0.8)/2
        assert rel.reliability == pytest.approx(0.6)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/explorer/test_metrics.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'explorer.metrics'`.

- [ ] **Step 3: Write the implementation**

Create `explorer/metrics.py`:

```python
"""Reliability metrics for multi-iteration evaluation runs.

Each per-sample score is binary: 1.0 if the record's `execution_accuracy` is
truthy, else 0.0. For one instruction with sample scores g = [S_1 .. S_n]:

  - Average       P̄       = mean over instances of mean(g)
  - Aptitude      A^90     = mean over instances of percentile(g, 90)
  - Unreliability U_10^90  = mean over instances of [percentile(g,90) - percentile(g,10)]
  - Reliability   R        = 1 - U                 (scores are 0..1, not 0..100)
  - pass@k (Chen et al. 2021, unbiased): for an instance with n samples and c
    passes, 1 - C(n-c, k)/C(n, k) (= 1 when n-c < k), averaged across instances
    with n >= k. Returned as a sweep {k: value} for k = 1 .. max(n_i).

Percentiles use numpy's default (linear interpolation). NOTE: with binary 0/1
scores and small n, a per-instance 90th/10th percentile collapses to 0 or 1, so
Aptitude/Unreliability are coarse (near-degenerate) for small n. This is inherent
to applying the paper's graded-score formulas to a binary score.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ReliabilityStats:
    avg_performance: float        # P̄
    aptitude: float               # A^90
    unreliability: float          # U_10^90
    reliability: float            # 1 - U
    passk: dict[int, float]       # {k: pass@k}, k = 1..max_n


def _scores(group: list[dict]) -> list[float]:
    return [1.0 if r.get("execution_accuracy") else 0.0 for r in group]


def _passk_instance(n: int, c: int, k: int) -> float:
    """Unbiased pass@k for one instance: 1 - C(n-c, k)/C(n, k).

    Computed as 1 - prod_{i=0}^{k-1} (n-c-i)/(n-i) to avoid overflow.
    """
    if n - c < k:
        return 1.0
    prod = 1.0
    for i in range(k):
        prod *= (n - c - i) / (n - i)
    return 1.0 - prod


def passk_curve(groups: dict[str, list[dict]]) -> dict[int, float]:
    if not groups:
        return {}
    max_n = max(len(g) for g in groups.values())
    curve: dict[int, float] = {}
    for k in range(1, max_n + 1):
        vals: list[float] = []
        for g in groups.values():
            n = len(g)
            if n < k:
                continue
            c = int(sum(_scores(g)))
            vals.append(_passk_instance(n, c, k))
        if vals:
            curve[k] = sum(vals) / len(vals)
    return curve


def reliability_metrics(groups: dict[str, list[dict]]) -> ReliabilityStats:
    if not groups:
        return ReliabilityStats(0.0, 0.0, 0.0, 0.0, {})
    means: list[float] = []
    p90s: list[float] = []
    ranges: list[float] = []
    for g in groups.values():
        s = _scores(g)
        means.append(float(np.mean(s)))
        p90 = float(np.percentile(s, 90))
        p10 = float(np.percentile(s, 10))
        p90s.append(p90)
        ranges.append(p90 - p10)
    unreliability = sum(ranges) / len(ranges)
    return ReliabilityStats(
        avg_performance=sum(means) / len(means),
        aptitude=sum(p90s) / len(p90s),
        unreliability=unreliability,
        reliability=1.0 - unreliability,
        passk=passk_curve(groups),
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/explorer/test_metrics.py -v`
Expected: PASS (all tests).

- [ ] **Step 5: Commit**

```bash
git add explorer/metrics.py tests/explorer/test_metrics.py
git commit -m "feat(explorer): add reliability metrics module (pass@k, aptitude, unreliability)"
```

---

## Task 2: `explorer/loader.py` — discover & group per-iteration files

**Files:**
- Modify: `explorer/loader.py` (RunStats at :14, RunData at :27, `_compute_stats` at :68, `_has_results` at :116, `load_run` at :150)
- Test: `tests/explorer/test_loader.py` (add cases)

- [ ] **Step 1: Write the failing tests**

Append to `tests/explorer/test_loader.py` (after the `TestLoadRun` class):

```python
class TestLoadRunIterations:
    def _write_jsonl(self, path: Path, records: list[dict]) -> None:
        path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")

    def test_groups_across_iteration_files(self, tmp_path):
        self._write_jsonl(tmp_path / "results_iter0.jsonl", [
            make_record(instance_id="t1", execution_accuracy=True, iteration=0),
            make_record(instance_id="t2", execution_accuracy=False, iteration=0),
        ])
        self._write_jsonl(tmp_path / "results_iter1.jsonl", [
            make_record(instance_id="t1", execution_accuracy=False, iteration=1),
            make_record(instance_id="t2", execution_accuracy=False, iteration=1),
        ])
        run = load_run(tmp_path)
        assert run.n_iterations == 2
        assert len(run.records) == 4
        assert set(run.groups) == {"t1", "t2"}
        assert len(run.groups["t1"]) == 2
        # groups sorted by iteration
        assert [r["iteration"] for r in run.groups["t1"]] == [0, 1]

    def test_malformed_counted_across_iteration_files(self, tmp_path):
        (tmp_path / "results_iter0.jsonl").write_text(
            json.dumps(make_record(instance_id="t1")) + "\nNOT JSON\n", encoding="utf-8"
        )
        (tmp_path / "results_iter1.jsonl").write_text(
            json.dumps(make_record(instance_id="t1")) + "\n", encoding="utf-8"
        )
        run = load_run(tmp_path)
        assert run.malformed_count == 1
        assert len(run.records) == 2

    def test_old_single_file_is_one_iteration(self, tmp_path):
        self._write_jsonl(tmp_path / "results.jsonl", [make_record(instance_id="t1")])
        run = load_run(tmp_path)
        assert run.n_iterations == 1
        assert run.groups["t1"][0]["iteration"] == 0

    def test_reliability_stats_populated(self, tmp_path):
        self._write_jsonl(tmp_path / "results_iter0.jsonl", [
            make_record(instance_id="t1", execution_accuracy=True, iteration=0),
        ])
        self._write_jsonl(tmp_path / "results_iter1.jsonl", [
            make_record(instance_id="t1", execution_accuracy=True, iteration=1),
        ])
        run = load_run(tmp_path)
        assert run.stats.reliability.avg_performance == pytest.approx(1.0)


class TestHasResultsIterations:
    def test_list_runs_finds_iteration_files(self, tmp_path):
        d = tmp_path / "2026_05_29" / "10_00_00__slug"
        d.mkdir(parents=True)
        (d / "results_iter0.jsonl").write_text("{}\n", encoding="utf-8")
        assert list_runs(tmp_path) == {"2026_05_29": ["10_00_00__slug"]}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/explorer/test_loader.py::TestLoadRunIterations -v`
Expected: FAIL — `RunData` has no attribute `n_iterations` / `groups`.

- [ ] **Step 3: Update imports and dataclasses**

In `explorer/loader.py`, change the dataclass import (top of file, line 6) and add the metrics import:

```python
from dataclasses import dataclass, field
```

Add after the existing imports (after `import yaml` at line 10):

```python
try:  # bare import when Streamlit runs from explorer/; package import under pytest
    from metrics import ReliabilityStats, reliability_metrics
except ModuleNotFoundError:  # pragma: no cover - import-path shim
    from explorer.metrics import ReliabilityStats, reliability_metrics
```

Replace the `RunStats` dataclass (lines 13-24) by adding a `reliability` field with a default at the end:

```python
@dataclass
class RunStats:
    n_total: int
    n_passed: int
    avg_input_tokens: float
    avg_output_tokens: float
    avg_cost: float
    avg_budget_remaining: float
    accuracy_by_database: dict[str, float]  # database -> pass rate
    error_distribution: Counter[str]  # error_class -> count
    tool_usage: Counter[str]  # tool_name -> total calls
    reliability: ReliabilityStats | None = None
```

Replace the `RunData` dataclass (lines 26-32) to add `groups` and `n_iterations`:

```python
@dataclass
class RunData:
    records: list[dict]
    config: dict
    stats: RunStats
    malformed_count: int = 0
    source_file: str = "results.jsonl"
    groups: dict[str, list[dict]] = field(default_factory=dict)
    n_iterations: int = 0
```

- [ ] **Step 4: Update `_compute_stats` to accept and use groups**

Change the signature (line 68) and the return statement (lines 103-113):

```python
def _compute_stats(records: list[dict], groups: dict[str, list[dict]] | None = None) -> RunStats:
```

and the `return RunStats(...)` block becomes:

```python
    return RunStats(
        n_total=n_total,
        n_passed=n_passed,
        avg_input_tokens=avg_input,
        avg_output_tokens=avg_output,
        avg_cost=avg_cost,
        avg_budget_remaining=avg_budget,
        accuracy_by_database=accuracy_by_database,
        error_distribution=error_distribution,
        tool_usage=tool_usage,
        reliability=reliability_metrics(groups or {}),
    )
```

- [ ] **Step 5: Update `_has_results` to recognize iteration files**

Replace `_has_results` (lines 116-117):

```python
def _has_results(d: Path) -> bool:
    return (
        bool(list(d.glob("results_iter*.jsonl")))
        or (d / "results.jsonl").exists()
        or (d / "results_smaller.jsonl").exists()
    )
```

- [ ] **Step 6: Rewrite `load_run` to read/group iteration files**

Replace the entire `load_run` function (lines 150-188) with:

```python
def _read_jsonl(source: Path) -> tuple[list[dict], int]:
    records: list[dict] = []
    malformed = 0
    with source.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                malformed += 1
    return records, malformed


def load_run(path: Path) -> RunData:
    """Load records and config from results/<baseline>/<date>/<run>/.

    Prefers per-iteration files (results_iter*.jsonl). Falls back to the legacy
    single file (results_smaller.jsonl, then results.jsonl), treated as iteration 0.
    """
    iter_files = sorted(path.glob("results_iter*.jsonl"))
    records: list[dict] = []
    malformed = 0
    if iter_files:
        for f in iter_files:
            recs, m = _read_jsonl(f)
            records.extend(recs)
            malformed += m
        source_file = f"results_iter*.jsonl ({len(iter_files)} files)"
    else:
        smaller = path / "results_smaller.jsonl"
        full = path / "results.jsonl"
        source = smaller if smaller.exists() else full
        if source.exists():
            records, malformed = _read_jsonl(source)
        source_file = source.name

    for r in records:
        r["_error_class"] = classify_submit_error(r)
        r.setdefault("iteration", 0)

    groups: dict[str, list[dict]] = {}
    for r in records:
        groups.setdefault(r.get("instance_id", ""), []).append(r)
    for g in groups.values():
        g.sort(key=lambda r: r.get("iteration", 0))
    n_iterations = len({r.get("iteration", 0) for r in records}) if records else 0

    config: dict = {}
    config_path = path / "config.yaml"
    if config_path.exists():
        with config_path.open(encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

    return RunData(
        records=records,
        config=config,
        stats=_compute_stats(records, groups),
        malformed_count=malformed,
        source_file=source_file,
        groups=groups,
        n_iterations=n_iterations,
    )
```

- [ ] **Step 7: Run the loader tests**

Run: `uv run pytest tests/explorer/test_loader.py -v`
Expected: PASS — new `TestLoadRunIterations` / `TestHasResultsIterations` pass and all pre-existing `TestLoadRun` / `TestComputeStats` / `TestListRuns` still pass (the `reliability` and `groups`/`n_iterations` fields all have defaults, so existing constructions are unaffected).

- [ ] **Step 8: Commit**

```bash
git add explorer/loader.py tests/explorer/test_loader.py
git commit -m "feat(explorer): load and group per-iteration result files"
```

---

## Task 3: `explorer/loader.py` — `join_runs` emits `c/n` cells

**Files:**
- Modify: `explorer/loader.py` (`join_runs` at :191)
- Test: `tests/explorer/test_loader.py` (`TestJoinRuns` and its `_make_run_data` helper)

- [ ] **Step 1: Update the test helper and the two pass/fail assertions**

In `tests/explorer/test_loader.py`, replace `_make_run_data` (lines 198-215) so it populates `groups` from the records:

```python
def _make_run_data(records: list[dict]) -> RunData:
    from collections import Counter
    groups: dict[str, list[dict]] = {}
    for r in records:
        groups.setdefault(r.get("instance_id", ""), []).append(r)
    return RunData(
        records=records,
        config={},
        stats=RunStats(
            n_total=len(records),
            n_passed=sum(1 for r in records if r.get("execution_accuracy")),
            avg_input_tokens=0.0,
            avg_output_tokens=0.0,
            avg_cost=0.0,
            avg_budget_remaining=0.0,
            accuracy_by_database={},
            error_distribution=Counter(),
            tool_usage=Counter(),
        ),
        malformed_count=0,
        groups=groups,
        n_iterations=1,
    )
```

In `test_two_runs_same_task_pass_fail` (lines 219-226) change the cell assertions:

```python
        assert df.iloc[0]["run_a"] == "1/1"
        assert df.iloc[0]["run_b"] == "0/1"
```

In `test_task_absent_in_one_run_shows_dash` (lines 228-239) change:

```python
        assert t1["run_a"] == "1/1"
        assert t1["run_b"] == "—"
        assert t2["run_a"] == "—"
        assert t2["run_b"] == "0/1"
```

- [ ] **Step 2: Add a new test for multi-sample cells**

Add to `TestJoinRuns`:

```python
    def test_cell_shows_pass_count_over_samples(self):
        g = [
            make_record(instance_id="t1", execution_accuracy=True, selected_database="db1"),
            make_record(instance_id="t1", execution_accuracy=False, selected_database="db1"),
        ]
        runs = {"run_a": _make_run_data(g)}
        df = join_runs(runs)
        assert df.iloc[0]["run_a"] == "1/2"
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest tests/explorer/test_loader.py::TestJoinRuns -v`
Expected: FAIL — current `join_runs` still emits `✓`/`✗`, so the `1/1`, `0/1`, `1/2` assertions fail.

- [ ] **Step 4: Rewrite `join_runs`**

Replace the entire `join_runs` function (lines 191-228) with:

```python
def join_runs(runs: dict[str, RunData]) -> "pd.DataFrame":
    """Merge N RunData objects on instance_id into a comparison DataFrame.

    Columns: instance_id, database, Question, then one column per run label with
    values "c/n" (c passes out of n samples for that instance) or "—" (task
    absent in that run).
    """
    all_ids: dict[str, dict] = {}
    for run_data in runs.values():
        for r in run_data.records:
            iid = r.get("instance_id", "")
            if iid not in all_ids:
                q = r.get("amb_user_query") or r.get("not_ambiguos_query", "")
                all_ids[iid] = {
                    "instance_id": iid,
                    "database": r.get("selected_database", ""),
                    "Question": (q[:80] + "…") if len(q) > 80 else q,
                }

    rows = []
    for iid, base in all_ids.items():
        row = dict(base)
        for label, run_data in runs.items():
            group = run_data.groups.get(iid)
            if not group:
                row[label] = "—"
            else:
                c = sum(1 for r in group if r.get("execution_accuracy"))
                row[label] = f"{c}/{len(group)}"
        rows.append(row)

    return pd.DataFrame(rows)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/explorer/test_loader.py::TestJoinRuns -v`
Expected: PASS (all `TestJoinRuns` cases).

- [ ] **Step 6: Commit**

```bash
git add explorer/loader.py tests/explorer/test_loader.py
git commit -m "feat(explorer): join_runs reports c/n pass counts per instance"
```

---

## Task 4: `explorer/app.py` — reliability section, pass@k chart, per-instance table

**Files:**
- Modify: `explorer/app.py` (metrics row at :96-105; Tasks block at :149-208)

UI code cannot be unit-tested here; verify with `py_compile` (syntax) and a manual Streamlit run.

- [ ] **Step 1: Add the reliability section after the existing metric cards**

In `explorer/app.py`, immediately after the `c5.metric("Avg Budget Remaining", ...)` line (line 105) and before the `# Charts:` comment (line 107), insert:

```python

    # Reliability metrics (multi-iteration runs only)
    if run.n_iterations >= 2 and stats.reliability is not None:
        rel = stats.reliability
        st.subheader(f"Reliability ({run.n_iterations} iterations)")
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Average P̄", f"{rel.avg_performance * 100:.1f}%")
        r2.metric("Aptitude A⁹⁰", f"{rel.aptitude * 100:.1f}%")
        r3.metric("Unreliability U₁₀⁹⁰", f"{rel.unreliability * 100:.1f}%")
        r4.metric("Reliability R", f"{rel.reliability * 100:.1f}%")
        if rel.passk:
            import altair as alt
            passk_df = pd.DataFrame(
                [{"k": k, "pass@k": v} for k, v in sorted(rel.passk.items())]
            )
            st.altair_chart(
                alt.Chart(passk_df).mark_line(point=True).encode(
                    x=alt.X("k:Q", scale=alt.Scale(nice=False)),
                    y=alt.Y("pass@k:Q", scale=alt.Scale(domain=[0, 1])),
                ).properties(height=250, title="pass@k"),
                use_container_width=True,
            )
    elif run.n_iterations == 1:
        st.caption("Single-iteration run — reliability metrics not applicable.")
```

- [ ] **Step 2: Replace the Tasks block with a per-instance / per-record branch**

Replace everything from `st.subheader("Tasks")` (line 150) through the final
`render_conversation(selected_record)` (line 208) with:

```python
    st.subheader("Tasks")

    if run.n_iterations >= 2:
        # ── Per-instance view (multi-iteration) ──────────────────────────────
        f1, f2 = st.columns([1, 3])
        with f1:
            pass_filter = st.radio(
                "Pass / Fail", ["All", "All Passed", "All Failed"], horizontal=True
            )
        with f2:
            search = st.text_input("Search question", placeholder="substring…")

        instance_rows = []
        for iid, group in run.groups.items():
            n = len(group)
            c = sum(1 for r in group if r.get("execution_accuracy"))
            q = _question(group[0])
            instance_rows.append(
                {
                    "instance_id": iid,
                    "database": group[0].get("selected_database", ""),
                    "Question": (q[:80] + "…") if len(q) > 80 else q,
                    "pass_rate": f"{c}/{n}",
                    "_c": c,
                    "_n": n,
                    "avg_cost": sum(r.get("total_cost") or 0 for r in group) / n,
                }
            )

        if pass_filter == "All Passed":
            instance_rows = [r for r in instance_rows if r["_c"] == r["_n"]]
        elif pass_filter == "All Failed":
            instance_rows = [r for r in instance_rows if r["_c"] == 0]
        if search:
            instance_rows = [
                r for r in instance_rows if search.lower() in r["Question"].lower()
            ]

        if not instance_rows:
            st.info("No tasks match the current filters.")
            st.stop()

        df = pd.DataFrame(
            [
                {
                    "#": i + 1,
                    "instance_id": r["instance_id"],
                    "database": r["database"],
                    "Question": r["Question"],
                    "pass_rate": r["pass_rate"],
                    "avg_cost": r["avg_cost"],
                }
                for i, r in enumerate(instance_rows)
            ]
        )
        event = st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
        )
        selected_indices = event.selection.rows
        if not selected_indices:
            st.stop()

        selected_iid = instance_rows[selected_indices[0]]["instance_id"]
        group = run.groups[selected_iid]
        st.divider()
        iters = [r.get("iteration", 0) for r in group]
        chosen = st.selectbox("Iteration", iters, format_func=lambda i: f"iteration {i}")
        selected_record = next(r for r in group if r.get("iteration", 0) == chosen)
        render_conversation(selected_record)
    else:
        # ── Per-record view (single iteration) — unchanged behavior ──────────
        f1, f2, f3 = st.columns([1, 2, 3])
        with f1:
            pass_filter = st.radio("Pass / Fail", ["All", "Passed", "Failed"], horizontal=True)
        with f2:
            all_error_classes = sorted({r.get("_error_class", "Other") for r in run.records})
            error_filter = st.multiselect(
                "Error Class", all_error_classes, default=all_error_classes
            )
        with f3:
            search = st.text_input("Search question", placeholder="substring…")

        filtered = run.records
        if pass_filter == "Passed":
            filtered = [r for r in filtered if r.get("execution_accuracy")]
        elif pass_filter == "Failed":
            filtered = [r for r in filtered if not r.get("execution_accuracy")]
        if error_filter:
            filtered = [r for r in filtered if r.get("_error_class", "Other") in error_filter]
        if search:
            filtered = [r for r in filtered if search.lower() in _question(r).lower()]

        if not filtered:
            st.info("No tasks match the current filters.")
            st.stop()

        rows = []
        for i, r in enumerate(filtered):
            q = _question(r)
            rows.append(
                {
                    "#": i + 1,
                    "instance_id": r.get("instance_id", ""),
                    "database": r.get("selected_database", ""),
                    "Question": (q[:80] + "…") if len(q) > 80 else q,
                    "error_class": r.get("_error_class", ""),
                    "Accuracy": "✓" if r.get("execution_accuracy") else "✗",
                    "input_tokens": r.get("mean_prompt_tokens", 0),
                    "output_tokens": r.get("mean_completion_tokens", 0),
                    "cost": r.get("total_cost", 0.0),
                }
            )

        df = pd.DataFrame(rows)
        event = st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
        )

        selected_indices = event.selection.rows
        if not selected_indices:
            st.stop()

        selected_record = filtered[selected_indices[0]]
        st.divider()
        render_conversation(selected_record)
```

- [ ] **Step 3: Verify the file compiles**

Run: `uv run python -m py_compile explorer/app.py`
Expected: no output, exit 0.

- [ ] **Step 4: Manual smoke check**

Run: `uv run streamlit run explorer/app.py` (Ctrl-C after checking).
Expected: a single-iteration run (any existing one under `results/`) shows the
"Single-iteration run — reliability metrics not applicable." caption and the task table
behaves as before. (Multi-iteration verification happens once a `num_iterations > 1` run
exists; the loader tests already cover grouping.)

- [ ] **Step 5: Commit**

```bash
git add explorer/app.py
git commit -m "feat(explorer): reliability section, pass@k chart, per-instance task table"
```

---

## Task 5: `explorer/pages/compare.py` — per-run reliability + generalized filters

**Files:**
- Modify: `explorer/pages/compare.py` (stats column at :63-75; agreement filter at :149-165; conversation viewer at :204-215)

- [ ] **Step 1: Add per-run reliability metrics to the stats columns**

In `explorer/pages/compare.py`, after the `st.metric("Avg Budget Remaining", ...)` line
(line 75, inside the `for col, (label, run) in zip(...)` loop), insert:

```python
        if run.n_iterations >= 2 and run.stats.reliability is not None:
            rel = run.stats.reliability
            st.metric("Average P̄", f"{rel.avg_performance * 100:.1f}%")
            st.metric("Aptitude A⁹⁰", f"{rel.aptitude * 100:.1f}%")
            st.metric("Unreliability U₁₀⁹⁰", f"{rel.unreliability * 100:.1f}%")
            st.metric("Reliability R", f"{rel.reliability * 100:.1f}%")
            if rel.passk:
                st.metric("pass@N", f"{rel.passk[max(rel.passk)] * 100:.1f}%")
```

- [ ] **Step 2: Generalize the agreement filter to `c/n` cells**

Replace the `_is_pass` helper and the three filter branches (lines 149-165) with:

```python
def _cell_state(val: str) -> str | None:
    """Map a 'c/n' cell to 'pass' (c==n), 'fail' (c==0), 'partial', or None for '—'."""
    if val == "—":
        return None
    c_str, n_str = val.split("/")
    c, n = int(c_str), int(n_str)
    if c == n:
        return "pass"
    if c == 0:
        return "fail"
    return "partial"


if agreement_filter == "Disagreement only":
    def _disagrees(row: pd.Series) -> bool:
        states = [s for c in run_labels if (s := _cell_state(row[c])) is not None]
        return len(set(states)) > 1 if states else False
    task_df = task_df[task_df.apply(_disagrees, axis=1)]
elif agreement_filter == "All Passed":
    task_df = task_df[task_df[run_labels].apply(
        lambda row: all(_cell_state(v) == "pass" for v in row), axis=1)]
elif agreement_filter == "All Failed":
    task_df = task_df[task_df[run_labels].apply(
        lambda row: all(_cell_state(v) == "fail" for v in row), axis=1)]
```

- [ ] **Step 3: Add an iteration selectbox to each conversation pane**

Replace the conversation viewer loop (lines 204-215) with:

```python
conv_cols = st.columns(2)
for col, label in zip(conv_cols, [run_a_label, run_b_label]):
    with col:
        st.markdown(f"**{label}**")
        group = runs[label].groups.get(selected_instance_id, [])
        if not group:
            st.info("Task not found in this run.")
            continue
        if len(group) > 1:
            iters = [r.get("iteration", 0) for r in group]
            chosen = st.selectbox(
                "Iteration",
                iters,
                format_func=lambda i: f"iteration {i}",
                key=f"iter_{label}",
            )
            record = next(r for r in group if r.get("iteration", 0) == chosen)
        else:
            record = group[0]
        render_conversation(record)
```

- [ ] **Step 4: Verify the file compiles**

Run: `uv run python -m py_compile explorer/pages/compare.py`
Expected: no output, exit 0.

- [ ] **Step 5: Manual smoke check**

Run: `uv run streamlit run explorer/app.py`, navigate to "Run Comparison", select ≥2
existing (single-iteration) runs.
Expected: stats columns show accuracy as before (no reliability metrics, since
`n_iterations == 1`); the per-task table shows `1/1` / `0/1` / `—` cells; the
"All Passed" / "All Failed" / "Disagreement only" filters work.

- [ ] **Step 6: Commit**

```bash
git add explorer/pages/compare.py
git commit -m "feat(explorer): per-run reliability metrics and c/n filters on compare page"
```

---

## Task 6: Docs + full test sweep

**Files:**
- Modify: `explorer/README.md`

- [ ] **Step 1: Update the README feature tables**

In `explorer/README.md`, add a row to the main-page feature table (after the **Metrics**
row, line 32) and the compare-page table (after its **Stats** row, line 63):

Main-page table — add:

```markdown
| **Reliability** | For multi-iteration runs (`results_iter*.jsonl`): Average P̄, Aptitude A⁹⁰, Unreliability U₁₀⁹⁰, Reliability R, and a pass@k curve. Hidden for single-iteration runs. |
```

Compare-page table — add:

```markdown
| **Reliability** | Per-run P̄, Aptitude, Unreliability, Reliability, and pass@N (multi-iteration runs only). |
```

Also update the layout block (lines 15-23) to note the new file pattern — change the
`results.jsonl` line to:

```
        ├── results_iter*.jsonl    # one file per iteration (or legacy results.jsonl)
```

And add a sentence after the layout block:

```markdown
Runs with `num_iterations > 1` produce one `results_iter{i}.jsonl` per iteration; the
explorer groups them by `instance_id` to compute reliability metrics. Legacy single-file
runs (`results.jsonl` / `results_smaller.jsonl`) load as a single iteration.
```

- [ ] **Step 2: Run the full test suite**

Run: `uv run pytest tests/`
Expected: PASS — entire suite green (per `.claude/CLAUDE.md`).

- [ ] **Step 3: Commit**

```bash
git add explorer/README.md
git commit -m "docs(explorer): document reliability metrics and per-iteration loading"
```

---

## Self-review notes (for the implementer)

- **Spec coverage:** metrics module (Task 1) ✓; loader discovery/grouping/`n_iterations`/`groups`/`reliability` (Task 2) ✓; `join_runs` `c/n` (Task 3) ✓; main-page reliability section + pass@k chart + per-instance table + iteration selectbox (Task 4) ✓; compare-page per-run metrics + generalized filters + iteration selectbox (Task 5) ✓; tests (Tasks 1-3) and README (Task 6) ✓.
- **Type consistency:** `ReliabilityStats(avg_performance, aptitude, unreliability, reliability, passk)` is constructed identically in `metrics.py` and consumed by name in `app.py`/`compare.py`. `RunData.groups` / `RunData.n_iterations` / `RunStats.reliability` are read with those exact names everywhere.
- **Backward compatibility:** all new dataclass fields have defaults, so the existing `test_loader.py` constructions and any other `RunData`/`RunStats` callers keep working; `_compute_stats` keeps a one-arg call site valid via the `groups=None` default.
```
