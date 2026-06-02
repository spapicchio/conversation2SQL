# Experiments Index Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a git-tracked `experiments.csv` at the repo root that indexes every evaluation run (config as a single `args` flag-string + metrics from the explorer), is appended to at launch, reconciled on demand, and preserves hand-typed Notes.

**Architecture:** A new `explorer/index.py` module reuses `explorer.loader.load_run()` for metrics (single source of truth). `reconcile()` rescans `results/` and rebuilds the CSV while carrying over the `Notes` column (and any rows for still-running/archived runs) by `run_dir`. `append_stub()` adds a `status=running` config-only row at launch, called from the pipeline right after it writes the `config.yaml` snapshot. A `just index` recipe runs reconcile; `run_suite` runs a non-fatal reconcile after each run.

**Tech Stack:** Python 3.12 (stdlib `csv`, `argparse`, `pathlib`, `os.replace` for atomic writes), pytest, just, bash.

---

## File structure

- **Create:** `explorer/index.py` — the index module (render_args, status, row builders, append_stub, reconcile, `__main__` CLI). One responsibility: maintain `experiments.csv`. Depends on `explorer.loader`.
- **Create:** `tests/explorer/test_index.py` — unit tests for the module.
- **Modify:** `src/conversation2sql/eval_framework/main_pipe_workflow.py` — call `append_stub` after the config snapshot is saved.
- **Modify:** `bash_scripts/utils/utils_evaluate.sh` — non-fatal `reconcile` at the end of `run_suite`.
- **Modify:** `justfile` — add the `index` recipe.
- **Modify:** `.gitignore` is NOT touched (we WANT `experiments.csv` tracked; it sits at repo root which is not ignored).

**Canonical column order** (used everywhere):
```
run_dir, date, time, status, baseline, model, args,
iters_present, n_instances, n_total, accuracy, avg_cost, avg_in_tok,
avg_out_tok, avg_budget_remaining, reliability, aptitude, unreliability, Notes
```

---

### Task 1: Module skeleton + `render_args`

**Files:**
- Create: `explorer/index.py`
- Test: `tests/explorer/test_index.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/explorer/test_index.py
from pathlib import Path

import pytest

from explorer.index import render_args, COLUMNS


def _cfg(**over):
    cfg = {
        "pipeline": {"baseline": "no_tool", "num_iterations": 9},
        "reader": {
            "database_schema_type": "ddl",
            "is_kb_linearized": True,
            "read_only_gt_tables": True,
            "read_only_gt_kb": True,
            "make_data_ambiguous": False,
        },
        "predictor": {
            "model_name": "Qwen/Qwen3.5-9B",
            "model_provider": "hosted_vllm",
            "temperature": 0.6,
            "top_p": 0.95,
            "enable_thinking": True,
        },
        "user_simulator": {"model_name": "gpt-5.4-mini-2026-03-17"},
    }
    for section, vals in over.items():
        cfg[section] = {**cfg[section], **vals}
    return cfg


class TestRenderArgs:
    def test_full_string(self):
        assert render_args(_cfg()) == (
            "--schema-type ddl --kb-linearized --gt-db --gt-kb "
            "--num-iterations 9 --temperature 0.6 --top-p 0.95 --thinking "
            "--provider hosted_vllm --user-sim gpt-5.4-mini-2026-03-17"
        )

    def test_false_bools_omitted(self):
        s = render_args(_cfg(reader={"read_only_gt_kb": False}, predictor={"enable_thinking": False}))
        assert "--gt-kb" not in s
        assert "--thinking" not in s
        assert "--gt-db" in s  # still true

    def test_one_token_difference(self):
        a = render_args(_cfg())
        b = render_args(_cfg(reader={"read_only_gt_kb": False}))
        assert a.replace(" --gt-kb", "") == b

    def test_ambiguous_flag_appears(self):
        assert "--ambiguous" in render_args(_cfg(reader={"make_data_ambiguous": True}))

    def test_columns_constant_order(self):
        assert COLUMNS[:7] == ["run_dir", "date", "time", "status", "baseline", "model", "args"]
        assert COLUMNS[-1] == "Notes"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/explorer/test_index.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'explorer.index'`.

- [ ] **Step 3: Write minimal implementation**

```python
# explorer/index.py
"""Maintain experiments.csv — a git-tracked cross-run index of eval runs.

One row per run dir (keyed by path relative to results/). The `args` column is a
canonical flag string rendered from config.yaml so ablations diff as one token.
Metrics come from explorer.loader so the CSV never disagrees with the explorer app.
"""
from __future__ import annotations

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = _REPO_ROOT / "results"
DEFAULT_CSV = _REPO_ROOT / "experiments.csv"

COLUMNS = [
    "run_dir", "date", "time", "status", "baseline", "model", "args",
    "iters_present", "n_instances", "n_total", "accuracy", "avg_cost", "avg_in_tok",
    "avg_out_tok", "avg_budget_remaining", "reliability", "aptitude", "unreliability",
    "Notes",
]


def render_args(config: dict) -> str:
    """Canonical flag string built from a config.yaml dict.

    Fixed order; boolean flags emitted only when true. Derived from config (not the
    raw CLI) so launch-time and reconcile always produce the same string.
    """
    pipeline = config.get("pipeline") or {}
    reader = config.get("reader") or {}
    predictor = config.get("predictor") or {}
    user = config.get("user_simulator") or {}

    parts: list[str] = ["--schema-type", str(reader.get("database_schema_type", ""))]
    if reader.get("is_kb_linearized"):
        parts.append("--kb-linearized")
    if reader.get("read_only_gt_tables"):
        parts.append("--gt-db")
    if reader.get("read_only_gt_kb"):
        parts.append("--gt-kb")
    if reader.get("make_data_ambiguous"):
        parts.append("--ambiguous")
    parts += ["--num-iterations", str(pipeline.get("num_iterations", ""))]
    parts += ["--temperature", str(predictor.get("temperature", ""))]
    parts += ["--top-p", str(predictor.get("top_p", ""))]
    if predictor.get("enable_thinking"):
        parts.append("--thinking")
    parts += ["--provider", str(predictor.get("model_provider", ""))]
    parts += ["--user-sim", str(user.get("model_name", ""))]
    return " ".join(parts)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/explorer/test_index.py -q`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add explorer/index.py tests/explorer/test_index.py
git commit -m "feat(explorer): add render_args + index module skeleton"
```

---

### Task 2: `derive_status` + config-only row + `append_stub`

**Files:**
- Modify: `explorer/index.py`
- Test: `tests/explorer/test_index.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/explorer/test_index.py`:

```python
import csv

from explorer.index import derive_status, append_stub


def _write_config(run_dir: Path, cfg: dict) -> None:
    import yaml
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


class TestDeriveStatus:
    def test_error_when_dir_suffixed(self, tmp_path):
        d = tmp_path / "run__error"
        d.mkdir()
        assert derive_status(d, _cfg(), n_present=0) == "error"

    def test_error_when_error_jsonl_present(self, tmp_path):
        d = tmp_path / "run"
        d.mkdir()
        (d / "results_error.jsonl").write_text("{}\n")
        assert derive_status(d, _cfg(), n_present=3) == "error"

    def test_done_when_all_iterations_present(self, tmp_path):
        d = tmp_path / "run"
        d.mkdir()
        assert derive_status(d, _cfg(pipeline={"num_iterations": 3}), n_present=3) == "done"

    def test_partial_when_some_iterations(self, tmp_path):
        d = tmp_path / "run"
        d.mkdir()
        assert derive_status(d, _cfg(pipeline={"num_iterations": 9}), n_present=4) == "partial"

    def test_temp_zero_collapses_expected_to_one(self, tmp_path):
        d = tmp_path / "run"
        d.mkdir()
        cfg = _cfg(pipeline={"num_iterations": 9}, predictor={"temperature": 0.0})
        assert derive_status(d, cfg, n_present=1) == "done"


class TestAppendStub:
    def test_appends_running_row_with_config_columns(self, tmp_path):
        results = tmp_path / "results"
        run = results / "2026-06-02" / "08-48-35" / "no_tool__Qwen__ddl__iter9"
        _write_config(run, _cfg())
        csv_path = tmp_path / "experiments.csv"

        append_stub(run, csv_path=csv_path, results_root=results)

        rows = list(csv.DictReader(csv_path.open()))
        assert len(rows) == 1
        row = rows[0]
        assert row["run_dir"] == "2026-06-02/08-48-35/no_tool__Qwen__ddl__iter9"
        assert row["date"] == "2026-06-02"
        assert row["time"] == "08-48-35"
        assert row["status"] == "running"
        assert row["baseline"] == "no_tool"
        assert row["model"] == "Qwen/Qwen3.5-9B"
        assert "--schema-type ddl" in row["args"]
        assert row["accuracy"] == ""   # metrics blank until reconcile
        assert row["Notes"] == ""

    def test_idempotent(self, tmp_path):
        results = tmp_path / "results"
        run = results / "2026-06-02" / "08-48-35" / "slug"
        _write_config(run, _cfg())
        csv_path = tmp_path / "experiments.csv"

        append_stub(run, csv_path=csv_path, results_root=results)
        append_stub(run, csv_path=csv_path, results_root=results)

        rows = list(csv.DictReader(csv_path.open()))
        assert len(rows) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/explorer/test_index.py -q`
Expected: FAIL with `ImportError: cannot import name 'derive_status'`.

- [ ] **Step 3: Write minimal implementation**

Add to `explorer/index.py` (after `render_args`). Add `import csv`, `import os`, and `import yaml` to the imports at the top of the file.

```python
def derive_status(run_path: Path, config: dict, n_present: int) -> str:
    """running | partial | done | error for one run dir."""
    if run_path.name.endswith("__error") or (run_path / "results_error.jsonl").exists():
        return "error"
    pipeline = config.get("pipeline") or {}
    predictor = config.get("predictor") or {}
    expected = pipeline.get("num_iterations") or 0
    temp = predictor.get("temperature") or 0
    if temp <= 0:                       # temperature<=0 collapses to a single pass
        expected = 1
    if expected and n_present >= expected:
        return "done"
    if n_present > 0:
        return "partial"
    return "running"


def _load_config(run_dir: Path) -> dict:
    p = run_dir / "config.yaml"
    if not p.exists():
        return {}
    with p.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _parse_time(run_key: str) -> str:
    """Time tag from a run_key: 'HH-MM-SS/slug' (new) or 'HH_MM_SS__slug' (old flat)."""
    if "/" in run_key:
        return run_key.split("/", 1)[0]
    return run_key.split("__", 1)[0]


def _blank_metrics() -> dict:
    return {
        "iters_present": "", "n_instances": "", "n_total": "", "accuracy": "",
        "avg_cost": "", "avg_in_tok": "", "avg_out_tok": "", "avg_budget_remaining": "",
        "reliability": "", "aptitude": "", "unreliability": "",
    }


def _row_from_config(run_dir: str, date: str, time: str, config: dict, status: str) -> dict:
    pipeline = config.get("pipeline") or {}
    predictor = config.get("predictor") or {}
    row = {
        "run_dir": run_dir, "date": date, "time": time, "status": status,
        "baseline": pipeline.get("baseline", ""),
        "model": predictor.get("model_name", ""),
        "args": render_args(config),
        "Notes": "",
    }
    row.update(_blank_metrics())
    return row


def _read_csv(csv_path: Path) -> dict[str, dict]:
    """Existing rows keyed by run_dir (empty when the file is absent)."""
    if not csv_path.exists():
        return {}
    with csv_path.open(encoding="utf-8", newline="") as f:
        return {r["run_dir"]: r for r in csv.DictReader(f)}


def _write_csv(csv_path: Path, rows: dict[str, dict]) -> None:
    """Atomic write of all rows, sorted newest-first by (date, time)."""
    ordered = sorted(rows.values(), key=lambda r: (r.get("date", ""), r.get("time", "")), reverse=True)
    tmp = csv_path.with_suffix(csv_path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        for row in ordered:
            w.writerow({c: row.get(c, "") for c in COLUMNS})
    os.replace(tmp, csv_path)


def append_stub(run_dir_path: Path, csv_path: Path | None = None, results_root: Path | None = None) -> None:
    """Append a status=running, config-only row for run_dir_path (idempotent)."""
    csv_path = csv_path or DEFAULT_CSV
    results_root = results_root or DEFAULT_RESULTS
    rel = str(run_dir_path.resolve().relative_to(results_root.resolve()))
    existing = _read_csv(csv_path)
    if rel in existing:
        return
    config = _load_config(run_dir_path)
    date = rel.split("/", 1)[0]
    run_key = rel.split("/", 1)[1] if "/" in rel else rel
    row = _row_from_config(rel, date, _parse_time(run_key), config, status="running")
    existing[rel] = row
    _write_csv(csv_path, existing)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/explorer/test_index.py -q`
Expected: PASS (all tests in the file).

- [ ] **Step 5: Commit**

```bash
git add explorer/index.py tests/explorer/test_index.py
git commit -m "feat(explorer): add derive_status + append_stub"
```

---

### Task 3: `reconcile` (metrics + Notes preservation)

**Files:**
- Modify: `explorer/index.py`
- Test: `tests/explorer/test_index.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/explorer/test_index.py`:

```python
import json

from explorer.index import reconcile


def _make_record(execution_accuracy=False, instance_id="t1", database="mydb"):
    return {
        "execution_accuracy": execution_accuracy,
        "instance_id": instance_id,
        "selected_database": database,
        "mean_prompt_tokens": 100,
        "mean_completion_tokens": 50,
        "total_cost": 0.01,
        "updated_user_patience": 3,
        "tool_calls_in_order": [],
        "messages": [],
        "iteration": 0,
    }


def _make_run(results: Path, date: str, time: str, slug: str, cfg: dict, records: list[dict]) -> Path:
    run = results / date / time / slug
    _write_config(run, cfg)
    (run / "results_iter0.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in records), encoding="utf-8"
    )
    return run


class TestReconcile:
    def test_builds_row_with_metrics(self, tmp_path):
        results = tmp_path / "results"
        _make_run(
            results, "2026-06-02", "08-48-35", "no_tool__Qwen__ddl__iter1",
            _cfg(pipeline={"num_iterations": 1}, predictor={"temperature": 0.0}),
            [_make_record(execution_accuracy=True), _make_record(execution_accuracy=False, instance_id="t2")],
        )
        csv_path = tmp_path / "experiments.csv"

        reconcile(results_root=results, csv_path=csv_path)

        rows = list(csv.DictReader(csv_path.open()))
        assert len(rows) == 1
        row = rows[0]
        assert row["run_dir"] == "2026-06-02/08-48-35/no_tool__Qwen__ddl__iter1"
        assert row["status"] == "done"
        assert row["n_instances"] == "2"
        assert row["accuracy"] == "0.5"
        assert "--schema-type ddl" in row["args"]

    def test_preserves_notes_across_reconcile(self, tmp_path):
        results = tmp_path / "results"
        _make_run(
            results, "2026-06-02", "08-48-35", "slug",
            _cfg(pipeline={"num_iterations": 1}, predictor={"temperature": 0.0}),
            [_make_record(execution_accuracy=True)],
        )
        csv_path = tmp_path / "experiments.csv"
        reconcile(results_root=results, csv_path=csv_path)

        # User edits the Notes column, then adds another passing record.
        rows = list(csv.DictReader(csv_path.open()))
        rows[0]["Notes"] = "best ablation so far"
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

        reconcile(results_root=results, csv_path=csv_path)

        out = list(csv.DictReader(csv_path.open()))
        assert out[0]["Notes"] == "best ablation so far"

    def test_keeps_running_stub_not_rediscovered(self, tmp_path):
        results = tmp_path / "results"
        # A stub for a run that has no results yet (append_stub wrote it).
        run = results / "2026-06-02" / "09-00-00" / "pending"
        _write_config(run, _cfg())
        csv_path = tmp_path / "experiments.csv"
        append_stub(run, csv_path=csv_path, results_root=results)
        # A different, completed run.
        _make_run(
            results, "2026-06-02", "08-00-00", "done_slug",
            _cfg(pipeline={"num_iterations": 1}, predictor={"temperature": 0.0}),
            [_make_record(execution_accuracy=True)],
        )

        reconcile(results_root=results, csv_path=csv_path)

        out = {r["run_dir"]: r for r in csv.DictReader(csv_path.open())}
        assert "2026-06-02/09-00-00/pending" in out          # stub survived
        assert out["2026-06-02/09-00-00/pending"]["status"] == "running"
        assert "2026-06-02/08-00-00/done_slug" in out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/explorer/test_index.py -q`
Expected: FAIL with `ImportError: cannot import name 'reconcile'`.

- [ ] **Step 3: Write minimal implementation**

Add to `explorer/index.py`. Add the loader import near the top of the file (after the stdlib imports):

```python
try:  # bare import when Streamlit/this module runs from explorer/
    from loader import list_runs, load_run, RunData
except ModuleNotFoundError:
    from explorer.loader import list_runs, load_run, RunData
```

Then add the functions:

```python
def _row_from_run(run_dir: str, date: str, time: str, run_path: Path, run: "RunData") -> dict:
    st = run.stats
    rel = st.reliability
    config = run.config
    pipeline = config.get("pipeline") or {}
    predictor = config.get("predictor") or {}
    return {
        "run_dir": run_dir, "date": date, "time": time,
        "status": derive_status(run_path, config, run.n_iterations),
        "baseline": pipeline.get("baseline", ""),
        "model": predictor.get("model_name", ""),
        "args": render_args(config),
        "iters_present": run.n_iterations,
        "n_instances": st.n_instances,
        "n_total": st.n_total,
        "accuracy": round(st.pass_at_1, 4),
        "avg_cost": round(st.avg_cost, 6),
        "avg_in_tok": round(st.avg_input_tokens, 1),
        "avg_out_tok": round(st.avg_output_tokens, 1),
        "avg_budget_remaining": round(st.avg_budget_remaining, 3),
        "reliability": round(rel.reliability, 4) if rel else "",
        "aptitude": round(rel.aptitude, 4) if rel else "",
        "unreliability": round(rel.unreliability, 4) if rel else "",
        "Notes": "",
    }


def reconcile(results_root: Path | None = None, csv_path: Path | None = None) -> None:
    """Rescan results_root, rebuild the CSV, and preserve Notes by run_dir.

    Rows for runs that are no longer discoverable (e.g. still-running stubs with no
    jsonl yet, or archived dirs) are carried over verbatim so nothing is lost.
    """
    results_root = results_root or DEFAULT_RESULTS
    csv_path = csv_path or DEFAULT_CSV
    existing = _read_csv(csv_path)

    rows: dict[str, dict] = {}
    for date, run_keys in list_runs(results_root).items():
        for run_key in run_keys:
            rel = f"{date}/{run_key}"
            run_path = results_root / date / run_key
            run = load_run(run_path)
            row = _row_from_run(rel, date, _parse_time(run_key), run_path, run)
            row["Notes"] = existing.get(rel, {}).get("Notes", "")
            rows[rel] = row

    for rel, row in existing.items():            # keep stubs/archived rows
        rows.setdefault(rel, row)

    _write_csv(csv_path, rows)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/explorer/test_index.py -q`
Expected: PASS (all tests).

- [ ] **Step 5: Commit**

```bash
git add explorer/index.py tests/explorer/test_index.py
git commit -m "feat(explorer): add reconcile with Notes preservation"
```

---

### Task 4: `__main__` CLI (reconcile / append)

**Files:**
- Modify: `explorer/index.py`
- Test: `tests/explorer/test_index.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/explorer/test_index.py`:

```python
import subprocess
import sys


class TestCli:
    def test_reconcile_subcommand(self, tmp_path):
        results = tmp_path / "results"
        _make_run(
            results, "2026-06-02", "08-48-35", "slug",
            _cfg(pipeline={"num_iterations": 1}, predictor={"temperature": 0.0}),
            [_make_record(execution_accuracy=True)],
        )
        csv_path = tmp_path / "experiments.csv"
        # Call the module's main() directly so we exercise the argparse wiring.
        from explorer.index import main
        main(["reconcile", "--results-root", str(results), "--csv", str(csv_path)])
        rows = list(csv.DictReader(csv_path.open()))
        assert rows and rows[0]["run_dir"] == "2026-06-02/08-48-35/slug"

    def test_append_subcommand(self, tmp_path):
        results = tmp_path / "results"
        run = results / "2026-06-02" / "08-48-35" / "slug"
        _write_config(run, _cfg())
        csv_path = tmp_path / "experiments.csv"
        from explorer.index import main
        main(["append", str(run), "--csv", str(csv_path), "--results-root", str(results)])
        rows = list(csv.DictReader(csv_path.open()))
        assert rows[0]["status"] == "running"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/explorer/test_index.py -q`
Expected: FAIL with `ImportError: cannot import name 'main'`.

- [ ] **Step 3: Write minimal implementation**

Add to `explorer/index.py` (and `import argparse` at top):

```python
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="explorer.index", description="Maintain experiments.csv")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="path to experiments.csv")
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS, help="results/ root")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("reconcile", help="rescan results/ and rebuild the CSV")
    ap = sub.add_parser("append", help="append a running stub for one run dir")
    ap.add_argument("run_dir", type=Path)
    args = parser.parse_args(argv)

    if args.cmd == "reconcile":
        reconcile(results_root=args.results_root, csv_path=args.csv)
    elif args.cmd == "append":
        append_stub(args.run_dir, csv_path=args.csv, results_root=args.results_root)


if __name__ == "__main__":
    main()
```

Note: `--csv`/`--results-root` are defined on the top-level parser so they work
before the subcommand (e.g. `python -m explorer.index --csv X reconcile`). argparse
accepts them before the subcommand; the tests pass them after, which also works
because they are top-level optionals consumed by the parent parser.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/explorer/test_index.py -q`
Expected: PASS.

If the test fails because top-level optionals after the subcommand are not parsed,
move `--csv` and `--results-root` definitions to BOTH subparsers instead. Re-run.

- [ ] **Step 5: Commit**

```bash
git add explorer/index.py tests/explorer/test_index.py
git commit -m "feat(explorer): add index CLI (reconcile/append)"
```

---

### Task 5: `just index` recipe

**Files:**
- Modify: `justfile`

- [ ] **Step 1: Add the recipe**

Add after the `recover` recipe block (around line 159 in `justfile`):

```just
# ── index ──────────────────────────────────────────────────────────────────────
# Rebuild experiments.csv at the repo root: rescan results/, fill metrics from the
# explorer loader, and preserve the hand-edited Notes column. Safe to run anytime.
#
#   just index
index:
    uv run python -m explorer.index reconcile
```

- [ ] **Step 2: Verify the recipe is listed and runs**

Run: `just --list | grep index`
Expected: shows the `index` recipe.

Run: `just index`
Expected: exits 0; `experiments.csv` appears/updates at the repo root with one row per existing run under `results/`.

- [ ] **Step 3: Sanity-check the output**

Run: `head -1 experiments.csv` (header) and `wc -l experiments.csv`.
Expected: header matches the canonical column order; line count ≈ number of run dirs + 1.

- [ ] **Step 4: Commit**

```bash
git add justfile experiments.csv
git commit -m "feat(just): add 'just index' to rebuild experiments.csv"
```

---

### Task 6: Launch hook in the pipeline

**Files:**
- Modify: `src/conversation2sql/eval_framework/main_pipe_workflow.py:108` (right after `_save_configs_as_yaml(...)`)

- [ ] **Step 1: Add the best-effort append call**

Immediately after the `_save_configs_as_yaml(...)` call that ends at line 108, insert:

```python
    # Register this run in the experiments index (best-effort: never abort a run).
    # The repo root must be importable for the top-level `explorer` package, which
    # is not part of the installed `conversation2sql` distribution.
    try:
        import sys

        repo_root = str(Path(__file__).resolve().parents[3])
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        from explorer.index import append_stub

        append_stub(output_folder)
    except Exception as e:  # noqa: BLE001 - indexing must never break evaluation
        logger.warning("Could not append run to experiments index: %s", e)
```

(`logger` and `Path` are already imported in this module; verify the `logger`
name exists near the top — it is used elsewhere in the file.)

- [ ] **Step 2: Smoke-test the import path resolves**

Run:
```bash
uv run python -c "import sys; from pathlib import Path; sys.path.insert(0, str(Path('src/conversation2sql/eval_framework/main_pipe_workflow.py').resolve().parents[3])); from explorer.index import append_stub; print('ok', append_stub)"
```
Expected: prints `ok <function append_stub ...>`.

- [ ] **Step 3: Run the existing test suite (no regressions)**

Run: `uv run pytest tests/ -q`
Expected: PASS (no import errors from the pipeline change).

- [ ] **Step 4: Commit**

```bash
git add src/conversation2sql/eval_framework/main_pipe_workflow.py
git commit -m "feat(pipeline): register each run in experiments index at launch"
```

---

### Task 7: Auto-reconcile at end of `run_suite`

**Files:**
- Modify: `bash_scripts/utils/utils_evaluate.sh` (inside `run_suite`, after the `=== Done ... ===` log line near line 239)

- [ ] **Step 1: Add the non-fatal reconcile**

After the final `log_section "=== Done ... ==="` line in `run_suite`, add:

```bash
  # Best-effort: refresh the experiments index so this run's metrics + final
  # status land in experiments.csv. Never fail the run on an index error.
  ( cd "${BASE_WORK}" && uv run python -m explorer.index reconcile ) \
    || log_section "experiments index reconcile failed (non-fatal)" "${MY_SLURM_JOB_ID:-}"
```

- [ ] **Step 2: Static-check the script**

Run: `bash -n bash_scripts/utils/utils_evaluate.sh`
Expected: no output (syntax OK).

- [ ] **Step 3: Dry verification of the command shape**

Run: `cd /workspaces/conversation2SQL && uv run python -m explorer.index reconcile && echo OK`
Expected: `OK` and `experiments.csv` refreshed.

- [ ] **Step 4: Commit**

```bash
git add bash_scripts/utils/utils_evaluate.sh
git commit -m "feat(eval): reconcile experiments index after each run"
```

---

### Task 8: Document it

**Files:**
- Modify: `bash_scripts/README.md` (add a short "Experiment tracking" note) and/or `explorer/README.md`

- [ ] **Step 1: Add a short section**

Add to `explorer/README.md` (and a one-line pointer in `bash_scripts/README.md`):

```markdown
## Experiment tracking (`experiments.csv`)

`experiments.csv` at the repo root is a git-tracked index of every run under
`results/`. One row per run dir; the `args` column is a canonical flag string so
ablations diff as a single token. Metrics (`accuracy`, `avg_cost`, reliability, …)
come from `explorer.loader`, so the CSV agrees with the explorer app.

- A `status=running` stub is appended automatically when a run launches.
- Metrics + final status are filled by `reconcile`, which runs after each eval and
  on demand via `just index`.
- The `Notes` column is yours to edit; `reconcile` never overwrites it.
```

- [ ] **Step 2: Commit**

```bash
git add explorer/README.md bash_scripts/README.md
git commit -m "docs: document experiments.csv tracking"
```

---

## Self-review notes

- **Spec coverage:** CSV at repo root git-tracked (Task 5/6 produce it) ✓; dedicated `run_dir/date/time/status/baseline/model` + single `args` column (Task 1–3) ✓; metrics from `explorer.loader` (Task 3) ✓; Notes preservation invariant + test (Task 3) ✓; hybrid population — append at launch (Task 6) + reconcile (Task 3/5/7) ✓; `just index` (Task 5) ✓; tests incl. render_args, reconcile, Notes, append idempotency, status (Task 1–4) ✓.
- **Deviation from spec (intentional):** the launch hook lives in the pipeline (after the config snapshot is written), not in `run_suite`, because `config.yaml` does not exist yet when `run_suite` creates the dir. `run_suite` instead runs a post-run `reconcile` (Task 7). Status-quo intent ("append at launch, config in place") is preserved.
- **Type/name consistency:** `render_args`, `derive_status`, `append_stub`, `reconcile`, `main`, `COLUMNS`, `_write_csv`, `_read_csv` used consistently across tasks. Metric field names match `explorer.loader.RunStats` (`pass_at_1`, `avg_input_tokens`, `avg_output_tokens`, `avg_cost`, `avg_budget_remaining`, `n_instances`, `n_total`) and `ReliabilityStats` (`reliability`, `aptitude`, `unreliability`).
- **Out of scope (YAGNI):** no auto-commit of `experiments.csv`; no per-DB/per-error columns; no SQLite.
