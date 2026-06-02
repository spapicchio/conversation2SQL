"""Maintain experiments.csv — a git-tracked cross-run index of eval runs.

One row per run dir (keyed by path relative to results/). The `args` column is a
canonical flag string rendered from config.yaml so ablations diff as one token.
Metrics come from explorer.loader so the CSV never disagrees with the explorer app.
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = _REPO_ROOT / "results"
DEFAULT_CSV = _REPO_ROOT / "experiments.csv"

COLUMNS = [
    "run_dir", "date", "time", "status", "baseline", "model", "args",
    "iters_present", "n_instances", "n_total", "accuracy", "avg_cost", "avg_in_tok",
    "avg_out_tok", "avg_budget_remaining", "reliability", "aptitude", "unreliability",
    "Notes",
]

try:  # bare import when running from explorer/
    from loader import list_runs, load_run, RunData
except ModuleNotFoundError:
    from explorer.loader import list_runs, load_run, RunData


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


def derive_status(run_path: Path, config: dict, n_present: int) -> str:
    """running | partial | done | error for one run dir."""
    if run_path.name.endswith("__error") or (run_path / "results_error.jsonl").exists():
        return "error"
    pipeline = config.get("pipeline") or {}
    predictor = config.get("predictor") or {}
    expected = pipeline.get("num_iterations") or 0
    temp = predictor.get("temperature") or 0
    if temp <= 0:
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

    for rel, row in existing.items():
        rows.setdefault(rel, row)

    _write_csv(csv_path, rows)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="explorer.index", description="Maintain experiments.csv")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="path to experiments.csv")
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS, help="results/ root")
    sub = parser.add_subparsers(dest="cmd", required=True)

    rec_p = sub.add_parser("reconcile", help="rescan results/ and rebuild the CSV")
    rec_p.add_argument("--csv", type=Path, default=None, help="path to experiments.csv")
    rec_p.add_argument("--results-root", type=Path, default=None, help="results/ root")

    ap = sub.add_parser("append", help="append a running stub for one run dir")
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--csv", type=Path, default=None, help="path to experiments.csv")
    ap.add_argument("--results-root", type=Path, default=None, help="results/ root")

    args = parser.parse_args(argv)

    # Subparser values override parent defaults when provided
    csv_path = getattr(args, "csv", None) or DEFAULT_CSV
    results_root = getattr(args, "results_root", None) or DEFAULT_RESULTS

    if args.cmd == "reconcile":
        reconcile(results_root=results_root, csv_path=csv_path)
    elif args.cmd == "append":
        append_stub(args.run_dir, csv_path=csv_path, results_root=results_root)


if __name__ == "__main__":
    main()
