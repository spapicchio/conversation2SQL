"""Migrate old results layout to the new baseline/date/time__slug structure.

Old layout:
    results/<YYYY-MM-DD>/<HH-MM-SS>/

New layout:
    results/<baseline>/<YYYY_MM_DD>/<HH_MM_SS>__<slug>/

The slug is derived from the run's config.yaml using the same logic as
_build_run_slug in main_pipe_workflow.py.

Usage (dry-run, prints planned moves):
    uv run python explorer/_adjust_old_runs.py

Usage (execute moves):
    uv run python explorer/_adjust_old_runs.py --execute
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

import yaml

from conversation2sql.cli_parser import PydanticParser
from conversation2sql.config_input import ConfigPipeline, ConfigPredictor, ConfigReader, ConfigUserSimulator
from conversation2sql.eval_framework.main_pipe_workflow import _build_run_slug




# ── old-layout detection ───────────────────────────────────────────────────────

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")   # YYYY-MM-DD  (old)
_TIME_RE = re.compile(r"^\d{2}-\d{2}-\d{2}$")   # HH-MM-SS   (old)


def _find_old_runs(results_root: Path) -> list[tuple[Path, str, str]]:
    """Return (run_dir, date_str, time_str) for every old-format run folder."""
    old_runs: list[tuple[Path, str, str]] = []
    for date_dir in results_root.iterdir():
        if not date_dir.is_dir() or not _DATE_RE.match(date_dir.name):
            continue
        for time_dir in date_dir.iterdir():
            if not time_dir.is_dir() or not _TIME_RE.match(time_dir.name):
                continue
            old_runs.append((time_dir, date_dir.name, time_dir.name))
    return old_runs


# ── new path construction ──────────────────────────────────────────────────────

def _new_path(results_root: Path, run_dir: Path, date_str: str, time_str: str) -> Path:
    cfg_path = run_dir / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"No config.yaml in {run_dir}")

    parser = PydanticParser([ConfigPipeline, ConfigReader, ConfigPredictor, ConfigUserSimulator])
    config_pipeline, config_reader, config_predictor, _ = parser.parse_args_and_config(
        ["--config", str(cfg_path)]
    )
    slug = _build_run_slug(config_pipeline, config_reader, config_predictor)

    new_date = date_str.replace("-", "_")           # 2026-05-16 → 2026_05_16
    new_time = time_str.replace("-", "_")           # 09-18-44  → 09_18_44
    folder_name = f"{new_time}__{slug}"

    return results_root  / new_date / folder_name


# ── config.yaml patch ──────────────────────────────────────────────────────────

def _update_config_output_folder(config_path: Path, new_folder: Path) -> None:
    with config_path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["pipeline"]["output_folder"] = str(new_folder)
    with config_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", default="results", help="Path to the results root (default: results/)")
    parser.add_argument("--execute", action="store_true", help="Actually move folders (default is dry-run)")
    args = parser.parse_args()

    results_root = Path(args.results).resolve()
    if not results_root.is_dir():
        raise SystemExit(f"Results directory not found: {results_root}")

    old_runs = _find_old_runs(results_root)
    if not old_runs:
        print("No old-format runs found. Nothing to migrate.")
        return

    moves: list[tuple[Path, Path]] = []
    errors: list[str] = []

    for run_dir, date_str, time_str in sorted(old_runs):
        try:
            dst = _new_path(results_root, run_dir, date_str, time_str)
        except FileNotFoundError as exc:
            errors.append(f"  SKIP  {run_dir}  — {exc}")
            continue
        moves.append((run_dir, dst))

    print(f"Found {len(old_runs)} old run(s), {len(errors)} skipped.\n")

    if errors:
        print("Skipped (no config.yaml):")
        for msg in errors:
            print(msg)
        print()

    label = "Executing" if args.execute else "Planned moves (dry-run)"
    print(f"{label}:")
    for src, dst in moves:
        print(f"  {src}")
        print(f"  → {dst}\n")

    if not args.execute:
        print("Re-run with --execute to apply.")
        return

    # --- actually move ---
    moved_dates: set[Path] = set()
    for src, dst in moves:
        if dst.exists():
            print(f"  SKIP (already exists): {dst}")
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        cfg_path = dst / "config.yaml"
        if cfg_path.exists():
            _update_config_output_folder(cfg_path, dst)
        moved_dates.add(src.parent)
        print(f"  Moved → {dst}")

    # clean up now-empty date directories
    for date_dir in moved_dates:
        remaining = [p for p in date_dir.iterdir() if not p.name.startswith(".")]
        if not remaining:
            date_dir.rmdir()
            print(f"  Removed empty dir: {date_dir}")
            # also clean parent if empty
            if not any(date_dir.parent.iterdir()):
                date_dir.parent.rmdir()
                print(f"  Removed empty dir: {date_dir.parent}")

    print("\nDone.")


if __name__ == "__main__":
    main()
