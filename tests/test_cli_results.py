import json
from pathlib import Path

import yaml
from typer.testing import CliRunner

from conversation2sql.cli import app

runner = CliRunner()


def _make_run(
    tmp_path: Path,
    baseline: str,
    date: str,
    time: str,
    model: str,
    records: list[dict],
) -> Path:
    run_dir = tmp_path / baseline / date / time
    run_dir.mkdir(parents=True)
    config = {
        "pipeline": {"baseline": baseline},
        "predictor": {"model_name": model},
    }
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config))
    (run_dir / "results_smaller.jsonl").write_text(
        "\n".join(json.dumps(r) for r in records)
    )
    return run_dir


def test_results_summary_table(tmp_path):
    _make_run(
        tmp_path,
        "bird_full",
        "2026_05_07",
        "12_00_00",
        "gpt-4o",
        [
            {"instance_id": "1", "execution_accuracy": 1, "total_cost": 0.01, "messages": []},
            {"instance_id": "2", "execution_accuracy": 0, "total_cost": 0.03, "messages": []},
        ],
    )
    result = runner.invoke(app, ["results", "--dir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "bird_full" in result.output
    assert "gpt-4o" in result.output
    assert "0.50" in result.output   # mean execution_accuracy = (1+0)/2
    assert "2026_05_07" in result.output


def test_results_empty_dir_exits_with_error(tmp_path):
    result = runner.invoke(app, ["results", "--dir", str(tmp_path / "nonexistent")])
    assert result.exit_code != 0


def test_results_skips_incomplete_run(tmp_path):
    # Directory without results_smaller.jsonl — should be silently skipped
    run_dir = tmp_path / "bird_full" / "2026_05_07" / "12_00_00"
    run_dir.mkdir(parents=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump({"pipeline": {"baseline": "bird_full"}}))
    # no results_smaller.jsonl

    result = runner.invoke(app, ["results", "--dir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "bird_full" not in result.output  # nothing rendered for incomplete run


def test_results_drilldown(tmp_path):
    messages = [
        {"type": "tool", "name": "submit_sql", "content": "passed=True sql=SELECT 1"},
    ]
    _make_run(
        tmp_path,
        "no_tool",
        "2026_05_07",
        "09_00_00",
        "gpt-3.5-turbo",
        [{"instance_id": "42", "execution_accuracy": 1, "total_cost": 0.005, "messages": messages}],
    )
    result = runner.invoke(
        app,
        ["results", "--dir", str(tmp_path), "--run", "no_tool/2026_05_07/09_00_00"],
    )
    assert result.exit_code == 0, result.output
    assert "42" in result.output
    assert "passed=True" in result.output


def test_results_drilldown_missing_run(tmp_path):
    result = runner.invoke(
        app,
        ["results", "--dir", str(tmp_path), "--run", "bird_full/9999_99_99/99_99_99"],
    )
    assert result.exit_code != 0
