import subprocess
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "bash_scripts" / "recover_payload.sh"


def _write_snapshot(run_dir: Path, provider: str):
    run_dir.mkdir(parents=True, exist_ok=True)
    snapshot = {
        "pipeline": {"baseline": "no_tool"},
        "predictor": {
            "model_name": "Qwen/Qwen3.5-9B",
            "model_provider": provider,
            "enable_thinking": None,
        },
        "user_simulator": {"model_name": "x", "model_provider": "openai"},
    }
    (run_dir / "config.yaml").write_text(yaml.safe_dump(snapshot))


def _dry_run(run_dir: Path):
    env = {
        "PATH": "/usr/bin:/bin:/usr/local/bin",
        "DRY_RUN": "1",
        "RESUME_DIR": str(run_dir),
        "BASE_WORK": str(REPO),
    }
    return subprocess.run(
        ["bash", str(SCRIPT)], env=env, capture_output=True, text=True, cwd=str(REPO)
    )


def test_dryrun_emits_resume_run_command(tmp_path):
    run_dir = tmp_path / "run"
    _write_snapshot(run_dir, provider="hosted_vllm")
    out = _dry_run(run_dir).stdout
    assert "--config" in out
    assert str(run_dir / "config.yaml") in out
    assert "--output_folder" in out
    assert str(run_dir) in out
    assert "--resume true" in out


def test_missing_config_yaml_fails(tmp_path):
    run_dir = tmp_path / "empty"
    run_dir.mkdir()
    result = _dry_run(run_dir)
    assert result.returncode != 0
