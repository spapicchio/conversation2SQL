import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "bash_scripts" / "eval_payload.sh"


def _dry_run(**env_overrides):
    env = {
        "PATH": "/usr/bin:/bin",
        "DRY_RUN": "1",
        "MODEL": "qwen35",
        "VARIANT": "all_db_all_kb",
        "BASELINE": "no_tool",
        "PREDICTOR_MODEL_PROVIDER": "hosted_vllm",
        "CONCURRENCY": "16",
        "NUM_ITERATIONS": "1",
    }
    env.update(env_overrides)
    return subprocess.run(
        ["bash", str(SCRIPT)], env=env, capture_output=True, text=True, cwd=str(REPO)
    )


def test_dryrun_emits_profile_and_variant_flags():
    out = _dry_run().stdout
    assert "--model-profile qwen35" in out
    assert "--variant all_db_all_kb" in out
    assert "--baseline no_tool" in out


def test_dryrun_forwards_extra_passthrough():
    out = _dry_run(EXTRA="--predictor_top_p 0.8 --reader_user_patience_budget 4").stdout
    assert "--predictor_top_p 0.8" in out
    assert "--reader_user_patience_budget 4" in out


def test_dryrun_no_longer_exports_reader_schema_env():
    out = _dry_run().stdout
    assert "DATABASE_SCHEMA_TYPE=" not in out
