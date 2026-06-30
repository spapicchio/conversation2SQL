import os
from pathlib import Path

import pytest

from conversation2sql.eval_framework.agents.deep_agent import bash_tool
from conversation2sql.eval_framework.agents.bird_baseline.tools.bird_interact_env_tools import (
    PSQL_GUARDRAIL_REFUSAL,
)


def test_build_pg_env_maps_dsn_and_sets_readonly():
    env = bash_tool.build_pg_env("postgresql://alice:secret@db.local:5544/shop")
    assert env["PGHOST"] == "db.local"
    assert env["PGPORT"] == "5544"
    assert env["PGUSER"] == "alice"
    assert env["PGPASSWORD"] == "secret"
    assert env["PGDATABASE"] == "shop"
    assert "default_transaction_read_only=on" in env["PGOPTIONS"]
    # inherits the parent environment
    assert "PATH" in env


@pytest.fixture
def catalog(tmp_path) -> Path:
    (tmp_path / "tables").mkdir()
    (tmp_path / "tables" / "users.md").write_text("# users\nid, name\n")
    (tmp_path / "database_overview.md").write_text("shop db\n")
    return tmp_path


def _run(catalog: Path, command: str) -> str:
    bash = bash_tool.return_tool_bash(catalog, {**os.environ})
    return bash.invoke({"command": command})


def test_allowed_read_command_runs(catalog):
    out = _run(catalog, "cat tables/users.md")
    assert "exit=0" in out
    assert "id, name" in out


def test_pipe_of_allowed_commands_runs(catalog):
    out = _run(catalog, "cat tables/users.md | grep name")
    assert "exit=0" in out
    assert "name" in out


def test_non_whitelisted_command_is_refused_without_spawning(catalog):
    out = _run(catalog, "echo pwned")
    assert "Refused" in out
    assert "pwned" not in out  # never executed


def test_write_command_is_refused(catalog):
    out = _run(catalog, "rm -rf tables")
    assert "Refused" in out
    assert (catalog / "tables").exists()  # nothing deleted


def test_shell_metacharacters_are_refused(catalog):
    for bad in ["cat tables/users.md; rm -rf .", "cat $(whoami)", "cat a > b", "cat a && rm b", "cat /nonexistent || ls /"]:
        out = _run(catalog, bad)
        assert "Refused" in out, bad


def test_psql_host_meta_command_is_refused(catalog):
    out = _run(catalog, 'psql -c "\\! id"')
    assert out == PSQL_GUARDRAIL_REFUSAL


def test_unbalanced_quotes_are_refused(catalog):
    out = _run(catalog, 'cat "unclosed')
    assert "Refused" in out
