import os
from pathlib import Path

import pytest

from conversation2sql.eval_framework.agents.deep_agent.tools import bash_tool
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


def test_psql_semicolon_inside_double_quotes_is_allowed(catalog):
    # The tool's own docstring tells the model to terminate SQL with `;`
    # inside the quoted -c argument; that must not trip the metachar guard.
    out = _run(catalog, 'psql -c "SELECT 1;"')
    assert "Refused" not in out


def test_psql_command_substitution_inside_double_quotes_is_refused(catalog):
    # Bash still expands `$(...)` and backticks inside double quotes, so
    # these remain dangerous even though `;` inside quotes is now allowed.
    out = _run(catalog, 'psql -c "SELECT $(whoami)"')
    assert "Refused" in out
    out = _run(catalog, 'psql -c "SELECT `whoami`"')
    assert "Refused" in out


def test_bare_semicolon_outside_quotes_is_still_refused(catalog):
    out = _run(catalog, 'psql -c "SELECT 1;"; rm -rf .')
    assert "Refused" in out


def test_psql_comparison_operators_inside_double_quotes_are_allowed(catalog):
    out = _run(catalog, 'psql -c "SELECT * FROM t WHERE age < 30 AND age > 10;"')
    assert "Refused" not in out


def test_bare_redirect_outside_quotes_is_still_refused(catalog):
    out = _run(catalog, "cat a > b")
    assert "Refused" in out


def test_grep_with_stderr_to_devnull_is_allowed(catalog):
    (catalog / "knowledge_base").mkdir()
    (catalog / "knowledge_base" / "a.md").write_text("CCS token\n")
    out = _run(catalog, 'grep -l "CCS" knowledge_base/*.md 2>/dev/null')
    assert "Refused" not in out
    assert "a.md" in out


def test_stdout_redirect_to_devnull_is_still_refused(catalog):
    # Only the exact `2>/dev/null` (stderr) form is carved out.
    out = _run(catalog, "cat a >/dev/null")
    assert "Refused" in out


def test_redirect_to_real_file_is_still_refused(catalog):
    out = _run(catalog, "cat a 2>/tmp/pwned")
    assert "Refused" in out


def test_devnull_redirect_glued_to_other_text_is_still_refused(catalog):
    out = _run(catalog, "cat a2>/dev/nullx")
    assert "Refused" in out
