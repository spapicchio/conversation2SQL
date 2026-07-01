"""A single read-only `bash` tool for the deep_agent.

Explores the per-task catalog (cwd) with a small whitelist of read commands and
runs read-only SQL via `psql` (credentials injected via env). Replaces the
deepagents virtual filesystem and `execute_sql`.
"""
from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path

from langchain_core.tools import tool
from psycopg2.extensions import parse_dsn

from conversation2sql.eval_framework.agents.bird_baseline.tools.bird_interact_env_tools import (
    PSQL_GUARDRAIL_REFUSAL,
    _violates_psql_guardrail,
)
from conversation2sql.eval_framework.agents.tool_specs import ToolSpec

# Read-only commands the agent may run (psql is read-only via PGOPTIONS below).
ALLOWED_COMMANDS: frozenset[str] = frozenset(
    {"cat", "ls", "find", "grep", "head", "tail", "wc", "psql"}
)
# Shell control constructs we never allow (a first-token whitelist alone does not
# stop `;`/`&&`/command-substitution). Conservative by design: a quoted
# occurrence is also refused — an acceptable false-positive for read-only use.
_FORBIDDEN_METACHARS: tuple[str, ...] = (";", "&", "||", ">", "<", "`", "$(", "${", "\n")

_TIMEOUT_S = 65
_STDOUT_CAP = 8000
_STDERR_CAP = 2000

WHITELIST_REFUSAL = (
    "Refused: only read-only commands are allowed "
    f"({', '.join(sorted(ALLOWED_COMMANDS))}), optionally joined with a pipe (|)."
)
METACHAR_REFUSAL = (
    "Refused: shell control characters (; & > < ` $( ${ newline) are not allowed; "
    "use a single read-only command or a pipeline of them."
)

BASH_TOOL_SPECS: dict[str, ToolSpec] = {
    # Cost 1 to match the value the agent budgets against; charging 2 here
    # desynced its budget planning and forced premature submits.
    "bash": ToolSpec(
        "bash",
        1.0,
        "Run a read-only shell command to explore the database catalog in your working "
        "directory (cat, ls, find, grep, head, tail, wc; pipes allowed), or run a read-only"
        " SQL query with `psql -c \"SELECT …\"`"),
}


def build_pg_env(db_dsn: str) -> dict[str, str]:
    """Inherit os.environ and add PG* vars from the DSN + a read-only PGOPTIONS,
    so `psql -c "SELECT …"` connects with no credentials in the command."""
    info = parse_dsn(db_dsn)
    env = {**os.environ}
    # Credentials are injected as PG* env vars so the model emits bare
    # `psql -c "SELECT ..."` with no connection flags — psql picks these up automatically.
    if info.get("host"):
        env["PGHOST"] = str(info["host"])
    if info.get("port"):
        env["PGPORT"] = str(info["port"])
    if info.get("user"):
        env["PGUSER"] = str(info["user"])
    if info.get("password"):
        env["PGPASSWORD"] = str(info["password"])
    if info.get("dbname"):
        env["PGDATABASE"] = str(info["dbname"])
    # Server-side read-only enforcement: the server rejects writes/DDL before psql
    # even parses the SQL; statement_timeout caps runaway queries.
    env["PGOPTIONS"] = "-c default_transaction_read_only=on -c statement_timeout=60s"
    return env


def _segments(tokens: list[str]) -> list[list[str]]:
    """Split a shlex token list on bare `|` tokens into pipeline segments."""
    segments: list[list[str]] = [[]]
    for t in tokens:
        if t == "|":
            segments.append([])
        else:
            segments[-1].append(t)
    return segments


def _refusal(command: str) -> str | None:
    """Return a refusal message if `command` is not an allowed read-only shell
    line, else None. Order: metachars → tokenizing → per-segment whitelist →
    psql host-command guardrail."""
    if any(m in command for m in _FORBIDDEN_METACHARS):
        return METACHAR_REFUSAL
    try:
        tokens = shlex.split(command)
    except ValueError:
        return WHITELIST_REFUSAL  # unbalanced quotes, etc.
    if not tokens:
        return WHITELIST_REFUSAL
    has_psql = False
    for seg in _segments(tokens):
        if not seg or seg[0] not in ALLOWED_COMMANDS:
            return WHITELIST_REFUSAL
        if seg[0] == "psql":
            has_psql = True
    # TODO(unmasked-kb-leak): a superuser DB role can still read the on-disk
    # (unmasked) catalog via `psql -c "SELECT pg_read_file('…')"`. The guardrail
    # below only blocks backslash host meta-commands. Deferred — see the spec's
    # Risks section (path-containment, revoke pg_read_file, or block it here).
    if has_psql and _violates_psql_guardrail(command):
        return PSQL_GUARDRAIL_REFUSAL
    return None


def return_tool_bash(catalog_dir: Path, pg_env: dict[str, str]):
    """Build the per-task `bash` tool bound to its catalog cwd and DB env."""

    @tool
    def bash(command: str) -> str:
        """Run a read-only shell command to explore the database catalog in your
        working directory (cat, ls, find, grep, head, tail, wc; pipes allowed), or
        run a read-only SQL query with `psql -c "SELECT …"` (writes are rejected).

        To query the database emit: psql -c "SELECT col FROM table WHERE ...;"
        No connection flags are needed — credentials are pre-injected as env vars.
        Informational meta-commands such as \\dt, \\d <table>, \\l, \\df are allowed;
        host-reaching ones (\\!, \\o, \\copy, \\i) are blocked.
        Shell control characters (; & > < ` newline) are never allowed."""
        refusal = _refusal(command)
        if refusal is not None:
            return refusal
        try:
            proc = subprocess.run(
                ["bash", "-lc", command],
                cwd=str(catalog_dir),
                env=pg_env,
                capture_output=True,
                text=True,
                timeout=_TIMEOUT_S,
            )
        except subprocess.TimeoutExpired:
            return f"Command timed out after {_TIMEOUT_S}s."
        out = (proc.stdout or "")[:_STDOUT_CAP]
        err = (proc.stderr or "")[:_STDERR_CAP]
        return f"exit={proc.returncode}\n--- stdout ---\n{out}\n--- stderr ---\n{err}"

    return bash
