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
# stop `;`/`&&`/command-substitution) when they appear *outside* quotes, where
# the shell actually interprets them. `;`/`&`/`||`/`>`/`<`/newline are literal
# text once quoted (single or double), so e.g. `psql -c "SELECT 1;"` — the form
# our own docstring tells the model to use — is fine. Backtick and `$(`/`${`
# stay dangerous even inside double quotes, since bash still expands them
# there; only single quotes neutralize them.

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


_STDERR_TO_DEVNULL = "2>/dev/null"


def _is_stderr_devnull_at(command: str, i: int) -> bool:
    """True if `command[i:]` starts with a standalone `2>/dev/null` token —
    the one redirect we allow unquoted, since it only discards a whitelisted
    command's stderr and can't write anywhere or leak data (unlike a general
    `>`/`<` redirect)."""
    end = i + len(_STDERR_TO_DEVNULL)
    return (
        command[i:end] == _STDERR_TO_DEVNULL
        and (i == 0 or command[i - 1].isspace())
        and (end == len(command) or command[end].isspace())
    )


def _has_unquoted_forbidden_metachar(command: str) -> bool:
    """True if `command` contains a shell metacharacter the way the shell would
    actually interpret it: `;`/`&`/`||`/`>`/`<`/newline only count when
    unquoted; backtick and `$(`/`${` count even inside double quotes (bash
    still expands them there). Unbalanced quotes are left for `shlex.split` to
    reject with the whitelist message. The lone exception is a standalone
    unquoted `2>/dev/null` (see `_is_stderr_devnull_at`)."""
    state: str | None = None  # None, "'", or '"'
    i, n = 0, len(command)
    while i < n:
        c = command[i]
        if state == "'":
            if c == "'":
                state = None
            i += 1
            continue
        if c == "\\" and i + 1 < n:
            i += 2
            continue
        if c == "'" and state is None:
            state = "'"
            i += 1
            continue
        if c == '"':
            state = None if state == '"' else '"'
            i += 1
            continue
        if c == "`" or (c == "$" and i + 1 < n and command[i + 1] in "({"):
            return True
        if state is None:
            if c == "2" and _is_stderr_devnull_at(command, i):
                i += len(_STDERR_TO_DEVNULL)
                continue
            if c in (";", "&", ">", "<", "\n"):
                return True
            if c == "|" and i + 1 < n and command[i + 1] == "|":
                return True
        i += 1
    return False


def _refusal(command: str) -> str | None:
    """Return a refusal message if `command` is not an allowed read-only shell
    line, else None. Order: metachars → tokenizing → per-segment whitelist →
    psql host-command guardrail."""
    if _has_unquoted_forbidden_metachar(command):
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
        Shell control characters (; & > < ` newline) are never allowed outside of
        quotes — a `;` terminating your quoted SQL is fine. A trailing
        `2>/dev/null` to silence stderr is also allowed."""
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
