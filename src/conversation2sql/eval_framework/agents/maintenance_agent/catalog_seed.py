"""Materialize one task's maintenance workspace into a temp dir.

Layout (the agent's cwd):
  <tmp>/ISSUE.md                     (ticket body + growing "## Comments" thread)
  <tmp>/docs/database_overview.md    (deep_agent's catalog, nested under docs/)
  <tmp>/docs/tables/<table>.md
  <tmp>/docs/tables/_foreign_key_constraints.md
  <tmp>/docs/knowledge_base/<node>.md
  <tmp>/queries/answer.sql           (empty stub; the agent's deliverable)
  <tmp>/tests/test_contract.py       (read-only reference; see maintenance_tools.run_tests)

Reuses deep_agent.catalog_seed.materialize_catalog_dir for the docs/ content
(table/KB rendering, per-task masked-KB faithfulness) and nests its flat
output one level under docs/, per the framing spec's workspace anatomy.
"""
from __future__ import annotations

import shutil
from pathlib import Path

from conversation2sql.eval_framework.agents.deep_agent.catalog_seed import (
    materialize_catalog_dir,
)
from conversation2sql.eval_framework.agents.maintenance_agent.tools import MA_TOOL_COSTS
from conversation2sql.eval_framework.state import TaskData

STUB_QUERY_CONTENT = "-- TODO: replace this stub with your SQL query.\n"

ISSUE_TEMPLATE = "# Issue\n\n{task_question}\n\n## Comments\n"

TEST_CONTRACT_REFERENCE = r'''"""Visible contract test for queries/answer.sql.

Executability only: checks structure, never correctness, so it can never leak
the ground-truth answer. This file is a read-only reference; the sandboxed
bash tool in this environment cannot execute scripts directly, so the
equivalent check is exposed as the `run_tests` tool instead.
"""
import pathlib
import re

TARGET = pathlib.Path(__file__).parent.parent / "queries" / "answer.sql"


def _strip_sql_comments(text: str) -> str:
    text = re.sub(r"--.*", "", text)
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    return text.strip()


def test_query_is_written():
    """queries/answer.sql must no longer be the empty stub."""
    assert _strip_sql_comments(TARGET.read_text())


def test_query_parses():
    """queries/answer.sql must EXPLAIN successfully against the database."""
    # Run via the `run_tests` tool in this environment.
'''


def materialize_maintenance_workspace(task: TaskData) -> Path:
    """Write this task's maintenance workspace to a fresh temp dir and return it."""
    catalog_dir = materialize_catalog_dir(task)

    docs_dir = catalog_dir / "docs"
    docs_dir.mkdir()
    for name in ("database_overview.md", "tables", "knowledge_base"):
        src = catalog_dir / name
        if src.exists():
            shutil.move(str(src), str(docs_dir / name))

    (catalog_dir / "ISSUE.md").write_text(
        ISSUE_TEMPLATE.format(task_question=task.task_question), encoding="utf-8"
    )

    queries_dir = catalog_dir / "queries"
    queries_dir.mkdir()
    (queries_dir / "answer.sql").write_text(STUB_QUERY_CONTENT, encoding="utf-8")

    tests_dir = catalog_dir / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_contract.py").write_text(TEST_CONTRACT_REFERENCE, encoding="utf-8")

    return catalog_dir


def maintenance_tool_costs() -> dict[str, float]:
    """Bird-coin cost map the patience middleware consults for maintenance_agent.

    Derived from MA_TOOL_COSTS (tools/__init__.py), the single source of truth
    for per-tool costs, so this stays in sync with the prompt the agent reads.
    """
    return dict(MA_TOOL_COSTS)
