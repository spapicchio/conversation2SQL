"""Materialize one task's catalog into a temp dir for the deep_agent bash tool.

Layout (the bash tool's cwd):
  <tmp>/database_overview.md            (copied from disk if present, with any
                                          "## Knowledge Base" section replaced)
  <tmp>/tables/<table>.md               (copied verbatim)
  <tmp>/tables/_foreign_key_constraints.md
  <tmp>/knowledge_base/<node>.md        (re-rendered from masked_agent_kb)

The KB is re-rendered from `masked_agent_kb` (not read from disk), so masked
prerequisites never appear — faithful per-sample masking, no leak. Same for
the "## Knowledge Base" section of database_overview.md: the on-disk copy is
generated from the full, unmasked KB (a DB-level artifact), so its section is
stripped and replaced with one rendered fresh per task from `masked_agent_kb`.
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from conversation2sql.eval_framework.agents.bird_baseline.tools import TOOL_COSTS
from conversation2sql.eval_framework.agents.utils_kb_linearize import (
    build_kb_filenames,
    build_kb_overview,
    linearize_prerequisites,
)
from conversation2sql.eval_framework.state import TaskData


def _catalog_db_dir(task: TaskData) -> Path:
    return Path(task.deep_catalog_root) / task.selected_database


def materialize_catalog_dir(task: TaskData) -> Path:
    """Write this task's catalog to a fresh temp dir and return that dir."""
    db_dir = _catalog_db_dir(task)
    if not (db_dir / "tables").is_dir():
        raise FileNotFoundError(
            f"deep_agent tables catalog not found for database "
            f"'{task.selected_database}' at {db_dir / 'tables'}. Generate it with "
            f"scripts/generate_catalog.py (--database {task.selected_database} "
            f"--output-dir {task.deep_catalog_root})."
        )
    out = Path(tempfile.mkdtemp(prefix="deep_catalog_"))
    try:
        tables_out = out / "tables"
        tables_out.mkdir()
        for path in sorted((db_dir / "tables").glob("*.md")):
            shutil.copyfile(path, tables_out / path.name)

        filenames = build_kb_filenames(task.masked_agent_kb)

        overview = db_dir / "database_overview.md"
        if overview.is_file():
            # Drop any "## Knowledge Base" section already on disk (it would
            # be rendered from the full, unmasked KB) and replace it with one
            # rendered fresh from this task's masked KB, so a masked entry
            # never leaks through the disk copy.
            base, _, _ = overview.read_text(encoding="utf-8").partition(
                "\n## Knowledge Base"
            )
            base = base.rstrip("\n")
            kb_overview = build_kb_overview(task.masked_agent_kb, filenames)
            content = f"{base}\n\n## Knowledge Base\n{kb_overview}\n" if kb_overview else f"{base}\n"
            (out / "database_overview.md").write_text(content, encoding="utf-8")

        kb_out = out / "knowledge_base"
        kb_out.mkdir()
        for name in task.masked_agent_kb:
            content = linearize_prerequisites(name, task.masked_agent_kb)
            (kb_out / f"{filenames[name]}.md").write_text(content, encoding="utf-8")

        return out
    except BaseException:
        shutil.rmtree(out, ignore_errors=True)
        raise


def deep_tool_costs() -> dict[str, float]:
    """Bird-coin cost map the patience middleware consults for the deep_agent.

    `bash` is a flat read cost; submit/ask_user reuse the shared table."""
    return {
        "bash": 1.0,
        "submit_sql": TOOL_COSTS["submit_sql"],
        "ask_user": TOOL_COSTS["ask_user"],
    }
