"""Seed the deep agent's /db virtual filesystem from the on-disk catalog.

The deep_agent reads a per-database Markdown catalog (produced by
``scripts/generate_catalog.py``) through the deepagents FilesystemMiddleware.
Only the catalog folder of the task's own database is mounted, under /db:

  /db/tables/<table>.md                  (read from disk, as-is)
  /db/tables/_foreign_key_constraints.md (read from disk, as-is)
  /db/knowledge_base/<node>.md           (re-rendered per task, added in Task 3)
"""
from __future__ import annotations

from pathlib import Path

from deepagents.middleware.filesystem import FileData

from conversation2sql.eval_framework.agents.bird_baseline.tools import TOOL_COSTS
from conversation2sql.eval_framework.state import TaskData


def _text_file(content: str) -> FileData:
    return {"content": content, "encoding": "utf-8"}


def _catalog_db_dir(task: TaskData) -> Path:
    return Path(task.deep_catalog_root) / task.selected_database


def _seed_tables(db_dir: Path) -> dict[str, FileData]:
    """Mirror every <db_dir>/tables/*.md file at /db/tables/<name>, verbatim."""
    files: dict[str, FileData] = {}
    for path in sorted((db_dir / "tables").glob("*.md")):
        files[f"/db/tables/{path.name}"] = _text_file(path.read_text(encoding="utf-8"))
    return files


def build_db_filesystem(task: TaskData) -> dict[str, FileData]:
    """Render the /db filesystem for one task from its database's catalog folder."""
    db_dir = _catalog_db_dir(task)
    if not db_dir.is_dir():
        raise FileNotFoundError(
            f"deep_agent catalog not found for database '{task.selected_database}' "
            f"at {db_dir}. Generate it with scripts/generate_catalog.py "
            f"(--database {task.selected_database} --output-dir {task.deep_catalog_root})."
        )
    return _seed_tables(db_dir)


# Bird-coin costs for the deepagents filesystem read/write tools. Reads are cheap
# (static DB info); writes are only present under the deep_enable_fs_write ablation.
FS_TOOL_COSTS: dict[str, float] = {
    "ls": 0.5,
    "read_file": 0.5,
    "glob": 0.5,
    "grep": 0.5,
    "write_file": 0.5,
    "edit_file": 0.5,
}

_FS_READ_TOOLS = ("ls", "read_file", "glob", "grep")
_FS_WRITE_TOOLS = ("write_file", "edit_file")
# Reused tools the deep_agent always has; pull their costs from the shared table.
_REUSED_TOOLS = ("execute_sql", "ask_user", "submit_sql")


def deep_tool_costs(*, enable_fs_write: bool) -> dict[str, float]:
    """Cost map the patience middleware consults for the deep_agent's tool set."""
    names = list(_FS_READ_TOOLS)
    if enable_fs_write:
        names += list(_FS_WRITE_TOOLS)
    costs = {n: FS_TOOL_COSTS[n] for n in names}
    for n in _REUSED_TOOLS:
        costs[n] = TOOL_COSTS[n]
    return costs
