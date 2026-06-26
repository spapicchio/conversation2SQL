"""Build the virtual-filesystem seed exposing the Postgres DB info to the deep agent.

Instead of get_schema-style tools, the deep_agent reads these files via the
deepagents FilesystemMiddleware (ls/read_file/grep/glob). One file per concern.
"""
from __future__ import annotations

from deepagents.middleware.filesystem import FileData

from conversation2sql.eval_framework.agents.bird_baseline.tools import (
    TOOL_COSTS,
    get_all_column_meanings_impl,
    get_all_knowledge_definitions_impl,
)
from conversation2sql.eval_framework.agents.utils_kb_linearize import linearize_kb
from conversation2sql.eval_framework.state import TaskData

DB_FS_PATHS: tuple[str, str, str] = (
    "/db/schema.sql",
    "/db/column_meanings.md",
    "/db/knowledge_base.md",
)

_NONE_SENTINEL = "(none)"


def _text_file(content: str) -> FileData:
    body = content if content and content.strip() else _NONE_SENTINEL
    return {"content": body, "encoding": "utf-8"}


def _render_column_meanings(task: TaskData) -> str:
    """One block per column: the `db|table|column` key followed by its JSON meaning."""
    meanings = get_all_column_meanings_impl(task.column_meanings)["column_meanings"]
    return "\n".join(f"{key}\n{value}" for key, value in meanings.items())


def _render_knowledge_base(task: TaskData) -> str:
    """Reuse the same KB rendering the bird tools use (linearized or JSON-dumped)."""
    if task.is_kb_linearized:
        return linearize_kb(task.masked_agent_kb)
    entries = get_all_knowledge_definitions_impl(task.masked_agent_kb)["knowledge"]
    return "\n\n".join(entries)


def build_db_filesystem(task: TaskData) -> dict[str, FileData]:
    """Render the three DB-info files seeded into the deep agent's filesystem."""
    return {
        "/db/schema.sql": _text_file(task.ddl_database_schema or ""),
        "/db/column_meanings.md": _text_file(_render_column_meanings(task)),
        "/db/knowledge_base.md": _text_file(_render_knowledge_base(task)),
    }


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
