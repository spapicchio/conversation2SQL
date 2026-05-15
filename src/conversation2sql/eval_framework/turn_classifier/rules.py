"""Deterministic Level 2 classifier from tool_calls + YAML config."""
from __future__ import annotations

from pathlib import Path

import yaml

# Priority order — first match wins
_PRIORITY_ORDER = ["SQL_SUBMISSION", "USER_INTERACTION"]


def load_tool_categories(yaml_path: str | Path) -> dict[str, str]:
    """Load the tool_name → Level2Category mapping from a YAML file."""
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    return data.get("tool_categories", {})


def classify_level2(
    tool_calls: list[dict],
    tool_categories: dict[str, str],
    has_content: bool = True,
) -> tuple[str, list[str]]:
    """Return (level2_category, tools_called) for one AI message.

    Args:
        tool_calls: list of tool_call dicts, each with a 'tool_name' key.
        tool_categories: mapping of tool_name → category string (from YAML).
        has_content: True when the AI message has non-empty content items.

    Categories:
        SQL_SUBMISSION  — any submit_sql call present
        USER_INTERACTION — any ask_user call (no submit_sql)
        DB_EXPLORATION  — only DB-type tools
        KNOWLEDGE_LOOKUP — only KB-type tools
        MIXED           — tools span more than one category
        TEXT_ONLY       — no tools, non-empty content
        NO_ACTION       — no tools, empty content
        UNKNOWN         — a tool is present but not in the YAML config
    """
    tool_names = [tc.get("tool_name", "") for tc in tool_calls if tc.get("tool_name")]

    if not tool_names:
        return ("TEXT_ONLY" if has_content else "NO_ACTION"), []

    categories: set[str] = set()
    unknown_found = False
    for name in tool_names:
        cat = tool_categories.get(name)
        if cat is None:
            unknown_found = True
        else:
            categories.add(cat)

    if unknown_found:
        return "UNKNOWN", tool_names

    # Priority: SQL_SUBMISSION > USER_INTERACTION
    for priority_cat in _PRIORITY_ORDER:
        if priority_cat in categories:
            return priority_cat, tool_names

    if len(categories) == 1:
        return categories.pop(), tool_names

    return "MIXED", tool_names
