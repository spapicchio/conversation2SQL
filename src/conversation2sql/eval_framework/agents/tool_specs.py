"""Single source of truth for per-tool metadata (cost + short description).

Each agent tool's bird-coin ``cost`` and its one-line ``summary`` live in one
``ToolSpec``. Both the agent system prompt's "Available tools and costs" list
and each tool's LLM-facing schema description are derived from these specs, so
the cost and wording cannot drift between the two places.

The per-file spec dicts (``DB_TOOL_SPECS`` / ``USER_TOOL_SPECS``) live next to
the tools they describe; ``tools/__init__.py`` merges them into ``TOOL_SPECS``
and derives the legacy ``TOOL_COSTS`` mapping from it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

# Substring used both to render the stamped cost line and to detect that a tool
# description has already been stamped (idempotency / no double-cost guard).
_COST_MARKER = "bird-coin"


@dataclass(frozen=True)
class ToolSpec:
    """Metadata for one agent tool.

    Attributes:
        name: The tool name (matches the ``@tool`` function name / ``tool.name``).
        cost: Bird-coin cost charged when the agent calls the tool.
        summary: One-line description rendered in the prompt's tool list. Should
            read naturally with a trailing ``". Cost: <n>"`` appended (i.e. no
            terminal period of its own).
    """

    name: str
    cost: float
    summary: str


def format_cost(cost: float) -> str:
    """Render a coin cost without a trailing ``.0`` (e.g. ``1.0`` -> ``"1"``)."""
    return str(int(cost)) if cost == int(cost) else str(cost)


def stamp_cost_in_descriptions(tools: Iterable, specs: Mapping[str, ToolSpec]):
    """Append each tool's spec cost to its ``.description`` (single-sourced).

    Mutates the tools in place so the LLM sees ``Cost: <n> bird-coins.`` in the
    tool schema without that number being hardcoded in the docstring. Idempotent
    and a no-op for any tool whose description already mentions bird-coins, so it
    is safe to call on module-level tool singletons and on freshly built tools.
    """
    for tool in tools:
        name = getattr(tool, "name", None)
        spec = specs.get(name) if name is not None else None
        if spec is None:
            continue
        description = tool.description or ""
        if _COST_MARKER in description:
            continue
        tool.description = (
            f"{description.rstrip()}\nCost: {format_cost(spec.cost)} bird-coins."
        )
    return tools
