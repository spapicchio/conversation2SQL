"""Abstract base class for agent tools."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseTool(ABC):
    """Base class that every tool in the agent pipeline must inherit from.

    Subclasses **must** provide concrete ``name`` and ``description``
    attributes and implement the :meth:`execute` method.
    """

    name: str
    description: str

    @abstractmethod
    def execute(self, **kwargs: Any) -> Any:
        """Run the tool with the supplied keyword arguments.

        Args:
            **kwargs: Tool-specific parameters.

        Returns:
            The tool's output (format depends on the concrete tool).
        """
