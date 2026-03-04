"""Standardized protocol for tool-calling predictors.

Defines :class:`ToolCallingPredictor`, a :pep:`544` ``Protocol`` that
decouples inference generation from evaluation logic.  Any backend
(LiteLLM, vLLM, HuggingFace, …) can implement this protocol and be
used interchangeably in the evaluation pipeline.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ToolCallingPredictor(Protocol):
    """Protocol that every tool-calling predictor must satisfy.

    The input format follows the HuggingFace / OpenAI chat-template
    standard so that implementations can be swapped transparently.
    """

    async def generate(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Generate a response given a conversation and tool definitions.

        Args:
            messages: Chat history in OpenAI format, e.g.
                ``[{"role": "user", "content": "What is the weather?"}]``.
            tools: JSON-schema tool definitions (OpenAI function-calling
                format).

        Returns:
            A dict with **one** of the following shapes:

            *Text response*::

                {"type": "text", "content": "<model reply>"}

            *Tool call*::

                {"type": "tool_call",
                 "tool_name": "<function name>",
                 "arguments": { ... }}
        """
        ...
