"""LiteLLM-backed predictor implementation.

Uses :func:`litellm.acompletion` to call any LiteLLM-supported model and
translates the response into the unified :class:`ToolCallingPredictor`
output format.
"""

from __future__ import annotations

import json
from typing import Any

import litellm


class LiteLLMPredictor:
    """Predictor that delegates to LiteLLM for API-based inference.

    Args:
        model: Model identifier understood by LiteLLM
            (e.g. ``"gpt-4o"``, ``"anthropic/claude-3-sonnet"``).
        api_key: Optional API key forwarded to the provider.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens to generate.
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> None:
        self._model = model
        self._api_key = api_key
        self._temperature = temperature
        self._max_tokens = max_tokens

    async def generate(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Call LiteLLM and return a standardized response.

        Args:
            messages: Chat messages in OpenAI format.
            tools: Tool definitions in OpenAI function-calling schema.

        Returns:
            Unified dict with either ``"type": "text"`` or
            ``"type": "tool_call"``.
        """
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
        }
        if tools:
            kwargs["tools"] = tools
        if self._api_key is not None:
            kwargs["api_key"] = self._api_key

        response = await litellm.acompletion(**kwargs)

        choice = response.choices[0]  # type: ignore[union-attr]
        message = choice.message

        # Check for tool calls in the response
        if hasattr(message, "tool_calls") and message.tool_calls:
            tool_call = message.tool_calls[0]
            func = tool_call.function
            try:
                arguments = json.loads(func.arguments)
            except (json.JSONDecodeError, TypeError):
                arguments = {}
            return {
                "type": "tool_call",
                "tool_name": func.name,
                "arguments": arguments,
            }

        return {
            "type": "text",
            "content": message.content or "",
        }
