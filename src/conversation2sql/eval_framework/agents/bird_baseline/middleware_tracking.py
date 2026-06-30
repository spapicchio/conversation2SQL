"""Post-hoc detection of built-in middleware activations from a message trace.

The built-in limit / context-editing middlewares keep their bookkeeping in
``PrivateStateAttr`` channels that LangGraph strips from the state returned by
``agent.invoke`` — only ``messages`` survives. They do, however, leave a
detectable footprint *in the message trace*:

- ``ToolCallLimitMiddleware`` (``exit_behavior="continue"``) injects a
  ``ToolMessage`` whose content starts with ``"Tool call limit exceeded."``.
- ``ModelCallLimitMiddleware`` (``exit_behavior="end"``) injects a final
  ``AIMessage`` whose content starts with ``"Model call limits exceeded:"``.
- ``ContextEditingMiddleware`` stamps cleared ``ToolMessage``s with
  ``response_metadata["context_editing"]["cleared"] = True``.

So tracking needs no custom middleware and no out-of-band recorder: it is a pure
function over the final ``messages`` list, which the pipeline already walks in
``utils_process_agent_response``. Each event anchors to the message id so the
explorer can place it on the right turn.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

# Single source of truth lives next to the tool/middleware specification. Matched
# by prefix so both tool-limit variants ("Do not call 'x' again." / "Do not make
# additional tool calls.") are covered.
from conversation2sql.eval_framework.agents.bird_baseline.tools import (
    MODEL_CALL_LIMIT_PREFIX,
    TOOL_CALL_LIMIT_PREFIX,
)


def _is_text(content: Any) -> str:
    return content if isinstance(content, str) else ""


def extract_middleware_events(messages: list[BaseMessage]) -> list[dict[str, Any]]:
    """Return one event per built-in middleware activation, in message order.

    Each event is a JSON-friendly ``{"type": ..., "message_id": ...}`` dict where
    ``type`` is one of ``"tool_call_limit"``, ``"model_call_limit"``,
    ``"context_editing"``.
    """
    events: list[dict[str, Any]] = []
    for message in messages:
        if isinstance(message, ToolMessage):
            if (message.response_metadata or {}).get("context_editing", {}).get(
                "cleared"
            ):
                events.append({"type": "context_editing", "message_id": message.id})
            if _is_text(message.content).startswith(TOOL_CALL_LIMIT_PREFIX):
                events.append({"type": "tool_call_limit", "message_id": message.id})
        elif isinstance(message, AIMessage):
            if _is_text(message.content).startswith(MODEL_CALL_LIMIT_PREFIX):
                events.append({"type": "model_call_limit", "message_id": message.id})
    return events
    
