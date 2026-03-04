"""Custom Inspect AI solver using the ToolCallingPredictor protocol.

The solver extracts messages and tool definitions from the
:class:`inspect_ai.solver.TaskState`, passes them through any
:class:`ToolCallingPredictor` implementation, and appends the
result back to the task state.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from inspect_ai.solver import Solver, TaskState, solver

from src.predictors.protocol import ToolCallingPredictor


@solver
def predictor_solver(predictor: ToolCallingPredictor) -> Solver:
    """Create an Inspect AI solver backed by a :class:`ToolCallingPredictor`.

    Args:
        predictor: Any object satisfying the :class:`ToolCallingPredictor`
            protocol (LiteLLM, vLLM, HuggingFace, …).

    Returns:
        An Inspect AI :class:`Solver` callable.
    """

    async def _solve(state: TaskState, generate: Any) -> TaskState:
        """Run the predictor and update the task state.

        Args:
            state: Current Inspect AI task state.
            generate: Inspect AI generate callable (unused — we bring
                our own inference).

        Returns:
            Updated task state with the predictor's output appended.
        """
        # --- Extract messages in OpenAI format ---
        messages: list[dict[str, str]] = []
        for msg in state.messages:
            messages.append({
                "role": msg.role,
                "content": msg.text,
            })

        # --- Extract tool definitions from sample metadata ---
        tools: list[dict[str, Any]] = []
        if state.metadata and "tools" in state.metadata:
            tools = state.metadata["tools"]

        # --- Call the predictor ---
        result = await predictor.generate(messages, tools)

        # --- Append result to state ---
        from inspect_ai.model import ChatMessageAssistant

        if result.get("type") == "tool_call":
            content = json.dumps({
                "tool_name": result.get("tool_name", ""),
                "arguments": result.get("arguments", {}),
            })
        else:
            content = result.get("content", "")

        state.messages.append(ChatMessageAssistant(content=content))
        state.output = state.messages[-1]
        return state

    return _solve
