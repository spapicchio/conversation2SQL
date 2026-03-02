"""Inspect AI task definition for tool-calling evaluation.

Wires together the dataset, the predictor solver, and the scorer into
a single :func:`inspect_ai.Task`.
"""

from __future__ import annotations

import json
from typing import Any

from inspect_ai import Task, task
from inspect_ai.dataset import Sample, MemoryDataset

from src.inspect_task.scorer import tool_call_scorer
from src.inspect_task.solver import predictor_solver
from src.predictors.protocol import ToolCallingPredictor


def build_dataset(samples: list[dict[str, Any]]) -> MemoryDataset:
    """Convert a list of raw sample dicts into an Inspect AI dataset.

    Each sample dict is expected to have:

    - ``"messages"``: list of ``{"role": ..., "content": ...}`` dicts.
    - ``"tools"``: list of tool-schema dicts.
    - ``"expected"``: dict with ``"tool_name"`` and ``"arguments"``.

    Args:
        samples: Raw evaluation samples.

    Returns:
        An :class:`inspect_ai.dataset.MemoryDataset`.
    """
    inspect_samples: list[Sample] = []
    for s in samples:
        user_msg = ""
        for msg in s.get("messages", []):
            if msg.get("role") == "user":
                user_msg = msg.get("content", "")
                break

        inspect_samples.append(
            Sample(
                input=user_msg,
                target=json.dumps(s["expected"]),
                metadata={
                    "messages": s.get("messages", []),
                    "tools": s.get("tools", []),
                },
            )
        )

    return MemoryDataset(samples=inspect_samples)


@task
def tool_calling_task(
    predictor: ToolCallingPredictor,
    samples: list[dict[str, Any]] | None = None,
) -> Task:
    """Create an Inspect AI task for tool-calling evaluation.

    Args:
        predictor: Any :class:`ToolCallingPredictor` implementation.
        samples: Optional list of evaluation sample dicts.  When
            ``None`` a small built-in example set is used.

    Returns:
        A configured :class:`inspect_ai.Task`.
    """
    if samples is None:
        samples = [
            {
                "messages": [
                    {"role": "user", "content": "What is the weather in Tokyo?"},
                ],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "description": "Get the current weather",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "location": {
                                        "type": "string",
                                        "description": "City name",
                                    },
                                },
                                "required": ["location"],
                            },
                        },
                    },
                ],
                "expected": {
                    "tool_name": "get_weather",
                    "arguments": {"location": "Tokyo"},
                },
            },
        ]

    dataset = build_dataset(samples)

    return Task(
        dataset=dataset,
        solver=predictor_solver(predictor),
        scorer=tool_call_scorer(),
    )
