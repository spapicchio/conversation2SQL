"""Custom Inspect AI scorer for tool-calling evaluation.

Compares the predicted tool call against the expected target to
determine whether the prediction is correct.
"""

from __future__ import annotations

import json
from typing import Any

from inspect_ai.scorer import (
    Score,
    Scorer,
    Target,
    accuracy,
    scorer,
)
from inspect_ai.solver import TaskState


@scorer(metrics=[accuracy()])
def tool_call_scorer() -> Scorer:
    """Score a prediction by comparing it to the expected tool call.

    The target is expected to be a JSON string encoding a dict with
    ``"tool_name"`` and ``"arguments"`` keys.

    Returns:
        An Inspect AI :class:`Scorer` callable.
    """

    async def _score(state: TaskState, target: Target) -> Score:
        """Compare predicted output against the target.

        Args:
            state: Task state containing the model's output.
            target: Expected answer.

        Returns:
            A :class:`Score` with value ``"C"`` (correct) or ``"I"``
            (incorrect).
        """
        if state.output is None:
            return Score(value="I", explanation="No output produced.")

        predicted_text: str = state.output.text

        # Parse predicted output
        try:
            predicted = json.loads(predicted_text)
        except (json.JSONDecodeError, TypeError):
            predicted = {"content": predicted_text}

        # Parse expected target
        try:
            expected = json.loads(target.text)
        except (json.JSONDecodeError, TypeError):
            expected = {"content": target.text}

        # Compare tool calls
        if isinstance(predicted, dict) and isinstance(expected, dict):
            name_match = predicted.get("tool_name") == expected.get("tool_name")
            args_match = predicted.get("arguments") == expected.get("arguments")
            if name_match and args_match:
                return Score(
                    value="C",
                    explanation="Tool call matches.",
                )

        # Fallback: exact string match
        if predicted_text.strip() == target.text.strip():
            return Score(value="C", explanation="Exact text match.")

        return Score(
            value="I",
            explanation=(
                f"Mismatch. Predicted: {predicted_text!r}, "
                f"Expected: {target.text!r}"
            ),
        )

    return _score
