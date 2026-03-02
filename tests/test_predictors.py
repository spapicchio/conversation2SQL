"""Tests for the tool-calling predictor pipeline.

All tests are designed to run on a CPU-only machine without real API
keys by mocking external services (LiteLLM, vLLM) and using tiny
models for HuggingFace.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.predictors.protocol import ToolCallingPredictor

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_MESSAGES: list[dict[str, str]] = [
    {"role": "user", "content": "What is the weather in Tokyo?"},
]

SAMPLE_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a location.",
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
]


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


class _MockPredictor:
    """Minimal predictor that satisfies the Protocol for testing."""

    async def generate(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return {
            "type": "tool_call",
            "tool_name": "get_weather",
            "arguments": {"location": "Tokyo"},
        }


def test_protocol_conformance() -> None:
    """A class with the right signature must be recognized as a ToolCallingPredictor."""
    predictor = _MockPredictor()
    assert isinstance(predictor, ToolCallingPredictor)


# ---------------------------------------------------------------------------
# LiteLLMPredictor – mocked
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_litellm_predictor_tool_call() -> None:
    """LiteLLMPredictor returns a tool call when the API responds with one."""
    # Build a mock response matching litellm's structure
    mock_func = MagicMock()
    mock_func.name = "get_weather"
    mock_func.arguments = json.dumps({"location": "Tokyo"})

    mock_tool_call = MagicMock()
    mock_tool_call.function = mock_func

    mock_message = MagicMock()
    mock_message.tool_calls = [mock_tool_call]
    mock_message.content = None

    mock_choice = MagicMock()
    mock_choice.message = mock_message

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_response):
        from src.predictors.litellm_predictor import LiteLLMPredictor

        predictor = LiteLLMPredictor(model="gpt-4o")
        result = await predictor.generate(SAMPLE_MESSAGES, SAMPLE_TOOLS)

    assert result["type"] == "tool_call"
    assert result["tool_name"] == "get_weather"
    assert result["arguments"] == {"location": "Tokyo"}


@pytest.mark.asyncio
async def test_litellm_predictor_text_response() -> None:
    """LiteLLMPredictor returns text when no tool call is present."""
    mock_message = MagicMock()
    mock_message.tool_calls = None
    mock_message.content = "The weather in Tokyo is sunny."

    mock_choice = MagicMock()
    mock_choice.message = mock_message

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_response):
        from src.predictors.litellm_predictor import LiteLLMPredictor

        predictor = LiteLLMPredictor(model="gpt-4o")
        result = await predictor.generate(SAMPLE_MESSAGES, [])

    assert result["type"] == "text"
    assert result["content"] == "The weather in Tokyo is sunny."


# ---------------------------------------------------------------------------
# VLLMPredictor – fully mocked
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_vllm_predictor_tool_call() -> None:
    """VLLMPredictor parses a JSON tool call from vLLM output."""
    mock_output_text = json.dumps({
        "name": "get_weather",
        "arguments": {"location": "Tokyo"},
    })

    mock_output = MagicMock()
    mock_output.text = mock_output_text

    mock_request_output = MagicMock()
    mock_request_output.outputs = [mock_output]

    mock_tokenizer = MagicMock()
    mock_tokenizer.apply_chat_template.return_value = "<formatted prompt>"

    mock_llm = MagicMock()
    mock_llm.generate.return_value = [mock_request_output]
    mock_llm.get_tokenizer.return_value = mock_tokenizer

    from src.predictors.vllm_predictor import VLLMPredictor

    predictor = VLLMPredictor.__new__(VLLMPredictor)
    predictor._llm = mock_llm
    predictor._tokenizer = mock_tokenizer
    predictor._sampling_params = MagicMock()

    result = await predictor.generate(SAMPLE_MESSAGES, SAMPLE_TOOLS)

    assert result["type"] == "tool_call"
    assert result["tool_name"] == "get_weather"
    assert result["arguments"] == {"location": "Tokyo"}
    mock_tokenizer.apply_chat_template.assert_called_once()


@pytest.mark.asyncio
async def test_vllm_predictor_text_fallback() -> None:
    """VLLMPredictor returns text when the output is not valid JSON."""
    mock_output = MagicMock()
    mock_output.text = "I cannot determine that."

    mock_request_output = MagicMock()
    mock_request_output.outputs = [mock_output]

    mock_tokenizer = MagicMock()
    mock_tokenizer.apply_chat_template.return_value = "<prompt>"

    mock_llm = MagicMock()
    mock_llm.generate.return_value = [mock_request_output]

    from src.predictors.vllm_predictor import VLLMPredictor

    predictor = VLLMPredictor.__new__(VLLMPredictor)
    predictor._llm = mock_llm
    predictor._tokenizer = mock_tokenizer
    predictor._sampling_params = MagicMock()

    result = await predictor.generate(SAMPLE_MESSAGES, [])

    assert result["type"] == "text"
    assert result["content"] == "I cannot determine that."


# ---------------------------------------------------------------------------
# _parse_tool_call edge cases
# ---------------------------------------------------------------------------


def test_parse_tool_call_malformed_json() -> None:
    """Malformed JSON falls back to text."""
    from src.predictors.parsing import parse_tool_call

    result = parse_tool_call("{not valid json}")
    assert result["type"] == "text"


def test_parse_tool_call_no_braces() -> None:
    """Plain text without braces falls back to text."""
    from src.predictors.parsing import parse_tool_call

    result = parse_tool_call("just plain text")
    assert result["type"] == "text"
    assert result["content"] == "just plain text"


def test_parse_tool_call_json_without_tool_name() -> None:
    """JSON without a tool name falls back to text."""
    from src.predictors.parsing import parse_tool_call

    result = parse_tool_call('{"arguments": {"x": 1}}')
    assert result["type"] == "text"


def test_parse_tool_call_string_arguments() -> None:
    """String arguments are JSON-parsed when possible."""
    from src.predictors.parsing import parse_tool_call

    text = json.dumps({
        "name": "my_tool",
        "arguments": json.dumps({"key": "val"}),
    })
    result = parse_tool_call(text)
    assert result["type"] == "tool_call"
    assert result["arguments"] == {"key": "val"}


# ---------------------------------------------------------------------------
# Inspect AI integration – mocked
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_inspect_solver_and_scorer() -> None:
    """End-to-end test of the Inspect AI solver + scorer with a mock predictor."""
    from inspect_ai.model import ChatMessageUser, ChatMessageAssistant
    from inspect_ai.scorer import Target
    from inspect_ai.solver import TaskState

    expected = {"tool_name": "get_weather", "arguments": {"location": "Tokyo"}}

    # Build a mock predictor
    mock_predictor = _MockPredictor()

    # Build a solver
    from src.inspect_task.solver import predictor_solver

    solve_fn = predictor_solver(mock_predictor)

    # Build a minimal TaskState
    state = TaskState(
        model="mockmodel",
        sample_id="test-1",
        epoch=1,
        messages=[ChatMessageUser(content="What is the weather in Tokyo?")],
        input=[ChatMessageUser(content="What is the weather in Tokyo?")],
        output=ChatMessageAssistant(content=""),
        metadata={"tools": SAMPLE_TOOLS},
    )

    # Run solver
    updated_state = await solve_fn(state, generate=MagicMock())

    # Verify solver appended the response
    assert len(updated_state.messages) == 2
    predicted_text = updated_state.output.text

    # Run scorer
    from src.inspect_task.scorer import tool_call_scorer

    score_fn = tool_call_scorer()
    target = Target(json.dumps(expected))
    score = await score_fn(updated_state, target)

    assert score.value == "C"


@pytest.mark.asyncio
async def test_inspect_scorer_mismatch() -> None:
    """Scorer returns 'I' when tool names don't match."""
    from inspect_ai.model import ChatMessageUser, ChatMessageAssistant
    from inspect_ai.scorer import Target
    from inspect_ai.solver import TaskState

    state = TaskState(
        model="mockmodel",
        sample_id="test-2",
        epoch=1,
        messages=[ChatMessageUser(content="q")],
        input=[ChatMessageUser(content="q")],
        output=ChatMessageAssistant(
            content=json.dumps({"tool_name": "wrong", "arguments": {}})
        ),
        metadata={},
    )

    from src.inspect_task.scorer import tool_call_scorer

    score_fn = tool_call_scorer()
    target = Target(json.dumps({"tool_name": "get_weather", "arguments": {"location": "Tokyo"}}))
    score = await score_fn(state, target)

    assert score.value == "I"
