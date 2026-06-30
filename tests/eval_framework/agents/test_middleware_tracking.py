"""Tests for post-hoc middleware-activation extraction from a message trace."""

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from conversation2sql.eval_framework.agents.bird_baseline.middleware_tracking import (
    extract_middleware_events,
)


def test_no_activations_returns_empty():
    messages = [
        HumanMessage(content="go"),
        AIMessage(content="", id="ai-1"),
        ToolMessage(content="pong", tool_call_id="c1", name="ping", id="t-1"),
    ]
    assert extract_middleware_events(messages) == []


def test_tool_call_limit_detected():
    messages = [
        ToolMessage(
            content="Tool call limit exceeded. Do not make additional tool calls.",
            tool_call_id="c1",
            name="ping",
            id="t-1",
        ),
    ]
    events = extract_middleware_events(messages)
    assert events == [{"type": "tool_call_limit", "message_id": "t-1"}]


def test_model_call_limit_detected():
    messages = [
        AIMessage(content="Model call limits exceeded: run limit (3/3)", id="ai-9"),
    ]
    events = extract_middleware_events(messages)
    assert events == [{"type": "model_call_limit", "message_id": "ai-9"}]


def test_context_editing_detected():
    cleared = ToolMessage(
        content="[cleared]",
        tool_call_id="c1",
        name="ping",
        id="t-7",
        response_metadata={"context_editing": {"cleared": True}},
    )
    events = extract_middleware_events([cleared])
    assert events == [{"type": "context_editing", "message_id": "t-7"}]


def test_context_editing_not_flagged_when_metadata_absent():
    not_cleared = ToolMessage(
        content="pong",
        tool_call_id="c1",
        name="ping",
        id="t-8",
        response_metadata={"context_editing": {}},
    )
    assert extract_middleware_events([not_cleared]) == []


def test_multiple_activations_in_order():
    messages = [
        ToolMessage(
            content="pong",
            tool_call_id="c0",
            name="ping",
            id="t-0",
            response_metadata={"context_editing": {"cleared": True}},
        ),
        ToolMessage(
            content="Tool call limit exceeded. Do not call 'ping' again.",
            tool_call_id="c1",
            name="ping",
            id="t-1",
        ),
        AIMessage(content="Model call limits exceeded: run limit (3/3)", id="ai-2"),
    ]
    events = extract_middleware_events(messages)
    assert events == [
        {"type": "context_editing", "message_id": "t-0"},
        {"type": "tool_call_limit", "message_id": "t-1"},
        {"type": "model_call_limit", "message_id": "ai-2"},
    ]
