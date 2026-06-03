"""Deterministic (no-LLM) tool-interaction anti-pattern detection for result traces.

Operates on the normalized tool-event sequence extracted from each record's
``messages``. All aggregates are sample-average: averaged per instance first, then
across instances, mirroring ``loader.pass_at_1`` and ``metrics.reliability_metrics``.

This module must not import from ``explorer.loader`` (would be circular); it carries
its own small message-text helper instead.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field

import pandas as pd


def _message_text(content: object) -> str:
    """Best-effort text of a tool message's content (dict or str)."""
    if isinstance(content, dict):
        for key in ("content", "message", "error"):
            value = content.get(key)
            if isinstance(value, str):
                return value
        return json.dumps(content, ensure_ascii=False)
    if isinstance(content, str):
        return content
    return ""


@dataclass(frozen=True)
class ToolEvent:
    message_index: int   # index into record["messages"] of the tool-result message
    tool_name: str
    arguments: dict
    status: str          # "success" or an error status
    is_error: bool       # status != "success"
    result_text: str     # tool message content as text


def extract_tool_events(record: dict) -> list[ToolEvent]:
    """Walk messages, pairing each ai tool_call with its following tool result(s).

    When an ai turn has K tool_calls, the next K tool messages are its results.
    Arguments are matched to a result by tool name, in order (handles multi-call
    turns and back-to-back tool messages).
    """
    events: list[ToolEvent] = []
    pending: list[tuple[str, dict]] = []  # (tool_name, arguments) awaiting a result
    for idx, msg in enumerate(record.get("messages", [])):
        role = msg.get("role")
        if role == "ai":
            pending = [
                (tc.get("tool_name", ""), tc.get("arguments", {}) or {})
                for tc in (msg.get("tool_calls") or [])
            ]
        elif role == "tool":
            name = msg.get("tool_name", "")
            args: dict = {}
            for j, (pname, pargs) in enumerate(pending):
                if pname == name:
                    args = pargs
                    pending.pop(j)
                    break
            status = msg.get("status", "")
            events.append(
                ToolEvent(
                    message_index=idx,
                    tool_name=name,
                    arguments=args,
                    status=status,
                    is_error=status != "success",
                    result_text=_message_text(msg.get("content")),
                )
            )
    return events
