"""TurnClassifier: orchestrates Level 2 (rules) + Level 1 (LLM judge)."""
from __future__ import annotations

import json

from langchain_core.language_models import BaseChatModel

from conversation2sql.eval_framework.turn_classifier.prompts import build_judge_prompt
from conversation2sql.eval_framework.turn_classifier.rules import classify_level2
from conversation2sql.eval_framework.turn_classifier.schemas import (
    Level1Classification,
    TurnClassification,
)


def _extract_content_parts(content_items: list) -> tuple[str | None, str | None]:
    """Split content list into (thinking_text, visible_text)."""
    thinking_parts: list[str] = []
    text_parts: list[str] = []
    for item in content_items:
        if not isinstance(item, dict):
            continue
        t = item.get("type")
        text = item.get("text", "")
        if t == "thinking":
            thinking_parts.append(text)
        elif t == "text":
            text_parts.append(text)
    return (
        "\n".join(thinking_parts) or None,
        "\n".join(text_parts) or None,
    )


def _summarize_tool_calls(tool_calls: list[dict], max_chars: int = 300) -> str:
    if not tool_calls:
        return "(none)"
    parts = []
    for tc in tool_calls:
        name = tc.get("tool_name", "?")
        args = tc.get("arguments", {})
        args_str = json.dumps(args)[:80]
        parts.append(f"{name}({args_str})")
    return "; ".join(parts)[:max_chars]


class TurnClassifier:
    """Classify one AI turn at Level 2 (rules) and Level 1 (LLM judge)."""

    def __init__(self, model: BaseChatModel, tool_categories: dict[str, str]) -> None:
        self._tool_categories = tool_categories
        self._judge = model.with_structured_output(Level1Classification)

    def classify_turn(
        self,
        message_index: int,
        ai_msg: dict,
        prior_failed_submit: bool,
    ) -> TurnClassification:
        """Classify a single AI message dict from results_smaller.jsonl.

        Args:
            message_index: index of this message in the record's messages list.
            ai_msg: the AI message dict (keys: content, tool_calls, ...).
            prior_failed_submit: True if any earlier submit_sql call returned passed=False.
        """
        tool_calls: list[dict] = ai_msg.get("tool_calls", [])
        content_items: list = ai_msg.get("content", [])
        has_content = bool(content_items)

        level2_cat, tools_called = classify_level2(
            tool_calls, self._tool_categories, has_content=has_content
        )

        if level2_cat == "NO_ACTION":
            return TurnClassification(
                message_index=message_index,
                level2_category=level2_cat,
                level2_tools_called=tools_called,
                reasoning="Empty turn — no thinking, no text, no tool calls.",
                level1_category="MISSING",
                level1_alternatives=[],
                confidence="CERTAIN",
            )

        thinking, text = _extract_content_parts(content_items)
        tool_calls_summary = _summarize_tool_calls(tool_calls)

        messages = build_judge_prompt(
            params={
                "thinking": thinking,
                "text": text,
                "tool_calls_summary": tool_calls_summary,
                "level2_category": level2_cat,
                "prior_failed_submit": prior_failed_submit,
            }
        )

        level1: Level1Classification = self._judge.invoke(messages)  # pyrefly: ignore

        return TurnClassification(
            message_index=message_index,
            level2_category=level2_cat,
            level2_tools_called=tools_called,
            reasoning=level1.reasoning,
            level1_category=level1.level1_category,
            level1_alternatives=level1.level1_alternatives,
            confidence=level1.confidence,
        )
