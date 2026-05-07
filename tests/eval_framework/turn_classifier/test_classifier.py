"""Unit tests for TurnClassifier (LLM mocked)."""
from __future__ import annotations

from unittest.mock import MagicMock

from conversation2sql.eval_framework.turn_classifier.classifier import TurnClassifier
from conversation2sql.eval_framework.turn_classifier.schemas import (
    Level1Classification,
    TurnClassification,
)

TOOL_CATEGORIES = {
    "submit_sql": "SQL_SUBMISSION",
    "ask_user": "USER_INTERACTION",
    "execute_sql": "DB_EXPLORATION",
    "get_knowledge_definition": "KNOWLEDGE_LOOKUP",
}


def _make_classifier(level1: Level1Classification) -> TurnClassifier:
    structured = MagicMock()
    structured.invoke.return_value = level1
    model = MagicMock()
    model.with_structured_output.return_value = structured
    return TurnClassifier(model=model, tool_categories=TOOL_CATEGORIES)


def _l1(**kwargs) -> Level1Classification:
    defaults = dict(
        reasoning="some reasoning",
        level1_category="DISCUSSION",
        level1_alternatives=[],
        confidence="CONFIDENT",
    )
    return Level1Classification(**{**defaults, **kwargs})


# --- result type ---


def test_returns_turn_classification():
    clf = _make_classifier(_l1())
    result = clf.classify_turn(
        message_index=2,
        ai_msg={"content": [{"type": "thinking", "text": "hmm"}], "tool_calls": []},
        prior_failed_submit=False,
    )
    assert isinstance(result, TurnClassification)


# --- level 2 wiring ---


def test_sql_submission_level2():
    clf = _make_classifier(_l1(level1_category="ANSWER_ATTEMPT", confidence="CERTAIN"))
    result = clf.classify_turn(
        message_index=5,
        ai_msg={
            "content": [{"type": "thinking", "text": "I'll submit."}],
            "tool_calls": [{"tool_name": "submit_sql", "arguments": {"sql": "SELECT 1"}, "tool_cost": 0}],
        },
        prior_failed_submit=False,
    )
    assert result.level2_category == "SQL_SUBMISSION"
    assert result.level2_tools_called == ["submit_sql"]


def test_user_interaction_level2():
    clf = _make_classifier(_l1(level1_category="CLARIFICATION"))
    result = clf.classify_turn(
        message_index=3,
        ai_msg={
            "content": [{"type": "thinking", "text": "I need to ask."}],
            "tool_calls": [{"tool_name": "ask_user", "arguments": {"question": "What do you mean?"}, "tool_cost": 0}],
        },
        prior_failed_submit=False,
    )
    assert result.level2_category == "USER_INTERACTION"


def test_no_action_empty_turn():
    clf = _make_classifier(_l1(level1_category="MISSING", confidence="CERTAIN"))
    result = clf.classify_turn(
        message_index=10,
        ai_msg={"content": [], "tool_calls": []},
        prior_failed_submit=False,
    )
    assert result.level2_category == "NO_ACTION"
    assert result.level2_tools_called == []


def test_text_only_level2():
    clf = _make_classifier(_l1(level1_category="DISCUSSION"))
    result = clf.classify_turn(
        message_index=7,
        ai_msg={
            "content": [{"type": "text", "text": "Let me think about this."}],
            "tool_calls": [],
        },
        prior_failed_submit=False,
    )
    assert result.level2_category == "TEXT_ONLY"


# --- level 1 wiring ---


def test_level1_fields_passed_through():
    clf = _make_classifier(_l1(
        reasoning="Clearly asking one question.",
        level1_category="CLARIFICATION",
        level1_alternatives=[],
        confidence="CONFIDENT",
    ))
    result = clf.classify_turn(
        message_index=1,
        ai_msg={"content": [], "tool_calls": []},
        prior_failed_submit=False,
    )
    assert result.reasoning == "Clearly asking one question."
    assert result.level1_category == "CLARIFICATION"
    assert result.confidence == "CONFIDENT"
    assert result.level1_alternatives == []


def test_alternatives_passed_through():
    clf = _make_classifier(_l1(
        level1_category="REVISION",
        level1_alternatives=["ANSWER_ATTEMPT"],
        confidence="PLAUSIBLE",
    ))
    result = clf.classify_turn(
        message_index=8,
        ai_msg={"content": [], "tool_calls": []},
        prior_failed_submit=True,
    )
    assert result.level1_alternatives == ["ANSWER_ATTEMPT"]


# --- message_index ---


def test_message_index_preserved():
    clf = _make_classifier(_l1())
    result = clf.classify_turn(
        message_index=42,
        ai_msg={"content": [], "tool_calls": []},
        prior_failed_submit=False,
    )
    assert result.message_index == 42
