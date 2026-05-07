"""Tests for turn_classifier.schemas."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from conversation2sql.eval_framework.turn_classifier.schemas import (
    Level1Classification,
    TurnClassification,
)


def test_level1_classification_valid():
    obj = Level1Classification(
        reasoning="The agent asks one focused question.",
        level1_category="CLARIFICATION",
        level1_alternatives=[],
        confidence="CONFIDENT",
    )
    assert obj.level1_category == "CLARIFICATION"
    assert obj.level1_alternatives == []


def test_level1_classification_with_alternatives():
    obj = Level1Classification(
        reasoning="Could be CLARIFICATION or INTERROGATION.",
        level1_category="CLARIFICATION",
        level1_alternatives=["INTERROGATION"],
        confidence="PLAUSIBLE",
    )
    assert "INTERROGATION" in obj.level1_alternatives


def test_level1_invalid_confidence():
    with pytest.raises(ValidationError):
        Level1Classification(
            reasoning="...",
            level1_category="CLARIFICATION",
            level1_alternatives=[],
            confidence="MAYBE",  # not in Literal
        )


def test_turn_classification_valid():
    obj = TurnClassification(
        message_index=3,
        level2_category="USER_INTERACTION",
        level2_tools_called=["ask_user"],
        reasoning="Asks a single question.",
        level1_category="CLARIFICATION",
        level1_alternatives=[],
        confidence="CONFIDENT",
    )
    assert obj.message_index == 3
    assert obj.level2_category == "USER_INTERACTION"
