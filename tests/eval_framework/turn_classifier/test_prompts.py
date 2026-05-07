"""Tests for turn_classifier.prompts."""
from __future__ import annotations

from conversation2sql.eval_framework.turn_classifier.prompts import build_judge_prompt

ALL_LEVEL1_LABELS = [
    "ANSWER_ATTEMPT", "REVISION", "CLARIFICATION", "INTERROGATION",
    "ASSUMPTION", "CONFIRMATION", "DISCUSSION", "HEDGING", "REFUSAL", "MISSING",
]

ALL_CONFIDENCE_VALUES = ["CERTAIN", "CONFIDENT", "PLAUSIBLE", "UNCERTAIN"]


def _render(overrides: dict | None = None) -> list[dict]:
    params = {
        "thinking": "I should ask the user about the date range.",
        "text": "Could you clarify the date range?",
        "tool_calls_summary": "ask_user(question=Could you clarify the date range?)",
        "level2_category": "USER_INTERACTION",
        "prior_failed_submit": False,
    }
    if overrides:
        params.update(overrides)
    return build_judge_prompt(params)


def test_returns_list_of_dicts():
    msgs = _render()
    assert isinstance(msgs, list)
    assert all(isinstance(m, dict) for m in msgs)
    assert all("role" in m and "content" in m for m in msgs)


def test_has_system_and_user():
    msgs = _render()
    roles = [m["role"] for m in msgs]
    assert "system" in roles
    assert "user" in roles


def test_all_level1_labels_present():
    msgs = _render()
    full_text = " ".join(m["content"] for m in msgs)
    for label in ALL_LEVEL1_LABELS:
        assert label in full_text, f"Label {label!r} missing from prompt"


def test_all_confidence_values_present():
    msgs = _render()
    full_text = " ".join(m["content"] for m in msgs)
    for val in ALL_CONFIDENCE_VALUES:
        assert val in full_text, f"Confidence value {val!r} missing from prompt"


def test_thinking_included():
    msgs = _render()
    full_text = " ".join(m["content"] for m in msgs)
    assert "I should ask the user about the date range" in full_text


def test_prior_failed_submit_shown():
    msgs = _render({"prior_failed_submit": True})
    full_text = " ".join(m["content"] for m in msgs)
    assert "True" in full_text or "true" in full_text.lower()


def test_no_thinking_omits_thinking_section():
    msgs = _render({"thinking": None})
    full_text = " ".join(m["content"] for m in msgs)
    assert "I should ask the user about the date range" not in full_text


def test_level2_category_in_prompt():
    msgs = _render()
    full_text = " ".join(m["content"] for m in msgs)
    assert "USER_INTERACTION" in full_text
