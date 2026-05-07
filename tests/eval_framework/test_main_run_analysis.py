from __future__ import annotations

import json
from collections import Counter
from unittest.mock import MagicMock

from conversation2sql.eval_framework.turn_classifier.schemas import TurnClassification


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_tc(
    index: int = 0,
    l2: str = "TEXT_ONLY",
    l1: str = "DISCUSSION",
    confidence: str = "CERTAIN",
) -> TurnClassification:
    return TurnClassification(
        message_index=index,
        level2_category=l2,
        level2_tools_called=[],
        reasoning="test",
        level1_category=l1,
        level1_alternatives=[],
        confidence=confidence,
    )


# ── _iter_records ─────────────────────────────────────────────────────────────

def test_iter_records_compact_jsonl(tmp_path):
    from conversation2sql.eval_framework.main_run_analysis import _iter_records

    p = tmp_path / "test.jsonl"
    records = [{"id": 1, "v": "a"}, {"id": 2, "v": "b"}]
    p.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    assert list(_iter_records(p)) == records


def test_iter_records_pretty_json(tmp_path):
    from conversation2sql.eval_framework.main_run_analysis import _iter_records

    p = tmp_path / "test.jsonl"
    records = [{"id": 1}, {"id": 2}]
    p.write_text(json.dumps(records[0], indent=2) + "\n" + json.dumps(records[1], indent=2))
    assert list(_iter_records(p)) == records


def test_iter_records_empty_file(tmp_path):
    from conversation2sql.eval_framework.main_run_analysis import _iter_records

    p = tmp_path / "empty.jsonl"
    p.write_text("")
    assert list(_iter_records(p)) == []


# ── classify_record ───────────────────────────────────────────────────────────

def test_classify_record_single_ai_turn():
    from conversation2sql.eval_framework.main_run_analysis import classify_record

    tc = _make_tc(index=0)
    mock_classifier = MagicMock()
    mock_classifier.classify_turn.return_value = tc

    record = {
        "instance_id": "q1",
        "messages": [
            {"role": "ai", "content": [{"type": "text", "text": "hello"}], "tool_calls": []},
            {"role": "human", "content": "ok"},
        ],
    }
    enriched, classifications = classify_record(record, mock_classifier)

    assert classifications == [tc]
    assert enriched["turn_classifications"] == [tc.model_dump()]
    mock_classifier.classify_turn.assert_called_once_with(
        message_index=0,
        ai_msg=record["messages"][0],
        prior_failed_submit=False,
    )


def test_classify_record_prior_failed_submit_flag():
    from conversation2sql.eval_framework.main_run_analysis import classify_record

    tc_first = _make_tc(index=0)
    tc_second = _make_tc(index=2)
    mock_classifier = MagicMock()
    mock_classifier.classify_turn.side_effect = [tc_first, tc_second]

    record = {
        "instance_id": "q2",
        "messages": [
            {"role": "ai", "content": [], "tool_calls": []},
            {
                "role": "tool",
                "tool_name": "submit_sql",
                "content": {"passed": False},
            },
            {"role": "ai", "content": [], "tool_calls": []},
        ],
    }
    _, classifications = classify_record(record, mock_classifier)

    assert len(classifications) == 2
    # second call must see prior_failed_submit=True
    second_call = mock_classifier.classify_turn.call_args_list[1]
    assert second_call.kwargs["prior_failed_submit"] is True


def test_classify_record_no_ai_turns():
    from conversation2sql.eval_framework.main_run_analysis import classify_record

    mock_classifier = MagicMock()
    record = {"instance_id": "q3", "messages": [{"role": "human", "content": "hi"}]}
    enriched, classifications = classify_record(record, mock_classifier)

    assert classifications == []
    assert enriched["turn_classifications"] == []
    mock_classifier.classify_turn.assert_not_called()


# ── _update_summary ───────────────────────────────────────────────────────────

def test_update_summary_increments_all_counters():
    from conversation2sql.eval_framework.main_run_analysis import (
        AnalysisSummary,
        _update_summary,
    )

    summary = AnalysisSummary()
    tc = _make_tc(l2="TEXT_ONLY", l1="DISCUSSION", confidence="CERTAIN")
    _update_summary(summary, "q1", [tc])

    assert summary.l2_counts["TEXT_ONLY"] == 1
    assert summary.l1_counts["DISCUSSION"] == 1
    assert summary.confidence_counts["CERTAIN"] == 1
    assert "q1" in summary.per_instance
    assert summary.per_instance["q1"].n_turns == 1
    assert summary.per_instance["q1"].l1_counts["DISCUSSION"] == 1


def test_update_summary_accumulates_across_calls():
    from conversation2sql.eval_framework.main_run_analysis import (
        AnalysisSummary,
        _update_summary,
    )

    summary = AnalysisSummary()
    _update_summary(summary, "q1", [_make_tc(l1="DISCUSSION"), _make_tc(l1="CLARIFICATION")])
    _update_summary(summary, "q1", [_make_tc(l1="DISCUSSION")])

    assert summary.l1_counts["DISCUSSION"] == 2
    assert summary.l1_counts["CLARIFICATION"] == 1
    assert summary.per_instance["q1"].n_turns == 3


def test_update_summary_separate_instances():
    from conversation2sql.eval_framework.main_run_analysis import (
        AnalysisSummary,
        _update_summary,
    )

    summary = AnalysisSummary()
    _update_summary(summary, "q1", [_make_tc(l1="DISCUSSION")])
    _update_summary(summary, "q2", [_make_tc(l1="CLARIFICATION")])

    assert "q1" in summary.per_instance
    assert "q2" in summary.per_instance
    assert summary.per_instance["q1"].n_turns == 1
    assert summary.per_instance["q2"].n_turns == 1
    assert summary.l1_counts["DISCUSSION"] == 1
    assert summary.l1_counts["CLARIFICATION"] == 1
