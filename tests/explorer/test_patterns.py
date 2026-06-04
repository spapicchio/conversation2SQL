from __future__ import annotations

from explorer.patterns import ToolEvent, extract_tool_events


def _ai(*tool_calls):
    """An ai message with the given (tool_name, arguments) tool calls."""
    return {
        "role": "ai",
        "tool_calls": [{"tool_name": n, "arguments": a} for n, a in tool_calls],
    }


def _tool(name, status="success", content="ok"):
    return {"role": "tool", "tool_name": name, "status": status, "content": content}


def _record(*messages):
    return {"messages": list(messages)}


def test_extract_pairs_single_call_with_result():
    rec = _record(
        {"role": "system", "content": "sys"},
        {"role": "human", "content": "q"},
        _ai(("get_schema", {})),
        _tool("get_schema", content={"schema": "..."}),
    )
    events = extract_tool_events(rec)
    assert len(events) == 1
    e = events[0]
    assert isinstance(e, ToolEvent)
    assert e.tool_name == "get_schema"
    assert e.message_index == 3
    assert e.status == "success"
    assert e.is_error is False


def test_extract_multi_call_turn_back_to_back_tool_messages():
    rec = _record(
        _ai(
            ("get_knowledge_definition", {"knowledge_name": "A"}),
            ("get_knowledge_definition", {"knowledge_name": "B"}),
        ),
        _tool("get_knowledge_definition"),
        _tool("get_knowledge_definition"),
    )
    events = extract_tool_events(rec)
    assert [e.tool_name for e in events] == [
        "get_knowledge_definition",
        "get_knowledge_definition",
    ]
    # Arguments matched positionally by name, in order.
    assert events[0].arguments == {"knowledge_name": "A"}
    assert events[1].arguments == {"knowledge_name": "B"}
    assert events[1].message_index == 2


def test_extract_marks_errors_and_extracts_text():
    rec = _record(
        _ai(("execute_sql", {"sql": "SELECT 1"})),
        _tool("execute_sql", status="error", content={"message": "syntax error"}),
    )
    events = extract_tool_events(rec)
    assert events[0].is_error is True
    assert "syntax error" in events[0].result_text


def test_extract_empty_when_no_tool_messages():
    rec = _record({"role": "ai", "content": "hi", "tool_calls": []})
    assert extract_tool_events(rec) == []


from explorer.patterns import PATTERN_CATALOG, PATTERN_NAMES, detect_patterns


def _hits(rec) -> set[str]:
    return {h.name for h in detect_patterns(rec)}


def test_catalog_names_match():
    assert PATTERN_NAMES == [name for name, _ in PATTERN_CATALOG]
    assert "blind_submit" in PATTERN_NAMES


def test_blind_submit_fires_without_execute():
    rec = _record(_ai(("submit_sql", {"sql": "SELECT 1"})), _tool("submit_sql"))
    assert "blind_submit" in _hits(rec)


def test_blind_submit_quiet_when_execute_precedes():
    rec = _record(
        _ai(("execute_sql", {"sql": "SELECT 1"})), _tool("execute_sql"),
        _ai(("submit_sql", {"sql": "SELECT 1"})), _tool("submit_sql"),
    )
    assert "blind_submit" not in _hits(rec)


def test_repeated_identical_call_ignores_whitespace_in_sql():
    rec = _record(
        _ai(("execute_sql", {"sql": "SELECT  1"})), _tool("execute_sql"),
        _ai(("execute_sql", {"sql": "SELECT 1"})), _tool("execute_sql"),
    )
    assert "repeated_identical_call" in _hits(rec)


def test_repeated_identical_call_quiet_when_args_differ():
    rec = _record(
        _ai(("execute_sql", {"sql": "SELECT 1"})), _tool("execute_sql"),
        _ai(("execute_sql", {"sql": "SELECT 2"})), _tool("execute_sql"),
    )
    assert "repeated_identical_call" not in _hits(rec)


def test_submit_after_error_fires_on_identical_failed_sql():
    rec = _record(
        _ai(("execute_sql", {"sql": "SELECT x"})),
        _tool("execute_sql", status="error", content="column x does not exist"),
        _ai(("submit_sql", {"sql": "SELECT x"})), _tool("submit_sql"),
    )
    assert "submit_after_error" in _hits(rec)


def test_unrecovered_error_loop_needs_three_consecutive():
    two = _record(
        _ai(("execute_sql", {"sql": "a"})), _tool("execute_sql", status="error"),
        _ai(("execute_sql", {"sql": "b"})), _tool("execute_sql", status="error"),
    )
    assert "unrecovered_error_loop" not in _hits(two)
    three = _record(
        _ai(("execute_sql", {"sql": "a"})), _tool("execute_sql", status="error"),
        _ai(("execute_sql", {"sql": "b"})), _tool("execute_sql", status="error"),
        _ai(("execute_sql", {"sql": "c"})), _tool("execute_sql", status="error"),
    )
    assert "unrecovered_error_loop" in _hits(three)


def test_unrecovered_error_loop_resets_on_success():
    rec = _record(
        _ai(("execute_sql", {"sql": "a"})), _tool("execute_sql", status="error"),
        _ai(("execute_sql", {"sql": "b"})), _tool("execute_sql", status="success"),
        _ai(("execute_sql", {"sql": "c"})), _tool("execute_sql", status="error"),
        _ai(("execute_sql", {"sql": "d"})), _tool("execute_sql", status="error"),
    )
    assert "unrecovered_error_loop" not in _hits(rec)


def test_kb_blind_fires_when_gt_kb_needed_and_unused():
    rec = _record(_ai(("execute_sql", {"sql": "x"})), _tool("execute_sql"))
    rec["gt_knowledge_base"] = {"BFR": "Bandwidth ratio"}
    assert "kb_blind" in _hits(rec)


def test_kb_blind_quiet_when_gt_kb_not_needed_even_if_masked_kb_present():
    # The full browsable KB is present, but the gold solution needs no KB entry,
    # so the task does not actually require external knowledge → not applicable.
    rec = _record(_ai(("execute_sql", {"sql": "x"})), _tool("execute_sql"))
    rec["masked_agent_kb"] = {"BFR": "Bandwidth ratio", "X": "y"}
    rec["gt_knowledge_base"] = {}
    assert "kb_blind" not in _hits(rec)


def test_kb_blind_quiet_when_gt_kb_queried():
    rec = _record(
        _ai(("get_knowledge_definition", {"knowledge_name": "BFR"})),
        _tool("get_knowledge_definition"),
    )
    rec["gt_knowledge_base"] = {"BFR": "Bandwidth ratio"}
    assert "kb_blind" not in _hits(rec)


def test_no_submission_fires_when_never_submitted():
    rec = _record(_ai(("execute_sql", {"sql": "x"})), _tool("execute_sql"))
    assert "no_submission" in _hits(rec)


def test_budget_death_fires_without_successful_submit_and_low_budget():
    rec = _record(_ai(("execute_sql", {"sql": "x"})), _tool("execute_sql"))
    rec["updated_user_patience"] = 0
    assert "budget_death" in _hits(rec)


def test_budget_death_quiet_after_successful_submit():
    rec = _record(_ai(("submit_sql", {"sql": "x"})), _tool("submit_sql", status="success"))
    rec["updated_user_patience"] = -2
    assert "budget_death" not in _hits(rec)


from explorer.patterns import (
    PatternStat,
    aggregate_patterns,
    clean_fraction,
    positional_tool_distribution,
)


def test_pattern_stat_fields():
    s = PatternStat(rate=0.25, applicable_n=8)
    assert s.rate == 0.25
    assert s.applicable_n == 8


def _submit_only(sql="SELECT 1"):
    # A clean validated submission: execute then submit, no anti-patterns.
    return _record(
        _ai(("execute_sql", {"sql": sql})), _tool("execute_sql"),
        _ai(("submit_sql", {"sql": sql})), _tool("submit_sql", status="success"),
    )


def test_aggregate_is_sample_average_not_pooled():
    # One instance, 2 samples: one blind-submits, one is clean.
    blind = _record(_ai(("submit_sql", {"sql": "x"})), _tool("submit_sql", status="success"))
    groups = {"inst1": [blind, _submit_only()]}
    freq = aggregate_patterns(groups)
    # Per-instance mean = 1/2 = 0.5 (not pooled over a flat list, but same here).
    assert freq["blind_submit"].rate == 0.5
    assert freq["no_submission"].rate == 0.0
    # Every sample is applicable to blind_submit → denominator is all samples.
    assert freq["blind_submit"].applicable_n == 2


def test_aggregate_weights_instances_equally():
    # inst_a: 2 samples both blind (1.0). inst_b: 1 sample clean (0.0).
    blind = _record(_ai(("submit_sql", {"sql": "x"})), _tool("submit_sql", status="success"))
    groups = {"a": [blind, dict(blind)], "b": [_submit_only()]}
    freq = aggregate_patterns(groups)
    # Mean over instances of per-instance rate = (1.0 + 0.0) / 2 = 0.5,
    # NOT pooled 2/3 = 0.667.
    assert freq["blind_submit"].rate == 0.5


def test_aggregate_kb_blind_denominator_only_counts_applicable_samples():
    # inst_a needs KB (gt non-empty) and ignores it → hit. inst_b needs no KB.
    needs_kb = _record(_ai(("execute_sql", {"sql": "x"})), _tool("execute_sql"))
    needs_kb["gt_knowledge_base"] = {"BFR": "ratio"}
    no_kb = _submit_only()
    no_kb["gt_knowledge_base"] = {}
    groups = {"a": [needs_kb], "b": [no_kb]}
    freq = aggregate_patterns(groups)
    # Only inst_a is applicable: rate = 1.0 over a single applicable sample,
    # and the non-applicable sample is excluded from the denominator entirely.
    assert freq["kb_blind"].rate == 1.0
    assert freq["kb_blind"].applicable_n == 1
    # A pattern applicable to everything still counts all samples.
    assert freq["no_submission"].applicable_n == 2


def test_clean_fraction_sample_average():
    blind = _record(_ai(("submit_sql", {"sql": "x"})), _tool("submit_sql", status="success"))
    groups = {"inst1": [blind, _submit_only()]}
    # 1 of 2 samples is clean.
    assert clean_fraction(groups) == 0.5


def test_positional_distribution_shares_sum_to_one_per_position():
    groups = {"inst1": [_submit_only()]}
    df = positional_tool_distribution(groups, max_pos=8)
    sums = df.groupby("position")["share"].sum()
    for pos, total in sums.items():
        assert abs(total - 1.0) < 1e-9, f"position {pos} sums to {total}"


def test_positional_distribution_no_call_band_appears_after_end():
    # 2-event sequence; positions 3..8 should be "(no call)".
    groups = {"inst1": [_submit_only()]}
    df = positional_tool_distribution(groups, max_pos=8)
    pos3 = df[(df["position"] == "3") & (df["tool"] == "(no call)")]["share"]
    assert float(pos3.iloc[0]) == 1.0
