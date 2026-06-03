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


def test_kb_blind_fires_when_kb_present_and_unused():
    rec = _record(_ai(("execute_sql", {"sql": "x"})), _tool("execute_sql"))
    rec["masked_agent_kb"] = {"BFR": "Bandwidth ratio"}
    assert "kb_blind" in _hits(rec)


def test_kb_blind_quiet_without_kb():
    rec = _record(_ai(("execute_sql", {"sql": "x"})), _tool("execute_sql"))
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
