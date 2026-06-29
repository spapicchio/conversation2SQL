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


def test_blind_submit_quiet_when_psql_console_query_precedes():
    # In the psql_console ablation the query is validated by running SQL in the
    # terminal, not via execute_sql.
    rec = _record(
        _ai(("psql_console", {"command": "SELECT 1"})), _tool("psql_console"),
        _ai(("submit_sql", {"sql": "SELECT 1"})), _tool("submit_sql"),
    )
    assert "blind_submit" not in _hits(rec)


def test_blind_submit_fires_when_only_psql_meta_command_precedes():
    # A backslash meta-command (\dt, \d …) inspects schema but never runs the
    # query, so it does not count as validation.
    rec = _record(
        _ai(("psql_console", {"command": "\\dt"})), _tool("psql_console"),
        _ai(("submit_sql", {"sql": "SELECT 1"})), _tool("submit_sql"),
    )
    assert "blind_submit" in _hits(rec)


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


def test_submit_after_error_fires_on_psql_console_error():
    # psql_console is used instead of execute_sql in the psql ablation; the
    # detector must recognise the same SQL erroring there before submission.
    rec = _record(
        _ai(("psql_console", {"command": "SELECT x FROM t"})),
        _tool("psql_console", status="error", content="column x does not exist"),
        _ai(("submit_sql", {"sql": "SELECT x FROM t"})), _tool("submit_sql"),
    )
    assert "submit_after_error" in _hits(rec)


def test_submit_after_error_quiet_for_psql_meta_command_error():
    # A backslash meta-command does not run the query, so an error on \d t
    # should not be linked to a later submission of any SQL.
    rec = _record(
        _ai(("psql_console", {"command": "\\d t"})),
        _tool("psql_console", status="error", content="table t does not exist"),
        _ai(("submit_sql", {"sql": "SELECT * FROM t"})), _tool("submit_sql"),
    )
    assert "submit_after_error" not in _hits(rec)


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


def test_unrecovered_error_loop_fires_on_psql_console_errors():
    # Three consecutive psql_console SQL errors (no execute_sql) must fire.
    rec = _record(
        _ai(("psql_console", {"command": "SELECT a"})), _tool("psql_console", status="error"),
        _ai(("psql_console", {"command": "SELECT b"})), _tool("psql_console", status="error"),
        _ai(("psql_console", {"command": "SELECT c"})), _tool("psql_console", status="error"),
    )
    assert "unrecovered_error_loop" in _hits(rec)


def test_unrecovered_error_loop_fires_on_mixed_execute_and_psql_errors():
    rec = _record(
        _ai(("execute_sql", {"sql": "a"})), _tool("execute_sql", status="error"),
        _ai(("psql_console", {"command": "SELECT b"})), _tool("psql_console", status="error"),
        _ai(("execute_sql", {"sql": "c"})), _tool("execute_sql", status="error"),
    )
    assert "unrecovered_error_loop" in _hits(rec)


def test_unrecovered_error_loop_psql_meta_command_does_not_extend_or_reset_streak():
    # A \dt meta-command in between two real SQL errors must be ignored —
    # it neither increments the streak nor resets it.
    rec = _record(
        _ai(("psql_console", {"command": "SELECT a"})), _tool("psql_console", status="error"),
        _ai(("psql_console", {"command": "\\dt"})), _tool("psql_console"),
        _ai(("psql_console", {"command": "SELECT b"})), _tool("psql_console", status="error"),
        _ai(("psql_console", {"command": "SELECT c"})), _tool("psql_console", status="error"),
    )
    assert "unrecovered_error_loop" in _hits(rec)


def test_unrecovered_error_loop_resets_on_psql_console_success():
    rec = _record(
        _ai(("psql_console", {"command": "SELECT a"})), _tool("psql_console", status="error"),
        _ai(("psql_console", {"command": "SELECT b"})), _tool("psql_console", status="error"),
        _ai(("psql_console", {"command": "SELECT ok"})), _tool("psql_console", status="success"),
        _ai(("psql_console", {"command": "SELECT c"})), _tool("psql_console", status="error"),
        _ai(("psql_console", {"command": "SELECT d"})), _tool("psql_console", status="error"),
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


# Verbatim shape of the message the budget middleware emits when it blocks a
# tool call (`tool_wrapper_patience_and_submit` in agent_callback.py).
_BLOCKED_TEXT = "Budget exhausted (1.0 remaining). You MUST call submit_sql now with your best SQL."


def test_budget_death_quiet_when_answer_passed():
    rec = _record(_ai(("submit_sql", {"sql": "x"})), _tool("submit_sql", status="success"))
    rec["execution_accuracy"] = True
    rec["updated_user_patience"] = -2
    assert "budget_death" not in _hits(rec)


def test_budget_death_fires_on_failed_forced_submit():
    # The classic death: budget-blocked tool → forced submit → SQL fails. The
    # submit's *tool status* is "success" (the tool ran fine), but no passing
    # answer landed, so this is still a budget death.
    rec = _record(
        _ai(("execute_sql", {"sql": "x"})),
        _tool("execute_sql", content=_BLOCKED_TEXT),
        _ai(("submit_sql", {"sql": "x"})),
        _tool("submit_sql", content={"passed": False, "message": "Your SQL is not correct."}),
    )
    rec["execution_accuracy"] = False
    rec["updated_user_patience"] = -2
    assert "budget_death" in _hits(rec)


# ── budget-blocked pseudo-events ────────────────────────────────────────────────
# When the budget middleware blocks a tool it emits a ToolMessage *named after
# the blocked tool* ("Budget exhausted… You MUST call submit_sql…") with success
# status — but that tool never ran, so detectors must not treat it as a real call.


def test_extract_marks_budget_blocked_events():
    rec = _record(
        _ai(("execute_sql", {"sql": "x"})),
        _tool("execute_sql", content=_BLOCKED_TEXT),
    )
    e = extract_tool_events(rec)[0]
    assert e.blocked is True
    assert extract_tool_events(
        _record(_ai(("execute_sql", {"sql": "x"})), _tool("execute_sql"))
    )[0].blocked is False


def test_blind_submit_fires_when_only_validation_was_budget_blocked():
    # A budget-blocked execute_sql never ran the query, so the following
    # submit is still blind.
    rec = _record(
        _ai(("execute_sql", {"sql": "SELECT 1"})),
        _tool("execute_sql", content=_BLOCKED_TEXT),
        _ai(("submit_sql", {"sql": "SELECT 1"})),
        _tool("submit_sql"),
    )
    assert "blind_submit" in _hits(rec)


def test_error_loop_streak_survives_budget_blocked_call():
    # The blocked pseudo-event has success status; it must neither extend nor
    # reset the consecutive-error streak.
    rec = _record(
        _ai(("execute_sql", {"sql": "a"})), _tool("execute_sql", status="error"),
        _ai(("execute_sql", {"sql": "b"})), _tool("execute_sql", status="error"),
        _ai(("execute_sql", {"sql": "c"})), _tool("execute_sql", content=_BLOCKED_TEXT),
        _ai(("execute_sql", {"sql": "d"})), _tool("execute_sql", status="error"),
    )
    assert "unrecovered_error_loop" in _hits(rec)


def test_repeated_identical_call_quiet_when_repeat_was_budget_blocked():
    # The re-issue never executed (it was blocked), so it is not a real repeat.
    rec = _record(
        _ai(("execute_sql", {"sql": "SELECT 1"})), _tool("execute_sql"),
        _ai(("execute_sql", {"sql": "SELECT 1"})), _tool("execute_sql", content=_BLOCKED_TEXT),
    )
    assert "repeated_identical_call" not in _hits(rec)


def test_truncated_generation_in_catalog():
    assert "truncated_generation" in PATTERN_NAMES


def test_truncated_generation_fires_on_finish_reason_length():
    rec = _record(
        {"role": "ai", "finish_reason": "stop", "tool_calls": []},
        _tool("execute_sql"),
        {"role": "ai", "finish_reason": "length", "tool_calls": []},
    )
    hits = detect_patterns(rec)
    hit = next(h for h in hits if h.name == "truncated_generation")
    assert hit.message_indices == [2]  # the truncated AIMessage's trace index


def test_truncated_generation_quiet_when_no_length_finish():
    rec = _record(
        {"role": "ai", "finish_reason": "stop", "tool_calls": []},
        {"role": "ai", "finish_reason": "tool_calls", "tool_calls": []},
    )
    assert "truncated_generation" not in _hits(rec)


def _with_events(*types):
    """A record carrying middleware_events of the given types."""
    rec = _record()
    rec["middleware_events"] = [{"type": t, "message_id": f"m-{i}"} for i, t in enumerate(types)]
    return rec


def test_middleware_patterns_in_catalog():
    for name in ("tool_call_limit", "model_call_limit", "context_editing"):
        assert name in PATTERN_NAMES


def test_tool_call_limit_fires_from_middleware_event():
    assert "tool_call_limit" in _hits(_with_events("tool_call_limit"))


def test_model_call_limit_fires_from_middleware_event():
    assert "model_call_limit" in _hits(_with_events("model_call_limit"))


def test_context_editing_fires_from_middleware_event():
    assert "context_editing" in _hits(_with_events("context_editing"))


def test_middleware_patterns_quiet_without_events():
    # No middleware_events key at all (e.g. an old run) → none fire.
    quiet = _hits(_record(_ai(("submit_sql", {"sql": "x"})), _tool("submit_sql")))
    assert {"tool_call_limit", "model_call_limit", "context_editing"} & quiet == set()


def test_middleware_patterns_independent():
    # Only the event types present fire; others stay quiet.
    hits = _hits(_with_events("context_editing"))
    assert "context_editing" in hits
    assert "tool_call_limit" not in hits
    assert "model_call_limit" not in hits


def test_model_call_limit_fires_from_trace_without_events_field():
    # Reproduces museum_4: run generated before `middleware_events` existed, so the
    # field is absent — but the injected AIMessage is right there in the trace.
    rec = _record(
        _ai(("execute_sql", {"sql": "x"})),
        _tool("execute_sql"),
        {"role": "ai", "content": "Model call limits exceeded: run limit (17/17)"},
    )
    assert "middleware_events" not in rec
    hits = detect_patterns(rec)
    hit = next(h for h in hits if h.name == "model_call_limit")
    assert hit.message_indices == [2]  # the offending AIMessage's trace index


def test_tool_call_limit_fires_from_trace_without_events_field():
    # The tool-limit ToolMessage serializes to {"content": "...", "parse_error": ...};
    # its text still starts with the limit prefix.
    rec = _record(
        _ai(("execute_sql", {"sql": "x"})),
        _tool(
            "execute_sql",
            status="error",
            content={
                "content": "Tool call limit exceeded. Do not make additional tool calls.",
                "parse_error": "Expecting value: line 1 column 1",
            },
        ),
    )
    assert "middleware_events" not in rec
    hits = detect_patterns(rec)
    hit = next(h for h in hits if h.name == "tool_call_limit")
    assert hit.message_indices == [1]


def test_context_editing_only_from_field_not_trace():
    # No trace footprint exists for context editing, so a record without the field
    # cannot fire it (documented limitation for pre-wiring runs).
    rec = _record({"role": "tool", "tool_name": "x", "status": "success", "content": "[cleared]"})
    assert "context_editing" not in _hits(rec)


from explorer.patterns import (
    PatternStat,
    aggregate_patterns,
    clean_fraction,
    positional_tool_distribution,
)


def test_pattern_stat_fields():
    s = PatternStat(rate=0.25, applicable_n=8, applicable_instances=4)
    assert s.rate == 0.25
    assert s.applicable_n == 8
    assert s.applicable_instances == 4


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


def test_aggregate_instances_denominator_distinct_from_samples():
    # 2 instances × 2 samples each = 4 applicable samples but 2 instances.
    blind = _record(_ai(("submit_sql", {"sql": "x"})), _tool("submit_sql", status="success"))
    groups = {"a": [dict(blind), dict(blind)], "b": [_submit_only(), _submit_only()]}
    freq = aggregate_patterns(groups)
    bs = freq["blind_submit"]
    assert bs.applicable_n == 4          # samples
    assert bs.applicable_instances == 2  # the rate's true denominator
    # rate is per-instance: inst a = 1.0, inst b = 0.0 → 0.5.
    assert bs.rate == 0.5
    # effective flagged instances = rate × instances = 1 (a clean integer here).
    assert abs(bs.rate * bs.applicable_instances - 1.0) < 1e-9


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


def test_positional_distribution_shares_sum_to_one_per_quintile():
    groups = {"inst1": [_submit_only()]}
    df = positional_tool_distribution(groups)
    sums = df.groupby("quintile")["share"].sum()
    for q, total in sums.items():
        assert abs(total - 1.0) < 1e-9, f"quintile {q} sums to {total}"


def test_positional_distribution_excludes_budget_blocked_calls():
    rec = _record(
        _ai(("execute_sql", {"sql": "x"})), _tool("execute_sql"),
        _ai(("get_schema", {})), _tool("get_schema", content=_BLOCKED_TEXT),
    )
    df = positional_tool_distribution({"inst1": [rec]})
    assert "get_schema" not in set(df["tool"])


def test_positional_distribution_bins_by_normalized_position():
    # 2-event sequence (execute_sql, submit_sql): index 0/2=0% → 0-20%,
    # index 1/2=50% → 40-60%. Unreached quintiles yield no rows.
    groups = {"inst1": [_submit_only()]}
    df = positional_tool_distribution(groups)
    early = df[(df["quintile"] == "0-20%") & (df["tool"] == "execute_sql")]["share"]
    mid = df[(df["quintile"] == "40-60%") & (df["tool"] == "submit_sql")]["share"]
    assert float(early.iloc[0]) == 1.0
    assert float(mid.iloc[0]) == 1.0
    assert df[df["quintile"] == "20-40%"].empty  # short conversation skips this fifth


from explorer.patterns import per_iteration_pattern_rates


def _record_it(iteration, *messages):
    return {"messages": list(messages), "iteration": iteration}


def test_per_iteration_buckets_by_iteration_field():
    # blind_submit hits in iter 0 (submit, no execute), clean in iter 1.
    hit = _record_it(0, _ai(("submit_sql", {"sql": "SELECT 1"})), _tool("submit_sql"))
    clean = _record_it(
        1,
        _ai(("execute_sql", {"sql": "SELECT 1"})), _tool("execute_sql"),
        _ai(("submit_sql", {"sql": "SELECT 1"})), _tool("submit_sql"),
    )
    df = per_iteration_pattern_rates([hit, clean])
    bs = df[df["pattern"] == "Blind submit"].set_index("iteration")["rate"]
    assert bs[0] == 1.0
    assert bs[1] == 0.0


def test_per_iteration_mean_matches_sample_average_for_complete_run():
    # Two instances, two iterations each; one instance always blind-submits.
    blind = lambda it: _record_it(it, _ai(("submit_sql", {"sql": "SELECT 1"})), _tool("submit_sql"))
    ok = lambda it: _record_it(
        it,
        _ai(("execute_sql", {"sql": "SELECT 1"})), _tool("execute_sql"),
        _ai(("submit_sql", {"sql": "SELECT 1"})), _tool("submit_sql"),
    )
    records = [blind(0), blind(1), ok(0), ok(1)]
    df = per_iteration_pattern_rates(records)
    bs = df[df["pattern"] == "Blind submit"]["rate"]
    # Each iteration: 1 of 2 records blind → 0.5; mean over iterations → 0.5.
    assert list(bs) == [0.5, 0.5]
    assert abs(bs.mean() - 0.5) < 1e-9


def test_per_iteration_covers_every_pattern_label():
    df = per_iteration_pattern_rates([_record_it(0, _ai(("submit_sql", {})), _tool("submit_sql"))])
    assert set(df["pattern"]) == {lbl for _, lbl in PATTERN_CATALOG}


from explorer.patterns import repeated_identical_tool_counts


def test_repeated_counts_surplus_per_tool():
    # execute_sql repeated 3× (2 surplus), get_schema 2× (1 surplus).
    rec = _record(
        _ai(("execute_sql", {"sql": "SELECT 1"})), _tool("execute_sql"),
        _ai(("execute_sql", {"sql": "SELECT 1"})), _tool("execute_sql"),
        _ai(("execute_sql", {"sql": "SELECT 1"})), _tool("execute_sql"),
        _ai(("get_schema", {})), _tool("get_schema"),
        _ai(("get_schema", {})), _tool("get_schema"),
    )
    df = repeated_identical_tool_counts([rec])
    counts = dict(zip(df["tool"], df["repeats"]))
    assert counts == {"execute_sql": 2, "get_schema": 1}
    # sorted descending by repeats
    assert list(df["tool"]) == ["execute_sql", "get_schema"]


def test_repeated_counts_sums_across_records_and_ignores_whitespace():
    rec_a = _record(
        _ai(("execute_sql", {"sql": "SELECT  1"})), _tool("execute_sql"),
        _ai(("execute_sql", {"sql": "SELECT 1"})), _tool("execute_sql"),
    )
    rec_b = _record(
        _ai(("execute_sql", {"sql": "SELECT 2"})), _tool("execute_sql"),
        _ai(("execute_sql", {"sql": "SELECT 2"})), _tool("execute_sql"),
    )
    df = repeated_identical_tool_counts([rec_a, rec_b])
    assert dict(zip(df["tool"], df["repeats"])) == {"execute_sql": 2}


def test_repeated_counts_empty_when_no_repeats():
    rec = _record(
        _ai(("execute_sql", {"sql": "SELECT 1"})), _tool("execute_sql"),
        _ai(("execute_sql", {"sql": "SELECT 2"})), _tool("execute_sql"),
    )
    df = repeated_identical_tool_counts([rec])
    assert df.empty
    assert list(df.columns) == ["tool", "repeats"]


from explorer.patterns import first_submit_accuracy_by_quintile


def _passed(rec, ok=True):
    rec["execution_accuracy"] = ok
    return rec


def test_first_submit_accuracy_buckets_early_and_late():
    # Immediate submit (1 event → 0/1 = 0%) vs submit after 4 calls (4/5 = 80%).
    early = _passed(_record(_ai(("submit_sql", {"sql": "x"})), _tool("submit_sql")), ok=False)
    late = _passed(_record(
        _ai(("execute_sql", {"sql": "a"})), _tool("execute_sql"),
        _ai(("execute_sql", {"sql": "b"})), _tool("execute_sql"),
        _ai(("execute_sql", {"sql": "c"})), _tool("execute_sql"),
        _ai(("execute_sql", {"sql": "d"})), _tool("execute_sql"),
        _ai(("submit_sql", {"sql": "x"})), _tool("submit_sql"),
    ), ok=True)
    df = first_submit_accuracy_by_quintile([early, late]).set_index("quintile")
    assert df.loc["0-20%", "accuracy"] == 0.0
    assert df.loc["0-20%", "n"] == 1
    assert df.loc["80-100%", "accuracy"] == 1.0
    assert df.loc["80-100%", "n"] == 1


def test_first_submit_accuracy_pooled_within_bucket():
    # Two immediate submits in the same bucket: one pass, one fail → 0.5.
    p = _passed(_record(_ai(("submit_sql", {"sql": "x"})), _tool("submit_sql")), ok=True)
    f = _passed(_record(_ai(("submit_sql", {"sql": "x"})), _tool("submit_sql")), ok=False)
    df = first_submit_accuracy_by_quintile([p, f]).set_index("quintile")
    assert df.loc["0-20%", "accuracy"] == 0.5
    assert df.loc["0-20%", "n"] == 2


def test_first_submit_accuracy_excludes_no_submission():
    no_sub = _passed(_record(_ai(("execute_sql", {"sql": "x"})), _tool("execute_sql")), ok=False)
    df = first_submit_accuracy_by_quintile([no_sub])
    assert df["n"].sum() == 0


def test_first_submit_accuracy_returns_all_bins_in_order():
    df = first_submit_accuracy_by_quintile([])
    assert list(df["quintile"]) == ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]


def test_first_submit_accuracy_ignores_budget_blocked_calls():
    # A blocked call never ran; with it excluded the submit is the 1st of 2
    # real events (1/2 → 40-60%), not the 2nd of 3 (2/3 → 60-80%).
    rec = _passed(_record(
        _ai(("execute_sql", {"sql": "a"})), _tool("execute_sql"),
        _ai(("execute_sql", {"sql": "b"})), _tool("execute_sql", content=_BLOCKED_TEXT),
        _ai(("submit_sql", {"sql": "x"})), _tool("submit_sql"),
    ), ok=True)
    df = first_submit_accuracy_by_quintile([rec]).set_index("quintile")
    assert df.loc["40-60%", "n"] == 1
    assert df.loc["60-80%", "n"] == 0


from explorer.patterns import tool_position_accuracy


def test_tool_position_accuracy_excludes_budget_blocked_calls():
    rec = _passed(_record(
        _ai(("execute_sql", {"sql": "x"})), _tool("execute_sql"),
        _ai(("get_schema", {})), _tool("get_schema", content=_BLOCKED_TEXT),
    ), ok=True)
    df = tool_position_accuracy([rec])
    assert "get_schema" not in set(df["tool"])
    assert "execute_sql" in set(df["tool"])


def test_tool_position_accuracy_places_tools_by_normalized_position():
    # 5 events: execute(0/5=0%), execute(1/5=20%), execute(2/5=40%),
    #           get_schema(3/5=60%), submit(4/5=80%).
    rec = _passed(_record(
        _ai(("execute_sql", {"sql": "a"})), _tool("execute_sql"),
        _ai(("execute_sql", {"sql": "b"})), _tool("execute_sql"),
        _ai(("execute_sql", {"sql": "c"})), _tool("execute_sql"),
        _ai(("get_schema", {})), _tool("get_schema"),
        _ai(("submit_sql", {"sql": "x"})), _tool("submit_sql"),
    ), ok=True)
    cell = tool_position_accuracy([rec]).set_index(["tool", "quintile"])
    assert cell.loc[("get_schema", "60-80%"), "accuracy"] == 1.0
    assert cell.loc[("submit_sql", "80-100%"), "accuracy"] == 1.0
    assert cell.loc[("execute_sql", "0-20%"), "n"] == 1


def test_tool_position_accuracy_dedupes_repeat_in_same_quintile():
    # 10 events; execute_sql at indices 0 and 1 both fall in 0-20% (0/10, 1/10).
    msgs = []
    for i in range(10):
        name = "execute_sql" if i < 2 else "get_schema"
        args = {"sql": str(i)} if name == "execute_sql" else {}
        msgs += [_ai((name, args)), _tool(name)]
    rec = _passed(_record(*msgs), ok=True)
    cell = tool_position_accuracy([rec]).set_index(["tool", "quintile"])
    # execute_sql called twice in the first quintile → conversation counted once.
    assert cell.loc[("execute_sql", "0-20%"), "n"] == 1


def test_tool_position_accuracy_pooled_across_conversations():
    a = _passed(_record(_ai(("submit_sql", {"sql": "x"})), _tool("submit_sql")), ok=True)
    b = _passed(_record(_ai(("submit_sql", {"sql": "x"})), _tool("submit_sql")), ok=False)
    cell = tool_position_accuracy([a, b]).set_index(["tool", "quintile"])
    assert cell.loc[("submit_sql", "0-20%"), "accuracy"] == 0.5
    assert cell.loc[("submit_sql", "0-20%"), "n"] == 2


from explorer.patterns import pattern_cooccurrence


def test_cooccurrence_diagonal_is_pattern_total():
    # One conversation hits both no_submission and unrecovered_error_loop (3 errors).
    rec = _record(
        _ai(("execute_sql", {"sql": "x"})), _tool("execute_sql", status="error"),
        _ai(("execute_sql", {"sql": "x"})), _tool("execute_sql", status="error"),
        _ai(("execute_sql", {"sql": "x"})), _tool("execute_sql", status="error"),
    )
    df = pattern_cooccurrence([rec]).set_index(["given", "pattern"])
    # Diagonal: each present pattern's own total over the records.
    assert df.loc[("No submission", "No submission"), "count"] == 1
    assert df.loc[("No submission", "No submission"), "conditional"] == 1.0


def test_cooccurrence_offdiagonal_counts_shared_conversations():
    # Both records share no_submission; only the first also loops.
    looped = _record(
        _ai(("execute_sql", {"sql": "x"})), _tool("execute_sql", status="error"),
        _ai(("execute_sql", {"sql": "x"})), _tool("execute_sql", status="error"),
        _ai(("execute_sql", {"sql": "x"})), _tool("execute_sql", status="error"),
    )
    plain = _record(_ai(("get_schema", {})), _tool("get_schema"))  # no_submission only
    df = pattern_cooccurrence([looped, plain]).set_index(["given", "pattern"])
    # 2 records hit no_submission; of those, 1 also loops → P(loop | no_submission)=0.5.
    cell = df.loc[("No submission", "Unrecovered error loop")]
    assert cell["count"] == 1
    assert cell["n_given"] == 2
    assert cell["conditional"] == 0.5
    # Asymmetric: every looped record also has no_submission → P=1.0.
    assert df.loc[("Unrecovered error loop", "No submission"), "conditional"] == 1.0


def test_cooccurrence_only_active_patterns_appear():
    rec = _record(_ai(("execute_sql", {"sql": "x"})), _tool("execute_sql"))  # no_submission only
    df = pattern_cooccurrence([rec])
    assert set(df["given"]) == {"No submission"}
    assert set(df["pattern"]) == {"No submission"}


def test_cooccurrence_empty_frame_has_columns():
    # A clean conversation (execute then successful submit fires no pattern) yields
    # no rows but stable columns.
    clean = _record(
        _ai(("execute_sql", {"sql": "x"})), _tool("execute_sql"),
        _ai(("submit_sql", {"sql": "x"})), _tool("submit_sql"),
    )
    df = pattern_cooccurrence([clean])
    assert df.empty
    assert list(df.columns) == ["given", "pattern", "count", "n_given", "conditional"]


# ── resubmit_unchanged ──────────────────────────────────────────────────────────


def test_resubmit_unchanged_in_catalog():
    assert "resubmit_unchanged" in PATTERN_NAMES


def test_resubmit_unchanged_fires_when_same_sql_submitted_twice():
    rec = _record(
        _ai(("execute_sql", {"sql": "SELECT x"})), _tool("execute_sql"),
        _ai(("submit_sql", {"sql": "SELECT x"})),
        _tool("submit_sql", content="Your SQL is not correct."),
        _ai(("submit_sql", {"sql": "SELECT x"})), _tool("submit_sql"),
    )
    assert "resubmit_unchanged" in _hits(rec)


def test_resubmit_unchanged_quiet_when_sql_differs_between_submits():
    rec = _record(
        _ai(("execute_sql", {"sql": "SELECT x"})), _tool("execute_sql"),
        _ai(("submit_sql", {"sql": "SELECT x"})),
        _tool("submit_sql", content="Your SQL is not correct."),
        _ai(("execute_sql", {"sql": "SELECT y"})), _tool("execute_sql"),
        _ai(("submit_sql", {"sql": "SELECT y"})), _tool("submit_sql"),
    )
    assert "resubmit_unchanged" not in _hits(rec)


def test_resubmit_unchanged_ignores_whitespace_differences():
    rec = _record(
        _ai(("submit_sql", {"sql": "SELECT  x"})),
        _tool("submit_sql", content="Your SQL is not correct."),
        _ai(("submit_sql", {"sql": "SELECT x"})), _tool("submit_sql"),
    )
    assert "resubmit_unchanged" in _hits(rec)


def test_resubmit_unchanged_fires_even_with_other_calls_between_submits():
    # Intervening tool calls (schema exploration, etc.) between two identical
    # submits do not break the pattern — the SQL is still unchanged.
    rec = _record(
        _ai(("submit_sql", {"sql": "SELECT x"})),
        _tool("submit_sql", content="Your SQL is not correct."),
        _ai(("get_schema", {})), _tool("get_schema"),
        _ai(("psql_console", {"command": "\\dt"})), _tool("psql_console"),
        _ai(("submit_sql", {"sql": "SELECT x"})), _tool("submit_sql"),
    )
    assert "resubmit_unchanged" in _hits(rec)


def test_resubmit_unchanged_quiet_on_single_submission():
    rec = _record(
        _ai(("execute_sql", {"sql": "SELECT x"})), _tool("execute_sql"),
        _ai(("submit_sql", {"sql": "SELECT x"})), _tool("submit_sql"),
    )
    assert "resubmit_unchanged" not in _hits(rec)


# ── recovered_after_wrong_submit ─────────────────────────────────────────────────


def test_recovered_after_wrong_submit_in_catalog():
    assert "recovered_after_wrong_submit" in PATTERN_NAMES


def test_recovered_after_wrong_submit_fires_when_failed_then_passed():
    rec = _record(
        _ai(("submit_sql", {"sql": "SELECT x"})),
        _tool("submit_sql", content={"passed": False, "message": "wrong"}),
        _ai(("submit_sql", {"sql": "SELECT y"})),
        _tool("submit_sql", content={"passed": True, "message": "correct"}),
    )
    assert "recovered_after_wrong_submit" in _hits(rec)


def test_recovered_after_wrong_submit_quiet_when_first_submit_passes():
    rec = _record(
        _ai(("submit_sql", {"sql": "SELECT y"})),
        _tool("submit_sql", content={"passed": True, "message": "correct"}),
    )
    assert "recovered_after_wrong_submit" not in _hits(rec)


def test_recovered_after_wrong_submit_quiet_when_never_recovers():
    rec = _record(
        _ai(("submit_sql", {"sql": "SELECT x"})),
        _tool("submit_sql", content={"passed": False, "message": "wrong"}),
        _ai(("submit_sql", {"sql": "SELECT z"})),
        _tool("submit_sql", content={"passed": False, "message": "still wrong"}),
    )
    assert "recovered_after_wrong_submit" not in _hits(rec)


def test_recovered_after_wrong_submit_evidence_links_both_submits():
    rec = _record(
        _ai(("submit_sql", {"sql": "SELECT x"})),
        _tool("submit_sql", content={"passed": False, "message": "wrong"}),
        _ai(("submit_sql", {"sql": "SELECT y"})),
        _tool("submit_sql", content={"passed": True, "message": "correct"}),
    )
    hit = next(h for h in detect_patterns(rec) if h.name == "recovered_after_wrong_submit")
    assert hit.message_indices == [1, 3]


def test_scope_by_accuracy_filters_pass_fail_and_passes_all_through():
    from explorer.patterns import scope_by_accuracy

    recs = [
        {"instance_id": "a", "execution_accuracy": True},
        {"instance_id": "b", "execution_accuracy": False},
        {"instance_id": "c"},  # missing key → treated as failed
    ]
    assert scope_by_accuracy(recs, "All") == recs
    assert scope_by_accuracy(recs, "Passed (1)") == [recs[0]]
    assert scope_by_accuracy(recs, "Failed (0)") == [recs[1], recs[2]]
    # Unknown scope is a no-op (defensive default).
    assert scope_by_accuracy(recs, "???") == recs
