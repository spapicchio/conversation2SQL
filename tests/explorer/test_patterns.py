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


from explorer.patterns import tool_position_accuracy


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
