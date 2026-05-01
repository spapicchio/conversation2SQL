"""Unit tests for the user-side Bird-Interact tools.

The pure ``*_impl`` functions are exercised directly so the LLM user-simulator
and the database executor can both be replaced with mocks.
"""
from __future__ import annotations

import json
from unittest.mock import patch

import psycopg2
import pytest

from conversation2sql.eval_framework.agents.bird_baseline.tools import bird_interact_user_tools as user_tools
from conversation2sql.eval_framework.agents.bird_baseline.tools import (
    _extract_group_in_tag_pattern,
    ask_user_impl,
    stage_1_parse_action,
    stage_2_generator,
    submit_sql_impl,
)


# ---------------------------------------------------------------------------
# _extract_group_in_tag_pattern
# ---------------------------------------------------------------------------
class TestExtractGroupInTagPattern:
    """The user-simulator wraps its structured output in pseudo-XML tags
    (e.g. ``<s>...</s>``). ``_extract_group_in_tag_pattern`` is the regex
    helper that pulls that payload out — the four cases below pin its
    contract: single match, multiple matches, no match, and case
    insensitivity. These are the inputs the LLM actually produces in
    practice, so each is a real failure mode rather than a hypothetical."""

    def test_returns_inner_text_for_single_match(self):
        """Trivial happy path — the most common shape from the LLM."""
        assert _extract_group_in_tag_pattern("<s>hello</s>", "s") == "hello"

    def test_returns_last_match_when_multiple(self):
        """When the LLM emits multiple tagged blocks (e.g. it ``<s>``-tags
        a draft and then a final answer), the final one is authoritative.
        Returning the *last* match avoids picking up a stale/early draft."""
        assert _extract_group_in_tag_pattern("<s>one</s>\n<s>two</s>", "s") == "two"

    def test_returns_none_when_tag_missing(self):
        """Missing tag → ``None``. Callers (stage_1/stage_2) interpret None
        as "fall back to the canned reply" rather than crashing."""
        assert _extract_group_in_tag_pattern("no tags here", "s") is None

    def test_is_case_insensitive(self):
        """Models occasionally upper-case the tag; treating the match as
        case-insensitive avoids spurious fallbacks for cosmetic differences."""
        assert _extract_group_in_tag_pattern("<S>capital</S>", "s") == "capital"


# ---------------------------------------------------------------------------
# stage_1_parse_action / stage_2_generator
# ---------------------------------------------------------------------------
class TestStageFunctions:
    """The user simulator runs in two LLM stages:

    * **stage 1** — classify the agent's clarification question into one of
      the labelled action types (``labeled("Amb")``, ``unanswerable()`` …),
    * **stage 2** — generate a natural-language reply consistent with that
      action and the task's ground-truth ambiguity.

    Each stage parses a tagged ``<s>...</s>`` block out of the model output
    and falls back to a fixed apology string when no tag is present. The
    tests cover both the success path and that fallback for each stage.
    """

    def test_stage_1_returns_parsed_tag_content(self, task_data, make_chat_model):
        """Stage 1 must strip any chain-of-thought (``<think>...</think>``)
        and return only the action label inside ``<s>...</s>`` — that's what
        stage 2 receives as input."""
        model = make_chat_model("<think>...</think>\n<s>labeled(\"Amb\")</s>")
        action = stage_1_parse_action("what is X?", task_data, model)
        assert action == 'labeled("Amb")'
        # Sanity: the LLM is invoked exactly once per stage call.
        model.invoke.assert_called_once()

    def test_stage_1_returns_fallback_when_no_tag(self, task_data, make_chat_model):
        """If the parser model produces unstructured text, stage 1 falls
        back to the canned apology so the conversation can keep going
        rather than hard-failing."""
        model = make_chat_model("free-form text without tags")
        action = stage_1_parse_action("what is X?", task_data, model)
        assert action == "I'm not sure I understand your question."

    def test_stage_2_returns_parsed_tag_content(self, task_data, make_chat_model):
        """Stage 2 happy path: extract the natural-language reply from
        ``<s>...</s>`` so it can be returned as the user's answer."""
        model = make_chat_model("<s>The active users are the recent ones.</s>")
        answer = stage_2_generator("labeled(\"Amb\")", "what is active?", task_data, model)
        assert answer == "The active users are the recent ones."

    def test_stage_2_returns_fallback_when_no_tag(self, task_data, make_chat_model):
        """Same fallback contract as stage 1 — unstructured generator
        output yields the canned apology rather than raising."""
        model = make_chat_model("no answer here")
        answer = stage_2_generator("labeled(\"Amb\")", "what is active?", task_data, model)
        assert answer == "I'm not sure I understand your question."


# ---------------------------------------------------------------------------
# ask_user_impl
# ---------------------------------------------------------------------------
class TestAskUserImpl:
    """``ask_user_impl`` orchestrates stage 1 + stage 2 to produce a single
    user reply. Two distinct LLMs are used (parser + generator) so each can
    be specialised; the tests below verify that both are invoked and that
    the generator's fallback string surfaces when its output is unparseable."""

    def test_runs_both_stages_and_returns_user_answer(self, task_data, make_chat_model):
        """End-to-end happy path: parser returns an action, generator
        returns a reply, ``ask_user_impl`` returns the reply under the
        canonical ``user_answer`` key. Both models must be called exactly
        once — calling either zero or twice indicates the orchestration
        is broken."""
        parser_model = make_chat_model("<s>labeled(\"Amb\")</s>")
        generator_model = make_chat_model("<s>I meant active users.</s>")

        result = ask_user_impl(
            clarification_question="who counts as a user?",
            task=task_data,
            model_user_parsing=parser_model,
            model_user_generator=generator_model,
        )

        assert result == {"user_answer": "I meant active users."}
        # Each stage runs exactly once per call — no retries, no skips.
        parser_model.invoke.assert_called_once()
        generator_model.invoke.assert_called_once()

    def test_falls_back_when_generator_returns_no_tag(self, task_data, make_chat_model):
        """If the generator emits unstructured text, ``ask_user_impl`` must
        substitute the canned apology rather than echoing the raw output —
        otherwise prompt-injection-style content from the generator could
        leak into the agent's transcript."""
        parser_model = make_chat_model("<s>unanswerable()</s>")
        generator_model = make_chat_model("plain text — no <s> tag")

        result = ask_user_impl(
            clarification_question="why?",
            task=task_data,
            model_user_parsing=parser_model,
            model_user_generator=generator_model,
        )
        assert result == {"user_answer": "I'm not sure I understand your question."}


# ---------------------------------------------------------------------------
# submit_sql_impl
# ---------------------------------------------------------------------------
class TestSubmitSqlImpl:
    """``submit_sql_impl`` is the grader: it runs the candidate SQL and the
    ground-truth SQL against the same DB and compares result sets.

    The non-trivial behaviours under test are:

    * **set vs. list semantics** — ``conditions["order"]`` toggles whether
      row order matters. With ``order=False`` (the default), results are
      compared as sets so two correct but differently-ordered queries both
      pass.
    * **two executions per call** — target first, then candidate. The mock
      side-effects below mirror that ordering.
    * **error handling asymmetry** — a candidate failure is captured as a
      failed grade (the agent should be able to retry); a *target* failure
      is propagated because it indicates a broken dataset, not an agent
      mistake."""

    def test_passes_when_unordered_results_match_as_set(self):
        """Same rows in different order should pass when ordering is not
        required — the grader compares result sets, not lists."""
        # ``_execute_query`` is called twice: first for the target SQL,
        # then for the candidate. ``side_effect`` feeds them in order.
        target_rows = [(1,), (2,)]
        pred_rows = [(2,), (1,)]
        with patch.object(
            user_tools, "_execute_query",
            side_effect=[(target_rows, None), (pred_rows, None)],
        ):
            result = submit_sql_impl(
                sql="SELECT id FROM users;",
                sol_sqls=["SELECT id FROM users;"],
                db_dsn="dsn",
                conditions={"order": False},
            )
        assert result["passed"] is True

    def test_fails_when_unordered_results_differ(self):
        """Different rows fail regardless of ordering. ``conditions=None``
        confirms the absence of a ``conditions`` block defaults to the
        unordered-comparison path."""
        with patch.object(
            user_tools, "_execute_query",
            side_effect=[([(1,)], None), ([(2,)], None)],
        ):
            result = submit_sql_impl(
                sql="SELECT id FROM users;",
                sol_sqls=["SELECT id FROM users;"],
                db_dsn="dsn",
                conditions=None,
            )
        assert result["passed"] is False
        # The user-facing message is fixed — pinning it ensures the agent's
        # retry heuristics keep working when this string is referenced.
        assert result["message"] == "Your SQL is not correct."

    def test_passes_when_ordered_results_match_in_order(self):
        """When ``conditions["order"]=True`` (e.g. queries with ORDER BY),
        identical lists in the same order pass."""
        rows = [(1,), (2,)]
        with patch.object(
            user_tools, "_execute_query",
            side_effect=[(rows, None), (rows, None)],
        ):
            result = submit_sql_impl(
                sql="SELECT id FROM users ORDER BY id;",
                sol_sqls=["SELECT id FROM users ORDER BY id;"],
                db_dsn="dsn",
                conditions={"order": True},
            )
        assert result["passed"] is True

    def test_fails_when_ordered_results_have_wrong_order(self):
        """The complement of the previous test: same rows, wrong order →
        fail when order is required. This is what catches a candidate that
        forgot ORDER BY on a question that demanded it."""
        with patch.object(
            user_tools, "_execute_query",
            side_effect=[([(1,), (2,)], None), ([(2,), (1,)], None)],
        ):
            result = submit_sql_impl(
                sql="SELECT id FROM users;",
                sol_sqls=["SELECT id FROM users ORDER BY id;"],
                db_dsn="dsn",
                conditions={"order": True},
            )
        assert result["passed"] is False
        assert result["message"] == "Your SQL is not correct."

    def test_query_canceled_after_target_runs(self):
        """Statement timeouts on the candidate side are reported as a
        graceful failure with a 'timed out' message (so the agent learns
        to write more efficient SQL), not raised as an exception."""
        with patch.object(
            user_tools, "_execute_query",
            side_effect=[
                ([(1,)], None),  # target completes successfully
                psycopg2.extensions.QueryCanceledError("timeout"),  # candidate times out
            ],
        ):
            result = submit_sql_impl(
                sql="SELECT pg_sleep(120);",
                sol_sqls=["SELECT 1;"],
                db_dsn="dsn",
                conditions=None,
            )
        assert result["passed"] is False
        assert "timed out" in result["message"]

    def test_database_error_on_prediction_is_reported(self):
        """A ``DatabaseError`` from the candidate (e.g. typo, wrong column)
        is wrapped into the failure message — the error class name appears
        so the agent has a hint at what went wrong without crashing the run."""
        with patch.object(
            user_tools, "_execute_query",
            side_effect=[
                ([(1,)], None),
                psycopg2.DatabaseError("boom"),
            ],
        ):
            result = submit_sql_impl(
                sql="SELECT bad_col;",
                sol_sqls=["SELECT 1;"],
                db_dsn="dsn",
                conditions=None,
            )
        assert result["passed"] is False
        assert "DatabaseError" in result["message"]

    def test_target_database_error_propagates(self):
        """If the *target* SQL fails, that's a dataset bug — silently
        passing or failing the candidate would mislead the evaluation, so
        the exception is re-raised to surface the broken sample."""
        with patch.object(
            user_tools, "_execute_query",
            side_effect=psycopg2.DatabaseError("ground-truth blew up"),
        ):
            # Note the asymmetry vs. the previous test: a candidate-side
            # error is captured, but a target-side error must escape.
            with pytest.raises(psycopg2.DatabaseError):
                submit_sql_impl(
                    sql="SELECT 1;",
                    sol_sqls=["SELECT does_not_exist;"],
                    db_dsn="dsn",
                    conditions=None,
                )


# ---------------------------------------------------------------------------
# Wrappers — sanity check the langchain @tool wires through to the impls.
#
# These tests mirror the wrapper tests in the env-tools suite: confirm that
# the LangChain ``@tool`` glue threads runtime context to the impl and
# JSON-serialises the result. They're shallow on purpose; the impls above
# already cover the semantics.
# ---------------------------------------------------------------------------
class _Runtime:
    """Stub LangGraph runtime exposing only ``.context`` (all the user-side
    tools read)."""

    def __init__(self, context):
        self.context = context


def _invoke_tool(tool_obj, *, runtime, **kwargs):
    """Invoke a langchain ``@tool`` directly while injecting our stub runtime."""
    return tool_obj.func(runtime=runtime, **kwargs)


def test_ask_user_wrapper_returns_serialized_user_answer(task_data, make_chat_model):
    """Wiring check for ``ask_user``: ``return_tool_ask_user`` must bind the
    parser/generator models into the resulting tool, and the tool must
    serialise the impl's dict to JSON."""
    parser_model = make_chat_model("<s>labeled(\"Amb\")</s>")
    generator_model = make_chat_model("<s>You meant active users.</s>")
    # ``return_tool_ask_user`` is a factory that captures the two LLMs in
    # closure — exercising it confirms that closure works end-to-end.
    ask_user = user_tools.return_tool_ask_user(parser_model, generator_model)

    raw = _invoke_tool(
        ask_user,
        clarification_question="active?",
        runtime=_Runtime(task_data),
    )
    assert json.loads(raw) == {"user_answer": "You meant active users."}


def test_submit_sql_wrapper_serializes_passed_result(task_data):
    """Wiring check for ``submit_sql``: the wrapper must read ``sol_sqls``,
    ``db_dsn`` and ``conditions`` from the task context (the agent only
    supplies the candidate ``sql``) and JSON-serialise the grade."""
    rows = [(1,)]
    # Identical target/candidate rows → expect ``passed=True`` after JSON
    # round-trip. The two side-effects model the target+candidate runs.
    with patch.object(user_tools, "_execute_query",
                      side_effect=[(rows, None), (rows, None)]):
        raw = _invoke_tool(
            user_tools.submit_sql,
            sql="SELECT id FROM users;",
            runtime=_Runtime(task_data),
        )
    decoded = json.loads(raw)
    assert decoded["passed"] is True
