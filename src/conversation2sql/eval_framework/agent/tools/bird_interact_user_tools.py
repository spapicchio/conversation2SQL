"""
Bird-Interact user interaction tools.

These tools correspond to the ``User`` interaction object in the Bird-Interact
agent prompt.  They allow the agent to ask the user for clarification and to
submit a final SQL query for evaluation.

Cost summary (mirrors the original prompt):
    ask_user    → 2 patience  (clarification question)
    submit_sql  → 3 patience  (final SQL submission; typically ends the episode)

The ``ask_user`` tool calls an LLM user-simulator whose prompt templates and
parameters are stored in ``ToolUserContext``.  ``submit_sql`` records the
submission in ``template_params.submitted_sql`` and returns a stub response;
the evaluation harness (scorer) reads that attribute to grade the answer.

Each langchain ``@tool`` is a thin wrapper that pulls the relevant fields out
of ``runtime.context`` and delegates to a plain Python ``*_impl`` function.
The ``*_impl`` functions hold all the real logic and are unit-tested directly
in ``tests/eval_framework/tools/`` without needing the LangGraph runtime.
"""
import json
import re

import psycopg2
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime

from conversation2sql.eval_framework.agent.agent_code_state import CustomAgentState
from conversation2sql.eval_framework.agent.tools.bird_user_prompt import build_llm_as_a_parser_messages, \
    build_llm_as_a_generator_messages
from conversation2sql.eval_framework.agent.tools.utils import _segment_sql_and_parse_in_str, remove_round, \
    remove_distinct, remove_comments
from conversation2sql.eval_framework.agent.tools.utils_db_execute import _execute_query, preprocess_results
from conversation2sql.eval_framework.state import TaskData

USER_TOOL_COSTS: dict[str, float] = {
    "ask_user": 2.0,
    "submit_sql": 3.0,
}


def _extract_group_in_tag_pattern(content: str, pattern_tag: str = 's') -> str | None:
    pattern = re.compile(
        rf"\s*<{pattern_tag}>\s*([\s\S]*?)\s*</{pattern_tag}>\s*",
        flags=re.DOTALL | re.MULTILINE | re.IGNORECASE,
    )
    pattern_match = pattern.findall(content)
    if pattern_match:
        return pattern_match[-1].strip()
    return None


def stage_1_parse_action(clarification_question, task: TaskData, model_user_parsing: BaseChatModel) -> str:
    """Stage 1: Action Parser — maps clarification question to action (AMB/LOC/UNA)."""

    sql_segments = "\n===\n".join(_segment_sql_and_parse_in_str(sql) for sql in task.sol_sql)

    messages = build_llm_as_a_parser_messages({
        'ambiguities_json': json.dumps(task.user_query_ambiguity, indent=4),
        'sql_segments': sql_segments,
        'clarification_question': clarification_question,
    })

    content: str = model_user_parsing.invoke(messages).content  # pyrefly: ignore
    parsed_content = _extract_group_in_tag_pattern(content, 's')
    return "I'm not sure I understand your question." if parsed_content is None else parsed_content


def stage_2_generator(action,
                      clarification_question,
                      task: TaskData,
                      model_user_generator: BaseChatModel) -> str:
    sql_segments = "\n===\n".join(_segment_sql_and_parse_in_str(sql) for sql in task.sol_sql)

    messages = build_llm_as_a_generator_messages({
        "db_schema": task.ddl_database_schema,
        "ambiguities_json": json.dumps(task.user_query_ambiguity, indent=4),
        "not_ambig_question": task.not_ambiguos_query,
        "gt_sql": task.sol_sql,
        "sql_segments": sql_segments,
        "asked_question": clarification_question,
        "action": action
    })

    content = model_user_generator.invoke(messages).content
    if not isinstance(content, str):
        raise ValueError(f"Expected content to be a string, got {type(content)}")

    parsed_content = _extract_group_in_tag_pattern(content, 's')
    return "I'm not sure I understand your question." if parsed_content is None else parsed_content


# ---------------------------------------------------------------------------
# Pure implementation functions (testable without LangGraph runtime)
# ---------------------------------------------------------------------------
def ask_user_impl(
        clarification_question: str,
        task: TaskData,
        model_user_parsing: BaseChatModel,
        model_user_generator: BaseChatModel,
) -> dict:
    action = stage_1_parse_action(clarification_question, task, model_user_parsing)
    generated = stage_2_generator(action, clarification_question, task, model_user_generator)
    return {"user_answer": generated}


def submit_sql_impl(
        sql: str,
        sol_sqls: list[str],
        db_dsn: str,
        conditions: dict | None,
) -> dict:
    pred_sql = remove_round(remove_distinct(remove_comments(sql)))
    target_sql = remove_round(remove_distinct(remove_comments(sol_sqls[0])))
    target_result, _ = _execute_query(query=target_sql, db_dsn=db_dsn)
    passed = False
    try:
        pred_result, _ = _execute_query(query=pred_sql, db_dsn=db_dsn)
        pred_result = preprocess_results(pred_result)
        target_result = preprocess_results(target_result)
        if conditions and conditions.get("order", False):
            if pred_result == target_result:
                passed = True
                message = "Phase 1 correct!. Task finished.",
            else:
                message = "Your SQL is not correct."
        else:
            if set(pred_result) == set(target_result):
                passed = True
                message = "Phase 1 correct! Task finished.",
            else:
                message = "Your SQL is not correct."
    except psycopg2.extensions.QueryCanceledError as e:
        message = f"Submitted SQL execution timed out: {e}"
    except psycopg2.DatabaseError as e:
        message = f"DatabaseError executing submitted SQL: {e}"

    return {"passed": passed, "message": message}


# ---------------------------------------------------------------------------
# Ask user (clarification) — cost: 2 patience
# ---------------------------------------------------------------------------
def return_tool_ask_user(model_user_parsing: BaseChatModel,
                         model_user_generator: BaseChatModel):
    @tool
    def ask_user(
            clarification_question: str,
            runtime: ToolRuntime[TaskData, CustomAgentState],
    ) -> str:
        """Ask the user a clarification question about their query.
        Use this when the user's request is ambiguous and you need more information.
        Cost: 2 bird-coins.

        Args:
            clarification_question: The clarification question to ask the user.

        Returns:
            The user's response to your question.
        """
        return json.dumps(
            ask_user_impl(
                clarification_question=clarification_question,
                task=runtime.context,
                model_user_parsing=model_user_parsing,
                model_user_generator=model_user_generator,
            ),
            indent=2,
        )

    return ask_user


# ---------------------------------------------------------------------------
# Submit SQL — cost: 3 patience
# ---------------------------------------------------------------------------
@tool
def submit_sql(
        sql: str,
        runtime: ToolRuntime[TaskData, CustomAgentState],
) -> str:
    """Submit your final SQL query for evaluation.
    This tests your SQL against the ground truth. Only submit when confident.
    Cost: 3 bird-coins.

    Args:
        sql: The final PostgreSQL SQL query to submit.

    Returns:
        Evaluation result including pass/fail, reward, and any follow-up instructions.
    """
    return json.dumps(
        submit_sql_impl(
            sql=sql,
            sol_sqls=runtime.context.sol_sql,
            db_dsn=runtime.context.db_dsn,
            conditions=runtime.context.sql_query_conditions,
        ),
        indent=2,
    )
