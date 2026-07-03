"""The four new maintenance_agent tools: write_query, run_tests, comment_on_issue,
submit. `bash` is reused unchanged from deep_agent.tools.bash_tool.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import psycopg2
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime

from conversation2sql.eval_framework.agents.bird_baseline.agent_code_state import (
    CustomAgentState,
)
from conversation2sql.eval_framework.agents.bird_baseline.tools import USER_TOOL_COSTS
from conversation2sql.eval_framework.agents.bird_baseline.tools.bird_interact_user_tools import (
    ask_user_impl,
)
from conversation2sql.eval_framework.agents.bird_baseline.tools.utils_db_execute import (
    _execute_query,
)
from conversation2sql.eval_framework.agents.tool_specs import (
    ToolSpec,
    stamp_cost_in_descriptions,
)
from conversation2sql.eval_framework.state import TaskData

MAINTENANCE_TOOL_SPECS: dict[str, ToolSpec] = {
    "write_query": ToolSpec(
        "write_query", 1.0,
        "overwrite queries/answer.sql with the full SQL query text (not a diff)",
    ),
    "comment_on_issue": ToolSpec(
        "comment_on_issue", USER_TOOL_COSTS["ask_user"],
        "post a clarification question to the issue thread and get the author's reply",
    ),
    "run_tests": ToolSpec(
        "run_tests", 1.0,
        "check that queries/answer.sql is written and EXPLAINs successfully "
        "(structure only, not correctness)",
    ),
    "submit": ToolSpec(
        "submit", USER_TOOL_COSTS["submit_sql"],
        "submit your work for review; ends the episode with no pass/fail feedback",
    ),
}


def _strip_sql_comments(text: str) -> str:
    text = re.sub(r"--.*", "", text)
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    return text.strip()


# ---------------------------------------------------------------------------
# write_query — cost: 1.0
# ---------------------------------------------------------------------------
def return_tool_write_query(catalog_dir: Path):
    """Build the per-task `write_query` tool bound to its fixed target path."""
    target = catalog_dir / "queries" / "answer.sql"

    @tool
    def write_query(content: str) -> str:
        """Overwrite queries/answer.sql with the full SQL query text.

        This always replaces the WHOLE file — there is no partial edit.
        Include the complete query, not a diff.

        Args:
            content: The full SQL query text to write to queries/answer.sql.

        Returns:
            A confirmation message.
        """
        target.write_text(content, encoding="utf-8")
        return f"Wrote {len(content)} characters to queries/answer.sql."

    stamp_cost_in_descriptions([write_query], MAINTENANCE_TOOL_SPECS)
    return write_query


# ---------------------------------------------------------------------------
# run_tests — cost: 1.0. Structural check only, never a correctness oracle.
# ---------------------------------------------------------------------------
def run_tests_impl(query_path: Path, db_dsn: str) -> dict:
    content = query_path.read_text(encoding="utf-8")
    stripped = _strip_sql_comments(content)
    if not stripped:
        return {
            "passed": False,
            "message": "queries/answer.sql is still the empty stub.",
        }
    try:
        _execute_query(query=f"EXPLAIN {stripped}", db_dsn=db_dsn)
    except psycopg2.DatabaseError as e:
        return {
            "passed": False,
            "message": f"queries/answer.sql does not parse: {e}",
        }
    return {
        "passed": True,
        "message": "queries/answer.sql is non-empty and parses successfully.",
    }


def return_tool_run_tests(catalog_dir: Path, db_dsn: str):
    """Build the per-task `run_tests` tool bound to its fixed target path + DSN."""
    query_path = catalog_dir / "queries" / "answer.sql"

    @tool
    def run_tests() -> str:
        """Run the visible structural check on queries/answer.sql: (1) the file
        is no longer the empty stub, (2) it EXPLAINs successfully against the
        database. This checks structure only — it never reveals whether the
        query is semantically correct.

        Returns:
            A JSON string with `passed` (bool) and `message` (str).
        """
        return json.dumps(run_tests_impl(query_path, db_dsn), indent=2)

    stamp_cost_in_descriptions([run_tests], MAINTENANCE_TOOL_SPECS)
    return run_tests


# ---------------------------------------------------------------------------
# comment_on_issue — cost: USER_TOOL_COSTS["ask_user"] (2.0)
# ---------------------------------------------------------------------------
def return_tool_comment_on_issue(
    catalog_dir: Path,
    model_user_parsing: BaseChatModel,
    model_user_generator: BaseChatModel,
):
    """Build the per-task `comment_on_issue` tool bound to its ISSUE.md path."""
    issue_path = catalog_dir / "ISSUE.md"

    @tool
    def comment_on_issue(
        question: str,
        runtime: ToolRuntime[TaskData, CustomAgentState],
    ) -> str:
        """Post a comment on the issue thread asking the ticket author a
        clarification question. Use this when the request is ambiguous.
        Ask one question at a time.

        Args:
            question: The clarification question to post.

        Returns:
            The author's reply.
        """
        answer = ask_user_impl(
            clarification_question=question,
            task=runtime.context,
            model_user_parsing=model_user_parsing,
            model_user_generator=model_user_generator,
        )["user_answer"]
        with issue_path.open("a", encoding="utf-8") as f:
            f.write(f"\n**Agent:** {question}\n\n**Author:** {answer}\n")
        return answer

    stamp_cost_in_descriptions([comment_on_issue], MAINTENANCE_TOOL_SPECS)
    return comment_on_issue


# ---------------------------------------------------------------------------
# submit — cost: USER_TOOL_COSTS["submit_sql"] (3.0). Always terminal, silent —
# see make_tool_wrapper_patience_and_submit_silent in bird_baseline/agent_callback.py.
# ---------------------------------------------------------------------------
@tool
def submit() -> str:
    """Submit your work for review. This ends the episode immediately: you
    will receive no pass/fail feedback and cannot retry. Only call this once
    queries/answer.sql is complete — verify with run_tests first.

    Returns:
        A confirmation string. No correctness feedback is given.
    """
    return "Submitted. No further actions will be taken."


stamp_cost_in_descriptions([submit], MAINTENANCE_TOOL_SPECS)
