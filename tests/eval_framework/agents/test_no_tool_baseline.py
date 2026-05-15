import pytest

from conversation2sql.eval_framework.agents.no_tool_baseline.baseline_model import (
    extract_sql_from_response,
)


class TestExtractSqlFromResponse:
    def test_sql_fenced_block(self):
        text = "Here is the answer:\n```sql\nSELECT 1;\n```\n"
        assert extract_sql_from_response(text) == "SELECT 1;"

    def test_bare_fenced_block(self):
        text = "```\nSELECT 1\n```"
        assert extract_sql_from_response(text) == "SELECT 1"

    def test_returns_last_block_when_multiple(self):
        text = (
            "First draft:\n```sql\nSELECT 0;\n```\n"
            "Final answer:\n```sql\nSELECT 1;\n```\n"
        )
        assert extract_sql_from_response(text) == "SELECT 1;"

    def test_no_block_returns_none(self):
        text = "I think the answer is SELECT 1;"
        assert extract_sql_from_response(text) is None

    def test_block_followed_by_prose(self):
        text = "```sql\nSELECT 1;\n```\nThis returns one row."
        assert extract_sql_from_response(text) == "SELECT 1;"

    def test_whitespace_only_block_returns_none(self):
        text = "```sql\n   \n```"
        assert extract_sql_from_response(text) is None

    def test_no_sql_start(self):
        text = "```\n   \n```"
        assert extract_sql_from_response(text) is None

    def test_unclosed_fence_with_semicolon(self):
        text = "```sql\nSELECT 1;"
        assert extract_sql_from_response(text) == "SELECT 1;"

    def test_unclosed_fence_without_semicolon(self):
        text = "```sql\nSELECT 1"
        assert extract_sql_from_response(text) == "SELECT 1"

    def test_unclosed_fence_stops_at_first_semicolon(self):
        text = "```sql\nSELECT 1;\nSELECT 2"
        assert extract_sql_from_response(text) == "SELECT 1;"

    def test_closed_fence_takes_priority_over_unclosed(self):
        # closed block first, then an unclosed block — closed wins
        text = "```sql\nSELECT 1;\n```\n```sql\nSELECT 2;"
        assert extract_sql_from_response(text) == "SELECT 1;"

    def test_raw_sql_select_starts_line(self):
        text = "The answer is:\nSELECT 1;"
        assert extract_sql_from_response(text) == "SELECT 1;"

    def test_raw_sql_with_keyword(self):
        text = "Reasoning...\nWITH cte AS (SELECT 1) SELECT * FROM cte;"
        assert extract_sql_from_response(text) == "WITH cte AS (SELECT 1) SELECT * FROM cte;"

    def test_raw_sql_last_keyword_wins(self):
        text = "First try:\nSELECT 0;\nBetter answer:\nSELECT 1;"
        assert extract_sql_from_response(text) == "SELECT 1;"

    def test_raw_sql_no_semicolon(self):
        text = "Reasoning prose\nSELECT * FROM t"
        assert extract_sql_from_response(text) == "SELECT * FROM t"

    def test_raw_sql_keyword_inside_word_not_matched(self):
        # SELECTING never starts a line — should return None
        text = "I am SELECTING rows from the table"
        assert extract_sql_from_response(text) is None

    def test_closed_fence_takes_priority_over_raw_sql(self):
        text = "```sql\nSELECT 1;\n```\nYou could also do SELECT 2;"
        assert extract_sql_from_response(text) == "SELECT 1;"

from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage

from conversation2sql.eval_framework.agents.no_tool_baseline.baseline_model import (
    run_baseline_no_tool,
)


def _make_task() -> MagicMock:
    task = MagicMock()
    task.ddl_database_schema = "CREATE TABLE t (id INT);"
    task.task_question = "How many rows?"
    return task


def _make_ai(
    content: str, prompt_tokens: int = 100, completion_tokens: int = 50
) -> AIMessage:
    msg = AIMessage(content=content)
    msg.usage_metadata = {
        "input_tokens": prompt_tokens,
        "output_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }
    msg.response_metadata = {"finish_reason": "stop", "_response_cost": 0.001}
    return msg


class TestRunBaselineNoTool:
    @patch(
        "conversation2sql.eval_framework.agents.no_tool_baseline.baseline_model.submit_sql_impl"
    )
    def test_passes_when_sql_extracted_and_submit_passes(self, mock_submit):
        mock_submit.return_value = {"passed": True, "pred_result": [], "gt_result": []}
        model = MagicMock()
        model.invoke.return_value = _make_ai("```sql\nSELECT COUNT(*) FROM t;\n```")

        result = run_baseline_no_tool(_make_task(), model)

        assert result["execution_accuracy"] is True
        assert result["tool_calls_in_order"] == []
        assert result["initial_user_patience"] is None
        assert result["updated_user_patience"] is None
        # message log: user + ai + synthetic submit_sql_offline tool message
        assert len(result["messages"]) == 3
        assert result["messages"][-1]["tool_name"] == "submit_sql_offline"
        assert result["total_tokens"] == 150

    @patch(
        "conversation2sql.eval_framework.agents.no_tool_baseline.baseline_model.submit_sql_impl"
    )
    def test_records_failure_when_no_sql_block(self, mock_submit):
        model = MagicMock()
        model.invoke.return_value = _make_ai("I cannot answer this.")

        result = run_baseline_no_tool(_make_task(), model)

        assert result["execution_accuracy"] is False
        assert result["messages"][-1]["content"]["error"] == "no_sql_block_found"
        mock_submit.assert_not_called()
