import json

import psycopg2

from conversation2sql.eval_framework.agents.maintenance_agent.tools import (
    maintenance_tools as mt,
)
from conversation2sql.eval_framework.agents.maintenance_agent.tools.maintenance_tools import (
    return_tool_comment_on_issue,
    return_tool_run_tests,
    return_tool_write_query,
    run_tests_impl,
    submit,
)


class _Runtime:
    """Stub LangGraph runtime exposing only `.context` (all comment_on_issue reads)."""

    def __init__(self, context):
        self.context = context


class TestWriteQuery:
    def test_writes_full_content_to_fixed_path(self, tmp_path):
        (tmp_path / "queries").mkdir()
        (tmp_path / "queries" / "answer.sql").write_text("-- stub\n")
        tool = return_tool_write_query(tmp_path)

        tool.func(content="SELECT 1;")

        assert (tmp_path / "queries" / "answer.sql").read_text() == "SELECT 1;"

    def test_overwrite_replaces_not_appends(self, tmp_path):
        (tmp_path / "queries").mkdir()
        (tmp_path / "queries" / "answer.sql").write_text("SELECT 1;")
        tool = return_tool_write_query(tmp_path)

        tool.func(content="SELECT 2;")

        assert (tmp_path / "queries" / "answer.sql").read_text() == "SELECT 2;"

    def test_ignores_path_like_content_in_argument(self, tmp_path):
        # The target path is fixed in the tool closure, not derived from the
        # argument — a filename-shaped `content` string is written verbatim as
        # text, never treated as a path.
        (tmp_path / "queries").mkdir()
        (tmp_path / "queries" / "answer.sql").write_text("-- stub\n")
        tool = return_tool_write_query(tmp_path)

        tool.func(content="../../etc/passwd")

        assert (tmp_path / "queries" / "answer.sql").read_text() == "../../etc/passwd"
        assert not (tmp_path / "etc").exists()


class TestRunTestsImpl:
    def test_stub_fails(self, tmp_path):
        path = tmp_path / "answer.sql"
        path.write_text("-- TODO: replace this stub with your SQL query.\n")

        result = run_tests_impl(path, db_dsn="postgresql://x")

        assert result["passed"] is False
        assert "stub" in result["message"]

    def test_valid_query_passes(self, tmp_path, monkeypatch):
        path = tmp_path / "answer.sql"
        path.write_text("SELECT 1;")
        monkeypatch.setattr(mt, "_execute_query", lambda query, db_dsn: ([], ()))

        result = run_tests_impl(path, db_dsn="postgresql://x")

        assert result["passed"] is True

    def test_invalid_query_fails_structurally(self, tmp_path, monkeypatch):
        path = tmp_path / "answer.sql"
        path.write_text("SELEKT 1;")

        def _raise(query, db_dsn):
            raise psycopg2.DatabaseError("syntax error")

        monkeypatch.setattr(mt, "_execute_query", _raise)

        result = run_tests_impl(path, db_dsn="postgresql://x")

        assert result["passed"] is False
        assert "does not parse" in result["message"]


class TestRunTestsTool:
    def test_tool_serializes_impl_result_to_json(self, tmp_path, monkeypatch):
        (tmp_path / "queries").mkdir()
        (tmp_path / "queries" / "answer.sql").write_text("SELECT 1;")
        monkeypatch.setattr(mt, "_execute_query", lambda query, db_dsn: ([], ()))
        tool = return_tool_run_tests(tmp_path, db_dsn="postgresql://x")

        raw = tool.func()

        assert json.loads(raw)["passed"] is True


class TestCommentOnIssue:
    def test_appends_question_and_answer_to_issue_md(
        self, tmp_path, make_chat_model, task_data
    ):
        (tmp_path / "ISSUE.md").write_text("# Issue\n\nq\n\n## Comments\n")
        model_user_parsing = make_chat_model("<s>AMB</s>")
        model_user_generator = make_chat_model("<s>Use the users table.</s>")
        tool = return_tool_comment_on_issue(
            tmp_path, model_user_parsing, model_user_generator
        )

        answer = tool.func(question="Which table?", runtime=_Runtime(task_data))

        assert answer == "Use the users table."
        issue = (tmp_path / "ISSUE.md").read_text()
        assert "**Agent:** Which table?" in issue
        assert "**Author:** Use the users table." in issue


class TestSubmit:
    def test_returns_confirmation_with_no_pass_fail_signal(self):
        result = submit.func()
        assert "passed" not in result.lower()
        assert "submitted" in result.lower()
