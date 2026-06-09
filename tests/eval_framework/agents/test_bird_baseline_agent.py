from unittest.mock import MagicMock

import pytest


def _make_models():
    return MagicMock(name="agent"), MagicMock(name="parser"), MagicMock(name="generator")


def _make_task():
    task = MagicMock()
    task.task_budget = 26
    task.task_question = "q"
    task.instance_id = "test_1"
    # Default OFF — matches the baseline; a real TaskData defaults these to False.
    # MagicMock attrs are truthy by default, so set them explicitly.
    task.enable_table_schema_tools = False
    task.enable_psql_console = False
    return task


class TestRunAgentBirdBaseline:
    def test_enable_ask_user_false_excludes_ask_user_tool(self, monkeypatch):
        from conversation2sql.eval_framework.agents.bird_baseline import agent_code

        captured = {}

        def fake_create_agent(model, tools, **kwargs):
            captured["tools"] = tools
            agent = MagicMock()
            agent.invoke = MagicMock(return_value={
                "messages": [],
                "initial_user_patience": 26,
                "updated_user_patience": 26,
                "tool_called_patience": [],
            })
            return agent

        monkeypatch.setattr(agent_code, "create_agent", fake_create_agent)
        monkeypatch.setattr(agent_code, "utils_process_agent_response", MagicMock(return_value={}))

        model_agent, _, _ = _make_models()
        agent_code.run_agent_bird_baseline(
            _make_task(), model_agent, None, None, enable_ask_user=False,
        )

        tool_names = [getattr(t, "name", getattr(t, "__name__", str(t))) for t in captured["tools"]]
        assert "ask_user" not in tool_names
        assert "submit_sql" in tool_names
        assert "execute_sql" in tool_names

    def test_enable_ask_user_true_includes_ask_user_tool(self, monkeypatch):
        from conversation2sql.eval_framework.agents.bird_baseline import agent_code

        captured = {}

        def fake_create_agent(model, tools, **kwargs):
            captured["tools"] = tools
            agent = MagicMock()
            agent.invoke = MagicMock(return_value={
                "messages": [],
                "initial_user_patience": 26,
                "updated_user_patience": 26,
                "tool_called_patience": [],
            })
            return agent

        monkeypatch.setattr(agent_code, "create_agent", fake_create_agent)
        monkeypatch.setattr(agent_code, "utils_process_agent_response", MagicMock(return_value={}))

        model_agent, parser, gen = _make_models()
        agent_code.run_agent_bird_baseline(
            _make_task(), model_agent, parser, gen, enable_ask_user=True,
        )

        tool_names = [getattr(t, "name", getattr(t, "__name__", str(t))) for t in captured["tools"]]
        assert "ask_user" in tool_names

    def _capture_tools(self, monkeypatch, task):
        from conversation2sql.eval_framework.agents.bird_baseline import agent_code

        captured = {}

        def fake_create_agent(model, tools, **kwargs):
            captured["tools"] = tools
            agent = MagicMock()
            agent.invoke = MagicMock(return_value={
                "messages": [],
                "initial_user_patience": 26,
                "updated_user_patience": 26,
                "tool_called_patience": [],
            })
            return agent

        monkeypatch.setattr(agent_code, "create_agent", fake_create_agent)
        monkeypatch.setattr(agent_code, "utils_process_agent_response", MagicMock(return_value={}))
        model_agent, _, _ = _make_models()
        agent_code.run_agent_bird_baseline(
            task, model_agent, None, None, enable_ask_user=False,
        )
        return [getattr(t, "name", getattr(t, "__name__", str(t))) for t in captured["tools"]]

    def test_table_schema_tools_excluded_when_flag_false(self, monkeypatch):
        """Default (flag off) keeps the baseline: get_schema is present, but the
        granular get_table_names / get_table_schema tools are not."""
        tool_names = self._capture_tools(monkeypatch, _make_task())
        assert "get_schema" in tool_names
        assert "get_table_names" not in tool_names
        assert "get_table_schema" not in tool_names

    def test_table_schema_tools_included_when_flag_true(self, monkeypatch):
        """With the flag on, both granular table tools are added alongside the
        unchanged get_schema."""
        task = _make_task()
        task.enable_table_schema_tools = True
        tool_names = self._capture_tools(monkeypatch, task)
        assert "get_schema" in tool_names
        assert "get_table_names" in tool_names
        assert "get_table_schema" in tool_names

    def test_enable_ask_user_true_with_none_models_raises(self, monkeypatch):
        from conversation2sql.eval_framework.agents.bird_baseline import agent_code

        monkeypatch.setattr(agent_code, "create_agent", MagicMock())
        model_agent, _, _ = _make_models()

        with pytest.raises(AssertionError):
            agent_code.run_agent_bird_baseline(
                _make_task(), model_agent, None, None, enable_ask_user=True,
            )


class TestUtilsProcessAgentResponse:
    def _ai(self, input_tokens, output_tokens, finish_reason=None):
        from langchain_core.messages import AIMessage

        kwargs = {}
        if finish_reason is not None:
            kwargs["response_metadata"] = {"finish_reason": finish_reason}
        return AIMessage(
            content="answer",
            usage_metadata={
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            },
            **kwargs,
        )

    def test_mean_tokens_average_over_ai_messages_only(self):
        """mean_*_tokens must divide by the number of LLM calls (AI messages),
        not by every message — tool/human/system messages carry 0 tokens and
        would otherwise dilute the per-call mean for tool-using baselines."""
        from langchain_core.messages import HumanMessage, ToolMessage

        from conversation2sql.eval_framework.agents.bird_baseline import agent_code

        messages = [
            HumanMessage(content="question"),
            self._ai(100, 20),
            ToolMessage(content="rows", tool_call_id="c1", name="execute_sql"),
            self._ai(300, 40),
        ]
        out = agent_code.utils_process_agent_response({"messages": messages})

        # 2 AI calls: (100+300)/2 and (20+40)/2 — NOT divided by 4 total messages.
        assert out["mean_prompt_tokens"] == 200.0
        assert out["mean_completion_tokens"] == 30.0

    def test_total_tokens_sum_across_ai_calls(self):
        """total_*_tokens is the cumulative tokens the model processed across
        the conversation — re-sent history makes input grow each turn."""
        from langchain_core.messages import ToolMessage

        from conversation2sql.eval_framework.agents.bird_baseline import agent_code

        messages = [
            self._ai(100, 20),
            ToolMessage(content="rows", tool_call_id="c1", name="execute_sql"),
            self._ai(300, 40),
        ]
        out = agent_code.utils_process_agent_response({"messages": messages})

        assert out["total_prompt_tokens"] == 400
        assert out["total_completion_tokens"] == 60

    def test_no_ai_messages_means_zero(self):
        from langchain_core.messages import HumanMessage

        from conversation2sql.eval_framework.agents.bird_baseline import agent_code

        out = agent_code.utils_process_agent_response(
            {"messages": [HumanMessage(content="q")]}
        )

        assert out["mean_prompt_tokens"] == 0.0
        assert out["mean_completion_tokens"] == 0.0
        assert out["total_prompt_tokens"] == 0
        assert out["total_completion_tokens"] == 0

    def test_truncated_calls_counted_from_finish_reason_length(self):
        """A model call cut off by the max-model-len cap comes back with
        finish_reason='length'. The record must surface how many calls were
        truncated and a was_truncated flag, so the silent truncation is
        visible downstream instead of looking like a normal run."""
        from conversation2sql.eval_framework.agents.bird_baseline import agent_code

        messages = [
            self._ai(100, 20, finish_reason="stop"),
            self._ai(300, 40, finish_reason="length"),
        ]
        out = agent_code.utils_process_agent_response({"messages": messages})

        assert out["num_truncated_calls"] == 1
        assert out["was_truncated"] is True

    def test_no_truncation_when_all_calls_stop(self):
        from conversation2sql.eval_framework.agents.bird_baseline import agent_code

        messages = [
            self._ai(100, 20, finish_reason="stop"),
            self._ai(300, 40, finish_reason="tool_calls"),
        ]
        out = agent_code.utils_process_agent_response({"messages": messages})

        assert out["num_truncated_calls"] == 0
        assert out["was_truncated"] is False

    def test_truncation_logs_warning(self):
        """Truncation must be loud during a run, not only discoverable post-hoc."""
        from loguru import logger as loguru_logger

        from conversation2sql.eval_framework.agents.bird_baseline import agent_code

        captured: list[str] = []
        sink_id = loguru_logger.add(
            lambda m: captured.append(m), level="WARNING", filter=lambda r: True
        )
        try:
            messages = [self._ai(300, 40, finish_reason="length")]
            agent_code.utils_process_agent_response({"messages": messages})
        finally:
            loguru_logger.remove(sink_id)

        assert any(
            "truncat" in m.lower() for m in captured
        ), "expected a WARNING mentioning truncation"

    def test_middleware_events_surfaced_in_output(self):
        """The response dict must carry middleware_events so it lands in the JSONL
        record the explorer reads."""
        from langchain_core.messages import ToolMessage

        from conversation2sql.eval_framework.agents.bird_baseline import agent_code

        messages = [
            self._ai(100, 20),
            ToolMessage(
                content="[cleared]",
                tool_call_id="c1",
                name="execute_sql",
                id="t-1",
                response_metadata={"context_editing": {"cleared": True}},
            ),
        ]
        out = agent_code.utils_process_agent_response({"messages": messages})

        assert out["middleware_events"] == [
            {"type": "context_editing", "message_id": "t-1"}
        ]
