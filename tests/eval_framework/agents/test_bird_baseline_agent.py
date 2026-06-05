from unittest.mock import MagicMock

import pytest


def _make_models():
    return MagicMock(name="agent"), MagicMock(name="parser"), MagicMock(name="generator")


def _make_task():
    task = MagicMock()
    task.task_budget = 26
    task.task_question = "q"
    task.instance_id = "test_1"
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

    def test_enable_ask_user_true_with_none_models_raises(self, monkeypatch):
        from conversation2sql.eval_framework.agents.bird_baseline import agent_code

        monkeypatch.setattr(agent_code, "create_agent", MagicMock())
        model_agent, _, _ = _make_models()

        with pytest.raises(AssertionError):
            agent_code.run_agent_bird_baseline(
                _make_task(), model_agent, None, None, enable_ask_user=True,
            )


class TestUtilsProcessAgentResponse:
    def _ai(self, input_tokens, output_tokens):
        from langchain_core.messages import AIMessage

        return AIMessage(
            content="answer",
            usage_metadata={
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            },
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
