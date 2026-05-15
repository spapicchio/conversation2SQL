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
