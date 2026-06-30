from pathlib import Path

from conversation2sql.eval_framework.agents.deep_agent import agent_code
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from deepagents import SubAgentMiddleware


def _task(task_data, **flags):
    for k, v in flags.items():
        setattr(task_data, k, v)
    return task_data


def _fake_model() -> FakeListChatModel:
    # deepagents introspects the model when compiling a subagent, so the
    # middleware tests need a real BaseChatModel rather than a MagicMock.
    return FakeListChatModel(responses=["ok"])


def test_middleware_minimal_by_default(task_data):
    mws = agent_code._build_deep_middleware(_task(task_data), _fake_model())
    assert not any(isinstance(m, SubAgentMiddleware) for m in mws)
    from conversation2sql.eval_framework.agents.bird_baseline.agent_callback import (
        tool_wrapper_patience_and_submit,
    )
    assert tool_wrapper_patience_and_submit in mws


def test_middleware_subagents_flag_adds_component(task_data):
    mws = agent_code._build_deep_middleware(
        _task(task_data, deep_enable_subagents=True), _fake_model()
    )
    assert any(isinstance(m, SubAgentMiddleware) for m in mws)


def test_tools_are_bash_submit_ask_user(task_data, make_chat_model, tmp_path):
    tools = agent_code._build_deep_tools(
        _task(task_data),
        model_user_parsing=make_chat_model("<s>x</s>"),
        model_user_generator=make_chat_model("<s>y</s>"),
        catalog_dir=tmp_path,
        pg_env={},
    )
    names = {t.name for t in tools}
    assert names == {"bash", "submit_sql", "ask_user"}
    assert "execute_sql" not in names
    assert "read_file" not in names


import pytest


def test_catalog_dir_cleaned_up_on_exception(task_data, make_chat_model, tmp_path, monkeypatch):
    import conversation2sql.eval_framework.agents.deep_agent.agent_code as ac

    created = {}

    def fake_materialize(task):
        d = tmp_path / "catalog"
        d.mkdir()
        created["dir"] = d
        return d

    class _Boom:
        def invoke(self, *a, **k):
            raise RuntimeError("boom")

    monkeypatch.setattr(ac, "materialize_catalog_dir", fake_materialize)
    monkeypatch.setattr(ac, "create_agent", lambda *a, **k: _Boom())

    with pytest.raises(RuntimeError):
        ac.run_agent_deep_agent(
            task_data,
            model_agent=make_chat_model("x"),
            model_user_parsing=make_chat_model("<s>x</s>"),
            model_user_generator=make_chat_model("<s>y</s>"),
        )
    assert not created["dir"].exists()


def test_system_and_user_travel_together_in_initial_state(
    task_data, make_chat_model, tmp_path, monkeypatch
):
    # The system prompt is no longer split out to create_agent's system_prompt=;
    # both the system and user turns are passed in the initial state messages.
    import conversation2sql.eval_framework.agents.deep_agent.agent_code as ac

    monkeypatch.setattr(ac, "materialize_catalog_dir", lambda task: tmp_path)
    monkeypatch.setattr(ac, "build_pg_env", lambda dsn: {})

    captured = {}

    class _Recorder:
        def invoke(self, state, *a, **k):
            captured["state"] = state
            return {"messages": []}

    monkeypatch.setattr(
        ac, "create_agent", lambda *a, **k: (captured.setdefault("kwargs", k), _Recorder())[1]
    )
    monkeypatch.setattr(ac, "_extract_predicted_sql", lambda msgs: None)
    monkeypatch.setattr(ac, "utils_process_agent_response", lambda *a, **k: {})

    ac.run_agent_deep_agent(
        task_data,
        model_agent=make_chat_model("x"),
        model_user_parsing=make_chat_model("<s>x</s>"),
        model_user_generator=make_chat_model("<s>y</s>"),
    )

    # system_prompt= is not used; the system turn rides in the state messages.
    assert "system_prompt" not in captured["kwargs"]
    msgs = captured["state"]["messages"]
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
    assert "captured_system_prompt" not in captured["state"]
