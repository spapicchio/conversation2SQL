from pathlib import Path
from types import SimpleNamespace

from langchain_core.messages import ToolMessage

from conversation2sql.eval_framework.agents.deep_agent import agent_code
from conversation2sql.eval_framework.agents.deep_agent.catalog_seed import (
    deep_tool_costs,
)


def test_middleware_includes_patience_budget(task_data):
    mws = agent_code._build_deep_middleware()
    names = [type(m).__name__ for m in mws]
    assert "tool_wrapper_patience_and_submit" in names


def test_bash_tool_call_is_charged_deep_agent_cost(task_data):
    # The bash tool is deep_agent's only DB-facing tool, costed at
    # deep_tool_costs()["bash"]. The patience middleware must charge that
    # cost, not silently default to 0.0 because "bash" is absent from
    # bird_baseline's TOOL_COSTS table.
    mws = agent_code._build_deep_middleware()
    tool_wrapper = next(
        m for m in mws if type(m).__name__ == "tool_wrapper_patience_and_submit"
    )
    request = SimpleNamespace(
        tool_call={"name": "bash", "id": "call-1"},
        runtime=SimpleNamespace(state={"updated_user_patience": 10.0}),
    )
    response = ToolMessage(content="output", tool_call_id="call-1", name="bash")
    out = tool_wrapper.wrap_tool_call(request, lambda _req: response)
    assert out.update["tool_called_patience"] == [deep_tool_costs()["bash"]]


def test_tools_are_bash_submit_when_ask_user_disabled(task_data, tmp_path):
    # Default (non-ambiguous) deep_agent: no ask_user, no user-sim models needed.
    tools = agent_code._build_deep_tools(
        task_data,
        model_user_parsing=None,
        model_user_generator=None,
        catalog_dir=tmp_path,
        pg_env={},
        enable_ask_user=False,
    )
    names = {t.name for t in tools}
    assert names == {"bash", "submit_sql"}
    assert "execute_sql" not in names
    assert "read_file" not in names


def test_tools_include_ask_user_when_enabled(task_data, make_chat_model, tmp_path):
    tools = agent_code._build_deep_tools(
        task_data,
        model_user_parsing=make_chat_model("<s>x</s>"),
        model_user_generator=make_chat_model("<s>y</s>"),
        catalog_dir=tmp_path,
        pg_env={},
        enable_ask_user=True,
    )
    names = {t.name for t in tools}
    assert names == {"bash", "submit_sql", "ask_user"}
    assert "execute_sql" not in names


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
