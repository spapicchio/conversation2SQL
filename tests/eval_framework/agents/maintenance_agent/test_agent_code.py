from types import SimpleNamespace

import pytest
from langchain_core.messages import ToolMessage

from conversation2sql.eval_framework.agents.maintenance_agent import agent_code
from conversation2sql.eval_framework.agents.maintenance_agent.catalog_seed import (
    maintenance_tool_costs,
)


def test_middleware_includes_silent_patience_budget(task_data):
    mws = agent_code._build_maintenance_middleware()
    names = [type(m).__name__ for m in mws]
    assert "tool_wrapper_patience_and_submit_silent" in names


def test_bash_tool_call_is_charged_maintenance_cost(task_data):
    mws = agent_code._build_maintenance_middleware()
    tool_wrapper = next(
        m for m in mws if type(m).__name__ == "tool_wrapper_patience_and_submit_silent"
    )
    request = SimpleNamespace(
        tool_call={"name": "bash", "id": "call-1"},
        runtime=SimpleNamespace(state={"updated_user_patience": 10.0}),
    )
    response = ToolMessage(content="output", tool_call_id="call-1", name="bash")
    out = tool_wrapper.wrap_tool_call(request, lambda _req: response)
    assert out.update["tool_called_patience"] == [maintenance_tool_costs()["bash"]]


def test_tools_are_bash_write_run_submit_when_ask_user_disabled(task_data, tmp_path):
    (tmp_path / "queries").mkdir()
    (tmp_path / "queries" / "answer.sql").write_text("-- stub\n")
    tools = agent_code._build_maintenance_tools(
        task_data,
        model_user_parsing=None,
        model_user_generator=None,
        catalog_dir=tmp_path,
        pg_env={},
        enable_ask_user=False,
    )
    names = {t.name for t in tools}
    assert names == {"bash", "write_query", "run_tests", "submit"}
    assert "comment_on_issue" not in names


def test_tools_include_comment_on_issue_when_enabled(task_data, make_chat_model, tmp_path):
    (tmp_path / "queries").mkdir()
    (tmp_path / "queries" / "answer.sql").write_text("-- stub\n")
    tools = agent_code._build_maintenance_tools(
        task_data,
        model_user_parsing=make_chat_model("<s>x</s>"),
        model_user_generator=make_chat_model("<s>y</s>"),
        catalog_dir=tmp_path,
        pg_env={},
        enable_ask_user=True,
    )
    names = {t.name for t in tools}
    assert names == {"bash", "write_query", "run_tests", "submit", "comment_on_issue"}


def test_catalog_dir_cleaned_up_on_exception(
    task_data, make_chat_model, tmp_path, monkeypatch
):
    import conversation2sql.eval_framework.agents.maintenance_agent.agent_code as ac

    created = {}

    def fake_materialize(task):
        d = tmp_path / "workspace"
        d.mkdir()
        (d / "queries").mkdir()
        (d / "queries" / "answer.sql").write_text("-- stub\n")
        created["dir"] = d
        return d

    class _Boom:
        def invoke(self, *a, **k):
            raise RuntimeError("boom")

    monkeypatch.setattr(ac, "materialize_maintenance_workspace", fake_materialize)
    monkeypatch.setattr(ac, "create_agent", lambda *a, **k: _Boom())

    with pytest.raises(RuntimeError):
        ac.run_agent_maintenance(
            task_data,
            model_agent=make_chat_model("x"),
            model_user_parsing=make_chat_model("<s>x</s>"),
            model_user_generator=make_chat_model("<s>y</s>"),
        )
    assert not created["dir"].exists()


def test_predicted_sql_read_from_disk_after_run(
    task_data, make_chat_model, tmp_path, monkeypatch
):
    import conversation2sql.eval_framework.agents.maintenance_agent.agent_code as ac

    def fake_materialize(task):
        d = tmp_path / "workspace"
        d.mkdir()
        (d / "queries").mkdir()
        (d / "queries" / "answer.sql").write_text("SELECT 1;")
        return d

    monkeypatch.setattr(ac, "materialize_maintenance_workspace", fake_materialize)
    monkeypatch.setattr(ac, "build_pg_env", lambda dsn: {})

    class _Recorder:
        def invoke(self, state, *a, **k):
            return {"messages": []}

    monkeypatch.setattr(ac, "create_agent", lambda *a, **k: _Recorder())
    monkeypatch.setattr(
        ac, "submit_sql_impl", lambda **kwargs: {"passed": True, "message": "ok"}
    )
    monkeypatch.setattr(
        ac, "utils_process_agent_response", lambda *a, **k: {"execution_accuracy": False}
    )

    output = ac.run_agent_maintenance(
        task_data,
        model_agent=make_chat_model("x"),
        model_user_parsing=make_chat_model("<s>x</s>"),
        model_user_generator=make_chat_model("<s>y</s>"),
    )

    assert output["predicted_sql"] == "SELECT 1;"


def test_execution_accuracy_overridden_from_hidden_grader_not_from_run_tests(
    task_data, make_chat_model, tmp_path, monkeypatch
):
    # Regression: utils_process_agent_response's generic `passed`-key scan
    # would otherwise pick up run_tests' *structural* pass/fail (also a dict
    # tool message with a "passed" key) instead of the real grade. Pin that
    # the hidden submit_sql_impl result always wins.
    import conversation2sql.eval_framework.agents.maintenance_agent.agent_code as ac

    def fake_materialize(task):
        d = tmp_path / "workspace"
        d.mkdir()
        (d / "queries").mkdir()
        (d / "queries" / "answer.sql").write_text("SELECT 1;")
        return d

    monkeypatch.setattr(ac, "materialize_maintenance_workspace", fake_materialize)
    monkeypatch.setattr(ac, "build_pg_env", lambda dsn: {})

    class _Recorder:
        def invoke(self, state, *a, **k):
            return {"messages": []}

    monkeypatch.setattr(ac, "create_agent", lambda *a, **k: _Recorder())
    # Hidden grader says the SQL is WRONG...
    monkeypatch.setattr(
        ac, "submit_sql_impl", lambda **kwargs: {"passed": False, "message": "wrong"}
    )
    # ...even though utils_process_agent_response's own (unrelated) scan would
    # have reported True (simulating a run_tests structural pass leaking in).
    monkeypatch.setattr(
        ac, "utils_process_agent_response", lambda *a, **k: {"execution_accuracy": True}
    )

    output = ac.run_agent_maintenance(
        task_data,
        model_agent=make_chat_model("x"),
        model_user_parsing=make_chat_model("<s>x</s>"),
        model_user_generator=make_chat_model("<s>y</s>"),
    )

    assert output["execution_accuracy"] is False


def test_system_and_user_travel_together_in_initial_state(
    task_data, make_chat_model, tmp_path, monkeypatch
):
    import conversation2sql.eval_framework.agents.maintenance_agent.agent_code as ac

    def fake_materialize(task):
        d = tmp_path / "workspace"
        d.mkdir()
        (d / "queries").mkdir()
        (d / "queries" / "answer.sql").write_text("-- stub\n")
        return d

    monkeypatch.setattr(ac, "materialize_maintenance_workspace", fake_materialize)
    monkeypatch.setattr(ac, "build_pg_env", lambda dsn: {})

    captured = {}

    class _Recorder:
        def invoke(self, state, *a, **k):
            captured["state"] = state
            return {"messages": []}

    monkeypatch.setattr(
        ac,
        "create_agent",
        lambda *a, **k: (captured.setdefault("kwargs", k), _Recorder())[1],
    )
    monkeypatch.setattr(
        ac, "submit_sql_impl", lambda **kwargs: {"passed": False, "message": "x"}
    )
    monkeypatch.setattr(ac, "utils_process_agent_response", lambda *a, **k: {})

    ac.run_agent_maintenance(
        task_data,
        model_agent=make_chat_model("x"),
        model_user_parsing=make_chat_model("<s>x</s>"),
        model_user_generator=make_chat_model("<s>y</s>"),
    )

    assert "system_prompt" not in captured["kwargs"]
    msgs = captured["state"]["messages"]
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
