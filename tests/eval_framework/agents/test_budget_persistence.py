"""The remaining-budget the agent sees must survive into the serialized record.

`wrap_model_append_tool_message` appends `[SYSTEM NOTE: Remaining budget: x/y]`
to the *transient* model-request copy of the last tool message so the model sees
its budget. That mutation is discarded after the model call, so historically the
budget never reached the persisted state / serialized record and every explorer
"budget spent" came out 0.

The middleware must therefore also persist the budget as structured metadata on
the tool message in state (via the returned `Command`, replacing the message by
id), without changing the content the model sees on later turns.
"""

import json
from types import SimpleNamespace

from langchain_core.messages import ToolMessage

from conversation2sql.eval_framework.agents.bird_baseline.agent_callback import (
    wrap_model_append_tool_message,
)


def _drive(state: dict, messages: list):
    request = SimpleNamespace(state=state, messages=messages)
    resp = wrap_model_append_tool_message.wrap_model_call(
        request, lambda _req: SimpleNamespace()
    )
    return request, resp


def test_persists_remaining_budget_as_metadata():
    tool = ToolMessage(
        content=json.dumps({"passed": False}),
        tool_call_id="c1",
        name="execute_sql",
        id="m1",
    )
    state = {
        "tool_called_patience": [3.0],
        "updated_user_patience": 10.0,
        "initial_user_patience": 12.0,
    }

    request, resp = _drive(state, [tool])

    # The model still sees the note on this turn's last tool message.
    assert "[SYSTEM NOTE: Remaining budget: 7.0/12.0]" in request.messages[-1].content

    # The budget is persisted as structured metadata, replacing the message by id,
    # with the original (note-free) content so later turns' model view is unchanged.
    persisted = resp.command.update["messages"][0]
    assert persisted.id == "m1"
    assert persisted.additional_kwargs["remaining_budget"] == 7.0
    assert persisted.additional_kwargs["total_budget"] == 12.0
    assert "SYSTEM NOTE" not in str(persisted.content)


def test_budget_clamped_at_blocked_floor_is_persisted():
    tool = ToolMessage(
        content="Budget exhausted.", tool_call_id="c1", name="execute_sql", id="m2"
    )
    state = {
        "tool_called_patience": [5.0],
        "updated_user_patience": 1.0,  # 1 - 5 clamps to PATIENCE_BLOCKED (-1)
        "initial_user_patience": 12.0,
    }

    _request, resp = _drive(state, [tool])

    persisted = resp.command.update["messages"][0]
    assert persisted.additional_kwargs["remaining_budget"] == -1.0
    assert persisted.additional_kwargs["total_budget"] == 12.0


def test_no_tool_call_this_turn_persists_no_message():
    # tool_called_patience empty -> nothing deducted, no budget note, no message
    # replacement (the existing early-return path).
    tool = ToolMessage(content="x", tool_call_id="c1", name="execute_sql", id="m3")
    state = {
        "tool_called_patience": [],
        "updated_user_patience": 10.0,
        "initial_user_patience": 12.0,
    }

    _request, resp = _drive(state, [tool])

    assert resp.command is None
