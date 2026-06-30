"""The budget note appended to tool messages must be surfaced to the explorer.

`wrap_model_append_tool_message` appends `[SYSTEM NOTE: Remaining budget: x/y]`
to each tool message. `utils_process_single_msg` strips that note from the
rendered content but should expose the numbers as `remaining_budget` /
`total_budget` so the explorer can show the budget diminishing across a trace.
"""

import json

from langchain_core.messages import ToolMessage

from conversation2sql.eval_framework.agents.utils import utils_process_single_msg


def test_tool_message_exposes_remaining_budget():
    content = json.dumps({"passed": True}) + (
        "\n\n[SYSTEM NOTE: Remaining budget: 7.0/12.0]"
    )
    msg = ToolMessage(content=content, tool_call_id="1", name="execute_sql")

    out = utils_process_single_msg(msg, tool_costs={})

    assert out["remaining_budget"] == 7.0
    assert out["total_budget"] == 12.0
    # The note is still stripped from the visible content.
    assert out["content"] == {"passed": True}


def test_tool_message_handles_negative_remaining_budget():
    content = "Budget exhausted.\n\n[SYSTEM NOTE: Remaining budget: -1.0/12.0]"
    msg = ToolMessage(content=content, tool_call_id="1", name="execute_sql")

    out = utils_process_single_msg(msg, tool_costs={})

    assert out["remaining_budget"] == -1.0
    assert out["total_budget"] == 12.0


def test_tool_message_exposes_budget_from_metadata():
    # The middleware persists the budget as structured metadata (not note text)
    # so the model-facing content is unchanged; the serializer must read it.
    msg = ToolMessage(
        content=json.dumps({"passed": True}),
        tool_call_id="1",
        name="execute_sql",
        additional_kwargs={"remaining_budget": 5.0, "total_budget": 12.0},
    )

    out = utils_process_single_msg(msg, tool_costs={})

    assert out["remaining_budget"] == 5.0
    assert out["total_budget"] == 12.0
    assert out["content"] == {"passed": True}


def test_tool_message_without_note_has_no_budget_fields():
    msg = ToolMessage(
        content=json.dumps({"passed": False}), tool_call_id="1", name="submit_sql"
    )

    out = utils_process_single_msg(msg, tool_costs={})

    assert "remaining_budget" not in out
    assert "total_budget" not in out
