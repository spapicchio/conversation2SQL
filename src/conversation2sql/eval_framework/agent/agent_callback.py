import json
from typing import Callable, Any

from langchain.agents.middleware import wrap_tool_call, before_model
from langchain_core.messages import ToolMessage
from langgraph.prebuilt import ToolRuntime
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.runtime import Runtime
from langgraph.types import Command

from conversation2sql.eval_framework.agent.agent_code_state import CustomAgentState
from conversation2sql.eval_framework.agent.tools import DB_TOOL_COSTS, USER_TOOL_COSTS
from conversation2sql.eval_framework.state import TaskData

TOOL_COSTS: dict[str, float] = {**DB_TOOL_COSTS, **USER_TOOL_COSTS}


@wrap_tool_call
def tool_wrapper_append_budget(
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command]
) -> ToolMessage | Command:
    tool_name = request.tool_call['name']
    cost = TOOL_COSTS.get(tool_name, 0.0)
    runtime: ToolRuntime[TaskData, CustomAgentState] = request.runtime

    user_patience = runtime.state["updated_user_patience"]
    initial_user_patience = runtime.state["initial_user_patience"]

    if user_patience < cost and tool_name != "submit_sql":
        return ToolMessage(
            content=f"Budget exhausted ({user_patience:.1f} remaining). "
                    "You MUST call submit_sql now with your best SQL.",
            tool_call_id=request.tool_call["id"],
        )

    # Execute the tool normally
    response = handler(request)

    # Deduct cost and compute new patience
    new_patience = max(user_patience - cost, -1)

    modified_content = (
        f"{response.content}"
        f"\n\n[SYSTEM NOTE: Remaining budget: {new_patience:.1f}/{initial_user_patience:.1f}]"
    )

    # Return a Command that both updates messages AND decrements user_patience
    return Command(
        update={
            "messages": [response.model_copy(deep=True, update={"content": modified_content})],
            "updated_user_patience": new_patience,
        },
    )


@wrap_tool_call
def tool_wrapper_submit_sql(
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command]
) -> ToolMessage | Command:
    response = handler(request)

    tool_name = request.tool_call['name']
    if tool_name == "submit_sql" and json.loads(response.content)['passed']:
        return Command(
            update={
                "messages": [ToolMessage(
                    content=json.loads(response.content)['message'],
                    tool_call_id=request.tool_call["id"],
                )],
            },
            goto="end",
        )

    return response


@before_model(state_schema=CustomAgentState, can_jump_to=["end"])
def model_budget_exhausted(state: CustomAgentState, runtime: Runtime) -> dict[str, Any] | None:
    updated_user_patience = state.get("updated_user_patience", 0)
    if updated_user_patience < 0:
        return {"jump_to": "end"}
    return None
