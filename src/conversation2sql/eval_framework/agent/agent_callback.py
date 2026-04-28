import json
from typing import Callable

from langchain.agents.middleware import (
    wrap_tool_call,
    wrap_model_call,
    ModelRequest,
    ModelResponse,
    ExtendedModelResponse,
)
from langchain_core.messages import ToolMessage
from langgraph.prebuilt import ToolRuntime
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command, Overwrite

from conversation2sql.eval_framework.agent.agent_code_state import CustomAgentState
from conversation2sql.eval_framework.agent.tools import DB_TOOL_COSTS, USER_TOOL_COSTS
from conversation2sql.eval_framework.state import TaskData

TOOL_COSTS: dict[str, float] = {**DB_TOOL_COSTS, **USER_TOOL_COSTS}


@wrap_model_call(state_schema=CustomAgentState)
def wrap_model_append_tool_message(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse[CustomAgentState]],
) -> ExtendedModelResponse:
    # https://docs.langchain.com/oss/python/langgraph/use-graph-api#bypass-reducers-with-overwrite
    tool_called_patience = request.state["tool_called_patience"]  # pyrefly: ignore

    if not tool_called_patience:
        response = handler(request)
        return ExtendedModelResponse(model_response=response)
    else:
        initial_user_patience = request.state["initial_user_patience"]  # pyrefly: ignore
        updated_user_patience = request.state["updated_user_patience"]  # pyrefly: ignore
        updated_user_patience = max(
            updated_user_patience - sum(tool_called_patience), -1
        )
        # update user_patience and tool called patience
        command = Command(
            update={
                "updated_user_patience": updated_user_patience,
                "tool_called_patience": Overwrite([]),
            }
        )
        message = request.messages[-1]
        # this must be a tool call
        if isinstance(message, ToolMessage):
            modified_content = (
                f"{message.content}"
                f"\n\n[SYSTEM NOTE: Remaining budget: {updated_user_patience:.1f}/{initial_user_patience:.1f}]"
            )
            request.messages[-1].content = modified_content

        response = handler(request)
        return ExtendedModelResponse(
            model_response=response,
            command=command,
        )


@wrap_tool_call
def tool_wrapper_patience_and_submit(
    request: ToolCallRequest,
    handler: Callable[[ToolCallRequest], ToolMessage | Command],
) -> ToolMessage | Command:
    tool_name = request.tool_call["name"]
    cost = TOOL_COSTS.get(tool_name, 0.0)
    runtime: ToolRuntime[TaskData, CustomAgentState] = request.runtime
    user_patience = runtime.state["updated_user_patience"]

    # Block any non-`submit_sql` tool whose cost would exceed the remaining budget.
    # `submit_sql` is always allowed so the agent can finalize even when out of budget.
    if user_patience < cost and tool_name != "submit_sql":
        return ToolMessage(
            content=f"Budget exhausted ({user_patience:.1f} remaining). "
            "You MUST call submit_sql now with your best SQL.",
            tool_call_id=request.tool_call["id"],
        )

    response = handler(request)

    # `submit_sql` may terminate the graph: either the budget is gone, or the SQL passed.
    if tool_name == "submit_sql":
        tool_output = json.loads(response.content)
        message = tool_output["message"]

        if user_patience < 0:
            # Out of budget → end the conversation with an explanatory note.
            return Command(
                update={
                    "messages": [
                        ToolMessage(
                            content=f"{message}\n\n [SYSTEM NOTE] Budget exhausted conversation ended.",
                            tool_call_id=request.tool_call["id"],
                        )
                    ],
                },
                goto="end",
            )

        if tool_output["passed"]:
            # SQL passed evaluation → end the conversation successfully.
            return Command(
                update={
                    "messages": [
                        ToolMessage(
                            content=message,
                            tool_call_id=request.tool_call["id"],
                        )
                    ],
                },
                goto="end",
            )

        # SQL did not pass yet → fall through to record the cost and let the agent retry.

    # Default path: persist the tool message and record the cost so that
    # `wrap_model_append_tool_message` can deduct it from the patience budget.
    return Command(
        update={
            "messages": [response.model_copy(deep=True)],
            "tool_called_patience": [cost],
        },
    )


# """

# SELECT:
# SELECT ROUND(CAST(om."mttrh" / (om."mtbfh" + om."mttrh") AS numeric), 4)
# FROM:
# FROM operational_metrics om
# JOIN:
# JOIN plant_record pr ON om."snapops" = pr."snapkey"
# JOIN:
# JOIN plants p ON pr."sitetie" = p."sitekey"
# WHERE:
# WHERE LOWER(p."sitelabel") = 'solar plant west davidport'
# LIMIT:
# LIMIT 1;



"""
SELECT om.mttrh / (om.mtbfh + om.mttrh) AS downtime_score\nFROM plant_record pr\nJOIN operational_metrics om ON pr.snapkey = om.snapops\nJOIN plants p ON pr.sitetie = p.sitekey\nWHERE p.sitelabel = 'Solar Plant West Davidport';"}
"""
# """

[
    'SELECT ROUND(CAST(om."mttrh" / (om."mtbfh" + om."mttrh") AS numeric), 4)\nFROM operational_metrics om\nJOIN plant_record pr ON om."snapops" = pr."snapkey"\nJOIN plants p ON pr."sitetie" = p."sitekey"\nWHERE LOWER(p."sitelabel") = \'solar plant west davidport\'\nLIMIT 1;'
]
