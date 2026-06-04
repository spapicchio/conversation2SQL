import json
from typing import Callable, Any

from langchain.agents.middleware import (
    wrap_tool_call,
    wrap_model_call,
    ModelRequest,
    ModelResponse,
    ExtendedModelResponse, before_model,
)
from langchain_core.messages import ToolMessage, AIMessage
from langgraph.prebuilt import ToolRuntime
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.runtime import Runtime
from langgraph.types import Command, Overwrite

from conversation2sql.eval_framework.agents.bird_baseline.agent_code_state import CustomAgentState
from conversation2sql.eval_framework.agents.bird_baseline.tools import TOOL_COSTS
from conversation2sql.eval_framework.state import TaskData


def _strip_thinking_from_history(messages: list) -> None:
    """Flatten reasoning out of *historical* assistant turns, in place.

    Reasoning models (e.g. Qwen3) return an AIMessage whose ``content`` is a list
    of blocks like ``[{"type": "thinking", ...}, {"type": "text", ...}]``.
    LangChain echoes that list straight back as the assistant ``content`` on the
    next request. The Qwen3 chat template mis-renders a prior assistant turn that
    carries a ``thinking`` block in its content: the next generation comes back
    ``finish_reason=stop`` with NO tool call (the model writes the call inside a
    ``<think>`` block that the tool parser never sees), so the agent loop dies
    before any SQL is submitted.

    Reproduced directly against vLLM: the same 2-turn tool conversation succeeds
    when the prior assistant content is a plain string and fails when it is a
    list with a thinking block. Tool calls live in ``tool_calls`` (not content),
    so dropping the reasoning blocks and keeping only ``text`` is lossless for the
    agent loop and makes multi-turn tool calling work in thinking mode.


    Note: Taken from https://huggingface.co/Qwen/Qwen3.5-9B
    "No Thinking Content in History: In multi-turn conversations, 
    the historical model output should only include the final output part and does
    not need to include the thinking content. 
    It is implemented in the provided chat template in Jinja2. 
    However, for frameworks that do not directly use the Jinja2 chat template,
    it is up to the developers to ensure that the best practice is followed."
    """
    for m in messages:
        if isinstance(m, AIMessage) and isinstance(m.content, list):
            thinking_blocks: list[str] = []
            text_blocks: list[str] = []

            for block in m.content:
                if not isinstance(block, dict):
                    continue
                block_type = block.get("type")
                if block_type == "thinking":
                    thinking_blocks.append(block.get("thinking", ""))
                elif block_type == "text":
                    text_blocks.append(block.get("text", ""))

            if thinking_blocks:
                meta = m.response_metadata or {}
                if not isinstance(meta, dict):
                    meta = {}
                if not meta.get("thinking"):
                    meta["thinking"] = "\n\n".join(
                        t for t in thinking_blocks if t
                    )
                    m.response_metadata = meta

            m.content = "".join(text_blocks)


@wrap_model_call(state_schema=CustomAgentState)
def sanitize_thinking_history(
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse[CustomAgentState]],
) -> ModelResponse:
    """Strip reasoning blocks from assistant history before each model call.

    See ``_strip_thinking_from_history`` for why this is required for multi-turn
    tool calling with thinking-enabled models.
    """
    _strip_thinking_from_history(request.messages)
    return handler(request)


@before_model(can_jump_to=["end"])
def check_budget_limit(state: CustomAgentState, runtime: Runtime) -> dict[str, Any] | None:
    if state["updated_user_patience"] < -1:
        return {
            "messages": [AIMessage("Conversation limit reached. User patience exhausted")],
            "jump_to": "end"
        }
    return None


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
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f"Budget exhausted ({user_patience:.1f} remaining). "
                                "You MUST call submit_sql now with your best SQL.",
                        tool_call_id=request.tool_call["id"],
                        # Name the message after the blocked tool so downstream
                        # serialisation/explorer code never sees a None tool name.
                        name=tool_name,
                    )
                ],
                'updated_user_patience': -1
            },
        )

    response = handler(request)

    if tool_name == "submit_sql":
        tool_output = json.loads(response.content)

        # Either out of budget or the SQL passed → the episode is terminal.
        # Preserve the FULL tool response (with the `passed` field and the
        # submit_sql tool name) so downstream metric extraction in
        # `utils_process_agent_response` can read `execution_accuracy` and the
        # explorer renders the turn with its tool name. Rewriting the content
        # to just the message string would drop `passed` and break scoring.
        if user_patience < 0 or tool_output["passed"]:
            return Command(
                update={
                    "messages": [response.model_copy(deep=True)],
                    "updated_user_patience": -2,
                },
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
