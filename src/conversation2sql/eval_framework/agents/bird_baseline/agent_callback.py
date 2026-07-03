import json
from dataclasses import replace
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

# Sentinel values written to ``updated_user_patience`` to encode a terminal episode.
# Live budget is always > -1. The state reducer keeps the *smallest* write (see
# ``CustomAgentState``), so the sentinels are ordered by priority: if the agent
# emits several submit_sql calls in one super-step, the success sentinel (the
# smallest) wins over a forced/exhausted one.
PATIENCE_BLOCKED = -1.0           # budget too low for the requested tool; agent must still submit (not terminal)
PATIENCE_SUBMIT_EXHAUSTED = -2.0  # submit_sql reached only because the budget ran out (SQL may be wrong)
PATIENCE_SUBMIT_PASSED = -3.0     # submit_sql returned passed=True -> clean success


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
    patience = state["updated_user_patience"]
    # Only a terminal submit_sql drives patience below the blocked floor (-1).
    # Anything at PATIENCE_BLOCKED or above means the episode is still live (the
    # agent is being pushed to submit, not stopped).
    if patience is None or patience > PATIENCE_SUBMIT_EXHAUSTED:
        return None

    # A passing submit is a clean finish: the submit_sql ToolMessage already records
    # the outcome, so end the run silently rather than tacking on a misleading
    # "patience exhausted" note.
    if patience <= PATIENCE_SUBMIT_PASSED:
        return {"jump_to": "end"}

    # PATIENCE_SUBMIT_EXHAUSTED: the agent submitted only because it ran out of
    # budget, so the patience-exhaustion note is the accurate explanation.
    return {
        "messages": [AIMessage("Conversation limit reached. User patience exhausted.")],
        "jump_to": "end",
    }


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
        updated_user_patience - sum(tool_called_patience), PATIENCE_BLOCKED
    )
    update: dict[str, Any] = {
        "updated_user_patience": updated_user_patience,
        "tool_called_patience": Overwrite([]),
    }
    message = request.messages[-1]
    # this must be a tool call
    if isinstance(message, ToolMessage):
        # Persist the budget as structured metadata on the *same* tool message
        # (matched by id, so the add_messages reducer replaces it in place) with
        # the original, note-free content. This is what reaches the serialized
        # record; mutating ``request.messages`` below only affects the transient
        # copy the model sees this turn, so it never survives into state on its
        # own. Built from the original content *before* the in-place mutation.
        update["messages"] = [
            message.model_copy(
                update={
                    "additional_kwargs": {
                        **message.additional_kwargs,
                        "remaining_budget": updated_user_patience,
                        "total_budget": initial_user_patience,
                    }
                }
            )
        ]
        # Show the model its remaining budget on this turn (transient request copy).
        request.messages[-1].content = (
            f"{message.content}"
            f"\n\n[SYSTEM NOTE: Remaining budget: {updated_user_patience:.1f}/{initial_user_patience:.1f}]"
        )

    response = handler(request)
    return ExtendedModelResponse(
        model_response=response,
        command=Command(update=update),
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
                'updated_user_patience': PATIENCE_BLOCKED
            },
        )

    response = handler(request)

    if tool_name == "submit_sql":
        tool_output = json.loads(response.content)

        # Decide whether this submit ends the episode, and *why*. A passing submit
        # is a clean success; a non-passing submit reached only because the budget
        # ran out is a forced finalize. The two map to distinct terminal sentinels
        # so `check_budget_limit` can word the closing message correctly.
        if tool_output["passed"]:
            terminal_patience = PATIENCE_SUBMIT_PASSED
        elif user_patience < 0:
            terminal_patience = PATIENCE_SUBMIT_EXHAUSTED
        else:
            terminal_patience = None  # wrong SQL but budget remains → let the agent retry

        if terminal_patience is not None:
            # Preserve the FULL tool response (with the `passed` field and the
            # submit_sql tool name) so downstream metric extraction in
            # `utils_process_agent_response` can read `execution_accuracy` and the
            # explorer renders the turn with its tool name. Rewriting the content
            # to just the message string would drop `passed` and break scoring.
            return Command(
                update={
                    "messages": [response.model_copy(deep=True)],
                    "updated_user_patience": terminal_patience,
                },
            )

        # SQL did not pass yet → fall through to record the cost and let the agent retry.

    # deepagents' state-updating tools (write_todos / task / write_file / edit_file)
    # return a langgraph `Command` rather than a `ToolMessage`. A Command has no
    # `model_copy`; rewrapping it would also drop the tool's own state update (its
    # messages plus e.g. `todos` / `files`). Thread the cost into the Command's
    # existing update instead — `tool_called_patience` uses an additive reducer,
    # so it accumulates alongside any cost already written this super-step.
    if isinstance(response, Command) and isinstance(response.update, dict):
        prior = response.update.get("tool_called_patience", [])
        merged = {**response.update, "tool_called_patience": [*prior, cost]}
        return replace(response, update=merged)

    # Default path: persist the tool message and record the cost so that
    # `wrap_model_append_tool_message` can deduct it from the patience budget.
    return Command(
        update={
            "messages": [response.model_copy(deep=True)],
            "tool_called_patience": [cost],
        },
    )


def make_tool_wrapper_patience_and_submit_silent(
    tool_costs: dict[str, float], submit_tool_name: str = "submit"
):
    """Build a patience/submit tool-wrapper middleware for a *silent* submit tool.

    Unlike ``make_tool_wrapper_patience_and_submit`` (bird_baseline's
    ``submit_sql``, which reports ``passed`` and lets a failing submit retry),
    ``submit_tool_name`` here carries no pass/fail signal at all: calling it
    always ends the episode. This is maintenance_agent's silent-submit design
    (see docs/superpowers/specs/2026-07-03-maintenance-agent-baseline-design.md)
    — the agent never learns whether its SQL was graded correct, so there is
    no retry-on-failed-submit branch to preserve.
    """

    @wrap_tool_call
    def tool_wrapper_patience_and_submit_silent(
            request: ToolCallRequest,
            handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        tool_name = request.tool_call["name"]
        cost = tool_costs.get(tool_name, 0.0)
        runtime: ToolRuntime[TaskData, CustomAgentState] = request.runtime
        user_patience = runtime.state["updated_user_patience"]

        # Block any non-submit tool whose cost would exceed the remaining
        # budget. The submit tool is always allowed so the agent can finalize
        # even when out of budget.
        if user_patience < cost and tool_name != submit_tool_name:
            return Command(
                update={
                    "messages": [
                        ToolMessage(
                            content=f"Budget exhausted ({user_patience:.1f} remaining). "
                                    f"You MUST call {submit_tool_name} now.",
                            tool_call_id=request.tool_call["id"],
                            name=tool_name,
                        )
                    ],
                    'updated_user_patience': PATIENCE_BLOCKED
                },
            )

        response = handler(request)

        if tool_name == submit_tool_name:
            # No passed/failed signal exists for a silent submit — calling it
            # is always terminal. Only the *reason* differs: a voluntary
            # submit with budget left is a clean finish; a submit reached
            # only after being blocked is a forced finalize. Reusing the two
            # bird_baseline sentinels keeps check_budget_limit's message
            # wording and downstream turn-classification code working
            # unmodified — no correctness signal is attached to either
            # sentinel here, unlike in bird_baseline where PASSED specifically
            # means "SQL was graded correct."
            terminal_patience = (
                PATIENCE_SUBMIT_EXHAUSTED if user_patience < 0 else PATIENCE_SUBMIT_PASSED
            )
            return Command(
                update={
                    "messages": [response.model_copy(deep=True)],
                    "updated_user_patience": terminal_patience,
                },
            )

        if isinstance(response, Command) and isinstance(response.update, dict):
            prior = response.update.get("tool_called_patience", [])
            merged = {**response.update, "tool_called_patience": [*prior, cost]}
            return replace(response, update=merged)

        return Command(
            update={
                "messages": [response.model_copy(deep=True)],
                "tool_called_patience": [cost],
            },
        )

    return tool_wrapper_patience_and_submit_silent
