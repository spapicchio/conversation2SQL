import json
import re
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import (
    ToolCallLimitMiddleware,
    ModelCallLimitMiddleware,
    ModelRetryMiddleware,
    ToolRetryMiddleware,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

from conversation2sql.eval_framework.agent.agent_callback import (
    tool_wrapper_patience_and_submit,
    TOOL_COSTS,
    wrap_model_append_tool_message, check_budget_limit,
)
from conversation2sql.eval_framework.agent.agent_code_state import CustomAgentState
from conversation2sql.eval_framework.agent.prompts import (
    build_bird_interact_agent_messages,
)
from conversation2sql.eval_framework.agent.tools import (
    execute_sql,
    get_all_column_meanings,
    get_schema,
    get_column_meaning,
    get_all_external_knowledge_names,
    get_knowledge_definition,
    get_all_knowledge_definitions,
    return_tool_ask_user,
    submit_sql,
)
from conversation2sql.eval_framework.state import TaskData
from conversation2sql.logger import get_logger

logger = get_logger(__name__)


def run_agent(
        single_task: TaskData,
        model_agent: BaseChatModel,
        model_user_parsing: BaseChatModel,
        model_user_generator: BaseChatModel,
) -> CustomAgentState:
    messages = build_bird_interact_agent_messages(
        params={
            "total_budget": single_task.task_budget,
            "amb_user_query": single_task.amb_user_query,
            # "amb_user_query": 'This is a debug message, call only ask_user as tool with an invented question and return without submitting'
        }
    )

    tools = [
        execute_sql,
        get_all_column_meanings,
        get_schema,
        get_column_meaning,
        get_all_external_knowledge_names,
        get_knowledge_definition,
        get_all_knowledge_definitions,
        return_tool_ask_user(model_user_parsing, model_user_generator),
        submit_sql,
    ]

    agent = create_agent(
        model_agent,
        tools,
        state_schema=CustomAgentState,  # mutable from the tool
        context_schema=TaskData,  # immutable cannot be changed in the tool
        middleware=[  # pyrefly: ignore
            ModelRetryMiddleware(max_delay=60.0, on_failure="error"),
            ToolRetryMiddleware(max_delay=60.0, on_failure="error"),
            ModelCallLimitMiddleware(run_limit=single_task.task_budget + 5),
            ToolCallLimitMiddleware(
                # Maximum tool calls per single invocation (one user message → response cycle).
                # Resets with each new user message.
                run_limit=single_task.task_budget + 5,
                # Maximum tool calls across all runs in a thread (conversation).
                # Persists across multiple invocations with the same thread ID.
                # Requires a checkpointer to maintain state. None means no thread limit.
                thread_limit=single_task.task_budget * 2,
            ),
            check_budget_limit,
            wrap_model_append_tool_message,
            tool_wrapper_patience_and_submit,
        ],
    )

    agent_state: CustomAgentState = {
        "messages": messages,  # pyrefly: ignore,
        "initial_user_patience": single_task.task_budget,
        "updated_user_patience": single_task.task_budget,
        "tool_called_patience": list(),
    }

    response: CustomAgentState = agent.invoke(agent_state, context=single_task)  # pyrefly: ignore
    return _process_agent_response(response)


def _process_agent_response(response: CustomAgentState) -> Any:
    messages = [_process_single_msg(m) for m in response.pop("messages")]
    total_cost = 0
    total_tokens = 0
    mean_prompt_tokens = []
    mean_completion_tokens = []
    tool_calls_in_order = []
    passed = False
    for msg in messages:
        if msg["role"] == "tool":
            passed = msg["content"].get("passed", False)

        total_cost += msg.get("cost_usd", 0)

        total_tokens += msg.get("total_tokens", 0)
        mean_prompt_tokens.append(msg.get("prompt_tokens", 0))
        mean_completion_tokens.append(msg.get("completion_tokens", 0))
        tool_calls_in_order.extend(msg.get("tool_calls", []))

    return {
        **response,
        "total_cost": total_cost,
        "total_tokens": total_tokens,
        "mean_prompt_tokens": sum(mean_prompt_tokens) / len(mean_prompt_tokens),
        "mean_completion_tokens": sum(mean_completion_tokens)
                                  / len(mean_completion_tokens),
        "tool_calls_in_order": tool_calls_in_order,
        "execution_accuracy": passed,
        "messages": messages,
    }


def _process_single_msg(message: BaseMessage) -> dict:
    # https://docs.langchain.com/oss/python/langchain/messages
    base = {
        "role": message.type,  # 'ai' | 'human' | 'system' | 'tool'
        "content": message.content,  # str, list[str | dict]
    }

    if isinstance(message, AIMessage):
        return {**base, **_extract_ai_metadata(message)}

    if isinstance(message, ToolMessage):
        content = base.pop("content")
        clean = re.sub(r"\s*\[SYSTEM NOTE:.*?\]\s*$", "", content, flags=re.DOTALL)
        try:
            clean = json.loads(clean)
        except json.decoder.JSONDecodeError as e:
            clean = {"content": clean, "parse_error": str(e)}

        return {
            **base,
            "tool_name": message.name,
            # 'tool_call_id': message.tool_call_id,
            "status": message.status,  # 'success' | 'error'
            "content": clean,
        }

    # HumanMessage / SystemMessage
    return base


def _extract_ai_metadata(message: AIMessage) -> dict:
    # --- token usage (LangChain-normalised; LiteLLM populates this for all providers) ---
    um = (
            message.usage_metadata or {}
    )  # https://reference.langchain.com/python/langchain-core/messages/ai/UsageMetadata?_gl=1*11ucany*_gcl_au*NDc0Mzc2NTAuMTc3Mjc5MTAyOA..*_ga*MjA2NDMyNTk0Ny4xNzcyNzkxMDI4*_ga_47WX3HKKY2*czE3NzcyODQ1ODckbzQ3JGcwJHQxNzc3Mjg0NTg3JGo2MCRsMCRoMA..

    prompt_tokens = um.get("input_tokens", -1)
    completion_tokens = um.get("output_tokens", -1)
    total_tokens = um.get("total_tokens") or (
        prompt_tokens + completion_tokens
        if prompt_tokens >= 0 and completion_tokens >= 0
        else -1
    )

    # --- response metadata (provider-specific; LiteLLM normalises key names) ---
    meta = message.response_metadata or {}

    # model identifier: ChatLiteLLM sets 'model_name' as "provider/model"
    model_name = meta.get("model_name") or meta.get("model") or "unknown"

    # cost: LiteLLM injects _response_cost into response_metadata
    cost_usd = (
            meta.get("_response_cost")
            or meta.get("response_cost")
            or meta.get("token_usage", {}).get("cost")
            or 0.0
    )

    # tool calls — use LangChain-normalised list (works across all providers)
    tool_calls = [
        {
            "tool_name": tc["name"],
            "arguments": tc["args"],
            "tool_cost": TOOL_COSTS.get(
                tc["name"], -1
            ),  # custom mapping of tool name to cost
        }
        for tc in (message.tool_calls or [])
    ]
    # malformed tool calls the model emitted but couldn't parse
    invalid_tool_calls = [
        {"tool_name": tc.get("name"), "error": tc.get("error")}
        for tc in (message.invalid_tool_calls or [])
    ]

    return {
        # --- token counts ---
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        # fine-grained breakdown (-1 = not reported by this provider/model)
        "model_name": model_name,
        # --- termination ---
        "finish_reason": meta.get("finish_reason", None),
        # --- cost ---
        "cost_usd": cost_usd,
        # --- tool calls ---
        "tool_calls": tool_calls,
        "invalid_tool_calls": invalid_tool_calls,
    }
