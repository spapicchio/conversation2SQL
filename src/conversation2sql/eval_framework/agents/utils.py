from __future__ import annotations

import json
import re
from typing import Any

from jinja2 import Template
from langchain_core.messages import BaseMessage, AIMessage, ToolMessage
from langchain_litellm import ChatLiteLLM

from conversation2sql.eval_framework.agents.bird_baseline.agent_code_state import (
    CustomAgentState,
)


def utils_process_agent_response(response: CustomAgentState, tool_costs: dict) -> Any:
    messages = [utils_process_single_msg(m, tool_costs=tool_costs) for m in response.pop("messages")]
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
        "total_prompt_tokens": sum(mean_prompt_tokens),
        "total_completion_tokens": sum(mean_completion_tokens),
        "mean_prompt_tokens": sum(mean_prompt_tokens) / len(mean_prompt_tokens),
        "mean_completion_tokens": sum(mean_completion_tokens)
        / len(mean_completion_tokens),
        "tool_calls_in_order": tool_calls_in_order,
        "execution_accuracy": passed,
        "messages": messages,
    }


def utils_process_single_msg(message: BaseMessage, tool_costs: dict) -> dict:
    # https://docs.langchain.com/oss/python/langchain/messages
    base = {
        "role": message.type,  # 'ai' | 'human' | 'system' | 'tool'
        "content": message.content,  # str, list[str | dict]
    }

    if isinstance(message, AIMessage):
        return {**base, **utils_extract_ai_metadata(message, tool_costs=tool_costs)}

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


def utils_extract_ai_metadata(message: AIMessage, tool_costs: dict) -> dict:
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
            "tool_cost": tool_costs.get(
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


def utils_create_model(
    model_name: str,
    model_provider: str,
    temperature: float,
    max_tokens: int,
    top_p: float | None = None,
) -> ChatLiteLLM:
    # LiteLLM uses "{provider}/{model}" format
    # https://docs.litellm.ai/docs/providers

    litellm_model = f"{model_provider}/{model_name}"
    # reasoning + result
    model_kwargs = {"max_completion_tokens": max_tokens + 2000}
    kwargs = dict(
        model=litellm_model,
        temperature=temperature,
        max_tokens=max_tokens,
        model_kwargs=model_kwargs,
    )
    if top_p is not None:
        kwargs["top_p"] = top_p

    return ChatLiteLLM(**kwargs)


def utils_render_jinja(template_str: str, params: dict) -> str:
    return Template(template_str).render(**params)


def utils_build_messages(
    system_str: str | None,
    user_str: str,
    params: dict,
) -> list[dict]:
    msgs: list[dict] = []
    if system_str:
        msgs.append(dict(role="system", content=utils_render_jinja(system_str, params)))
    msgs.append(dict(role="user", content=utils_render_jinja(user_str, params)))
    return msgs
