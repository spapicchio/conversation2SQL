from __future__ import annotations
import json
import re

from jinja2 import Template
from langchain_core.messages import BaseMessage, AIMessage, ToolMessage
from langchain_litellm import ChatLiteLLM

from conversation2sql.eval_framework.agents.utils_extract_sql_from_response import (
    extract_sql_from_response,
)


def utils_single_msg_to_str(message: BaseMessage) -> str:
    """Convert a single LangChain `BaseMessage` to a human-readable string.

    This is used for logging and test assertions where we want a simple
    string representation of the message content. For `ToolMessage` objects
    it attempts to JSON-decode the content for better readability.

    Parameters
    - message: A LangChain `BaseMessage` (or subclass) instance.

    Returns
    A string representation of the message content.
    """

    if isinstance(message.content, str):
        raw_text = message.content
    elif isinstance(message.content, list):
        raw_text = "\n\n".join(
            f"# {block['type']}\n{block[block['type']]}"
            if isinstance(block, dict) and "type" in block and block["type"] in block
            else str(block)
            for block in message.content
        )
    else:
        raw_text = str(message.content)

    return raw_text


def utils_extract_sql_from_ai_message(message: AIMessage) -> str | None:
    """Extract a SQL block from an `AIMessage`.

    Reasoning models emit `content` as a list of blocks; prefer the `text`
    block over the `thinking` block so we parse the answer rather than the
    chain-of-thought, then fall back to the whole stringified content so
    non-reasoning models still work.
    """
    if isinstance(message.content, list):
        for block in message.content:
            if isinstance(block, dict) and block.get("type") == "text":
                sql = extract_sql_from_response(block["text"])
                if sql is not None:
                    return sql
    return extract_sql_from_response(utils_single_msg_to_str(message))


def utils_process_single_msg(message: BaseMessage, tool_costs: dict) -> dict:
    """Convert a single LangChain `BaseMessage` to a serialisable dict.

    The returned dict always includes `role` and `content`. For `AIMessage`
    objects it appends token/cost/finish metadata extracted via
    `utils_extract_ai_metadata`. For `ToolMessage` objects it attempts to
    JSON-decode the tool output and exposes `tool_name` and `status`.

    Parameters
    - message: A LangChain `BaseMessage` (or subclass) instance.
    - tool_costs: Mapping from tool name -> cost used to annotate tool calls

    Returns
    A dict representation suitable for logging and downstream metrics.
    """

    # https://docs.langchain.com/oss/python/langchain/messages
    base = {
        "role": message.type,  # 'ai' | 'human' | 'system' | 'tool'
        "content": message.content,  # str, list[str | dict]
    }

    if isinstance(message, AIMessage):
        meta = message.response_metadata or {}
        thinking = meta.get("thinking") if isinstance(meta, dict) else None
        thinking_field = {"thinking": thinking} if thinking else {}
        return {
            **base,
            **utils_extract_ai_metadata(message, tool_costs=tool_costs),
            **thinking_field,
        }

    if isinstance(message, ToolMessage):
        base.pop("content")
        # ToolMessage.content may be a list of content blocks (LangChain spec),
        # not just a str — normalise to text before the regex / JSON decode.
        content = utils_single_msg_to_str(message)
        # The budget middleware appends "[SYSTEM NOTE: Remaining budget: x/y]" to
        # the tool message before the next model call. Capture the numbers so the
        # explorer can show how the budget diminishes, then strip the note from
        # the content the same way it was stripped before.
        budget = _parse_remaining_budget(content)
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
            **budget,
        }

    # HumanMessage / SystemMessage
    return base


_REMAINING_BUDGET_RE = re.compile(
    r"\[SYSTEM NOTE: Remaining budget:\s*(-?\d+(?:\.\d+)?)\s*/\s*(-?\d+(?:\.\d+)?)\]"
)


def _parse_remaining_budget(content: str) -> dict:
    """Extract the `[SYSTEM NOTE: Remaining budget: x/y]` note into fields.

    The budget middleware (`wrap_model_append_tool_message`) annotates each tool
    message with the patience budget the model saw on the next call. We surface
    `remaining_budget` / `total_budget` so the explorer can show the budget
    diminishing across the trace. Returns an empty dict when no note is present
    (e.g. a terminal `submit_sql` message that no model call followed).
    """
    match = _REMAINING_BUDGET_RE.search(content)
    if not match:
        return {}
    return {
        "remaining_budget": float(match.group(1)),
        "total_budget": float(match.group(2)),
    }


def utils_extract_ai_metadata(message: AIMessage, tool_costs: dict) -> dict:
    """Extract normalised metadata from an `AIMessage`.

    The langchain `AIMessage` object may include `usage_metadata` and
    `response_metadata` which vary across providers; this helper converts
    them into a consistent dictionary with token counts, model id, cost,
    and a list of parsed tool-calls annotated with their configured costs.

    Parameters
    - message: `AIMessage` instance returned by LangChain/LiteLLM.
    - tool_costs: mapping from tool name to cost used to annotate tool call

    Returns
    A dict with keys: `prompt_tokens`, `completion_tokens`, `total_tokens`,
    `model_name`, `finish_reason`, `cost_usd`, `tool_calls`, and
    `invalid_tool_calls`.
    """

    # --- token usage (LangChain-normalised; LiteLLM populates this for all providers) ---
    um = (
        message.usage_metadata or {}
    )  # https://reference.langchain.com/python/langchain-core/messages/ai/UsageMetadata

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
            "tool_cost": c if (c := tool_costs.get(tc["name"])) is not None else -1,
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
    top_k: int | None = None,
    api_base: str | None = None,
    min_p: float | None = None,
    presence_penalty: float | None = None,
    repetition_penalty: float | None = None,
    enable_thinking: bool | None = None,
    reasoning_effort: str | None = None,
    request_timeout: float | None = None,
    num_retries: int | None = None,
    http_client: object | None = None,
) -> ChatLiteLLM:
    """Create a configured `ChatLiteLLM` instance for agent use.

    This wraps the common argument translation between the project's
    higher-level model config fields and the `ChatLiteLLM` constructor.

    Parameters mirror the pipeline config and allow lightweight tuning:
    - `model_name` / `model_provider`: combined into the provider/model string
    - `temperature`, `top_p`, `top_k`: sampling controls
    - `max_tokens`: maximum completion length, sent as `max_completion_tokens`.
      Omitted on the local vLLM path (`api_base` set) where the server's
      `--max-model-len` already bounds generation; only sent to hosted
      providers, which have no server-side cap.
    - `enable_thinking`, `reasoning_effort`: provider-specific extras

    Returns
    A ready-to-use `ChatLiteLLM` instance. Use this when creating the
    agent's model clients so behaviour and token limits are consistent.
    """

    # LiteLLM uses "{provider}/{model}" format
    # https://docs.litellm.ai/docs/providers
    # When an api_base is given we target a local vLLM OpenAI-compatible server,
    # which LiteLLM routes via the "hosted_vllm/" prefix instead of the provider.
    litellm_model = (
        f"hosted_vllm/{model_name}"
        if api_base is not None
        else f"{model_provider}/{model_name}"
    )
    # reasoning + result
    # For local vLLM (hosted_vllm/, i.e. api_base set) we deliberately omit
    # max_completion_tokens: the server's --max-model-len already bounds
    # generation to (max_model_len - prompt_tokens), so letting vLLM fill the
    # remaining context avoids both a fixed cap and the prompt+completion >
    # max_model_len rejection. Hosted providers (OpenRouter, etc.) have no such
    # server-side cap, so we still send max_tokens to bound their generation.
    model_kwargs: dict = {}
    if api_base is None:
        model_kwargs["max_completion_tokens"] = max_tokens
    if reasoning_effort is not None:
        model_kwargs["reasoning_effort"] = reasoning_effort

    if min_p is not None:
        model_kwargs["min_p"] = min_p
    if presence_penalty is not None:
        model_kwargs["presence_penalty"] = presence_penalty
    if repetition_penalty is not None:
        model_kwargs["repetition_penalty"] = repetition_penalty
    if enable_thinking is not None:
        model_kwargs["chat_template_kwargs"] = {"enable_thinking": enable_thinking}
    # When hitting a local vLLM (hosted_vllm/), LiteLLM otherwise reuses a single
    # cached httpx client whose default pool caps at max_connections=100 — which
    # silently throttles in-flight requests below the pipeline concurrency. Passing
    # an explicit client (sized to the concurrency) lets LiteLLM use a wider pool.
    # `client` is only honoured on the hosted_vllm/openai path, so gate on api_base.
    if http_client is not None and api_base is not None:
        model_kwargs["client"] = http_client

    kwargs = dict(
        model=litellm_model,
        temperature=temperature,
        model_kwargs=model_kwargs,
    )
    if top_p is not None:
        kwargs["top_p"] = top_p
    if top_k is not None:
        kwargs["top_k"] = top_k
    if api_base is not None:
        kwargs["api_base"] = api_base
    # A per-request timeout makes a wedged connection fail fast instead of
    # blocking its worker thread until asyncio's 300s executor-join watchdog
    # trips (which surfaces as "[Errno 9] Bad file descriptor").
    if request_timeout is not None:
        kwargs["request_timeout"] = request_timeout
    if num_retries is not None:
        kwargs["max_retries"] = num_retries

    return ChatLiteLLM(**kwargs)


def utils_render_jinja(template_str: str, params: dict) -> str:
    """Render a Jinja template string with the provided parameters.

    A tiny convenience wrapper used by `utils_build_messages` to keep
    message construction concise and testable.

    Parameters
    - template_str: Jinja template as a string
    - params: mapping of variables to render into the template

    Returns
    The rendered string.
    """

    return Template(template_str).render(**params)


def utils_build_messages(
    system_str: str | None,
    user_str: str,
    params: dict,
) -> list[dict]:
    """Build a small list of message dicts for a chat model call.

    The function renders the optional `system_str` and required `user_str`
    using `utils_render_jinja` and returns a list of dictionaries with
    `role`/`content` keys that downstream code (or tests) can consume.

    Example
    - `utils_build_messages("You are an assistant.", "Answer: {{q}}", {"q": "hi"})`

    Returns
    A list like `[{'role': 'system', 'content': '...'}, {'role': 'user', 'content': '...'}]`.
    """

    msgs: list[dict] = []
    if system_str:
        msgs.append(dict(role="system", content=utils_render_jinja(system_str, params)))
    msgs.append(dict(role="user", content=utils_render_jinja(user_str, params)))
    return msgs
