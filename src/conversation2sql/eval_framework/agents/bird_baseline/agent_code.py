from langchain.agents.middleware import ClearToolUsesEdit
from langchain.agents.middleware import ContextEditingMiddleware
from typing import Any
from conversation2sql.eval_framework.agents.utils import (
    utils_extract_sql_from_ai_message,
    utils_process_single_msg,
)
from langchain.agents import create_agent
from langchain.agents.middleware import (
    ToolCallLimitMiddleware,
    ModelCallLimitMiddleware,
    ModelRetryMiddleware,
    ToolRetryMiddleware,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage

from conversation2sql.eval_framework.agents.bird_baseline.agent_callback import (
    tool_wrapper_patience_and_submit,
    wrap_model_append_tool_message,
    check_budget_limit,
    sanitize_thinking_history,
)
from conversation2sql.eval_framework.agents.bird_baseline.agent_code_state import (
    CustomAgentState,
)
from conversation2sql.eval_framework.agents.bird_baseline.middleware_tracking import (
    extract_middleware_events,
)
from conversation2sql.eval_framework.agents.bird_baseline.prompts import (
    build_bird_interact_agent_messages,
)
from conversation2sql.eval_framework.agents.bird_baseline.tools import (
    execute_sql,
    psql_console,
    create_python_udf,
    get_all_column_meanings,
    get_schema,
    get_table_names,
    get_table_schema,
    get_column_meaning,
    get_all_external_knowledge_names,
    get_knowledge_definition,
    get_all_knowledge_definitions,
    return_tool_ask_user,
    submit_sql,
    cleanup_python_udfs_impl,
    _safe_instance_prefix,
)
from conversation2sql.eval_framework.agents.bird_baseline.tools import TOOL_COSTS


from conversation2sql.eval_framework.state import TaskData
from conversation2sql.logger import get_logger

logger = get_logger(__name__)


def _select_db_tools(single_task: TaskData) -> list:
    """Pick the database-facing tools for this task.

    Default: execute_sql + get_schema (+ get_table_names/get_table_schema when
    enable_table_schema_tools). When enable_psql_console is set, a single
    read-only psql_console tool replaces all of them. The two ablation flags are
    mutually exclusive (also enforced in ConfigReader).
    """
    if single_task.enable_psql_console and single_task.enable_table_schema_tools:
        raise ValueError(
            "enable_psql_console and enable_table_schema_tools are mutually "
            "exclusive; enable at most one DB-tool ablation."
        )
    if single_task.enable_psql_console:
        return [psql_console]
    db_tools = [execute_sql, get_schema]
    if single_task.enable_table_schema_tools:
        db_tools.extend([get_table_names, get_table_schema])
    return db_tools


def run_agent_bird_baseline(
    single_task: TaskData,
    model_agent: BaseChatModel,
    model_user_parsing: BaseChatModel,
    model_user_generator: BaseChatModel,
    *,
    enable_ask_user: bool,
) -> CustomAgentState:
    if enable_ask_user:
        assert model_user_parsing is not None and model_user_generator is not None, (
            "model_user_parsing and model_user_generator must not be None when enable_ask_user=True"
        )

    messages = build_bird_interact_agent_messages(
        params={
            "total_budget": single_task.task_budget,
            "amb_user_query": single_task.task_question,
            # "amb_user_query": 'This is a debug message, call only ask_user as tool with an invented question and return without submitting'
            "enable_ask_user": enable_ask_user,
            "enable_table_schema_tools": single_task.enable_table_schema_tools,
            "enable_psql_console": single_task.enable_psql_console,
            "enable_python_udf": single_task.enable_python_udf,
        }
    )

    tools = [
        *_select_db_tools(single_task),
        get_all_column_meanings,
        get_column_meaning,
        get_all_external_knowledge_names,
        get_knowledge_definition,
        get_all_knowledge_definitions,
        submit_sql,
    ]
    if enable_ask_user:
        tools.append(return_tool_ask_user(model_user_parsing, model_user_generator))
    if single_task.enable_python_udf:
        tools.append(create_python_udf)

    agent = create_agent(
        model_agent,
        tools,
        state_schema=CustomAgentState,  # mutable from the tool
        context_schema=TaskData,  # immutable cannot be changed in the tool
        middleware=[  # pyrefly: ignore
            ModelRetryMiddleware(max_delay=60.0, on_failure="error"),
            ToolRetryMiddleware(max_delay=60.0, on_failure="error"),
            # ModelCallLimitMiddleware(run_limit=single_task.task_budget + 5),
            # ToolCallLimitMiddleware(
                # Maximum tool calls per single invocation (one user message → response cycle).
                # Resets with each new user message.
                # run_limit=single_task.task_budget + 5,
                # Maximum tool calls across all runs in a thread (conversation).
                # Persists across multiple invocations with the same thread ID.
                # Requires a checkpointer to maintain state. None means no thread limit.
                # thread_limit=single_task.task_budget * 2,
            # ),
            ContextEditingMiddleware(
                edits=[
                    ClearToolUsesEdit(
                        trigger=100_000,
                        clear_at_least=25_000,
                        keep=3,
                        clear_tool_inputs=False,
                        exclude_tools=[],
                        placeholder="[cleared]",
                    ),
                ],
            ),
            check_budget_limit,
            sanitize_thinking_history,
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
    if single_task.enable_python_udf:
        cleanup_python_udfs_impl(
            db_dsn=single_task.db_dsn,
            prefix=_safe_instance_prefix(single_task.instance_id),
        )
    # Recover the predicted SQL from the raw messages *before* utils_process_agent_response
    # pops "messages" off the state.
    predicted_sql = _extract_predicted_sql(response["messages"])  # pyrefly: ignore
    output = utils_process_agent_response(response, tool_costs=TOOL_COSTS)
    output["predicted_sql"] = predicted_sql
    return output


def _extract_predicted_sql(messages: list[BaseMessage]) -> str | None:
    """Recover the SQL the agent settled on.

    Prefers the `sql` argument of the last `submit_sql` tool call (the SQL the
    agent actually submitted for evaluation). Falls back to parsing the last
    AIMessage's text when the agent never submitted (e.g. budget exhausted).
    """
    for message in reversed(messages):
        for tool_call in getattr(message, "tool_calls", None) or []:
            if tool_call.get("name") == "submit_sql":
                sql = (tool_call.get("args") or {}).get("sql")
                if sql:
                    return sql
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            sql = utils_extract_sql_from_ai_message(message)
            if sql is not None:
                return sql
    return None


def utils_process_agent_response(
    response: CustomAgentState, tool_costs: dict | None = None
) -> Any:
    """Normalise an agent `CustomAgentState` into a report dict.

    This takes the mutable `response` (which contains a `messages` list of
    LangChain `BaseMessage` subclasses) and returns a compact dictionary
    summarising token usage, cost, tool calls and the parsed messages.

    Parameters
    - response: The agent state (expected to implement mapping access and
      contain a `messages` entry with LangChain message objects).
    - tool_costs: Optional mapping from tool name to its coin/cost value.

    Returns
    A dictionary with keys:
    - `total_cost`, `total_tokens`, `total_prompt_tokens`,
      `total_completion_tokens` (cumulative across every LLM call — the full
      prompt is re-sent each turn, so this is the real billed token count),
      `num_model_calls` (number of LLM calls), `mean_prompt_tokens`,
      `mean_completion_tokens` (per-call averages: sum / `num_model_calls`)
    - `tool_calls_in_order`: flattened sequence of tool call entries
    - `execution_accuracy`: boolean derived from the last tool (submit)
    - `messages`: list of parsed message dicts (see `utils_process_single_msg`)

    Why: downstream logging and metrics expect a stable, JSON-friendly
    shape rather than rich LangChain objects.
    """

    tool_costs = tool_costs or {}
    raw_messages = response.pop("messages")  # pyrefly: ignore
    middleware_events = extract_middleware_events(raw_messages)
    messages = [
        utils_process_single_msg(m, tool_costs=tool_costs) for m in raw_messages
    ]
    total_cost = 0
    total_tokens = 0
    # Only AIMessages carry token counts (one entry per LLM call). Tool/human/
    # system messages contribute nothing, so we collect prompt/completion tokens
    # per AI call to keep the mean a true per-call average. The full prompt is
    # re-sent every turn, so summing these gives the real cumulative token cost.
    prompt_tokens_per_call = []
    completion_tokens_per_call = []
    tool_calls_in_order = []
    passed = False
    for msg in messages:
        if msg["role"] == "tool":
            passed = msg["content"].get("passed", False)

        total_cost += msg.get("cost_usd", 0)

        total_tokens += msg.get("total_tokens", 0)
        if msg["role"] == "ai":
            prompt_tokens_per_call.append(msg.get("prompt_tokens", 0))
            completion_tokens_per_call.append(msg.get("completion_tokens", 0))
        tool_calls_in_order.extend(msg.get("tool_calls", []))

    # finish_reason == "length" means the model hit the max-model-len cap and
    # was silently truncated (no error is raised, since we no longer send
    # max_tokens on the local vLLM path). Surface a count + flag so a truncated
    # run is distinguishable from a clean one downstream / in the explorer.
    num_truncated_calls = sum(
        1
        for msg in messages
        if msg["role"] == "ai" and msg.get("finish_reason") == "length"
    )
    if num_truncated_calls:
        logger.warning(
            "%d model call(s) truncated by max-model-len (finish_reason='length'); "
            "generation was cut off — check the run's token budget.",
            num_truncated_calls,
        )

    n_calls = len(prompt_tokens_per_call)
    return {
        **response,
        "total_cost": total_cost,
        "total_tokens": total_tokens,
        "total_prompt_tokens": sum(prompt_tokens_per_call),
        "total_completion_tokens": sum(completion_tokens_per_call),
        "num_model_calls": n_calls,
        "num_truncated_calls": num_truncated_calls,
        "was_truncated": num_truncated_calls > 0,
        "mean_prompt_tokens": sum(prompt_tokens_per_call) / n_calls if n_calls else 0.0,
        "mean_completion_tokens": (
            sum(completion_tokens_per_call) / n_calls if n_calls else 0.0
        ),
        "tool_calls_in_order": tool_calls_in_order,
        "execution_accuracy": passed,
        "middleware_events": middleware_events,
        "messages": messages,
    }
