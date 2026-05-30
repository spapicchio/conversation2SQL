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
    wrap_model_append_tool_message, check_budget_limit,
)
from conversation2sql.eval_framework.agents.bird_baseline.agent_code_state import CustomAgentState
from conversation2sql.eval_framework.agents.bird_baseline.prompts import (
    build_bird_interact_agent_messages,
)
from conversation2sql.eval_framework.agents.bird_baseline.tools import (
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
from conversation2sql.eval_framework.agents.bird_baseline.tools import TOOL_COSTS


from conversation2sql.eval_framework.state import TaskData
from conversation2sql.logger import get_logger

logger = get_logger(__name__)

def run_agent_bird_baseline(
        single_task: TaskData,
        model_agent: BaseChatModel,
        model_user_parsing: BaseChatModel,
        model_user_generator: BaseChatModel,
        *,
        enable_ask_user: bool,
) -> CustomAgentState:
    if enable_ask_user:
        assert (
            model_user_parsing is not None and model_user_generator is not None
        ), "model_user_parsing and model_user_generator must not be None when enable_ask_user=True"

    messages = build_bird_interact_agent_messages(
        params={
            "total_budget": single_task.task_budget,
            "amb_user_query": single_task.task_question,
            # "amb_user_query": 'This is a debug message, call only ask_user as tool with an invented question and return without submitting'
            "enable_ask_user": enable_ask_user,
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
        submit_sql,
    ]
    if enable_ask_user:
        tools.append(return_tool_ask_user(model_user_parsing, model_user_generator))

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
      `total_completion_tokens`, `mean_prompt_tokens`, `mean_completion_tokens`
    - `tool_calls_in_order`: flattened sequence of tool call entries
    - `execution_accuracy`: boolean derived from the last tool (submit)
    - `messages`: list of parsed message dicts (see `utils_process_single_msg`)

    Why: downstream logging and metrics expect a stable, JSON-friendly
    shape rather than rich LangChain objects.
    """

    tool_costs = tool_costs or {}
    messages = [
        utils_process_single_msg(m, tool_costs=tool_costs)
        for m in response.pop("messages") #pyrefly: ignore
    ]
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

