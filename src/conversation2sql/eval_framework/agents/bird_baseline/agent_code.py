from langchain.agents import create_agent
from langchain.agents.middleware import (
    ToolCallLimitMiddleware,
    ModelCallLimitMiddleware,
    ModelRetryMiddleware,
    ToolRetryMiddleware,
)
from langchain_core.language_models import BaseChatModel

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
from conversation2sql.eval_framework.agents.utils import utils_process_agent_response
from conversation2sql.eval_framework.state import TaskData
from conversation2sql.logger import get_logger

logger = get_logger(__name__)


def run_agent_bird_baseline(
        single_task: TaskData,
        model_agent: BaseChatModel,
        model_user_parsing: BaseChatModel,
        model_user_generator: BaseChatModel,
) -> CustomAgentState:
    messages = build_bird_interact_agent_messages(
        params={
            "total_budget": single_task.task_budget,
            "amb_user_query": single_task.task_question,
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
    return utils_process_agent_response(response)
