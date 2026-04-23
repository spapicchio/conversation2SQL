from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallLimitMiddleware, ModelCallLimitMiddleware
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage, ToolCall, ToolMessage

from conversation2sql.config_input import ConfigPredictor
from conversation2sql.eval_framework.agent.agent_callback import (
    tool_wrapper_append_budget,
    tool_wrapper_submit_sql,
    model_budget_exhausted
)
from conversation2sql.eval_framework.agent.agent_code_state import CustomAgentState
from conversation2sql.eval_framework.agent.prompts import build_bird_interact_agent_messages
from conversation2sql.eval_framework.agent.tools import (
    execute_sql,
    get_all_column_meanings,
    get_schema, get_column_meaning,
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
        model_agent: BaseChatModel,
        model_user_parsing: BaseChatModel,
        model_user_generator: BaseChatModel,
        single_task: TaskData,
        config_predictor: ConfigPredictor,
):
    messages = build_bird_interact_agent_messages(
        params={
            "total_budget": single_task.task_budget,
            # "amb_user_query": single_task.amb_user_query
            "amb_user_query": 'This is a debug message, call only get_all_column_meanings as tool and return without submitting'
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
            ModelCallLimitMiddleware(
                run_limit=config_predictor.user_patience_budget + 5
            ),
            ToolCallLimitMiddleware(
                # Maximum tool calls per single invocation (one user message → response cycle).
                # Resets with each new user message.
                run_limit=config_predictor.user_patience_budget + 5,
                # Maximum tool calls across all runs in a thread (conversation).
                # Persists across multiple invocations with the same thread ID.
                # Requires a checkpointer to maintain state. None means no thread limit.
                thread_limit=config_predictor.user_patience_budget * 3,
            ),
            tool_wrapper_append_budget,
            tool_wrapper_submit_sql,
            model_budget_exhausted,
        ],
    )
    agent_state: CustomAgentState = {
        "messages": messages,  # pyrefly: ignore,
        "initial_user_patience": single_task.task_budget,
        "updated_user_patience": single_task.task_budget,
    }

    response: CustomAgentState = agent.invoke(agent_state, context=single_task)
    return _process_agent_response(response)


def _process_agent_response(response: CustomAgentState) -> Any:
    messages = [_process_single_msg(m) for m in response.pop('messages')]
    return {**response, "messages": messages}


def _process_single_msg(message: BaseMessage) -> dict:
    def _process_tool_call(tool_call: ToolCall):
        return {'tool_name': tool_call['name'], 'arguments': tool_call['args']}
    #  https://docs.langchain.com/oss/javascript/langchain/messages?search=ResponseMetadata
    response = {}
    if message.response_metadata:
        response = {
            'completion_tokens': message.response_metadata['token_usage']['completion_tokens'],
            'prompt_tokens': message.response_metadata['token_usage']['prompt_tokens'],
            'total_tokens': message.response_metadata['token_usage']['total_tokens'],
            'reasoning_tokens': message.response_metadata['token_usage']['completion_tokens_details'][
                'reasoning_tokens'],
            'model_name': f"{message.response_metadata['model_provider']}:{message.response_metadata['model_name']}",
            'finish_reason': message.response_metadata['finish_reason'],
            'tool_calls': [tool['name'] for tool in message.response_metadata['tool_calls']]

        }
    message_dict = {
        "role": message.type,
        "content": message.content,
        "response_metadata": response,
    }
    if isinstance(message, AIMessage):
        message_dict['tool_calls'] = [_process_tool_call(tool) for tool in message.tool_calls]
    elif isinstance(message, ToolMessage):
        message_dict['tool_name'] = message.name

    return message_dict
