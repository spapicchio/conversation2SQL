"""deep_agent baseline: deepagents FilesystemMiddleware + bird patience budget."""
from __future__ import annotations

from langchain.agents import create_agent
from langchain.agents.middleware import (
    ModelRetryMiddleware,
    ToolRetryMiddleware,
    SummarizationMiddleware,
    TodoListMiddleware,
)
from langchain_core.language_models import BaseChatModel
from deepagents import FilesystemMiddleware, SubAgentMiddleware
from deepagents.backends.state import StateBackend
from deepagents.middleware.subagents import SubAgent

from conversation2sql.eval_framework.agents.bird_baseline.agent_callback import (
    check_budget_limit,
    sanitize_thinking_history,
    tool_wrapper_patience_and_submit,
    wrap_model_append_tool_message,
)
from conversation2sql.eval_framework.agents.bird_baseline.agent_code import (
    _extract_predicted_sql,
    utils_process_agent_response,
)
from conversation2sql.eval_framework.agents.bird_baseline.tools import (
    execute_sql,
    submit_sql,
    return_tool_ask_user,
)
from conversation2sql.eval_framework.agents.deep_agent.agent_code_state import (
    DeepAgentCustomState,
)
from conversation2sql.eval_framework.agents.deep_agent.filesystem_seed import (
    build_db_filesystem,
    deep_tool_costs,
)
from conversation2sql.eval_framework.agents.deep_agent.prompts import (
    build_deep_agent_messages,
)
from conversation2sql.eval_framework.state import TaskData
from conversation2sql.logger import get_logger

logger = get_logger(__name__)

_FS_READ_TOOLS = ("ls", "read_file", "glob", "grep")
_FS_WRITE_TOOLS = ("write_file", "edit_file")

# Minimal general-purpose subagent. deepagents' SubAgentMiddleware requires at
# least one subagent (each needing a model); tuning subagent prompts/tools is
# out of scope per the spec (subagents are off by default — this only exists so
# the deep_enable_subagents flag can be toggled).
def _general_subagent(model_agent: BaseChatModel) -> SubAgent:
    return {
        "name": "general",
        "description": "A general-purpose subagent for an isolated, self-contained sub-task.",
        "system_prompt": (
            "You are a focused sub-agent. Complete the assigned sub-task using the "
            "tools available to you and return a single concise result."
        ),
        "model": model_agent,
        "tools": [execute_sql, submit_sql],
    }


def _build_fs_middleware(*, enable_fs_write: bool) -> FilesystemMiddleware:
    """State-backed filesystem, restricted to a read-only subset by default.

    The middleware ships ls/read_file/write_file/edit_file/glob/grep/execute; we
    filter its `.tools` list down to the read subset (plus writes when enabled).
    `execute` is always dropped — it errors on the non-sandbox StateBackend.
    """
    mw = FilesystemMiddleware(backend=StateBackend())
    allowed = set(_FS_READ_TOOLS)
    if enable_fs_write:
        allowed |= set(_FS_WRITE_TOOLS)
    mw.tools = [t for t in mw.tools if t.name in allowed]
    return mw


def _build_deep_tools(
    single_task: TaskData,
    model_user_parsing: BaseChatModel,
    model_user_generator: BaseChatModel,
) -> list:
    return [
        execute_sql,
        submit_sql,
        return_tool_ask_user(model_user_parsing, model_user_generator),
    ]


def _build_deep_middleware(single_task: TaskData, model_agent: BaseChatModel) -> list:
    mws: list = [
        ModelRetryMiddleware(max_delay=60.0, on_failure="error"),
        ToolRetryMiddleware(max_delay=60.0, on_failure="error"),
        _build_fs_middleware(enable_fs_write=single_task.deep_enable_fs_write),
    ]
    if single_task.deep_enable_todos:
        mws.append(TodoListMiddleware())
    if single_task.deep_enable_summarization:
        # SummarizationMiddleware requires a model; reuse the agent's model.
        mws.append(SummarizationMiddleware(model=model_agent))
    if single_task.deep_enable_subagents:
        mws.append(
            SubAgentMiddleware(
                backend=StateBackend(),
                subagents=[_general_subagent(model_agent)],
            )
        )
    # Patience budget — appended last, same relative order as bird_baseline.
    mws += [
        check_budget_limit,
        sanitize_thinking_history,
        wrap_model_append_tool_message,
        tool_wrapper_patience_and_submit,
    ]
    return mws


def run_agent_deep_agent(
    single_task: TaskData,
    model_agent: BaseChatModel,
    model_user_parsing: BaseChatModel,
    model_user_generator: BaseChatModel,
    *,
    enable_ask_user: bool = True,
) -> dict:
    assert model_user_parsing is not None and model_user_generator is not None, (
        "deep_agent always uses ask_user; user-sim models must not be None"
    )
    messages = build_deep_agent_messages(
        params={
            "total_budget": single_task.task_budget,
            "amb_user_query": single_task.task_question,
            "enable_fs_write": single_task.deep_enable_fs_write,
            "enable_todos": single_task.deep_enable_todos,
            "enable_subagents": single_task.deep_enable_subagents,
        }
    )
    agent = create_agent(
        model_agent,
        _build_deep_tools(single_task, model_user_parsing, model_user_generator),
        state_schema=DeepAgentCustomState,
        context_schema=TaskData,
        middleware=_build_deep_middleware(single_task, model_agent),  # pyrefly: ignore
    )
    agent_state = {
        "messages": messages,
        "files": build_db_filesystem(single_task),
        "initial_user_patience": single_task.task_budget,
        "updated_user_patience": single_task.task_budget,
        "tool_called_patience": [],
    }
    response = agent.invoke(agent_state, context=single_task)  # pyrefly: ignore
    predicted_sql = _extract_predicted_sql(response["messages"])
    output = utils_process_agent_response(
        response,  # pyrefly: ignore
        tool_costs=deep_tool_costs(enable_fs_write=single_task.deep_enable_fs_write),
    )
    output["predicted_sql"] = predicted_sql
    return output
