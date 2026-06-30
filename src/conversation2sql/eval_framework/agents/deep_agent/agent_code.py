"""deep_agent baseline: one read-only bash tool + bird patience budget."""
from __future__ import annotations

import shutil
from pathlib import Path

from langchain.agents import create_agent
from langchain.agents.middleware import (
    ModelRetryMiddleware,
    ToolRetryMiddleware,
)
from langchain_core.language_models import BaseChatModel

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
from conversation2sql.eval_framework.agents.deep_agent.bash_tool import (
    build_pg_env,
    return_tool_bash,
)
from conversation2sql.eval_framework.agents.deep_agent.catalog_seed import (
    deep_tool_costs,
    materialize_catalog_dir,
)
from conversation2sql.eval_framework.agents.deep_agent.prompts import (
    build_deep_agent_messages,
)
from conversation2sql.eval_framework.state import TaskData
from conversation2sql.logger import get_logger

logger = get_logger(__name__)


def _build_deep_tools(
    single_task: TaskData,
    model_user_parsing: BaseChatModel,
    model_user_generator: BaseChatModel,
    catalog_dir: Path,
    pg_env: dict[str, str],
) -> list:
    return [
        return_tool_bash(catalog_dir, pg_env),
        submit_sql,
        return_tool_ask_user(model_user_parsing, model_user_generator),
    ]


def _build_deep_middleware() -> list:
    mws: list = [
        ModelRetryMiddleware(max_delay=60.0, on_failure="error"),
        ToolRetryMiddleware(max_delay=60.0, on_failure="error"),
    ]
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
        }
    )
    catalog_dir = materialize_catalog_dir(single_task)
    try:
        pg_env = build_pg_env(single_task.db_dsn)
        agent = create_agent(
            model_agent,
            _build_deep_tools(
                single_task,
                model_user_parsing,
                model_user_generator,
                catalog_dir,
                pg_env,
            ),
            state_schema=DeepAgentCustomState,
            context_schema=TaskData,
            middleware=_build_deep_middleware(),  # pyrefly: ignore
        )
        # The system + user turns travel together in the initial state. The old
        # FilesystemMiddleware injected its own /db system message (forcing the
        # prompt through create_agent's system_prompt= to avoid two leading system
        # messages); with the on-disk catalog there is no such injection, so the
        # system message can live in the history directly — like bird_baseline.
        agent_state = {
            "messages": messages,
            "initial_user_patience": single_task.task_budget,
            "updated_user_patience": single_task.task_budget,
            "tool_called_patience": [],
        }
        response = agent.invoke(agent_state, context=single_task)  # pyrefly: ignore
        predicted_sql = _extract_predicted_sql(response["messages"])
        output = utils_process_agent_response(
            response,  # pyrefly: ignore
            tool_costs=deep_tool_costs(),
        )
        output["predicted_sql"] = predicted_sql
        return output
    finally:
        shutil.rmtree(catalog_dir, ignore_errors=True)
