"""maintenance_agent baseline: bash + write_query + run_tests + comment_on_issue +
silent submit, under the bird patience budget."""
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
    make_tool_wrapper_patience_and_submit_silent,
    sanitize_thinking_history,
    wrap_model_append_tool_message,
)
from conversation2sql.eval_framework.agents.bird_baseline.agent_code import (
    utils_process_agent_response,
)
from conversation2sql.eval_framework.agents.bird_baseline.tools import submit_sql_impl
from conversation2sql.eval_framework.agents.deep_agent.tools.bash_tool import (
    build_pg_env,
    return_tool_bash,
)
from conversation2sql.eval_framework.agents.maintenance_agent.agent_code_state import (
    MaintenanceAgentCustomState,
)
from conversation2sql.eval_framework.agents.maintenance_agent.catalog_seed import (
    maintenance_tool_costs,
    materialize_maintenance_workspace,
)
from conversation2sql.eval_framework.agents.maintenance_agent.prompts import (
    build_maintenance_agent_messages,
)
from conversation2sql.eval_framework.agents.maintenance_agent.tools.maintenance_tools import (
    return_tool_comment_on_issue,
    return_tool_run_tests,
    return_tool_write_query,
    submit,
)
from conversation2sql.eval_framework.state import TaskData


def _build_maintenance_tools(
    single_task: TaskData,
    model_user_parsing: BaseChatModel | None,
    model_user_generator: BaseChatModel | None,
    catalog_dir: Path,
    pg_env: dict[str, str],
    *,
    enable_ask_user: bool,
) -> list:
    tools: list = [
        return_tool_bash(catalog_dir, pg_env),
        return_tool_write_query(catalog_dir),
        return_tool_run_tests(catalog_dir, single_task.db_dsn),
        submit,
    ]
    # Only surface comment_on_issue when it's enabled — binding it without
    # also describing it in the prompt hides the tool from the model.
    if enable_ask_user:
        assert model_user_parsing is not None and model_user_generator is not None, (
            "enable_ask_user=True requires the user-sim models"
        )
        tools.append(
            return_tool_comment_on_issue(
                catalog_dir, model_user_parsing, model_user_generator
            )
        )
    return tools


def _build_maintenance_middleware() -> list:
    mws: list = [
        ModelRetryMiddleware(max_delay=60.0, on_failure="error"),
        ToolRetryMiddleware(max_delay=60.0, on_failure="error"),
    ]
    # Patience budget — appended last, same relative order as bird_baseline /
    # deep_agent. Costed from maintenance_tool_costs(), and using the *silent*
    # submit factory: this baseline's `submit` carries no pass/fail signal, so
    # there is no retry-on-failed-submit branch (see Task 1).
    mws += [
        check_budget_limit,
        sanitize_thinking_history,
        wrap_model_append_tool_message,
        make_tool_wrapper_patience_and_submit_silent(
            maintenance_tool_costs(), submit_tool_name="submit"
        ),
    ]
    return mws


def run_agent_maintenance(
    single_task: TaskData,
    model_agent: BaseChatModel,
    model_user_parsing: BaseChatModel | None,
    model_user_generator: BaseChatModel | None,
    *,
    enable_ask_user: bool = True,
) -> dict:
    messages = build_maintenance_agent_messages(
        params={
            "total_budget": single_task.task_budget,
            "amb_user_query": single_task.task_question,
            "enable_ask_user": enable_ask_user,
        }
    )
    catalog_dir = materialize_maintenance_workspace(single_task)
    try:
        pg_env = build_pg_env(single_task.db_dsn)
        agent = create_agent(
            model_agent,
            _build_maintenance_tools(
                single_task,
                model_user_parsing,
                model_user_generator,
                catalog_dir,
                pg_env,
                enable_ask_user=enable_ask_user,
            ),
            state_schema=MaintenanceAgentCustomState,
            context_schema=TaskData,
            middleware=_build_maintenance_middleware(),  # pyrefly: ignore
        )
        agent_state = {
            "messages": messages,
            "initial_user_patience": single_task.task_budget,
            "updated_user_patience": single_task.task_budget,
            "tool_called_patience": [],
        }
        response = agent.invoke(agent_state, context=single_task)  # pyrefly: ignore
        # The workspace *is* the state: no tool-call parsing needed. This is
        # unconditional — whether the episode ended via an explicit `submit`
        # call or via forced budget exhaustion, the file's final on-disk
        # content is the predicted SQL.
        predicted_sql = (catalog_dir / "queries" / "answer.sql").read_text(
            encoding="utf-8"
        )
        # Hidden grading: never exposed to the agent (silent submit). Computed
        # purely so the pipeline can report execution_accuracy.
        grading = submit_sql_impl(
            sql=predicted_sql,
            sol_sqls=single_task.sol_sql,
            db_dsn=single_task.db_dsn,
            conditions=single_task.sql_query_conditions,
        )
        output = utils_process_agent_response(
            response,  # pyrefly: ignore
            tool_costs=maintenance_tool_costs(),
        )
        output["predicted_sql"] = predicted_sql
        # Never trust utils_process_agent_response's own `execution_accuracy`
        # derivation here: run_tests also returns a dict tool message with a
        # `passed` key (a structural check), which would otherwise be picked
        # up as the correctness signal. The real grade always comes from the
        # hidden submit_sql_impl call above, run on the final on-disk file.
        output["execution_accuracy"] = grading["passed"]
        return output
    finally:
        shutil.rmtree(catalog_dir, ignore_errors=True)
