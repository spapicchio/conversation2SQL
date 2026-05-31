from typing import Any
from conversation2sql.eval_framework.agents.utils_kb_linearize import linearize_kb


from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage

from conversation2sql.eval_framework.agents.bird_baseline.tools import submit_sql_impl
from conversation2sql.eval_framework.agents.no_tool_baseline.prompts import (
    build_omnisql_prompt,
)
from conversation2sql.eval_framework.agents.utils import (
    utils_extract_ai_metadata,
    utils_extract_sql_from_ai_message,
    utils_single_msg_to_str,
)
from conversation2sql.eval_framework.state import TaskData
from conversation2sql.logger import get_logger

logger = get_logger(__name__)


def run_baseline_no_tool(
    single_task: TaskData,
    model_agent: BaseChatModel,
) -> dict[str, Any]:
    kb_for_prompt = (
        linearize_kb(single_task.masked_agent_kb)
        if single_task.is_kb_linearized
        else single_task.masked_agent_kb
    )
    user_messages = build_omnisql_prompt(
        params={
            "schema": single_task.ddl_database_schema,
            "question": single_task.task_question,
            "kb": kb_for_prompt,
        }
    )

    ai_msg: AIMessage = model_agent.invoke(user_messages)  # pyrefly: ignore

    sql = utils_extract_sql_from_ai_message(ai_msg)
    raw_text = utils_single_msg_to_str(ai_msg)

    if sql is None:
        submit_outcome: dict = {
            "passed": False,
            "error": "no_sql_block_found",
            "raw_text": raw_text[-2000:],
        }
    else:
        submit_outcome = submit_sql_impl(
            sql=sql,
            sol_sqls=single_task.sol_sql,
            db_dsn=single_task.db_dsn,
            conditions=single_task.sql_query_conditions,
        )

    ai_meta = utils_extract_ai_metadata(ai_msg, tool_costs={})

    user_msg_dict = {"role": "user", "content": user_messages[-1]["content"]}
    ai_msg_dict = {"role": "ai", "content": raw_text, **ai_meta}
    synthetic_tool_msg = {
        "role": "tool",
        "tool_name": "submit_sql_offline",
        "status": "success" if submit_outcome.get("passed") else "error",
        "content": submit_outcome,
    }
    messages = [user_msg_dict, ai_msg_dict, synthetic_tool_msg]

    prompt_tokens = ai_meta.get("prompt_tokens", 0) or 0
    completion_tokens = ai_meta.get("completion_tokens", 0) or 0
    total_tokens = ai_meta.get("total_tokens", 0) or 0

    return {
        "messages": messages,
        "initial_user_patience": None,
        "updated_user_patience": None,
        "tool_called_patience": [],
        "total_cost": ai_meta.get("cost_usd", 0.0) or 0.0,
        "total_tokens": total_tokens,
        "total_prompt_tokens": prompt_tokens,
        "total_completion_tokens": completion_tokens,
        "mean_prompt_tokens": float(prompt_tokens),
        "mean_completion_tokens": float(completion_tokens),
        "tool_calls_in_order": [],
        "predicted_sql": sql,
        "target_sql": single_task.sol_sql[0],
        "execution_accuracy": bool(submit_outcome.get("passed", False)),
    }
