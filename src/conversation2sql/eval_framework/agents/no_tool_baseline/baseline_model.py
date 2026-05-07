from langchain_core.language_models import BaseChatModel

from conversation2sql.eval_framework.agents.no_tool_baseline.prompts import build_omnisql_prompt
from conversation2sql.eval_framework.agents.utils import utils_process_agent_response
from conversation2sql.eval_framework.state import TaskData
from conversation2sql.logger import get_logger

logger = get_logger(__name__)


def run_baseline_model(
        single_task: TaskData,
        model_agent: BaseChatModel,
):
    messages = build_omnisql_prompt(
        params={
            "schema": single_task.ddl_database_schema,
            "question": single_task.task_question,
        }
    )

    response = model_agent.invoke(messages)

    return utils_process_agent_response(response, tool_costs={})