"""
Bird-Interact user interaction tools.

These tools correspond to the ``User`` interaction object in the Bird-Interact
agent prompt.  They allow the agent to ask the user for clarification and to
submit a final SQL query for evaluation.

Cost summary (mirrors the original prompt):
    ask_user    → 2 patience  (clarification question)
    submit_sql  → 3 patience  (final SQL submission; typically ends the episode)

The ``ask_user`` tool calls an LLM user-simulator whose prompt templates and
parameters are stored in ``UserContext``.  ``submit_sql`` records the
submission in ``UserContext.template_params["submitted_sql"]`` and returns a
stub response; the evaluation harness (scorer) reads that key to grade the answer.
"""

from frozendict import frozendict
from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime

from conversation2sql.eval.interfaces import UserContext
from conversation2sql.eval.predictors.available_tools._patience_utils import deduct_and_note
from conversation2sql.eval.predictors.langchain_agent_factory import AgentState, get_cached_model
from conversation2sql.eval.registry import tool_registry
from conversation2sql.prompt_factory import get_cached_prompt_factory


# ---------------------------------------------------------------------------
# Ask user (clarification) — cost: 2 patience
# ---------------------------------------------------------------------------

@tool_registry.register(name="ask_user")
@tool(
    description=(
        "Ask the user a single clarification question to resolve ambiguity in their request. "
        "Only ask ONE question at a time to minimise patience cost. "
        "Cost: 2 patience."
    )
)
def ask_user(question: str, runtime: ToolRuntime[UserContext, AgentState]) -> str:
    """Invoke the user-simulator LLM with a clarification question (cost: 2 patience)."""
    note = deduct_and_note(runtime, cost=2)

    model = get_cached_model(
        model_name="o4-mini",
        model_provider="openai",
        temperature=1.0,
        max_tokens=1024,
        top_p=1.0,
        top_k=None,
    )

    template_params = runtime.context.template_params
    template_params["clarification_question"] = question

    prompt_factory = get_cached_prompt_factory(runtime.context.user_simulator_prompt_folder)
    messages = []

    if runtime.context.user_simulator_system_prompt is not None:
        messages.append(
            prompt_factory.render_template(
                template_name=runtime.context.user_simulator_system_prompt,
                template_params=frozendict(**template_params),
            )
        )

    messages.append(
        prompt_factory.render_template(
            template_name=runtime.context.user_simulator_user_prompt,
            template_params=frozendict(**template_params),
        )
    )

    output = model.invoke(input=messages)
    return output.content + note


# ---------------------------------------------------------------------------
# Submit SQL — cost: 3 patience
# ---------------------------------------------------------------------------

@tool_registry.register(name="submit_sql")
@tool(
    description=(
        "Submit your final SQL query to the user for evaluation. "
        "The user will test the SQL against the database and return feedback on its correctness. "
        "Use this only when you are confident the SQL is correct. "
        "Cost: 3 patience."
    )
)
def submit_sql(sql: str, runtime: ToolRuntime[UserContext, AgentState]) -> str:
    """Submit the final SQL to the user-simulator for evaluation (cost: 3 patience).

    The submitted SQL is stored in ``template_params['submitted_sql']`` so the
    scorer and user-simulator can access it.  Override this tool or extend
    ``UserContext`` to wire up real execution-based feedback.
    """
    note = deduct_and_note(runtime, cost=3)
    runtime.context.template_params["submitted_sql"] = sql
    return (
        "[SUBMITTED] Your SQL has been submitted for evaluation. "
        "Await feedback from the user."
        + note
    )
