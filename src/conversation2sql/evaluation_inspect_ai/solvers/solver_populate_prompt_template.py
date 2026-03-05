"""Inspect AI solvers that use PromptFactory for Jinja2-based prompt rendering."""

from typing import Any

from inspect_ai.model import ChatMessage, ChatMessageSystem
from inspect_ai.solver import Generate, TaskState, solver, Solver

from conversation2sql.prompt_factory import PromptFactory


@solver
def populate_prompt_template(
        task: str,
        prompt_dir: str,
        **params: dict[str, Any],
) -> Solver:
    """
    Create the Messages for the task by rendering the system and user prompts using PromptFactory.

    Args:
        task: the task name to determine which prompt template to use
        prompt_dir: the directory where the prompt templates are located
        **params: additional parameters to pass to the PromptFactory for rendering the templates

    Returns:
        A Solver function that populates the TaskState with the rendered messages.
    """
    factory = PromptFactory(prompt_dir=prompt_dir)

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        system_prompt = factory.get_system_prompt(task, **params)
        user_prompt = factory.get_user_prompt(task, **params)
        messages = [ChatMessageSystem(content=system_prompt), ChatMessage(content=user_prompt)]
        state.messages = messages
        return state

    return solve
