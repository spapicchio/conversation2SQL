"""Message-building helpers that map typed PromptParams to BaseMessage lists.

Separates template rendering from dataset-reader logic so that message
construction can be reused across readers and tested independently.
"""
from __future__ import annotations

from conversation2sql.eval.interfaces import BaseMessage
from conversation2sql.eval.prompt_params import BirdInteractAgentParams
from conversation2sql.prompt_factory import PromptFactory


def build_agent_messages(
    prompt_factory: PromptFactory,
    system_prompt: str | None,
    user_prompt: str,
    params: BirdInteractAgentParams,
) -> list[BaseMessage]:
    """Render system + user templates into a ``list[BaseMessage]``.

    Parameters
    ----------
    prompt_factory:
        Initialised ``PromptFactory`` pointing at the correct prompts directory.
    system_prompt:
        Template path relative to the prompt directory, e.g.
        ``"bird_interact_a_agent/system.jinja"``.  ``None`` omits the system
        message entirely.
    user_prompt:
        Template path for the initial user turn.
    params:
        Typed parameter model; all required template variables must be set.
    """
    messages: list[BaseMessage] = []

    if system_prompt:
        content = prompt_factory.render_template(system_prompt, params)
        messages.append(BaseMessage(role="system", content=content))

    content = prompt_factory.render_template(user_prompt, params)
    messages.append(BaseMessage(role="user", content=content))

    return messages
