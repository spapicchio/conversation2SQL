"""Typed parameter models for Jinja2 prompt templates.

Each Pydantic model corresponds to a template family under ``prompts/``.
Pass these to ``PromptFactory.render_template`` instead of raw dicts to get
static type checking and Pydantic validation at construction time.

Runtime-injectable fields (``clarification_question``, ``action``,
``submitted_sql``) are defined on the base class with empty-string defaults
so that tools can call ``model_copy(update={"clarification_question": q})``
on any ``PromptParams`` instance without knowing its concrete type.
"""
from __future__ import annotations

from pydantic import BaseModel


class PromptParams(BaseModel):
    """Base class for all prompt parameter models.

    The three fields below are injected by tools at runtime and therefore
    carry empty-string defaults; they are not required at construction time.
    Having them on the base class lets ``ask_user`` / ``submit_sql`` call
    ``model_copy(update={...})`` without casting to a concrete subtype.
    """

    # Injected by ask_user tool
    clarification_question: str = ""
    # Injected by step_2 LLM generator (action chosen in step_1)
    action: str = ""
    # Recorded by submit_sql tool so the scorer can read the final answer
    submitted_sql: str = ""


# ---------------------------------------------------------------------------
# bird_interact_a_agent/ — agent-facing prompts
# system.jinja  vars: database_engine
# user.jinja    vars: user_query, total_budget
# ---------------------------------------------------------------------------

class BirdInteractAgentParams(PromptParams):
    """Parameters for ``bird_interact_a_agent/system.jinja`` and ``user.jinja``."""

    database_engine: str  # e.g. "postgresql"
    user_query: str       # the (possibly ambiguous) user query shown to the agent
    total_budget: int     # total patience budget shown in the user prompt


# ---------------------------------------------------------------------------
# bird_interact_user_simulator/ — user simulator prompts
# simulator_base.jinja    vars: db_name, db_schema, user_query,
#                               ambiguities_json, correct_sql,
#                               clarification_question
# step_1_llm_parser.jinja vars: ambiguities_json, sql_segments,
#                               clarification_question
# step_2_llm_generator.jinja vars: db_schema, ambiguities_json, user_query,
#                                  database_engine, correct_sql, sql_segments,
#                                  clarification_question, action
# ---------------------------------------------------------------------------

class BirdInteractUserSimulatorParams(PromptParams):
    """Parameters for the ``bird_interact_user_simulator/`` template family.

    ``sql_segments`` is optional because it is only needed by the step_1 /
    step_2 templates; it defaults to an empty string for the base template.
    ``clarification_question``, ``action``, and ``submitted_sql`` are
    inherited from ``PromptParams`` and injected by tools at runtime.
    """

    db_name: str           # database name, e.g. "student_1"
    db_schema: str         # full DDL schema text
    user_query: str        # the ambiguous user query
    ambiguities_json: str  # pre-serialised JSON string of ambiguity points
    correct_sql: str       # ground-truth SQL (visible to the simulator only)
    database_engine: str   # e.g. "postgresql"
    sql_segments: str = "" # SQL segments used by step_1 / step_2 templates


# ---------------------------------------------------------------------------
# user_simulator/ — generic user simulator prompts
# system.jinja vars: original_query, user_query_ambiguity (opt),
#                    external_knowledge (opt)
# user.jinja   vars: ambiguous_query, reference_sql, clarifications,
#                    dialogue_history
# ---------------------------------------------------------------------------

class UserSimulatorParams(PromptParams):
    """Parameters for the generic ``user_simulator/`` template family."""

    original_query: str
    ambiguous_query: str
    reference_sql: str
    clarifications: str
    dialogue_history: str
    # Optional — templates guard these with {% if ... %}
    user_query_ambiguity: list | None = None
    external_knowledge: list | None = None


# ---------------------------------------------------------------------------
# eval/ — generic single-turn eval prompts
# system.jinja — no template vars (extends base_system.jinja)
# user.jinja   vars: prompt, schema (opt)
# ---------------------------------------------------------------------------

class EvalParams(PromptParams):
    """Parameters for the generic ``eval/`` template family."""

    prompt: str
    schema: str = ""
