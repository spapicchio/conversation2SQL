"""Jinja2 prompt templates for the Level 1 LLM judge."""
from __future__ import annotations

from conversation2sql.eval_framework.agents.utils import utils_build_messages

_JUDGE_SYSTEM = """\
You are a conversation analysis expert classifying agent turns in a text-to-SQL dialogue system.
Classify the single agent turn below using the Level 1 taxonomy.
Output JSON only — no prose outside the JSON fields.\
"""

_JUDGE_USER = """\
## Turn to Classify

{% if thinking %}
### Agent Thinking (internal monologue):
{{ thinking[:2000] }}
{% endif %}

{% if text %}
### Agent Text (visible to user):
{{ text[:1000] }}
{% endif %}

### Tool Calls Summary:
{{ tool_calls_summary }}

### Level 2 Category (deterministic, from tool calls):
{{ level2_category }}

### Prior Failed Submit:
{{ prior_failed_submit }}

---

## Level 1 Taxonomy (10 labels)

| Label | Description |
|---|---|
| ANSWER_ATTEMPT | Agent commits to an SQL answer — first submission with no prior failed context |
| REVISION | Agent identifies a previous failed/wrong answer and self-corrects autonomously without asking the user |
| CLARIFICATION | Single focused question to the user about one ambiguity |
| INTERROGATION | Multiple questions to the user in one turn |
| ASSUMPTION | Agent explicitly resolves an ambiguity by making an interpretive choice without consulting the user |
| CONFIRMATION | Agent grounds the user's previous answer — echoes back its understanding before acting |
| DISCUSSION | Explores or reasons about the problem without submitting, asking, or refusing |
| HEDGING | Presents multiple conditional SQL candidates based on different interpretations |
| REFUSAL | Declines to proceed without a follow-up question or request |
| MISSING | Empty turn — no thinking, no text, no tool calls |

### Key Disambiguation Rules

- ASSUMPTION vs CLARIFICATION: assumption = agent decides unilaterally; clarification = agent asks the user
- REVISION vs ANSWER_ATTEMPT: REVISION requires prior_failed_submit=True and evidence the agent is correcting the failure
- CONFIRMATION vs DISCUSSION: CONFIRMATION explicitly references the user's prior answer and signals readiness to act

---

## Confidence Scale

| Value | Meaning |
|---|---|
| CERTAIN | Structurally determined — tool calls alone make the label unambiguous (e.g. submit_sql → ANSWER_ATTEMPT) |
| CONFIDENT | Text/thinking strongly supports one label; a second interpretation exists but is clearly weaker |
| PLAUSIBLE | Two labels are in competition; chosen one is best fit but the other is genuinely defensible |
| UNCERTAIN | Forced choice — turn is mixed or ambiguous enough that the label could easily be wrong |

---

## Instructions

1. Write your step-by-step reasoning in the `reasoning` field.
2. Choose the single best-fit label for `level1_category`.
3. If confidence is PLAUSIBLE or UNCERTAIN, populate `level1_alternatives` with ALL other labels that could reasonably apply. When confidence is CERTAIN or CONFIDENT, leave `level1_alternatives` empty.
4. Set `confidence` based on the scale above.\
"""


def build_judge_prompt(params: dict) -> list[dict]:
    """Return a list of role/content dicts suitable for model.invoke()."""
    return utils_build_messages(_JUDGE_SYSTEM, _JUDGE_USER, params)
