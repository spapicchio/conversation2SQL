# Turn Classifier Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a two-level turn classifier that reads `results_smaller.jsonl` traces, applies deterministic Level 2 tool-action rules and an LLM judge for Level 1 semantic labels, and writes enriched JSONL with `turn_classifications` appended per record.

**Architecture:** Level 2 is fully deterministic (YAML-configured tool→category map, no LLM). Level 1 uses `model.with_structured_output(Level1Classification)` — LangChain enforces the JSON schema, the model writes CoT reasoning before committing to a label. `classifier.py` assembles both levels into a `TurnClassification` per turn.

**Tech Stack:** Python 3.12, Pydantic v2, LangChain + LiteLLM (`ChatLiteLLM`), Jinja2, PyYAML, tqdm, pytest

---

## File Map

| Action | Path | Responsibility |
|--------|------|---------------|
| Create | `configs/turn_classifier/tool_categories.yaml` | Tool→Level2Category mapping (YAML, no code) |
| Create | `src/conversation2sql/eval_framework/turn_classifier/__init__.py` | Package init, public re-exports |
| Create | `src/conversation2sql/eval_framework/turn_classifier/schemas.py` | Pydantic models: `Level1Classification`, `TurnClassification` |
| Create | `src/conversation2sql/eval_framework/turn_classifier/rules.py` | Deterministic Level 2 from tool_calls + YAML |
| Create | `src/conversation2sql/eval_framework/turn_classifier/prompts.py` | Jinja2 judge prompt strings + builder function |
| Create | `src/conversation2sql/eval_framework/turn_classifier/classifier.py` | `TurnClassifier`: orchestrates Level 2 + Level 1 |
| Create | `scripts/classify_turns.py` | CLI entrypoint |
| Create | `tests/eval_framework/turn_classifier/__init__.py` | Test package |
| Create | `tests/eval_framework/turn_classifier/test_rules.py` | Unit tests for `rules.py` |
| Create | `tests/eval_framework/turn_classifier/test_prompts.py` | Unit tests for `prompts.py` |
| Create | `tests/eval_framework/turn_classifier/test_classifier.py` | Unit tests for `classifier.py` (mocked LLM) |

---

## Task 1: YAML Tool-Category Config

**Files:**
- Create: `configs/turn_classifier/tool_categories.yaml`

- [ ] **Step 1: Create the config directory and file**

```yaml
# Tool name → Level 2 category
# Valid categories: SQL_SUBMISSION, USER_INTERACTION, DB_EXPLORATION, KNOWLEDGE_LOOKUP
# Priority when multiple categories appear in one turn:
#   SQL_SUBMISSION > USER_INTERACTION > specific single category > MIXED
tool_categories:
  submit_sql: SQL_SUBMISSION
  ask_user: USER_INTERACTION
  execute_sql: DB_EXPLORATION
  get_schema: DB_EXPLORATION
  get_column_meaning: DB_EXPLORATION
  get_all_column_meanings: DB_EXPLORATION
  get_all_external_knowledge_names: KNOWLEDGE_LOOKUP
  get_knowledge_definition: KNOWLEDGE_LOOKUP
  get_all_knowledge_definitions: KNOWLEDGE_LOOKUP
```

- [ ] **Step 2: Verify the file parses cleanly**

```bash
uv run python -c "
import yaml
with open('configs/turn_classifier/tool_categories.yaml') as f:
    data = yaml.safe_load(f)
print(data['tool_categories'])
"
```

Expected: prints a dict with 9 entries, all values are one of the 4 valid categories.

- [ ] **Step 3: Commit**

```bash
git add configs/turn_classifier/tool_categories.yaml
git commit -m "chore: add turn_classifier tool-categories YAML config"
```

---

## Task 2: Pydantic Schemas

**Files:**
- Create: `src/conversation2sql/eval_framework/turn_classifier/schemas.py`
- Create: `src/conversation2sql/eval_framework/turn_classifier/__init__.py`

- [ ] **Step 1: Write the failing test**

Create `tests/eval_framework/turn_classifier/__init__.py` (empty) then
create `tests/eval_framework/turn_classifier/test_schemas.py`:

```python
"""Tests for turn_classifier.schemas."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from conversation2sql.eval_framework.turn_classifier.schemas import (
    Level1Classification,
    TurnClassification,
)


def test_level1_classification_valid():
    obj = Level1Classification(
        reasoning="The agent asks one focused question.",
        level1_category="CLARIFICATION",
        level1_alternatives=[],
        confidence="CONFIDENT",
    )
    assert obj.level1_category == "CLARIFICATION"
    assert obj.level1_alternatives == []


def test_level1_classification_with_alternatives():
    obj = Level1Classification(
        reasoning="Could be CLARIFICATION or INTERROGATION.",
        level1_category="CLARIFICATION",
        level1_alternatives=["INTERROGATION"],
        confidence="PLAUSIBLE",
    )
    assert "INTERROGATION" in obj.level1_alternatives


def test_level1_invalid_confidence():
    with pytest.raises(ValidationError):
        Level1Classification(
            reasoning="...",
            level1_category="CLARIFICATION",
            level1_alternatives=[],
            confidence="MAYBE",  # not in Literal
        )


def test_turn_classification_valid():
    obj = TurnClassification(
        message_index=3,
        level2_category="USER_INTERACTION",
        level2_tools_called=["ask_user"],
        reasoning="Asks a single question.",
        level1_category="CLARIFICATION",
        level1_alternatives=[],
        confidence="CONFIDENT",
    )
    assert obj.message_index == 3
    assert obj.level2_category == "USER_INTERACTION"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/eval_framework/turn_classifier/test_schemas.py -v
```

Expected: `ImportError` — `turn_classifier` module does not exist yet.

- [ ] **Step 3: Create schemas.py**

```python
"""Pydantic models for the turn classifier."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

Confidence = Literal["CERTAIN", "CONFIDENT", "PLAUSIBLE", "UNCERTAIN"]


class Level1Classification(BaseModel):
    """Returned by the LLM judge — semantic fields only.

    reasoning is declared first so the model writes CoT before committing to a label.
    """

    reasoning: str
    level1_category: str
    level1_alternatives: list[str]
    confidence: Confidence


class TurnClassification(BaseModel):
    """Per-turn output assembled from Level 2 (rules) + Level 1 (LLM)."""

    message_index: int
    level2_category: str
    level2_tools_called: list[str]
    reasoning: str
    level1_category: str
    level1_alternatives: list[str]
    confidence: Confidence
```

- [ ] **Step 4: Create __init__.py**

```python
"""Turn classifier package."""
from conversation2sql.eval_framework.turn_classifier.schemas import (
    Level1Classification,
    TurnClassification,
)
from conversation2sql.eval_framework.turn_classifier.classifier import TurnClassifier

__all__ = ["Level1Classification", "TurnClassification", "TurnClassifier"]
```

- [ ] **Step 5: Run test to verify it passes**

```bash
uv run pytest tests/eval_framework/turn_classifier/test_schemas.py -v
```

Expected: 4 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/conversation2sql/eval_framework/turn_classifier/schemas.py \
        src/conversation2sql/eval_framework/turn_classifier/__init__.py \
        tests/eval_framework/turn_classifier/__init__.py \
        tests/eval_framework/turn_classifier/test_schemas.py
git commit -m "feat: add turn_classifier Pydantic schemas"
```

---

## Task 3: Rule-Based Level 2 Classifier

**Files:**
- Create: `src/conversation2sql/eval_framework/turn_classifier/rules.py`
- Create/Modify: `tests/eval_framework/turn_classifier/test_rules.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/eval_framework/turn_classifier/test_rules.py`:

```python
"""Unit tests for turn_classifier.rules — deterministic Level 2 classification."""
from __future__ import annotations

import pytest

from conversation2sql.eval_framework.turn_classifier.rules import (
    classify_level2,
    load_tool_categories,
)

# Inline fixture — no YAML dependency in unit tests
CATEGORIES = {
    "submit_sql": "SQL_SUBMISSION",
    "ask_user": "USER_INTERACTION",
    "execute_sql": "DB_EXPLORATION",
    "get_schema": "DB_EXPLORATION",
    "get_knowledge_definition": "KNOWLEDGE_LOOKUP",
}


def _tc(name: str) -> dict:
    """Build a minimal tool_call dict."""
    return {"tool_name": name, "arguments": {}, "tool_cost": 0}


# --- single-tool cases ---


def test_sql_submission():
    cat, tools = classify_level2([_tc("submit_sql")], CATEGORIES, has_content=False)
    assert cat == "SQL_SUBMISSION"
    assert tools == ["submit_sql"]


def test_user_interaction():
    cat, tools = classify_level2([_tc("ask_user")], CATEGORIES, has_content=False)
    assert cat == "USER_INTERACTION"
    assert tools == ["ask_user"]


def test_db_exploration_single():
    cat, tools = classify_level2([_tc("execute_sql")], CATEGORIES, has_content=True)
    assert cat == "DB_EXPLORATION"


def test_knowledge_lookup():
    cat, tools = classify_level2([_tc("get_knowledge_definition")], CATEGORIES, has_content=False)
    assert cat == "KNOWLEDGE_LOOKUP"


# --- no-tool cases ---


def test_no_action_empty_content():
    cat, tools = classify_level2([], CATEGORIES, has_content=False)
    assert cat == "NO_ACTION"
    assert tools == []


def test_text_only():
    cat, tools = classify_level2([], CATEGORIES, has_content=True)
    assert cat == "TEXT_ONLY"
    assert tools == []


# --- multi-tool cases ---


def test_db_exploration_multi():
    calls = [_tc("execute_sql"), _tc("get_schema")]
    cat, tools = classify_level2(calls, CATEGORIES, has_content=False)
    assert cat == "DB_EXPLORATION"
    assert set(tools) == {"execute_sql", "get_schema"}


def test_mixed():
    calls = [_tc("execute_sql"), _tc("get_knowledge_definition")]
    cat, tools = classify_level2(calls, CATEGORIES, has_content=False)
    assert cat == "MIXED"


# --- priority ---


def test_sql_submission_priority_over_user():
    calls = [_tc("submit_sql"), _tc("ask_user")]
    cat, _ = classify_level2(calls, CATEGORIES, has_content=False)
    assert cat == "SQL_SUBMISSION"


def test_sql_submission_priority_over_db():
    calls = [_tc("submit_sql"), _tc("execute_sql")]
    cat, _ = classify_level2(calls, CATEGORIES, has_content=False)
    assert cat == "SQL_SUBMISSION"


def test_user_interaction_priority_over_db():
    calls = [_tc("ask_user"), _tc("execute_sql")]
    cat, _ = classify_level2(calls, CATEGORIES, has_content=False)
    assert cat == "USER_INTERACTION"


# --- unknown tool ---


def test_unknown_tool():
    cat, tools = classify_level2([_tc("mystery_tool")], CATEGORIES, has_content=False)
    assert cat == "UNKNOWN"
    assert "mystery_tool" in tools


def test_load_tool_categories(tmp_path):
    yaml_content = "tool_categories:\n  submit_sql: SQL_SUBMISSION\n"
    p = tmp_path / "cats.yaml"
    p.write_text(yaml_content)
    cats = load_tool_categories(p)
    assert cats == {"submit_sql": "SQL_SUBMISSION"}
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/eval_framework/turn_classifier/test_rules.py -v
```

Expected: `ImportError` — `rules` module does not exist yet.

- [ ] **Step 3: Implement rules.py**

```python
"""Deterministic Level 2 classifier from tool_calls + YAML config."""
from __future__ import annotations

from pathlib import Path

import yaml

# Priority order — first match wins
_PRIORITY_ORDER = ["SQL_SUBMISSION", "USER_INTERACTION"]


def load_tool_categories(yaml_path: str | Path) -> dict[str, str]:
    """Load the tool_name → Level2Category mapping from a YAML file."""
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    return data.get("tool_categories", {})


def classify_level2(
    tool_calls: list[dict],
    tool_categories: dict[str, str],
    has_content: bool = True,
) -> tuple[str, list[str]]:
    """Return (level2_category, tools_called) for one AI message.

    Args:
        tool_calls: list of tool_call dicts, each with a 'tool_name' key.
        tool_categories: mapping of tool_name → category string (from YAML).
        has_content: True when the AI message has non-empty content items.

    Categories:
        SQL_SUBMISSION  — any submit_sql call present
        USER_INTERACTION — any ask_user call (no submit_sql)
        DB_EXPLORATION  — only DB-type tools
        KNOWLEDGE_LOOKUP — only KB-type tools
        MIXED           — tools span more than one category
        TEXT_ONLY       — no tools, non-empty content
        NO_ACTION       — no tools, empty content
        UNKNOWN         — a tool is present but not in the YAML config
    """
    tool_names = [tc["tool_name"] for tc in tool_calls]

    if not tool_names:
        return ("TEXT_ONLY" if has_content else "NO_ACTION"), []

    categories: set[str] = set()
    unknown_found = False
    for name in tool_names:
        cat = tool_categories.get(name)
        if cat is None:
            unknown_found = True
        else:
            categories.add(cat)

    if unknown_found:
        return "UNKNOWN", tool_names

    # Priority: SQL_SUBMISSION > USER_INTERACTION
    for priority_cat in _PRIORITY_ORDER:
        if priority_cat in categories:
            return priority_cat, tool_names

    if len(categories) == 1:
        return categories.pop(), tool_names

    return "MIXED", tool_names
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/eval_framework/turn_classifier/test_rules.py -v
```

Expected: 14 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/conversation2sql/eval_framework/turn_classifier/rules.py \
        tests/eval_framework/turn_classifier/test_rules.py
git commit -m "feat: add turn_classifier Level 2 rule-based classifier"
```

---

## Task 4: LLM Judge Prompt Template

**Files:**
- Create: `src/conversation2sql/eval_framework/turn_classifier/prompts.py`
- Create: `tests/eval_framework/turn_classifier/test_prompts.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/eval_framework/turn_classifier/test_prompts.py`:

```python
"""Tests for turn_classifier.prompts."""
from __future__ import annotations

from conversation2sql.eval_framework.turn_classifier.prompts import build_judge_prompt

ALL_LEVEL1_LABELS = [
    "ANSWER_ATTEMPT", "REVISION", "CLARIFICATION", "INTERROGATION",
    "ASSUMPTION", "CONFIRMATION", "DISCUSSION", "HEDGING", "REFUSAL", "MISSING",
]

ALL_CONFIDENCE_VALUES = ["CERTAIN", "CONFIDENT", "PLAUSIBLE", "UNCERTAIN"]


def _render(overrides: dict | None = None) -> list[dict]:
    params = {
        "thinking": "I should ask the user about the date range.",
        "text": "Could you clarify the date range?",
        "tool_calls_summary": "ask_user(question=Could you clarify the date range?)",
        "level2_category": "USER_INTERACTION",
        "prior_failed_submit": False,
    }
    if overrides:
        params.update(overrides)
    return build_judge_prompt(params)


def test_returns_list_of_dicts():
    msgs = _render()
    assert isinstance(msgs, list)
    assert all(isinstance(m, dict) for m in msgs)
    assert all("role" in m and "content" in m for m in msgs)


def test_has_system_and_user():
    msgs = _render()
    roles = [m["role"] for m in msgs]
    assert "system" in roles
    assert "user" in roles


def test_all_level1_labels_present():
    msgs = _render()
    full_text = " ".join(m["content"] for m in msgs)
    for label in ALL_LEVEL1_LABELS:
        assert label in full_text, f"Label {label!r} missing from prompt"


def test_all_confidence_values_present():
    msgs = _render()
    full_text = " ".join(m["content"] for m in msgs)
    for val in ALL_CONFIDENCE_VALUES:
        assert val in full_text, f"Confidence value {val!r} missing from prompt"


def test_thinking_included():
    msgs = _render()
    full_text = " ".join(m["content"] for m in msgs)
    assert "I should ask the user about the date range" in full_text


def test_prior_failed_submit_shown():
    msgs = _render({"prior_failed_submit": True})
    full_text = " ".join(m["content"] for m in msgs)
    assert "True" in full_text or "true" in full_text.lower()


def test_no_thinking_omits_thinking_section():
    msgs = _render({"thinking": None})
    full_text = " ".join(m["content"] for m in msgs)
    # The thinking section header should not appear when there's nothing to show
    assert "I should ask the user about the date range" not in full_text


def test_level2_category_in_prompt():
    msgs = _render()
    full_text = " ".join(m["content"] for m in msgs)
    assert "USER_INTERACTION" in full_text
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/eval_framework/turn_classifier/test_prompts.py -v
```

Expected: `ImportError` — `prompts` module does not exist yet.

- [ ] **Step 3: Implement prompts.py**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/eval_framework/turn_classifier/test_prompts.py -v
```

Expected: 8 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/conversation2sql/eval_framework/turn_classifier/prompts.py \
        tests/eval_framework/turn_classifier/test_prompts.py
git commit -m "feat: add turn_classifier Jinja2 judge prompt"
```

---

## Task 5: TurnClassifier

**Files:**
- Create: `src/conversation2sql/eval_framework/turn_classifier/classifier.py`
- Create: `tests/eval_framework/turn_classifier/test_classifier.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/eval_framework/turn_classifier/test_classifier.py`:

```python
"""Unit tests for TurnClassifier (LLM mocked)."""
from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from conversation2sql.eval_framework.turn_classifier.classifier import TurnClassifier
from conversation2sql.eval_framework.turn_classifier.schemas import (
    Level1Classification,
    TurnClassification,
)

TOOL_CATEGORIES = {
    "submit_sql": "SQL_SUBMISSION",
    "ask_user": "USER_INTERACTION",
    "execute_sql": "DB_EXPLORATION",
    "get_knowledge_definition": "KNOWLEDGE_LOOKUP",
}


def _make_classifier(level1: Level1Classification) -> TurnClassifier:
    structured = MagicMock()
    structured.invoke.return_value = level1
    model = MagicMock()
    model.with_structured_output.return_value = structured
    return TurnClassifier(model=model, tool_categories=TOOL_CATEGORIES)


def _l1(**kwargs) -> Level1Classification:
    defaults = dict(
        reasoning="some reasoning",
        level1_category="DISCUSSION",
        level1_alternatives=[],
        confidence="CONFIDENT",
    )
    return Level1Classification(**{**defaults, **kwargs})


# --- result type ---


def test_returns_turn_classification():
    clf = _make_classifier(_l1())
    result = clf.classify_turn(
        message_index=2,
        ai_msg={"content": [{"type": "thinking", "text": "hmm"}], "tool_calls": []},
        prior_failed_submit=False,
    )
    assert isinstance(result, TurnClassification)


# --- level 2 wiring ---


def test_sql_submission_level2():
    clf = _make_classifier(_l1(level1_category="ANSWER_ATTEMPT", confidence="CERTAIN"))
    result = clf.classify_turn(
        message_index=5,
        ai_msg={
            "content": [{"type": "thinking", "text": "I'll submit."}],
            "tool_calls": [{"tool_name": "submit_sql", "arguments": {"sql": "SELECT 1"}, "tool_cost": 0}],
        },
        prior_failed_submit=False,
    )
    assert result.level2_category == "SQL_SUBMISSION"
    assert result.level2_tools_called == ["submit_sql"]


def test_user_interaction_level2():
    clf = _make_classifier(_l1(level1_category="CLARIFICATION"))
    result = clf.classify_turn(
        message_index=3,
        ai_msg={
            "content": [{"type": "thinking", "text": "I need to ask."}],
            "tool_calls": [{"tool_name": "ask_user", "arguments": {"question": "What do you mean?"}, "tool_cost": 0}],
        },
        prior_failed_submit=False,
    )
    assert result.level2_category == "USER_INTERACTION"


def test_no_action_empty_turn():
    clf = _make_classifier(_l1(level1_category="MISSING", confidence="CERTAIN"))
    result = clf.classify_turn(
        message_index=10,
        ai_msg={"content": [], "tool_calls": []},
        prior_failed_submit=False,
    )
    assert result.level2_category == "NO_ACTION"
    assert result.level2_tools_called == []


def test_text_only_level2():
    clf = _make_classifier(_l1(level1_category="DISCUSSION"))
    result = clf.classify_turn(
        message_index=7,
        ai_msg={
            "content": [{"type": "text", "text": "Let me think about this."}],
            "tool_calls": [],
        },
        prior_failed_submit=False,
    )
    assert result.level2_category == "TEXT_ONLY"


# --- level 1 wiring ---


def test_level1_fields_passed_through():
    clf = _make_classifier(_l1(
        reasoning="Clearly asking one question.",
        level1_category="CLARIFICATION",
        level1_alternatives=[],
        confidence="CONFIDENT",
    ))
    result = clf.classify_turn(
        message_index=1,
        ai_msg={"content": [], "tool_calls": []},
        prior_failed_submit=False,
    )
    assert result.reasoning == "Clearly asking one question."
    assert result.level1_category == "CLARIFICATION"
    assert result.confidence == "CONFIDENT"
    assert result.level1_alternatives == []


def test_alternatives_passed_through():
    clf = _make_classifier(_l1(
        level1_category="REVISION",
        level1_alternatives=["ANSWER_ATTEMPT"],
        confidence="PLAUSIBLE",
    ))
    result = clf.classify_turn(
        message_index=8,
        ai_msg={"content": [], "tool_calls": []},
        prior_failed_submit=True,
    )
    assert result.level1_alternatives == ["ANSWER_ATTEMPT"]


# --- message_index ---


def test_message_index_preserved():
    clf = _make_classifier(_l1())
    result = clf.classify_turn(
        message_index=42,
        ai_msg={"content": [], "tool_calls": []},
        prior_failed_submit=False,
    )
    assert result.message_index == 42
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/eval_framework/turn_classifier/test_classifier.py -v
```

Expected: `ImportError` — `classifier` module does not exist yet.

- [ ] **Step 3: Implement classifier.py**

```python
"""TurnClassifier: orchestrates Level 2 (rules) + Level 1 (LLM judge)."""
from __future__ import annotations

import json

from langchain_core.language_models import BaseChatModel

from conversation2sql.eval_framework.turn_classifier.prompts import build_judge_prompt
from conversation2sql.eval_framework.turn_classifier.rules import classify_level2
from conversation2sql.eval_framework.turn_classifier.schemas import (
    Level1Classification,
    TurnClassification,
)


def _extract_content_parts(content_items: list) -> tuple[str | None, str | None]:
    """Split content list into (thinking_text, visible_text)."""
    thinking_parts: list[str] = []
    text_parts: list[str] = []
    for item in content_items:
        if not isinstance(item, dict):
            continue
        t = item.get("type")
        text = item.get("text", "")
        if t == "thinking":
            thinking_parts.append(text)
        elif t == "text":
            text_parts.append(text)
    return (
        "\n".join(thinking_parts) or None,
        "\n".join(text_parts) or None,
    )


def _summarize_tool_calls(tool_calls: list[dict], max_chars: int = 300) -> str:
    if not tool_calls:
        return "(none)"
    parts = []
    for tc in tool_calls:
        name = tc.get("tool_name", "?")
        args = tc.get("arguments", {})
        args_str = json.dumps(args)[:80]
        parts.append(f"{name}({args_str})")
    return "; ".join(parts)[:max_chars]


class TurnClassifier:
    """Classify one AI turn at Level 2 (rules) and Level 1 (LLM judge)."""

    def __init__(self, model: BaseChatModel, tool_categories: dict[str, str]) -> None:
        self._tool_categories = tool_categories
        self._judge = model.with_structured_output(Level1Classification)

    def classify_turn(
        self,
        message_index: int,
        ai_msg: dict,
        prior_failed_submit: bool,
    ) -> TurnClassification:
        """Classify a single AI message dict from results_smaller.jsonl.

        Args:
            message_index: index of this message in the record's messages list.
            ai_msg: the AI message dict (keys: content, tool_calls, ...).
            prior_failed_submit: True if any earlier submit_sql call returned passed=False.
        """
        tool_calls: list[dict] = ai_msg.get("tool_calls", [])
        content_items: list = ai_msg.get("content", [])
        has_content = bool(content_items)

        level2_cat, tools_called = classify_level2(
            tool_calls, self._tool_categories, has_content=has_content
        )

        thinking, text = _extract_content_parts(content_items)
        tool_calls_summary = _summarize_tool_calls(tool_calls)

        messages = build_judge_prompt(
            params={
                "thinking": thinking,
                "text": text,
                "tool_calls_summary": tool_calls_summary,
                "level2_category": level2_cat,
                "prior_failed_submit": prior_failed_submit,
            }
        )

        level1: Level1Classification = self._judge.invoke(messages)

        return TurnClassification(
            message_index=message_index,
            level2_category=level2_cat,
            level2_tools_called=tools_called,
            reasoning=level1.reasoning,
            level1_category=level1.level1_category,
            level1_alternatives=level1.level1_alternatives,
            confidence=level1.confidence,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/eval_framework/turn_classifier/test_classifier.py -v
```

Expected: 9 tests PASS.

- [ ] **Step 5: Run the full test suite to catch regressions**

```bash
uv run pytest tests/ -v
```

Expected: all existing tests still pass; new tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/conversation2sql/eval_framework/turn_classifier/classifier.py \
        src/conversation2sql/eval_framework/turn_classifier/__init__.py \
        tests/eval_framework/turn_classifier/test_classifier.py
git commit -m "feat: add TurnClassifier orchestrating Level 2 + Level 1 LLM judge"
```

---

## Task 6: CLI Script

**Files:**
- Create: `scripts/classify_turns.py`

- [ ] **Step 1: Implement classify_turns.py**

```python
#!/usr/bin/env python3
"""CLI: classify every agent turn in one or more results_smaller.jsonl files.

Usage:
    uv run python scripts/classify_turns.py results/*/results_smaller.jsonl \\
        --output results/classified.jsonl \\
        --model openai/gpt-4o-mini \\
        --tool-categories configs/turn_classifier/tool_categories.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv
from tqdm import tqdm

from conversation2sql.eval_framework.agents.utils import utils_create_model
from conversation2sql.eval_framework.turn_classifier.classifier import TurnClassifier
from conversation2sql.eval_framework.turn_classifier.rules import load_tool_categories


def iter_records(path: Path):
    """Yield JSON records handling both pretty-printed and compact JSONL formats."""
    text = path.read_text()
    decoder = json.JSONDecoder()
    pos = 0
    while pos < len(text):
        stripped = text[pos:]
        lstripped = stripped.lstrip()
        if not lstripped:
            break
        whitespace_chars = len(stripped) - len(lstripped)
        try:
            obj, end = decoder.raw_decode(lstripped)
            yield obj
            pos += whitespace_chars + end
        except json.JSONDecodeError:
            break


def classify_record(record: dict, classifier: TurnClassifier) -> dict:
    """Add turn_classifications list to a single result record."""
    messages = record.get("messages", [])
    prior_failed_submit = False
    turn_classifications = []

    for i, msg in enumerate(messages):
        role = msg.get("role")
        if role == "ai":
            tc = classifier.classify_turn(
                message_index=i,
                ai_msg=msg,
                prior_failed_submit=prior_failed_submit,
            )
            turn_classifications.append(tc.model_dump())
        elif role == "tool" and msg.get("tool_name") == "submit_sql":
            content = msg.get("content", {})
            if isinstance(content, dict) and not content.get("passed", True):
                prior_failed_submit = True

    return {**record, "turn_classifications": turn_classifications}


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Classify agent turns in BIRD-Interact result traces."
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="Input JSONL file(s)")
    parser.add_argument("--output", type=Path, default=None, help="Output JSONL path")
    parser.add_argument("--model", default="openai/gpt-4o-mini", help="LiteLLM model string (provider/model)")
    parser.add_argument(
        "--tool-categories",
        default="configs/turn_classifier/tool_categories.yaml",
        help="Path to tool_categories.yaml",
    )
    args = parser.parse_args()

    if args.output is None:
        if len(args.inputs) == 1:
            args.output = args.inputs[0].parent / "results_classified.jsonl"
        else:
            args.output = Path("results_classified.jsonl")

    provider, model_name = args.model.split("/", 1)
    model = utils_create_model(
        model_name=model_name,
        model_provider=provider,
        temperature=0.0,
        max_tokens=1024,
    )

    tool_categories = load_tool_categories(args.tool_categories)
    classifier = TurnClassifier(model=model, tool_categories=tool_categories)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    processed = 0
    errors = 0

    with open(args.output, "w") as out_f:
        for input_path in args.inputs:
            records = list(iter_records(input_path))
            for record in tqdm(records, desc=str(input_path)):
                instance_id = record.get("instance_id", "?")
                try:
                    classified = classify_record(record, classifier)
                    out_f.write(json.dumps(classified) + "\n")
                    processed += 1
                except Exception as exc:
                    print(
                        f"ERROR [{instance_id}]: {exc}",
                        file=sys.stderr,
                    )
                    errors += 1

    print(f"\nDone — {processed} records classified, {errors} errors.")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify the script is importable (no LLM call)**

```bash
uv run python -c "import scripts.classify_turns; print('ok')" 2>/dev/null || \
uv run python scripts/classify_turns.py --help
```

Expected: help text printed, no import errors.

- [ ] **Step 3: Smoke-test against the fixture file (requires API key)**

```bash
uv run python scripts/classify_turns.py \
  results/2026-05-01/10-28-14/main/results_smaller.jsonl \
  --model openai/gpt-4o-mini \
  --tool-categories configs/turn_classifier/tool_categories.yaml
```

Expected:
- Output written to `results/2026-05-01/10-28-14/main/results_classified.jsonl`
- Progress bar shows 1 record processed
- The output file contains the original record plus a `turn_classifications` list with one entry per AI message (11 AI messages in the trace)
- Each entry has keys: `message_index`, `level2_category`, `level2_tools_called`, `reasoning`, `level1_category`, `level1_alternatives`, `confidence`

Verify the output:

```bash
uv run python -c "
import json
with open('results/2026-05-01/10-28-14/main/results_classified.jsonl') as f:
    record = json.loads(f.readline())
tcs = record['turn_classifications']
print(f'Turn classifications: {len(tcs)}')
for tc in tcs:
    print(f'  msg {tc[\"message_index\"]}: L2={tc[\"level2_category\"]}, L1={tc[\"level1_category\"]}, conf={tc[\"confidence\"]}')
"
```

Expected: 11 lines printed, each with a valid L2 category and L1 label.

- [ ] **Step 4: Run full test suite to confirm no regressions**

```bash
uv run pytest tests/ -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add scripts/classify_turns.py
git commit -m "feat: add classify_turns CLI script"
```

---

## Self-Review

### Spec coverage check

| Spec requirement | Task |
|---|---|
| Level 1 taxonomy (10 labels) | Task 4 (prompt) + Task 2 (schemas) |
| Level 2 taxonomy (7+ categories, YAML-driven) | Task 1 (YAML) + Task 3 (rules.py) |
| YAML-based tool categories, no hardcoded names | Task 1 + Task 3 |
| `prior_failed_submit` cross-turn context | Task 5 (classifier), Task 6 (classify_record) |
| `model.with_structured_output(Level1Classification)` | Task 5 |
| CoT: reasoning before label | Task 2 (field order in Pydantic) + Task 4 (prompt) |
| Confidence: 4-level discrete scale | Task 2 (schemas) + Task 4 (prompt) |
| `level1_alternatives` when PLAUSIBLE/UNCERTAIN | Task 2 + Task 4 |
| Both thinking AND text content extracted | Task 5 (`_extract_content_parts`) |
| Handles pretty-printed + compact JSONL | Task 6 (`iter_records` with `raw_decode`) |
| CLI: multi-file input, `--output`, `--model`, `--tool-categories` | Task 6 |
| Default output path | Task 6 |
| Per-record error handling: log and continue | Task 6 |
| Unit tests for rules.py | Task 3 |
| Unit tests for classifier.py (mocked LLM) | Task 5 |
| Integration/smoke test against fixture | Task 6 step 3 |

All spec requirements are covered.

### Placeholder scan

No TBD, TODO, or placeholder text in any code block. All steps show complete code.

### Type consistency

- `classify_level2` returns `tuple[str, list[str]]` — matches usage in `classifier.py`
- `TurnClassifier.__init__` takes `model: BaseChatModel` — matches `utils_create_model` return type (`ChatLiteLLM` extends `BaseChatModel`)
- `build_judge_prompt(params: dict) -> list[dict]` — matches `model.invoke(messages)` call in `classifier.py`
- `tc.model_dump()` on `TurnClassification` — Pydantic v2 method, correct
- `level1_alternatives: list[str]` — consistent across `Level1Classification`, `TurnClassification`, and test assertions
