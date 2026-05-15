# BIRD-Interact Baselines Ablation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a 4-cell ablation matrix (`no_tool`, `tools_only`, `tools_user`, `bird_full`) selectable via a single `ConfigPipeline.baseline` enum, sharing the existing middleware/budget machinery via a parameterizable agent. Also fix the broken `no_tool_baseline` evaluation path.

**Architecture:** A single parameterizable `run_agent_bird_baseline(..., enable_ask_user)` covers three of the four baselines; `run_baseline_no_tool` covers the text-only one. The pipeline's `_resolve_baseline_settings` maps the `baseline` enum to `(make_data_ambiguous, runner, needs_user_sim)`. Budget formula is baseline-aware (`6 + 2*patience` when no ambiguity, `6 + 2*m_amb + 2*patience` otherwise).

**Tech Stack:** Python 3.12, `uv`, LangChain `create_agent`, LangGraph middleware, ChatLiteLLM, Pydantic, pytest with `asyncio_mode=auto`.

**Spec:** `docs/superpowers/specs/2026-05-07-baselines-ablation-design.md`

**Commit policy:** This repo's user commits manually. Each task ends with `git add` only — DO NOT run `git commit`. Commit boundaries are noted so the user can commit grouped changes themselves.

---

## File map

**Source files modified:**
- `src/conversation2sql/config_input.py` — add `baseline` enum to `ConfigPipeline`
- `src/conversation2sql/eval_framework/dataset_readers/bird_interact_reader.py` — `_calculate_initial_budget` gains `count_ambiguity` param
- `src/conversation2sql/eval_framework/agents/no_tool_baseline/prompts.py` — fix system/user swap bug
- `src/conversation2sql/eval_framework/agents/no_tool_baseline/baseline_model.py` — rename + rewrite to use `submit_sql_impl`
- `src/conversation2sql/eval_framework/agents/bird_baseline/prompts.py` — Jinja conditional on `enable_ask_user`
- `src/conversation2sql/eval_framework/agents/bird_baseline/agent_code.py` — `enable_ask_user` kwarg, conditional `ask_user` tool
- `src/conversation2sql/eval_framework/agents/__init__.py` — re-export `run_baseline_no_tool`
- `src/conversation2sql/eval_framework/main_pipe_workflow.py` — dispatch logic, per-baseline output path

**CLAUDE.md updates:**
- `src/conversation2sql/eval_framework/agents/CLAUDE.md` — function rename, new variants
- `src/conversation2sql/eval_framework/agents/no_tool_baseline/CLAUDE.md` — function rename, new behavior
- `src/conversation2sql/eval_framework/agents/bird_baseline/CLAUDE.md` — `enable_ask_user` knob

**New test files:**
- `tests/eval_framework/agents/__init__.py`
- `tests/eval_framework/agents/test_no_tool_baseline.py`
- `tests/eval_framework/agents/test_bird_baseline_agent.py`
- `tests/eval_framework/dataset_readers/__init__.py`
- `tests/eval_framework/dataset_readers/test_budget.py`
- `tests/eval_framework/test_main_pipe_workflow.py`
- `tests/eval_framework/integration/__init__.py`
- `tests/eval_framework/integration/test_baselines_smoke.py` (gated `@pytest.mark.integration`)

---

## Task 1: Fix `no_tool_baseline/prompts.py` system/user swap bug

The current code passes `_BASE_MODEL_SYSTEM` (empty string) as both system and user — the actual user prompt is never rendered.

**Files:**
- Modify: `src/conversation2sql/eval_framework/agents/no_tool_baseline/prompts.py:38`

- [ ] **Step 1: Apply the fix**

Replace the `build_omnisql_prompt` body:

```python
def build_omnisql_prompt(
        params: dict,
) -> list[dict]:
    return utils_build_messages(_BASE_MODEL_SYSTEM, _BASE_MODEL_USER, params)
```

- [ ] **Step 2: Verify by import**

```bash
uv run python -c "from conversation2sql.eval_framework.agents.no_tool_baseline.prompts import build_omnisql_prompt; msgs = build_omnisql_prompt({'schema': 'CREATE TABLE t (id INT);', 'question': 'How many?'}); print(msgs)"
```

Expected: a list with one user message whose content includes both `CREATE TABLE t (id INT);` and `How many?`. (System message is omitted because `_BASE_MODEL_SYSTEM` is empty — `utils_build_messages` skips empty system strings.)

- [ ] **Step 3: Stage**

```bash
git add src/conversation2sql/eval_framework/agents/no_tool_baseline/prompts.py
```

Commit point — leave for user to commit.

---

## Task 2: Add `extract_sql_from_response` helper (TDD)

Pure helper that pulls SQL out of the model's free-text response. Lives next to `run_baseline_no_tool` in `baseline_model.py`.

**Files:**
- Modify: `src/conversation2sql/eval_framework/agents/no_tool_baseline/baseline_model.py`
- Create: `tests/eval_framework/agents/__init__.py`
- Create: `tests/eval_framework/agents/test_no_tool_baseline.py`

- [ ] **Step 1: Create test package init**

Create `tests/eval_framework/agents/__init__.py` as an empty file.

```bash
touch tests/eval_framework/agents/__init__.py
```

- [ ] **Step 2: Write the failing tests**

Create `tests/eval_framework/agents/test_no_tool_baseline.py`:

```python
import pytest

from conversation2sql.eval_framework.agents.no_tool_baseline.baseline_model import (
    extract_sql_from_response,
)


class TestExtractSqlFromResponse:
    def test_sql_fenced_block(self):
        text = "Here is the answer:\n```sql\nSELECT 1;\n```\n"
        assert extract_sql_from_response(text) == "SELECT 1;"

    def test_bare_fenced_block(self):
        text = "```\nSELECT 1\n```"
        assert extract_sql_from_response(text) == "SELECT 1"

    def test_returns_last_block_when_multiple(self):
        text = (
            "First draft:\n```sql\nSELECT 0;\n```\n"
            "Final answer:\n```sql\nSELECT 1;\n```\n"
        )
        assert extract_sql_from_response(text) == "SELECT 1;"

    def test_no_block_returns_none(self):
        text = "I think the answer is SELECT 1;"
        assert extract_sql_from_response(text) is None

    def test_block_followed_by_prose(self):
        text = "```sql\nSELECT 1;\n```\nThis returns one row."
        assert extract_sql_from_response(text) == "SELECT 1;"

    def test_whitespace_only_block_returns_none(self):
        text = "```sql\n   \n```"
        assert extract_sql_from_response(text) is None
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
uv run pytest tests/eval_framework/agents/test_no_tool_baseline.py -v
```

Expected: ImportError or 6 FAILs because `extract_sql_from_response` doesn't exist yet.

- [ ] **Step 4: Implement the helper in `baseline_model.py`**

At the top of `src/conversation2sql/eval_framework/agents/no_tool_baseline/baseline_model.py`, add the helper alongside the existing imports. Keep `run_baseline_model` untouched for now (next task rewrites it).

```python
import re

_FENCED_SQL_RE = re.compile(
    r"```(?:sql)?\s*\n?(.*?)\n?```",
    re.IGNORECASE | re.DOTALL,
)


def extract_sql_from_response(text: str) -> str | None:
    """Extract SQL from a model's free-text response.

    Returns the *last* fenced code block (handles models that draft and refine).
    Accepts both ```sql and bare ``` fences. Returns None when no block is
    found or the block is whitespace-only.
    """
    matches = _FENCED_SQL_RE.findall(text)
    for block in reversed(matches):
        stripped = block.strip()
        if stripped:
            return stripped
    return None
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
uv run pytest tests/eval_framework/agents/test_no_tool_baseline.py -v
```

Expected: 6 PASS.

- [ ] **Step 6: Stage**

```bash
git add tests/eval_framework/agents/__init__.py \
        tests/eval_framework/agents/test_no_tool_baseline.py \
        src/conversation2sql/eval_framework/agents/no_tool_baseline/baseline_model.py
```

Commit point.

---

## Task 3: Rewrite `no_tool` runner to use `submit_sql_impl`

Replace the broken `run_baseline_model` with `run_baseline_no_tool`. Goal: emit a result dict whose top-level keys match what `run_agent_bird_baseline` produces, so `_save_record` and the `_smaller.jsonl` projection work without a special case.

**Files:**
- Modify: `src/conversation2sql/eval_framework/agents/no_tool_baseline/baseline_model.py`
- Modify: `tests/eval_framework/agents/test_no_tool_baseline.py`

- [ ] **Step 1: Add a test for the result-dict shape (mocked model + mocked submit_sql_impl)**

Append to `tests/eval_framework/agents/test_no_tool_baseline.py`:

```python
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage

from conversation2sql.eval_framework.agents.no_tool_baseline.baseline_model import (
    run_baseline_no_tool,
)


def _make_task() -> MagicMock:
    task = MagicMock()
    task.ddl_database_schema = "CREATE TABLE t (id INT);"
    task.task_question = "How many rows?"
    return task


def _make_ai(content: str, prompt_tokens: int = 100, completion_tokens: int = 50) -> AIMessage:
    msg = AIMessage(content=content)
    msg.usage_metadata = {
        "input_tokens": prompt_tokens,
        "output_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }
    msg.response_metadata = {"finish_reason": "stop", "_response_cost": 0.001}
    return msg


class TestRunBaselineNoTool:
    @patch("conversation2sql.eval_framework.agents.no_tool_baseline.baseline_model.submit_sql_impl")
    def test_passes_when_sql_extracted_and_submit_passes(self, mock_submit):
        mock_submit.return_value = {"passed": True, "pred_result": [], "gt_result": []}
        model = MagicMock()
        model.invoke.return_value = _make_ai("```sql\nSELECT COUNT(*) FROM t;\n```")

        result = run_baseline_no_tool(_make_task(), model)

        assert result["execution_accuracy"] is True
        assert result["tool_calls_in_order"] == []
        assert result["initial_user_patience"] is None
        assert result["updated_user_patience"] is None
        # message log: user + ai + synthetic submit_sql_offline tool message
        assert len(result["messages"]) == 3
        assert result["messages"][-1]["tool_name"] == "submit_sql_offline"
        assert result["total_tokens"] == 150

    @patch("conversation2sql.eval_framework.agents.no_tool_baseline.baseline_model.submit_sql_impl")
    def test_records_failure_when_no_sql_block(self, mock_submit):
        model = MagicMock()
        model.invoke.return_value = _make_ai("I cannot answer this.")

        result = run_baseline_no_tool(_make_task(), model)

        assert result["execution_accuracy"] is False
        assert result["messages"][-1]["content"]["error"] == "no_sql_block_found"
        mock_submit.assert_not_called()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/eval_framework/agents/test_no_tool_baseline.py -v
```

Expected: 2 new FAILs (ImportError on `run_baseline_no_tool`).

- [ ] **Step 3: Rewrite `baseline_model.py`**

Replace the entire file:

```python
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage

from conversation2sql.eval_framework.agents.bird_baseline.tools import submit_sql_impl
from conversation2sql.eval_framework.agents.no_tool_baseline.prompts import build_omnisql_prompt
from conversation2sql.eval_framework.agents.utils import utils_extract_ai_metadata
from conversation2sql.eval_framework.state import TaskData
from conversation2sql.logger import get_logger

import re

logger = get_logger(__name__)

_FENCED_SQL_RE = re.compile(
    r"```(?:sql)?\s*\n?(.*?)\n?```",
    re.IGNORECASE | re.DOTALL,
)


def extract_sql_from_response(text: str) -> str | None:
    """Extract SQL from a model's free-text response.

    Returns the last fenced code block (handles draft-then-refine).
    Accepts ```sql or bare ``` fences. Returns None on no/whitespace-only block.
    """
    matches = _FENCED_SQL_RE.findall(text)
    for block in reversed(matches):
        stripped = block.strip()
        if stripped:
            return stripped
    return None


def run_baseline_no_tool(
        single_task: TaskData,
        model_agent: BaseChatModel,
) -> dict[str, Any]:
    user_messages = build_omnisql_prompt(
        params={
            "schema": single_task.ddl_database_schema,
            "question": single_task.task_question,
        }
    )

    ai_msg: AIMessage = model_agent.invoke(user_messages)  # pyrefly: ignore
    raw_text = ai_msg.content if isinstance(ai_msg.content, str) else str(ai_msg.content)

    sql = extract_sql_from_response(raw_text)
    if sql is None:
        submit_outcome: dict = {
            "passed": False,
            "error": "no_sql_block_found",
            "raw_text": raw_text[:2000],
        }
    else:
        submit_outcome = submit_sql_impl(sql, single_task)

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
        "execution_accuracy": bool(submit_outcome.get("passed", False)),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/eval_framework/agents/test_no_tool_baseline.py -v
```

Expected: 8 PASS (6 from Task 2 + 2 new).

- [ ] **Step 5: Type-check**

```bash
uv run pyrefly check src/conversation2sql/eval_framework/agents/no_tool_baseline/
```

Expected: no new errors (the `pyrefly: ignore` comment on `model_agent.invoke` covers the LangChain message-list shape mismatch already present in `bird_baseline`).

- [ ] **Step 6: Stage**

```bash
git add src/conversation2sql/eval_framework/agents/no_tool_baseline/baseline_model.py \
        tests/eval_framework/agents/test_no_tool_baseline.py
```

Commit point.

---

## Task 4: Update `_calculate_initial_budget` signature (TDD)

Add a `count_ambiguity: bool` param. When `False`, drop the `m_amb` term.

**Files:**
- Modify: `src/conversation2sql/eval_framework/dataset_readers/bird_interact_reader.py:58`
- Create: `tests/eval_framework/dataset_readers/__init__.py`
- Create: `tests/eval_framework/dataset_readers/test_budget.py`

- [ ] **Step 1: Create test package init**

```bash
touch tests/eval_framework/dataset_readers/__init__.py
```

- [ ] **Step 2: Write failing tests**

Create `tests/eval_framework/dataset_readers/test_budget.py`:

```python
from conversation2sql.eval_framework.dataset_readers.bird_interact_reader import (
    _calculate_initial_budget,
)


def _make_line(critical_count: int = 0, knowledge_count: int = 0) -> dict:
    return {
        "user_query_ambiguity": {
            "critical_ambiguity": [{"i": i} for i in range(critical_count)]
        },
        "knowledge_ambiguity": [{"j": j} for j in range(knowledge_count)],
    }


class TestCalculateInitialBudget:
    def test_with_ambiguity_counted(self):
        line = _make_line(critical_count=2, knowledge_count=1)
        # 6 + 2*3 + 2*10 = 32
        assert _calculate_initial_budget(line, user_patience=10, count_ambiguity=True) == 32.0

    def test_with_ambiguity_ignored(self):
        line = _make_line(critical_count=2, knowledge_count=1)
        # 6 + 2*10 = 26 (m_amb dropped)
        assert _calculate_initial_budget(line, user_patience=10, count_ambiguity=False) == 26.0

    def test_no_ambiguity_in_line(self):
        line = _make_line()
        # both flags give the same answer when m_amb=0
        assert _calculate_initial_budget(line, user_patience=5, count_ambiguity=True) == 16.0
        assert _calculate_initial_budget(line, user_patience=5, count_ambiguity=False) == 16.0

    def test_missing_keys_default_to_zero(self):
        # Reader sometimes pops these dict keys before computing budget
        line = {}
        assert _calculate_initial_budget(line, user_patience=3, count_ambiguity=True) == 12.0
        assert _calculate_initial_budget(line, user_patience=3, count_ambiguity=False) == 12.0
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
uv run pytest tests/eval_framework/dataset_readers/test_budget.py -v
```

Expected: 4 FAILs (TypeError: got unexpected keyword `count_ambiguity`).

- [ ] **Step 4: Update the function**

Replace `_calculate_initial_budget` in `src/conversation2sql/eval_framework/dataset_readers/bird_interact_reader.py:58-70`:

```python
def _calculate_initial_budget(line: dict, user_patience: int, count_ambiguity: bool = True) -> float:
    """a-interact budget in bird-coins (per task, paper Section 3.2).

    With count_ambiguity=True (default): 6 + 2*m_amb + 2*patience
    With count_ambiguity=False: 6 + 2*patience (used for the no-ambiguity ablations).
    """
    if not count_ambiguity:
        return 6.0 + 2.0 * user_patience
    critical = len(line.get("user_query_ambiguity", {}).get("critical_ambiguity", []))
    knowledge = len(line.get("knowledge_ambiguity", []))
    m_amb = critical + knowledge
    return 6.0 + 2.0 * m_amb + 2.0 * user_patience
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
uv run pytest tests/eval_framework/dataset_readers/test_budget.py -v
```

Expected: 4 PASS.

- [ ] **Step 6: Stage**

```bash
git add tests/eval_framework/dataset_readers/__init__.py \
        tests/eval_framework/dataset_readers/test_budget.py \
        src/conversation2sql/eval_framework/dataset_readers/bird_interact_reader.py
```

Commit point.

---

## Task 5: Thread `count_ambiguity` through `load_bird_interact_as_tasks`

The reader's caller controls budget shape via `make_data_ambiguous`.

**Files:**
- Modify: `src/conversation2sql/eval_framework/dataset_readers/bird_interact_reader.py:166`

- [ ] **Step 1: Update the call site**

In `load_bird_interact_as_tasks`, change the `task_budget` line to pass `count_ambiguity=make_data_ambiguous`:

```python
                task_budget=_calculate_initial_budget(
                    line, user_patience_budget, count_ambiguity=make_data_ambiguous
                ),
```

- [ ] **Step 2: Smoke-test the reader still loads**

```bash
uv run python -c "
from conversation2sql.eval_framework.dataset_readers.bird_interact_reader import load_bird_interact_as_tasks
samples = load_bird_interact_as_tasks(
    dataset_path='data/bird_interact/bird-interact-full',
    dataset_name_jsonl='data/bird_interact/bird-interact-full/bird_interact_data_GT.jsonl',
    filter_query_category=True,
    db_dsn_template='postgresql://root:123123@localhost:5433/{database}',
    user_patience_budget=10,
    make_data_ambiguous=False,
)
print(f'Loaded {len(samples)}; first task_budget = {samples[0].task_budget}')
print(f'Expected 6 + 2*10 = 26.0 since count_ambiguity=False')
assert samples[0].task_budget == 26.0
"
```

Expected: prints budget, asserts 26.0. (If you can't reach Postgres, the assertion still runs because it's pure math on JSON.)

- [ ] **Step 3: Run the budget tests again to confirm no regression**

```bash
uv run pytest tests/eval_framework/dataset_readers/ -v
```

Expected: all PASS.

- [ ] **Step 4: Stage**

```bash
git add src/conversation2sql/eval_framework/dataset_readers/bird_interact_reader.py
```

Commit point.

---

## Task 6: Make `bird_baseline/prompts.py` Jinja-conditional on `enable_ask_user`

**Files:**
- Modify: `src/conversation2sql/eval_framework/agents/bird_baseline/prompts.py`

- [ ] **Step 1: Update the system template**

Replace `_BIRD_AGENT_SYSTEM` and `build_bird_interact_agent_messages`:

```python
_BIRD_AGENT_SYSTEM = """
You are a helpful PostgreSQL agent that interacts with a {{ "user and a " if enable_ask_user else "" }}database to solve the user's question.

Task description:
Your goal is to understand the user's {{ "ambiguous " if enable_ask_user else "" }}question{{ " involving external knowledge retrieval" if enable_ask_user else "" }} and generate the correct SQL query to solve it.
You can:
{% if enable_ask_user %}1. Interact with the user to ask clarifying questions or submit the SQL query.
2. Interact with the database environment to explore the database and retrieve relevant information.{% else %}1. Interact with the database environment to explore the database and retrieve relevant information.
2. Submit the SQL query when ready.{% endif %}

The interaction ends when you submit the correct SQL query or the budget runs out.
Each action costs bird-coins, so you should be efficient.

Available tools and costs:
- execute_sql: execute a PostgreSQL query. Cost: 1
- get_schema: get the database schema. Cost: 1
- get_all_column_meanings: get all column meanings. Cost: 1
- get_column_meaning: get the meaning of one column. Cost: 0.5
- get_all_external_knowledge_names: get all external knowledge names. Cost: 0.5
- get_knowledge_definition: get one external knowledge definition. Cost: 0.5
- get_all_knowledge_definitions: get all external knowledge definitions. Cost: 1
{% if enable_ask_user %}- ask_user: ask the user a clarification question. Cost: 2
{% endif %}- submit_sql: submit the SQL for evaluation. Cost: 3

Important strategy tips:
- First explore the database schema, column meanings, and relevant external knowledge to understand the task.
{% if enable_ask_user %}- If the user's intent is ambiguous, ask clarifying questions to figure out the real intent before committing to SQL.
- Ask one clarification question at a time.
{% endif %}- Be efficient with your actions to conserve budget.
- Make sure the submitted SQL is valid and addresses all aspects of the question.
- Keep track of the remaining budget and prioritize actions accordingly.
- Be careful with broad retrieval tools such as get_all_column_meanings and get_all_knowledge_definitions because they may return a long context.
- Test SQL with execute_sql before submit_sql when useful.
- If a submission fails and budget remains, debug and try again.
{% if enable_ask_user %}- After a successful phase-1 submission, you may receive a follow-up question for phase 2.
{% endif %}"""

_BIRD_AGENT_USER = """
User's Question: 
{{ amb_user_query }}

[SYSTEM NOTE: You have a total action budget of {{ total_budget }} units. Each action consumes budget. If the budget runs out, you must submit.]
"""


def build_bird_interact_agent_messages(
        params: dict,
) -> list[dict]:
    return utils_build_messages(_BIRD_AGENT_SYSTEM, _BIRD_AGENT_USER, params)
```

- [ ] **Step 2: Verify both prompt variants render correctly**

```bash
uv run python -c "
from conversation2sql.eval_framework.agents.bird_baseline.prompts import build_bird_interact_agent_messages
on = build_bird_interact_agent_messages({'total_budget': 26, 'amb_user_query': 'q', 'enable_ask_user': True})
off = build_bird_interact_agent_messages({'total_budget': 26, 'amb_user_query': 'q', 'enable_ask_user': False})
assert 'ask_user' in on[0]['content'] and 'ambiguous' in on[0]['content']
assert 'ask_user' not in off[0]['content'] and 'ambiguous' not in off[0]['content']
print('OK — both variants render correctly')
"
```

Expected: prints `OK`.

- [ ] **Step 3: Stage**

```bash
git add src/conversation2sql/eval_framework/agents/bird_baseline/prompts.py
```

Commit point.

---

## Task 7: Add `enable_ask_user` kwarg to `run_agent_bird_baseline` (with tests)

**Files:**
- Modify: `src/conversation2sql/eval_framework/agents/bird_baseline/agent_code.py`
- Create: `tests/eval_framework/agents/test_bird_baseline_agent.py`

- [ ] **Step 1: Write failing tests for tool-list assembly**

Create `tests/eval_framework/agents/test_bird_baseline_agent.py`:

```python
from unittest.mock import MagicMock

import pytest


def _make_models():
    return MagicMock(name="agent"), MagicMock(name="parser"), MagicMock(name="generator")


def _make_task():
    task = MagicMock()
    task.task_budget = 26
    task.task_question = "q"
    task.instance_id = "test_1"
    return task


class TestRunAgentBirdBaseline:
    def test_enable_ask_user_false_excludes_ask_user_tool(self, monkeypatch):
        from conversation2sql.eval_framework.agents.bird_baseline import agent_code

        captured = {}

        def fake_create_agent(model, tools, **kwargs):
            captured["tools"] = tools
            agent = MagicMock()
            agent.invoke = MagicMock(return_value={
                "messages": [],
                "initial_user_patience": 26,
                "updated_user_patience": 26,
                "tool_called_patience": [],
            })
            return agent

        monkeypatch.setattr(agent_code, "create_agent", fake_create_agent)

        model_agent, _, _ = _make_models()
        agent_code.run_agent_bird_baseline(
            _make_task(), model_agent, None, None, enable_ask_user=False,
        )

        tool_names = [getattr(t, "name", t.__name__) for t in captured["tools"]]
        assert "ask_user" not in tool_names
        assert "submit_sql" in tool_names
        assert "execute_sql" in tool_names
        assert len(captured["tools"]) == 8  # 7 db tools + submit_sql

    def test_enable_ask_user_true_includes_ask_user_tool(self, monkeypatch):
        from conversation2sql.eval_framework.agents.bird_baseline import agent_code

        captured = {}

        def fake_create_agent(model, tools, **kwargs):
            captured["tools"] = tools
            agent = MagicMock()
            agent.invoke = MagicMock(return_value={
                "messages": [],
                "initial_user_patience": 26,
                "updated_user_patience": 26,
                "tool_called_patience": [],
            })
            return agent

        monkeypatch.setattr(agent_code, "create_agent", fake_create_agent)

        model_agent, parser, gen = _make_models()
        agent_code.run_agent_bird_baseline(
            _make_task(), model_agent, parser, gen, enable_ask_user=True,
        )

        tool_names = [getattr(t, "name", t.__name__) for t in captured["tools"]]
        assert "ask_user" in tool_names
        assert len(captured["tools"]) == 9  # 7 db + submit_sql + ask_user

    def test_enable_ask_user_true_with_none_models_raises(self, monkeypatch):
        from conversation2sql.eval_framework.agents.bird_baseline import agent_code

        monkeypatch.setattr(agent_code, "create_agent", MagicMock())
        model_agent, _, _ = _make_models()

        with pytest.raises(AssertionError):
            agent_code.run_agent_bird_baseline(
                _make_task(), model_agent, None, None, enable_ask_user=True,
            )
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/eval_framework/agents/test_bird_baseline_agent.py -v
```

Expected: 3 FAILs (TypeError: unexpected `enable_ask_user`).

- [ ] **Step 3: Update `agent_code.py`**

Modify `run_agent_bird_baseline` in `src/conversation2sql/eval_framework/agents/bird_baseline/agent_code.py`:

```python
def run_agent_bird_baseline(
        single_task: TaskData,
        model_agent: BaseChatModel,
        model_user_parsing: BaseChatModel | None,
        model_user_generator: BaseChatModel | None,
        *,
        enable_ask_user: bool,
) -> CustomAgentState:
    messages = build_bird_interact_agent_messages(
        params={
            "total_budget": single_task.task_budget,
            "amb_user_query": single_task.task_question,
            "enable_ask_user": enable_ask_user,
        }
    )

    tools = [
        execute_sql,
        get_all_column_meanings,
        get_schema,
        get_column_meaning,
        get_all_external_knowledge_names,
        get_knowledge_definition,
        get_all_knowledge_definitions,
        submit_sql,
    ]
    if enable_ask_user:
        assert (
            model_user_parsing is not None and model_user_generator is not None
        ), "ask_user requires user-simulator models"
        tools.append(return_tool_ask_user(model_user_parsing, model_user_generator))

    agent = create_agent(
        model_agent,
        tools,
        state_schema=CustomAgentState,
        context_schema=TaskData,
        middleware=[  # pyrefly: ignore
            ModelRetryMiddleware(max_delay=60.0, on_failure="error"),
            ToolRetryMiddleware(max_delay=60.0, on_failure="error"),
            ModelCallLimitMiddleware(run_limit=single_task.task_budget + 5),
            ToolCallLimitMiddleware(
                run_limit=single_task.task_budget + 5,
                thread_limit=single_task.task_budget * 2,
            ),
            check_budget_limit,
            wrap_model_append_tool_message,
            tool_wrapper_patience_and_submit,
        ],
    )

    agent_state: CustomAgentState = {
        "messages": messages,  # pyrefly: ignore
        "initial_user_patience": single_task.task_budget,
        "updated_user_patience": single_task.task_budget,
        "tool_called_patience": list(),
    }

    response: CustomAgentState = agent.invoke(agent_state, context=single_task)  # pyrefly: ignore
    return utils_process_agent_response(response, tool_costs=TOOL_COSTS)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/eval_framework/agents/test_bird_baseline_agent.py -v
```

Expected: 3 PASS.

- [ ] **Step 5: Stage**

```bash
git add src/conversation2sql/eval_framework/agents/bird_baseline/agent_code.py \
        tests/eval_framework/agents/test_bird_baseline_agent.py
```

Commit point.

---

## Task 8: Add `baseline` enum to `ConfigPipeline`

**Files:**
- Modify: `src/conversation2sql/config_input.py`

- [ ] **Step 1: Add the field**

Update `ConfigPipeline`:

```python
from typing import Literal

from pydantic import BaseModel, Field


class ConfigPipeline(BaseModel):
    debug: bool = True
    mode: str = 'a-interact'  # a-interact | c-interact | oracle
    output_folder: str = "results"
    concurrency: int = Field(default=5, description="Number of parallel tasks to run")
    baseline: Literal['no_tool', 'tools_only', 'tools_user', 'bird_full'] = 'bird_full'
```

- [ ] **Step 2: Verify CLI flag picks it up**

```bash
uv run python -c "
from conversation2sql.cli_parser import PydanticParser
from conversation2sql.config_input import ConfigPipeline, ConfigReader, ConfigPredictor, ConfigUserSimulator
import sys
sys.argv = ['prog', '--baseline', 'tools_only']
parser = PydanticParser([ConfigPipeline, ConfigReader, ConfigPredictor, ConfigUserSimulator])
cp, *_ = parser.parse_args_and_config()
print(f'baseline={cp.baseline}')
assert cp.baseline == 'tools_only'
"
```

Expected: prints `baseline=tools_only`. (If `PydanticParser` auto-prefixes due to a name collision, use `--pipeline_baseline` instead and update the printed assertion command.)

- [ ] **Step 3: Stage**

```bash
git add src/conversation2sql/config_input.py
```

Commit point.

---

## Task 9: Add `_resolve_baseline_settings` helper to `main_pipe_workflow.py`

**Files:**
- Modify: `src/conversation2sql/eval_framework/main_pipe_workflow.py`

- [ ] **Step 1: Add the helper**

Near the top of `src/conversation2sql/eval_framework/main_pipe_workflow.py`, after the existing imports, add:

```python
from typing import Callable

from conversation2sql.eval_framework.agents import (
    run_agent_bird_baseline,
    run_baseline_no_tool,
)


def _resolve_baseline_settings(baseline: str) -> tuple[bool, Callable, bool]:
    """Map ConfigPipeline.baseline to (make_data_ambiguous, runner, needs_user_sim).

    no_tool    -> clean query, no agent loop, no user-sim
    tools_only -> clean query, agent without ask_user, no user-sim
    tools_user -> clean query, agent with ask_user, user-sim required
    bird_full  -> ambiguous query, agent with ask_user, user-sim required
    """
    table = {
        "no_tool":    (False, run_baseline_no_tool, False),
        "tools_only": (False, run_agent_bird_baseline, False),
        "tools_user": (False, run_agent_bird_baseline, True),
        "bird_full":  (True,  run_agent_bird_baseline, True),
    }
    if baseline not in table:
        raise ValueError(f"Unknown baseline: {baseline!r}; expected one of {list(table)}")
    return table[baseline]
```

(Note: `run_baseline_no_tool` will be exported in Task 11. This import will fail until then — that's intentional; Task 11 fixes it.)

- [ ] **Step 2: Stage (no test yet — wired into pipeline in Task 12)**

```bash
git add src/conversation2sql/eval_framework/main_pipe_workflow.py
```

(Don't run pipeline yet — broken import. Continue to Task 10.)

---

## Task 10: Make `_init_models` conditional on `needs_user_sim`

**Files:**
- Modify: `src/conversation2sql/eval_framework/main_pipe_workflow.py:80-101`

- [ ] **Step 1: Update the function**

Replace the `_init_models` function:

```python
def _init_models(
        config_predictor: ConfigPredictor,
        config_user: ConfigUserSimulator,
        needs_user_sim: bool,
) -> tuple:
    model_agent = utils_create_model(
        model_name=config_predictor.model_name,
        model_provider=config_predictor.model_provider,
        temperature=config_predictor.temperature,
        top_p=config_predictor.top_p,
        max_tokens=config_predictor.max_new_tokens,
    )
    if not needs_user_sim:
        return model_agent, (None, None)

    model_user_parsing = utils_create_model(
        model_name=config_user.model_name,
        model_provider=config_user.model_provider,
        temperature=config_user.temperature,
        max_tokens=config_user.max_new_tokens,
    )
    model_user_generator = utils_create_model(
        model_name=config_user.model_name,
        model_provider=config_user.model_provider,
        temperature=config_user.temperature,
        max_tokens=config_user.max_new_tokens,
    )
    return model_agent, (model_user_parsing, model_user_generator)
```

- [ ] **Step 2: Stage**

```bash
git add src/conversation2sql/eval_framework/main_pipe_workflow.py
```

(Pipeline still partially broken — Task 11 wires the dispatch.)

---

## Task 11: Re-export `run_baseline_no_tool` from `agents/__init__.py`

**Files:**
- Modify: `src/conversation2sql/eval_framework/agents/__init__.py`

- [ ] **Step 1: Update the export**

Replace the file contents:

```python
from conversation2sql.eval_framework.agents.bird_baseline.agent_code import run_agent_bird_baseline
from conversation2sql.eval_framework.agents.no_tool_baseline.baseline_model import run_baseline_no_tool

__all__ = [
    "run_agent_bird_baseline",
    "run_baseline_no_tool",
]
```

- [ ] **Step 2: Verify the import chain works**

```bash
uv run python -c "
from conversation2sql.eval_framework.main_pipe_workflow import _resolve_baseline_settings
print(_resolve_baseline_settings('tools_only'))
print(_resolve_baseline_settings('bird_full'))
print(_resolve_baseline_settings('no_tool'))
"
```

Expected: three tuples printed, each with `(make_data_ambiguous, callable, needs_user_sim)`.

- [ ] **Step 3: Stage**

```bash
git add src/conversation2sql/eval_framework/agents/__init__.py
```

---

## Task 12: Wire dispatch + per-baseline output path in `workflow_evaluation_pipeline`

**Files:**
- Modify: `src/conversation2sql/eval_framework/main_pipe_workflow.py`
- Create: `tests/eval_framework/test_main_pipe_workflow.py`

- [ ] **Step 1: Write failing dispatch tests**

Create `tests/eval_framework/test_main_pipe_workflow.py`:

```python
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from conversation2sql.eval_framework.main_pipe_workflow import (
    _resolve_baseline_settings,
    workflow_evaluation_pipeline,
)
from conversation2sql.config_input import (
    ConfigPipeline, ConfigReader, ConfigPredictor, ConfigUserSimulator,
)


class TestResolveBaselineSettings:
    @pytest.mark.parametrize("baseline,expected_amb,expected_user_sim", [
        ("no_tool", False, False),
        ("tools_only", False, False),
        ("tools_user", False, True),
        ("bird_full", True, True),
    ])
    def test_resolves(self, baseline, expected_amb, expected_user_sim):
        amb, runner, needs = _resolve_baseline_settings(baseline)
        assert amb is expected_amb
        assert needs is expected_user_sim
        assert callable(runner)

    def test_unknown_baseline_raises(self):
        with pytest.raises(ValueError):
            _resolve_baseline_settings("nonsense")


@pytest.fixture
def configs(tmp_path):
    return (
        ConfigPipeline(debug=True, output_folder=str(tmp_path)),
        ConfigReader(make_data_ambiguous=True),  # intentionally inconsistent w/ tools_only
        ConfigPredictor(),
        ConfigUserSimulator(),
    )


def _fake_task():
    t = MagicMock()
    t.instance_id = "task_1"
    t.model_dump.return_value = {"instance_id": "task_1"}
    return t


@patch("conversation2sql.eval_framework.main_pipe_workflow.run_agent_bird_baseline")
@patch("conversation2sql.eval_framework.main_pipe_workflow.run_baseline_no_tool")
@patch("conversation2sql.eval_framework.main_pipe_workflow.load_bird_interact_as_tasks")
@patch("conversation2sql.eval_framework.main_pipe_workflow.utils_create_model")
class TestDispatch:
    def _stub_response(self):
        return {
            "messages": [], "initial_user_patience": 0, "updated_user_patience": 0,
            "tool_called_patience": [], "total_cost": 0, "total_tokens": 0,
            "total_prompt_tokens": 0, "total_completion_tokens": 0,
            "mean_prompt_tokens": 0, "mean_completion_tokens": 0,
            "tool_calls_in_order": [], "execution_accuracy": False,
        }

    def test_no_tool_dispatches_to_run_baseline_no_tool(
        self, mock_create, mock_load, mock_no_tool, mock_agent, configs,
    ):
        cp, cr, cpred, cu = configs
        cp.baseline = "no_tool"
        mock_load.return_value = [_fake_task()]
        mock_no_tool.return_value = self._stub_response()
        mock_create.return_value = MagicMock()

        workflow_evaluation_pipeline(cp, cr, cpred, cu)

        mock_no_tool.assert_called_once()
        mock_agent.assert_not_called()
        # make_data_ambiguous should be overridden to False for no_tool
        assert mock_load.call_args.kwargs["make_data_ambiguous"] is False

    def test_tools_user_dispatches_to_agent_with_ask_user_true(
        self, mock_create, mock_load, mock_no_tool, mock_agent, configs,
    ):
        cp, cr, cpred, cu = configs
        cp.baseline = "tools_user"
        mock_load.return_value = [_fake_task()]
        mock_agent.return_value = self._stub_response()
        mock_create.return_value = MagicMock()

        workflow_evaluation_pipeline(cp, cr, cpred, cu)

        mock_agent.assert_called_once()
        assert mock_agent.call_args.kwargs["enable_ask_user"] is True
        # user-sim models constructed: agent + parser + generator = 3 calls
        assert mock_create.call_count == 3

    def test_tools_only_skips_user_sim_construction(
        self, mock_create, mock_load, mock_no_tool, mock_agent, configs,
    ):
        cp, cr, cpred, cu = configs
        cp.baseline = "tools_only"
        mock_load.return_value = [_fake_task()]
        mock_agent.return_value = self._stub_response()
        mock_create.return_value = MagicMock()

        workflow_evaluation_pipeline(cp, cr, cpred, cu)

        assert mock_agent.call_args.kwargs["enable_ask_user"] is False
        # only the agent model is built
        assert mock_create.call_count == 1

    def test_bird_full_keeps_make_data_ambiguous_true(
        self, mock_create, mock_load, mock_no_tool, mock_agent, configs,
    ):
        cp, cr, cpred, cu = configs
        cp.baseline = "bird_full"
        cr.make_data_ambiguous = True
        mock_load.return_value = [_fake_task()]
        mock_agent.return_value = self._stub_response()
        mock_create.return_value = MagicMock()

        workflow_evaluation_pipeline(cp, cr, cpred, cu)

        assert mock_load.call_args.kwargs["make_data_ambiguous"] is True

    def test_output_path_includes_baseline(
        self, mock_create, mock_load, mock_no_tool, mock_agent, configs, tmp_path,
    ):
        cp, cr, cpred, cu = configs
        cp.baseline = "tools_only"
        mock_load.return_value = [_fake_task()]
        mock_agent.return_value = self._stub_response()
        mock_create.return_value = MagicMock()

        workflow_evaluation_pipeline(cp, cr, cpred, cu)

        # one results.jsonl somewhere under <tmp>/tools_only/
        matches = list(Path(tmp_path).glob("tools_only/**/results.jsonl"))
        assert len(matches) == 1
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
uv run pytest tests/eval_framework/test_main_pipe_workflow.py -v
```

Expected: tests fail because `workflow_evaluation_pipeline` doesn't yet do dispatch / output-path remap.

- [ ] **Step 3: Rewrite `workflow_evaluation_pipeline`**

Replace `workflow_evaluation_pipeline` body in `src/conversation2sql/eval_framework/main_pipe_workflow.py`:

```python
def workflow_evaluation_pipeline(
        config_pipeline: ConfigPipeline,
        config_reader: ConfigReader,
        config_predictor: ConfigPredictor,
        config_user: ConfigUserSimulator,
) -> list[dict]:
    logger.info(f"config_pipeline: {config_pipeline}")
    logger.info(f"config_reader: {config_reader}")
    logger.info(f"config_predictor: {config_predictor}")
    logger.info(f"config_user: {config_user}")

    # Resolve baseline settings and apply to reader config
    forced_amb, runner, needs_user_sim = _resolve_baseline_settings(config_pipeline.baseline)
    if config_reader.make_data_ambiguous != forced_amb:
        logger.warning(
            "baseline=%s forces make_data_ambiguous=%s; overriding ConfigReader.make_data_ambiguous=%s",
            config_pipeline.baseline, forced_amb, config_reader.make_data_ambiguous,
        )
    config_reader = config_reader.model_copy(update={"make_data_ambiguous": forced_amb})

    # Per-baseline output folder
    output_folder = Path(config_pipeline.output_folder) / config_pipeline.baseline
    config_pipeline = config_pipeline.model_copy(update={"output_folder": str(output_folder)})

    # save config in (per-baseline) output folder
    _save_configs_as_yaml(
        output_folder=output_folder,
        config_pipeline=config_pipeline,
        config_reader=config_reader,
        config_predictor=config_predictor,
        config_user=config_user,
    )

    file_result = output_folder / 'results.jsonl'

    model_agent, (model_user_parsing, model_user_generator) = _init_models(
        config_predictor, config_user, needs_user_sim=needs_user_sim,
    )

    dataset: list[TaskData] = load_bird_interact_as_tasks(**config_reader.model_dump())

    i = -1
    result = []
    try:
        for i, task in tqdm.tqdm(enumerate(dataset), desc="Processing dataset"):
            if config_pipeline.baseline == "no_tool":
                response = runner(task, model_agent)
            else:
                response = runner(
                    task, model_agent, model_user_parsing, model_user_generator,
                    enable_ask_user=(config_pipeline.baseline in ("tools_user", "bird_full")),
                )
            task_output = {
                "config_predictor": config_predictor.model_dump(),
                "config_user": config_user.model_dump(),
                "config_pipeline": config_pipeline.model_dump(),
                "config_reader": config_reader.model_dump(),
                **task.model_dump(),
                **response,
            }
            _save_record(response=task_output, output_path_jsonl=file_result)
            logger.info(f"Saved response for task_id={task.instance_id} to {file_result}")
            result.append(task_output)

            if config_pipeline.debug:
                break

    except Exception as e:
        logger.error(f"Error occurred: {e}")
        response_error = {'error': str(e), 'last_processed_task': i}
        output = file_result.parent / f"{file_result.stem}_error.jsonl"
        _save_record(response=response_error, output_path_jsonl=output)
        logger.info(f"Saved ERROR to {output}")
        raise e

    return result
```

- [ ] **Step 4: Run dispatch tests to verify they pass**

```bash
uv run pytest tests/eval_framework/test_main_pipe_workflow.py -v
```

Expected: all PASS.

- [ ] **Step 5: Run the full unit-test suite to confirm no regressions**

```bash
uv run pytest tests/ -v --ignore=tests/eval_framework/integration
```

Expected: all PASS.

- [ ] **Step 6: Type-check**

```bash
uv run pyrefly check
```

Expected: no new errors.

- [ ] **Step 7: Stage**

```bash
git add src/conversation2sql/eval_framework/main_pipe_workflow.py \
        tests/eval_framework/test_main_pipe_workflow.py
```

Commit point.

---

## Task 13: Update CLAUDE.md files for renames and new behavior

**Files:**
- Modify: `src/conversation2sql/eval_framework/agents/CLAUDE.md`
- Modify: `src/conversation2sql/eval_framework/agents/no_tool_baseline/CLAUDE.md`
- Modify: `src/conversation2sql/eval_framework/agents/bird_baseline/CLAUDE.md`

- [ ] **Step 1: Update `agents/CLAUDE.md`**

Replace the variants table:

```markdown
## Variants

`ConfigPipeline.baseline` selects one of four ablation cells. Two runner functions cover them:

| `baseline` enum | Runner | DB tools | `ask_user` | Query |
|-----------------|--------|----------|-----------|-------|
| `no_tool`       | `run_baseline_no_tool` | — | — | clean |
| `tools_only`    | `run_agent_bird_baseline(enable_ask_user=False)` | ✅ | — | clean |
| `tools_user`    | `run_agent_bird_baseline(enable_ask_user=True)`  | ✅ | ✅ | clean |
| `bird_full`     | `run_agent_bird_baseline(enable_ask_user=True)`  | ✅ | ✅ | ambiguous |

`bird_baseline/agent_code.py` is the parameterizable agent — `enable_ask_user` toggles the `ask_user` tool and the corresponding Jinja blocks in `prompts.py`. The middleware stack is identical across the three agentic variants.
```

- [ ] **Step 2: Update `no_tool_baseline/CLAUDE.md`**

Replace the file:

```markdown
# no_tool_baseline

Single-shot text-to-SQL baseline — no tools, no user interaction.

## Entry function

`run_baseline_no_tool(single_task, model_agent)` in `baseline_model.py`:
1. Renders the OmniSQL-style prompt via `build_omnisql_prompt` (Jinja2, schema + question).
2. Calls `model_agent.invoke(messages)` once.
3. Extracts SQL from the model's fenced code block via `extract_sql_from_response`.
4. If SQL was extracted, calls `submit_sql_impl(sql, single_task)` against Postgres for `passed`.
5. Returns a result dict whose top-level keys mirror `run_agent_bird_baseline` (a synthetic
   `submit_sql_offline` ToolMessage stands in for the missing real submit).

## When to use

Lower-bound baseline for the ablation. No patience-budget mechanics; no agent loop.

## Gotchas

- `extract_sql_from_response` returns the **last** fenced block (handles draft-then-refine).
- When no fenced block is present, `execution_accuracy=False` is recorded with
  `error="no_sql_block_found"` — the task is not retried.
```

- [ ] **Step 3: Update `bird_baseline/CLAUDE.md`**

Add to the "Key files" section, right after the `agent_code.py` line:

```markdown
- `agent_code.py` exposes `enable_ask_user: bool` (kwarg). When `False` the `ask_user` tool is omitted and the system prompt renders without ambiguity-related guidance. The middleware stack is unchanged across both modes.
```

- [ ] **Step 4: Stage**

```bash
git add src/conversation2sql/eval_framework/agents/CLAUDE.md \
        src/conversation2sql/eval_framework/agents/no_tool_baseline/CLAUDE.md \
        src/conversation2sql/eval_framework/agents/bird_baseline/CLAUDE.md
```

Commit point.

---

## Task 14: Add gated integration smoke test

**Files:**
- Create: `tests/eval_framework/integration/__init__.py`
- Create: `tests/eval_framework/integration/test_baselines_smoke.py`

- [ ] **Step 1: Create test package init**

```bash
touch tests/eval_framework/integration/__init__.py
```

- [ ] **Step 2: Write the smoke test**

Create `tests/eval_framework/integration/test_baselines_smoke.py`:

```python
"""End-to-end smoke tests for the four baselines.

Gated: run with `uv run pytest tests/eval_framework/integration/ -m integration`.
Requires:
- Postgres containers from `.devcontainer/docker-compose.yml` running
- API keys for the configured provider in `.env`
- The bird-interact dataset on disk
"""
import pytest

from conversation2sql.config_input import (
    ConfigPipeline, ConfigReader, ConfigPredictor, ConfigUserSimulator,
)
from conversation2sql.eval_framework.main_pipe_workflow import (
    workflow_evaluation_pipeline,
)


REQUIRED_KEYS = {
    "execution_accuracy", "total_cost", "total_tokens", "tool_calls_in_order",
    "messages", "config_pipeline",
}


@pytest.mark.integration
@pytest.mark.parametrize("baseline", ["no_tool", "tools_only", "tools_user", "bird_full"])
def test_baseline_runs_one_task_and_emits_expected_shape(baseline, tmp_path):
    cp = ConfigPipeline(debug=True, baseline=baseline, output_folder=str(tmp_path))
    cr = ConfigReader()  # _resolve_baseline_settings overrides make_data_ambiguous
    cpred = ConfigPredictor()
    cu = ConfigUserSimulator()

    results = workflow_evaluation_pipeline(cp, cr, cpred, cu)

    assert len(results) == 1
    record = results[0]
    missing = REQUIRED_KEYS - record.keys()
    assert not missing, f"Missing keys for {baseline}: {missing}"
    assert isinstance(record["execution_accuracy"], bool)
```

- [ ] **Step 3: Verify the marker is registered (or register it)**

```bash
grep -n "integration" pyproject.toml || echo "MARKER_NOT_REGISTERED"
```

If `MARKER_NOT_REGISTERED`, add to `[tool.pytest.ini_options]` in `pyproject.toml`:

```toml
markers = [
    "integration: requires live Postgres + provider API keys",
]
```

- [ ] **Step 4: Verify the test is collected but skipped by default**

```bash
uv run pytest tests/eval_framework/integration/ --collect-only -q
```

Expected: 4 tests collected (one per `baseline`), all marked `integration`.

```bash
uv run pytest tests/eval_framework/integration/ -v
```

Expected: 4 tests deselected (default config skips `integration` mark) — confirms the gate works.

- [ ] **Step 5: Stage**

```bash
git add tests/eval_framework/integration/__init__.py \
        tests/eval_framework/integration/test_baselines_smoke.py \
        pyproject.toml
```

Commit point.

---

## Task 15: Final verification

- [ ] **Step 1: Full unit test suite**

```bash
uv run pytest tests/ -v --ignore=tests/eval_framework/integration
```

Expected: all PASS, no skipped tests other than ones already skipped before this work.

- [ ] **Step 2: Type-check**

```bash
uv run pyrefly check
```

Expected: no new errors. (Pre-existing errors are out of scope for this plan.)

- [ ] **Step 3: Manual smoke run of one baseline**

If the Postgres container is reachable:

```bash
uv run python main.py \
    --config configs/eval_pipeline_config.yaml \
    --pipeline_baseline tools_only \
    --pipeline_debug true
```

Expected: one JSONL line under `<output_folder>/tools_only/<date>/<time>/results.jsonl` with `execution_accuracy` set, no errors.

(If `--pipeline_baseline` is not the correct flag name due to `PydanticParser`'s prefixing rules — `baseline` is a unique field, so the bare `--baseline` should work; try whichever the parser accepts.)

- [ ] **Step 4: Hand off to user for commit**

Print a summary of staged changes:

```bash
git status
git diff --cached --stat
```

Then ask the user to commit when ready. **Do not run `git commit`.**

---

## Notes for the implementer

- **Commit policy:** This repo's user commits manually. Do not run `git commit`. Always stop after `git add`.
- **Test discovery:** `pytest.ini_options` sets `testpaths = ["tests"]` and `asyncio_mode = "auto"`. New test files must live under `tests/`.
- **Always `uv run`:** Use `uv run python` and `uv run pytest`. The base image `verlai/verl:vllm017.latest` ships system packages, but project deps are pinned in `.venv`.
- **Postgres caveat:** `submit_sql_impl` in Task 3 hits a live DB. Tests in Tasks 2/3 mock it. The integration test in Task 14 hits a live DB.
- **Reading order:** Tasks 9-12 are tightly coupled and intentionally land "broken-then-fixed" within a single staging window. Implementer can split commits at task boundaries 1-8, then commit Tasks 9-12 as a single change if the user prefers cleaner history.
