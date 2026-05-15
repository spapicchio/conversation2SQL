# Turn-Level Conversation Classifier — Design Spec

**Date:** 2026-05-07
**Branch:** feat/turn-classifier (from feat/baselines-ablation)
**Goal:** Classify every agent turn in BIRD-Interact result traces at two levels for paper-quality analysis: a semantic Level 1 (comparable to open-domain QA literature) and a tool-action Level 2 (text-to-SQL specific).

---

## 1. Problem

`results_smaller.jsonl` files produced by `workflow_evaluation_pipeline` contain a `messages` list per record. Each `ai` message represents one agent turn. No taxonomy currently exists to describe *what the agent was doing* on that turn — clarifying, exploring the DB, assuming an interpretation, revising a failed SQL, etc.

The classifier fills this gap so we can:
- Report per-baseline strategy distributions (% turns spent on clarification vs. DB exploration vs. self-repair)
- Correlate turn-type patterns with `execution_accuracy`
- Produce qualitative examples for paper sections

---

## 2. Taxonomy

### Level 1 — Semantic (LLM judge, 10 labels)

Adapted from the QA dialogue act taxonomy (Rahmani et al.) with three additions for disambiguation and agentic SQL solving.

| Label | Description |
|---|---|
| `ANSWER_ATTEMPT` | Agent commits to an SQL answer — first submission with no prior failed context |
| `REVISION` | Agent identifies a previous failed/wrong answer and self-corrects autonomously without asking the user |
| `CLARIFICATION` | Single focused question to the user about one ambiguity |
| `INTERROGATION` | Multiple questions to the user in one turn |
| `ASSUMPTION` | Agent explicitly resolves an ambiguity by making an interpretive choice without consulting the user |
| `CONFIRMATION` | Agent grounds the user's previous answer — echoes back its understanding before acting |
| `DISCUSSION` | Explores or reasons about the problem without submitting, asking, or refusing |
| `HEDGING` | Presents multiple conditional SQL candidates based on different interpretations |
| `REFUSAL` | Declines to proceed without a follow-up question or request |
| `MISSING` | Empty turn — no thinking, no text, no tool calls |

**Key distinctions for disambiguation context:**
- `ASSUMPTION` vs `CLARIFICATION`: assumption = agent decides unilaterally; clarification = agent asks the user
- `REVISION` vs `ANSWER_ATTEMPT`: revision requires evidence of a prior failed attempt in message history
- `CONFIRMATION` vs `DISCUSSION`: confirmation explicitly references user's prior answer and signals readiness to act

### Level 2 — Tool Action (rule-based, configurable)

Determined deterministically from the `tool_calls` list in each AI message. Tool→category mapping lives in `configs/turn_classifier/tool_categories.yaml` — no tool names are hardcoded in library code.

| Category | Rule |
|---|---|
| `SQL_SUBMISSION` | any `submit_sql` call present |
| `USER_INTERACTION` | any `ask_user` call present (no `submit_sql`) |
| `DB_EXPLORATION` | only DB-type tools (e.g. `execute_sql`, `get_schema`, `get_column_meaning`) |
| `KNOWLEDGE_LOOKUP` | only KB-type tools (e.g. `get_all_external_knowledge_names`, `get_knowledge_definition`) |
| `MIXED` | tool calls span more than one category |
| `TEXT_ONLY` | no tool calls, non-empty content |
| `NO_ACTION` | no tool calls, empty content |
| `UNKNOWN` | tool present but not in the YAML config |

Priority order when multiple rules match: `SQL_SUBMISSION` > `USER_INTERACTION` > specific category > `MIXED`.

---

## 3. Module Layout

```
src/conversation2sql/eval_framework/turn_classifier/
  __init__.py
  classifier.py      # TurnClassifier: orchestrates Level 2 + Level 1
  schemas.py         # Pydantic: TurnClassification, TurnClassifierConfig
  rules.py           # deterministic Level 2 from tool_calls + YAML
  prompts.py         # Jinja2 template strings for LLM judge

configs/turn_classifier/
  tool_categories.yaml   # tool_name → Level2Category

scripts/
  classify_turns.py      # CLI entrypoint
```

Follows the existing `agents/no_tool_baseline/` pattern: schemas → prompts → model logic → script.

---

## 4. Data Flow

### Input

Any `results_smaller.jsonl` produced by the evaluation pipeline. The file may contain one record (debug mode, pretty-printed) or many records (one compact JSON line each). The script handles both formats via `json.JSONDecoder.raw_decode`.

### Per-record processing

```
record
  └─ messages → filter role == 'ai' → ai_turns list
       for each ai_turn:
         ├─ Level 2 (rules.py, no LLM)
         │    tool_calls → YAML lookup → category string
         │
         └─ Level 1 (classifier.py, LLM judge)
              context = {
                thinking: str | None,      # content items with type='thinking'
                text: str | None,          # content items with type='text'
                tool_calls_summary: str,   # tool names + key args, truncated to 300 chars
                level2_category: str,      # from above
                prior_failed_submit: bool, # True if any previous submit_sql returned passed=False
              }
              model.with_structured_output(TurnClassification).invoke(prompt)
              → TurnClassification

→ record['turn_classifications'] = list[TurnClassification]
→ append record to output JSONL
```

### Output record addition

```json
"turn_classifications": [
  {
    "message_index": 14,
    "level2_category": "USER_INTERACTION",
    "level2_tools_called": ["ask_user"],
    "reasoning": "The agent presents four numbered sub-questions covering two independent ambiguities ...",
    "level1_category": "INTERROGATION",
    "level1_alternatives": [],
    "confidence": "CERTAIN"
  },
  {
    "message_index": 22,
    "level2_category": "SQL_SUBMISSION",
    "level2_tools_called": ["submit_sql"],
    "reasoning": "Agent submits SQL after a prior failed attempt, but the thinking shows no explicit acknowledgment of the failure ...",
    "level1_category": "REVISION",
    "level1_alternatives": ["ANSWER_ATTEMPT"],
    "confidence": "PLAUSIBLE"
  },
  ...
]
```

---

## 5. LLM Judge Design

### Strategy: Single-pass CoT with structured output

- **Model:** configurable via `--model` flag (default: `openai/gpt-4o-mini` via LiteLLM). Same `utils_create_model` factory used by the rest of the codebase.
- **Structured output:** `model.with_structured_output(Level1Classification)` — LangChain enforces JSON schema, no regex parsing.
- **CoT:** the Pydantic schema places `reasoning: str` before `level1_category: str`, so the model writes its rationale before committing to a label.
- **Cross-turn context:** `prior_failed_submit` flag is injected so the model can detect `REVISION` reliably without re-reading the full history.
- **Prompt template:** Jinja2, in `prompts.py`. Includes all 10 category definitions, confidence level definitions, and the instruction: *"When confidence is PLAUSIBLE or UNCERTAIN, list every other label that could reasonably apply in `level1_alternatives`. When confidence is CERTAIN or CONFIDENT, leave `level1_alternatives` empty."*

### Confidence scale (discrete, 4 levels — evidence-anchored)

| Value | Meaning |
|---|---|
| `CERTAIN` | Structurally determined — tool calls alone make the label unambiguous (e.g. `submit_sql` → ANSWER_ATTEMPT) |
| `CONFIDENT` | Text/thinking strongly supports one label; a second interpretation exists but is clearly weaker |
| `PLAUSIBLE` | Two labels are in competition; the chosen one is the best fit but the other is genuinely defensible — **must populate `level1_alternatives`** |
| `UNCERTAIN` | Forced choice — turn is mixed or ambiguous enough that the label could easily be wrong — **must populate `level1_alternatives`** |

### Pydantic schemas

Two schemas keep responsibilities clean:

```python
class Level1Classification(BaseModel):
    """What the LLM judge returns — only semantic fields."""
    reasoning: str                  # CoT — written before label
    level1_category: str            # primary (best-fit) label, one of 10
    level1_alternatives: list[str]  # competing labels; non-empty only when PLAUSIBLE or UNCERTAIN
    confidence: Literal["CERTAIN", "CONFIDENT", "PLAUSIBLE", "UNCERTAIN"]

class TurnClassification(BaseModel):
    """Final assembled output per turn (rules + LLM merged by classifier.py)."""
    message_index: int
    level2_category: str            # from rules.py
    level2_tools_called: list[str]  # from rules.py
    reasoning: str                  # from LLM
    level1_category: str            # from LLM — primary label
    level1_alternatives: list[str]  # from LLM — competing labels when PLAUSIBLE/UNCERTAIN
    confidence: Literal["CERTAIN", "CONFIDENT", "PLAUSIBLE", "UNCERTAIN"]  # from LLM
```

`model.with_structured_output(Level1Classification)` is called; `classifier.py` then assembles `TurnClassification` by merging the Level 2 fields from `rules.py` with the Level 1 fields from the LLM.

---

## 6. CLI Interface

```bash
uv run python scripts/classify_turns.py \
  results/2026-05-01/10-28-14/main/results_smaller.jsonl \
  [results/other_run/results_smaller.jsonl ...] \
  --output results/classified.jsonl \
  --model openai/gpt-4o-mini \
  --tool-categories configs/turn_classifier/tool_categories.yaml
```

- Accepts one or more input paths (glob-friendly via shell expansion)
- `--output` defaults to `<input_dir>/results_classified.jsonl` when a single input is given
- Processes records sequentially; logs progress with `tqdm`
- On error for a single record: logs and skips, continues to next record

---

## 7. Extensibility

- **New tool:** add one line to `tool_categories.yaml`, no code change
- **New Level 2 category:** add the string to the YAML values; `rules.py` uses the values as-is
- **New Level 1 label:** add to `schemas.py` and update the prompt template in `prompts.py` (both category definition and disambiguation notes)
- **Confidence scale:** fixed at 4 values; changing it requires updating the `Literal` in `schemas.py` and the prompt
- **Self-consistency:** wrap `TurnClassifier.classify_turn` with K-run majority vote by passing `n_votes > 1` — stub in `classifier.py`, not implemented by default

---

## 8. Testing

- Unit test `rules.py` with fixture tool_calls lists covering each Level 2 case
- Integration test: run classifier over the existing `results/2026-05-01/10-28-14/main/results_smaller.jsonl` fixture and assert output schema is valid (no LLM call needed if `rules.py` is tested independently; LLM tests marked `@pytest.mark.slow`)
