# BIRD-Interact Dataset Reader

## Overview

[BIRD-Interact](https://bird-interact.github.io) (ICLR 2026 Oral, [paper](https://arxiv.org/abs/2510.05318)) is an interactive text-to-SQL benchmark built on top of [LiveSQLBench](https://livesqlbench.ai). Unlike traditional single-turn Text-to-SQL benchmarks, BIRD-Interact evaluates models through **dynamic multi-turn interactions** where the model must resolve ambiguities in user queries by asking clarifying questions or using available tools.

The reader is implemented in [`bird_interact_reader.py`](bird_interact_reader.py) as `BirdInteractReader`. It loads the `birdsql/bird-interact-lite` (or `full`) dataset from HuggingFace and merges it with a local ground-truth JSONL file that provides additional fields not publicly released (SQL solutions, test cases, external knowledge).

### Key numbers

| Version | Tasks | DB Engine |
|---------|-------|-----------|
| Lite    | 300   | PostgreSQL |
| Full    | 600   | PostgreSQL |

---

## Dataset Fields

### Fields from HuggingFace

These fields come directly from the public `birdsql/bird-interact-lite` HuggingFace dataset.

| Field | Type | Description |
|-------|------|-------------|
| `instance_id` | `str` | Unique identifier for the task instance. Used as `sample_id` in the pipeline. |
| `selected_database` | `str` | Name of the PostgreSQL database this task operates on (e.g. `"alien"`). |
| `query` | `str` | The **unambiguous** ground-truth user query from LiveSQLBench. Not shown to the agent. |
| `amb_user_query` | `str` | The **ambiguous** user query shown to the agent — the original `query` with injected ambiguities. |
| `user_query_ambiguity` | `list[dict]` | JSON listing the ambiguities injected into `amb_user_query`. Each entry describes one ambiguity point. Provided to the user simulator to respond to clarifications. |
| `non_critical_ambiguity` | `list[dict]` | Subset of ambiguities that are non-critical (e.g. ordering, limit). These do not affect correctness. |
| `knowledge_ambiguity` | `list[dict]` | Subset of ambiguities created by masking external knowledge (e.g. domain-specific formulas or terminology). Requires `external_knowledge` to resolve. |
| `follow_up` | `list[dict]` | Labeled follow-up questions for the task. Not used in the current evaluation pipeline (only the initial `amb_user_query` is evaluated). |
| `preprocess_sql` | `list[str]` | SQL statements executed **before** running the predicted SQL. Used to set up the database state (e.g. INSERT rows). |
| `clean_up_sqls` | `list[str]` | SQL statements executed **after** the test cases to revert any database changes (e.g. DELETE inserted rows). Note: the HF dataset field is named `clean_up_sql` (singular); internally the reader stores this as `clean_up_sqls`. |

### Fields from the Ground-Truth JSONL

These fields are merged in from the local GT file (`bird_interact_lite_gt_kg_testcases_1008.jsonl` / `bird_interact_full_gt_kg_testcases_1008.jsonl`). The GT file is **not publicly released** — it must be requested from the BIRD team.

| Field | Type | Description |
|-------|------|-------------|
| `sol_sql` | `str` | The ground-truth SQL query. The reader takes `sol_sql[0]` (first element) from the GT file, as it is stored as a list. Used as `target` in the pipeline. |
| `test_cases` | `list[list[str]]` | Executable test cases to validate the predicted SQL. Each inner list is one test case. The HF dataset schema for this field is cast to `Sequence(Sequence(Value("string")))` to match the GT format. |
| `external_knowledge` | `list[dict]` | Domain-specific knowledge entries needed to resolve `knowledge_ambiguity`. Each entry has fields: `knowledge` (name), `description`, `definition`, `type`, and `children_knowledge`. |

---

## Processing Pipeline

### 1. Loading and Merging (`read`)

```
HuggingFace dataset["dev"]
        ↓
Cast test_cases schema → Sequence(Sequence(Value("string")))
        ↓
for line in dataset.to_list()   ← sequential, merges GT fields per instance_id
        ↓
list[Sample]
```

The `read()` method:
1. Loads the HuggingFace dataset (split `"dev"`).
2. Casts the `test_cases` feature to `Sequence(Sequence(Value("string")))` to match the nested list format in the GT file.
3. Iterates over each row to build prompt inputs and merge GT fields.
4. Constructs a `ToolUserContext` inline for each sample to configure the user simulator.
5. Returns `list[Sample]` objects.

**Note:** Follow-up questions (`follow_up`) are intentionally ignored — only the initial ambiguous query is evaluated.

### 2. Per-Sample Processing (inline in `read`)

For each sample, the following fields are resolved:

| Field | Source | Description |
|-------|--------|-------------|
| `target` | HF line `sol_sql` | The ground-truth SQL query. |
| `test_cases` | GT JSONL (`instance2sol`) | Executable test cases; taken from the GT file. |
| `external_knowledge` | HF line `kb_database` | KB entries referenced by the GT `external_knowledge` ID list. |
| `user_patience` | Config (`user_patience`) | Number of interaction turns the user simulator will tolerate. Injected as `total_budget` in the agent prompt. |
| `messages` | `_build_messages()` | The formatted input messages (system + user) for the agent. |
| `user_context` | inline `ToolUserContext(...)` | A `ToolUserContext` object configuring the user simulator. Set after Sample construction. |

### 3. Agent Prompt (`_build_messages`)

Renders the agent's initial messages using Jinja2 templates from `prompts/bird_interact_a_agent/`:

| Template variable | Value |
|-------------------|-------|
| `database_engine` | From config (e.g. `"postgresql"`) |
| `user_query` | `amb_user_query` — the ambiguous query shown to the agent |
| `total_budget` | `user_patience` — max interaction turns |

If `is_chat_template=True` (default for instruction-tuned models), the output is a `list[BaseMessage]` with `system` and `user` roles. Otherwise it is a single concatenated string for completion models.

### 4. User Simulator Context (`_build_user_context`)

Builds a `UserContext` object that configures the LLM-based user simulator. The simulator is invoked by the `ask_user` tool during agentic interaction.

The simulator has access to:

| Context | Description |
|---------|-------------|
| `db_schema` | Name of the database (`selected_database`) |
| `user_query` | The ambiguous query (`amb_user_query`) |
| `ambiguities_json` | The full `user_query_ambiguity` list |
| `correct_sql` | The ground-truth `sol_sql` (so it can verify agent clarifications) |
| `external_knowledge` | Domain knowledge entries for resolving `knowledge_ambiguity` |
| `db_dsn` | PostgreSQL DSN for the specific database, built from `db_dsn_template` in config |

Simulator prompts live in `prompts/bird_interact_user_simulator/`:
- `simulator_base.jinja` — base system prompt
- `step_1_llm_parser.jinja` — parses the agent's question
- `step_2_llm_generator.jinja` — generates the user's response

---

## Sample Object Structure

Each sample produced by `read()` is a `Sample` with:

```python
Sample(
    sample_id=instance_id,        # e.g. "bird_interact_lite_0042"
    messages=[                    # agent's initial prompt
        BaseMessage(role="system", content="..."),
        BaseMessage(role="user", content="..."),
    ],
    target=sol_sql,               # ground-truth SQL string
    user_context=UserContext(...),# user simulator config
    metadata={
        "selected_database": ..., # DB name, used by scorer to connect
        "unambig_query": ...,     # original unambiguous query (not shown to agent)
        "knowledge_ambiguity": ...,
        "user_query_ambiguity": ...,
        "preprocess_sql": ...,    # run before evaluating predictions
        "clean_up_sqls": ...,     # run after evaluation
        "test_cases": ...,        # test cases for scoring
        "external_knowledge": ...,
    },
)
```

---

## Database Example: `alien`

The `alien` database (located in `alien/`) serves as a reference example of the kind of PostgreSQL databases in BIRD-Interact. It models a radio-signal observatory for SETI research.

### Schema overview

| Table | Description |
|-------|-------------|
| `observatories` | Weather and environmental conditions at each observatory station |
| `telescopes` | Equipment status, calibration, and connectivity per telescope |
| `signals` | Primary signal detection records (frequency, strength, modulation, etc.) |
| `observationalconditions` | Date, time, and duration of each observation |
| `signalprobabilities` | Probability scores for signal origin (biological, artificial, natural) |
| `signalclassification` | Pattern, periodicity, and complexity classification of signals |
| `signaldecoding` | Encoding, decoding method, and confidence for each signal |
| `signaldynamics` | Temporal/spectral stability and coherence metrics |
| `signaladvancedphenomena` | Quantum effects, encryption evidence, cultural significance |
| `researchprocess` | Peer review, publication, and funding status per signal |
| `sourceproperties` | Astronomical properties of the signal's origin (RA/Dec, distance, star type) |

### Column meanings

Column meanings are stored in `alien_column_meaning_base.json` as a flat dict keyed by `"<db>|<Table>|<Column>"`. Each value is a human-readable explanation including full name, data type, and example values or categories. This format is used internally to provide schema context to the user simulator and agent.

### External knowledge (`alien_kb.jsonl`)

Each line is a JSON object with:

| Field | Description |
|-------|-------------|
| `id` | Integer ID |
| `knowledge` | Short name (e.g. `"Signal-to-Noise Quality Indicator (SNQI)"`) |
| `description` | One-sentence summary |
| `definition` | Mathematical formula or decision rule (LaTeX) |
| `type` | One of: `calculation_knowledge`, `domain_knowledge`, `value_illustration` |
| `children_knowledge` | List of `id`s this entry depends on, or `-1` if none |

> **`children_knowledge` direction:** despite the name, these are *prerequisites*, not outputs. An entry lists the IDs it needs to be understood or computed first. For example, if entry 3 has `children_knowledge: [0, 2]`, it means entry 3 depends on entries 0 and 2 — not the other way around.
>
> Example chain from `museum_kb.jsonl`:
> ```
> 0 (CPI)         children: -1       ← leaf, no deps
> 1 (SensWeights) children: -1       ← leaf, no deps
> 2 (ERF)         children: [1]      ← needs SensWeights to map Low/Med/High → 1/5/10
> 3 (AVS)         children: [0, 2]   ← AVS = CPI × ERF, needs both
> ```
> Dependency graph: `0 → 3` and `1 → 2 → 3`.

Knowledge types:
- **`calculation_knowledge`** — formulas derived from table columns (e.g. SNQI, AOI, TOLS)
- **`domain_knowledge`** — conceptual definitions and classification systems (e.g. "Technosignature", "Target of Opportunity")
- **`value_illustration`** — explains what specific column values mean in context (e.g. `WeathProfile: Clear`)
