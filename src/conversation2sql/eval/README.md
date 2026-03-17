# Evaluation Pipeline

This package implements a three-stage, registry-based evaluation pipeline for conversation-to-SQL tasks.

## Pipeline Overview

```
Config YAML / CLI args
  │
  ▼
PydanticParser  (cli_parser.py)
  │  merges defaults < env vars < YAML < CLI overrides
  ▼
EvalPipelineInput
  ├── ConfigReader
  ├── ConfigPredictor
  └── ConfigScorer
  │
  ▼ workflow_evaluation_pipeline()  (workflow.py)
  │
  ├─[1]─ reader_registry.build(reader_name)
  │         reader.read()
  │         → list[Sample]
  │
  ├─[2]─ predictor_registry.build(predictor_name)
  │         predictor.predict(list[Sample])
  │         → list[SampleWithPred]
  │
  └─[3]─ scorer_registry.build(scorer_name)
            scorer.score(list[SampleWithPred])
            → list[SampleWithPredScore]
```

Each stage is **pluggable**: swap any component by changing a single name in the YAML config. Components are auto-discovered via a decorator-based registry — no central factory to edit.

---

## Package Layout

```
eval/
├── __init__.py              # public exports: registries, base classes, data models
├── interfaces.py            # data models (Sample, SampleWithPred, …) and base classes
├── registry.py              # Registry class + reader_registry, predictor_registry, scorer_registry
├── workflow.py              # workflow_evaluation_pipeline() — wires the three stages together
├── dataset_readers/         # Stage 1 — load and format dataset samples
│   ├── __init__.py
│   ├── placeholder_reader.py
│   └── bird_interact_reader.py
├── predictors/              # Stage 2 — call an LLM to produce predictions
│   ├── __init__.py
│   ├── placeholder_predictor.py
│   ├── langchain_predictor.py
│   ├── langchain_agent_factory.py
│   └── available_tools/     # LangChain tools bound to agents
│       ├── __init__.py
│       └── tool_placeholder.py
└── scorers/                 # Stage 3 — grade predictions against gold answers
    ├── __init__.py
    └── placeholder_scorer.py
```

---

## Data Models (`interfaces.py`)

Data flows through the pipeline as immutable Pydantic models. Each stage extends the previous one.

```
Sample
  └── SampleWithPred
        └── SampleWithPredScore
```

| Model | Key fields |
|---|---|
| `Sample` | `sample_id`, `messages: list[BaseMessage]`, `target: str`, `user_context`, `metadata` |
| `SampleWithPred` | + `metadata_pred` |
| `SampleWithPredScore` | + `score: float`, `metadata_score` |

`BaseMessage` is a `TypedDict` with `role` (`system`/`user`/`assistant`/`tool`) and `content`.

`UserContext` carries the template parameters and prompt folder for the user-simulator LLM. It is set by the reader and consumed by the `ask_user` tool during multi-turn agent evaluation.

---

## Registry Mechanism (`registry.py`)

Three global registries map string names → classes:

| Registry | Required method | Used for |
|---|---|---|
| `reader_registry` | `read()` | Dataset readers |
| `predictor_registry` | `predict()` | LLM predictors |
| `scorer_registry` | `score()` | Scorers |

A fourth registry, `tool_registry`, maps tool names → LangChain tool callables.

**Registration** happens at import time via the `@registry.register` decorator:

```python
@reader_registry.register       # registers as "MyReader"
class MyReader(BaseReader):
    ...
```

`workflow.py` imports all three subpackages at startup to guarantee every built-in component is registered before `registry.build(name)` is called.

**Instantiation** by name:

```python
reader = reader_registry.build("BirdInteractReader", config_reader=config_reader)
```

---

## Stage 1 — Readers (`dataset_readers/`)

**Contract**: `read() → list[Sample]`

A reader loads a dataset from any source and converts each example into a `Sample`. It is responsible for:
- Fetching raw data (HuggingFace, local files, etc.).
- Merging supplementary ground-truth files if needed.
- Rendering per-sample Jinja2 prompt templates into `messages`.
- Populating `target` (gold SQL) and `metadata` (DB name, test cases, …).
- Optionally building a `UserContext` for multi-turn agent evaluation.

Built-in readers:

| Class | Description |
|---|---|
| `PlaceholderReader` | Returns two hard-coded samples; no I/O. Used in tests. |
| `BirdInteractReader` | Loads `birdsql/bird-interact-lite/full` from HuggingFace, merges with a local GT JSONL. |

See `.claude/eval_dataset_readers.md` for full field-by-field documentation.

---

## Stage 2 — Predictors (`predictors/`)

**Contract**: `predict(list[Sample]) → list[SampleWithPred]`

A predictor calls an LLM for each sample and returns the full updated message history. It must preserve the input order and produce exactly one `SampleWithPred` per input `Sample`.

Built-in predictors:

| Class | Description |
|---|---|
| `PlaceholderPredictor` | Returns `prediction='placeholder'` for every sample; no LLM call. Used in tests. |
| `LangChainPredictor` | Runs a LangGraph ReAct agent. Calls `agent.invoke` per sample, passing `user_context` as LangGraph runtime context. |

`LangChainAgentFactory` (not a registered component) builds the agent from `ConfigPredictor`. It binds tools from `tool_registry` and attaches middleware to cap model and tool call counts.

**Tools** (`available_tools/`) are `@tool`-decorated LangChain functions also registered in `tool_registry`. The `ask_user` tool is the only non-stub tool: when the agent needs clarification it calls `ask_user`, which invokes a user-simulator LLM using the `UserContext` injected by LangGraph at runtime.

See `.claude/eval_predictors.md` for full documentation.

---

## Stage 3 — Scorers (`scorers/`)

**Contract**: `score(list[SampleWithPred]) → list[SampleWithPredScore]`

A scorer compares each prediction against the gold answer (`sample.target`) and assigns a `score` in `[0.0, 1.0]`. For SQL tasks the typical approach is execution-based: run both predicted and gold SQL against the database and compare result sets, using `metadata['preprocess_sql']`, `metadata['test_cases']`, and `metadata['clean_up_sqls']`.

Built-in scorers:

| Class | Description |
|---|---|
| `PlaceholderScorer` | Always returns `score=0.0`. Used in tests. |

See `.claude/eval_scorers.md` for full documentation.

---

## Configuration

Components are selected and configured via a YAML file (see `configs/eval_pipeline_config.yaml`). The three top-level sections map directly to Pydantic models in `config_input.py`:

```yaml
reader:
  reader_name: BirdInteractReader
  dataset_name: birdsql/bird-interact-lite
  dataset_kwargs:
    gt_path_jsonl: /path/to/gt_lite.jsonl
  database_engine: sqlite
  prompt_dir: prompts
  system_prompt: bird_interact_a_agent/system.jinja
  user_prompt: bird_interact_a_agent/user.jinja
  is_chat_template: true

predictor:
  predictor_name: LangChainPredictor
  model_name: gpt-4o
  model_provider: openai
  tool_names: [user, sql_query_tool]
  temperature: 0.0
  top_k: 1
  top_p: 1.0
  max_new_tokens: 2048

scorer:
  scorer_name: PlaceholderScorer
```

Run the pipeline:

```bash
uv run python main.py --config configs/eval_pipeline_config.yaml
```

---

## Contributing a New Component

All three component types follow the same four-step pattern.

### Adding a Reader

1. Create `eval/dataset_readers/my_reader.py`:

   ```python
   from conversation2sql.config_input import ConfigReader
   from conversation2sql.eval import reader_registry, Sample
   from conversation2sql.eval.interfaces import BaseReader

   @reader_registry.register
   class MyReader(BaseReader):
       def __init__(self, config_reader: ConfigReader, *args, **kwargs):
           super().__init__(config_reader, *args, **kwargs)

       def read(self) -> list[Sample]:
           ...
   ```

2. Add to `eval/dataset_readers/__init__.py`:

   ```python
   from conversation2sql.eval.dataset_readers.my_reader import MyReader
   ```

3. Set `reader_name: MyReader` in the YAML config.

### Adding a Predictor

1. Create `eval/predictors/my_predictor.py`:

   ```python
   from conversation2sql.config_input import ConfigPredictor
   from conversation2sql.eval import predictor_registry, Sample, SampleWithPred
   from conversation2sql.eval.interfaces import BasePredictor

   @predictor_registry.register
   class MyPredictor(BasePredictor):
       def __init__(self, config_predictor: ConfigPredictor, *args, **kwargs):
           super().__init__(config_predictor, *args, **kwargs)

       def predict(self, samples: list[Sample]) -> list[SampleWithPred]:
           ...
   ```

2. Add to `eval/predictors/__init__.py`.
3. Set `predictor_name: MyPredictor` in the YAML config.

### Adding a Tool

1. Add a `@tool`-decorated function in `available_tools/` and decorate with `@tool_registry.register(name='my_tool')`.
2. Export from `available_tools/__init__.py`.
3. List `my_tool` under `tool_names` in the YAML config.

If the tool needs access to `UserContext` at runtime, accept `runtime: ToolRuntime[UserContext]` — LangGraph injects it automatically.

### Adding a Scorer

1. Create `eval/scorers/my_scorer.py`:

   ```python
   from conversation2sql.config_input import ConfigScorer
   from conversation2sql.eval import scorer_registry, SampleWithPred, SampleWithPredScore
   from conversation2sql.eval.interfaces import BaseScorer

   @scorer_registry.register
   class MyScorer(BaseScorer):
       def __init__(self, config_scorer: ConfigScorer, *args, **kwargs):
           super().__init__(config_scorer, *args, **kwargs)

       def score(self, predictions: list[SampleWithPred]) -> list[SampleWithPredScore]:
           ...
   ```

2. Add to `eval/scorers/__init__.py`.
3. Set `scorer_name: MyScorer` in the YAML config.

### Contribution Checklist

- [ ] Class decorated with the correct `@*_registry.register`.
- [ ] Class exported from its subpackage `__init__.py`.
- [ ] `read()` / `predict()` / `score()` returns one output object per input object, in the same order.
- [ ] No mutable state shared across samples (pipeline may run samples in parallel in future).
- [ ] Tests added under `tests/` — mock all external I/O; tests must run CPU-only with no network access.

---

## Further Reading

- `.claude/eval_dataset_readers.md` — detailed reader documentation.
- `.claude/eval_predictors.md` — detailed predictor and tool documentation.
- `.claude/eval_scorers.md` — detailed scorer documentation.
- `CLAUDE.md` — overall project architecture and setup instructions.