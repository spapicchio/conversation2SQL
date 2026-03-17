# Dataset Readers — Structure, Role, and API

## Role in the Eval Pipeline

Dataset readers are the **first stage** of the evaluation pipeline. Their sole responsibility is to load a dataset from any source (HuggingFace Hub, local files, hard-coded fixtures, etc.) and convert every example into a `Sample` object that downstream components (predictor, scorer) consume.

Pipeline position:

```
Config YAML
  → ConfigReader (Pydantic)
  → reader_registry.build(reader_name, config_reader=…)
  → reader.read()            ← dataset readers live here
  → list[Sample]
  → predictor.predict(…)
  → scorer.score(…)
```

The workflow entry point (`eval/workflow.py::workflow_evaluation_pipeline`) instantiates the reader by name from `reader_registry` and calls `.read()` exactly once per pipeline run.

---

## File Layout

```
src/conversation2sql/eval/dataset_readers/
├── __init__.py               # imports all concrete readers (triggers auto-registration)
├── placeholder_reader.py     # hard-coded toy reader — no I/O, useful for testing
└── bird_interact_reader.py   # production reader for the BIRD-Interact benchmark
```

---

## Base Contract (`eval/interfaces.py`)

Every reader must subclass `BaseReader` and implement one method:

```python
class BaseReader(ABC):
    def __init__(self, *args, **kwargs): ...

    @abstractmethod
    def read(self) -> list[Sample]: ...
```

`Sample` (also in `interfaces.py`) is the data model that readers must produce:

| Field | Type | Description |
|---|---|---|
| `sample_id` | `str` | Unique identifier for this example |
| `messages` | `list[BaseMessage]` | Chat message list (system/user/assistant turns) or a single string prompt |
| `target` | `str` | Gold-standard answer (e.g. the correct SQL query) |
| `user_context` | `UserContext \| None` | Parameters for the user-simulator LLM (used by interactive agents) |
| `metadata` | `dict[str, Any]` | Arbitrary per-sample data carried through the pipeline (DB name, test cases, …) |

`BaseMessage` is a `TypedDict` with `role` and `content` fields.

`UserContext` bundles the template parameters and prompt folder path needed to instantiate a user-simulator LLM during multi-turn agent evaluation.

---

## Auto-Registration Mechanism (`eval/registry.py`)

The global `reader_registry` (a `Registry` instance) maps string names → reader classes.

Decorating a class with `@reader_registry.register` registers it under `cls.__name__`. Registration happens at **import time**, so it is enough to import the module. The `dataset_readers/__init__.py` imports every concrete reader, which is itself imported by `workflow.py` via:

```python
import conversation2sql.eval.dataset_readers  # noqa: F401
```

This guarantees all built-in readers are available before `reader_registry.build(name)` is called.

To add a new reader:
1. Create `eval/dataset_readers/my_reader.py`, subclass `BaseReader`, decorate with `@reader_registry.register`.
2. Import it in `eval/dataset_readers/__init__.py`.
3. Reference it by class name in `config_reader.reader_name` in the YAML config.

---

## Configuration (`config_input.py::ConfigReader`)

Readers receive a `ConfigReader` Pydantic model on construction:

| Field | Description |
|---|---|
| `reader_name` | Class name used to look up the reader in `reader_registry` |
| `dataset_name` | HuggingFace dataset identifier or local path |
| `dataset_kwargs` | Freeform dict for reader-specific options (e.g. `gt_path_jsonl`) |
| `database_engine` | Target DB engine (`sqlite`, `postgresql`, …) — injected into prompts |
| `prompt_dir` | Root directory for Jinja2 templates |
| `system_prompt` | Optional Jinja template name for the system turn |
| `user_prompt` | Jinja template name for the user turn (default: `user.jinja`) |
| `is_chat_template` | If `True`, produce a `list[BaseMessage]`; if `False`, concatenate into a single string |
| `user_simulator_prompt_folder` | Prompt folder for the user-simulator LLM |
| `user_simulator_system_prompt` | Optional system prompt template for the user simulator |
| `user_simulator_user_prompt` | User prompt template for the user simulator |

---

## Concrete Readers

### `PlaceholderReader`

Returns two hard-coded `Sample` objects. No I/O, no config required. Used in unit tests and as a minimal example for implementing a new reader.

### `BirdInteractReader`

Loads the `birdsql/bird-interact-lite` (or `-full`) dataset from HuggingFace and merges it with a local GT JSONL file (required field: `dataset_kwargs.gt_path_jsonl`) that provides `sol_sql`, `test_cases`, and `external_knowledge`.

Key implementation details:

- **GT merge**: the HF dataset is public but omits gold SQL and test cases; those are loaded from the GT JSONL file keyed by `instance_id`.
- **Lite/full guard**: raises `ValueError` if the HF dataset name and GT file path disagree on lite vs full.
- **Prompt rendering**: uses `PromptFactory` to render `system_prompt` and `user_prompt` Jinja templates per sample. Required template variables: `database_engine`, `user_query`, `total_budget`.
- **`UserContext`**: each `Sample` includes a `UserContext` populated with `db_schema`, `amb_user_query`, `user_query_ambiguity`, and `correct_sql` — used by the user-simulator during multi-turn agent evaluation.
- **`metadata`** per sample includes: `selected_database`, `unambig_query`, `knowledge_ambiguity`, `user_query_ambiguity`, `preprocess_sql`, `clean_up_sqls`, `test_cases`, `external_knowledge`.

---

## External API (where readers are used)

| Location | How |
|---|---|
| `eval/workflow.py::workflow_evaluation_pipeline` | `reader_registry.build(name, config_reader=…)` then `reader.read()` |
| `eval/__init__.py` | exports `BaseReader`, `Sample`, `reader_registry` |
| `main.py` (entry point) | constructs `EvalPipelineInput` with `ConfigReader` from YAML, calls `workflow_evaluation_pipeline` |
| `tests/test_datasets_bird_interact.py` | directly instantiates `BirdInteractReader` for unit testing |

The only method callers outside the `dataset_readers/` package ever invoke is `.read() → list[Sample]`. Everything else (`_process_line`, `_build_predictor_input`, `_build_user_context`) is internal to each reader class.

---

## How to Contribute a New Reader

1. **Create the module** — add `src/conversation2sql/eval/dataset_readers/my_reader.py`.

2. **Subclass and register**:

   ```python
   from conversation2sql.config_input import ConfigReader
   from conversation2sql.eval import reader_registry, Sample
   from conversation2sql.eval.interfaces import BaseReader

   @reader_registry.register
   class MyReader(BaseReader):
       def __init__(self, config_reader: ConfigReader, *args, **kwargs):
           super().__init__(config_reader, *args, **kwargs)
           # initialise your data source here

       def read(self) -> list[Sample]:
           # load and return list[Sample]
           ...
   ```

3. **Export it** — add the import to `eval/dataset_readers/__init__.py`:

   ```python
   from conversation2sql.eval.dataset_readers.my_reader import MyReader
   ```

4. **Reference it in config** — set `reader_name: MyReader` in your YAML config (or `--reader_name MyReader` on the CLI). Add any reader-specific keys under `dataset_kwargs`.

5. **Write tests** — add `tests/test_datasets_my_reader.py`. Mock HuggingFace or filesystem I/O; tests run CPU-only with no external services.

### Checklist

- [ ] `read()` returns `list[Sample]` with a unique `sample_id` per item.
- [ ] `messages` is either `list[BaseMessage]` (chat template) or a `str` (completion template), matching `config_reader.is_chat_template`.
- [ ] `target` contains the gold answer used by the scorer.
- [ ] `user_context` is set only if the predictor will use the `ask_user` tool; otherwise leave `None`.
- [ ] Class is exported from `dataset_readers/__init__.py`.

---

## See Also

- [eval_predictors.md](eval_predictors.md) — predictors: how the LLM is called and how to add a new one.
- [eval_scorers.md](eval_scorers.md) — scorers: how predictions are graded and how to add a new one.