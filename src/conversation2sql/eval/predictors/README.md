# Predictors — Structure, Role, and API

## Role in the Eval Pipeline

Predictors are the **second stage** of the evaluation pipeline. They receive the list of `Sample` objects produced by a reader, call an LLM (or return a stub), and return `SampleWithPred` objects that carry the updated message history. The scorer then grades these predictions.

Pipeline position:

```
list[Sample]          ← from reader
  → predictor_registry.build(predictor_name, config_predictor=…)
  → predictor.predict(samples)     ← predictors live here
  → list[SampleWithPred]
  → scorer.score(…)
```

---

## File Layout

```
src/conversation2sql/eval/predictors/
├── __init__.py                      # imports all concrete predictors (triggers auto-registration)
├── placeholder_predictor.py         # stub predictor — no LLM call, for testing
├── langchain_predictor.py           # production predictor wrapping a LangChain agent
├── langchain_agent_factory.py       # builds the LangChain/LangGraph agent from ConfigPredictor
└── available_tools/
    ├── __init__.py                  # imports all tools (triggers tool_registry registration)
    └── tool_placeholder.py          # built-in LangChain tools registered in tool_registry
```

---

## Base Contract (`eval/interfaces.py`)

Every predictor must subclass `BasePredictor` and implement one method:

```python
class BasePredictor(ABC):
    def __init__(self, *args, **kwargs): ...

    @abstractmethod
    def predict(self, samples: list[Sample]) -> list[SampleWithPred]: ...
```

`SampleWithPred` extends `Sample` with:

| Field | Type | Description |
|---|---|---|
| `messages` | `list[BaseMessage]` | Full conversation history including the model's final reply |
| `metadata_pred` | `dict[str, Any]` | Optional per-sample prediction metadata (latency, token counts, …) |

All other fields from `Sample` are inherited unchanged.

---

## Auto-Registration Mechanism

Same pattern as readers: decorating a class with `@predictor_registry.register` registers it under `cls.__name__`. The `predictors/__init__.py` imports every concrete predictor, which is itself imported by `workflow.py` at startup.

---

## Configuration (`config_input.py::ConfigPredictor`)

Predictors receive a `ConfigPredictor` Pydantic model on construction:

| Field | Description |
|---|---|
| `predictor_name` | Class name used to look up the predictor in `predictor_registry` |
| `model_name` | LLM identifier (e.g. `gpt-4o`, `o4-mini`) |
| `model_provider` | LangChain provider string (e.g. `openai`, `huggingface`) |
| `tool_names` | List of tool names from `tool_registry` to bind to the agent |
| `temperature` | Sampling temperature |
| `top_k` | Top-k sampling parameter |
| `top_p` | Nucleus sampling probability |
| `max_new_tokens` | Maximum tokens to generate |

---

## Concrete Predictors

### `PlaceholderPredictor`

Returns a `SampleWithPred` with `prediction='placeholder'` for every sample. No LLM call, no config fields required. Used in unit tests and as a minimal implementation reference.

### `LangChainPredictor`

Wraps a LangChain/LangGraph ReAct agent. For each sample it calls `agent.invoke({'messages': sample.messages}, context=sample.user_context)` and converts the response message list back to `BaseMessage` dicts.

Key design points:

- **Agent construction** is delegated to `LangChainAgentFactory.create_agent()`.
- **`user_context`** is passed as the LangGraph `context` kwarg so tools can access it at runtime (see `ask_user` tool).
- Samples are processed **sequentially** (one `invoke` per sample). A batched path using `agent.batch(…, config={'max_concurrency': 5})` is present but currently commented out.
- Requires all samples to have `messages` as a `list[BaseMessage]` — raises `ValueError` otherwise.

---

## Agent Factory (`langchain_agent_factory.py`)

`LangChainAgentFactory` is a helper (not a registered component) used internally by `LangChainPredictor`:

- `create_agent()` — builds a compiled LangGraph `StateGraph` via `langchain.agents.create_agent` with:
  - The model initialised via `init_chat_model` (provider-agnostic LangChain factory).
  - Tools looked up by name from `tool_registry`.
  - `ModelCallLimitMiddleware(run_limit=3)` — caps total LLM calls per trajectory.
  - `ToolCallLimitMiddleware(run_limit=1, thread_limit=1)` — caps tool calls per round and across the full conversation.
- `get_cached_model(…)` — module-level `@cache` function used by tools to reuse model instances across calls.

---

## Available Tools (`available_tools/`)

Tools are LangChain `@tool`-decorated functions also registered in the global `tool_registry` (from `eval/registry.py`). The registry maps string names → tool callables.

`LangChainAgentFactory.create_agent()` looks up `config_predictor.tool_names` in `tool_registry` to bind tools to the agent.

### Built-in tools (`tool_placeholder.py`)

| Registry name | Function | Description |
|---|---|---|
| `search` | `search(query)` | Stub web search |
| `get_weather` | `get_weather(location)` | Stub weather lookup |
| `sql_query_tool` | `sql_query_tool(query)` | Stub SQL execution |
| `db_query_tool` | `db_query_tool(query)` | Stub DB execution |
| `user` | `ask_user(question, runtime)` | **User-simulator tool** — calls an LLM to answer clarification questions |

The `ask_user` tool is the only non-stub tool. It:
1. Receives the clarification `question` from the agent and a `ToolRuntime[UserContext]` injected by LangGraph.
2. Appends `clarification_question` to `UserContext.template_params`.
3. Uses `get_cached_prompt_factory` to render the user-simulator Jinja templates.
4. Calls a cached LLM (`o4-mini` by default) and returns its response as a string.

---

## External API (where predictors are used)

| Location | How |
|---|---|
| `eval/workflow.py::workflow_evaluation_pipeline` | `predictor_registry.build(name, config_predictor=…)` then `predictor.predict(samples)` |
| `eval/__init__.py` | exports `BasePredictor`, `SampleWithPred`, `predictor_registry` |
| `tests/test_predictors.py` | directly instantiates predictors for unit testing |

The only method callers outside the `predictors/` package ever invoke is `.predict(list[Sample]) → list[SampleWithPred]`.

---

## How to Contribute a New Predictor

1. **Create the module** — add `src/conversation2sql/eval/predictors/my_predictor.py`.

2. **Subclass and register**:

   ```python
   from conversation2sql.config_input import ConfigPredictor
   from conversation2sql.eval import predictor_registry, Sample, SampleWithPred
   from conversation2sql.eval.interfaces import BasePredictor

   @predictor_registry.register
   class MyPredictor(BasePredictor):
       def __init__(self, config_predictor: ConfigPredictor, *args, **kwargs):
           super().__init__(config_predictor, *args, **kwargs)
           # initialise your model/client here

       def predict(self, samples: list[Sample]) -> list[SampleWithPred]:
           # call LLM, return list[SampleWithPred]
           ...
   ```

3. **Export it** — add the import to `eval/predictors/__init__.py`:

   ```python
   from conversation2sql.eval.predictors.my_predictor import MyPredictor
   ```

4. **Reference it in config** — set `predictor_name: MyPredictor` in your YAML config.

### Adding a New Tool

1. Add a `@tool`-decorated function in `available_tools/` (or a new submodule).
2. Decorate it with `@tool_registry.register(name='my_tool')`.
3. Import it from `available_tools/__init__.py`.
4. List `my_tool` under `tool_names` in the YAML config.

If the tool needs access to `UserContext` at runtime, accept a `runtime: ToolRuntime[UserContext]` parameter — LangGraph injects it automatically (see `ask_user` for reference).

### Checklist

- [ ] `predict()` returns one `SampleWithPred` per input `Sample`, in the same order.
- [ ] `messages` in the returned object is the **full** conversation history (input turns + model reply).
- [ ] Class is exported from `predictors/__init__.py`.
- [ ] Any new tools are exported from `available_tools/__init__.py` and registered in `tool_registry`.

---

## See Also

- [eval_dataset_readers.md](eval_dataset_readers.md) — readers: how dataset samples are loaded.
- [eval_scorers.md](eval_scorers.md) — scorers: how predictions are graded and how to add a new one.
