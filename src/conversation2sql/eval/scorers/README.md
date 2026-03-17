# Scorers — Structure, Role, and API

## Role in the Eval Pipeline

Scorers are the **third and final stage** of the evaluation pipeline. They receive the list of `SampleWithPred` objects produced by a predictor, compare each prediction against the gold answer, and return `SampleWithPredScore` objects carrying a numeric score and optional score metadata.

Pipeline position:

```
list[SampleWithPred]    ← from predictor
  → scorer_registry.build(scorer_name, config_scorer=…)
  → scorer.score(predictions)      ← scorers live here
  → list[SampleWithPredScore]
```

---

## File Layout

```
src/conversation2sql/eval/scorers/
├── __init__.py              # imports all concrete scorers (triggers auto-registration)
└── placeholder_scorer.py    # stub scorer — always returns score=0.0, for testing
```

---

## Base Contract (`eval/interfaces.py`)

Every scorer must subclass `BaseScorer` and implement one method:

```python
class BaseScorer(ABC):
    def __init__(self, *args, **kwargs): ...

    @abstractmethod
    def score(self, predictions: list[SampleWithPred]) -> list[SampleWithPredScore]: ...
```

`SampleWithPredScore` extends `SampleWithPred` with:

| Field | Type | Description |
|---|---|---|
| `score` | `float` | Correctness value — `0.0` (wrong) to `1.0` (correct), or a continuous metric |
| `metadata_score` | `dict[str, Any]` | Optional per-sample scoring metadata (execution results, error messages, …) |

All other fields from `Sample` and `SampleWithPred` are inherited unchanged.

---

## Auto-Registration Mechanism

Same pattern as readers and predictors: decorating a class with `@scorer_registry.register` registers it under `cls.__name__`. The `scorers/__init__.py` imports every concrete scorer, which is itself imported by `workflow.py` at startup.

---

## Configuration (`config_input.py::ConfigScorer`)

Scorers receive a `ConfigScorer` Pydantic model on construction:

| Field | Description |
|---|---|
| `scorer_name` | Class name used to look up the scorer in `scorer_registry` |
| `run_score_in_parallel` | Whether to score samples concurrently (default: `False`) |

---

## Concrete Scorers

### `PlaceholderScorer`

Returns a `SampleWithPredScore` with `score=0.0` for every sample. No comparison logic, no config fields required. Used in unit tests and as a minimal implementation reference.

---

## External API (where scorers are used)

| Location | How |
|---|---|
| `eval/workflow.py::workflow_evaluation_pipeline` | `scorer_registry.build(name, config_scorer=…)` then `scorer.score(predictions)` |
| `eval/__init__.py` | exports `BaseScorer`, `SampleWithPredScore`, `scorer_registry` |

The only method callers outside the `scorers/` package ever invoke is `.score(list[SampleWithPred]) → list[SampleWithPredScore]`.

---

## How to Contribute a New Scorer

1. **Create the module** — add `src/conversation2sql/eval/scorers/my_scorer.py`.

2. **Subclass and register**:

   ```python
   from conversation2sql.config_input import ConfigScorer
   from conversation2sql.eval import scorer_registry, SampleWithPred, SampleWithPredScore
   from conversation2sql.eval.interfaces import BaseScorer

   @scorer_registry.register
   class MyScorer(BaseScorer):
       def __init__(self, config_scorer: ConfigScorer, *args, **kwargs):
           super().__init__(config_scorer, *args, **kwargs)
           # initialise your scoring logic here

       def score(self, predictions: list[SampleWithPred]) -> list[SampleWithPredScore]:
           results = []
           for pred in predictions:
               # compare pred.messages[-1]['content'] against pred.target
               s = ...  # float between 0.0 and 1.0
               results.append(SampleWithPredScore(score=s, **pred.model_dump()))
           return results
   ```

3. **Export it** — add the import to `eval/scorers/__init__.py`:

   ```python
   from conversation2sql.eval.scorers.my_scorer import MyScorer
   ```

4. **Reference it in config** — set `scorer_name: MyScorer` in your YAML config.

### Execution-based scoring for SQL

For SQL tasks the standard approach is:
- Extract the predicted SQL from the last assistant message.
- Run `sample.metadata['preprocess_sql']` before execution.
- Execute both predicted SQL and gold SQL (`sample.target`) against the target database.
- Compare result sets.
- Run `sample.metadata['clean_up_sqls']` after execution.
- Use `sample.metadata['test_cases']` for fine-grained validation.

### Checklist

- [ ] `score()` returns one `SampleWithPredScore` per input `SampleWithPred`, in the same order.
- [ ] `score` is a `float` in `[0.0, 1.0]` (or a documented continuous metric).
- [ ] Any per-sample scoring details (execution errors, partial matches) go in `metadata_score`.
- [ ] Class is exported from `scorers/__init__.py`.

---

## See Also

- [eval_dataset_readers.md](eval_dataset_readers.md) — readers: how dataset samples are loaded.
- [eval_predictors.md](eval_predictors.md) — predictors: how the LLM is called and how to add a new one.
