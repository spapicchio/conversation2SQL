"""
LangGraph @entrypoint pipeline for the evaluation workflow.

The ``evaluation_pipeline`` function wires the three @task steps together
and adds checkpoint-based resume logic so that a re-run with the same
``thread_id`` skips stages that already completed successfully.

Serialization design
--------------------
LangGraph's ``InMemorySaver`` (msgpack) checkpoints the *inputs* passed to
``.invoke()``.  Reader, Predictor, and Scorer objects are arbitrary Python
classes and cannot be serialized.

Solution: pass components via the ``context`` kwarg to ``.invoke()`` /
``.stream()``.  LangGraph injects ``context`` into the entrypoint function
through the ``runtime`` parameter — context is **not** checkpointed.
Only serializable metadata lives in ``inputs``.

Usage
-----
    from conversation2sql.eval.workflow import evaluation_pipeline

    config  = {"configurable": {"thread_id": "my-eval-run-001"}}
    inputs  = {"metadata": {"model": "gpt-4o-mini"}}          # serializable
    context = {                                                # NOT serialized
        "reader":    my_reader,
        "predictor": my_predictor,
        "scorer":    my_scorer,
    }

    result: EvalState = evaluation_pipeline.invoke(inputs, config, context=context)
    print(result["scores"])

Resume after a crash
--------------------
Call ``.invoke()`` again with the *same* ``thread_id`` and ``context``.
The ``previous`` parameter will contain the EvalState saved by the last
successful checkpoint, so completed stages are not repeated.
"""

from __future__ import annotations

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.func import entrypoint, task
from pydantic import BaseModel, Field

# Trigger registration of all built-in components.
import conversation2sql.eval.dataset_readers  # noqa: F401
import conversation2sql.eval.predictors  # noqa: F401
import conversation2sql.eval.scorers  # noqa: F401
from conversation2sql.config_input import ConfigReader, ConfigPredictor, ConfigScorer
from conversation2sql.eval import (
    reader_registry,
    predictor_registry,
    scorer_registry,
    BasePredictor,
    BaseReader,
    BaseScorer,
    SampleWithPred,
    Sample,
    SampleWithPredScore
)


# ---------------------------------------------------------------------------
# Composite Pipeline Input
# ---------------------------------------------------------------------------


class EvalPipelineInput(BaseModel):
    config_reader: ConfigReader
    config_predictor: ConfigPredictor
    config_scorer: ConfigScorer

    dataset: list[Sample] = Field(default_factory=list)
    dataset_with_pred: list[SampleWithPred] = Field(default_factory=list)
    dataset_with_score: list[SampleWithPredScore] = Field(default_factory=list)


@task
def _run_pred(data_input: EvalPipelineInput, predictor: BasePredictor) -> list[SampleWithPred]:
    return predictor.predict(data_input.dataset)


@task
def _run_score(data_input: EvalPipelineInput, scorer: BaseScorer) -> list[SampleWithPredScore]:
    return scorer.score(data_input.dataset_with_pred)


@entrypoint(checkpointer=InMemorySaver())
def workflow_evaluation_pipeline(
        data_input: EvalPipelineInput,
        # The input (e.g., passed via `invoke`) must be JSON serializable to be checkpointed.
) -> EvalPipelineInput:
    # Step 1: instantiate Reader, Predictor, and Scorer.
    reader: BaseReader = reader_registry.build(data_input.config_reader.reader_name)
    predictor: BasePredictor = predictor_registry.build(data_input.config_predictor.predictor_name)
    scorer: BaseScorer = scorer_registry.build(data_input.config_scorer.scorer_name)

    # step 2: read the dataset using the Reader.
    data_input.dataset = reader.read()

    # step 3: run the prediction
    data_input.dataset_with_pred = _run_pred(data_input, predictor).result()

    # Step 4: run score
    data_input.dataset_with_score = _run_score(data_input, scorer).result()

    return data_input
