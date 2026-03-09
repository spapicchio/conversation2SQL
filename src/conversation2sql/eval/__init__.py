"""conversation2sql.eval — LangGraph Functional API evaluation pipeline."""

from conversation2sql.eval.interfaces import (
    BasePredictor,
    BaseReader,
    BaseScorer,
    Sample,
    SampleWithPred,
    SampleWithPredScore,
)
from conversation2sql.eval.registry import (
    reader_registry,
    predictor_registry,
    scorer_registry,
)
from conversation2sql.eval.workflow import workflow_evaluation_pipeline

__all__ = [
    # Registry
    "reader_registry",
    "predictor_registry",
    "scorer_registry",
    # Protocols
    "BaseReader",
    "BasePredictor",
    "BaseScorer",
    # Data models
    "Sample",
    "SampleWithPred",
    "SampleWithPredScore",
    # Workflow
    "workflow_evaluation_pipeline",
]
