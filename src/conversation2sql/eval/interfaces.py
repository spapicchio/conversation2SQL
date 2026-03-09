"""
Protocols, data models, and state definitions for the evaluation pipeline.

Implement these Protocols to swap readers, predictors, and scorers:
    - BaseReader → provides EvalSample objects
    - BasePredictor → calls an LLM and returns a message list
    - BaseScorer → compares a Prediction against the gold EvalSample
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable, Literal
from typing import TypedDict

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class BaseMessage(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str


class Sample(BaseModel):
    """A single evaluation example from the dataset."""
    sample_id: str
    conversation: list[BaseMessage]  # conversation history fed to the LLM
    target: str  # gold-standard answer (e.g. a SQL query)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SampleWithPred(Sample):
    """The LLM's response for a single EvalSample."""
    prediction: str  # extracted final text answer
    metadata_pred: dict[str, Any] = Field(default_factory=dict)


class SampleWithPredScore(SampleWithPred):
    """Evaluation outcome for a single Prediction."""
    score: float  # 0.0 (wrong) – 1.0 (correct), or a continuous value
    metadata_score: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Protocols — implement these to plug in custom components
# ---------------------------------------------------------------------------

@runtime_checkable
class BaseReader(Protocol):
    """Reads evaluation samples from any source (file, HuggingFace, DB, …)."""

    def read(self) -> list[Sample]:
        """Return the full list of samples for this evaluation run."""
        ...


@runtime_checkable
class BasePredictor(Protocol):
    """Interfaces with an LLM to produce a response for a message list.

    Implementations must handle the standard LangChain tool-calling cycle:
        AIMessage(tool_calls=[…]) → ToolMessage(…) → AIMessage(content=…)

    The returned list must end with a final AIMessage whose ``.content``
    contains the answer and whose ``.tool_calls`` list is empty.
    """

    def predict(self, samples: list[Sample]) -> list[SampleWithPred]:
        """Call the LLM and return the complete updated message list."""
        ...


@runtime_checkable
class BaseScorer(Protocol):
    """Evaluates a Prediction against its corresponding EvalSample."""

    def score(self, predictions: list[SampleWithPred]) -> list[SampleWithPredScore]:
        """Return a ScoreResult indicating correctness."""
        ...
