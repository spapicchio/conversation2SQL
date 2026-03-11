"""
Protocols, data models, and state definitions for the evaluation pipeline.

Implement these Protocols to swap readers, predictors, and scorers:
    - BaseReader → provides EvalSample objects
    - BasePredictor → calls an LLM and returns a message list
    - BaseScorer → compares a Prediction against the gold EvalSample
"""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from typing import Any, Literal
from typing import TypedDict

from pydantic import BaseModel, Field, ConfigDict


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class BaseMessage(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str


class Sample(BaseModel):
    """A single evaluation example from the dataset."""
    model_config = ConfigDict(extra='ignore')  # extra data is ignored but it is stored

    sample_id: str
    # The input may be a chat message list or a single string (e.g. a question)
    # For chat template only Chat models are used, instead for string template only next token prediction models are used.
    messages: list[BaseMessage]
    target: str  # gold-standard answer (e .g. a SQL query)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SampleWithPred(Sample):
    """The LLM's response for a single EvalSample."""
    metadata_pred: dict[str, Any] = Field(default_factory=dict)


class SampleWithPredScore(SampleWithPred):
    """Evaluation outcome for a single Prediction."""
    score: float  # 0.0 (wrong) – 1.0 (correct), or a continuous value
    metadata_score: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Protocols — implement these to plug in custom components
# ---------------------------------------------------------------------------

class BaseReader(ABC):
    """Reads evaluation samples from any source (file, HuggingFace, DB, …)."""

    def __init__(self, *args, **kwargs):
        """Initialize the reader with any necessary parameters (e.g. file path)."""
        pass

    @abstractmethod
    def read(self) -> list[Sample]:
        """Return the full list of samples for this evaluation run."""
        ...


class BasePredictor(ABC):
    """Interfaces with an LLM to produce a response for a message list.

    Implementations must handle the standard LangChain tool-calling cycle:
        AIMessage(tool_calls=[…]) → ToolMessage(…) → AIMessage(content=…)

    The returned list must end with a final AIMessage whose ``.content``
    contains the answer and whose ``.tool_calls`` list is empty.
    """

    def __init__(self, *args, **kwargs):
        """Initialize the predictor with any necessary parameters (e.g. model name)."""
        pass

    @abstractmethod
    def predict(self, samples: list[Sample]) -> list[SampleWithPred]:
        """Call the LLM and return the complete updated message list."""
        ...


class BaseScorer(ABC):
    """Evaluates a Prediction against its corresponding EvalSample."""

    def __init__(self, *args, **kwargs):
        """Initialize the scorer with any necessary parameters (e.g. evaluation criteria)."""
        pass

    @abstractmethod
    def score(self, predictions: list[SampleWithPred]) -> list[SampleWithPredScore]:
        """Return a ScoreResult indicating correctness."""
        ...
