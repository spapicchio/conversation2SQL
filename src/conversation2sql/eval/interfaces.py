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

from frozendict import frozendict
from langchain.agents import AgentState as LangChainAgentState
from pydantic import BaseModel, Field, ConfigDict

from conversation2sql.eval.prompt_params import PromptParams


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------
# This TypedDict is used for the short memory into a conversation with tools
# The budget is the number tool interaction the model can do based on the user patience
# This is used to keep a state that can be modified in the conversation
class CustomAgentState(LangChainAgentState):
    user_patience: float  # must be set with invoke

# This is used to give context to the tool and cannot be modified
class ToolUserContext(BaseModel):
    """Used to define the LLM as a user. Must be specified if tool 'ask_user' is used."""
    # User simulator config
    template_params: PromptParams  # typed params for the user-simulator templates
    user_simulator_prompt_folder: str
    user_simulator_system_prompt: str | None = None
    user_simulator_user_prompt: str

    sample: Sample | None = None  # the current sample being evaluated, for use in the user simulator prompts



class BaseMessage(TypedDict):
    role: Literal["system", "user", "assistant", "tool", "function"]
    content: str


class ExternalKnowledgeEntry(BaseModel):
    # Refer to dataset_readers/bird_interact.md for full description of these fields.
    id: int  # id: the integer id of the kb entry, which is the same as the id in the database kb jsonl file.
    knowledge: str  # a short name associated with the knowledge entry.
    description: str  # the description of the knowledge entry.
    definition: str  # the definition of the knowledge entry based on Mathematical formula or decision rule.
    type: str  # the type of the knowledge entry, which can be one of "calculation_knowledge", "domain_knowledge", "value_illustration".
    children_knowledge: list[int]  # list of IDs this entry depends on, or -1 if none


class Sample(BaseModel):
    """A single evaluation example from the dataset."""

    model_config = ConfigDict(extra="allow")  # extra data is ignored but it is stored

    sample_id: str
    # The input may be a chat message list or a single string (e.g. a question)
    # For chat template only Chat models are used, instead for string template only next token prediction models are used.
    messages: list[BaseMessage]
    target: str  # gold-standard answer (e .g. a SQL query)
    user_context: ToolUserContext | None = None

    user_patience: int
    db_dsn: str | None = None  # PostgreSQL DSN, e.g. "postgresql://root:123123@localhost:5432/mydb"
    database_engine: str = "postgresql"
    ddl_database_schema: str
    # External knowledge and column meanings loaded from the dataset
    # List of dicts: {knowledge, description, definition}
    external_knowledge: list[ExternalKnowledgeEntry] = Field(default_factory=list)
    # Nested dict: {table_name: {column_name: meaning | field_meaning}}
    column_meanings: dict[str, dict[str, str | dict]] = Field(default_factory=dict)

    test_cases: list[str] # possible string representing the python code to run as a unit test
    metadata: dict[str, Any] = Field(default_factory=dict)


class SampleWithPred(Sample):
    """The LLM's response for a single EvalSample."""
    # Update the messages directly, no new field
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

