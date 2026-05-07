"""Pydantic models for the turn classifier."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

Confidence = Literal["CERTAIN", "CONFIDENT", "PLAUSIBLE", "UNCERTAIN"]


class Level1Classification(BaseModel):
    """Returned by the LLM judge — semantic fields only.

    reasoning is declared first so the model writes CoT before committing to a label.
    """

    reasoning: str
    level1_category: str
    level1_alternatives: list[str]
    confidence: Confidence


class TurnClassification(BaseModel):
    """Per-turn output assembled from Level 2 (rules) + Level 1 (LLM)."""

    message_index: int
    level2_category: str
    level2_tools_called: list[str]
    reasoning: str
    level1_category: str
    level1_alternatives: list[str]
    confidence: Confidence
