"""Orchestration for turn classification analysis over result traces."""
from __future__ import annotations

import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console
from rich.table import Table

from conversation2sql.eval_framework.agents.utils import utils_create_model
from conversation2sql.eval_framework.turn_classifier.classifier import TurnClassifier
from conversation2sql.eval_framework.turn_classifier.rules import load_tool_categories
from conversation2sql.eval_framework.turn_classifier.schemas import TurnClassification

console = Console()


@dataclass
class InstanceStats:
    n_turns: int = 0
    l1_counts: Counter[str] = field(default_factory=Counter)


@dataclass
class AnalysisSummary:
    l2_counts: Counter[str] = field(default_factory=Counter)
    l1_counts: Counter[str] = field(default_factory=Counter)
    confidence_counts: Counter[str] = field(default_factory=Counter)
    per_instance: dict[str, InstanceStats] = field(default_factory=dict)
    total_records: int = 0
    total_errors: int = 0


def _iter_records(path: Path):
    """Yield JSON objects from a file that is either compact JSONL or pretty-printed JSON."""
    text = path.read_text()
    decoder = json.JSONDecoder()
    pos = 0
    while pos < len(text):
        stripped = text[pos:]
        lstripped = stripped.lstrip()
        if not lstripped:
            break
        whitespace_chars = len(stripped) - len(lstripped)
        try:
            obj, end = decoder.raw_decode(lstripped)
            yield obj
            pos += whitespace_chars + end
        except json.JSONDecodeError:
            break


def classify_record(
    record: dict,
    classifier: TurnClassifier,
) -> tuple[dict, list[TurnClassification]]:
    """Classify all AI turns in a record.

    Returns:
        (enriched_record, classifications) — enriched_record has a
        'turn_classifications' list added.
    """
    messages = record.get("messages", [])
    prior_failed_submit = False
    turn_classifications: list[TurnClassification] = []

    for i, msg in enumerate(messages):
        role = msg.get("role")
        if role == "ai":
            tc = classifier.classify_turn(
                message_index=i,
                ai_msg=msg,
                prior_failed_submit=prior_failed_submit,
            )
            turn_classifications.append(tc)
        elif role == "tool" and msg.get("tool_name") == "submit_sql":
            content = msg.get("content", {})
            if isinstance(content, dict) and not content.get("passed", True):
                prior_failed_submit = True

    enriched = {
        **record,
        "turn_classifications": [tc.model_dump() for tc in turn_classifications],
    }
    return enriched, turn_classifications
