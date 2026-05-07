"""Orchestration for turn classification analysis over result traces."""
from __future__ import annotations

import json
import sys
from collections import Counter
from collections.abc import Iterator
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


def _iter_records(path: Path) -> Iterator[dict]:
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
            print(f"WARN: unparseable content at position {pos} in {path}, stopping early", file=sys.stderr)
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
        # Sticky: once a submit fails, all subsequent turns see prior_failed_submit=True
        elif role == "tool" and msg.get("tool_name") == "submit_sql":
            content = msg.get("content", {})
            if isinstance(content, dict) and not content.get("passed", True):
                prior_failed_submit = True

    enriched = {
        **record,
        "turn_classifications": [tc.model_dump() for tc in turn_classifications],
    }
    return enriched, turn_classifications


def _update_summary(
    summary: AnalysisSummary,
    instance_id: str,
    classifications: list[TurnClassification],
) -> None:
    if instance_id not in summary.per_instance:
        summary.per_instance[instance_id] = InstanceStats()
    stats = summary.per_instance[instance_id]
    for tc in classifications:
        summary.l2_counts[tc.level2_category] += 1
        summary.l1_counts[tc.level1_category] += 1
        summary.confidence_counts[tc.confidence] += 1
        stats.n_turns += 1
        stats.l1_counts[tc.level1_category] += 1


def workflow_classification_pipeline(
    inputs: list[Path],
    output: Path,
    model_str: str,
    tool_categories_path: Path,
) -> AnalysisSummary:
    """Classify every AI turn in the given input files and write enriched JSONL.

    Args:
        inputs: one or more results_smaller.jsonl files.
        output: destination JSONL path (parent directory created if absent).
        model_str: LiteLLM model string in "provider/model" format.
        tool_categories_path: path to tool_categories.yaml.

    Returns:
        AnalysisSummary with aggregate counters and per-instance stats.
    """
    tool_categories = load_tool_categories(tool_categories_path)
    provider, model_name = model_str.split("/", 1)
    model = utils_create_model(
        model_name=model_name,
        model_provider=provider,
        temperature=0.0,
        max_tokens=1024,
    )
    classifier = TurnClassifier(model=model, tool_categories=tool_categories)

    summary = AnalysisSummary()
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, "w") as out_f:
        for input_path in inputs:
            try:
                records = list(_iter_records(input_path))
            except Exception as exc:
                print(f"ERROR reading {input_path}: {exc}", file=sys.stderr)
                summary.total_errors += 1
                continue
            for record in records:
                instance_id = str(record.get("instance_id", "?"))
                try:
                    enriched, classifications = classify_record(record, classifier)
                    out_f.write(json.dumps(enriched) + "\n")
                    _update_summary(summary, instance_id, classifications)
                    summary.total_records += 1
                except Exception as exc:
                    print(f"ERROR [{instance_id}]: {exc}", file=sys.stderr)
                    summary.total_errors += 1

    return summary


def print_summary_tables(summary: AnalysisSummary, output: Path) -> None:
    """Print four Rich tables: L2 distribution, L1 distribution, confidence, per-instance."""
    total_turns = sum(summary.l2_counts.values())

    def _pct(n: int) -> str:
        return f"{100 * n / total_turns:.1f}%" if total_turns else "0%"

    l2_table = Table(title="L2 Category Distribution")
    l2_table.add_column("category", justify="left")
    l2_table.add_column("count", justify="right")
    l2_table.add_column("%", justify="right")
    for cat, count in sorted(summary.l2_counts.items(), key=lambda x: -x[1]):
        l2_table.add_row(cat, str(count), _pct(count))
    console.print(l2_table)

    l1_table = Table(title="L1 Category Distribution")
    l1_table.add_column("category", justify="left")
    l1_table.add_column("count", justify="right")
    l1_table.add_column("%", justify="right")
    for cat, count in sorted(summary.l1_counts.items(), key=lambda x: -x[1]):
        l1_table.add_row(cat, str(count), _pct(count))
    console.print(l1_table)

    conf_table = Table(title="Confidence Distribution")
    conf_table.add_column("level", justify="left")
    conf_table.add_column("count", justify="right")
    conf_table.add_column("%", justify="right")
    for level, count in sorted(summary.confidence_counts.items(), key=lambda x: -x[1]):
        conf_table.add_row(level, str(count), _pct(count))
    console.print(conf_table)

    inst_table = Table(title="Per-Instance Breakdown")
    inst_table.add_column("instance_id", justify="left")
    inst_table.add_column("n_turns", justify="right")
    inst_table.add_column("top_l1", justify="left")
    inst_table.add_column("top_l1_count", justify="right")
    for iid, stats in sorted(summary.per_instance.items()):
        top_l1, top_count = (
            stats.l1_counts.most_common(1)[0] if stats.l1_counts else ("—", 0)
        )
        inst_table.add_row(iid, str(stats.n_turns), top_l1, str(top_count))
    console.print(inst_table)

    console.print(
        f"\nDone — {summary.total_records} records classified, "
        f"{summary.total_errors} errors → {output}"
    )
