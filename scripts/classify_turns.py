#!/usr/bin/env python3
"""CLI: classify every agent turn in one or more results_smaller.jsonl files.

Usage:
    uv run python scripts/classify_turns.py results/*/results_smaller.jsonl \\
        --output results/classified.jsonl \\
        --model openai/gpt-4o-mini \\
        --tool-categories configs/turn_classifier/tool_categories.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

from conversation2sql.eval_framework.agents.utils import utils_create_model
from conversation2sql.eval_framework.turn_classifier.classifier import TurnClassifier
from conversation2sql.eval_framework.turn_classifier.rules import load_tool_categories


def iter_records(path: Path):
    """Yield JSON records handling both pretty-printed and compact JSONL formats."""
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


def classify_record(record: dict, classifier: TurnClassifier) -> dict:
    """Add turn_classifications list to a single result record."""
    messages = record.get("messages", [])
    prior_failed_submit = False
    turn_classifications = []

    for i, msg in enumerate(messages):
        role = msg.get("role")
        if role == "ai":
            tc = classifier.classify_turn(
                message_index=i,
                ai_msg=msg,
                prior_failed_submit=prior_failed_submit,
            )
            turn_classifications.append(tc.model_dump())
        elif role == "tool" and msg.get("tool_name") == "submit_sql":
            content = msg.get("content", {})
            if isinstance(content, dict) and not content.get("passed", True):
                prior_failed_submit = True

    return {**record, "turn_classifications": turn_classifications}


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Classify agent turns in BIRD-Interact result traces."
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="Input JSONL file(s)")
    parser.add_argument("--output", type=Path, default=None, help="Output JSONL path")
    parser.add_argument(
        "--model",
        default="openai/gpt-4o-mini",
        help="LiteLLM model string (provider/model)",
    )
    parser.add_argument(
        "--tool-categories",
        default="configs/turn_classifier/tool_categories.yaml",
        help="Path to tool_categories.yaml",
    )
    args = parser.parse_args()

    if args.output is None:
        if len(args.inputs) == 1:
            args.output = args.inputs[0].parent / "results_classified.jsonl"
        else:
            args.output = Path("results_classified.jsonl")

    provider, model_name = args.model.split("/", 1)
    model = utils_create_model(
        model_name=model_name,
        model_provider=provider,
        temperature=0.0,
        max_tokens=1024,
    )

    tool_categories = load_tool_categories(args.tool_categories)
    classifier = TurnClassifier(model=model, tool_categories=tool_categories)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    processed = 0
    errors = 0

    with open(args.output, "w") as out_f:
        for input_path in args.inputs:
            records = list(iter_records(input_path))
            for record in tqdm(records, desc=str(input_path)):
                instance_id = record.get("instance_id", "?")
                try:
                    classified = classify_record(record, classifier)
                    out_f.write(json.dumps(classified) + "\n")
                    processed += 1
                except Exception as exc:
                    print(
                        f"ERROR [{instance_id}]: {exc}",
                        file=sys.stderr,
                    )
                    errors += 1

    print(f"\nDone — {processed} records classified, {errors} errors.")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
