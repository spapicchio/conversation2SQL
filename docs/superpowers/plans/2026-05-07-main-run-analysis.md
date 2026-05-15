# main_run_analysis + classify-turns CLI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `src/conversation2sql/eval_framework/main_run_analysis.py` that orchestrates turn classification over result files and prints aggregate Rich tables, then wire it into the typer CLI as `classify-turns`.

**Architecture:** `main_run_analysis.py` mirrors `main_pipe_workflow.py` — it owns the classify loop, aggregate counters, and Rich table printing; `cli.py` gets a thin `classify-turns` command that resolves the output path default and delegates to `workflow_classification_pipeline`. Tests cover the three pure functions (`_iter_records`, `classify_record`, `_update_summary`) and the pipeline via mocked dependencies.

**Tech Stack:** Python 3.12, `typer`, `rich`, `pydantic`, `unittest.mock`, `pytest`, `uv run pytest`

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `src/conversation2sql/eval_framework/main_run_analysis.py` | Orchestration, aggregation, Rich tables |
| Modify | `src/conversation2sql/cli.py` | Add `classify-turns` command |
| Create | `tests/eval_framework/test_main_run_analysis.py` | Unit + integration tests |

---

### Task 1: Data types + `_iter_records` + `classify_record`

**Files:**
- Create: `src/conversation2sql/eval_framework/main_run_analysis.py`
- Create: `tests/eval_framework/test_main_run_analysis.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/eval_framework/test_main_run_analysis.py`:

```python
from __future__ import annotations

import json
from collections import Counter
from unittest.mock import MagicMock

import pytest

from conversation2sql.eval_framework.turn_classifier.schemas import TurnClassification


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_tc(
    index: int = 0,
    l2: str = "TEXT_ONLY",
    l1: str = "DISCUSSION",
    confidence: str = "CERTAIN",
) -> TurnClassification:
    return TurnClassification(
        message_index=index,
        level2_category=l2,
        level2_tools_called=[],
        reasoning="test",
        level1_category=l1,
        level1_alternatives=[],
        confidence=confidence,
    )


# ── _iter_records ─────────────────────────────────────────────────────────────

def test_iter_records_compact_jsonl(tmp_path):
    from conversation2sql.eval_framework.main_run_analysis import _iter_records

    p = tmp_path / "test.jsonl"
    records = [{"id": 1, "v": "a"}, {"id": 2, "v": "b"}]
    p.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    assert list(_iter_records(p)) == records


def test_iter_records_pretty_json(tmp_path):
    from conversation2sql.eval_framework.main_run_analysis import _iter_records

    p = tmp_path / "test.jsonl"
    records = [{"id": 1}, {"id": 2}]
    p.write_text(json.dumps(records[0], indent=2) + "\n" + json.dumps(records[1], indent=2))
    assert list(_iter_records(p)) == records


def test_iter_records_empty_file(tmp_path):
    from conversation2sql.eval_framework.main_run_analysis import _iter_records

    p = tmp_path / "empty.jsonl"
    p.write_text("")
    assert list(_iter_records(p)) == []


# ── classify_record ───────────────────────────────────────────────────────────

def test_classify_record_single_ai_turn():
    from conversation2sql.eval_framework.main_run_analysis import classify_record

    tc = _make_tc(index=0)
    mock_classifier = MagicMock()
    mock_classifier.classify_turn.return_value = tc

    record = {
        "instance_id": "q1",
        "messages": [
            {"role": "ai", "content": [{"type": "text", "text": "hello"}], "tool_calls": []},
            {"role": "human", "content": "ok"},
        ],
    }
    enriched, classifications = classify_record(record, mock_classifier)

    assert classifications == [tc]
    assert enriched["turn_classifications"] == [tc.model_dump()]
    mock_classifier.classify_turn.assert_called_once_with(
        message_index=0,
        ai_msg=record["messages"][0],
        prior_failed_submit=False,
    )


def test_classify_record_prior_failed_submit_flag():
    from conversation2sql.eval_framework.main_run_analysis import classify_record

    tc_first = _make_tc(index=0)
    tc_second = _make_tc(index=2)
    mock_classifier = MagicMock()
    mock_classifier.classify_turn.side_effect = [tc_first, tc_second]

    record = {
        "instance_id": "q2",
        "messages": [
            {"role": "ai", "content": [], "tool_calls": []},
            {
                "role": "tool",
                "tool_name": "submit_sql",
                "content": {"passed": False},
            },
            {"role": "ai", "content": [], "tool_calls": []},
        ],
    }
    _, classifications = classify_record(record, mock_classifier)

    assert len(classifications) == 2
    # second call must see prior_failed_submit=True
    second_call = mock_classifier.classify_turn.call_args_list[1]
    assert second_call.kwargs["prior_failed_submit"] is True


def test_classify_record_no_ai_turns():
    from conversation2sql.eval_framework.main_run_analysis import classify_record

    mock_classifier = MagicMock()
    record = {"instance_id": "q3", "messages": [{"role": "human", "content": "hi"}]}
    enriched, classifications = classify_record(record, mock_classifier)

    assert classifications == []
    assert enriched["turn_classifications"] == []
    mock_classifier.classify_turn.assert_not_called()
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
uv run pytest tests/eval_framework/test_main_run_analysis.py -v
```

Expected: `ImportError` — `main_run_analysis` does not exist yet.

- [ ] **Step 3: Create `main_run_analysis.py` with data types + `_iter_records` + `classify_record`**

Create `src/conversation2sql/eval_framework/main_run_analysis.py`:

```python
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
```

- [ ] **Step 4: Run tests — expect pass**

```bash
uv run pytest tests/eval_framework/test_main_run_analysis.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/conversation2sql/eval_framework/main_run_analysis.py \
        tests/eval_framework/test_main_run_analysis.py
git commit -m "feat: add main_run_analysis.py with data types, _iter_records, classify_record"
```

---

### Task 2: `_update_summary` aggregation

**Files:**
- Modify: `src/conversation2sql/eval_framework/main_run_analysis.py`
- Modify: `tests/eval_framework/test_main_run_analysis.py`

- [ ] **Step 1: Add failing tests**

Append to `tests/eval_framework/test_main_run_analysis.py`:

```python
# ── _update_summary ───────────────────────────────────────────────────────────

def test_update_summary_increments_all_counters():
    from conversation2sql.eval_framework.main_run_analysis import (
        AnalysisSummary,
        _update_summary,
    )

    summary = AnalysisSummary()
    tc = _make_tc(l2="TEXT_ONLY", l1="DISCUSSION", confidence="CERTAIN")
    _update_summary(summary, "q1", [tc])

    assert summary.l2_counts["TEXT_ONLY"] == 1
    assert summary.l1_counts["DISCUSSION"] == 1
    assert summary.confidence_counts["CERTAIN"] == 1
    assert "q1" in summary.per_instance
    assert summary.per_instance["q1"].n_turns == 1
    assert summary.per_instance["q1"].l1_counts["DISCUSSION"] == 1


def test_update_summary_accumulates_across_calls():
    from conversation2sql.eval_framework.main_run_analysis import (
        AnalysisSummary,
        _update_summary,
    )

    summary = AnalysisSummary()
    _update_summary(summary, "q1", [_make_tc(l1="DISCUSSION"), _make_tc(l1="CLARIFICATION")])
    _update_summary(summary, "q1", [_make_tc(l1="DISCUSSION")])

    assert summary.l1_counts["DISCUSSION"] == 2
    assert summary.l1_counts["CLARIFICATION"] == 1
    assert summary.per_instance["q1"].n_turns == 3


def test_update_summary_separate_instances():
    from conversation2sql.eval_framework.main_run_analysis import (
        AnalysisSummary,
        _update_summary,
    )

    summary = AnalysisSummary()
    _update_summary(summary, "q1", [_make_tc(l1="DISCUSSION")])
    _update_summary(summary, "q2", [_make_tc(l1="CLARIFICATION")])

    assert "q1" in summary.per_instance
    assert "q2" in summary.per_instance
    assert summary.per_instance["q1"].n_turns == 1
    assert summary.per_instance["q2"].n_turns == 1
    assert summary.l1_counts["DISCUSSION"] == 1
    assert summary.l1_counts["CLARIFICATION"] == 1
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
uv run pytest tests/eval_framework/test_main_run_analysis.py -k "update_summary" -v
```

Expected: `ImportError` — `_update_summary` not yet defined.

- [ ] **Step 3: Add `_update_summary` to `main_run_analysis.py`**

Add after `classify_record` in `src/conversation2sql/eval_framework/main_run_analysis.py`:

```python
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
```

- [ ] **Step 4: Run tests — expect pass**

```bash
uv run pytest tests/eval_framework/test_main_run_analysis.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/conversation2sql/eval_framework/main_run_analysis.py \
        tests/eval_framework/test_main_run_analysis.py
git commit -m "feat: add _update_summary aggregation to main_run_analysis"
```

---

### Task 3: `workflow_classification_pipeline` + `print_summary_tables`

**Files:**
- Modify: `src/conversation2sql/eval_framework/main_run_analysis.py`
- Modify: `tests/eval_framework/test_main_run_analysis.py`

- [ ] **Step 1: Add failing tests**

Append to `tests/eval_framework/test_main_run_analysis.py`:

```python
# ── workflow_classification_pipeline ─────────────────────────────────────────

def test_workflow_writes_enriched_jsonl(tmp_path):
    from unittest.mock import patch
    from conversation2sql.eval_framework.main_run_analysis import workflow_classification_pipeline

    input_file = tmp_path / "results_smaller.jsonl"
    input_file.write_text(
        json.dumps({
            "instance_id": "q1",
            "messages": [{"role": "ai", "content": [], "tool_calls": []}],
        }) + "\n"
    )
    output = tmp_path / "out.jsonl"
    tc = _make_tc(index=0, l2="NO_ACTION", l1="MISSING")

    with (
        patch("conversation2sql.eval_framework.main_run_analysis.utils_create_model"),
        patch("conversation2sql.eval_framework.main_run_analysis.load_tool_categories", return_value={}),
        patch("conversation2sql.eval_framework.main_run_analysis.TurnClassifier") as mock_cls,
    ):
        mock_cls.return_value.classify_turn.return_value = tc
        summary = workflow_classification_pipeline(
            inputs=[input_file],
            output=output,
            model_str="openai/gpt-4o-mini",
            tool_categories_path=tmp_path / "tool_categories.yaml",
        )

    assert summary.total_records == 1
    assert summary.total_errors == 0
    assert output.exists()
    record = json.loads(output.read_text().strip())
    assert "turn_classifications" in record
    assert record["turn_classifications"][0]["level1_category"] == "MISSING"


def test_workflow_counts_unreadable_file_as_error(tmp_path):
    from unittest.mock import patch
    from conversation2sql.eval_framework.main_run_analysis import workflow_classification_pipeline

    missing = tmp_path / "does_not_exist.jsonl"
    output = tmp_path / "out.jsonl"

    with (
        patch("conversation2sql.eval_framework.main_run_analysis.utils_create_model"),
        patch("conversation2sql.eval_framework.main_run_analysis.load_tool_categories", return_value={}),
        patch("conversation2sql.eval_framework.main_run_analysis.TurnClassifier"),
    ):
        summary = workflow_classification_pipeline(
            inputs=[missing],
            output=output,
            model_str="openai/gpt-4o-mini",
            tool_categories_path=tmp_path / "tool_categories.yaml",
        )

    assert summary.total_records == 0
    assert summary.total_errors == 1


def test_workflow_aggregates_summary(tmp_path):
    from unittest.mock import patch
    from conversation2sql.eval_framework.main_run_analysis import workflow_classification_pipeline

    input_file = tmp_path / "results_smaller.jsonl"
    input_file.write_text(
        "\n".join([
            json.dumps({"instance_id": "q1", "messages": [{"role": "ai", "content": [], "tool_calls": []}]}),
            json.dumps({"instance_id": "q2", "messages": [{"role": "ai", "content": [], "tool_calls": []}]}),
        ])
    )
    output = tmp_path / "out.jsonl"
    tc = _make_tc(l2="TEXT_ONLY", l1="DISCUSSION", confidence="CERTAIN")

    with (
        patch("conversation2sql.eval_framework.main_run_analysis.utils_create_model"),
        patch("conversation2sql.eval_framework.main_run_analysis.load_tool_categories", return_value={}),
        patch("conversation2sql.eval_framework.main_run_analysis.TurnClassifier") as mock_cls,
    ):
        mock_cls.return_value.classify_turn.return_value = tc
        summary = workflow_classification_pipeline(
            inputs=[input_file],
            output=output,
            model_str="openai/gpt-4o-mini",
            tool_categories_path=tmp_path / "tc.yaml",
        )

    assert summary.total_records == 2
    assert summary.l2_counts["TEXT_ONLY"] == 2
    assert summary.l1_counts["DISCUSSION"] == 2
    assert "q1" in summary.per_instance
    assert "q2" in summary.per_instance
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
uv run pytest tests/eval_framework/test_main_run_analysis.py -k "workflow" -v
```

Expected: `ImportError` — `workflow_classification_pipeline` not yet defined.

- [ ] **Step 3: Add `workflow_classification_pipeline` and `print_summary_tables` to `main_run_analysis.py`**

Append to `src/conversation2sql/eval_framework/main_run_analysis.py`:

```python
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
```

- [ ] **Step 4: Run all tests — expect pass**

```bash
uv run pytest tests/eval_framework/test_main_run_analysis.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Run full suite to catch regressions**

```bash
uv run pytest tests/ -v
```

Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/conversation2sql/eval_framework/main_run_analysis.py \
        tests/eval_framework/test_main_run_analysis.py
git commit -m "feat: add workflow_classification_pipeline and print_summary_tables"
```

---

### Task 4: Add `classify-turns` command to `cli.py`

**Files:**
- Modify: `src/conversation2sql/cli.py`

- [ ] **Step 1: Add the import at the top of `cli.py`**

In `src/conversation2sql/cli.py`, add to the existing imports block:

```python
from conversation2sql.eval_framework.main_run_analysis import (
    workflow_classification_pipeline,
    print_summary_tables,
)
```

- [ ] **Step 2: Add the `classify-turns` command**

Append to `src/conversation2sql/cli.py` (after the `results` command):

```python
@app.command("classify-turns")
def classify_turns(
    inputs: list[Path] = typer.Argument(..., help="Input results_smaller.jsonl file(s)."),
    output: Optional[Path] = typer.Option(None, "--output", help="Output JSONL path."),
    model: str = typer.Option(
        "openai/gpt-4o-mini",
        "--model",
        help="LiteLLM model string (provider/model).",
    ),
    tool_categories: Path = typer.Option(
        Path("configs/turn_classifier/tool_categories.yaml"),
        "--tool-categories",
        help="Path to tool_categories.yaml.",
    ),
) -> None:
    """Classify agent turns in BIRD-Interact result traces and show aggregate statistics."""
    load_dotenv(".env")
    if output is None:
        output = (
            inputs[0].parent / "results_classified.jsonl"
            if len(inputs) == 1
            else Path("results_classified.jsonl")
        )
    summary = workflow_classification_pipeline(inputs, output, model, tool_categories)
    print_summary_tables(summary, output)
```

- [ ] **Step 3: Verify the command appears in the CLI help**

```bash
uv run c2sql --help
```

Expected output includes `classify-turns` in the commands list.

```bash
uv run c2sql classify-turns --help
```

Expected: shows `--output`, `--model`, `--tool-categories` options.

- [ ] **Step 4: Run full test suite**

```bash
uv run pytest tests/ -v
```

Expected: all tests PASS (no regressions in existing `run` and `results` commands).

- [ ] **Step 5: Commit**

```bash
git add src/conversation2sql/cli.py
git commit -m "feat: add classify-turns CLI command wired to workflow_classification_pipeline"
```

---

## Self-Review

**Spec coverage check:**

| Spec requirement | Task |
|---|---|
| `main_run_analysis.py` in `eval_framework/` | Task 1 |
| `AnalysisSummary` + `InstanceStats` dataclasses | Task 1 |
| `workflow_classification_pipeline` function | Task 3 |
| `print_summary_tables` with 4 Rich tables | Task 3 |
| L2 distribution table | Task 3 |
| L1 distribution table | Task 3 |
| Confidence distribution table | Task 3 |
| Per-instance breakdown table | Task 3 |
| `classify-turns` CLI command | Task 4 |
| Output path default logic | Task 4 |
| `--model`, `--output`, `--tool-categories` options | Task 4 |
| File-level errors increment `total_errors` | Task 3 |
| Record-level errors increment `total_errors` | Task 3 |
| `scripts/classify_turns.py` left unchanged | (no task touches it) ✓ |

All requirements covered. No placeholders. Type signatures consistent across tasks (`AnalysisSummary`, `InstanceStats`, `TurnClassification` used identically in Tasks 1-4).
