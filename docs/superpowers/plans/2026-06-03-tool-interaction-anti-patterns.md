# Tool-Interaction Anti-Pattern Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add script-only (no LLM) anti-pattern detection and a positional tool-usage plot to the `explorer/` Streamlit app, for diagnosing broken tool-use behavior in BIRD-Interact result traces — both single-run and cross-run.

**Architecture:** A new pure module `explorer/patterns.py` extracts a normalized tool-event sequence from each record's `messages`, runs a registry of deterministic detectors, and aggregates results **sample-average style** (per-instance first, then across instances) from the `groups` dict that `load_run` already builds. `loader.py` attaches per-record hits and exposes aggregate frequencies on `RunStats`. A new page `explorer/pages/patterns.py` renders single-run diagnosis (frequency bar, positional stacked bar, drill-down); `explorer/pages/compare.py` gains a per-run frequency table.

**Tech Stack:** Python 3.12, `pandas`, `altair`, `streamlit`, `pytest`, `uv run pytest`. Detectors are Streamlit-free and unit-tested.

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `explorer/patterns.py` | Tool-event extraction, detector registry, sample-average aggregation, positional distribution |
| Create | `tests/explorer/test_patterns.py` | Unit tests for extraction, every detector, aggregation, positional frame |
| Modify | `explorer/loader.py` | Attach `_pattern_hits` per record; add `pattern_frequency` + `clean_fraction` to `RunStats` |
| Modify | `tests/explorer/test_loader.py` | Assert new `RunStats` fields are sample-average |
| Modify | `explorer/render.py` | Optional `highlight_indices` arg to flag offending messages |
| Modify | `tests/explorer/test_render.py` | Assert highlight marker renders |
| Create | `explorer/pages/patterns.py` | Single-run diagnosis page (frequency bar, positional plot, drill-down) |
| Modify | `explorer/pages/compare.py` | Per-run anti-pattern frequency table |

**Import convention:** test files import from `explorer.patterns` / `explorer.loader`. Streamlit pages (run with `explorer/` on `sys.path`) import bare (`from patterns import ...`), matching the existing `from loader import ...` in `pages/compare.py`. `loader.py` imports `patterns` with the same `try/except ModuleNotFoundError` fallback it already uses for `metrics`. `patterns.py` must NOT import from `loader` (avoids a circular import) — it defines its own small message-text helper.

---

### Task 1: Tool-event extraction + dataclasses

**Files:**
- Create: `explorer/patterns.py`
- Create: `tests/explorer/test_patterns.py`

- [ ] **Step 1: Write the failing test**

Create `tests/explorer/test_patterns.py`:

```python
from __future__ import annotations

from explorer.patterns import ToolEvent, extract_tool_events


def _ai(*tool_calls):
    """An ai message with the given (tool_name, arguments) tool calls."""
    return {
        "role": "ai",
        "tool_calls": [{"tool_name": n, "arguments": a} for n, a in tool_calls],
    }


def _tool(name, status="success", content="ok"):
    return {"role": "tool", "tool_name": name, "status": status, "content": content}


def _record(*messages):
    return {"messages": list(messages)}


def test_extract_pairs_single_call_with_result():
    rec = _record(
        {"role": "system", "content": "sys"},
        {"role": "human", "content": "q"},
        _ai(("get_schema", {})),
        _tool("get_schema", content={"schema": "..."}),
    )
    events = extract_tool_events(rec)
    assert len(events) == 1
    e = events[0]
    assert isinstance(e, ToolEvent)
    assert e.tool_name == "get_schema"
    assert e.message_index == 3
    assert e.status == "success"
    assert e.is_error is False


def test_extract_multi_call_turn_back_to_back_tool_messages():
    rec = _record(
        _ai(
            ("get_knowledge_definition", {"knowledge_name": "A"}),
            ("get_knowledge_definition", {"knowledge_name": "B"}),
        ),
        _tool("get_knowledge_definition"),
        _tool("get_knowledge_definition"),
    )
    events = extract_tool_events(rec)
    assert [e.tool_name for e in events] == [
        "get_knowledge_definition",
        "get_knowledge_definition",
    ]
    # Arguments matched positionally by name, in order.
    assert events[0].arguments == {"knowledge_name": "A"}
    assert events[1].arguments == {"knowledge_name": "B"}
    assert events[1].message_index == 2


def test_extract_marks_errors_and_extracts_text():
    rec = _record(
        _ai(("execute_sql", {"sql": "SELECT 1"})),
        _tool("execute_sql", status="error", content={"message": "syntax error"}),
    )
    events = extract_tool_events(rec)
    assert events[0].is_error is True
    assert "syntax error" in events[0].result_text


def test_extract_empty_when_no_tool_messages():
    rec = _record({"role": "ai", "content": "hi", "tool_calls": []})
    assert extract_tool_events(rec) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/explorer/test_patterns.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'explorer.patterns'`

- [ ] **Step 3: Write the dataclasses + extraction**

Create `explorer/patterns.py`:

```python
"""Deterministic (no-LLM) tool-interaction anti-pattern detection for result traces.

Operates on the normalized tool-event sequence extracted from each record's
``messages``. All aggregates are sample-average: averaged per instance first, then
across instances, mirroring ``loader.pass_at_1`` and ``metrics.reliability_metrics``.

This module must not import from ``explorer.loader`` (would be circular); it carries
its own small message-text helper instead.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field

import pandas as pd


def _message_text(content: object) -> str:
    """Best-effort text of a tool message's content (dict or str)."""
    if isinstance(content, dict):
        for key in ("content", "message", "error"):
            value = content.get(key)
            if isinstance(value, str):
                return value
        return json.dumps(content, ensure_ascii=False)
    if isinstance(content, str):
        return content
    return ""


@dataclass(frozen=True)
class ToolEvent:
    message_index: int   # index into record["messages"] of the tool-result message
    tool_name: str
    arguments: dict
    status: str          # "success" or an error status
    is_error: bool       # status != "success"
    result_text: str     # tool message content as text


def extract_tool_events(record: dict) -> list[ToolEvent]:
    """Walk messages, pairing each ai tool_call with its following tool result(s).

    When an ai turn has K tool_calls, the next K tool messages are its results.
    Arguments are matched to a result by tool name, in order (handles multi-call
    turns and back-to-back tool messages).
    """
    events: list[ToolEvent] = []
    pending: list[tuple[str, dict]] = []  # (tool_name, arguments) awaiting a result
    for idx, msg in enumerate(record.get("messages", [])):
        role = msg.get("role")
        if role == "ai":
            pending = [
                (tc.get("tool_name", ""), tc.get("arguments", {}) or {})
                for tc in (msg.get("tool_calls") or [])
            ]
        elif role == "tool":
            name = msg.get("tool_name", "")
            args: dict = {}
            for j, (pname, pargs) in enumerate(pending):
                if pname == name:
                    args = pargs
                    pending.pop(j)
                    break
            status = msg.get("status", "")
            events.append(
                ToolEvent(
                    message_index=idx,
                    tool_name=name,
                    arguments=args,
                    status=status,
                    is_error=status != "success",
                    result_text=_message_text(msg.get("content")),
                )
            )
    return events
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/explorer/test_patterns.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add explorer/patterns.py tests/explorer/test_patterns.py
git commit -m "feat(explorer): tool-event extraction for anti-pattern detection"
```

---

### Task 2: Anti-pattern detectors + registry

**Files:**
- Modify: `explorer/patterns.py`
- Modify: `tests/explorer/test_patterns.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/explorer/test_patterns.py`:

```python
from explorer.patterns import PATTERN_CATALOG, PATTERN_NAMES, detect_patterns


def _hits(rec) -> set[str]:
    return {h.name for h in detect_patterns(rec)}


def test_catalog_names_match():
    assert PATTERN_NAMES == [name for name, _ in PATTERN_CATALOG]
    assert "blind_submit" in PATTERN_NAMES


def test_blind_submit_fires_without_execute():
    rec = _record(_ai(("submit_sql", {"sql": "SELECT 1"})), _tool("submit_sql"))
    assert "blind_submit" in _hits(rec)


def test_blind_submit_quiet_when_execute_precedes():
    rec = _record(
        _ai(("execute_sql", {"sql": "SELECT 1"})), _tool("execute_sql"),
        _ai(("submit_sql", {"sql": "SELECT 1"})), _tool("submit_sql"),
    )
    assert "blind_submit" not in _hits(rec)


def test_repeated_identical_call_ignores_whitespace_in_sql():
    rec = _record(
        _ai(("execute_sql", {"sql": "SELECT  1"})), _tool("execute_sql"),
        _ai(("execute_sql", {"sql": "SELECT 1"})), _tool("execute_sql"),
    )
    assert "repeated_identical_call" in _hits(rec)


def test_repeated_identical_call_quiet_when_args_differ():
    rec = _record(
        _ai(("execute_sql", {"sql": "SELECT 1"})), _tool("execute_sql"),
        _ai(("execute_sql", {"sql": "SELECT 2"})), _tool("execute_sql"),
    )
    assert "repeated_identical_call" not in _hits(rec)


def test_submit_after_error_fires_on_identical_failed_sql():
    rec = _record(
        _ai(("execute_sql", {"sql": "SELECT x"})),
        _tool("execute_sql", status="error", content="column x does not exist"),
        _ai(("submit_sql", {"sql": "SELECT x"})), _tool("submit_sql"),
    )
    assert "submit_after_error" in _hits(rec)


def test_unrecovered_error_loop_needs_three_consecutive():
    two = _record(
        _ai(("execute_sql", {"sql": "a"})), _tool("execute_sql", status="error"),
        _ai(("execute_sql", {"sql": "b"})), _tool("execute_sql", status="error"),
    )
    assert "unrecovered_error_loop" not in _hits(two)
    three = _record(
        _ai(("execute_sql", {"sql": "a"})), _tool("execute_sql", status="error"),
        _ai(("execute_sql", {"sql": "b"})), _tool("execute_sql", status="error"),
        _ai(("execute_sql", {"sql": "c"})), _tool("execute_sql", status="error"),
    )
    assert "unrecovered_error_loop" in _hits(three)


def test_unrecovered_error_loop_resets_on_success():
    rec = _record(
        _ai(("execute_sql", {"sql": "a"})), _tool("execute_sql", status="error"),
        _ai(("execute_sql", {"sql": "b"})), _tool("execute_sql", status="success"),
        _ai(("execute_sql", {"sql": "c"})), _tool("execute_sql", status="error"),
        _ai(("execute_sql", {"sql": "d"})), _tool("execute_sql", status="error"),
    )
    assert "unrecovered_error_loop" not in _hits(rec)


def test_kb_blind_fires_when_kb_present_and_unused():
    rec = _record(_ai(("execute_sql", {"sql": "x"})), _tool("execute_sql"))
    rec["masked_agent_kb"] = {"BFR": "Bandwidth ratio"}
    assert "kb_blind" in _hits(rec)


def test_kb_blind_quiet_without_kb():
    rec = _record(_ai(("execute_sql", {"sql": "x"})), _tool("execute_sql"))
    assert "kb_blind" not in _hits(rec)


def test_no_submission_fires_when_never_submitted():
    rec = _record(_ai(("execute_sql", {"sql": "x"})), _tool("execute_sql"))
    assert "no_submission" in _hits(rec)


def test_budget_death_fires_without_successful_submit_and_low_budget():
    rec = _record(_ai(("execute_sql", {"sql": "x"})), _tool("execute_sql"))
    rec["updated_user_patience"] = 0
    assert "budget_death" in _hits(rec)


def test_budget_death_quiet_after_successful_submit():
    rec = _record(_ai(("submit_sql", {"sql": "x"})), _tool("submit_sql", status="success"))
    rec["updated_user_patience"] = -2
    assert "budget_death" not in _hits(rec)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/explorer/test_patterns.py -v`
Expected: FAIL — `ImportError: cannot import name 'detect_patterns'`

- [ ] **Step 3: Implement detectors + registry**

Append to `explorer/patterns.py`:

```python
@dataclass(frozen=True)
class PatternHit:
    name: str                    # "blind_submit"
    label: str                   # "Blind submit"
    detail: str                  # human evidence string
    message_indices: list[int] = field(default_factory=list)


ERROR_LOOP_MIN = 3  # consecutive execute_sql errors that constitute an unrecovered loop


def _norm_sql(sql: object) -> str:
    return " ".join(sql.split()) if isinstance(sql, str) else ""


def _norm_args(arguments: dict) -> str:
    a = dict(arguments or {})
    if isinstance(a.get("sql"), str):
        a["sql"] = _norm_sql(a["sql"])
    return json.dumps(a, sort_keys=True, ensure_ascii=False)


def _detect_blind_submit(events: list[ToolEvent], record: dict) -> PatternHit | None:
    seen_execute = False
    for e in events:
        if e.tool_name == "execute_sql":
            seen_execute = True
        elif e.tool_name == "submit_sql":
            if not seen_execute:
                return PatternHit(
                    "blind_submit", "Blind submit",
                    "submitted without any prior execute_sql", [e.message_index],
                )
            return None  # first submission was validated
    return None


def _detect_repeated_identical_call(events: list[ToolEvent], record: dict) -> PatternHit | None:
    seen: dict[tuple[str, str], list[int]] = {}
    for e in events:
        seen.setdefault((e.tool_name, _norm_args(e.arguments)), []).append(e.message_index)
    for (name, _), idxs in seen.items():
        if len(idxs) >= 2:
            return PatternHit(
                "repeated_identical_call", "Repeated identical call",
                f"`{name}` called {len(idxs)}× with identical arguments", idxs,
            )
    return None


def _detect_submit_after_error(events: list[ToolEvent], record: dict) -> PatternHit | None:
    errored: dict[str, int] = {}
    for e in events:
        if e.tool_name == "execute_sql" and e.is_error:
            errored[_norm_sql(e.arguments.get("sql"))] = e.message_index
        elif e.tool_name == "submit_sql":
            key = _norm_sql(e.arguments.get("sql"))
            if key and key in errored:
                return PatternHit(
                    "submit_after_error", "Submit after error",
                    "submitted SQL is identical to an execute_sql that errored",
                    [errored[key], e.message_index],
                )
    return None


def _detect_unrecovered_error_loop(events: list[ToolEvent], record: dict) -> PatternHit | None:
    streak: list[int] = []
    for e in events:
        if e.tool_name != "execute_sql":
            continue
        if e.is_error:
            streak.append(e.message_index)
            if len(streak) >= ERROR_LOOP_MIN:
                return PatternHit(
                    "unrecovered_error_loop", "Unrecovered error loop",
                    f"{len(streak)} consecutive execute_sql errors", list(streak),
                )
        else:
            streak = []
    return None


def _detect_kb_blind(events: list[ToolEvent], record: dict) -> PatternHit | None:
    kb = record.get("masked_agent_kb") or record.get("gt_knowledge_base") or {}
    if not kb:
        return None
    if not any(e.tool_name == "get_knowledge_definition" for e in events):
        return PatternHit(
            "kb_blind", "KB-blind",
            f"{len(kb)} KB entries available but get_knowledge_definition never called",
        )
    return None


def _detect_budget_death(events: list[ToolEvent], record: dict) -> PatternHit | None:
    if any(e.tool_name == "submit_sql" and not e.is_error for e in events):
        return None
    budget = record.get("updated_user_patience")
    exhausted = any("budget exhausted" in e.result_text.lower() for e in events) or (
        isinstance(budget, (int, float)) and budget <= 0
    )
    if exhausted:
        return PatternHit(
            "budget_death", "Budget death",
            "budget exhausted before a successful submit_sql",
        )
    return None


def _detect_no_submission(events: list[ToolEvent], record: dict) -> PatternHit | None:
    if not any(e.tool_name == "submit_sql" for e in events):
        return PatternHit(
            "no_submission", "No submission",
            "conversation ended with no submit_sql call",
        )
    return None


Detector = Callable[[list[ToolEvent], dict], "PatternHit | None"]

ANTI_PATTERNS: list[Detector] = [
    _detect_blind_submit,
    _detect_repeated_identical_call,
    _detect_submit_after_error,
    _detect_unrecovered_error_loop,
    _detect_kb_blind,
    _detect_budget_death,
    _detect_no_submission,
]

# (name, label) for display + enumeration independent of whether a detector fires.
PATTERN_CATALOG: list[tuple[str, str]] = [
    ("blind_submit", "Blind submit"),
    ("repeated_identical_call", "Repeated identical call"),
    ("submit_after_error", "Submit after error"),
    ("unrecovered_error_loop", "Unrecovered error loop"),
    ("kb_blind", "KB-blind"),
    ("budget_death", "Budget death"),
    ("no_submission", "No submission"),
]
PATTERN_NAMES: list[str] = [name for name, _ in PATTERN_CATALOG]


def detect_patterns(record: dict) -> list[PatternHit]:
    events = extract_tool_events(record)
    return [hit for d in ANTI_PATTERNS if (hit := d(events, record)) is not None]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/explorer/test_patterns.py -v`
Expected: PASS (all detector tests green)

- [ ] **Step 5: Commit**

```bash
git add explorer/patterns.py tests/explorer/test_patterns.py
git commit -m "feat(explorer): anti-pattern detector registry"
```

---

### Task 3: Sample-average aggregation + positional distribution

**Files:**
- Modify: `explorer/patterns.py`
- Modify: `tests/explorer/test_patterns.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/explorer/test_patterns.py`:

```python
from explorer.patterns import (
    aggregate_patterns,
    clean_fraction,
    positional_tool_distribution,
)


def _submit_only(sql="SELECT 1"):
    # A clean validated submission: execute then submit, no anti-patterns.
    return _record(
        _ai(("execute_sql", {"sql": sql})), _tool("execute_sql"),
        _ai(("submit_sql", {"sql": sql})), _tool("submit_sql", status="success"),
    )


def test_aggregate_is_sample_average_not_pooled():
    # One instance, 2 samples: one blind-submits, one is clean.
    blind = _record(_ai(("submit_sql", {"sql": "x"})), _tool("submit_sql", status="success"))
    groups = {"inst1": [blind, _submit_only()]}
    freq = aggregate_patterns(groups)
    # Per-instance mean = 1/2 = 0.5 (not pooled over a flat list, but same here).
    assert freq["blind_submit"] == 0.5
    assert freq["no_submission"] == 0.0


def test_aggregate_weights_instances_equally():
    # inst_a: 2 samples both blind (1.0). inst_b: 1 sample clean (0.0).
    blind = _record(_ai(("submit_sql", {"sql": "x"})), _tool("submit_sql", status="success"))
    groups = {"a": [blind, dict(blind)], "b": [_submit_only()]}
    freq = aggregate_patterns(groups)
    # Mean over instances of per-instance rate = (1.0 + 0.0) / 2 = 0.5,
    # NOT pooled 2/3 = 0.667.
    assert freq["blind_submit"] == 0.5


def test_clean_fraction_sample_average():
    blind = _record(_ai(("submit_sql", {"sql": "x"})), _tool("submit_sql", status="success"))
    groups = {"inst1": [blind, _submit_only()]}
    # 1 of 2 samples is clean.
    assert clean_fraction(groups) == 0.5


def test_positional_distribution_shares_sum_to_one_per_position():
    groups = {"inst1": [_submit_only()]}
    df = positional_tool_distribution(groups, max_pos=8)
    sums = df.groupby("position")["share"].sum()
    for pos, total in sums.items():
        assert abs(total - 1.0) < 1e-9, f"position {pos} sums to {total}"


def test_positional_distribution_no_call_band_appears_after_end():
    # 2-event sequence; positions 3..8 should be "(no call)".
    groups = {"inst1": [_submit_only()]}
    df = positional_tool_distribution(groups, max_pos=8)
    pos3 = df[(df["position"] == "3") & (df["tool"] == "(no call)")]["share"]
    assert float(pos3.iloc[0]) == 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/explorer/test_patterns.py -k "aggregate or clean_fraction or positional" -v`
Expected: FAIL — `ImportError: cannot import name 'aggregate_patterns'`

- [ ] **Step 3: Implement aggregation + positional distribution**

Append to `explorer/patterns.py`:

```python
def _record_hits(record: dict) -> list[PatternHit]:
    """Use cached hits from loader if present, else compute."""
    cached = record.get("_pattern_hits")
    return cached if cached is not None else detect_patterns(record)


def aggregate_patterns(groups: dict[str, list[dict]]) -> dict[str, float]:
    """name -> mean over instances of (fraction of that instance's samples hitting it).

    Per record a pattern counts at most once (presence, not occurrence count).
    """
    per_instance: dict[str, list[float]] = {n: [] for n in PATTERN_NAMES}
    for samples in groups.values():
        if not samples:
            continue
        present = [{h.name for h in _record_hits(r)} for r in samples]
        for name in PATTERN_NAMES:
            per_instance[name].append(sum(name in p for p in present) / len(samples))
    return {
        name: (sum(v) / len(v) if v else 0.0) for name, v in per_instance.items()
    }


def clean_fraction(groups: dict[str, list[dict]]) -> float:
    """Sample-average fraction of samples with zero anti-pattern hits."""
    per_instance: list[float] = []
    for samples in groups.values():
        if not samples:
            continue
        clean = sum(1 for r in samples if not _record_hits(r))
        per_instance.append(clean / len(samples))
    return sum(per_instance) / len(per_instance) if per_instance else 0.0


def positional_tool_distribution(
    groups: dict[str, list[dict]], max_pos: int = 8
) -> pd.DataFrame:
    """Long-form frame [position, tool, share] for a 100%-stacked positional plot.

    Columns 1..max_pos-1 are individual call positions; the last column ">=max_pos"
    shows the tool at position max_pos for still-active conversations. Each sample
    contributes exactly one category per position (a tool name, or "(no call)" once
    the sequence has ended), so shares per position sum to 1. Aggregated sample-average:
    per-instance share averaged over ALL instances (missing categories imply 0).
    """
    positions = [str(p) for p in range(1, max_pos)] + [f"≥{max_pos}"]
    n_inst = sum(1 for s in groups.values() if s)
    if n_inst == 0:
        return pd.DataFrame(columns=["position", "tool", "share"])

    accum: dict[str, dict[str, float]] = {pos: {} for pos in positions}
    for samples in groups.values():
        if not samples:
            continue
        counts: dict[str, Counter] = {pos: Counter() for pos in positions}
        for r in samples:
            seq = [e.tool_name for e in extract_tool_events(r)]
            for i, pos in enumerate(positions):
                cat = seq[i] if i < len(seq) else "(no call)"
                counts[pos][cat] += 1
        n = len(samples)
        for pos in positions:
            for cat, c in counts[pos].items():
                accum[pos][cat] = accum[pos].get(cat, 0.0) + (c / n)

    rows = [
        {"position": pos, "tool": cat, "share": total / n_inst}
        for pos in positions
        for cat, total in accum[pos].items()
    ]
    return pd.DataFrame(rows)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/explorer/test_patterns.py -v`
Expected: PASS (all patterns tests green)

- [ ] **Step 5: Commit**

```bash
git add explorer/patterns.py tests/explorer/test_patterns.py
git commit -m "feat(explorer): sample-average aggregation + positional tool distribution"
```

---

### Task 4: Wire detection into the loader

**Files:**
- Modify: `explorer/loader.py`
- Modify: `tests/explorer/test_loader.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/explorer/test_loader.py`:

```python
def test_compute_stats_pattern_frequency_sample_average():
    # One instance, 2 samples: one blind-submit (no execute), one validated.
    blind = make_record(
        instance_id="i1",
        messages=[
            {"role": "ai", "tool_calls": [{"tool_name": "submit_sql", "arguments": {"sql": "x"}}]},
            {"role": "tool", "tool_name": "submit_sql", "status": "success", "content": "ok"},
        ],
    )
    validated = make_record(
        instance_id="i1",
        messages=[
            {"role": "ai", "tool_calls": [{"tool_name": "execute_sql", "arguments": {"sql": "x"}}]},
            {"role": "tool", "tool_name": "execute_sql", "status": "success", "content": "ok"},
            {"role": "ai", "tool_calls": [{"tool_name": "submit_sql", "arguments": {"sql": "x"}}]},
            {"role": "tool", "tool_name": "submit_sql", "status": "success", "content": "ok"},
        ],
    )
    groups = {"i1": [blind, validated]}
    stats = _compute_stats([blind, validated], groups)
    assert stats.pattern_frequency["blind_submit"] == 0.5
    assert stats.clean_fraction == 0.5
```

Note: confirm `make_record` (top of `test_loader.py`) forwards `**extra` into the record dict so `instance_id=` and `messages=` land as keys. If it does not, add them explicitly to the returned dict in `make_record`.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/explorer/test_loader.py::test_compute_stats_pattern_frequency_sample_average -v`
Expected: FAIL — `AttributeError: 'RunStats' object has no attribute 'pattern_frequency'`

- [ ] **Step 3: Add the import and `RunStats` fields**

In `explorer/loader.py`, just below the existing `metrics` import block (around line 15), add:

```python
try:  # pragma: no cover - bare import only when Streamlit runs from explorer/
    from patterns import aggregate_patterns, clean_fraction, detect_patterns
except ModuleNotFoundError:
    from explorer.patterns import aggregate_patterns, clean_fraction, detect_patterns
```

Add two fields to the `RunStats` dataclass (after `tool_usage`, before `reliability`):

```python
    pattern_frequency: dict[str, float] = field(default_factory=dict)  # name -> sample-avg rate
    clean_fraction: float = 0.0  # sample-avg fraction of samples with zero hits
```

(`field` is already imported in `loader.py`.)

- [ ] **Step 4: Populate the fields in `_compute_stats`**

In `_compute_stats`, change the `return RunStats(...)` call to pass the new fields. Add these two lines just before `return RunStats(`:

```python
    g = groups or {}
    pattern_frequency = aggregate_patterns(g)
    clean = clean_fraction(g)
```

and add to the `RunStats(...)` constructor args (after `tool_usage=tool_usage,`):

```python
        pattern_frequency=pattern_frequency,
        clean_fraction=clean,
```

- [ ] **Step 5: Attach per-record hits in `load_run`**

In `load_run`, the loop that sets `_error_class` reads:

```python
    for r in records:
        r["_error_class"] = classify_submit_error(r)
        r.setdefault("iteration", 0)
```

Add the hit attachment so the page can highlight without recomputing:

```python
    for r in records:
        r["_error_class"] = classify_submit_error(r)
        r["_pattern_hits"] = detect_patterns(r)
        r.setdefault("iteration", 0)
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/explorer/test_loader.py -v`
Expected: PASS (existing loader tests + the new one)

- [ ] **Step 7: Commit**

```bash
git add explorer/loader.py tests/explorer/test_loader.py
git commit -m "feat(explorer): expose anti-pattern frequency on RunStats"
```

---

### Task 5: Highlight offending messages in the conversation renderer

**Files:**
- Modify: `explorer/render.py`
- Modify: `tests/explorer/test_render.py`

`test_render.py` already provides a `_StubSt` (no-op streamlit stand-in, `__getattr__` returns a `_StubCtx`), a `_StubCtx` context manager, and a `patch_st` fixture that does `monkeypatch.setattr(render, "st", _StubSt())`. The stub records nothing, so the new test needs a *capturing* stub. Add a small capturing subclass and use it directly (not the `patch_st` fixture).

- [ ] **Step 1: Write the failing test**

Append to `tests/explorer/test_render.py` (reusing the file's existing `_StubSt`/`_StubCtx`):

```python
class _CapturingSt(_StubSt):
    """_StubSt that records markdown() text so assertions can inspect output."""

    def __init__(self):
        self.markdown_calls: list[str] = []

    def markdown(self, body="", *_args, **_kwargs):
        self.markdown_calls.append(str(body))
        return _StubCtx()


def test_render_marks_highlighted_messages(monkeypatch):
    import explorer.render as render

    cap = _CapturingSt()
    monkeypatch.setattr(render, "st", cap)
    record = {
        "instance_id": "i1",
        "messages": [{"role": "ai", "content": "hi", "tool_calls": []}],
    }
    render.render_conversation(record, highlight_indices={0})
    assert any("\U0001f6a9" in c for c in cap.markdown_calls)  # 🚩 marker rendered


def test_render_no_marker_without_highlight(monkeypatch):
    import explorer.render as render

    cap = _CapturingSt()
    monkeypatch.setattr(render, "st", cap)
    record = {"instance_id": "i1", "messages": [{"role": "ai", "content": "hi", "tool_calls": []}]}
    render.render_conversation(record)
    assert not any("\U0001f6a9" in c for c in cap.markdown_calls)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/explorer/test_render.py::test_render_marks_highlighted_messages -v`
Expected: FAIL — `render_conversation() got an unexpected keyword argument 'highlight_indices'`

- [ ] **Step 3: Add the `highlight_indices` parameter**

In `explorer/render.py`, change the signature:

```python
def render_conversation(record: dict, highlight_indices: set[int] | None = None) -> None:
```

Change the message loop to enumerate and mark highlighted turns. Replace:

```python
    for msg in record.get("messages", []):
        role = msg.get("role")
```

with:

```python
    highlight = highlight_indices or set()
    for i, msg in enumerate(record.get("messages", [])):
        role = msg.get("role")
        if i in highlight:
            st.markdown("\U0001f6a9 **flagged turn**")
```

(The marker renders just above whichever block the role branch produces.)

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/explorer/test_render.py -v`
Expected: PASS (existing render tests + the new one)

- [ ] **Step 5: Commit**

```bash
git add explorer/render.py tests/explorer/test_render.py
git commit -m "feat(explorer): optional message highlighting in conversation renderer"
```

---

### Task 6: Single-run anti-pattern diagnosis page

**Files:**
- Create: `explorer/pages/patterns.py`

This is a Streamlit page; it is verified by running the app (no unit test — all pure logic is already tested in Tasks 1-3).

- [ ] **Step 1: Create the page**

Create `explorer/pages/patterns.py`:

```python
"""Single-run tool-interaction anti-pattern diagnosis page."""

from __future__ import annotations

import os
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from loader import RunData, list_runs, load_run
from patterns import PATTERN_CATALOG, positional_tool_distribution
from render import render_conversation


RESULTS_ROOT = Path(os.environ.get("RESULTS_ROOT", str(Path(__file__).parent.parent.parent / "results")))

_LABELS = dict(PATTERN_CATALOG)  # name -> human label


@st.cache_data
def _load_run_cached(path_str: str) -> RunData:
    return load_run(Path(path_str))


def _question(r: dict) -> str:
    return r.get("amb_user_query") or r.get("not_ambiguos_query", "")


st.set_page_config(page_title="Tool Anti-Patterns", layout="wide")
st.title("Tool-Interaction Anti-Patterns")

# ── Sidebar: run selector (mirrors app.py) ──────────────────────────────────────
runs_tree = list_runs(RESULTS_ROOT)
if not runs_tree:
    st.error(f"No results found in `{RESULTS_ROOT.resolve()}`")
    st.stop()

with st.sidebar:
    st.header("Select Run")
    date = st.selectbox("Date", list(runs_tree.keys()))
    run_key = st.selectbox("Run", runs_tree[date], index=0)

run = _load_run_cached(str(RESULTS_ROOT / date / run_key))
stats = run.stats

# ── Headline: anti-pattern frequency (sample-average) ───────────────────────────
st.subheader("Anti-pattern frequency (sample-average)")
st.caption(f"Fraction of instances hitting each pattern · clean: {stats.clean_fraction * 100:.1f}%")
freq_df = pd.DataFrame(
    [{"Pattern": _LABELS[name], "Frequency": stats.pattern_frequency.get(name, 0.0)}
     for name, _ in PATTERN_CATALOG]
).sort_values("Frequency", ascending=False)
st.altair_chart(
    alt.Chart(freq_df).mark_bar().encode(
        x=alt.X("Frequency:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
        y=alt.Y("Pattern:N", sort="-x"),
    ).properties(height=260),
    use_container_width=True,
)

# ── Positional tool distribution (100%-stacked) ─────────────────────────────────
st.subheader("Tool by call position")
pos_df = positional_tool_distribution(run.groups)
if not pos_df.empty:
    pos_order = [str(p) for p in range(1, 8)] + ["≥8"]
    st.altair_chart(
        alt.Chart(pos_df).mark_bar().encode(
            x=alt.X("position:N", sort=pos_order, title="Call position"),
            y=alt.Y("share:Q", stack="normalize", axis=alt.Axis(format="%"), title="Share"),
            color=alt.Color("tool:N", title="Tool"),
            order=alt.Order("tool:N"),
        ).properties(height=320),
        use_container_width=True,
    )

# ── Drill-down: pick a pattern, list flagged tasks, inspect ─────────────────────
st.divider()
st.subheader("Drill-down")
pattern_label = st.selectbox(
    "Pattern", [lbl for _, lbl in PATTERN_CATALOG],
    format_func=lambda lbl: lbl,
)
pattern_name = next(name for name, lbl in PATTERN_CATALOG if lbl == pattern_label)

flagged = [
    r for r in run.records
    if any(h.name == pattern_name for h in r.get("_pattern_hits", []))
]
if not flagged:
    st.info("No records hit this pattern.")
    st.stop()

rows = []
for i, r in enumerate(flagged):
    hit = next(h for h in r["_pattern_hits"] if h.name == pattern_name)
    q = _question(r)
    rows.append({
        "#": i + 1,
        "instance_id": r.get("instance_id", ""),
        "iteration": r.get("iteration", 0),
        "database": r.get("selected_database", ""),
        "Question": (q[:70] + "…") if len(q) > 70 else q,
        "Evidence": hit.detail,
        "Accuracy": "✓" if r.get("execution_accuracy") else "✗",
    })
event = st.dataframe(
    pd.DataFrame(rows), use_container_width=True, hide_index=True,
    on_select="rerun", selection_mode="single-row",
)
selected = event.selection.rows
if not selected:
    st.stop()

record = flagged[selected[0]]
hit = next(h for h in record["_pattern_hits"] if h.name == pattern_name)
st.divider()
st.warning(f"**{pattern_label}** — {hit.detail}  ·  flagged messages: {hit.message_indices}")
render_conversation(record, highlight_indices=set(hit.message_indices))
```

- [ ] **Step 2: Verify the page renders**

Run (background it, then stop after checking):

```bash
RESULTS_ROOT=results uv run streamlit run explorer/app.py --server.headless true
```

In the browser sidebar, open the **Tool Anti-Patterns** page. Confirm: the frequency bar chart, the positional stacked bar (with a grey `(no call)` band at later positions), the pattern selector, the flagged-task table, and that selecting a task renders the conversation with 🚩 markers on the flagged turns. Stop the server when done.

- [ ] **Step 3: Commit**

```bash
git add explorer/pages/patterns.py
git commit -m "feat(explorer): single-run anti-pattern diagnosis page"
```

---

### Task 7: Cross-run anti-pattern frequency table

**Files:**
- Modify: `explorer/pages/compare.py`

- [ ] **Step 1: Add the import**

In `explorer/pages/compare.py`, after the existing `from loader import ...` lines, add:

```python
from patterns import PATTERN_CATALOG
```

- [ ] **Step 2: Add the frequency table block**

In `compare.py`, locate the Tool Usage block (the `if has_tools:` section that renders per-run `st.bar_chart`). Immediately **after** that block and **before** the `# ── Aptitude / Unreliability box plot` comment, insert:

```python
# ── Anti-pattern frequency (sample-average) ─────────────────────────────────────
st.divider()
st.subheader("Tool-interaction anti-patterns")
st.caption("Sample-average % of instances hitting each pattern. Red = worst (highest) per row.")

_pat_labels = dict(PATTERN_CATALOG)
pat_raw = pd.DataFrame(
    {label: {_pat_labels[name]: run.stats.pattern_frequency.get(name, 0.0)
             for name, _ in PATTERN_CATALOG}
     for label, run in runs.items()}
)
pat_display = pat_raw.applymap(lambda v: f"{v * 100:.1f}%")


def _highlight_worst(row: pd.Series) -> list[str]:
    if len(runs) < 2:
        return [""] * len(row)
    numeric = pat_raw.loc[str(row.name)]
    worst = numeric.max()  # higher anti-pattern rate is worse
    return ["background-color: rgba(220, 0, 0, 0.18)" if numeric[c] == worst and worst > 0 else ""
            for c in row.index]


st.dataframe(
    pat_display.style.apply(_highlight_worst, axis=1),
    use_container_width=True,
)
```

- [ ] **Step 3: Verify the comparison page renders**

Run:

```bash
RESULTS_ROOT=results uv run streamlit run explorer/app.py --server.headless true
```

Open the **Run Comparison** page, select ≥2 runs, and confirm the new "Tool-interaction anti-patterns" table appears with one row per pattern, one column per run, and red highlighting on the worst cell per row. Stop the server when done.

- [ ] **Step 4: Commit**

```bash
git add explorer/pages/compare.py
git commit -m "feat(explorer): cross-run anti-pattern frequency table"
```

---

### Task 8: Full suite + type check

**Files:** none (verification only)

- [ ] **Step 1: Run the full test suite**

Run: `uv run pytest tests/`
Expected: all tests pass (per CLAUDE.md, run after applying changes).

- [ ] **Step 2: Type-check**

Run: `uv run pyrefly check`
Expected: no new errors in `explorer/patterns.py`, `explorer/loader.py`, `explorer/render.py`.

- [ ] **Step 3: Commit any fixes**

If steps 1-2 surfaced fixes, commit them:

```bash
git add -A
git commit -m "fix(explorer): address test/type-check findings for anti-pattern analysis"
```

(Per repo convention, do not `git commit -a` blindly — stage only files this plan touched.)

---

## Self-Review Notes

- **Spec coverage:** §3 module layout → Tasks 1-7. §4 contract (`ToolEvent`, `extract_tool_events`, `PatternHit`, 7-detector catalog, `detect_patterns`, `aggregate_patterns`, `positional_tool_distribution`) → Tasks 1-3. §5 loader integration (`pattern_frequency`, `clean_fraction`, `_pattern_hits`) → Task 4. §6 single-run page + positional plot → Tasks 5-6; compare table → Task 7. §7 testing → Tasks 1-4 tests. §8 extensibility preserved (add a detector fn + a `PATTERN_CATALOG` row).
- **Sample-average invariant** locked by `test_aggregate_weights_instances_equally` (asserts 0.5, not pooled 0.667) and `test_compute_stats_pattern_frequency_sample_average`.
- **Type consistency:** `pattern_frequency: dict[str, float]` and `clean_fraction: float` are used identically in `loader.py`, `pages/patterns.py`, and `pages/compare.py`. `positional_tool_distribution` returns columns `position, tool, share` consumed by both the test and the page. `render_conversation(record, highlight_indices=None)` signature matches its sole new caller.
- **Known limitation (per spec, accepted):** `submit_after_error` uses whitespace-collapsed exact match; a one-char edit won't fire. `budget_death` and `no_submission` are independent flags and can co-fire on one record (intentional).
