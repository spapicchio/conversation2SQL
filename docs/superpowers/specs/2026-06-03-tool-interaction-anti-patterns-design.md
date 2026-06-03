# Tool-Interaction Anti-Pattern Analysis — Design Spec

**Date:** 2026-06-03
**Branch:** feat/resume-failed-instances (current)
**Goal:** Add script-only (no LLM judge) analysis to the `explorer/` Streamlit app that surfaces recurring **broken tool-use patterns** in BIRD-Interact result traces, so they can be exploited to improve the agent harness or the model.

---

## 1. Problem & Goal

The `tools_only` baseline lets the agent call DB/KB tools (`get_schema`, `execute_sql`,
`get_knowledge_definition`, …) and `submit_sql` under a patience budget. We want to
understand **how the agent interacts with these tools** — specifically to find common
*broken patterns* (blind submissions, repeated identical calls, error loops, …) that
are actionable: fix the harness prompt, the tool design, or target the model's weakness.

Constraint from the user: **everything is done with deterministic scripts. No LLM-as-judge.**
This supersedes the earlier `2026-05-07-turn-classifier-design.md`, whose semantic
"Level 1" layer required an LLM. The deterministic "Level 2" idea (tool-action categories
derived from `tool_calls`) survives in spirit here, recast as a per-conversation
anti-pattern catalog rather than per-turn labels.

The analysis serves **two equally-weighted uses**:
- **Single-run diagnosis:** deeply analyze one selected run.
- **Cross-run comparison:** compare anti-pattern frequencies across runs (e.g. before/after
  a harness tweak) to see whether a change reduced broken patterns.

---

## 2. Data available (and the one gotcha)

Each result record (`results_iter*.jsonl`) already carries:
- `tool_calls_in_order`: ordered list of `{tool_name, arguments, tool_cost}` — names, args,
  costs, but **NOT** per-call success/failure.
- `messages`: full trace. Each `role=="tool"` message has `tool_name`, `status`
  (`"success"` / error), and `content` (result text). Multiple `tool` messages can follow
  one `ai` turn.
- Context fields for detectors: `masked_agent_kb` / `gt_knowledge_base` (KB presence),
  `predicted_sql`, `execution_accuracy`, `updated_user_patience`, `task_budget`,
  `instance_id`, `selected_database`.

**Gotcha:** per-call success lives in `messages`, not `tool_calls_in_order`. So `patterns.py`
first builds one **normalized tool-event sequence** by walking `messages` and pairing each
`ai` tool_call with its following `tool` result. Every detector reads that sequence — never
the raw messages directly.

---

## 3. Module layout

Everything stays in `explorer/` — pure functions on a result `record`, no `src/` library
code, no `turn_classifier` package, no LLM. (Detectors are pure → trivially unit-testable,
and could be lifted into `src/` later if a CLI is ever wanted. YAGNI for now.)

```
explorer/
  patterns.py        # NEW — tool-event extraction, detector registry, aggregation (no Streamlit)
  loader.py          # MODIFY — attach pattern hits per record + aggregate into RunStats
  pages/
    patterns.py      # NEW — single-run diagnosis page
    compare.py       # MODIFY — add per-run anti-pattern frequency table

tests/
  explorer/
    test_patterns.py # NEW — one positive + one negative fixture per detector
```

---

## 4. `patterns.py` — contract

### Tool-event extraction

```python
@dataclass(frozen=True)
class ToolEvent:
    message_index: int        # index into record["messages"] of the tool-result message
    tool_name: str
    arguments: dict
    status: str               # "success" or an error status
    is_error: bool            # status != "success"
    result_text: str          # tool message content as text

def extract_tool_events(record: dict) -> list[ToolEvent]:
    """Walk messages, pair each ai tool_call with its following tool result(s), in order."""
```

Pairing rule: iterate messages in order; when an `ai` message has `tool_calls`, the
immediately-following `tool` messages (one or more, back-to-back) are its results, matched
by `tool_name` and order. Robust to back-to-back `tool` messages.

### Detector contract

```python
@dataclass(frozen=True)
class PatternHit:
    name: str                   # "blind_submit"
    label: str                  # "Blind submit"
    detail: str                 # human evidence, e.g. "submitted without any execute_sql"
    message_indices: list[int]  # offending messages, for drill-down highlighting

Detector = Callable[[list[ToolEvent], dict], PatternHit | None]

ANTI_PATTERNS: list[Detector] = [...]   # extensible registry — add a function, done

def detect_patterns(record: dict) -> list[PatternHit]:
    events = extract_tool_events(record)
    return [hit for d in ANTI_PATTERNS if (hit := d(events, record))]
```

### Sample-average aggregation (the core invariant)

Every aggregate is computed **per-instance first, then averaged across instances**, taking
the `groups` dict (`instance_id → [samples]`) that `load_run` already builds — exactly like
`pass_at_1` in `loader._compute_stats` and `reliability_metrics` in `metrics.py`. With a
single iteration this collapses to a plain fraction (today's behavior is unchanged), but
when multiple iterations exist it prevents instances with more samples from being
over-weighted, and it lines up with the Aptitude/Unreliability machinery consuming the same
`groups`. **No aggregate consumes a flat `records` list.**

```python
def aggregate_patterns(groups: dict[str, list[dict]]) -> dict[str, float]:
    """name -> mean over instances of (fraction of that instance's samples hitting it).

    Per-record a pattern counts at most once (presence, not occurrence count)."""

def positional_tool_distribution(
    groups: dict[str, list[dict]], max_pos: int = 8
) -> "pd.DataFrame":
    """Long-form frame: position (1..max_pos, with a '>=max_pos' tail bucket), tool, share.

    For each sample, the ordered tool sequence (including the terminal submit_sql) gives the
    tool at each position; positions past the sample's last call contribute to a synthetic
    '(no call)' tool. Shares are aggregated sample-average style: mean over instances of the
    per-instance mean share. Denominator is constant across positions (all instances), so the
    '(no call)' band grows with position and shows conversation attrition."""
```

### Initial catalog (all deterministic, script-only)

| Name | Label | Fires when |
|---|---|---|
| `blind_submit` | Blind submit | a `submit_sql` event with no prior `execute_sql` in the run |
| `repeated_identical_call` | Repeated identical call | same `(tool_name, normalized-args)` called ≥2× |
| `submit_after_error` | Submit after error | submitted SQL is byte-identical to an `execute_sql` whose result errored |
| `unrecovered_error_loop` | Unrecovered error loop | ≥3 consecutive `execute_sql` errors with no success between |
| `kb_blind` | KB-blind | record has KB entries but zero `get_knowledge_definition` calls |
| `budget_death` | Budget death | budget exhausted / never reached a successful `submit_sql` |
| `no_submission` | No submission | conversation ends with no `submit_sql` call at all |

Normalization for `repeated_identical_call`: `json.dumps(arguments, sort_keys=True)` after
stripping whitespace from string values; SQL args compared on whitespace-collapsed text.

---

## 5. `loader.py` integration

- In `load_run`, after `classify_submit_error`, attach `r["_pattern_hits"] = detect_patterns(r)`
  to each record (mirrors the existing `r["_error_class"]` pattern).
- Add to `RunStats`: `pattern_frequency: dict[str, float]` (via `aggregate_patterns(groups)`,
  sample-average) and `clean_fraction: float` (sample-average fraction of samples with zero
  hits). `_compute_stats` already receives `groups` — reuse it; do not aggregate over the flat
  `records` list.
- The positional plot frame is computed in the page from `run.groups` (not stored on
  `RunStats`), since it is page-specific and `groups` is already on `RunData`.

---

## 6. UI

### Single-run page (`pages/patterns.py`)

- **Headline bar chart:** sample-average frequency of each anti-pattern (sorted desc, as a
  %), from `stats.pattern_frequency`.
- **Positional tool plot:** 100%-stacked bar — x-axis = call position (`1`..`8`, then a
  `≥8` tail bucket), color = tool name (including terminal `submit_sql` and a grey
  `(no call)` band for instances that already ended), height = sample-average share. Built
  from `positional_tool_distribution(run.groups)`. Shows the typical workflow shape and when
  submission/attrition happens.
- **Cleanliness histogram:** how many records are clean vs hit 1 / 2 / 3+ patterns.
- **Drill-down:** select a pattern → filtered task table (reusing the existing dataframe
  table style from `app.py`) → select a task → `render_conversation`, with the hit's
  `message_indices` visually highlighted.

### Compare page (`compare.py`, extended)

- One row per anti-pattern, one column per selected run; cells = **sample-average %** hitting
  it (from each run's `stats.pattern_frequency`). So a tweak that drops
  `repeated_identical_call` 40%→5% is visible at a glance.
- (Sample-average % is the headline; absolute count not required per user.)

---

## 7. Testing (`tests/explorer/test_patterns.py`)

- Hand-built tiny `messages` fixtures — one positive and one negative per detector.
- A dedicated test for `extract_tool_events` covering back-to-back `tool` messages and an
  `ai` turn with multiple tool_calls.
- `aggregate_patterns` / `positional_tool_distribution`: a multi-sample `groups` fixture
  (one instance with 2 samples, one hitting a pattern) asserting the result is the
  per-instance mean (0.5), not the pooled fraction — locks in the sample-average invariant.
- A constant-denominator check on the positional frame: shares at each position (including
  `(no call)`) sum to ~1.0.
- No fixtures larger than a few messages; no LLM, no DB.
- Run via `uv run pytest tests/explorer/test_patterns.py`.

---

## 8. Extensibility

- **New anti-pattern:** add one detector function to `ANTI_PATTERNS`. No other code changes.
- **Tune a threshold** (e.g. error-loop length): edit the detector; keep it a module constant.
- Detectors stay pure and Streamlit-free so they remain unit-testable and portable.
