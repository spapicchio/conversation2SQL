# Explorer reliability metrics (pass@k, aptitude, unreliability)

**Date:** 2026-05-29
**Component:** `explorer/loader.py`, `explorer/metrics.py` (new), `explorer/app.py`,
`explorer/pages/compare.py`
**Related:** [multi-iteration eval design](2026-05-29-multi-iteration-eval-concurrency-design.md)

## Problem

The multi-iteration pipeline now runs the dataset N times for one variant and writes one
file per iteration (`results_iter{i}.jsonl`), each record tagged with an `iteration` field.
The explorer app, however, still globs a single `results.jsonl` and computes one aggregate
execution accuracy (`n_passed / n_total`). It has no concept of iterations, so it cannot
load multi-iteration runs at all, and it offers no measure of how *reliably* a model solves
each instruction across repeated samples.

We want the explorer to (a) load and group the per-iteration files, and (b) report
reliability metrics adapted from *"LLMs Get Lost in Multi-Turn Conversation"*
(arXiv:2505.06120) plus the pass@k estimator (Chen et al. 2021).

## Scope

- **In scope:** teach the loader to discover/read/group `results_iter*.jsonl`; a new
  `explorer/metrics.py` with the metric math; reliability metrics + pass@k on the main page
  and the compare page; tests.
- **Out of scope:** any change to the eval pipeline (it already emits the per-iteration
  files). No new dataset reads.
- **Out of scope:** seed plumbing or generating new runs.

## Metric definitions

Per-sample score `Sᵢ ∈ {0, 1}` taken from each record's `execution_accuracy`. For an
instruction with sample scores `g = [S₁ … Sₙ]` (n = that instance's sample count):

- **Average** `P̄` = mean over instances of `mean(g)`.
- **Aptitude** `A⁹⁰` = mean over instances of `percentile(g, 90)`.
- **Unreliability** `U₁₀⁹⁰` = mean over instances of `percentile(g, 90) − percentile(g, 10)`.
- **Reliability** `R = 1 − U`. (Scores are on a 0–1 scale, not the paper's 0–100, so
  `R = 1 − U`, not `100 − U`.)
- **pass@k** (unbiased Chen et al. 2021 estimator): for an instance with `n` samples and
  `c` passes,

  ```
  pass@k_instance = 1 − C(n − c, k) / C(n, k)      (= 1 when n − c < k)
  ```

  averaged across instances with `n ≥ k`. Reported as a sweep `{k: value}` for
  `k = 1 … max(nᵢ)`. `pass@1` equals `P̄` numerically.

**Decisions (confirmed):**
- Aptitude/Unreliability use the **paper-faithful** per-instance-percentile-then-average
  formulation (not a per-instance-pass-rate distribution).
- pass@k is the **full sweep** `1 … N`, rendered as a curve.
- Percentile convention: `numpy.percentile` default (linear interpolation).

**Caveat (documented in `metrics.py` docstring):** with binary 0/1 scores and small N, a
per-instance 90th/10th percentile collapses to 0 or 1, so Aptitude/Unreliability are coarse
(near-degenerate) for small N. This is inherent to applying the paper's graded-score
formulas to a binary score and is accepted.

## Design

### 1. `explorer/metrics.py` (new) — pure math

Framework-free, no Streamlit, no I/O. Operates on already-grouped data so it is unit-testable
in isolation.

```python
@dataclass
class ReliabilityStats:
    avg_performance: float        # P̄
    aptitude: float               # A⁹⁰
    unreliability: float          # U₁₀⁹⁰
    reliability: float            # 1 − U
    passk: dict[int, float]       # {k: pass@k}, k = 1..max_n

def _scores(group: list[dict]) -> list[float]:
    return [1.0 if r.get("execution_accuracy") else 0.0 for r in group]

def passk_curve(groups: dict[str, list[dict]]) -> dict[int, float]: ...
def reliability_metrics(groups: dict[str, list[dict]]) -> ReliabilityStats: ...
```

- `passk_curve` uses the unbiased estimator above. To avoid float overflow on the binomial
  ratio it computes `C(n−c, k)/C(n, k)` as a running product
  `∏_{i=0}^{k-1} (n−c−i)/(n−i)` (the standard numerically-stable form).
- Empty `groups` → zeros and `{}`. A group with `n < k` is skipped for that `k` (so `passk`
  only contains `k` values reachable by at least one instance).

### 2. `explorer/loader.py` — discovery & grouping

- `_has_results(d)` also returns True when `list(d.glob("results_iter*.jsonl"))` is non-empty.
- `load_run(path)`:
  - If `results_iter*.jsonl` files exist, read **all** of them; each record retains its
    `iteration` field (default to `0` if absent).
  - Else fall back to today's `results_smaller.jsonl` → `results.jsonl`, treating the single
    file as `iteration = 0`.
  - Build `groups: dict[instance_id, list[dict]]`, each group sorted by `iteration`.
  - `n_iterations` = number of distinct `iteration` values observed.
  - `malformed_count` sums malformed lines across all files read; `source_file` becomes a
    short descriptor (e.g. `"results_iter*.jsonl (N files)"`).
- `RunData` gains `groups: dict[str, list[dict]]` and `n_iterations: int`.
- `RunStats` gains a `reliability: ReliabilityStats` field, populated by `_compute_stats`
  (which now also receives `groups`). Existing fields (`n_passed`, `accuracy_by_database`,
  etc.) keep their current micro-average semantics over the flat `records` list, so existing
  views are unaffected.

`records` remains the flat list of **all** samples across iterations — existing filters,
search, and the conversation viewer keep working without change.

### 3. `explorer/app.py` — main page

- New **"Reliability (N iterations)"** section, shown only when `run.n_iterations >= 2`
  (else a one-line note: single-iteration run, reliability not applicable):
  - Metric cards: **Average P̄**, **Aptitude A⁹⁰**, **Unreliability U₁₀⁹⁰**,
    **Reliability R** — formatted as percentages.
  - **pass@k line chart** (Altair): x = `k` (1..N), y = pass@k, y-domain `[0, 1]`.
- Existing 5 cards (Accuracy, avg tokens, avg cost, avg budget) unchanged; "Accuracy"
  stays the micro-average over all samples.
- **Task table:** when `n_iterations >= 2`, switch to **one row per instance** with a
  `pass_rate` column (`c/N`) instead of the single ✓/✗. The Pass/Fail filter becomes:
  *Passed* = `c == N`, *Failed* = `c == 0`; partial rows appear only under *All*.
  Clicking a row opens the conversation viewer with an **iteration selectbox**
  (default iteration 0). When `n_iterations == 1`, the table is exactly as today.

### 4. `explorer/pages/compare.py` — compare page

- Per-run stats columns gain **P̄, Aptitude, Unreliability, Reliability, and pass@N**
  metrics, shown when that run has `n_iterations >= 2` (otherwise only the existing
  accuracy metric).
- `join_runs` cells become `c/N` pass-count strings (with `n_iterations == 1` rendering as
  `1/1` / `0/1`, equivalent to today's ✓/✗). Agreement filters generalize:
  *All Passed* = `c == N` in every run, *All Failed* = `c == 0` in every run,
  *Disagreement* = differing pass-rates across runs. Task-absent stays `—`.
- Conversation viewer panes get the same iteration selectbox.

### 5. Error handling

- Malformed lines in any iteration file are counted, not fatal (as today).
- A run with mixed presence (some instances missing in some iterations) is handled by
  per-instance sample counts; no crash, metrics computed on available samples.

## Testing

- **`tests/explorer/test_metrics.py` (new):**
  - pass@k hand-checked: e.g. n=10, c=3 → pass@1 ≈ 0.3, pass@10 = 1.0; verify a mid-k value
    against the closed-form estimator.
  - aptitude/unreliability on crafted groups (e.g. mix of all-pass, all-fail, flaky
    instances); reliability = 1 − unreliability.
  - degenerate inputs: empty groups → zeros/`{}`; single-iteration groups.
- **`tests/explorer/test_loader.py` (extend):**
  - multiple `results_iter*.jsonl` → correct `groups`, `n_iterations`, malformed counting
    across files.
  - old single `results.jsonl` / `results_smaller.jsonl` still loads as `n_iterations == 1`.
  - `_has_results` / `list_runs` recognize iteration files.
  - update `join_runs` tests for the `c/N` cell format and generalized agreement filters.
- `uv run pytest tests/` green (per CLAUDE.md).

## Risks / notes

- **`join_runs` cell format change** (`c/N` instead of ✓/✗) is a breaking change for its
  existing tests; they are updated in the same change.
- **Binary-score coarseness** of aptitude/unreliability (above) is accepted and documented.
- No multi-iteration runs exist on disk yet; tests synthesize `results_iter*.jsonl`
  fixtures. First real validation comes from a future `num_iterations > 1` pipeline run.
