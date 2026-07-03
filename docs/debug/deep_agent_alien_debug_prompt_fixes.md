# deep_agent / alien debug set: root-cause analysis + prompt fixes

**Date:** 2026-07-01
**Baseline:** `deep_agent`, model `Qwen/Qwen3.5-9B` (hosted_vllm, temperature 0.6), variant
`all_db_all_kb_linearized`, `--debug` (first 10 dataset rows — always the 10 `alien_*`
"Query"-category tasks, `dataset[:10]` in `main_pipe_workflow.py`).

**Runs referenced:**
- `results/2026-07-01/10-37-13/deep_agent__Qwen3.5-9B__ddl__lin__iter1__debug` — original, 6/10
- `results/2026_07_01/12_00_40__…__debug` — prompt v1, 3/10
- `results/2026_07_01/12_22_41__…__debug` — prompt v2, 4/10
- `results/2026_07_01/12_43_54__…__debug` — prompt v3, 4/10

Goal (user-set): get all 10 `alien_*` tasks passing. Not yet reached when this note was
written — see "State at handoff" below.

## TL;DR

1. Root-caused the original run's 4 failures (`alien_4/6/8/9`) by diffing `predicted_sql`
   against `sol_sql` and re-reading the actual on-disk KB catalog (not just inferring from
   agent behavior). All four were **KB comprehension / query-shape** failures, not budget
   or tool bugs.
2. Iteratively patched `deep_agent/prompts.py`'s system prompt (3 rounds) targeting those
   patterns. Pass rate did **not** monotonically improve run-over-run (6 → 3 → 4 → 4) —
   temperature 0.6 sampling variance is large enough on a 10-task set that single-run
   deltas are not reliable evidence for or against a prompt change.
3. Independent of prompt tuning, found a **deterministic, high-confidence bug**: the
   grading harness's `remove_round()` (`bird_baseline/tools/utils.py`) strips the textual
   `ROUND(...)` wrapper but not an inner `::numeric` cast — and Postgres's two-argument
   `ROUND(numeric, int)` *requires* that cast for a `double precision` expression. So any
   model that (reasonably) writes `ROUND(expr::numeric, n)` gets compared, post-stripping,
   as `expr::numeric` against the reference's plain `expr` (double precision) — different
   arithmetic path, different low-order decimal digits, exact-match fails even though the
   displayed rounded value is identical. Verified directly against the live DB (see below).
   This explained 3 of the 6 failures in the last observed run.
4. Also found one plain Postgres syntax bug (SELECT-list alias referenced in the same
   query's WHERE clause — illegal, but a model-generated pattern) and one likely
   **dataset/KB authoring inconsistency** (`alien_4`'s gold SQL omits a criterion — `InfoDense
   > 0.8` — that the KB entry it's drawn from explicitly requires).

## Root causes found (original run, 6/10)

Investigated via `tool_calls_in_order`, `predicted_sql` vs `sol_sql` in the results JSONL,
plus reading the actual catalog files under
`data/bird_interact/catalog_bird_interact_lite/alien/`.

- **`alien_6`** (RPI/CCS by observatory): computed CCS with an extra `ClassConf` factor
  not in `sol_sql`, then **guessed the "high confidence" threshold 7 times** via
  `submit_sql` (0.5, 0.3, 1, 0.2, 0.8, 0.7, 0.6 …) — 12 `submit_sql` calls total, ending on
  budget exhaustion. The correct formula (`ccs_approximation.md`, no ClassConf) and the
  exact threshold (`high_confidence_signals.md`: "CCS > 0.8") were both listed by name in
  `database_overview.md`'s Knowledge Base index (confirmed present in `masked_agent_kb`,
  i.e. not masked out) — the agent never opened either file, despite the task question
  literally containing the word "approximate".
- **`alien_8`** (NTM signal classification): final submitted query didn't even include a
  classification column (returned raw `signaldynamics.sigcoherence`). Trace shows dozens
  of near-byte-identical repeated `psql` calls and 12 `submit_sql` attempts cycling through
  different coherence-band cutoffs — never converged.
- **`alien_9`** (Observation Quality Factor / OCL ranking): enumerated *every* telescope
  status column found (`focusquality`, `coolsysstatus`, `powerstatus`, `datastorstatus`,
  `netstatus`, `procqueuestatus`) as "equipment issues", vs. `sol_sql`'s 3 KB-defined ones
  (`EquipStatus`, `CalibrStatus`, `CoolSysStatus`). Also collapsed to one row per
  observatory via `ROW_NUMBER() PARTITION BY observstation`, while `sol_sql` returns one
  row per telescope — a structural shape mismatch.
- **`alien_4`** (technosignature detection): added an unrequested filter
  `sc.infodense > 0.8` (with its own extra join), not present in `sol_sql`. *Later
  determined this filter is actually correct per the KB's own "Technosignature" definition
  — see "Likely dataset inconsistency" below.*

## Prompt changes applied (`deep_agent/prompts.py`, `_DEEP_AGENT_SYSTEM`)

Three rounds, cumulative, all just added strategy-tip bullets (system prompt structure/
tool descriptions untouched; `uv run pytest tests/eval_framework/agents/deep_agent/` kept
passing after each edit):

1. **v1** — after the `alien_6`/`alien_8` root-cause: "scan the whole KB index for
   approximation/threshold/classification variants before computing a formula"; "only
   select columns/joins/filters explicitly asked for"; "don't repeat identical psql
   commands". Result: **3/10** (down from 6) — the "re-scan the whole index" wording was
   too broad, driving 29-31 `bash` calls per task even on simple questions and burning
   budget before verification.
2. **v2** — tightened the KB-scan tip to be targeted ("only read entries related to THIS
   question"), added aggregation-scope guidance (use `COUNT(*) FILTER (WHERE …)` instead of
   a CTE-level filter that also restricts other aggregates), added "Postgres has no
   `MEDIAN()` — use `PERCENTILE_CONT`". Result: **4/10**. Confirmed via direct psql
   comparison: `submit_sql` guess-and-check dropped from 7-12 calls/task to 1-2; `alien_6`'s
   RPI/CCS formula now matches `sol_sql` byte-for-byte (previously wrong); `alien_8`
   started attempting a real classification column (previously just returned a raw value)
   — real qualitative progress masked by the pass-count staying flat, because *new*
   failures appeared elsewhere in the same run.
3. **v3** — added: (a) don't add a JOIN you don't reference in SELECT/WHERE/GROUP BY (an
   unused INNER JOIN can silently drop unmatched rows, corrupting COUNT/AVG — this is what
   was still happening in `alien_6`'s v2 output: a stray `JOIN signalclassification`); (b)
   when multiple KB entries have similar names, match by the literal noun phrase in the
   question, don't take the first plausible hit (targets `alien_8`'s confusion between
   `signaldynamics.sigcoherence` and the actual answer, `ntm_classification_system.md`,
   both semantically "coherence"-adjacent). Result: still **4/10** — but for a *different*
   reason discovered right after this run (see next section), not because these two fixes
   failed.

## The ROUND/`::numeric` finding (verified independently of any prompt change)

After the v3 run, diffed remaining failures against `sol_sql` by executing both directly
against the live `alien` Postgres DB (`localhost:5432`, matching `db_dsn_template`).
`alien_5`'s predicted/gold numeric columns looked identical to 2 decimal places, which
shouldn't fail the harness's set-equality check
(`bird_baseline/tools/bird_interact_user_tools.py:submit_sql_impl`, `set(pred_result) ==
set(target_result)`, after `remove_round()` strips `ROUND(...)` textually).

Direct test on the live DB:

```sql
WITH mcs_calc AS (...)
SELECT modtype, AVG(mcs) AS avg_plain, AVG(mcs::numeric) AS avg_cast,
       AVG(mcs) = AVG(mcs::numeric)::double precision AS equal_check
FROM mcs_calc GROUP BY modtype;
--  QAM: 3196.644985032344   vs 3196.6449850323449685   -> f
--  FM : 3131.386870654527   vs 3131.3868706545240588   -> f
--  AM : 10838.400502914363  vs  10838.40050291436897   -> f
--  PM : 2790.4245244948156  vs 2790.4245244948157460   -> t (coincidence)
```

`AVG(x::numeric)` and `AVG(x)` differ at low decimal digits for a `double precision`
column — `remove_round("ROUND(AVG(mcs::numeric), 2)")` yields `AVG(mcs::numeric)`, which
is compared against the gold query's plain `AVG(mcs)`. Cross-checked against the last run's
predicted SQL: **every currently-passing sample either has no `ROUND` at all, or uses
`ROUND` without an inner `::numeric` cast; every failing sample that uses
`ROUND(expr::numeric, n)` was failing** (`alien_4`, `alien_5`, `alien_6` in the v3 run).

Added as a v3.1 prompt rule: never `ROUND(...)` or `::numeric`-cast a float/aggregate
expression for display; leave it unrounded/uncast unless the column's own type forces a
cast (e.g. integer division). Not yet verified with a live rerun (see below).

## Other bugs found (not prompt-fixable / not yet addressed)

- **`alien_10`**: predicted SQL referenced a SELECT-list alias
  (`cip_classification_label`) inside that same query's `WHERE` clause — illegal in
  Postgres (aliases don't exist yet when WHERE is evaluated), so the query errored outright.
  Added a v3.1 prompt rule for this specific pattern (repeat the expression or wrap in a
  subquery/CTE instead).
- **`alien_4`, likely dataset inconsistency**: `knowledge_base/technosignature.md` defines
  Technosignature as `TechSigProb > 0.7 AND NatSrcProb < 0.3 AND ArtSrcProb < 50 AND BFR <
  0.001 AND InfoDense > 0.8` (5 conditions) — but `sol_sql` (the graded reference) only
  applies the first 4, omitting `InfoDense > 0.8` entirely. An agent that faithfully reads
  the KB (as the prompt now explicitly tells it to) produces a *stricter*, KB-correct
  filter that will never match this particular gold query. This looks like a benchmark
  authoring bug, not something fixable from the agent side without telling it to
  deliberately ignore part of the KB it just read — flagging rather than "fixing" by
  overfitting to this one instance.

## State at handoff

- `prompts.py` currently has all v1–v3 + the ROUND/`::numeric` and WHERE-alias rules
  applied; `uv run pytest tests/eval_framework/agents/deep_agent/` passes (43 tests).
- **Not yet re-run against the live model** — a 4th eval run (`goalrun4`) was about to be
  launched (GPU 0, same command as prior runs) when this summary was requested instead:
  ```bash
  MODEL=qwen35 VARIANT=all_db_all_kb_linearized BASELINE=deep_agent \
    CUDA_VISIBLE_DEVICES=0 DEBUG=true CONCURRENCY=16 NUM_ITERATIONS=1 \
    bash bash_scripts/eval_payload.sh
  ```
- Expectation for that run: `alien_5`/`alien_6` should improve (ROUND/cast fix is
  deterministic and directly verified against the DB), `alien_10` should improve (WHERE-alias
  syntax bug is unambiguous), `alien_9` unverified against the newest prompt, `alien_4`
  likely still fails (dataset inconsistency, not agent-fixable), `alien_8` improved in kind
  (real classification attempted) but not yet confirmed correct.
- Given temperature 0.6, a single 10-task run is noisy evidence in either direction; the
  ROUND/`::numeric` and WHERE-alias fixes are the two changes in this note with
  *deterministic, DB-verified* justification independent of any particular sampled run.
