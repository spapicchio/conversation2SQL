# psql_console ablation: token consumption and failure root cause

**Date:** 2026-06-11
**Runs compared (9 iterations each, 189 instances, Qwen3.5-9B, ddl + linearized KB):**
- psql: `results/2026-06-09/19-31-55/tools_only__Qwen3.5-9B__ddl__lin__iter9__psql`
- baseline (typed DB tools): `results/2026-06-08/21-52-35/tools_only__Qwen3.5-9B__ddl__lin__iter9__error_new_tools_no_limitations`

Aggregated with the explorer's own `load_run` (`explorer/loader.py`), same stats the
compare page shows.

## TL;DR

1. **psql *does* consume more tokens — +47% total input** (133k vs 91k per conversation).
   It only looks flat *per call* (7,003 vs 6,572 input tokens/call) because each
   `psql_console` payload is small while the conversation is ~36% longer (18.0 vs 13.3
   model calls).
2. **Where psql fails and the baseline passes, the root cause is KB-blindness**: the
   agent tunnels into the cheap console (`\d`, trial SELECTs), never calls
   `get_knowledge_definition`, and submits an invented formula → "Wrong SQL". It is not
   budget death, syntax errors, or truncation.
3. Overall accuracy is a wash: pass@1 32.7% (psql) vs 33.5% (baseline); disagreement is
   roughly symmetric (10 instances strongly favor baseline, 7 favor psql).

---

## Q1: why per-call tokens look flat

| Metric (per conversation) | psql | baseline |
|---|---|---|
| Avg total input tokens | **133,184** | 90,661 |
| Avg total output tokens | **6,683** | 5,335 |
| Avg model calls | **18.0** | 13.3 |
| Avg input tokens / call | 7,003 | 6,572 |

Three mechanisms keep individual psql payloads small:

- **`\d` output is compact**: ~840 chars avg (median 704) — bare names/types, no column
  comments, no sample rows. Compare `get_table_schema` ~2,700 chars (DDL with embedded
  column-meaning comments) and `get_schema` ~24,000 chars.
- **SQL results truncated identically**: `psql_console` SQL output gets the same
  `MAX_RESULT_LENGTH = 500` cut as `execute_sql`
  (`bird_interact_env_tools.py:57`; meta-commands are exempt by design).
- **Cheaper calls → longer conversations**: `psql_console` costs 0.5 coins vs
  `execute_sql` 1.0, so the same budget buys ~2× the exploration calls
  (20,529 psql_console vs 8,280 execute_sql calls). That call-count growth, not payload
  size, is what drives the higher *total*.

The single biggest token sink in the psql run is `get_all_column_meanings`: called 3.7×
more often (1,187 vs 321 calls, ~22k chars each, 25.9M chars total) to compensate for
`\d` showing no column descriptions.

> Stale doc note: `tools/CLAUDE.md` lists psql_console = 1.0 / execute_sql = 2.0; the
> code (`DB_TOOL_COSTS`) has had 0.5 / 1.0 since the tool was introduced (c64853a).

## Q2: root cause where psql fails but baseline passes

Per-instance pass-rate diff ≥ 4/9 in baseline's favor: `alien_2, solar_4, mental_10,
gaming_9, virtual_4, solar_8, fake_5, solar_7, news_7, insider_2`.

**Failure mode on these instances: "Wrong SQL" submissions.** Budgets end at −2
(= submit_sql reached), so no budget death; syntax errors and truncation are negligible.

**Dominant root cause — KB-blindness:**

- `kb_blind` (gold SQL needs KB entries but `get_knowledge_definition` never called):
  **13.4% of instances (psql) vs 9.4% (baseline)**.
- It is lethal: KB-blind samples pass at **~12%** vs **~36%** when the KB is consulted
  (both runs show the same split).
- It concentrates exactly on the worst instances: `solar_7` (KB-blind in 8/9 failing
  iters), `solar_4` (6/7), `news_7` (5/5), `alien_2` (4/8).

**Worked example — `solar_7`** (psql 0/9, baseline 4/9): the task needs KB entry
"Fill Factor Degradation Rate (FFDR)" = `(ffactorinit − ffactorcurr) / years × 100`.
The psql agent's entire trace is 17× `psql_console` + 3× `submit_sql` — no KB or
meaning tools — and it invents `(init − curr) / init / years` (normalized, no ×100).
The baseline run saw the column-meaning comments embedded in `get_table_schema` output
and converged to the correct formula.

**Secondary patterns (psql vs baseline, instance rates):**

- `repeated_identical_call`: 43.7% vs 30.1% — 647 psql conversations re-issue an
  identical console command (`\dt`, repeated SELECTs), burning calls without new info.
- `blind_submit`: 5.6% vs 2.4%.

Checked and ruled out: column-description visibility per se does not correlate with
failure (psql samples that never opened column meanings actually pass *more*, 43% vs
30% — confounded by easy tasks needing none). The KB channel, not the column-meaning
channel, is the driver.

## Takeaway / next lever

The console itself is fine — psql wins symmetric cases (`robot_6`, `credit_9`,
`credit_3`, …) where schema/data exploration suffices. The loss cases are tasks whose
answer is a KB-defined formula. Most targeted fix: prompt-side nudge in the psql
variant to check `get_all_external_knowledge_names` / `get_knowledge_definition`
before the first submit, mirroring the conditional schema-exploration tip added in
134518f.


# How to solve

1. Fix the KB-blindness with a dedicated prompt tip (highest leverage). The psql tip at prompts.py:47 already mentions external knowledge, but it's buried at the tail of the schema-exploration sentence â and the data says it gets ignored (KB-blind on 13.4% of instances, passing only ~12% there vs ~36% with KB). I'd split it into its own imperative tip for the psql variant, something like: "The schema alone does not define domain formulas (rates, scores, classifications). Before your first submit_sql, call get_all_external_knowledge_names and look up any entry matching the question with get_knowledge_definition â do not invent formulas." That directly targets the solar_7-style failures. A stronger structural variant: since get_all_external_knowledge_names output is only ~2k chars, you could inline the KB names list into the system prompt for free, so the agent sees that "Fill Factor Degradation Rate (FFDR)" exists without having to decide to look.

2. Break repeated-identical-call loops tool-side. 43.7% of psql instances (647 conversations) re-issue a byte-identical psql_console command. A small addition to tool_wrapper_patience_and_submit (or a tiny new wrapper) that detects an exact (tool, args) repeat and appends [NOTE: identical command already executed this conversation â result unchanged] to the result would stop the loop and stop the budget bleed. You could even make repeats free to avoid double-charging for no information.

3. Run a cost-calibration ablation: psql_console at 1.0. The 0.5 cost buys ~2Ã the exploration calls, which is exactly what produces the +47% token bill and arguably the console tunnel-vision. Bumping it to parity with execute_sql is a one-number change in DB_TOOL_COSTS and would tell you whether forced economy redirects budget toward KB lookups. (Also fix the stale 1.0/2.0 costs in tools/CLAUDE.md while there.)

4. (Optional, lower priority) Surface column comments in \d. You could COMMENT ON COLUMN at DB setup so \d+ shows meanings, closing the gap with get_table_schema. But I'd deprioritize this: the data showed column-description visibility does not correlate with failure â the KB channel is the driver, not the column-meaning channel.

5. Soft-guard blind submits. blind_submit doubled (5.6% vs 2.4%). The submit wrapper could warn when the submitted SQL was never executed in the console: "this query was never tested; consider running it first if budget allows." Cheap to add, modest payoff.

I'd do 1 + 2 together as the next __psql run (they're independent fixes for the two dominant patterns), and 3 as a separate ablation so the effects stay attributable. Want me to implement any of these?