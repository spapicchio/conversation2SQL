# Experiments Log

A running trace of observed anomalies, investigations, and findings.

---

## EXP-000 — Template [TODO]

**Date:** YYYY-MM-DD  
**Observation:** What caught your eye or the strange behavior you noticed.  
**Hypothesis:** What you think is causing it.  

### Steps

1. First command or action taken.
2. Second command, etc.

### Result

What you found. Whether it confirmed or refuted the hypothesis.

---

## EXP-001 — DB/Column submit errors are budget-exhausted execute_sql loops [TODO]

**Date:** 2026-06-21
**Run:** `results/2026-06-12/07-28-13/tools_only__Qwen3.5-9B__ddl__lin__iter9__double_ctx_bdg__error`
**Observation:** Suspected that "DB Error" and "Column/Relation Not Found" submit_sql errors increased in this run.
**Hypothesis:** The doubled context budget (`double_ctx_bdg`) lets the agent persist on un-executable queries longer, inflating these two error classes at forced-submit time.

### Steps

1. Loaded the run with `explorer.loader.load_run` and dumped the error distribution + the last `submit_sql` message for each Column/Relation and DB Error record.
2. Traced one instance end-to-end (`news_4`): schema, every `execute_sql` result, and the final submission.
3. Correlated each error class with `execute_sql` call count and budget exhaustion.
4. Compared error rates against every sibling `tools_only__Qwen3.5-9B__ddl` run (normal-budget 06-05 baseline, new_tools, psql, psql+double).
5. Measured SQL repetition (distinct vs total `execute_sql` queries) within the failing records.

### Result

**Hypothesis refuted — the errors did not increase.** This run has the *lowest* Column/Relation (2.7%) and DB Error (1.7%) rates of any run in the series (06-05 normal budget: 5.0% / 4.0%; 06-11 psql+double: 4.6% / 3.1%). Doubling the budget *reduced* both, shifting failures toward "Wrong SQL" (51.5%→56.8%).

**Root-cause mechanism (confirmed) of the errors that do occur:** a degenerate `execute_sql` retry loop. 100% of Column/Relation (45/45) and DB Error (28/28) records end in budget exhaustion after a mean ~21 `execute_sql` calls (vs 4.6 for passes, 0% exhausted). The schema is correct and Postgres returns the exact fix every time (e.g. `column "occubl" does not exist … HINT: … "occulbl"`); the agent even names the correct column in its thinking yet re-emits the typo — `news_4` ran the identical broken query 27 times before force-submitting it. Failure is Qwen3.5-9B not acting on explicit corrective feedback, not a schema/data problem.

### Proposed experiment to run

Add a **loop/no-progress guard** to `execute_sql` and re-run to measure impact on these error classes (and overall pass@1): detect when the last N `execute_sql` calls returned the same error (or near-identical query), and either (a) inject a stronger nudge that quotes the HINT and forces a column-name correction, or (b) short-circuit to `submit_sql`/`ask_user` instead of burning the remaining budget on the same broken query. Compare error distribution and budget-exhaustion rate against this run as the control.

---

## EXP-002 — psql_console regresses enum-heavy tasks (missing enum catalog) [TODO]

**Date:** 2026-06-21
**Observation:** Two runs identical except `enable_psql_console` (false vs true) land at the
*same* pass@1 (0.387), but several instances flip hard between them — one run passes >½ of its
9 iterations while the other almost never passes.
- no-psql: `results/2026-06-12/07-28-13/tools_only__Qwen3.5-9B__ddl__lin__iter9__double_ctx_bdg__error`
- psql:    `results/2026-06-11/16-49-34/tools_only__Qwen3.5-9B__ddl__lin__iter9__psql__double_ctx_bdg__error`

Largest disagreements (no-psql vs psql): `fake_7` 0/9 vs 6/9, `credit_3` 1/9 vs 7/9,
`robot_7` 2/9 vs 7/9 (psql helps); `museum_9` 7/9 vs 2/9, `fake_9` 6/9 vs 1/9,
`mental_5` 5/9 vs 0/9 (psql hurts).

**Hypothesis:** Enabling `psql_console` *replaces* `get_schema` and makes the agent abandon the
semantic helper tools, changing the failure mode rather than the overall rate.

### Steps

1. Loaded both runs via `explorer/loader.py`, grouped by `instance_id`, ranked by |pass-fraction delta|.
2. Diffed the two `config.yaml` — only `enable_psql_console` differs.
3. Per disagreement instance, compared error classes, tool usage, model calls, predicted SQL, and the last `submit_sql` message.
4. Inspected the psql agent's system prompt and tool calls for `mental_5`.

### Result — confirmed, with a precise root cause

- pass@1 is identical, but the **failure mode shifts**. Full-run error mix:
  Column/Relation-Not-Found 2.7%→4.6%, DB Error 1.7%→3.1%, Syntax Error 0.4%→1.3%;
  `get_column_meaning`/`get_all_column_meanings` calls collapse 2454→653; avg model calls 20.4→23.3.
- The psql agent produces **more SQL that does not execute** (enum/column/syntax errors) instead of executable-but-wrong SQL.
- **Root cause (verified):** in psql mode `get_schema` — whose DDL dump lists every
  `CREATE TYPE … AS ENUM (…)` with its allowed values — is gone. `\d <table>` shows only the enum
  *type name*, not its values. In `mental_5` the psql agent's system prompt contained **0** enum
  definitions and it **never ran `\dT`**, so it guessed literals and failed every iteration with
  `invalid input value for enum functionalimpairment_enum: "Significant"`. The DB defines two
  confusable enums (`functionalimpairment_enum` = Severe/Moderate/Mild vs
  `functionalimprovement_enum` = Moderate/Minimal/Significant); the agent took a value from the
  wrong one. The no-psql agent reads the enum values from `get_schema` and passes.

**Fix applied (Option B):** `extract_enum_types(ddl)` pulls the `CREATE TYPE … AS ENUM (…)`
statements out of the schema; `agent_code.py` computes the catalog when `enable_psql_console` and
injects it into the psql system prompt (`prompts.py`), restoring the upfront enum catalog the
non-psql agent gets from `get_schema`. Tests:
`tests/eval_framework/tools/test_bird_interact_env_tools.py::TestExtractEnumTypes` and
the enum cases in `tests/eval_framework/agents/test_psql_console_wiring.py`.

### Re-run command (psql + enum fix)

```bash
just eval \
  --variant all_db_all_kb_linearized \
  --model qwen35 \
  --baseline tools_only \
  --num-iterations 9 \
  --extra "--enable_psql_console true --reader_user_patience_budget 12" \
  --append-name double_ctx_bdg_enumfix
```

`RUNNER=slurm just eval …` submits via sbatch. New run-dir slug:
`tools_only__Qwen3.5-9B__ddl__lin__iter9__psql__double_ctx_bdg_enumfix`.

### Compare against (when finished)

- **Primary:** the pre-fix psql run
  `results/2026-06-11/16-49-34/tools_only__Qwen3.5-9B__ddl__lin__iter9__psql__double_ctx_bdg__error`
  — same settings, only the code (enum injection) differs. Expect DB/Column/Syntax error classes
  to drop and the psql-hurts instances (`mental_5`, `fake_9`, `museum_9`) to recover.
- **Reference:** the no-psql run
  `results/2026-06-12/07-28-13/tools_only__Qwen3.5-9B__ddl__lin__iter9__double_ctx_bdg__error`
  — the variant that never had the enum problem (upper bound for those instances).

Use the explorer compare page (`explorer/pages/compare.py`) or `join_runs` on the three run dirs.

---

## EXP-003 — Why the psql variant has *more* DB/Column submit errors (and what is NOT the cause) [TODO]

**Date:** 2026-06-21
**Runs (identical except `enable_psql_console`):**
- psql:    `results/2026-06-11/16-49-34/tools_only__Qwen3.5-9B__ddl__lin__iter9__psql__double_ctx_bdg__error`
- no-psql: `results/2026-06-12/07-28-13/tools_only__Qwen3.5-9B__ddl__lin__iter9__double_ctx_bdg__error`

**Observation:** the psql run has Column/Relation-Not-Found 4.6% vs 2.7% and DB Error 3.1% vs 1.7%
(combined 7.7% vs 4.4%). Config diff is a single flag: `enable_psql_console`.

### Steps

1. Diffed both `config.yaml` — only `enable_psql_console: true/false` differs.
2. Compared tool usage: psql run has `get_schema=0`, `get_column_meaning` 334 vs 1256,
   `get_all_column_meanings` 319 vs 1198 — the agent introspects via `\d` instead.
3. Traced `museum_11` (Column/Relation Not Found: `column d.count_adequate does not exist`):
   read every `psql_console` result, located the offending name in the `\d+` output.
4. Read the tool implementation (`tools/bird_interact_env_tools.py`, `tools/utils_db_execute.py`)
   to check what is and isn't truncated.
5. Counted malformed tool-call names across the whole psql run.

### Result — the increase is real; two of my first guesses were wrong

**Confirmed driver — weaker schema grounding.** `enable_psql_console` *replaces* the structured
`get_schema` (clean DDL: exact column names, `CREATE TYPE … AS ENUM (values)`, and all FK
constraints) with a raw `psql_console` the agent must self-introspect via `\d`. The agent also
leans on the semantic helpers far less (column-meaning calls collapse ~2454→653). Working from a
noisier view it hallucinates more column/enum names and mis-builds more multi-join SQL
(`missing FROM-clause entry for "s"`, `column reference … is ambiguous`, `function sum(text)…`).
Both variants end un-fixable queries in budget-exhausted retry loops (see EXP-001), so the worse
grounding surfaces as more *un-executable* submissions rather than merely wrong ones.

**Refuted — "column names are truncated/misread in `\d`."** `_is_psql_meta_command` returns `\d`/`\d+`
output **untruncated** by design; only SQL result rows are capped. `museum_11` is not a truncation:
the real column is `maintbudgetstatus character(25)` whose **description** lists
`'Review Required', 'Insufficient', 'Adequate'`. The agent invented `d.count_adequate` by
conflating a COUNT with the enum value `'Adequate'` — an enum-value hallucination, exactly the
class EXP-002's enum-catalog injection already targets.

**Refuted — "psql meta-command tool-calls are mis-parsed."** Only **4 of 39,003** tool calls
(0.010%, 3 records) had a malformed name (e.g. `'\\d+ account\n</parameter'`). Negligible; not a
driver of the error increase.

**Real (but separate) truncation knob:** `MAX_RESULT_ROWS = 3` (`utils_db_execute.py`) caps *SELECT
result rows* for both `execute_sql` and `psql_console`. When the agent runs `SELECT DISTINCT col`
to discover a column's value domain it sees only 3 of N (museum_11: `… [showing first 3 of 6 rows]`
on the dynasty list). This hurts value/enum **discovery** but is independent of the column-name
errors above — fixing it would not have fixed `museum_11`.

### Suggested next steps (in priority order)

1. **Schema grounding (highest leverage):** extend the EXP-002 enum-catalog injection to also inject
   a compact **FK/relationship summary** (and optionally the full column list) into the psql system
   prompt, restoring what `get_schema`'s DDL gave the non-psql agent without removing the console.
   Targets the `missing FROM-clause` / `ambiguous column` / bad-join DB errors (my point #3).
2. **Discovery truncation (optional, separate):** raise/relax `MAX_RESULT_ROWS` (or stop truncating
   single-column results) so value-domain probes (`SELECT DISTINCT …`) aren't cut to 3 rows.
   Mind the token-budget tradeoff (the prompt is re-sent each turn).
3. **Do not** invest in psql tool-call-format hardening — measured at 0.01%.

---

## EXP-004 — Would `create_python_udf` help on this run? (mostly no; one scoped exception)

**Date:** 2026-06-21
**Run analysed:** `results/2026-06-11/16-49-34/tools_only__Qwen3.5-9B__ddl__lin__iter9__psql__double_ctx_bdg__error`
(pass@1 0.383, 188 instances × 9 iters; `enable_python_udf` was **off** here).
**Question:** the `create_python_udf` tool (plpython3u, additive ablation) is fully wired
(`config_input.py` → `TaskData` → `agent_code._select`/cleanup → prompt → `__pyudf` slug). It's meant
for "formulas hard to express in pure SQL." Are there failures in this run it would actually fix?
**Hypothesis:** BIRD-Interact is formula-heavy (KB `calculation_knowledge` entries), so letting the
agent define a formula once in Python and call it from SQL could cut formula-transcription errors.

### Steps

1. Loaded all 9 iters, grouped by instance, computed per-instance pass fraction.
2. Counted KB entry types and split pass@1 by whether the GT KB carries a `calculation_knowledge` entry.
3. Scanned every GT `sol_sql` for constructs genuinely awkward in pure SQL (regex, stats fns,
   sqrt/exp/log/power, array/string parsing, recursive) — i.e. where a UDF adds *expressive* power.
4. Scanned KB formula definitions for non-trivial math (`\sqrt`, `\sum`, `log`, `percentile`, …).
5. Traced two always-failing (0/9) formula instances end-to-end (`archeology_4`, `virtual_6`):
   predicted SQL vs GT SQL vs the stated formula.
6. Counted GT SQLs that **repeat** a math sub-expression ≥2× (copy-pasted formula = a DRY/UDF signal)
   and correlated repeat-count with pass rate.

### Result — limited upside; the dominant blockers are not expressivity

- **Formulas dominate but are natively SQL-expressible.** 167/188 instances (89%) carry a
  `calculation_knowledge` KB entry and pass *lower* (0.379 vs 0.455 for the 21 non-formula ones). But
  almost every formula is arithmetic + `CASE` + functions PostgreSQL already has: only **16** GT SQLs
  use any non-trivial math, **9** any stats fn, **9** any string/array parsing — and those resolve to
  native `SQRT`/`LOG`/`POWER`/`PERCENTILE_CONT`. A Python UDF therefore adds **little expressive power**
  on this benchmark; SQL arithmetic is exactly its strong suit.
- **The real failure modes are grounding and interpretation, which a UDF does not touch.** A UDF body
  still references the same columns and encodes the same formula reading:
  - `archeology_4` (0/9): the agent wrote the RAR `sqrt` formula **correctly** but filtered on the
    wrong column (`logmethod LIKE '%Target%'` vs GT `refmark LIKE '%Target%'`) — a grounding error
    (EXP-003). Inside a UDF it would reference the same wrong column.
  - `virtual_6` (0/9): the agent implemented RRF **faithfully to the stated formula** (`(churnflag/3)*2`)
    but GT's SQL uses a different enum-numeric reading (raw 0/1/2/3) — an interpretation divergence,
    not an expressivity gap. A UDF picks the same interpretation it would have written inline.
  - These join the failure modes already documented in this series — retry loops (EXP-001), schema
    grounding (EXP-003), enum hallucination (EXP-002) — none of which a UDF addresses.
- **One genuine (scoped) win — copy-pasted formulas.** 10 GT SQLs reference a math sub-expression ≥2×;
  the heaviest repeaters all fail completely (`archeology_5` 12×, `archeology_1` 6×, `archeology_4`/`_8`
  4× → all 0/9) while light repeaters pass well (`archeology_3` 2× → 1.0, `archeology_11` 2× → 0.78,
  `archeology_6` 2× → 0.67). When one formula must appear in `SELECT` **and** a `CASE` **and** `ORDER BY`,
  every copy is an independent chance to diverge. Defining it once as a UDF and calling it N times
  removes that transcription-multiplier. This is the only place the tool plausibly moves the needle,
  and it concentrates on the `archeology` DB.

### Caveats / risks

1. **Blocked prerequisite:** the BIRD-Interact Postgres images do **not** ship `plpython3u`
   (verified: `CREATE EXTENSION plpython3u` → "could not open extension control file …/14/extension/
   plpython3u.control"). Until the image installs `postgresql-plpython3-14`, every `create_python_udf`
   call just returns that error — the ablation can't even be measured. (See the earlier wiring review.)
2. **Action-space bloat on a weak model:** EXP-003 showed Qwen3.5-9B already loses grounding when its
   tool surface grows; adding a non-trivial UDF tool may dilute rather than help.
3. **Comparison fragility:** plpython3u float math may round differently from inline SQL `numeric`
   arithmetic and fail `submit_sql`'s set comparison even when the formula is "right."

### Suggested next step (if pursued)

Don't run it as a blanket ablation. **Scope a test to the `archeology` DB** (the repeated-formula
cluster), *after* installing `plpython3u` in the DB image, and compare pass@1 on the 10 repeated-formula
instances against this run as control — watching specifically for rounding-mismatch submit failures.
Expected upside elsewhere is low; grounding/interpretation fixes (EXP-002/003) remain the higher-leverage work.

---

## EXP-005 — Terminal-only baseline: one shell tool instead of the structured tool suite [TODO]

**Date:** 2026-06-21
**Status:** Proposed (not yet run).
**Observation:** Today the agent gets a structured tool suite — `execute_sql`, `get_schema`
(+ optional `get_table_names`/`get_table_schema`), the semantic helpers
(`get_column_meaning`/`get_all_column_meanings`), the KB tools, and `submit_sql`. The existing
`enable_psql_console` ablation only swaps the *DB* tools for a single **read-only** `psql_console`
(`_select_db_tools` in `agent_code.py:60`); it still hands the agent `get_column_meaning`, the KB
tools, and `submit_sql`, and it injects a pre-built enum catalog (EXP-002) and DDL-derived context.
**Idea:** test the opposite extreme — give the model **only a terminal** and make it discover
everything itself: enumerate the schema (`\d`, `\d <table>`, `\dT+`), read column/enum semantics by
querying the catalog, and run arbitrary read **and** write SQL through the same shell. The question is
how far raw shell-driven self-service closes (or widens) the gap vs. the curated tool suite.

**Hypothesis:** A capable model with a real terminal can recover most of what `get_schema` +
the semantic helpers provide by introspecting Postgres directly, but a 9B model will lose ground —
EXP-001/002/003 already show Qwen3.5-9B fails to act on explicit DB feedback and hallucinates
column/enum names once the structured grounding is removed. Net pass@1 likely flat-to-down for the
small model; the value is measuring *which* failure classes the terminal shifts.

### Design / what to build

This needs more than the existing read-only `psql_console`, so it is a new ablation rather than a
config tweak:

1. **A single `terminal` tool** that executes a command against the task's Postgres DB and returns
   stdout/stderr. Two scoping options to decide before implementing:
   - **(a) psql-only terminal** — wrap `psql` so the model issues meta-commands (`\d`, `\dT+`,
     `\df`) *and* SQL through one tool, with **writes allowed** (drop the read-only guard in
     `utils_db_execute.py`). Closest to "detect the schema and run PostgreSQL" with minimal new
     surface.
   - **(b) full shell** — a real bash tool (e.g. `psql "$DSN" -c "…"`, plus `\copy`, scripts).
     Strictly more powerful but a much bigger sandbox/safety surface; only worth it if (a) is too
     limiting. Recommend starting with (a).
2. **Remove the curated scaffolding** so the test is honest: no pre-injected enum catalog
   (EXP-002), no `get_column_meaning`/`get_all_column_meanings`, no separate `get_schema`. Keep only
   `submit_sql` (the harness needs it to score) and — for the `tools_user`/`bird_full` cells —
   `ask_user`. The model recovers column meanings/enums by querying `information_schema` /
   `pg_catalog` itself.
3. **Register it** following the existing pattern: a `enable_terminal_only` flag in
   `config_input.py` (mutually exclusive with `enable_psql_console`/`enable_table_schema_tools`),
   tool selection in `_select_db_tools`, a cost entry in `DB_TOOL_COSTS`/`DB_TOOL_SPECS`
   (the unrestricted terminal probably warrants the `execute_sql` cost or higher), a `prompts.py`
   block telling the model it has only a terminal and must self-introspect, and a `__terminal`
   run-slug suffix in `presets.py`.

### Open questions to resolve before running

- **Read-only or read-write?** A write-capable terminal lets the model create temp tables / UDFs
  (overlaps with `enable_python_udf`) but risks mutating the shared eval DB — needs per-instance
  isolation or a transaction rollback wrapper.
- **Result truncation.** `MAX_RESULT_ROWS = 3` (`utils_db_execute.py`) will cripple self-service
  schema/value discovery (EXP-003 point #2); a terminal-only agent leans on `SELECT DISTINCT` far
  more, so this almost certainly must be relaxed for this variant.
- **Budget.** Self-introspection costs more turns; reuse the `double_ctx_bdg` setting
  (`--reader_user_patience_budget 12`) so the comparison isn't budget-starved.

### Compare against

- **Read-only-console reference:** `tools_only__Qwen3.5-9B__ddl__lin__iter9__psql__double_ctx_bdg`
  (EXP-002/003) — same "one DB tool" shape but read-only and *with* the semantic helpers + enum
  catalog. Isolates the effect of going fully self-service.
- **Structured-tools reference (upper bound):**
  `results/2026-06-12/07-28-13/tools_only__Qwen3.5-9B__ddl__lin__iter9__double_ctx_bdg__error`.
- **Metrics:** pass@1 and the error-class mix (Column/Relation-Not-Found, DB Error, Syntax,
  Wrong-SQL), plus tool-usage counts (expect a spike in catalog/`\d` queries replacing the
  collapsed `get_column_meaning` calls). Use the explorer compare page.

---


Why the text2SQL is not working for coding model? maybe because the model are perfect for coding and not for navigating dabase structure. If we model the problem as a coding problem for software engineer, the model perform better. 

- Q: what happen if the database and knowledge base is huge? we are paying space for performancescla