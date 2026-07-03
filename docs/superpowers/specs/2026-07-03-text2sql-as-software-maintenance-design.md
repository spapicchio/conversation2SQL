# Interactive Text2SQL as Software Maintenance — framing & methodology

**Date:** 2026-07-03
**Status:** approved framing (conceptual phase); implementation phases follow separately
**Predecessors:** `2026-06-24-deep-agent-baseline-design.md`, `2026-06-27-deep-agent-catalog-filesystem-design.md`, `2026-06-30-deep-agent-bash-tool-design.md`

## Thesis

Interactive Text2SQL (BIRD-Interact) can be recast as **software maintenance over a
database application workspace**. Each benchmark instance becomes an underspecified
ticket filed against a small repository of SQL assets, docs, and tests. This exposes
the task to mature coding-agent capabilities — repository navigation, shell-based
exploration, test-driven repair, patch generation, and hidden-test validation — and
lets one task format serve three later phases: an eval baseline in this repo, an
adapter for external harnesses (Claude Code, SWE-agent), and a training environment
(SFT/RL via VeRL).

The methodology is SWE-bench's **issue → patch → hidden-test** loop, transplanted onto
a per-database application repo, with BIRD-Interact's interactive-ambiguity and
patience-budget mechanics mapped in rather than dropped.

## Concept mapping

| BIRD-Interact concept | Software-maintenance counterpart |
|---|---|
| Ambiguous user query | Underspecified issue (`ISSUE.md`) filed against the application |
| Simulated user + `ask_user` | Ticket author; asking = commenting on the issue thread |
| Patience budget (bird-coins) | Reviewer/author patience; every action costs engineering time |
| Schema catalog + masked KB | The repo itself: `docs/` with schema files and in-house domain docs |
| GT SQL / `submit_sql` grading | Hidden FAIL_TO_PASS test (execution match vs. ground truth) |
| Follow-up turn | Repair ticket against a query that already exists in the repo |
| `execution_accuracy` | "% resolved" — same number, SWE-bench vocabulary |

## Task universe: two tickets per instance

Each BIRD-Interact instance expands into two maintenance tickets over the same
per-database application repo:

- **Ticket A — feature request.** The ambiguous phase-1 query. `queries/<name>.sql`
  exists only as an empty stub (header comment, no SQL) that the agent must fill; the
  issue asks for a new report. Fixing the target path keeps the visible tests
  well-defined. Resolving ambiguity through the issue thread is the core measured
  skill.
- **Ticket B — repair / change request.** The follow-up turn. The repo already
  contains phase-1's **ground-truth** SQL as `queries/<name>.sql`; the issue asks for
  a modification. Genuine patch-the-existing-code maintenance, built entirely from the
  benchmark's own data — no synthetic seeding, no cross-task leakage.

Ticket B always starts from the GT phase-1 query, never from the agent's Ticket-A
output, so the two tickets grade independently (the same isolation BIRD-Interact uses
for follow-ups).

Rejected seeding alternatives: sibling-task GT SQL (near-duplicate solution leakage
within a database), LLM-generated filler queries (synthetic noise the agent ignores),
and empty greenfield repos (no maintenance story, no regression surface).

## Workspace anatomy

Generated per task by extending the existing `scripts/generate_catalog.py` +
`materialize_catalog_dir` pipeline; `docs/` is exactly today's deep_agent catalog,
with the KB re-rendered per task from `masked_agent_kb` (masked entries never appear).

```
db_app/
├── ISSUE.md            # ticket body (ambiguous query) + comment thread, appended on disk
├── docs/
│   ├── database_overview.md
│   ├── tables/*.md     # DDL + column descriptions + FKs
│   └── knowledge_base/*.md
├── queries/            # the "application": target stub (Ticket A) or phase-1 GT SQL (Ticket B)
├── tests/              # visible contract tests
└── run_tests.sh        # the agent's red→green loop
```

The issue thread is **materialized on disk**: every question and author reply is
appended to `ISSUE.md`. Interaction state is therefore file-native — re-readable,
grep-able — and trajectories read like real dev sessions, which is what the training
phase needs.

## Agent–environment interface (coding primitives only)

| Primitive | Implementation | Cost |
|---|---|---|
| `bash` (cat/grep/ls/find/head/tail/wc + read-only psql) | existing deep_agent tool, unchanged | cheap (1.0) |
| `edit`/write, restricted to `queries/` (and scratch) | **new** — the only added capability | cheap |
| `comment_on_issue` | reskinned `ask_user`; Q&A also appended to `ISSUE.md` | expensive (existing ask_user cost) |
| `submit` | ends the episode, **silent** — no pass/fail echo | terminal |

Budget mechanics reuse the `bird_baseline` middleware verbatim (via the
`make_tool_wrapper_patience_and_submit(tool_costs)` factory with a new cost table).
Only the narration changes: commands are cheap, bothering the ticket author is
expensive, the episode ends when patience runs out.

Departure from BIRD-Interact semantics, by design: `submit` gives **no pass/fail
feedback** (no retry-on-failed-submit loop). Coding harnesses don't get a
ground-truth oracle at submission time; keeping submit silent is what makes the
format harness-agnostic and prevents oracle-guided guessing. The agent self-checks by
running its SQL through psql and the visible tests instead.

## Verification split

- **Visible tests** (`tests/`, via `run_tests.sh`): **executability only** — the
  target query file exists, parses, and executes read-only without error. Derived
  purely from unambiguous task facts, so zero oracle leakage by construction; red at
  episode start, green-able without resolving any ambiguity. A richer tier (assert
  expected output columns where derivable from GT SQL) is deferred as a future
  ablation — in tasks where the ambiguity concerns *what to return*, column names leak
  part of the answer, so that tier needs a per-task leakage filter.
- **Hidden tests** (grader, outside the workspace): today's `submit_sql_impl`
  execution match against GT results, honoring the `sql_query_conditions["order"]`
  flag — the FAIL_TO_PASS analog. Other `queries/*.sql` still executing correctly is
  the PASS_TO_PASS analog (meaningful in Ticket B; cheap to include everywhere).
- **Metric:** `execution_accuracy`, unchanged — results stay directly comparable with
  the existing `no_tool` / `tools_only` / `tools_user` / `bird_full` / `deep_agent`
  ablation table.

## Phased roadmap

1. **Framing (this document).** The mapping, workspace format, and verification split
   as the paper-facing methodology.
2. **Eval baseline.** A `maintenance_agent` baseline (or evolution of `deep_agent`)
   inside the existing LangGraph pipeline: add the write tool, the on-disk issue
   thread, and silent submit. Ticket A first; Ticket B (follow-up repair) after.
3. **External-harness adapter.** The workspace format *is* the adapter: `git init`
   the workspace, point an off-the-shelf harness at it, grade the resulting diff with
   the hidden tests. No separate task format.
4. **Training.** Trajectories are already file-edit/shell/test trajectories suitable
   for SFT; the hidden grader is the reward function for RL via VeRL.

Each phase gets its own design + plan; this spec fixes only the shared format and
semantics.

## Out of scope

- Embedding SQL in a host codebase (dbt models, Python query functions) — drifts from
  the benchmark and breaks grading comparability.
- Tuning subagents, multi-agent decomposition — orthogonal ablations.
- Any change to the patience-budget arithmetic (`6 + 2*m_amb + 2*user_patience`).
