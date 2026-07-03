# maintenance_agent baseline (phase 2, Ticket A)

**Date:** 2026-07-03
**Status:** design approved, spec pending final review
**Predecessor:** `2026-07-03-text2sql-as-software-maintenance-design.md` (the framing spec; this
document is its phase 2 — "Eval baseline")
**Scope:** Ticket A (feature request) only. Ticket B (repair/follow-up) is deferred to its own
brainstorm once Ticket A is built and validated, per the framing spec's own phasing.

## Goal

Build a new eval baseline, `maintenance_agent`, that evolves `deep_agent` into the
software-maintenance framing: the agent explores a per-task workspace (`docs/` catalog +
on-disk issue thread), writes a SQL file, self-checks with a structural test primitive, and
submits silently (no pass/fail feedback). This validates the coding-primitives methodology
end-to-end inside the existing LangGraph pipeline before phase 3 (external-harness adapter)
or phase 4 (training) are attempted.

## Package layout

New sibling package `maintenance_agent/` under `agents/`, same shape as `deep_agent/`:

```
maintenance_agent/
├── __init__.py
├── agent_code.py          # run_agent_maintenance + _build_tools/_build_middleware helpers
├── agent_code_state.py    # MaintenanceAgentCustomState (same 3 patience fields as DeepAgentCustomState)
├── catalog_seed.py        # materialize_maintenance_workspace + maintenance_tool_costs
├── prompts.py             # build_maintenance_agent_messages
└── tools/
    ├── __init__.py
    └── maintenance_tools.py   # write_query, comment_on_issue, run_tests, submit
```

Registered in `agents/__init__.py` and wired into `main_pipe_workflow.py` behind a new
`baseline` enum value, per the "Adding a new agent" checklist in `agents/CLAUDE.md`.

## Reuse

| From | What | Why |
|---|---|---|
| `deep_agent.catalog_seed.materialize_catalog_dir` | table/KB rendering into a temp dir | identical `docs/` content requirement; no need to re-derive it |
| `deep_agent.tools.bash_tool` | `return_tool_bash`, `build_pg_env` | exploration + ad-hoc `psql` is unchanged |
| `bird_baseline.tools.bird_interact_user_tools.ask_user_impl` | two-stage LLM user-sim pipeline | the engine behind `comment_on_issue`; only the LangGraph tool wrapper (and on-disk append) is new |
| `bird_baseline.tools.bird_interact_user_tools.submit_sql_impl` | GT execution-match comparison | the hidden grader, called by the harness after the run, never exposed to the agent |
| `bird_baseline.agent_callback` | `check_budget_limit`, `sanitize_thinking_history`, `wrap_model_append_tool_message` | unmodified — the budget-deduction/note-appending logic is identical |

## Workspace

`materialize_maintenance_workspace(task)` in `maintenance_agent/catalog_seed.py`:

1. Calls `deep_agent.catalog_seed.materialize_catalog_dir(task)` to get a temp dir populated
   with `database_overview.md`, `tables/*.md`, `knowledge_base/*.md` (per-task masked KB
   rendering, unchanged).
2. Creates `docs/` inside that dir and moves the three items above into it (the framing spec's
   workspace anatomy nests the catalog one level under `docs/`, unlike `deep_agent`'s flat
   layout).
3. Writes `ISSUE.md`: the ticket body is `task.task_question`, followed by an empty
   `## Comments` section that `comment_on_issue` appends to.
4. Writes `queries/answer.sql`: a fixed stub, `-- TODO: replace with your SQL\n`.
5. Writes `tests/test_contract.py`: a **reference artifact** — readable via `cat`, illustrating
   the same two checks `run_tests` performs (see below). Not executed by the agent in this
   phase (see rationale in Tools).

```
<tmp>/
├── ISSUE.md
├── docs/
│   ├── database_overview.md
│   ├── tables/*.md
│   └── knowledge_base/*.md
├── queries/
│   └── answer.sql
└── tests/
    └── test_contract.py
```

Cleanup: `shutil.rmtree` in a `finally` block, same as `deep_agent`.

## Tools

| Tool | Signature | Cost | Behavior |
|---|---|---|---|
| `bash` | reused unchanged | 1.0 | explore `docs/`, ad-hoc `psql -c "SELECT …"` |
| `write_query` | `write_query(content: str) -> str` | 1.0 | overwrites the fixed `queries/answer.sql` path — the path is hardcoded in the tool closure, never agent-supplied, so there is no traversal risk by construction |
| `comment_on_issue` | `comment_on_issue(question: str) -> str` | 2.0 | calls `ask_user_impl`, appends `**Agent:** <question>` / `**Author:** <answer>` to `ISSUE.md`'s `## Comments` section, returns the answer text directly to the agent (so no forced re-read, but the durable record is on disk per the framing spec) |
| `run_tests` | `run_tests() -> str` | 1.0 | Python-level structural check (not a shelled-out script — see rationale below): (a) stub marker stripped, content non-empty; (b) `EXPLAIN <query>` succeeds against the task's read-only DSN. Reports pass/fail on structure only — **never** on correctness, so it cannot leak the oracle |
| `submit` | `submit() -> str` | 3.0 | terminal, unconditional — ends the episode with no pass/fail feedback |

**Why `run_tests` is a tool, not an executed shell script.** The existing `bash` tool's
whitelist (`cat, ls, find, grep, head, tail, wc, psql`) deliberately excludes `bash`/`sh` as an
invocable command — that boundary is what makes the sandbox a whitelist rather than "anything
not explicitly denied." Adding a way to run `tests/test_contract.py` as a real script would
require extending that whitelist to arbitrary script execution, which defeats it. Instead,
`test_contract.py` stays on disk as a **readable, `cat`-able reference** — useful for the
agent's own understanding, and directly reusable verbatim by phase-3 external harnesses that
get real shell access — while `run_tests` exposes the identical check as a first-class,
sandboxed primitive today.

## Middleware

New factory `make_tool_wrapper_patience_and_submit_silent(tool_costs, submit_tool_name="submit")`
added to `bird_baseline/agent_callback.py` (co-located with the existing
`make_tool_wrapper_patience_and_submit`, since that file is the canonical home for
patience-budget logic per its module docstring). Same shape as the existing factory — blocks
non-submit tools whose cost exceeds remaining budget, records cost — but drops the
`passed`-JSON / retry branch entirely: calling `submit` is **always** terminal.

```python
if tool_name == submit_tool_name:
    terminal_patience = (
        PATIENCE_SUBMIT_EXHAUSTED if user_patience < 0 else PATIENCE_SUBMIT_PASSED
    )
    return Command(update={"messages": [...], "updated_user_patience": terminal_patience})
```

The existing factory's budget-exhausted block message is hardcoded to
`"You MUST call submit_sql now with your best SQL."`; the new factory parametrizes this on
`submit_tool_name` too (`f"You MUST call {submit_tool_name} now."`), so a blocked
`write_query`/`comment_on_issue`/`run_tests`/`bash` call correctly tells the agent to call
`submit`, not the nonexistent `submit_sql`.

`PATIENCE_SUBMIT_PASSED`/`PATIENCE_SUBMIT_EXHAUSTED` are reused as-is purely for their existing
message-wording behavior in `check_budget_limit` ("clean finish" vs. "conversation limit
reached" note) — no correctness signal is attached to either sentinel here, unlike in
`bird_baseline` where `PASSED` specifically means "SQL was graded correct." Reusing the names
keeps downstream turn-classification code (which already recognizes these sentinels) working
unmodified; the meaning shift (terminal-because-submitted vs. terminal-because-correct) is a
documented departure, not a naming collision.

`check_budget_limit`, `sanitize_thinking_history`, `wrap_model_append_tool_message` are reused
unmodified. Full middleware order (same relative position as `deep_agent`):
`ModelRetryMiddleware` → `ToolRetryMiddleware` → `check_budget_limit` →
`sanitize_thinking_history` → `wrap_model_append_tool_message` →
`make_tool_wrapper_patience_and_submit_silent(maintenance_tool_costs())`.

## Grading (no tool-call parsing)

`run_agent_maintenance` reads `catalog_dir / "queries" / "answer.sql"` from disk **after**
`agent.invoke` returns — regardless of whether the episode ended via an explicit `submit` call
or via forced budget exhaustion, the file's final on-disk content is `predicted_sql`,
unconditionally. This is simpler than `bird_baseline`/`deep_agent`'s `_extract_predicted_sql`
(which parses the last relevant tool call out of message history): the workspace *is* the
state, so there's nothing to parse.

`submit_sql_impl` (existing, reused unmodified) is then called by the harness itself — never
exposed to the agent as a tool — purely to populate `execution_accuracy` for our own metrics
output. This mirrors the framing spec's "hidden grader reads the file" design: the agent never
sees a pass/fail signal, but the pipeline still needs one to report results.

## Pipeline wiring

New `baseline` enum value `maintenance_agent`. `_resolve_baseline_settings` maps it to
`(make_data_ambiguous=True, needs_user_sim=True, enable_ask_user=True)` — unlike `deep_agent`
(which defaults ambiguity + `ask_user` off to validate bash+KB reading in isolation),
`maintenance_agent`'s entire point is ambiguity resolution through the issue thread, so it is
on by default with no separate ablation flag in this phase.

`maintenance_tool_costs()` in `catalog_seed.py` mirrors `deep_tool_costs()`'s shape:

```python
def maintenance_tool_costs() -> dict[str, float]:
    return {
        "bash": 1.0,
        "write_query": 1.0,
        "comment_on_issue": USER_TOOL_COSTS["ask_user"],
        "run_tests": 1.0,
        "submit": USER_TOOL_COSTS["submit_sql"],
    }
```

## Prompt

`build_maintenance_agent_messages` follows `deep_agent/prompts.py`'s inline-Jinja2 pattern,
describing: the workspace layout (`ISSUE.md`, `docs/`, `queries/answer.sql`, `tests/`), the five
tools and their costs, and the red→green loop (`write_query` → `run_tests` → iterate →
`submit`). Framed as engineering narration per the parent spec: commands are cheap, comments on
the issue cost the author's patience, the episode ends when patience runs out or `submit` is
called.

## Testing

Mirrors `tests/eval_framework/agents/deep_agent/`:

- `test_catalog_seed.py` — workspace shape (`docs/`, `queries/answer.sql` stub content,
  `ISSUE.md` header + empty `## Comments`), cleanup on both success and exception.
- `test_agent_code.py` — tool list wiring, middleware order, state schema.
- `tools/test_maintenance_tools.py`:
  - `write_query` — writes to the fixed path regardless of any path-like content in the
    argument; overwrite semantics (full replace, not append).
  - `comment_on_issue` — appends Q+A to `ISSUE.md`, returns the answer text.
  - `run_tests` — stub → fail; syntactically valid `SELECT` → pass; invalid SQL → fail with a
    structural (not correctness) message.
  - the new silent-submit middleware factory — terminal on every call; `PATIENCE_SUBMIT_PASSED`
    when `user_patience >= 0` at call time, `PATIENCE_SUBMIT_EXHAUSTED` when the call follows a
    block.

## Out of scope (this phase)

- Ticket B (repair/follow-up) — separate spec, once Ticket A is validated.
- An `enable_comment_on_issue` ablation switch to mirror `deep_agent`'s `enable_ask_user`
  default-off pattern — could be added later; not needed to validate the core loop.
- Any richer visible-test tier (asserting output columns) — deferred in the framing spec for
  leakage reasons; `run_tests` stays executability-only.
- Git-backed workspace / diff-based submission — that is phase 3's concern (external-harness
  adapter), where the workspace format itself becomes the adapter surface.
