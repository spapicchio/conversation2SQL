# Expand linearize_prerequisites to render 1-hop neighbors (both directions) — design

**Date:** 2026-07-01
**Component:** `src/conversation2sql/eval_framework/agents/utils_kb_linearize.py`
(shared by `deep_agent/catalog_seed.py` and
`bird_baseline/tools/bird_interact_env_tools.py::get_knowledge_definition`)

## Goal

Fix a root cause identified in `docs/debug/deep_agent_alien_debug_prompt_fixes.md`
(`alien_6`, `alien_8`): approximation/threshold/classification variants of a metric
are modeled in the KB as **dependents** of the base metric (they list the base
metric in their own `children_knowledge`), not as prerequisites of it. Example from
`data/bird_interact/bird-interact-lite/alien/alien_kb.jsonl`:

```
id 36  Confirmation Confidence Score (CCS)   children: [0]
id 47  CCS Approximation                      children: [36]   <- dependent of CCS
id 54  High Confidence Signals ("CCS > 0.8")  children: [36]   <- dependent of CCS
id 15  Narrowband Technological Marker (NTM)  children: [4]
id 39  NTM Classification System              children: [15, 4]  <- dependent of NTM (and BFR)
```

`linearize_prerequisites(name, kb)` — the function that renders each
`knowledge_base/<node>.md` catalog file — currently walks `children_knowledge`
**upward only** (an entry's own transitive ancestors/prerequisites) and
explicitly **excludes dependents**. So `knowledge_base/ccs.md` never mentions
`CCS Approximation` or `High Confidence Signals`, and the agent must separately
discover and open those files by name — which is exactly what failed in the
debug run (12 blind `submit_sql` threshold guesses in `alien_6`, never finding
`ccs_approximation.md` / `high_confidence_signals.md`).

## Current behavior (for reference)

- Collects `name`'s full transitive ancestor set (unbounded-depth stack walk
  over `children_knowledge`).
- Topologically sorts the collected set; `ordered[-1]` is always `name` itself
  (it depends, directly or transitively, on everything else collected).
- Renders **only** `ordered[-1]`'s own description/definition in full; every
  other collected ancestor is named only, via an edge list
  (`- "X" needs "Y"`), capped for display at `MAX_DEPTH` (10) entries.
- Dependents of `name` never appear at all.

This document changes that contract. `linearize_kb` (the full-KB-dump used by
`get_all_knowledge_definitions`) is **not** touched — every node in a connected
component already gets included there (just not always fully rendered — a
separate, pre-existing, out-of-scope quirk left as-is).

## New behavior

Treat the KB DAG as undirected for reachability (same neighbor-set
construction already used by `_find_connected_components`: every
`children_knowledge` edge counts both ways) and BFS outward from `name`,
recording each reached node's hop-distance.

| Distance | Direction | Rendering |
|----------|-----------|-----------|
| 0 | — (`name` itself) | Full block: `# {knowledge}` + description + formula (unchanged from today) |
| 1 | prerequisite (parent) **or** dependent (child) | Full block each: `# {knowledge}` + description + formula (**new** — today a 1-hop prerequisite is named only, and 1-hop dependents don't appear at all) |
| ≥2 | either direction | Name only, combined across both directions, capped at `MAX_DEPTH` (10) total entries |

Distance-≥2 entries are rendered as `- "X" needs "Y"` for every direct
`children_knowledge` edge that connects two nodes within the full discovered
set (not just edges touching `name` directly) — same textual style already
used today for the ancestor-chain edges, just no longer restricted to
ancestors only. Ties (equal distance) break by ascending `id`, matching the
existing determinism convention (`_topological_sort`, `_find_connected_components`).

Masked-out entries never appear (BFS only ever traverses
`masked_agent_kb.values()`, same as today) — no leak of masked prerequisites
or dependents.

### Example: `knowledge_base/ccs.md`

CCS has one 1-hop prerequisite (SNQI) and four 1-hop dependents (`CCS
Approximation`, `High Confidence Signals`, `Observation-Verified Signal`,
`High-Confidence Technosignature`); none of those five have further neighbors
outside this set, so there is no distance-≥2 section for this entry:

```
# Confirmation Confidence Score (CCS)
- **description**: Quantifies overall confidence in signal verification across multiple parameters
- **definition**: (1 - FalsePosProb) * DecodeConf * ClassConf * ...

# Signal-to-Noise Quality Indicator (SNQI)
- **description**: ...
- **definition**: ...

# CCS Approximation
- **description**: Simplified CCS calculation using direct signal-to-noise ratio values when full SNQI data is unavailable
- **definition**: (1 - FalsePosProb) * DecodeConf * ...

# High Confidence Signals
- **description**: Signal with Confirmation Confidence Score (CCS) > 0.8, indicating high reliability
- **definition**: CCS > 0.8

# Observation-Verified Signal
- **description**: ...
- **definition**: ...

# High-Confidence Technosignature
- **description**: ...
- **definition**: ...
```

### Example: `knowledge_base/ntm.md` (has a distance-2 tail)

NTM's 1-hop neighbors are `Bandwidth-Frequency Ratio (BFR)` (prerequisite) and
`NTM Classification System` (dependent), both rendered in full. `NTM
Classification System`'s own dependent `Research Critical Signal` is
distance 2 from NTM, so it is named only:

```
# Narrowband Technological Marker (NTM)
...full block...

# Bandwidth-Frequency Ratio (BFR)
...full block...

# NTM Classification System
...full block...

The following are other related definitions (up to 10):
- "Research Critical Signal" needs "NTM Classification System"
- ...
```

## Contract change (breaking, intentional)

This replaces the documented/tested guarantee "entries that depend on `name`
are excluded" with "1-hop dependents are included and fully rendered;
2-plus-hop dependents are named only, capped at 10." Existing tests in
`tests/eval_framework/agents/test_utils_kb_linearize.py` encode the old
contract and must be rewritten, not just left passing incidentally:

- `test_linearize_prerequisites_excludes_dependents` — currently asserts a
  1-hop dependent (`Top (C)`, where `A prereq_of B`, `B prereq_of C`, queried
  on `B`) is fully absent. Rewrite: `C` must now appear **fully rendered**
  (distance 1 from `B`). Add a distinct case for a distance-2 dependent (`D`
  depends on `C` depends on `B`) to confirm it is named-only, not fully
  rendered, and not excluded either.
- `test_linearize_prerequisites_includes_transitive_ancestors` — currently
  only checks name-only presence for a 1-hop ancestor. Update to assert the
  1-hop ancestor is now **fully rendered** (description/definition present,
  not just the edge line).
- `test_linearize_prerequisites_skips_masked_ancestor` — keep the "masked
  entry never appears" assertion; behavior unchanged for this case (masking
  applies identically regardless of direction).
- `test_linearize_prerequisites_leaf_has_no_edges_block` — a leaf with no
  prerequisites and no dependents still renders no edges/neighbor section.
  Add a sibling case: a leaf with a 1-hop dependent must show that dependent
  fully, still with no "needs" edge line for the leaf itself (since the leaf
  has no prerequisites of its own).

New tests to add (locking in the distance boundary precisely):

- 1-hop dependent → full block (description + definition both present).
- 1-hop prerequisite → full block (description + definition both present).
- distance-2 node (either direction) → name only (`- "X" needs "Y"`),
  description/definition text absent.
- distance-≥2 set larger than `MAX_DEPTH` → capped at 10, with the existing
  "Showing only the first N" notice preserved.
- a node reachable via **both** a distance-1 and a distance-2 path (diamond
  shape) is rendered once, at its shortest distance (full block, not also
  listed in the name-only tail).

## Docstring updates

`linearize_prerequisites`'s docstring currently says "Dependents are excluded."
Replace with the distance-based contract above. Update these to match:

- `agents/CLAUDE.md`'s one-line description of `linearize_prerequisites`.
- `agents/bird_baseline/tools/CLAUDE.md`'s line documenting
  `get_knowledge_definition`: "...the entry plus its transitive prerequisites
  (via `linearize_prerequisites`) — the dependency-edges + topo-ordered
  definitions section, scoped to the looked-up entry's ancestors (dependents
  excluded)" — the "(dependents excluded)" clause is no longer accurate.

## Out of scope

- `linearize_kb` / `get_all_knowledge_definitions` — untouched.
- `build_kb_overview` / the `database_overview.md` KB index — untouched (flat
  index only, no neighbor expansion there).
- No change to `MAX_DEPTH`'s value (stays 10) or to how prerequisite-only
  traversal depth is bounded (still unbounded collection, capped only at
  display time) — the same cap now also bounds the combined distance-≥2
  name-only tail.

## Testing

- `tests/eval_framework/agents/test_utils_kb_linearize.py`: rewrites + new
  cases listed above.
- `tests/eval_framework/agents/deep_agent/test_catalog_seed.py`: spot-check
  that a generated `knowledge_base/<node>.md` file contains a 1-hop
  dependent's full text (regression guard for the deep_agent call site).
- `tests/eval_framework/tools/test_bird_interact_env_tools.py::test_get_knowledge_definition_linearized_includes_transitive_prerequisites`
  breaks under the new contract: it asserts the literal edge string
  `'"derived (DRV)" needs "base (BASE)"'` for a 1-hop prerequisite, which the
  new design renders as a full block instead (no "needs" line at distance 1).
  Update the assertion to check for the full block (description/definition
  text) instead of the edge string; add a 3-node case (`derived2` depending
  on `derived`, itself depending on `base`) to confirm `base` — now
  distance 2 from `derived2` — keeps the old "needs" edge-string form.
- `uv run pytest tests/` after implementation (repo convention).
