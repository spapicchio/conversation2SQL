# Knowledge Base (KB) Structure and Linearization Strategies

## What the KB is

Each BIRD-Interact database ships with an external knowledge base: a small set of domain-specific
facts the agent needs to correctly answer queries. Examples include formulas ("Net Profit Margin =
Net Profit / Revenue"), decision rules ("a plant is considered active if status = 1"), and value
mappings ("severity 3 means critical").

### Data model

Each entry is an `ExternalKnowledgeEntry` with these fields:

| Field | Type | Description |
|---|---|---|
| `id` | `int` | Unique integer ID |
| `knowledge` | `str` | Human-readable name, often with a trailing `(ACRONYM)` |
| `description` | `str` | Plain-English explanation |
| `definition` | `str` | Formula or rule, written in LaTeX |
| `type` | `str` | One of `calculation_knowledge`, `domain_knowledge`, `value_illustration` |
| `children_knowledge` | `list[int]` | IDs of entries this entry depends on (prerequisites) |

### Graph structure

Entries form a **DAG**: `children_knowledge` points to prerequisites. For example, "Net Profit
Margin (NPM)" depends on "Net Profit (NP)" and "Revenue (REV)". Leaves have `children_knowledge =
[]`.

In practice, KB sizes range from ~5 entries (simple databases) to ~30+ (complex ones). Graphs are
shallow (depth 2–3 is typical) but wide.

### Ambiguity masking

When `make_data_ambiguous=True`, entries involved in a KB ambiguity are **deleted** from the KB
before it is handed to the agent (`masked_agent_kb`). The agent must ask the user to resolve the
ambiguity — it cannot look up the deleted entry.

---

## How the KB reaches the agent

The agent has three KB-related tools (all in `bird_interact_env_tools.py`):

| Tool | Cost | Returns |
|---|---|---|
| `get_all_external_knowledge_names` | 0.5 | List of entry names (keys) |
| `get_knowledge_definition` | 0.5 | Definition of one named entry |
| `get_all_knowledge_definitions` | 1.0 | Definitions of all entries |

When `is_kb_linearized=True`, tools read from `masked_agent_kb_linearized` instead of
`masked_agent_kb`. What that field contains depends on the linearization strategy.

---

## Current implementation (per-entry subgraph expansion)

`linearize_kb` calls `linearize_subgraph(entry, kb)` for every entry. Each call does a BFS from
the entry and collects its full transitive closure (the entry + all its prerequisites), then emits:

1. Dependency triples for the subgraph
2. All definitions in topological order (leaves first)

Result type: `dict[str, str]` — one string per entry, keyed by knowledge name.

### Example

KB: A (leaf), B (leaf), C → {A, B}, D → {C}

```
linearized["A"] = "[A] ..."
linearized["B"] = "[B] ..."
linearized["C"] = "(A, prerequisite_of, C)\n(B, prerequisite_of, C)\n[A]...\n[B]...\n[C]..."
linearized["D"] = "(A, prerequisite_of, C)\n(B, prerequisite_of, C)\n(C, prerequisite_of, D)\n[A]...\n[B]...\n[C]...\n[D]..."
```

### Problems

- **Redundancy**: A appears 3 times across the dict (in A, C, D). B also appears 3 times.
  Any entry that is a shared dependency is repeated once per downstream dependent.
- **Token waste on single-entry lookup**: `get_knowledge_definition("D")` returns A, B, and C in
  full even if the agent already retrieved them in prior turns.
- **Inconsistent granularity**: leaf entries return a single line; deep dependents return a
  paragraph — making it hard to predict tool output size.

---

## Strategy 1: Single flat KB string (recommended)

Produce **one string** for the whole KB: all dependency edges once at the top, then all entries
once in topological order (leaves first). Store this as a single `str` field.

```
# Dependency edges (prerequisite -> dependent)
(NP, prerequisite_of, NPM)
(REV, prerequisite_of, NPM)

# Definitions (topological order: leaves first)
[NP] Net Profit - earnings after all expenses - formula: Revenue - Total Costs
[REV] Revenue - total income from operations - formula: sum of all sales
[NPM] Net Profit Margin - ratio of profit to revenue - formula: (NP) / (REV)
```

`get_all_knowledge_definitions` returns this string.
`get_knowledge_definition(name)` extracts just the `[TOKEN]` line for that entry.

**Pros**
- Zero redundancy: every node appears exactly once.
- One `get_all_knowledge_definitions` call gives the complete picture — the LLM can reason over
  all relationships without further calls.
- Predictable and compact: token cost scales linearly with KB size, not with graph depth.
- Trivial to implement on top of the existing `_topological_sort` helper.

**Cons**
- `masked_agent_kb_linearized` changes type from `dict[str, str]` to `str` — requires updating
  `state.py`, the tool implementations, and any code that reads this field.
- `get_knowledge_definition` becomes a substring lookup rather than a dict access. Need to handle
  entries whose names share prefixes carefully.
- The agent loses the ability to get "entry + its context" in one call; it must read the full KB
  or combine the names list + single-entry lookup.

**Watch out for**
- Masked entries: the flat string must be built from `masked_agent_kb`, not `full_knowledge_base`,
  so deleted entries are absent from both the edges and the definitions sections.
- LaTeX in `definition`: `simplify_latex` does best-effort conversion; some exotic constructs pass
  through unchanged. Always verify on a sample with complex formulas.
- Empty KB: if `masked_agent_kb` is empty (all entries masked), return an empty string or a
  sentinel like `"No external knowledge available."` rather than bare headers.

---

## Strategy 2: Per-entry shallow (entry only, deps by name)

Keep `dict[str, str]` format. Each entry's string contains only its own definition line, plus a
`depends_on: [NAME1, NAME2]` annotation for direct dependencies — no content from deps.

```
linearized["NPM"] = "[NPM] Net Profit Margin - ratio of profit to revenue - formula: (NP) / (REV)\ndepends_on: [Net Profit, Revenue]"
```

**Pros**
- Zero redundancy: each node described exactly once, in its own key.
- Preserves per-entry lookup — `get_knowledge_definition("NPM")` returns only NPM.
- No type change to `masked_agent_kb_linearized`.

**Cons**
- Agent must chase dependencies manually: to fully understand NPM it needs 2–3 separate tool
  calls (one per dependency level). Each call costs 0.5 coins — adds up quickly on deep chains.
- Dependency names in `depends_on` must exactly match the keys of `masked_agent_kb`. If a
  dependency is masked (ambiguity), it will be absent from the KB but still listed in `depends_on`
  — could confuse the agent.

**Watch out for**
- The `depends_on` list should be filtered to only include dependencies that are actually present
  in `masked_agent_kb`. Omit masked deps entirely (the agent should discover their absence via
  `get_knowledge_definition` returning "Knowledge not found.").
- Coin cost: a chain of depth 3 costs 1.5 coins just to resolve one entry. Consider whether the
  0.5/call pricing still makes sense under this strategy.

---

## Strategy 3: Hybrid — flat string + per-entry index

Store two things:
1. A single flat string (`_all` key or a dedicated field) — used by `get_all_knowledge_definitions`
2. A `dict[str, str]` where each value is just the single line for that entry (no subgraph) — used
   by `get_knowledge_definition`

This is essentially Strategy 1 + Strategy 2 combined.

**Pros**
- Best of both: one call for the full picture, cheap per-entry lookup for targeted access.
- No chasing dependencies (the flat string gives full context).
- Backward-compatible dict interface preserved.

**Cons**
- Slightly more complex to build and store.
- Two representations to keep in sync — if masking logic changes, both must be updated together.

**Watch out for**
- Keep both representations built from the same masked KB in a single pass so they can't diverge.

---

## Recommendation

Start with **Strategy 1** (single flat string). It is the simplest to implement correctly, easiest
to test, and best matches how an LLM actually benefits from KB context: seeing the complete DAG in
one structured block is more useful than assembling it across multiple tool calls. Migrate to
Strategy 3 only if you observe the agent making redundant `get_all_knowledge_definitions` calls
and want to make single-entry lookup cheaper.
