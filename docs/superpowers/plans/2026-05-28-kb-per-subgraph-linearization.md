# KB Per-Subgraph Linearization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split `linearize_kb` output into one numbered `# Subgraph N` section per connected component of the KB DAG, each with its own dependency-edges block and topological-order definitions block.

**Architecture:** Add a `_find_connected_components` helper that treats the DAG as undirected and returns components sorted by minimum node id; refactor `linearize_kb` to call it and emit one section per component, joining sections with a blank line. `format_entry_line` is unchanged.

**Tech Stack:** Python 3.12, `uv run pytest`, existing `ExternalKnowledgeEntry` dataclass, no new dependencies.

---

## File Map

| Action | Path | What changes |
|--------|------|--------------|
| Modify | `src/conversation2sql/eval_framework/agents/utils_kb_linearize.py` | Add `_find_connected_components`; rewrite `linearize_kb` body |
| Modify | `tests/eval_framework/agents/test_utils_kb_linearize.py` | Add 4 new tests for multi-component and subgraph-header behaviour |

No other files need to change: `format_entry_line` signature and return type are unchanged; callers in `bird_interact_env_tools.py` consume the string opaquely.

---

## Task 1: Write failing tests for the new subgraph-numbered format

**Files:**
- Modify: `tests/eval_framework/agents/test_utils_kb_linearize.py`

- [ ] **Step 1.1: Append the four new tests to the test file**

Add the following block at the bottom of `tests/eval_framework/agents/test_utils_kb_linearize.py`:

```python
# ---------------------------------------------------------------------------
# Per-subgraph (connected-component) format
# ---------------------------------------------------------------------------

def test_single_entry_has_subgraph_1_header():
    """Even a single-entry KB must start with # Subgraph 1."""
    kb = {"Foo (F)": _entry(1, "Foo (F)", description="the foo")}
    result = linearize_kb(kb)
    assert result.startswith("# Subgraph 1")


def test_two_disconnected_entries_produce_two_subgraphs():
    """Two entries with no edge between them → two separate subgraph sections."""
    a = _entry(1, "Alpha (A)", description="first")
    b = _entry(2, "Beta (B)", description="second")
    kb = {"Alpha (A)": a, "Beta (B)": b}
    result = linearize_kb(kb)
    assert "# Subgraph 1" in result
    assert "# Subgraph 2" in result
    # Neither entry appears in the other's subgraph
    parts = result.split("# Subgraph 2")
    assert "[A]" not in parts[1]  # A belongs to subgraph 1, not 2


def test_two_components_each_with_edges():
    """Two separate chains each emit their own edges + definitions blocks."""
    # Component 1: X → Y
    x = _entry(1, "X (X)", description="base x")
    y = _entry(2, "Y (Y)", description="uses x", children=[1])
    # Component 2: P → Q
    p = _entry(3, "P (P)", description="base p")
    q = _entry(4, "Q (Q)", description="uses p", children=[3])
    kb = {"X (X)": x, "Y (Y)": y, "P (P)": p, "Q (Q)": q}
    result = linearize_kb(kb)
    assert "# Subgraph 1" in result
    assert "# Subgraph 2" in result
    # Each component has its own edge triple
    assert "(X, prerequisite_of, Y)" in result
    assert "(P, prerequisite_of, Q)" in result
    # Edges from component 1 don't bleed into component 2's section
    sub2 = result.split("# Subgraph 2")[1]
    assert "(X, prerequisite_of, Y)" not in sub2


def test_subgraph_ordering_by_min_node_id():
    """Subgraphs are ordered by the smallest node id in each component."""
    # Component with min id=5 and component with min id=1
    high = _entry(5, "High (H)", description="high id")
    low  = _entry(1, "Low (L)", description="low id")
    kb = {"High (H)": high, "Low (L)": low}
    result = linearize_kb(kb)
    # Subgraph 1 should contain the entry with id=1 (Low)
    sub1 = result.split("# Subgraph 2")[0]
    assert "[L]" in sub1
```

- [ ] **Step 1.2: Run the new tests and confirm they all FAIL**

```bash
uv run pytest tests/eval_framework/agents/test_utils_kb_linearize.py -v -k "subgraph"
```

Expected: 4 FAILED (current output has no `# Subgraph N` headers).

---

## Task 2: Implement `_find_connected_components` and update `linearize_kb`

**Files:**
- Modify: `src/conversation2sql/eval_framework/agents/utils_kb_linearize.py`

- [ ] **Step 2.1: Add `_find_connected_components` after `_topological_sort`**

In `utils_kb_linearize.py`, insert the following function immediately after the `_topological_sort` function (after line 185):

```python
def _find_connected_components(
    nodes: list[ExternalKnowledgeEntry],
) -> list[list[ExternalKnowledgeEntry]]:
    """Return connected components sorted by minimum node id (undirected view of DAG)."""
    node_by_id = {n.id: n for n in nodes}
    id_set = set(node_by_id)

    neighbors: dict[int, set[int]] = {n.id: set() for n in nodes}
    for n in nodes:
        for child_id in (n.children_knowledge or []):
            if child_id in id_set:
                neighbors[n.id].add(child_id)
                neighbors[child_id].add(n.id)

    visited: set[int] = set()
    components: list[list[ExternalKnowledgeEntry]] = []
    for n in sorted(nodes, key=lambda e: e.id):
        if n.id in visited:
            continue
        queue = [n.id]
        component_ids: list[int] = []
        while queue:
            curr = queue.pop()
            if curr in visited:
                continue
            visited.add(curr)
            component_ids.append(curr)
            queue.extend(neighbors[curr] - visited)
        components.append([node_by_id[nid] for nid in component_ids])

    components.sort(key=lambda comp: min(e.id for e in comp))
    return components
```

- [ ] **Step 2.2: Replace the body of `linearize_kb`**

Replace the entire `linearize_kb` function body (keeping its docstring and signature) with:

```python
def linearize_kb(masked_agent_kb: dict[str, ExternalKnowledgeEntry]) -> str:
    """Strategy 1: single flat string for the whole (masked) KB.

    Empty KB -> "".
    Each connected component gets its own # Subgraph N section.
    """
    if not masked_agent_kb:
        return ""

    entries = list(masked_agent_kb.values())
    components = _find_connected_components(entries)

    sections: list[str] = []
    for idx, component in enumerate(components, start=1):
        ordered = _topological_sort(component)
        in_kb = {n.id for n in ordered}
        token_of = {n.id: _extract_token(n.knowledge) for n in ordered}

        edges = [
            (token_of[child_id], token_of[n.id])
            for n in ordered
            for child_id in (n.children_knowledge or [])
            if child_id in in_kb
        ]

        lines: list[str] = [f"# Subgraph {idx}"]
        if edges:
            lines.append("# Dependency edges (prerequisite -> dependent)")
            lines.extend(f"({a}, prerequisite_of, {b})" for a, b in edges)
            lines.append("")
        lines.append("# Definitions (topological order: leaves first)")
        for n in ordered:
            lines.append(_format_line(n, token_of[n.id]))
        sections.append("\n".join(lines))

    return "\n\n".join(sections)
```

- [ ] **Step 2.3: Run ALL tests and confirm everything passes**

```bash
uv run pytest tests/eval_framework/agents/test_utils_kb_linearize.py -v
```

Expected: 13 passed (9 original + 4 new).

Check that the two tests most sensitive to output structure still pass:
- `test_format_entry_line_matches_linearize_kb_line` — filters lines starting with `[`, unaffected by `# Subgraph` headers.
- `test_sentinel_children_minus_one_treated_as_no_deps` — two isolated nodes, two components, still no `# Dependency edges` anywhere.

- [ ] **Step 2.4: Commit**

```bash
git add src/conversation2sql/eval_framework/agents/utils_kb_linearize.py \
        tests/eval_framework/agents/test_utils_kb_linearize.py
git commit -m "feat: linearize KB as per-subgraph sections (connected components)"
```

---

## Task 3: Run full test suite

**Files:** none changed

- [ ] **Step 3.1: Run full suite**

```bash
uv run pytest tests/ -v
```

Expected: all previously passing tests still pass; the 4 new tests pass.

- [ ] **Step 3.2: Commit docs**

```bash
git add docs/superpowers/plans/2026-05-28-kb-per-subgraph-linearization.md
git commit -m "docs: add implementation plan for KB per-subgraph linearization"
```
