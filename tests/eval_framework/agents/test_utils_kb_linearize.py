"""Tests for the Strategy-1 KB linearizer in agents/utils_kb_linearize.py."""
from __future__ import annotations

from conversation2sql.eval_framework.agents.utils_kb_linearize import (
    format_entry_line,
    linearize_kb,
    linearize_prerequisites,
)
from conversation2sql.eval_framework.state import ExternalKnowledgeEntry


def _entry(id: int, knowledge: str, definition: str = "", description: str = "",
           children: list[int] | None = None) -> ExternalKnowledgeEntry:
    return ExternalKnowledgeEntry(
        id=id,
        knowledge=knowledge,
        description=description,
        definition=definition,
        type="domain_knowledge",
        children_knowledge=children or [],
    )


def test_empty_kb_returns_empty_string():
    assert linearize_kb({}) == ""


def test_single_leaf_entry_no_edges_block():
    kb = {"Active User (AU)": _entry(1, "Active User (AU)",
                                     description="logged in ≤ 30 days",
                                     definition="days_since_login <= 30")}
    result = linearize_kb(kb)
    assert "prerequisite" not in result.lower()
    assert "Active User (AU)" in result
    assert "days_since_login <= 30" in result


def test_two_entries_prerequisite_produces_edge():
    a = _entry(1, "Net Profit (NP)", definition="Revenue - Costs")
    b = _entry(2, "Net Profit Margin (NPM)",
               definition=r"\frac{NP}{REV}", children=[1])
    kb = {"Net Profit (NP)": a, "Net Profit Margin (NPM)": b}
    result = linearize_kb(kb)
    assert '"Net Profit Margin (NPM)" needs "Net Profit (NP)"' in result
    assert "Net Profit Margin (NPM)" in result


def test_shared_prereq_appears_once_in_definitions():
    a = _entry(1, "Base (B)")
    b = _entry(2, "Derived1 (D1)", children=[1])
    c = _entry(3, "Derived2 (D2)", children=[1])
    kb = {"Base (B)": a, "Derived1 (D1)": b, "Derived2 (D2)": c}
    result = linearize_kb(kb)
    assert '"Derived1 (D1)" needs "Base (B)"' in result
    assert '"Derived2 (D2)" needs "Base (B)"' in result


def test_latex_frac_simplified():
    kb = {"Rate (R)": _entry(1, "Rate (R)", definition=r"\frac{A}{B}")}
    result = linearize_kb(kb)
    assert "(A) / (B)" in result
    assert r"\frac" not in result


def test_sentinel_children_minus_one_treated_as_no_deps():
    kb = {
        "Alpha": _entry(1, "Alpha", children=[-1]),
        "Beta":  _entry(2, "Beta",  children=[-1]),
    }
    result = linearize_kb(kb)
    assert "# Dependency edges" not in result


def test_format_entry_line_missing_name():
    kb = {"Foo (F)": _entry(1, "Foo (F)")}
    assert format_entry_line("does_not_exist", kb) == "Knowledge not found."


def test_format_entry_line_matches_linearize_kb_line():
    entry = _entry(1, "Active User (AU)",
                   description="logged in recently",
                   definition="days <= 30")
    kb = {"Active User (AU)": entry}
    flat = linearize_kb(kb)
    entry_text = format_entry_line("Active User (AU)", kb)
    assert entry_text in flat


def test_format_entry_line_no_subgraph_context():
    a = _entry(1, "Base (B)", description="the base")
    b = _entry(2, "Derived (D)", description="uses base", children=[1])
    kb = {"Base (B)": a, "Derived (D)": b}
    line = format_entry_line("Derived (D)", kb)
    assert "the base" not in line   # no edge info in single-entry format
    assert "uses base" in line      # description of the requested entry is present


# ---------------------------------------------------------------------------
# linearize_prerequisites — single entry + its transitive prerequisites
# ---------------------------------------------------------------------------

def test_linearize_prerequisites_missing_name():
    kb = {"Foo (F)": _entry(1, "Foo (F)")}
    assert linearize_prerequisites("does_not_exist", kb) == "Knowledge not found."


def test_linearize_prerequisites_leaf_has_no_edges_block():
    """An entry with no prerequisites renders only its own definition, no edges."""
    kb = {"Active User (AU)": _entry(1, "Active User (AU)",
                                     description="logged in recently",
                                     definition="days <= 30")}
    result = linearize_prerequisites("Active User (AU)", kb)
    assert "prerequisite" not in result.lower()
    assert "Active User (AU)" in result


def test_linearize_prerequisites_has_no_subgraph_header():
    """Single-entry lookup must not emit the per-component '# Subgraph N' header."""
    kb = {"Foo (F)": _entry(1, "Foo (F)", description="the foo")}
    result = linearize_prerequisites("Foo (F)", kb)
    assert "# Subgraph" not in result


def test_linearize_prerequisites_includes_transitive_ancestors():
    """Querying B (where A is prereq of B) pulls in A with the edge."""
    a = _entry(1, "Base (A)", definition="x")
    b = _entry(2, "Mid (B)", definition=r"\frac{A}{2}", children=[1])
    kb = {"Base (A)": a, "Mid (B)": b}
    result = linearize_prerequisites("Mid (B)", kb)
    assert '"Mid (B)" needs "Base (A)"' in result
    assert "Mid (B)" in result
    assert "Base (A)" in result


def test_linearize_prerequisites_excludes_dependents():
    """Querying B (A prereq_of B, B prereq_of C) includes ancestor A, excludes dependent C."""
    a = _entry(1, "Base (A)")
    b = _entry(2, "Mid (B)", children=[1])
    c = _entry(3, "Top (C)", children=[2])
    kb = {"Base (A)": a, "Mid (B)": b, "Top (C)": c}
    result = linearize_prerequisites("Mid (B)", kb)
    assert "Base (A)" in result
    assert "Mid (B)" in result
    assert "Top (C)" not in result
    assert '"Mid (B)" needs "Base (A)"' in result
    assert '"Top (C)" needs' not in result


def test_linearize_prerequisites_skips_masked_ancestor():
    """A prerequisite masked out of the KB is silently skipped (no edge to a ghost)."""
    # B depends on id=1, but id=1 is not present in the (masked) KB.
    b = _entry(2, "Mid (B)", children=[1])
    kb = {"Mid (B)": b}
    result = linearize_prerequisites("Mid (B)", kb)
    assert "Mid (B)" in result
    assert "prerequisite" not in result.lower()


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
    """Two separate chains each emit their own edges."""
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
    assert '"Y (Y)" needs "X (X)"' in result
    assert '"Q (Q)" needs "P (P)"' in result
    # Edges from component 1 don't bleed into component 2's section
    sub2 = result.split("# Subgraph 2")[1]
    assert '"Y (Y)" needs "X (X)"' not in sub2


def test_subgraph_ordering_by_min_node_id():
    """Subgraphs are ordered by the smallest node id in each component."""
    # Component with min id=5 and component with min id=1
    high = _entry(5, "High (H)", description="high id")
    low  = _entry(1, "Low (L)", description="low id")
    kb = {"High (H)": high, "Low (L)": low}
    result = linearize_kb(kb)
    # Subgraph 1 should contain the entry with id=1 (Low)
    sub1 = result.split("# Subgraph 2")[0]
    assert "Low (L)" in sub1
