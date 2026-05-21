"""Tests for the Strategy-1 KB linearizer in agents/utils_kb_linearize.py."""
from __future__ import annotations

from conversation2sql.eval_framework.agents.utils_kb_linearize import (
    format_entry_line,
    linearize_kb,
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
    assert "# Dependency edges" not in result
    assert "# Definitions" in result
    assert "[AU] Active User" in result
    assert "days_since_login <= 30" in result


def test_two_entries_prerequisite_produces_edge():
    a = _entry(1, "Net Profit (NP)", definition="Revenue - Costs")
    b = _entry(2, "Net Profit Margin (NPM)",
               definition=r"\frac{NP}{REV}", children=[1])
    kb = {"Net Profit (NP)": a, "Net Profit Margin (NPM)": b}
    result = linearize_kb(kb)
    assert "# Dependency edges" in result
    assert "(NP, prerequisite_of, NPM)" in result
    assert "# Definitions" in result
    def_section = result.split("# Definitions")[1]
    assert def_section.index("[NP]") < def_section.index("[NPM]")


def test_shared_prereq_appears_once_in_definitions():
    a = _entry(1, "Base (B)")
    b = _entry(2, "Derived1 (D1)", children=[1])
    c = _entry(3, "Derived2 (D2)", children=[1])
    kb = {"Base (B)": a, "Derived1 (D1)": b, "Derived2 (D2)": c}
    result = linearize_kb(kb)
    assert result.count("[B] Base") == 1
    assert "(B, prerequisite_of, D1)" in result
    assert "(B, prerequisite_of, D2)" in result


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
    def_lines = [l for l in flat.splitlines() if l.startswith("[")]
    assert len(def_lines) == 1
    assert format_entry_line("Active User (AU)", kb) == def_lines[0]


def test_format_entry_line_no_subgraph_context():
    a = _entry(1, "Base (B)", description="the base")
    b = _entry(2, "Derived (D)", description="uses base", children=[1])
    kb = {"Base (B)": a, "Derived (D)": b}
    line = format_entry_line("Derived (D)", kb)
    assert "[B]" not in line
    assert "[D]" in line
