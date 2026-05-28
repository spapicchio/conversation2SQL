"""Strategy-1 KB linearizer: per-subgraph sections for the KB DAG.

Public interface
----------------
linearize_kb(masked_agent_kb)           -> str   (one # Subgraph N section per connected component)
format_entry_line(name, masked_agent_kb) -> str  (one definition line, or not-found sentinel)
"""
from __future__ import annotations

import re

from conversation2sql.eval_framework.state import ExternalKnowledgeEntry

# ---------------------------------------------------------------------------
# LaTeX -> code-like notation
# ---------------------------------------------------------------------------
_SIMPLE_LATEX_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    (r"\times", " * "),
    (r"\cdot", " * "),
    (r"\div", " / "),
    (r"\leq", " <= "),
    (r"\geq", " >= "),
    (r"\neq", " != "),
    (r"\le ", " <= "),
    (r"\ge ", " >= "),
    (r"\ne ", " != "),
    (r"\pm", " +/- "),
    (r"\left", ""),
    (r"\right", ""),
    (r"\,", " "),
    (r"\;", " "),
    (r"\:", " "),
    (r"\!", ""),
    (r"\%", "%"),
    (r"\$", "$"),
)

_UNWRAP_COMMANDS: tuple[str, ...] = (
    r"\text",
    r"\textit",
    r"\textbf",
    r"\textrm",
    r"\mathrm",
    r"\mathbf",
    r"\mathit",
    r"\mathsf",
    r"\operatorname",
)


def _extract_braced(s: str, start: int) -> tuple[str, int]:
    if start >= len(s) or s[start] != "{":
        return "", start
    depth = 0
    for i in range(start, len(s)):
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                return s[start + 1 : i], i + 1
    return s[start + 1 :], len(s)


def _convert_cases(body: str) -> str:
    rows = re.split(r"\\\\", body)
    out: list[str] = []
    first = True
    for row in rows:
        row = row.strip()
        if not row:
            continue
        parts = row.split("&", 1)
        if len(parts) != 2:
            out.append(row)
            continue
        expr = parts[0].strip()
        cond = re.sub(r"\\text\{([^{}]*)\}", r"\1", parts[1]).strip()
        if "otherwise" in cond.lower() or cond.lower() in {"", "else"}:
            out.append(f"else: {expr}")
            first = False
            continue
        keyword = "if" if first else "elif"
        m = re.match(r"if\s+(.*)", cond, flags=re.IGNORECASE)
        condition = m.group(1).strip() if m else cond
        out.append(f"{keyword} {condition}: {expr}")
        first = False
    return "{ " + "; ".join(out) + " }"


def _process_braced_commands(s: str) -> str:
    candidates: list[tuple[str, int]] = []
    m = re.search(r"\\begin\{cases\}", s)
    if m:
        candidates.append(("cases", m.start()))
    for cmd in (r"\frac", r"\sqrt", *_UNWRAP_COMMANDS):
        idx = s.find(cmd)
        if idx >= 0:
            candidates.append((cmd, idx))
    if not candidates:
        return s
    candidates.sort(key=lambda pair: pair[1])
    cmd, idx = candidates[0]
    if cmd == "cases":
        end_match = re.search(r"\\end\{cases\}", s[idx:])
        if not end_match:
            return s
        body_start = idx + len(r"\begin{cases}")
        body_end = idx + end_match.start()
        return s[:idx] + _convert_cases(s[body_start:body_end]) + s[idx + end_match.end():]
    brace_start = idx + len(cmd)
    if cmd == r"\frac":
        a, after_a = _extract_braced(s, brace_start)
        if after_a == brace_start:
            return s
        b, after_b = _extract_braced(s, after_a)
        if after_b == after_a:
            return s
        return s[:idx] + f"({a}) / ({b})" + s[after_b:]
    if cmd == r"\sqrt":
        x, after = _extract_braced(s, brace_start)
        if after == brace_start:
            return s
        return s[:idx] + f"sqrt({x})" + s[after:]
    x, after = _extract_braced(s, brace_start)
    if after == brace_start:
        return s
    return s[:idx] + x + s[after:]


def simplify_latex(definition: str) -> str:
    """Rewrite LaTeX formula into simple code-like notation (best-effort)."""
    if not definition:
        return ""
    s = definition
    for old, new in _SIMPLE_LATEX_REPLACEMENTS:
        s = s.replace(old, new)
    prev: str | None = None
    while prev != s:
        prev = s
        s = _process_braced_commands(s)
    s = re.sub(r"\^\{([^{}]*)\}", r"**(\1)", s)
    s = re.sub(r"\^(\w)", r"**\1", s)
    s = re.sub(r"\\\\", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ---------------------------------------------------------------------------
# Token helpers
# ---------------------------------------------------------------------------
_TOKEN_RE = re.compile(r"\(([A-Za-z][A-Za-z0-9_]{0,15})\)\s*$")


def _extract_token(knowledge: str) -> str:
    m = _TOKEN_RE.search(knowledge)
    return m.group(1) if m else knowledge.strip()


def _strip_token(knowledge: str) -> str:
    return _TOKEN_RE.sub("", knowledge).strip()


# ---------------------------------------------------------------------------
# Topological sort (Kahn's algorithm, stable on id)
# ---------------------------------------------------------------------------
def _topological_sort(
    nodes: list[ExternalKnowledgeEntry],
) -> list[ExternalKnowledgeEntry]:
    node_ids = {n.id for n in nodes}
    prereqs: dict[int, set[int]] = {
        n.id: {c for c in (n.children_knowledge or []) if c in node_ids}
        for n in nodes
    }
    remaining = {n.id: n for n in nodes}
    placed: set[int] = set()
    ordered: list[ExternalKnowledgeEntry] = []
    while remaining:
        ready = sorted(nid for nid in remaining if prereqs[nid].issubset(placed))
        if not ready:
            ready = [min(remaining)]
        for nid in ready:
            ordered.append(remaining.pop(nid))
            placed.add(nid)
    return ordered


# ---------------------------------------------------------------------------
# Connected-component decomposition
# ---------------------------------------------------------------------------
def _find_connected_components(
    nodes: list[ExternalKnowledgeEntry],
) -> list[list[ExternalKnowledgeEntry]]:
    """Return connected components (DFS traversal) sorted by minimum node id (undirected view of DAG)."""
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
        stack = [n.id]
        component_ids: list[int] = []
        while stack:
            curr = stack.pop()
            if curr in visited:
                continue
            visited.add(curr)
            component_ids.append(curr)
            stack.extend(neighbors[curr] - visited)
        components.append([node_by_id[nid] for nid in component_ids])

    # components are already in min-id order (outer loop iterates nodes ascending)
    return components


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------
def _format_line(entry: ExternalKnowledgeEntry, token: str) -> str:
    name = _strip_token(entry.knowledge)
    desc = (entry.description or "").strip().rstrip(".")
    formula = simplify_latex(entry.definition or "")
    suffix = f" - formula: {formula}" if formula else ""
    prefix = f"[{token}] {name}" if name else f"[{token}]"
    return f"{prefix} - {desc}{suffix}" if desc else f"{prefix}{suffix}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def linearize_kb(masked_agent_kb: dict[str, ExternalKnowledgeEntry]) -> str:
    """Strategy 1: one # Subgraph N section per connected component of the (masked) KB.

    Empty KB -> "".
    Single component -> one section. Multiple disconnected components -> multiple sections.
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


def format_entry_line(
    name: str,
    masked_agent_kb: dict[str, ExternalKnowledgeEntry],
) -> str:
    """Return the single [TOKEN] definition line for `name`.

    Returns "Knowledge not found." when `name` is absent from the KB.
    """
    entry = masked_agent_kb.get(name)
    if entry is None:
        return "Knowledge not found."
    return _format_line(entry, _extract_token(entry.knowledge))
