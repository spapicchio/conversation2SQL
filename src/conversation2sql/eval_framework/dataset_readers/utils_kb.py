"""Helpers for presenting the external knowledge base (KB) to the agent.

The KB is a small DAG: each entry depends on zero or more "children". Rather
than dump every entry as a flat list (or as raw LaTeX), we build a per-entry
*subgraph linearization*: the entry plus its transitive prerequisites,
emitted as

  1. dependency triples ``(prereq, prerequisite_of, dependent)`` — the
     structural backbone, and
  2. one-line definitions in topological order (leaves first), with the
     formula rewritten from LaTeX into a code-like notation.

Single-node entries collapse to just the definition line.
"""

from __future__ import annotations

import re
from collections import deque

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

# Commands of the form ``\name{arg}`` that should be replaced by the bare arg.
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
    """Return ``(inner, idx_after_close)`` for the braced group at ``s[start]``.

    If ``s[start]`` is not ``{``, returns ``("", start)``.
    """
    if start >= len(s) or s[start] != "{":
        return "", start
    depth = 0
    for i in range(start, len(s)):
        ch = s[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return s[start + 1 : i], i + 1
    return s[start + 1 :], len(s)


def _convert_cases(body: str) -> str:
    """Convert the body of ``\\begin{cases}...\\end{cases}`` into ``if/elif/else``.

    Rows are separated by ``\\\\``; each row is ``expression & condition``.
    Conditions wrapped in ``\\text{...}`` are unwrapped here so the result is
    readable even before further LaTeX simplification passes run.
    """
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
    """One simplification pass: handle the leftmost LaTeX braced command.

    The caller runs this to a fixed point. Order of preference is determined
    purely by source position so nesting works naturally.
    """
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
        replacement = _convert_cases(s[body_start:body_end])
        return s[:idx] + replacement + s[idx + end_match.end() :]

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

    # Unwrap commands: \text{X}, \mathrm{X}, etc. -> X
    x, after = _extract_braced(s, brace_start)
    if after == brace_start:
        return s
    return s[:idx] + x + s[after:]


def simplify_latex(definition: str) -> str:
    """Rewrite a LaTeX-formatted formula into simple, code-like notation.

    Best-effort and lossy: covers the constructs the BIRD-Interact KB actually
    uses (``\\frac``, ``\\text``, ``\\sqrt``, ``\\begin{cases}``, simple
    operators). Anything we don't recognise is passed through.
    """
    if not definition:
        return definition

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
# Subgraph linearization
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"\(([A-Za-z][A-Za-z0-9_]{1,15})\)\s*$")


def _extract_token(knowledge: str) -> str:
    """Pick a short token for triples — the trailing ``(ACRONYM)`` if present."""
    m = _TOKEN_RE.search(knowledge)
    return m.group(1) if m else knowledge.strip()


def _strip_token(knowledge: str) -> str:
    return _TOKEN_RE.sub("", knowledge).strip()


def _id_index(
    kb: dict[str, ExternalKnowledgeEntry],
) -> dict[int, ExternalKnowledgeEntry]:
    return {entry.id: entry for entry in kb.values()}


def _collect_subgraph(
    root: ExternalKnowledgeEntry,
    by_id: dict[int, ExternalKnowledgeEntry],
) -> list[ExternalKnowledgeEntry]:
    """BFS from ``root`` over ``children_knowledge`` IDs that exist in ``by_id``."""
    seen: set[int] = {root.id}
    out: list[ExternalKnowledgeEntry] = [root]
    queue: deque[ExternalKnowledgeEntry] = deque([root])
    while queue:
        node = queue.popleft()
        for child_id in node.children_knowledge or []:
            if child_id in by_id and child_id not in seen:
                seen.add(child_id)
                child = by_id[child_id]
                out.append(child)
                queue.append(child)
    return out


def _topological_sort(
    nodes: list[ExternalKnowledgeEntry],
) -> list[ExternalKnowledgeEntry]:
    """Leaves first, dependents last. Stable on entry id within each layer."""
    node_ids = {n.id for n in nodes}
    prereqs: dict[int, set[int]] = {
        n.id: {c for c in (n.children_knowledge or []) if c in node_ids}
        for n in nodes
    }
    remaining = {n.id: n for n in nodes}
    placed: set[int] = set()
    ordered: list[ExternalKnowledgeEntry] = []

    while remaining:
        ready = sorted(
            nid for nid, _ in remaining.items() if prereqs[nid].issubset(placed)
        )
        if not ready:
            # Cycle (shouldn't happen on a real DAG) — break it deterministically.
            ready = [min(remaining)]
        for nid in ready:
            ordered.append(remaining.pop(nid))
            placed.add(nid)

    return ordered


def linearize_subgraph(
    root: ExternalKnowledgeEntry,
    kb: dict[str, ExternalKnowledgeEntry],
) -> str:
    """Linearize the subgraph rooted at ``root`` (root + transitive prereqs).

    Prerequisites that are not present in ``kb`` (e.g. masked away) are
    silently dropped from both the triples and the definitions list.
    """
    by_id = _id_index(kb)
    by_id.setdefault(root.id, root)

    subgraph = _collect_subgraph(root, by_id)
    ordered = _topological_sort(subgraph)
    token_of = {n.id: _extract_token(n.knowledge) for n in ordered}
    in_subgraph = {n.id for n in ordered}

    lines: list[str] = []

    edges: list[tuple[str, str]] = []
    for n in ordered:
        for child_id in n.children_knowledge or []:
            if child_id in in_subgraph:
                edges.append((token_of[child_id], token_of[n.id]))

    if edges:
        lines.append("# Dependency edges (prerequisite -> dependent)")
        lines.extend(f"({a}, prerequisite_of, {b})" for a, b in edges)
        lines.append("")

    lines.append("# Definitions (topological order: leaves first)")
    for n in ordered:
        token = token_of[n.id]
        name = _strip_token(n.knowledge)
        desc = (n.description or "").strip().rstrip(".")
        formula = simplify_latex(n.definition or "")
        suffix = f" - formula: {formula}" if formula else ""
        prefix = f"[{token}] {name}" if name else f"[{token}]"
        if desc:
            lines.append(f"{prefix} - {desc}{suffix}")
        else:
            lines.append(f"{prefix}{suffix}")

    return "\n".join(lines)


def linearize_kb(
    kb: dict[str, ExternalKnowledgeEntry],
) -> dict[str, str]:
    """Per-entry linearized subgraphs, keyed by the original ``knowledge`` name."""
    return {name: linearize_subgraph(entry, kb) for name, entry in kb.items()}
