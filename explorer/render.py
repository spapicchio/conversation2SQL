"""Shared Streamlit rendering helpers for conversation records."""

from __future__ import annotations

import html
import json

import streamlit as st

try:  # pragma: no cover - bare import only when Streamlit runs from explorer/
    from colors import stable_color
except ModuleNotFoundError:
    from explorer.colors import stable_color


def _pattern_badge(label: str) -> str:
    """Inline colored HTML chip for an anti-pattern label (color pinned in colors.py)."""
    color = stable_color(label, kind="pattern")
    return (
        f"<span style='background:{color};color:white;border-radius:6px;"
        f"padding:1px 7px;font-size:0.8em;white-space:nowrap'>{html.escape(label)}</span>"
    )


def _pattern_banner_html(hits: list) -> str:
    """A banner listing every anti-pattern a conversation hit, with its evidence."""
    items = "".join(
        f"<div style='margin:3px 0'>{_pattern_badge(h.label)} "
        f"<span style='color:#666;font-size:0.85em'>{html.escape(h.detail)}</span></div>"
        for h in hits
    )
    return (
        "<div style='border-left:4px solid #d62728;padding:6px 12px;margin:8px 0;"
        f"background:#fff5f5;border-radius:4px'>🚩 <b>Anti-patterns flagged "
        f"({len(hits)})</b>{items}</div>"
    )


def render_ai_content(content) -> None:
    """Render AI message content: str passes through, list-of-blocks separates thinking from text."""
    if isinstance(content, str):
        st.markdown(content)
    elif isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                st.markdown(str(block))
                continue
            block_type = block.get("type", "")
            if block_type == "thinking":
                with st.expander("Thinking — click to expand"):
                    st.text(block.get("thinking", ""))
            elif block_type == "text":
                st.markdown(block.get("text", ""))
            else:
                st.json(block)
    else:
        st.text(str(content))


def render_conversation(
    record: dict,
    highlight_indices: set[int] | None = None,
    pattern_hits: list | None = None,
) -> None:
    """Render a conversation trace.

    ``highlight_indices`` flags individual message rows (a generic emphasis).
    ``pattern_hits`` (a list of ``PatternHit``) additionally renders a banner of
    every anti-pattern the conversation hit and tags each flagged turn with the
    label(s) of the pattern(s) that fired there — so a single conversation shows
    *all* its (non-mutually-exclusive) patterns at once. Conversation-level
    patterns (no ``message_indices``) appear in the banner only.
    """
    hits = list(pattern_hits or [])
    # message index -> labels of the pattern(s) that flagged that exact turn.
    idx_labels: dict[int, list[str]] = {}
    for h in hits:
        for mi in getattr(h, "message_indices", None) or []:
            idx_labels.setdefault(mi, []).append(getattr(h, "label", ""))

    acc = record.get("execution_accuracy", False)
    badge = "✓ PASS" if acc else "✗ FAIL"
    st.subheader(
        f"Conversation: `{record.get('instance_id', '')}` · "
        f"db: `{record.get('selected_database', '')}` · **{badge}**"
    )
    if hits:
        st.markdown(_pattern_banner_html(hits), unsafe_allow_html=True)

    # Questions
    clean_q = record.get("not_ambiguos_query", "")
    amb_q = record.get("amb_user_query", "")
    if clean_q:
        st.markdown(f"**Question:** {clean_q}")
    if amb_q and amb_q != clean_q:
        st.markdown(f"**Ambiguous question:** {amb_q}")

    col1, col2 = st.columns(2)
    with col1:
        with st.expander("Ground-truth SQL"):
            sol = record.get("sol_sql", "")
            if isinstance(sol, list):
                sol = "\n\n".join(sol)
            st.code(sol, language="sql")
    with col2:
        with st.expander("Predicted SQL"):
            st.code(record.get("predicted_sql", "") or "", language="sql")

    # Task context: schema, KB, column meanings
    schema = record.get("ddl_database_schema", "")
    # kb_linearized: dict = record.get("masked_agent_kb_linearized") or {}
    kb_linearized: dict = {}
    kb_raw: dict = record.get("masked_agent_kb") or {}
    gt_kb: dict = record.get("gt_knowledge_base") or {}
    col_meanings: dict = record.get("column_meanings") or {}

    ctx_cols = st.columns(4)
    with ctx_cols[0]:
        with st.expander("Schema (DDL)"):
            with st.container(height=300):
                st.code(schema or "(none)", language="sql")
    with ctx_cols[1]:
        with st.expander(f"Agent KB ({len(kb_linearized or kb_raw)} entries)"):
            with st.container(height=300):
                if kb_linearized:
                    for name, text in kb_linearized.items():
                        st.markdown(f"**{name}**")
                        st.text(text)
                elif kb_raw:
                    st.json(kb_raw)
                else:
                    st.text("(none)")
    with ctx_cols[2]:
        with st.expander(f"Ground-truth KB ({len(gt_kb)} entries)"):
            with st.container(height=300):
                if gt_kb:
                    st.json(gt_kb)
                else:
                    st.text("(none)")
    with ctx_cols[3]:
        with st.expander(f"Column meanings ({len(col_meanings)} entries)"):
            with st.container(height=300):
                if col_meanings:
                    st.json(col_meanings)
                else:
                    st.text("(none)")

    st.divider()
    st.markdown("### Conversation trace")

    # Render every message in order as a chronological trace (LangSmith style):
    # the long system prompt stays collapsed; the user question, each assistant
    # turn (reasoning + text + tool-call chips), and each tool result render as
    # their own distinctly-styled turn.
    highlight = (highlight_indices or set()) | set(idx_labels)
    for i, msg in enumerate(record.get("messages", [])):
        role = msg.get("role")
        if i in highlight:
            labels = idx_labels.get(i)
            if labels:
                st.markdown(
                    "🚩 " + " ".join(_pattern_badge(lbl) for lbl in labels),
                    unsafe_allow_html=True,
                )
            else:
                st.markdown("\U0001f6a9 **flagged turn**")

        if role == "system":
            with st.expander("⚙️ System prompt — click to expand"):
                st.text(msg.get("content", ""))

        elif role in ("user", "human"):
            with st.chat_message("user"):
                st.markdown("**User**")
                content = msg.get("content", "")
                st.markdown(content if isinstance(content, str) else f"```\n{content}\n```")

        elif role == "ai":
            with st.chat_message("assistant"):
                st.markdown("**Assistant**")
                content = msg.get("content", "")
                thinking = msg.get("thinking", "")
                if thinking and not isinstance(content, list):
                    with st.expander("Thinking (Not passed in History!) — click to expand"):
                        st.text(thinking)
                if content:
                    render_ai_content(content)
                tool_calls = msg.get("tool_calls", [])
                for tc in tool_calls:
                    tool_name = tc.get("tool_name", "unknown")
                    args = tc.get("arguments", {})
                    args_str = json.dumps(args, ensure_ascii=False)
                    label = (
                        f"🔧 calls `{tool_name}`({args_str[:60]}…)"
                        if len(args_str) > 60
                        else f"🔧 calls `{tool_name}`({args_str})"
                    )
                    with st.expander(label):
                        st.json(args)
                st.caption(
                    f"tokens: {msg.get('prompt_tokens', 0)}↑ {msg.get('completion_tokens', 0)}↓"
                    f" | cost: ${msg.get('cost_usd', 0):.5f}"
                    f" | finish: {msg.get('finish_reason', 'unknown')}"
                    + (f" | {len(tool_calls)} tool call(s)" if tool_calls else "")
                )

        elif role == "tool":
            tool_name = msg.get("tool_name") or "tool"
            status = msg.get("status", "")
            status_badge = "✓ success" if status == "success" else f"✗ {status or 'error'}"
            remaining = msg.get("remaining_budget")
            total = msg.get("total_budget")
            budget_badge = (
                f" · 🪙 budget: **{remaining:g}/{total:g}**"
                if remaining is not None and total is not None
                else ""
            )
            with st.chat_message("tool", avatar="🔧"):
                st.markdown(
                    f"**Tool result · `{tool_name}`** — {status_badge}{budget_badge}"
                )
                content = msg.get("content", "")
                with st.expander("Tool output — click to expand"):
                    if isinstance(content, dict):
                        st.json(content)
                    else:
                        st.text(str(content))
        else:
            raise ValueError(f"Unknown message role: {role}")