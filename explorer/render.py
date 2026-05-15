"""Shared Streamlit rendering helpers for conversation records."""

from __future__ import annotations

import json

import streamlit as st


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


def render_conversation(record: dict) -> None:
    acc = record.get("execution_accuracy", False)
    badge = "✓ PASS" if acc else "✗ FAIL"
    st.subheader(
        f"Conversation: `{record.get('instance_id', '')}` · "
        f"db: `{record.get('selected_database', '')}` · **{badge}**"
    )

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
    col_meanings: dict = record.get("column_meanings") or {}

    ctx_cols = st.columns(3)
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
        with st.expander(f"Column meanings ({len(col_meanings)} entries)"):
            with st.container(height=300):
                if col_meanings:
                    st.json(col_meanings)
                else:
                    st.text("(none)")

    st.divider()

    for msg in record.get("messages", []):
        role = msg.get("role")

        if role in ("user", "system"):
            with st.expander(f"{role.capitalize()} prompt — click to expand"):
                st.text(msg.get("content", ""))

        elif role == "ai":
            with st.chat_message("assistant"):
                content = msg.get("content", "")
                if content:
                    render_ai_content(content)
                st.caption(
                    f"tokens: {msg.get('prompt_tokens', 0)}↑ {msg.get('completion_tokens', 0)}↓"
                    f" | cost: ${msg.get('cost_usd', 0):.5f}"
                    f" | finish: {msg.get('finish_reason', 'unknown')}"
                )
                for tc in msg.get("tool_calls", []):
                    tool_name = tc.get("tool_name", "unknown")
                    args = tc.get("arguments", {})
                    args_str = json.dumps(args, ensure_ascii=False)
                    label = (
                        f"🔧 {tool_name}({args_str[:60]}…)"
                        if len(args_str) > 60
                        else f"🔧 {tool_name}({args_str})"
                    )
                    with st.expander(label):
                        st.json(args)

        elif role == "tool":
            tool_name = msg.get("tool_name", "tool")
            status = msg.get("status", "")
            status_badge = "✓" if status == "success" else "✗"
            with st.chat_message("user"):
                st.markdown(f"**{tool_name}** {status_badge}")
                content = msg.get("content", "")
                with st.expander("Tool output — click to expand"):
                    if isinstance(content, dict):
                        st.json(content)
                    else:
                        st.text(str(content))
        else:
            raise ValueError(f"Unknown message role: {role}")