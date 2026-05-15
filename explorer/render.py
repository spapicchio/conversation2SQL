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

    for msg in record.get("messages", []):
        role = msg.get("role")

        if role in ("user", "system"):
            with st.expander("System prompt — click to expand"):
                st.text(msg.get("content", ""))

        elif role == "human":
            with st.chat_message("user"):
                st.markdown(msg.get("content", ""))

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
