"""Tests for explorer.render conversation rendering."""

from __future__ import annotations

import contextlib

import pytest


class _StubSt:
    """Minimal stand-in for the streamlit module.

    Records nothing; every method is a no-op and every container/layout helper
    returns a context manager (or list of them) so `with st.expander(...)` works.
    """

    def __getattr__(self, _name):
        def _method(*_args, **_kwargs):
            return _StubCtx()

        return _method

    def columns(self, n, *_args, **_kwargs):
        count = n if isinstance(n, int) else len(n)
        return [_StubCtx() for _ in range(count)]


class _StubCtx:
    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


@pytest.fixture
def patch_st(monkeypatch):
    import explorer.render as render

    monkeypatch.setattr(render, "st", _StubSt())
    return render


def _record_with_roles(roles):
    return {
        "instance_id": "demo_1",
        "selected_database": "demo",
        "not_ambiguos_query": "q",
        "messages": [{"role": r, "content": "x"} for r in roles],
    }


def test_render_conversation_handles_human_role(patch_st):
    """LangChain stores the user role as 'human'; render must not raise on it."""
    record = _record_with_roles(["system", "human", "ai", "tool"])
    # AI/tool messages need their expected shapes.
    record["messages"][2] = {"role": "ai", "content": "answer", "tool_calls": []}
    record["messages"][3] = {
        "role": "tool",
        "tool_name": "submit_sql",
        "status": "success",
        "content": {"passed": True},
    }
    patch_st.render_conversation(record)  # must not raise


def test_render_conversation_rejects_truly_unknown_role(patch_st):
    record = _record_with_roles(["bogus"])
    with pytest.raises(ValueError):
        patch_st.render_conversation(record)


class _CapturingSt(_StubSt):
    """_StubSt that records markdown() text so assertions can inspect output."""

    def __init__(self):
        self.markdown_calls: list[str] = []

    def markdown(self, body="", *_args, **_kwargs):
        self.markdown_calls.append(str(body))
        return _StubCtx()


def test_render_marks_highlighted_messages(monkeypatch):
    import explorer.render as render

    cap = _CapturingSt()
    monkeypatch.setattr(render, "st", cap)
    record = {
        "instance_id": "i1",
        "messages": [{"role": "ai", "content": "hi", "tool_calls": []}],
    }
    render.render_conversation(record, highlight_indices={0})
    assert any("\U0001f6a9" in c for c in cap.markdown_calls)  # 🚩 marker rendered


def test_render_no_marker_without_highlight(monkeypatch):
    import explorer.render as render

    cap = _CapturingSt()
    monkeypatch.setattr(render, "st", cap)
    record = {"instance_id": "i1", "messages": [{"role": "ai", "content": "hi", "tool_calls": []}]}
    render.render_conversation(record)
    assert not any("\U0001f6a9" in c for c in cap.markdown_calls)
