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


class _Hit:
    """Minimal PatternHit duck-type for render tests."""

    def __init__(self, label, detail, message_indices):
        self.label = label
        self.detail = detail
        self.message_indices = message_indices


def test_render_pattern_hits_banner_lists_every_pattern(monkeypatch):
    import explorer.render as render

    cap = _CapturingSt()
    monkeypatch.setattr(render, "st", cap)
    record = {"instance_id": "i1", "messages": [{"role": "ai", "content": "hi", "tool_calls": []}]}
    hits = [
        _Hit("Blind submit", "submitted blind", [0]),
        _Hit("Budget death", "ran out of budget", []),  # conversation-level, no index
    ]
    render.render_conversation(record, pattern_hits=hits)
    banner = "\n".join(cap.markdown_calls)
    # Banner names every pattern, including the index-less conversation-level one.
    assert "Anti-patterns flagged (2)" in banner
    assert "Blind submit" in banner and "Budget death" in banner


def test_render_pattern_hits_tag_the_flagged_turn(monkeypatch):
    import explorer.render as render

    cap = _CapturingSt()
    monkeypatch.setattr(render, "st", cap)
    record = {
        "instance_id": "i1",
        "messages": [
            {"role": "ai", "content": "first", "tool_calls": []},
            {"role": "ai", "content": "second", "tool_calls": []},
        ],
    }
    render.render_conversation(record, pattern_hits=[_Hit("Blind submit", "x", [1])])
    # The label badge appears as an in-trace turn tag (not only in the banner).
    tag_calls = [c for c in cap.markdown_calls if "\U0001f6a9" in c and "Blind submit" in c]
    assert tag_calls


class _CapturingCaptionSt(_StubSt):
    """_StubSt that records caption() text so budget-badge assertions can inspect it."""

    def __init__(self):
        self.caption_calls: list[str] = []

    def caption(self, body="", *_args, **_kwargs):
        self.caption_calls.append(str(body))
        return _StubCtx()


def test_render_shows_budget_on_assistant_tool_calls(monkeypatch):
    """Each assistant turn that makes a tool call shows the budget it had available.

    The first turn sees the full ``task_budget``; subsequent turns see the
    ``remaining_budget`` annotated on the preceding tool message.
    """
    import explorer.render as render

    cap = _CapturingCaptionSt()
    monkeypatch.setattr(render, "st", cap)
    record = {
        "instance_id": "i1",
        "task_budget": 12,
        "messages": [
            {"role": "ai", "content": "a", "tool_calls": [{"tool_name": "get_schema"}]},
            {
                "role": "tool",
                "tool_name": "get_schema",
                "status": "success",
                "content": {},
                "remaining_budget": 11,
                "total_budget": 12,
            },
            {"role": "ai", "content": "b", "tool_calls": [{"tool_name": "execute_sql"}]},
        ],
    }
    render.render_conversation(record)
    budget_caps = [c for c in cap.caption_calls if "🪙 budget" in c]
    assert "🪙 budget: 12/12" in budget_caps[0]  # first turn: full budget
    assert "🪙 budget: 11/12" in budget_caps[1]  # second turn: after the tool's note


def test_render_omits_budget_when_task_budget_missing(monkeypatch):
    """Records without budget info (e.g. no_tool baseline) render no budget badge."""
    import explorer.render as render

    cap = _CapturingCaptionSt()
    monkeypatch.setattr(render, "st", cap)
    record = {
        "instance_id": "i1",
        "messages": [
            {"role": "ai", "content": "a", "tool_calls": [{"tool_name": "get_schema"}]},
        ],
    }
    render.render_conversation(record)
    assert not any("🪙 budget" in c for c in cap.caption_calls)
