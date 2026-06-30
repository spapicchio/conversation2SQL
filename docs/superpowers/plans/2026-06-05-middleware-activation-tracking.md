# Middleware Activation Tracking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Record when LangChain middlewares (context editing, retries, call limits) fire during a BIRD-Interact agent run and surface those activations inline in the explorer trace.

**Architecture:** An out-of-band `MiddlewareEventRecorder` is built per run and captured by thin subclasses of each tracked middleware. Each subclass wraps the `handler` it is given (so it reuses the base middleware's logic unchanged) and calls `recorder.record(...)` on activation. A registry dict makes adding a new tracked middleware a small subclass + one entry. Events are attached to the output record (`middleware_events`) and rendered by the explorer using stable anchors (`AIMessage.id` / `tool_call_id`).

**Tech Stack:** Python 3.12, LangChain `1.2.10` agent middleware, Streamlit explorer, pytest (run via `uv run`).

---

## Background facts (verified against installed source)

- `ContextEditingMiddleware.wrap_model_call` edits a **deepcopy** of the messages and only passes them to a single model call via `request.override(messages=...)` — the `[cleared]` marker never reaches persisted state (`.venv/.../middleware/context_editing.py:251`).
- `ModelRetryMiddleware.wrap_model_call` / `ToolRetryMiddleware.wrap_tool_call` loop `for attempt in range(self.max_retries + 1): try: return handler(request)` (`model_retry.py:232`, `tool_retry.py:314`). Delegating to `super().wrap_*_call(request, wrapped_handler)` reuses the backoff loop.
- `ModelCallLimitMiddleware.before_model` and `ToolCallLimitMiddleware.after_model` are decorated `@hook_config(can_jump_to=["end"])` and return `{"jump_to": "end", ...}` when the limit is hit. `hook_config` only sets `func.__can_jump_to__` (`types.py:856`), so an override must re-apply it.
- `ModelResponse.result: list[BaseMessage]` holds the produced `AIMessage` (`types.py:282`); `ExtendedModelResponse.model_response` wraps a `ModelResponse` (`types.py:307`).
- All of `ContextEditingMiddleware`, `ModelRetryMiddleware`, `ToolRetryMiddleware`, `ModelCallLimitMiddleware`, `ToolCallLimitMiddleware`, `ModelResponse`, `hook_config` are importable from `langchain.agents.middleware`.

## File structure

- **Create** `src/conversation2sql/eval_framework/agents/bird_baseline/middleware_tracking.py` — `MiddlewareEvent`, `MiddlewareEventRecorder`, `_response_ai_id`, the five tracked subclasses, and the `TRACKED_MIDDLEWARE` registry. One responsibility: instrumentation.
- **Modify** `src/conversation2sql/eval_framework/agents/bird_baseline/agent_code.py` — build a recorder, construct the middlewares via the registry, attach `middleware_events` to the output.
- **Modify** `src/conversation2sql/eval_framework/agents/utils.py` — serialize `id` (AI messages) and `tool_call_id` (tool messages) in `utils_process_single_msg`.
- **Modify** `explorer/render.py` — pure grouping/formatting helpers + inline rendering of events.
- **Create** `tests/eval_framework/agents/test_middleware_tracking.py` — unit tests for the recorder and the five subclasses.
- **Modify** `tests/eval_framework/agents/test_utils_process_msg_budget.py` (existing) **or create** `tests/eval_framework/agents/test_utils_serialize_ids.py` — serialization tests. This plan creates a new focused file.
- **Modify** `tests/explorer/test_render.py` — render helper + smoke tests.

---

### Task 1: Recorder, event dataclass, and AI-id helper

**Files:**
- Create: `src/conversation2sql/eval_framework/agents/bird_baseline/middleware_tracking.py`
- Test: `tests/eval_framework/agents/test_middleware_tracking.py`

- [ ] **Step 1: Write the failing test**

Create `tests/eval_framework/agents/test_middleware_tracking.py`:

```python
from __future__ import annotations

from langchain_core.messages import AIMessage
from langchain.agents.middleware import ModelResponse

from conversation2sql.eval_framework.agents.bird_baseline.middleware_tracking import (
    MiddlewareEvent,
    MiddlewareEventRecorder,
    _response_ai_id,
)


def test_recorder_appends_event():
    rec = MiddlewareEventRecorder()
    rec.record("context_editing", "cleared", anchor={"ai_message_id": "a1"}, n_cleared=3)
    assert rec.events == [
        MiddlewareEvent(
            middleware="context_editing",
            kind="cleared",
            anchor={"ai_message_id": "a1"},
            detail={"n_cleared": 3},
        )
    ]


def test_recorder_defaults_anchor_to_empty_dict():
    rec = MiddlewareEventRecorder()
    rec.record("model_call_limit", "limit_reached")
    assert rec.events[0].anchor == {}
    assert rec.events[0].detail == {}


def test_response_ai_id_from_model_response():
    resp = ModelResponse(result=[AIMessage(content="hi", id="ai_1")])
    assert _response_ai_id(resp) == "ai_1"


def test_response_ai_id_from_bare_ai_message():
    assert _response_ai_id(AIMessage(content="hi", id="ai_2")) == "ai_2"


def test_response_ai_id_none_when_absent():
    assert _response_ai_id(object()) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/eval_framework/agents/test_middleware_tracking.py -v`
Expected: FAIL with `ModuleNotFoundError`/`ImportError` (module not created yet).

- [ ] **Step 3: Write minimal implementation**

Create `src/conversation2sql/eval_framework/agents/bird_baseline/middleware_tracking.py`:

```python
"""Out-of-band tracking of LangChain middleware activations.

The middlewares in `agent_code.py` mostly act invisibly: their effect never
reaches persisted agent state (e.g. `ContextEditingMiddleware` edits a deepcopy
of the messages), so the saved trace gives no signal that they fired. This module
adds a per-run `MiddlewareEventRecorder` and thin subclasses that record an event
when each tracked middleware activates. The recorder is read after `agent.invoke()`
and attached to the output record as `middleware_events`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from langchain_core.messages import AIMessage


@dataclass
class MiddlewareEvent:
    """One middleware activation, anchored to a message in the saved trace."""

    middleware: str  # registry key, e.g. "context_editing"
    kind: str  # "cleared" | "retry" | "limit_reached"
    anchor: dict[str, Any] = field(default_factory=dict)  # {"ai_message_id"|"tool_call_id": ...} or {}
    detail: dict[str, Any] = field(default_factory=dict)  # middleware-specific payload


class MiddlewareEventRecorder:
    """Collects `MiddlewareEvent`s for a single agent run.

    Built fresh per run in `run_agent_bird_baseline` and captured by closure in
    each tracked middleware, so concurrent tasks never share a recorder.
    """

    def __init__(self) -> None:
        self.events: list[MiddlewareEvent] = []

    def record(
        self,
        middleware: str,
        kind: str,
        *,
        anchor: dict[str, Any] | None = None,
        **detail: Any,
    ) -> None:
        self.events.append(MiddlewareEvent(middleware, kind, anchor or {}, detail))


def _response_ai_id(response: Any) -> str | None:
    """Best-effort extraction of the produced `AIMessage.id` from a model response.

    Handles `ExtendedModelResponse` (`.model_response.result`), `ModelResponse`
    (`.result`), and a bare `AIMessage`. Returns None when no id is available.
    """
    model_response = getattr(response, "model_response", response)
    result = getattr(model_response, "result", None)
    if result:
        for message in result:
            if isinstance(message, AIMessage):
                return message.id
    if isinstance(response, AIMessage):
        return response.id
    return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/eval_framework/agents/test_middleware_tracking.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add src/conversation2sql/eval_framework/agents/bird_baseline/middleware_tracking.py tests/eval_framework/agents/test_middleware_tracking.py
git commit -m "feat(tracking): add middleware event recorder"
```

---

### Task 2: TrackedContextEditingMiddleware

**Files:**
- Modify: `src/conversation2sql/eval_framework/agents/bird_baseline/middleware_tracking.py`
- Test: `tests/eval_framework/agents/test_middleware_tracking.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/eval_framework/agents/test_middleware_tracking.py`:

```python
from dataclasses import dataclass as _dataclass

from langchain_core.messages import ToolMessage
from langchain.agents.middleware import ClearToolUsesEdit, ModelResponse

from conversation2sql.eval_framework.agents.bird_baseline.middleware_tracking import (
    TrackedContextEditingMiddleware,
)


@_dataclass
class _FakeRequest:
    """Minimal stand-in for ModelRequest: only `.messages` and `.override` are used
    on the approximate-token path of ContextEditingMiddleware.wrap_model_call."""

    messages: list

    def override(self, *, messages):
        return _FakeRequest(messages=messages)


def _ai_with_call(call_id):
    return AIMessage(content="x", tool_calls=[{"name": "execute_sql", "args": {}, "id": call_id}])


def test_context_editing_records_cleared_event():
    rec = MiddlewareEventRecorder()
    mw = TrackedContextEditingMiddleware(
        recorder=rec,
        edits=[ClearToolUsesEdit(trigger=10, keep=0, clear_at_least=0)],
    )
    messages = [
        _ai_with_call("c1"),
        ToolMessage(content="y" * 400, tool_call_id="c1", name="execute_sql"),
        _ai_with_call("c2"),
        ToolMessage(content="z" * 400, tool_call_id="c2", name="execute_sql"),
    ]
    request = _FakeRequest(messages=messages)

    def handler(req):
        return ModelResponse(result=[AIMessage(content="done", id="ai_x")])

    mw.wrap_model_call(request, handler)

    assert len(rec.events) == 1
    event = rec.events[0]
    assert event.middleware == "context_editing"
    assert event.kind == "cleared"
    assert event.anchor == {"ai_message_id": "ai_x"}
    assert event.detail["n_cleared"] >= 1


def test_context_editing_silent_below_trigger():
    rec = MiddlewareEventRecorder()
    mw = TrackedContextEditingMiddleware(
        recorder=rec,
        edits=[ClearToolUsesEdit(trigger=1_000_000, keep=0)],
    )
    request = _FakeRequest(
        messages=[_ai_with_call("c1"), ToolMessage(content="y", tool_call_id="c1", name="execute_sql")]
    )
    mw.wrap_model_call(request, lambda req: ModelResponse(result=[AIMessage(content="ok", id="ai_y")]))
    assert rec.events == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/eval_framework/agents/test_middleware_tracking.py -k context_editing -v`
Expected: FAIL with `ImportError` (`TrackedContextEditingMiddleware` not defined).

- [ ] **Step 3: Write minimal implementation**

Add imports at the top of `middleware_tracking.py` (extend the existing import block):

```python
from langchain_core.messages import AIMessage, ToolMessage
from langchain.agents.middleware import ContextEditingMiddleware
```

Append the class to `middleware_tracking.py`:

```python
class TrackedContextEditingMiddleware(ContextEditingMiddleware):
    """ContextEditingMiddleware that records when tool outputs are actually cleared.

    Delegates to `super().wrap_model_call` with a wrapped handler. The base computes
    the edited message list and calls our handler with it, so the wrapped handler
    sees the post-clear messages (carrying `response_metadata.context_editing.cleared`)
    and the produced AIMessage — no base logic is duplicated.
    """

    def __init__(self, *, recorder: MiddlewareEventRecorder, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.recorder = recorder

    def wrap_model_call(self, request, handler):  # type: ignore[override]
        def tracking_handler(edited_request):
            n_cleared = sum(
                1
                for message in edited_request.messages
                if isinstance(message, ToolMessage)
                and (message.response_metadata or {}).get("context_editing", {}).get("cleared")
            )
            response = handler(edited_request)
            if n_cleared:
                self.recorder.record(
                    "context_editing",
                    "cleared",
                    anchor={"ai_message_id": _response_ai_id(response)},
                    n_cleared=n_cleared,
                )
            return response

        return super().wrap_model_call(request, tracking_handler)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/eval_framework/agents/test_middleware_tracking.py -k context_editing -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/conversation2sql/eval_framework/agents/bird_baseline/middleware_tracking.py tests/eval_framework/agents/test_middleware_tracking.py
git commit -m "feat(tracking): track context-editing clears"
```

---

### Task 3: TrackedModelRetryMiddleware + TrackedToolRetryMiddleware

**Files:**
- Modify: `src/conversation2sql/eval_framework/agents/bird_baseline/middleware_tracking.py`
- Test: `tests/eval_framework/agents/test_middleware_tracking.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/eval_framework/agents/test_middleware_tracking.py`:

```python
from langchain_core.messages import ToolMessage as _ToolMessage

from conversation2sql.eval_framework.agents.bird_baseline.middleware_tracking import (
    TrackedModelRetryMiddleware,
    TrackedToolRetryMiddleware,
)


def test_model_retry_records_retry_event():
    rec = MiddlewareEventRecorder()
    mw = TrackedModelRetryMiddleware(recorder=rec, max_retries=2, initial_delay=0.0, max_delay=0.0)
    calls = {"n": 0}

    def handler(req):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ValueError("boom")
        return ModelResponse(result=[AIMessage(content="ok", id="ai_r")])

    mw.wrap_model_call(object(), handler)

    assert len(rec.events) == 1
    event = rec.events[0]
    assert event.middleware == "model_retry"
    assert event.kind == "retry"
    assert event.detail["attempts"] == 2
    assert event.anchor == {"ai_message_id": "ai_r"}
    assert "ValueError: boom" in event.detail["errors"]


def test_model_retry_silent_on_first_success():
    rec = MiddlewareEventRecorder()
    mw = TrackedModelRetryMiddleware(recorder=rec, max_retries=2, initial_delay=0.0, max_delay=0.0)
    mw.wrap_model_call(object(), lambda req: ModelResponse(result=[AIMessage(content="ok", id="a")]))
    assert rec.events == []


class _FakeToolRequest:
    def __init__(self):
        self.tool = None
        self.tool_call = {"id": "t1", "name": "execute_sql"}


def test_tool_retry_records_retry_event():
    rec = MiddlewareEventRecorder()
    mw = TrackedToolRetryMiddleware(recorder=rec, max_retries=2, initial_delay=0.0, max_delay=0.0)
    calls = {"n": 0}

    def handler(req):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("io")
        return _ToolMessage(content="ok", tool_call_id="t1", name="execute_sql")

    mw.wrap_tool_call(_FakeToolRequest(), handler)

    assert len(rec.events) == 1
    event = rec.events[0]
    assert event.middleware == "tool_retry"
    assert event.kind == "retry"
    assert event.anchor == {"tool_call_id": "t1"}
    assert event.detail["attempts"] == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/eval_framework/agents/test_middleware_tracking.py -k retry -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Write minimal implementation**

Extend the import in `middleware_tracking.py`:

```python
from langchain.agents.middleware import (
    ContextEditingMiddleware,
    ModelRetryMiddleware,
    ToolRetryMiddleware,
)
```

Append both classes to `middleware_tracking.py`:

```python
class TrackedModelRetryMiddleware(ModelRetryMiddleware):
    """ModelRetryMiddleware that records when a retry actually occurred.

    Wraps the handler to count attempts and capture exceptions, then delegates to
    the stock backoff loop via `super().wrap_model_call`.
    """

    def __init__(self, *, recorder: MiddlewareEventRecorder, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.recorder = recorder

    def wrap_model_call(self, request, handler):  # type: ignore[override]
        attempts = 0
        errors: list[str] = []

        def counting_handler(req):
            nonlocal attempts
            attempts += 1
            try:
                return handler(req)
            except Exception as exc:  # noqa: BLE001 - re-raised after recording
                errors.append(f"{type(exc).__name__}: {exc}")
                raise

        try:
            result = super().wrap_model_call(request, counting_handler)
        except Exception:
            if attempts > 1:
                self.recorder.record(
                    "model_retry", "retry", anchor={}, attempts=attempts, errors=errors, exhausted=True
                )
            raise
        if attempts > 1:
            self.recorder.record(
                "model_retry",
                "retry",
                anchor={"ai_message_id": _response_ai_id(result)},
                attempts=attempts,
                errors=errors,
            )
        return result


class TrackedToolRetryMiddleware(ToolRetryMiddleware):
    """ToolRetryMiddleware that records when a tool retry actually occurred."""

    def __init__(self, *, recorder: MiddlewareEventRecorder, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.recorder = recorder

    def wrap_tool_call(self, request, handler):  # type: ignore[override]
        tool_call_id = request.tool_call["id"]
        attempts = 0
        errors: list[str] = []

        def counting_handler(req):
            nonlocal attempts
            attempts += 1
            try:
                return handler(req)
            except Exception as exc:  # noqa: BLE001 - re-raised after recording
                errors.append(f"{type(exc).__name__}: {exc}")
                raise

        try:
            result = super().wrap_tool_call(request, counting_handler)
        except Exception:
            if attempts > 1:
                self.recorder.record(
                    "tool_retry",
                    "retry",
                    anchor={"tool_call_id": tool_call_id},
                    attempts=attempts,
                    errors=errors,
                    exhausted=True,
                )
            raise
        if attempts > 1:
            self.recorder.record(
                "tool_retry",
                "retry",
                anchor={"tool_call_id": tool_call_id},
                attempts=attempts,
                errors=errors,
            )
        return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/eval_framework/agents/test_middleware_tracking.py -k retry -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/conversation2sql/eval_framework/agents/bird_baseline/middleware_tracking.py tests/eval_framework/agents/test_middleware_tracking.py
git commit -m "feat(tracking): track model and tool retries"
```

---

### Task 4: TrackedModelCallLimitMiddleware + TrackedToolCallLimitMiddleware

**Files:**
- Modify: `src/conversation2sql/eval_framework/agents/bird_baseline/middleware_tracking.py`
- Test: `tests/eval_framework/agents/test_middleware_tracking.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/eval_framework/agents/test_middleware_tracking.py`:

```python
from conversation2sql.eval_framework.agents.bird_baseline.middleware_tracking import (
    TrackedModelCallLimitMiddleware,
    TrackedToolCallLimitMiddleware,
)


def test_model_call_limit_records_when_exceeded():
    rec = MiddlewareEventRecorder()
    mw = TrackedModelCallLimitMiddleware(recorder=rec, run_limit=1, exit_behavior="end")
    state = {"run_model_call_count": 1, "thread_model_call_count": 0}
    result = mw.before_model(state, None)
    assert result is not None and result.get("jump_to") == "end"
    assert len(rec.events) == 1
    assert rec.events[0].middleware == "model_call_limit"
    assert rec.events[0].kind == "limit_reached"
    assert rec.events[0].detail["run_count"] == 1


def test_model_call_limit_silent_below_limit():
    rec = MiddlewareEventRecorder()
    mw = TrackedModelCallLimitMiddleware(recorder=rec, run_limit=5, exit_behavior="end")
    state = {"run_model_call_count": 1, "thread_model_call_count": 0}
    assert mw.before_model(state, None) is None
    assert rec.events == []


def test_tool_call_limit_records_when_exceeded():
    rec = MiddlewareEventRecorder()
    mw = TrackedToolCallLimitMiddleware(recorder=rec, thread_limit=1, run_limit=1, exit_behavior="end")
    state = {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "execute_sql", "args": {}, "id": "t1"},
                    {"name": "execute_sql", "args": {}, "id": "t2"},
                ],
            )
        ],
        "thread_tool_call_count": {},
        "run_tool_call_count": {},
    }
    result = mw.after_model(state, None)
    assert result is not None and result.get("jump_to") == "end"
    assert len(rec.events) == 1
    assert rec.events[0].middleware == "tool_call_limit"
    assert rec.events[0].kind == "limit_reached"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/eval_framework/agents/test_middleware_tracking.py -k limit -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Write minimal implementation**

Extend the import in `middleware_tracking.py`:

```python
from langchain.agents.middleware import (
    ClearToolUsesEdit,  # noqa: F401 - re-exported for callers building edits (optional)
    ContextEditingMiddleware,
    ModelCallLimitMiddleware,
    ModelRetryMiddleware,
    ToolCallLimitMiddleware,
    ToolRetryMiddleware,
    hook_config,
)
```

(If `ClearToolUsesEdit` re-export is undesirable, omit that line — it is only a convenience.)

Append both classes to `middleware_tracking.py`:

```python
class TrackedModelCallLimitMiddleware(ModelCallLimitMiddleware):
    """ModelCallLimitMiddleware that records when the limit terminates the run.

    `before_model` is re-decorated with `@hook_config(can_jump_to=["end"])` so the
    graph compiler still sees the jump edge on this subclass.
    """

    def __init__(self, *, recorder: MiddlewareEventRecorder, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.recorder = recorder

    @hook_config(can_jump_to=["end"])
    def before_model(self, state, runtime):  # type: ignore[override]
        result = super().before_model(state, runtime)
        if result and result.get("jump_to") == "end":
            self.recorder.record(
                "model_call_limit",
                "limit_reached",
                anchor={},
                run_count=state.get("run_model_call_count", 0),
                thread_count=state.get("thread_model_call_count", 0),
                run_limit=self.run_limit,
                thread_limit=self.thread_limit,
            )
        return result


class TrackedToolCallLimitMiddleware(ToolCallLimitMiddleware):
    """ToolCallLimitMiddleware that records when the limit terminates the run."""

    def __init__(self, *, recorder: MiddlewareEventRecorder, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.recorder = recorder

    @hook_config(can_jump_to=["end"])
    def after_model(self, state, runtime):  # type: ignore[override]
        result = super().after_model(state, runtime)
        if result and result.get("jump_to") == "end":
            self.recorder.record(
                "tool_call_limit",
                "limit_reached",
                anchor={},
                run_limit=self.run_limit,
                thread_limit=self.thread_limit,
            )
        return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/eval_framework/agents/test_middleware_tracking.py -k limit -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/conversation2sql/eval_framework/agents/bird_baseline/middleware_tracking.py tests/eval_framework/agents/test_middleware_tracking.py
git commit -m "feat(tracking): track model and tool call-limit termination"
```

---

### Task 5: TRACKED_MIDDLEWARE registry

**Files:**
- Modify: `src/conversation2sql/eval_framework/agents/bird_baseline/middleware_tracking.py`
- Test: `tests/eval_framework/agents/test_middleware_tracking.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/eval_framework/agents/test_middleware_tracking.py`:

```python
from conversation2sql.eval_framework.agents.bird_baseline.middleware_tracking import (
    TRACKED_MIDDLEWARE,
)


def test_registry_builds_each_tracked_middleware():
    rec = MiddlewareEventRecorder()
    built = {
        "model_retry": TRACKED_MIDDLEWARE["model_retry"](rec, max_retries=2),
        "tool_retry": TRACKED_MIDDLEWARE["tool_retry"](rec, max_retries=2),
        "model_call_limit": TRACKED_MIDDLEWARE["model_call_limit"](rec, run_limit=10),
        "tool_call_limit": TRACKED_MIDDLEWARE["tool_call_limit"](rec, run_limit=10, thread_limit=20),
        "context_editing": TRACKED_MIDDLEWARE["context_editing"](rec),
    }
    assert isinstance(built["model_retry"], TrackedModelRetryMiddleware)
    assert isinstance(built["context_editing"], TrackedContextEditingMiddleware)
    # every tracked middleware shares the same recorder instance
    assert all(mw.recorder is rec for mw in built.values())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/eval_framework/agents/test_middleware_tracking.py -k registry -v`
Expected: FAIL with `ImportError` (`TRACKED_MIDDLEWARE` not defined).

- [ ] **Step 3: Write minimal implementation**

Append to `middleware_tracking.py`:

```python
# name -> factory(recorder, **kwargs) -> AgentMiddleware. To track a new middleware,
# add a Tracked* subclass above and a single entry here.
TRACKED_MIDDLEWARE: dict[str, Callable[..., Any]] = {
    "context_editing": lambda recorder, **kw: TrackedContextEditingMiddleware(recorder=recorder, **kw),
    "model_retry": lambda recorder, **kw: TrackedModelRetryMiddleware(recorder=recorder, **kw),
    "tool_retry": lambda recorder, **kw: TrackedToolRetryMiddleware(recorder=recorder, **kw),
    "model_call_limit": lambda recorder, **kw: TrackedModelCallLimitMiddleware(recorder=recorder, **kw),
    "tool_call_limit": lambda recorder, **kw: TrackedToolCallLimitMiddleware(recorder=recorder, **kw),
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/eval_framework/agents/test_middleware_tracking.py -k registry -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add src/conversation2sql/eval_framework/agents/bird_baseline/middleware_tracking.py tests/eval_framework/agents/test_middleware_tracking.py
git commit -m "feat(tracking): add tracked-middleware registry"
```

---

### Task 6: Serialize message ids and tool_call_id

**Files:**
- Modify: `src/conversation2sql/eval_framework/agents/utils.py` (`utils_process_single_msg`, ~lines 82-115)
- Test: `tests/eval_framework/agents/test_utils_serialize_ids.py` (create)

- [ ] **Step 1: Write the failing test**

Create `tests/eval_framework/agents/test_utils_serialize_ids.py`:

```python
from __future__ import annotations

from langchain_core.messages import AIMessage, ToolMessage

from conversation2sql.eval_framework.agents.utils import utils_process_single_msg


def test_ai_message_serializes_id():
    msg = AIMessage(content="hi", id="ai_42")
    out = utils_process_single_msg(msg, tool_costs={})
    assert out["id"] == "ai_42"


def test_tool_message_serializes_tool_call_id():
    msg = ToolMessage(content="{}", tool_call_id="call_7", name="execute_sql")
    out = utils_process_single_msg(msg, tool_costs={})
    assert out["tool_call_id"] == "call_7"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/eval_framework/agents/test_utils_serialize_ids.py -v`
Expected: FAIL with `KeyError: 'id'` (and `KeyError: 'tool_call_id'`).

- [ ] **Step 3: Write minimal implementation**

In `src/conversation2sql/eval_framework/agents/utils.py`, in the `AIMessage` branch of `utils_process_single_msg`, add `"id": message.id` to the returned dict:

```python
    if isinstance(message, AIMessage):
        meta = message.response_metadata or {}
        thinking = meta.get("thinking") if isinstance(meta, dict) else None
        thinking_field = {"thinking": thinking} if thinking else {}
        return {
            **base,
            "id": message.id,
            **utils_extract_ai_metadata(message, tool_costs=tool_costs),
            **thinking_field,
        }
```

In the `ToolMessage` branch, re-enable the `tool_call_id` line (currently commented at `utils.py:111`):

```python
        return {
            **base,
            "tool_name": message.name,
            "tool_call_id": message.tool_call_id,
            "status": message.status,  # 'success' | 'error'
            "content": clean,
            **budget,
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/eval_framework/agents/test_utils_serialize_ids.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/conversation2sql/eval_framework/agents/utils.py tests/eval_framework/agents/test_utils_serialize_ids.py
git commit -m "feat(serialize): expose AI message id and tool_call_id in trace"
```

---

### Task 7: Wire registry + recorder into the agent

**Files:**
- Modify: `src/conversation2sql/eval_framework/agents/bird_baseline/agent_code.py` (imports, middleware list at lines 90-119, return path at lines 122-135)

This task has no unit test of its own (it is wiring exercised by the full suite and the existing `test_budget_reducer_init.py`); correctness is verified by Step 4 running the whole suite plus an import smoke check.

- [ ] **Step 1: Replace the middleware imports and add tracking imports**

In `agent_code.py`, the current import block (lines 9-14) imports `ToolCallLimitMiddleware, ModelCallLimitMiddleware, ModelRetryMiddleware, ToolRetryMiddleware` from `langchain.agents.middleware`. Remove those four names from that import (keep `create_agent`), and keep the `ClearToolUsesEdit` / `ContextEditingMiddleware` imports at lines 1-2 (still used to build the `edits=[...]`). Add a new import:

```python
from conversation2sql.eval_framework.agents.bird_baseline.middleware_tracking import (
    MiddlewareEventRecorder,
    TRACKED_MIDDLEWARE,
)
from dataclasses import asdict
```

- [ ] **Step 2: Build the recorder and construct middlewares via the registry**

In `run_agent_bird_baseline`, immediately before the `agent = create_agent(...)` call, add:

```python
    recorder = MiddlewareEventRecorder()
```

Replace the `middleware=[...]` list (lines 90-119) with:

```python
        middleware=[  # pyrefly: ignore
            TRACKED_MIDDLEWARE["model_retry"](recorder, max_delay=60.0, on_failure="error"),
            TRACKED_MIDDLEWARE["tool_retry"](recorder, max_delay=60.0, on_failure="error"),
            TRACKED_MIDDLEWARE["model_call_limit"](recorder, run_limit=single_task.task_budget + 5),
            TRACKED_MIDDLEWARE["tool_call_limit"](
                recorder,
                run_limit=single_task.task_budget + 5,
                thread_limit=single_task.task_budget * 2,
            ),
            TRACKED_MIDDLEWARE["context_editing"](
                recorder,
                edits=[
                    ClearToolUsesEdit(
                        trigger=100_000,
                        clear_at_least=25_000,
                        keep=3,
                        clear_tool_inputs=False,
                        exclude_tools=[],
                        placeholder="[cleared]",
                    ),
                ],
            ),
            check_budget_limit,
            sanitize_thinking_history,
            wrap_model_append_tool_message,
            tool_wrapper_patience_and_submit,
        ],
```

- [ ] **Step 3: Attach the events to the output**

In `run_agent_bird_baseline`, after the line `output["predicted_sql"] = predicted_sql` and before `return output`, add:

```python
    output["middleware_events"] = [asdict(event) for event in recorder.events]
```

- [ ] **Step 4: Run the import smoke check and the full suite**

Run: `uv run python -c "from conversation2sql.eval_framework.agents.bird_baseline.agent_code import run_agent_bird_baseline; print('ok')"`
Expected: prints `ok` (no graph-compile error — confirms the `@hook_config` jump edges still register on the subclasses).

Run: `uv run pytest tests/ -q`
Expected: PASS (all green).

- [ ] **Step 5: Commit**

```bash
git add src/conversation2sql/eval_framework/agents/bird_baseline/agent_code.py
git commit -m "feat(agent): build middlewares via tracking registry and emit middleware_events"
```

---

### Task 8: Render middleware events in the explorer

**Files:**
- Modify: `explorer/render.py` (add helpers near the top; call them in `render_conversation`)
- Test: `tests/explorer/test_render.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/explorer/test_render.py`:

```python
from explorer.render import (
    group_middleware_events,
    format_event_badge,
    summarize_middleware_events,
)


def test_group_events_resolves_ai_and_tool_anchors_and_terminal():
    record = {
        "messages": [
            {"role": "ai", "id": "ai_1", "tool_calls": []},
            {"role": "tool", "tool_call_id": "t1"},
        ],
        "middleware_events": [
            {"middleware": "context_editing", "kind": "cleared", "anchor": {"ai_message_id": "ai_1"}, "detail": {"n_cleared": 2}},
            {"middleware": "tool_retry", "kind": "retry", "anchor": {"tool_call_id": "t1"}, "detail": {"attempts": 2, "errors": ["RuntimeError: io"]}},
            {"middleware": "model_call_limit", "kind": "limit_reached", "anchor": {}, "detail": {}},
        ],
    }
    by_index, terminal = group_middleware_events(record)
    assert by_index[0][0]["middleware"] == "context_editing"
    assert by_index[1][0]["middleware"] == "tool_retry"
    assert len(terminal) == 1 and terminal[0]["middleware"] == "model_call_limit"


def test_group_events_empty_when_no_events():
    by_index, terminal = group_middleware_events({"messages": []})
    assert by_index == {} and terminal == []


def test_format_event_badge_variants():
    assert "Context cleared" in format_event_badge(
        {"middleware": "context_editing", "kind": "cleared", "detail": {"n_cleared": 2}}
    )
    assert "attempt" in format_event_badge(
        {"middleware": "model_retry", "kind": "retry", "detail": {"attempts": 2, "errors": ["X: y"]}}
    )
    assert "reached" in format_event_badge(
        {"middleware": "model_call_limit", "kind": "limit_reached", "detail": {}}
    )


def test_summary_counts_by_middleware():
    summary = summarize_middleware_events(
        [{"middleware": "context_editing"}, {"middleware": "model_retry"}, {"middleware": "model_retry"}]
    )
    assert "×2" in summary  # model_retry appeared twice


def test_render_conversation_shows_event_badge(monkeypatch):
    import explorer.render as render

    cap = _CapturingSt()
    monkeypatch.setattr(render, "st", cap)
    record = {
        "instance_id": "i1",
        "messages": [{"role": "ai", "id": "ai_1", "content": "hi", "tool_calls": []}],
        "middleware_events": [
            {"middleware": "context_editing", "kind": "cleared", "anchor": {"ai_message_id": "ai_1"}, "detail": {"n_cleared": 4}}
        ],
    }
    render.render_conversation(record)
    assert any("Context cleared" in c for c in cap.markdown_calls)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/explorer/test_render.py -k "event or summary or badge" -v`
Expected: FAIL with `ImportError` (helpers not defined).

- [ ] **Step 3: Write minimal implementation**

In `explorer/render.py`, add after the imports (top of file):

```python
from collections import Counter

_MW_ICON = {
    "context_editing": "🧹",
    "model_retry": "🔁",
    "tool_retry": "🔁",
    "model_call_limit": "⛔",
    "tool_call_limit": "⛔",
}


def group_middleware_events(record: dict) -> tuple[dict[int, list[dict]], list[dict]]:
    """Resolve each middleware event to the index of its anchor message.

    Returns `(by_index, terminal)` where `by_index` maps a message index to the
    events anchored there and `terminal` holds events with no resolvable anchor
    (rendered at the end of the trace).
    """
    messages = record.get("messages", [])
    ai_index: dict[str, int] = {}
    tool_index: dict[str, int] = {}
    for i, message in enumerate(messages):
        if message.get("role") == "ai" and message.get("id"):
            ai_index[message["id"]] = i
        tool_call_id = message.get("tool_call_id")
        if tool_call_id:
            tool_index[tool_call_id] = i

    by_index: dict[int, list[dict]] = {}
    terminal: list[dict] = []
    for event in record.get("middleware_events", []):
        anchor = event.get("anchor") or {}
        index = None
        if "ai_message_id" in anchor:
            index = ai_index.get(anchor["ai_message_id"])
        elif "tool_call_id" in anchor:
            index = tool_index.get(anchor["tool_call_id"])
        if index is None:
            terminal.append(event)
        else:
            by_index.setdefault(index, []).append(event)
    return by_index, terminal


def format_event_badge(event: dict) -> str:
    """One-line human-readable badge for a single middleware event."""
    middleware = event.get("middleware", "")
    kind = event.get("kind", "")
    detail = event.get("detail", {}) or {}
    icon = _MW_ICON.get(middleware, "⚙️")
    if kind == "cleared":
        return f"{icon} Context cleared: {detail.get('n_cleared', '?')} tool output(s)"
    if kind == "retry":
        first_error = (detail.get("errors") or ["?"])[0]
        return f"{icon} {detail.get('attempts', '?')} attempt(s) ({first_error})"
    if kind == "limit_reached":
        return f"{icon} {middleware.replace('_', ' ')} reached"
    return f"{icon} {middleware}: {kind}"


def summarize_middleware_events(events: list[dict]) -> str:
    """Compact per-middleware count summary, e.g. `🧹×1 · 🔁×2`."""
    counts = Counter(event.get("middleware") for event in events)
    return " · ".join(f"{_MW_ICON.get(name, '⚙️')}×{count}" for name, count in counts.items())
```

Then wire into `render_conversation`. Just after the existing `st.markdown("### Conversation trace")` line (currently `render.py:90`), insert the summary line and compute the grouping:

```python
    events = record.get("middleware_events", [])
    if events:
        st.caption(f"Middleware activations: {summarize_middleware_events(events)}")
    events_by_index, terminal_events = group_middleware_events(record)
```

Inside the `for i, msg in enumerate(...)` loop (currently `render.py:97`), immediately after the `highlight` marker block and before the `if role == "system":` branch, add:

```python
        for event in events_by_index.get(i, []):
            st.markdown(format_event_badge(event))
```

After the loop ends (after the final `else: raise ValueError(...)` block), add:

```python
    for event in terminal_events:
        st.markdown(format_event_badge(event))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/explorer/test_render.py -v`
Expected: PASS (all render tests, including the new ones).

- [ ] **Step 5: Commit**

```bash
git add explorer/render.py tests/explorer/test_render.py
git commit -m "feat(explorer): render middleware activations inline in the trace"
```

---

### Task 9: Full-suite verification

**Files:** none (verification only).

- [ ] **Step 1: Run the entire test suite**

Run: `uv run pytest tests/ -q`
Expected: PASS (all green) — per the repo's "Apply changes" rule.

- [ ] **Step 2: Type-check the new/changed code**

Run: `uv run pyrefly check`
Expected: No new errors introduced by `middleware_tracking.py`, `agent_code.py`, `utils.py`, `render.py`. (Pre-existing repo errors, if any, are out of scope — note them but do not fix here.)

- [ ] **Step 3: Commit any lint/type fixes**

If Step 2 surfaced fixable issues in the new code, fix them inline and:

```bash
git add -A
git commit -m "chore(tracking): satisfy type checker"
```

---

## Self-review notes

- **Spec coverage:** context editing (Task 2), model+tool retries (Task 3), model+tool call-limits (Task 4), registry/extensibility (Task 5), serialization of `middleware_events` + anchors (Tasks 6-7), explorer rendering + summary line (Task 8), always-on with no config flag (Task 7 wiring has no flag), full-suite check (Task 9). All spec sections map to a task.
- **Anchor types consistent:** `{"ai_message_id": ...}` for model-level events, `{"tool_call_id": ...}` for tool-level events, `{}` for terminal — used identically in `middleware_tracking.py` and resolved identically in `group_middleware_events`.
- **Names consistent across tasks:** `MiddlewareEventRecorder.record(middleware, kind, *, anchor=None, **detail)`, `_response_ai_id`, `TRACKED_MIDDLEWARE`, `group_middleware_events`, `format_event_badge`, `summarize_middleware_events` — referenced with the same signatures everywhere.
- **Risk:** retry instrumentation depends on `super().wrap_*_call(request, wrapped_handler)`; `@hook_config` re-application on the limit subclasses preserves jump edges. Both are validated by Task 7 Step 4 (graph compile via import smoke + full suite).
