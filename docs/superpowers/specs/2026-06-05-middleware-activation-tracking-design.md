# Middleware Activation Tracking — Design

Date: 2026-06-05
Status: Approved (pending implementation plan)

## Problem

The BIRD-Interact agent (`run_agent_bird_baseline`) runs under a stack of LangChain
middlewares. Several of them act *invisibly*: their effect never reaches the
persisted agent state, so the saved JSONL trace gives no signal that they fired.

The clearest example is `ContextEditingMiddleware`. Its `wrap_model_call` edits a
**deepcopy** of the messages and passes them only into a single model call via
`request.override(messages=...)`:

```python
# langchain/agents/middleware/context_editing.py:251-255
edited_messages = deepcopy(list(request.messages))
for edit in self.edits:
    edit.apply(edited_messages, count_tokens=count_tokens)
return handler(request.override(messages=edited_messages))
```

The `[cleared]` placeholder and the `response_metadata["context_editing"]` marker
**never reach the persisted state**. When the pipeline saves `response["messages"]`,
there is no trace that context editing ran. The same is broadly true for retries
(a successful retry leaves only the success) and the call-limit middlewares
(control-flow jumps with no distinctive artifact).

We want the explorer to show **which middleware fired and where** in a trace, so we
can analyze how often context editing/retries/limits actually trigger.

## Key constraint

Activation cannot be reliably detected post-hoc by scanning saved messages — it must
be **actively instrumented**. Each tracked middleware records an event into an
out-of-band recorder; the events are attached to the output record and rendered by
the explorer.

## Scope

In scope (chosen by the user):

- `ContextEditingMiddleware` — tool-output clearing fired.
- `ModelRetryMiddleware` / `ToolRetryMiddleware` — retries occurred.
- `ModelCallLimitMiddleware` / `ToolCallLimitMiddleware` — limit terminated the run.

Plus: a **registry** so a future middleware can be tracked with a small subclass +
one dict entry.

Non-goals:

- Always-on; **no new config flag** (recording overhead is negligible).
- The existing patience `[SYSTEM NOTE]` budget parsing is left as-is — it is already
  visible in the trace and is not folded into the event system.
- Fully *automatic*, zero-code tracking of an arbitrary middleware is explicitly not
  a goal: each middleware's notion of "activated" differs, so each needs a tiny
  subclass. The registry keeps that cost to ~15 lines + one line.

## Architecture

Three layers: **instrument → serialize → render**.

### Layer 1 — Instrumentation (out-of-band recorder)

New module `src/conversation2sql/eval_framework/agents/bird_baseline/middleware_tracking.py`:

```python
@dataclass
class MiddlewareEvent:
    middleware: str   # registry key, e.g. "context_editing"
    kind: str         # "cleared" | "retry" | "limit_reached"
    anchor: dict      # {"ai_message_id": ...} | {"tool_call_id": ...} | {}  (terminal)
    detail: dict      # middleware-specific payload

class MiddlewareEventRecorder:
    def __init__(self) -> None:
        self.events: list[MiddlewareEvent] = []
    def record(self, middleware: str, kind: str, *, anchor: dict | None = None, **detail) -> None:
        self.events.append(MiddlewareEvent(middleware, kind, anchor or {}, detail))
```

The recorder is a plain object — **not** a LangGraph state channel. It is built fresh
per run and captured by closure in each tracked middleware. Per-task isolation is
automatic because `run_agent_bird_baseline` already builds its middleware list (and
will now build the recorder) fresh on every call, so concurrent tasks never share a
recorder.

Each tracked middleware is a **thin subclass** taking `recorder` as an extra ctor
argument. The uniform trick is to **wrap the `handler`** passed into the hook so we
never reimplement the base middleware's logic:

- `TrackedContextEditingMiddleware(ContextEditingMiddleware)` — override
  `wrap_model_call`. Detect a clear by checking whether any message gained a
  `response_metadata["context_editing"]["cleared"]` marker during `edit.apply` (or by
  subclassing `ClearToolUsesEdit.apply`). Record `kind="cleared"` with
  `{tokens_before, n_cleared}`. Anchor to the produced `AIMessage.id`, captured from
  the wrapped handler's response.
- `TrackedModelRetryMiddleware(ModelRetryMiddleware)` /
  `TrackedToolRetryMiddleware(ToolRetryMiddleware)` — wrap the handler to count
  exceptions/attempts, then delegate to `super().wrap_model_call(request, wrapped_handler)` /
  `super().wrap_tool_call(...)`. This **reuses the stock backoff loop** (version-robust;
  `model_retry.py:232`, `tool_retry.py:314` both loop `for attempt in range(max_retries+1)`
  calling `handler` in a try/except). If `attempts > 0`, record `kind="retry"` with
  `{attempts, errors}`. Anchor: produced `AIMessage.id` / `tool_call_id`.
- `TrackedModelCallLimitMiddleware(ModelCallLimitMiddleware)` /
  `TrackedToolCallLimitMiddleware(ToolCallLimitMiddleware)` — override the enforcing
  hook, call `super()`, and if it jumps to `end` / emits the limit message, record
  `kind="limit_reached"` with `{run_count, run_limit, thread_count, thread_limit}`.
  Anchor: terminal `{}` (no produced message) — rendered at the end of the trace.

### Layer 1b — Registry (the extensibility point)

```python
# name -> factory(recorder, **kwargs) -> AgentMiddleware
TRACKED_MIDDLEWARE: dict[str, Callable[..., AgentMiddleware]] = {
    "context_editing":  lambda rec, **kw: TrackedContextEditingMiddleware(recorder=rec, **kw),
    "model_retry":      lambda rec, **kw: TrackedModelRetryMiddleware(recorder=rec, **kw),
    "tool_retry":       lambda rec, **kw: TrackedToolRetryMiddleware(recorder=rec, **kw),
    "model_call_limit": lambda rec, **kw: TrackedModelCallLimitMiddleware(recorder=rec, **kw),
    "tool_call_limit":  lambda rec, **kw: TrackedToolCallLimitMiddleware(recorder=rec, **kw),
}
```

`agent_code.py` builds one `recorder` per run and constructs the tracked middlewares
via the registry (threading the recorder + the existing kwargs such as
`max_delay`/`run_limit`/`thread_limit`/`edits`). The custom patience middlewares are
unchanged. Adding a future tracked middleware = write its subclass + add one registry
line.

### Layer 2 — Serialization

- `run_agent_bird_baseline`: after `agent.invoke()`,
  `output["middleware_events"] = [asdict(e) for e in recorder.events]`.
- `utils_process_single_msg` (`agents/utils.py`): add `"id": message.id` to AI
  messages and re-expose `tool_call_id` on tool messages (currently the commented-out
  line at `agents/utils.py:111`). These are the anchors the explorer needs.

The record written to `results_iter*.jsonl` then carries a `middleware_events` list.

### Layer 3 — Rendering (explorer)

`explorer/render.py::render_conversation`:

- Build `{ai_message_id -> index}` and `{tool_call_id -> index}` maps from
  `record["messages"]`.
- Group `record["middleware_events"]` by resolved message index (terminal events go to
  a "trace end" bucket).
- In the existing per-message loop, when rendering message `i`, emit a small callout
  for any event anchored there, e.g.:
  - `🧹 Context cleared: 4 tool outputs (~26k tok)`
  - `🔁 2 retries (Timeout)`
  - `⛔ Model call limit reached`
- A minimal **summary line** near the top of the trace shows per-middleware counts
  (e.g. `Middleware: 🧹×1 · 🔁×2`). Rendered only when `middleware_events` is non-empty.

## Testing

- Unit test the recorder and each tracked subclass in isolation: force activation
  (handler raises then succeeds for retries; messages over `trigger` for context
  editing; exceed the limit for call-limit) and assert exactly one event with the
  correct `middleware`, `kind`, `anchor`, and `detail`.
- Serialization test: a processed response includes `middleware_events`, AI messages
  carry `id`, tool messages carry `tool_call_id`.
- A render smoke test: `render_conversation` on a record with events does not error and
  references the events (Streamlit rendering is exercised via the existing explorer
  test patterns).
- Full suite: `uv run pytest tests/` per the repo's "Apply changes" rule.

## Files touched

- New: `agents/bird_baseline/middleware_tracking.py` (recorder, tracked subclasses, registry).
- `agents/bird_baseline/agent_code.py` — build recorder, construct middlewares via
  registry, attach `middleware_events` to output.
- `agents/utils.py` — serialize AI `id` and tool `tool_call_id` in
  `utils_process_single_msg`.
- `explorer/render.py` — render events inline + summary line.
- Tests under `tests/eval_framework/agents/` and `tests/explorer/`.

## Risks / open questions

- **Anchor precision for context editing**: capturing the produced `AIMessage.id` from
  the wrapped handler is the robust path; if a provider does not populate `id`, the
  event falls back to a terminal/unanchored render (still counted in the summary).
- **Retry instrumentation depends on delegating to `super().wrap_*_call`** with a
  wrapped handler. If a future LangChain version changes the hook signature, the
  subclass needs a one-line update — acceptable and localized.
