# vLLM Gemma 4: architecture support & thinking mode

**Date:** 2026-06-08
**vLLM version:** 0.22.1
**Verified on:** GPU 0 (RTX PRO 6000 Blackwell), `google/gemma-4-E4B-it`

## TL;DR

- The dense **12B/31B** and the **E-series** are *different models under different
  architecture classes*. vLLM 0.22.1 registers the E-series class but **not** the dense one.
- `Gemma4UnifiedForConditionalGeneration` is **not** something you need — it's just the class
  Google ships the dense 12B/31B under. E4B doesn't use it.
- The 12B startup crash is **not** caused by `--reasoning-parser` / `--tool-call-parser` /
  chat template. It's purely the unsupported architecture.
- E4B thinking is **off by default**. Turning it on without `--reasoning-parser gemma4` leaks
  the raw `<|channel>thought…` block into `content`.

---

## 1. Architecture support (the real root cause)

vLLM 0.22.1 registry (`vllm/model_executor/models/registry.py`):

| Arch class | `config.json` of | Registered? |
|---|---|---|
| `Gemma4ForConditionalGeneration` (`gemma4_mm`) | `gemma-4-E4B-it` | ✅ native → **works** |
| `Gemma4UnifiedForConditionalGeneration` (`gemma4_unified`) | `gemma-4-12B-it`, `31B` | ❌ **absent** → Transformers fallback → profiling crash |
| `Gemma4ForCausalLM` | text-only checkpoints | ✅ native |

Confirmed on boot: E4B logs `Resolved architecture: Gemma4ForConditionalGeneration`.

Note: `gemma-4-12B-it` and `gemma-4-12b-it` are the **same** HF snapshot
(`5926caa4…`) — the Hub is case-insensitive. Both are the Unified (unsupported) model;
case does not select a text-only variant.

### How to make the 12B/31B work — **upgrade vLLM**

Stable **0.22.1 cannot run the dense 12B/31B** — the `Gemma4UnifiedForConditionalGeneration`
(encoder-free) class simply isn't in its registry, and no flag/override fixes that reliably.

Support landed in **vLLM PR [#44429](https://github.com/vllm-project/vllm/pull/44429)**, which
as of 2026-06 has **not shipped in a stable release** — it's only in the **nightly wheel** or the
**Docker image pinned by the vLLM Gemma 4 recipe**. So the real fix is:

1. **Bump vLLM to a build that includes PR #44429** — use the image/tag the
   [vLLM Gemma 4 recipe](https://docs.vllm.ai/projects/recipes/en/latest/Google/Gemma4.html)
   pins, or a nightly wheel. (Our `.devcontainer/docker-compose.yml` currently uses
   `vllm/vllm-openai:latest` — pin it to the recipe's tag for the 12B.) After upgrading, the
   original failing command (reasoning + tool parser + chat template) should just work.

Stop-gaps that do **not** properly fix it on 0.22.1:
- **Use E4B instead** — natively supported today, but it's a different (smaller) model.
- `--hf-overrides '{"architectures": ["..."]}'` — risky: the 12B `text_config` is
  `gemma4_unified_text`, not `gemma4_text`, so weights may not line up. Don't trust without testing.

---

## 2. Thinking mode on E4B

### How it's triggered
The chat template injects the `<|think|>` token (id **98**) into the system turn **only** when
the request passes `chat_template_kwargs={"enable_thinking": true}`. Default is off.

The model then emits its reasoning as `<|channel>thought\n…<channel|><final answer>`.
`--reasoning-parser gemma4` keys off those `<|channel>…<channel|>` delimiters to split
`reasoning_content` (the thought) from `content` (the final answer).

### What happens with the plain serve command
```bash
vllm serve google/gemma-4-E4B-it \
  --max-model-len 8000 --limit-mm-per-prompt '{"image": 0, "audio": 0}'
```
No `--reasoning-parser` → no separation:

| Request | Result |
|---|---|
| default (no `enable_thinking`) | thinking off; answer in `content`; `reasoning_content: null` |
| `enable_thinking: true` | thinking generated but **dumped raw into `content`** as `<|channel>thought…`; `reasoning_content: null` |

To get a clean split you must add `--reasoning-parser gemma4` to the serve command.

---

## 3. Reproduce

### Launch (GPU 0)
```bash
CUDA_VISIBLE_DEVICES=0 vllm serve google/gemma-4-E4B-it \
  --max-model-len 8000 --limit-mm-per-prompt '{"image": 0, "audio": 0}' \
  --port 8000
# add `--reasoning-parser gemma4` to get reasoning_content populated
```

### Confirm the `<|think|>` (id 98) injection — render endpoint
```bash
# without thinking -> no 98
curl -s localhost:8000/v1/chat/completions/render -H 'Content-Type: application/json' \
  -d '{"model":"google/gemma-4-E4B-it","messages":[{"role":"user","content":"hi"}]}' \
  | jq -c '.token_ids'
# -> [2,105,2364,107,2202,106,107,105,4368,107]

# with thinking -> 98 appears
curl -s localhost:8000/v1/chat/completions/render -H 'Content-Type: application/json' \
  -d '{"model":"google/gemma-4-E4B-it","messages":[{"role":"user","content":"hi"}],
       "chat_template_kwargs":{"enable_thinking":true}}' \
  | jq -c '.token_ids'
# -> [2,105,9731,107,98,107,106,107,105,2364,107,2202,106,107,105,4368,107]
```

### See thinking leak into `content` (no reasoning parser)
`skip_special_tokens:false` makes the channel delimiters visible:
```bash
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model":"google/gemma-4-E4B-it",
  "messages":[{"role":"user","content":"A bat and ball cost $1.10 total. The bat costs $1 more than the ball. How much is the ball? Reason carefully."}],
  "max_tokens":400,"temperature":0,
  "chat_template_kwargs":{"enable_thinking":true},
  "skip_special_tokens":false
}' | jq '{content: .choices[0].message.content, reasoning_content: .choices[0].message.reasoning_content}'
```
Output (abridged):
```json
{
  "content": "<|channel>thought\nHere's a thinking process ... L = 0.05 ...",
  "reasoning_content": null
}
```

With `--reasoning-parser gemma4` on the server, the same request instead yields the thought in
`reasoning_content` and only the final answer in `content`.

---

## 4. Why does vLLM pair the "thought" chat template with tool calling? (explain-like-I'm-5)

Imagine Gemma 4 is a kid who was taught **one habit: always whisper to yourself first, then
speak.** That whisper is the *thought channel* (`<|channel>thought … <channel|>`). The kid does
it before *everything* — before answering, and before picking up a toy (calling a tool). You
can't switch the habit off just for tool time.

So three things follow:

1. **The whisper always comes out.** If your template/parser ignores it, the whisper gets mixed
   into the kid's spoken answer, and the part of vLLM that reads tool calls
   (`--tool-call-parser`) trips over the whisper. `--reasoning-parser gemma4` is the grown-up who
   takes the whisper out first (`reasoning_content`) so the tool call is clean.

2. **The kid needs to remember what it whispered.** When the toy (tool) gives something back,
   the kid has to recall *"why did I grab that toy?"* to keep going. The tool template copies the
   earlier whisper back into the conversation **only between a tool call and its answer** (the
   `tool_calls`-guarded block) so the kid doesn't lose the plot across steps.

3. **One template, two moods.** Pass `enable_thinking=true` → the kid whispers for real. Leave it
   off → the template hands the kid an *already-finished empty whisper*
   (`<|channel>thought\n<channel|>`) so it skips straight to talking. Same template, your choice.

**Bottom line:** vLLM doesn't suggest "thought" because tool calls *need* reasoning — it's that
Gemma 4 *always* opens its mouth with a thought, so the only correct way to read its tool calls
(and to remember its plan across steps) is a thought-aware template + the reasoning parser.

### Technical version (mapped to the jinja)

Same three points, but as the actual control flow in
[tool_chat_template_gemma4.jinja](bash_scripts/utils/tool_chat_template_gemma4.jinja). Line refs
are to that file.

**0. When the system turn (and thus the thought slot) opens at all.**
[L179](bash_scripts/utils/tool_chat_template_gemma4.jinja#L179) gates the whole leading block on
`enable_thinking or tools or messages[0].role in ['system','developer']`. So as soon as you pass
`tools`, a `<|turn>system\n` turn is emitted even with no system message — that's why tool calling
and the thinking machinery are entangled in *one* template, not two.

**1. The thought channel is always present; only its *content* is conditional.**
- Thinking on: [L182-L185](bash_scripts/utils/tool_chat_template_gemma4.jinja#L182-L185) inject
  `<|think|>\n` (token 98) at the very top of the first system turn.
- Generation prompt: [L356-L362](bash_scripts/utils/tool_chat_template_gemma4.jinja#L356-L362).
  When `add_generation_prompt` and the model is about to speak, if `enable_thinking` is **off** the
  template appends a literal **empty** thought block `<|channel>thought\n<channel|>` after
  `<|turn>model\n`. That's the "already-finished empty whisper" — the channel structure is emitted
  either way, so `--tool-call-parser` always sees the same shape and `--reasoning-parser gemma4`
  always has well-formed `<|channel>…<channel|>` delimiters to strip.
- Past assistant *content* gets its inline thought removed on re-render:
  `strip_thinking()` ([L148-L158](bash_scripts/utils/tool_chat_template_gemma4.jinja#L148-L158))
  drops everything between `<|channel>` and `<channel|>`, and it's applied **only** to `role ==
  'model'` content ([L318-L319](bash_scripts/utils/tool_chat_template_gemma4.jinja#L318-L319),
  [L326-L327](bash_scripts/utils/tool_chat_template_gemma4.jinja#L326-L327)). So leaked thoughts in
  prior answers don't accumulate in context.

**2. The "remember what it whispered" copy-back is guarded to the current tool-call sub-turn.**
[L238-L241](bash_scripts/utils/tool_chat_template_gemma4.jinja#L238-L241) re-emits a structured
`reasoning`/`reasoning_content` as `<|channel>thought\n…<channel|>` **only when both**:
- `loop.index0 > ns_turn.last_user_idx` — the message is *after* the last user turn
  (`last_user_idx` is pre-scanned at
  [L207-L213](bash_scripts/utils/tool_chat_template_gemma4.jinja#L207-L213)), **and**
- `message.get('tool_calls')` — the assistant message actually calls a tool.

So a thought is preserved across `assistant(tool_call) → tool_response → assistant(tool_call)…`
within the *current* user turn, but is dropped once a new user message arrives. That's exactly the
"copies the whisper back only between a tool call and its answer" claim — it's the
`tool_calls`-guarded, post-last-user block.

**3. Where the tool plumbing lives (so you can see why a stray thought breaks parsing).**
Tool *declarations* are rendered into the system turn at
[L196-L203](bash_scripts/utils/tool_chat_template_gemma4.jinja#L196-L203); the assistant's tool
*calls* are emitted as `<|tool_call>call:<name>{…}<tool_call|>` at
[L243-L260](bash_scripts/utils/tool_chat_template_gemma4.jinja#L243-L260); tool *responses* come
back via `format_tool_response_block` (OpenAI-style `role:tool` messages are forward-scanned at
[L270-L314](bash_scripts/utils/tool_chat_template_gemma4.jinja#L270-L314)). All of this sits
*after* the thought channel — so if the thought isn't carved out into `reasoning_content` first,
the `<|channel>thought…` text lands in the same span `--tool-call-parser` reads, which is the
failure mode point #1 describes.

**Net:** the template never has an "off switch" for the thought channel — it has an *on* state
(real `<|think|>` + re-emitted `<|channel>thought>` on tool-call turns) and an *empty* state
(`<|channel>thought\n<channel|>` placeholder). Both keep the channel shape stable so the parsers
stay aligned; `--reasoning-parser gemma4` is what makes the on-state content land in
`reasoning_content` instead of `content`.

## Files involved
- `bash_scripts/utils/tool_chat_template_gemma4.jinja` — repo tool template; differs from the
  E4B built-in one by appending an empty `<|channel>thought\n<channel|>` when thinking is off
  (pins thinking off). Built-in E4B template ends at `<|turn>model\n`.
- `/usr/local/lib/python3.12/dist-packages/vllm/reasoning/gemma4_reasoning_parser.py`
  — parser; `<|think|>` = token 98, delimiters `<|channel>` / `<channel|>`.
