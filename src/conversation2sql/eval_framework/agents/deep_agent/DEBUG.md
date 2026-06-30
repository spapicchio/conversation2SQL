# deep_agent — debug log

End-to-end shakeout of the `deep_agent` baseline and its four ablation flags on a
local vLLM server (GPU 0 + GPU 1). The baseline was **broken out of the box**;
three real bugs were found and fixed, then verified end-to-end.

## Summary

The new `deep_agent` baseline was run across all four ablation flags on GPU 0+1
(one vLLM server, `Qwen/Qwen3.5-9B`, data-parallel=2). It was completely broken
out of the box — three real bugs found and fixed, then verified end-to-end.

### Bugs found & fixed

1. **vLLM never told to parse tool calls** (`presets.py`) — `deep_agent` was
   missing from `_TOOL_BASELINES`, so `baseline_uses_tools('deep_agent')`
   returned `False` and the server launched *without*
   `--enable-auto-tool-choice --tool-call-parser`. A tool-only agent can't
   function. → added `deep_agent` to the set (+ test).

2. **"System message must be at the beginning"** (vLLM 400 on every task) —
   `deep_agent` passed its system prompt as a `SystemMessage` inside the state
   messages, but `FilesystemMiddleware` injects its own `/db` system message that
   `create_agent` prepends → two system messages, the second mid-history. →
   route the system text through `create_agent(system_prompt=...)`, keep only the
   human turn in state (new `_split_system_prompt` helper + test).

3. **`'Command' object has no attribute 'model_copy'`** (todos/subagents/fswrite)
   — the shared patience wrapper (`bird_baseline/agent_callback.py`) assumed
   every tool returns a `ToolMessage`, but deepagents'
   `write_todos`/`task`/`write_file`/`edit_file` return a langgraph `Command`. →
   thread the cost into the Command's own update via `dataclasses.replace`
   (+ test). Also fixed a sibling `'list' object has no attribute 'get'` in
   metrics extraction (`bird_baseline/agent_code.py`) where a subagent result's
   content JSON-decodes to a list (+ test).

### One orthogonal issue, *not* a deep_agent bug

The only remaining error is deterministic on instance `alien_2` and identical
across `base` and `todos`: `ask_user` routes the user-simulator
(`gpt-5.4-mini`, an OpenAI model) to the local vLLM endpoint → 404. It is
triggered by passing `--user_simulator_vllm_api_base` (which `run_suite` also
does unconditionally), so it would affect `bird_full` the same way. Not
root-caused; worth a look since it could silently break `ask_user` whenever the
user-sim provider is `openai` but a vLLM api-base is set.

---

## Files changed

| File | Change |
|------|--------|
| `presets.py` | `_TOOL_BASELINES` now includes `deep_agent` (bug 1) |
| `deep_agent/agent_code.py` | `_split_system_prompt` helper; `create_agent(system_prompt=...)`, human-only state (bug 2) |
| `bird_baseline/agent_callback.py` | `tool_wrapper_patience_and_submit` handles `Command`-returning tools (bug 3) |
| `bird_baseline/agent_code.py` | `passed` extraction guards non-dict tool content (bug 3 sibling) |
| Tests | `test_presets.py`, `deep_agent/test_agent_code.py`, `test_bird_baseline_agent.py`, `test_budget_terminal_state.py` |

---

## How to reproduce

### 1. Start one vLLM server on GPU 0 + GPU 1 (data-parallel = 2)

The server flags mirror what `just eval --baseline deep_agent` resolves
(confirm with `just dry --variant all_db_all_kb --baseline deep_agent`). The
`--enable-auto-tool-choice --tool-call-parser qwen3_coder` pair only appears
*after* the bug-1 fix.

```bash
PORT=8769
CUDA_VISIBLE_DEVICES=0,1 \
uv run vllm serve Qwen/Qwen3.5-9B \
  --port $PORT --max-model-len 64000 --uvicorn-log-level warning --max-num-seqs 16 \
  --enable-prefix-caching --gpu-memory-utilization 0.9 \
  --tensor-parallel-size 1 --data-parallel-size 2 --reasoning-parser qwen3 \
  --chat-template bash_scripts/utils/tool_chat_template_qwen35.jinja \
  --language-model-only --enable-auto-tool-choice --tool-call-parser qwen3_coder

# wait for health
until curl -s http://127.0.0.1:$PORT/health >/dev/null 2>&1; do sleep 3; done
```

### 2. Run each ablation against that one server

`--debug true` caps the dataset to the first 10 tasks. Each ablation reuses the
same server (cheaper than 5 server boots). The user-sim api-base is set the same
way `run_suite` does it.

```bash
PORT=8769; API="http://127.0.0.1:$PORT/v1"
for name in base todos subagents summar fswrite; do
  case $name in
    base)      extra="" ;;
    todos)     extra="--deep_enable_todos true" ;;
    subagents) extra="--deep_enable_subagents true" ;;
    summar)    extra="--deep_enable_summarization true" ;;
    fswrite)   extra="--deep_enable_fs_write true" ;;
  esac
  CUDA_VISIBLE_DEVICES=0,1 TOKENIZERS_PARALLELISM=true VLLM_WORKER_MULTIPROC_METHOD=spawn \
  uv run conv2sql run --config configs/eval_pipeline_config.yaml \
    --model-profile qwen35 --variant all_db_all_kb --baseline deep_agent \
    --predictor_model_provider hosted_vllm --predictor_enable_thinking true \
    --concurrency 16 --num-iterations 1 --debug true \
    --output_folder results_debug/$name \
    --predictor_vllm_api_base "$API" --user_simulator_vllm_api_base "$API" \
    $extra
done
```

### 3. Inspect results

```bash
for n in base todos subagents summar fswrite; do
  d=results_debug/$n
  res=$(cat "$d"/results_iter*.jsonl 2>/dev/null | wc -l)
  err=$([ -s "$d/results_error.jsonl" ] && wc -l < "$d/results_error.jsonl" || echo 0)
  printf "%-10s results=%s errors=%s\n" "$n" "$res" "$err"
done

# which tools the agent actually invoked (proves the Command path was exercised)
grep -ho '"tool_name": "[a-z_]*"' results_debug/fswrite/results_iter*.jsonl | sort | uniq -c
cat results_debug/*/results_error.jsonl   # see remaining errors verbatim
```

---

## Results

### Before the fixes

- **Bug 1** — `just dry --variant all_db_all_kb --baseline deep_agent` omitted
  `--enable-auto-tool-choice --tool-call-parser qwen3_coder` from the
  `vllm serve` line (present for `bird_full`).
- **Bug 2** — every task errored:
  ```
  litellm.BadRequestError: Hosted_vllmException - {"error":{"message":
  "System message must be at the beginning.","type":"BadRequestError","code":400}}
  ```
  → `base / todos / subagents / summar / fswrite` all `results=0 errors=10`.

### After the fixes (10-task debug runs)

| ablation | results | errors |
|---|---|---|
| `base` (read-only FS) | 9 | 1\* |
| `deep_enable_todos` | 9 | 1\* |
| `deep_enable_subagents` | 10 | 0 |
| `deep_enable_summarization` | 10 | 0 |
| `deep_enable_fs_write` | 10 | 0 |

\* The single error in `base`/`todos` is the orthogonal user-simulator routing
issue on instance `alien_2` (see Summary), **not** a deep_agent bug:

```json
{"instance_id": "alien_2", "iteration": 0, "error":
 "litellm.NotFoundError: Hosted_vllmException - {\"error\":{\"message\":
  \"The model `gpt-5.4-mini-2026-03-17` does not exist.\",\"code\":404}}"}
```

**Direct e2e proof of bug-3 fix:** the `fswrite` run actually invoked the
`Command`-returning `write_file` tool 3 times and completed `10 results, 0
errors` — exactly the path that raised `'Command' object has no attribute
'model_copy'` before the fix. `subagents` went from a hard failure
(`'list' object has no attribute 'get'`) to `10/0`.

An earlier run (after only bugs 1–2 were fixed) reproduced bug 3 directly:

```
todos      results=8  errors=2   # 'Command' object has no attribute 'model_copy'
subagents  results=9  errors=1   # 'list' object has no attribute 'get'
```

### Tests

```bash
uv run pytest tests/eval_framework/agents/test_budget_terminal_state.py \
  tests/eval_framework/agents/test_bird_baseline_agent.py \
  tests/eval_framework/agents/deep_agent/ tests/test_presets.py -q
# 69 passed
```

The 17 failures in the full `uv run pytest tests/` are **pre-existing WIP
breakage** (KB-linearize / catalog / psql-enum), confirmed by stashing these
changes and re-running: they fail identically without any of the fixes above.
