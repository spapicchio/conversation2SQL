# agents

Two agent implementations for the BIRD-Interact benchmark.

## Variants

`ConfigPipeline.baseline` selects one of four ablation cells. Two runner functions cover them:

| `baseline` enum | Runner | DB tools | `ask_user` | Query |
|-----------------|--------|----------|-----------|-------|
| `no_tool`       | `run_baseline_no_tool` | — | — | clean |
| `tools_only`    | `run_agent_bird_baseline(enable_ask_user=False)` | ✅ | — | clean |
| `tools_user`    | `run_agent_bird_baseline(enable_ask_user=True)`  | ✅ | ✅ | clean |
| `bird_full`     | `run_agent_bird_baseline(enable_ask_user=True)`  | ✅ | ✅ | ambiguous |

`bird_baseline/agent_code.py` is the parameterizable agent — `enable_ask_user` toggles the `ask_user` tool and the corresponding Jinja blocks in `prompts.py`. The middleware stack is identical across the three agentic variants.

## utils.py

Shared helpers used by both agents:
- `utils_create_model` — builds a `ChatLiteLLM` instance from provider/model/temperature config. Uses `"{provider}/{model}"` format required by LiteLLM.
- `utils_process_agent_response` — walks message history, extracts token usage, cost, tool calls, and `execution_accuracy` from the last `submit_sql` result.
- `utils_build_messages` / `utils_render_jinja` — Jinja2 prompt rendering helpers.

## Adding a new agent

1. Create a new subdirectory package with `__init__.py`, an entry function, and prompts.
2. Register the entry function in `agents/__init__.py`.
3. Wire it into `main_pipe_workflow.py` behind the appropriate config flag.
