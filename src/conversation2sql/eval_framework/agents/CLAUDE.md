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

## utils_kb_linearize.py

Shared KB linearization helpers used by both agents when `is_kb_linearized=True`.

- `linearize_kb(masked_agent_kb)` — emits one `# Subgraph N` section per connected component of the KB DAG. Each section has a `# Dependency edges` block (if that component has edges) and a `# Definitions` block in topological order (leaves first). Returns `""` for an empty KB.
- `linearize_prerequisites(name, masked_agent_kb)` — returns one entry plus all of its **transitive prerequisites** (follows `children_knowledge`), as a headerless `# Dependency edges` + `# Definitions` section (topo order, leaves first). Dependents are excluded; masked-out prerequisites are skipped. Returns `"Knowledge not found."` if `name` is absent.
- `format_entry_line(name, masked_agent_kb)` — formats the single `[TOKEN] name - desc - formula: …` line for one entry. Returns `"Knowledge not found."` if the name is absent.
- `_render_section(ordered, in_kb, token_of)` — private helper shared by `linearize_kb` and `linearize_prerequisites` that renders the edges + definitions blocks (no `# Subgraph` header).

Used by `bird_baseline/tools/bird_interact_env_tools.py` (KB tools when `is_kb_linearized=True`: `get_knowledge_definition` → `linearize_prerequisites`, `get_all_knowledge_definitions` → `linearize_kb`) and `no_tool_baseline/baseline_model.py` (prompt rendering when `is_kb_linearized=True`).

## Adding a new agent

1. Create a new subdirectory package with `__init__.py`, an entry function, and prompts.
2. Register the entry function in `agents/__init__.py`.
3. Wire it into `main_pipe_workflow.py` behind the appropriate config flag.
