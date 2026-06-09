# bird_baseline/tools

LangGraph tools exposed to the agent, each with a bird-coin patience cost.

## Tool costs (from `DB_TOOL_COSTS` / `USER_TOOL_COSTS`)

| Tool | Cost | Source |
|------|------|--------|
| `execute_sql` | 2.0 | `bird_interact_env_tools.py` |
| `get_schema` | 1.0 | `bird_interact_env_tools.py` |
| `get_table_names` | 0.5 | `bird_interact_env_tools.py` |
| `get_table_schema` | 0.5 | `bird_interact_env_tools.py` |
| `get_all_column_meanings` | 1.0 | `bird_interact_env_tools.py` |
| `get_column_meaning` | 0.5 | `bird_interact_env_tools.py` |
| `get_all_external_knowledge_names` | 0.5 | `bird_interact_env_tools.py` |
| `get_knowledge_definition` | 0.5 | `bird_interact_env_tools.py` |
| `get_all_knowledge_definitions` | 1.0 | `bird_interact_env_tools.py` |
| `psql_console` | 1.0 | `bird_interact_env_tools.py` |
| `ask_user` | 2.0 | `bird_interact_user_tools.py` |
| `submit_sql` | 3.0 | `bird_interact_user_tools.py` |

`TOOL_COSTS = {**DB_TOOL_COSTS, **USER_TOOL_COSTS}` is the merged dict used by the middleware.

## Design pattern

Each `@tool` is a thin wrapper that extracts fields from `runtime.context` (a `TaskData`) and delegates to a `*_impl` function. The `*_impl` functions have no LangGraph dependency and are unit-tested directly in `tests/eval_framework/tools/`.

## ask_user two-stage pipeline

`ask_user` runs a two-stage LLM pipeline:
1. `stage_1_parse_action` — classifies the clarification question as AMB/LOC/UNA via `model_user_parsing`.
2. `stage_2_generator` — generates the simulated user's response conditioned on the action via `model_user_generator`.

`return_tool_ask_user(model_user_parsing, model_user_generator)` is a factory that closes over both models; call it at agent-construction time.

## submit_sql evaluation

`submit_sql_impl` executes both the predicted SQL and the ground-truth SQL against Postgres, then compares result sets (unordered by default; ordered if `sql_query_conditions["order"] == True`). `ROUND`/`DISTINCT`/comments are stripped before comparison.

## Gotchas

- `execute_sql` only allows `SELECT`/`WITH`/`EXPLAIN` — write queries are rejected immediately.
- `get_knowledge_definition` returns `"Knowledge not found."` for entries masked by KB ambiguity (by design). When `is_kb_linearized=True` it returns the entry **plus its transitive prerequisites** (via `linearize_prerequisites`) — the dependency-edges + topo-ordered definitions section, scoped to the looked-up entry's ancestors (dependents excluded). The legacy (non-linearized) branch still JSON-dumps the single entry's visible fields.
- `get_column_meaning` key format: `"{db_name}|{table.lower()}|{column.lower()}"` — case matters for the lookup.
- `MAX_RESULT_LENGTH = 500` truncates `execute_sql` output sent back to the agent.
- `get_table_names` / `get_table_schema` parse the static `ddl_database_schema` blob at call time via `_parse_ddl` (no DB query). They mirror the KB granular tools (`get_all_external_knowledge_names` / `get_knowledge_definition`) against `get_schema` (the full dump). `get_table_schema` returns the table's `CREATE TABLE` block + its "First 3 rows" sample + every `ALTER TABLE … FOREIGN KEY` line where the table is the **child or the referenced parent** (both directions, so joinable tables surface either way); unknown names return `"Table not found."`.
- These two tools are **gated behind the `enable_table_schema_tools` flag** (default `False`). It is a `ConfigReader` field threaded onto `TaskData`; `run_agent_bird_baseline` reads `single_task.enable_table_schema_tools` to decide whether to add them to the tool list and render their lines in the prompt. The baseline keeps only `get_schema`; flip the flag on (e.g. `--extra "--enable_table_schema_tools true"`) for ablations.
- `psql_console` (ablation, `enable_psql_console`, default off) runs a real `psql -X -c <command>` subprocess under `PGOPTIONS='-c default_transaction_read_only=on -c statement_timeout=60s'` — `SELECT` + read-only meta-commands (`\dt`, `\d`, `\l`, `\df`) work; writes/DDL are rejected by the server. A guardrail (`_violates_psql_guardrail`) refuses host-reaching meta-commands (`\!`, `\o`, `\copy`, `\i`, `\e`, `\w`, `\s`, and `\g`/`\gx` with a file/pipe arg) **before** psql is spawned, returning `PSQL_GUARDRAIL_REFUSAL`. On nonzero exit it returns stderr; output is truncated to `MAX_RESULT_LENGTH`. When on it **replaces** `execute_sql`/`get_schema`/`get_table_names`/`get_table_schema` and is **mutually exclusive** with `enable_table_schema_tools` (enforced in `ConfigReader` and `run_agent_bird_baseline._select_db_tools`). Drive it via `--extra "--enable_psql_console true"`; the run-dir slug gets a `__psql` suffix.
