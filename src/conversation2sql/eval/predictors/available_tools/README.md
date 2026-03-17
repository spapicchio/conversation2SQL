# Available Tools — Reference

This directory contains LangChain `@tool`-decorated functions that are registered in the global `tool_registry` and can be bound to any predictor via the `tool_names` field in `ConfigPredictor`.

---

## File Layout

```
available_tools/
├── __init__.py                   # imports all tools (triggers tool_registry registration)
├── tool_placeholder.py           # generic stubs (search, get_weather, ask_user via LLM)
├── bird_interact_env_tools.py    # Bird-Interact database / environment tools
└── bird_interact_user_tools.py   # Bird-Interact user interaction tools (ask + submit)
```

---

## Tool Registry

Every tool is registered under a string key via `@tool_registry.register(name="…")`.  Pass those keys in your YAML config:

```yaml
predictor:
  tool_names:
    - get_schema
    - get_column_meaning
    - get_all_external_knowledge_names
    - get_knowledge_definition
    - execute_sql
    - ask_user
    - submit_sql
```

---

## Bird-Interact Environment Tools (`bird_interact_env_tools.py`)

These tools correspond to the **Environment** interaction object in the Bird-Interact agent prompt.  They require `UserContext.db_path` to be set (populated by the dataset reader).

| Registry name | Function | Description | Patience cost |
|---|---|---|---|
| `execute_sql` | `execute_sql(sql, runtime)` | Execute one or more SQL statements; returns JSON rows | 1 |
| `get_schema` | `get_schema(runtime)` | Return DDL + 3 sample rows per table | 1 |
| `get_all_column_meanings` | `get_all_column_meanings(runtime)` | Return all column descriptions from `UserContext.column_meanings` | 1 |
| `get_column_meaning` | `get_column_meaning(table_name, column_name, runtime)` | Return description for a single column | 0.5 |
| `get_all_external_knowledge_names` | `get_all_external_knowledge_names(runtime)` | List all knowledge entry names from `UserContext.external_knowledge` | 0.5 |
| `get_knowledge_definition` | `get_knowledge_definition(knowledge_name, runtime)` | Return the full definition for one knowledge entry | 0.5 |
| `get_all_knowledge_definitions` | `get_all_knowledge_definitions(runtime)` | Return all knowledge entries with definitions | 1 |

### Required `UserContext` fields

| Field | Type | Description |
|---|---|---|
| `db_path` | `str \| None` | Absolute path to the SQLite database file |
| `database_engine` | `str` | Database engine identifier (default `"sqlite"`) |
| `external_knowledge` | `list[dict]` | List of `{knowledge, description, definition}` dicts from the GT file |
| `column_meanings` | `dict[str, dict[str, str]]` | Nested map `{table: {column: meaning}}` from the GT file |

---

## Bird-Interact User Tools (`bird_interact_user_tools.py`)

These tools correspond to the **User** interaction object in the Bird-Interact agent prompt.

| Registry name | Function | Description | Patience cost |
|---|---|---|---|
| `ask_user` | `ask_user(question, runtime)` | Ask the user-simulator LLM one clarification question | 2 |
| `submit_sql` | `submit_sql(sql, runtime)` | Submit the final SQL for evaluation; stores it in `template_params["submitted_sql"]` | 3 |

`ask_user` calls the LLM configured in `UserContext` (user-simulator prompt folder + templates).
`submit_sql` stores the SQL in `UserContext.template_params["submitted_sql"]` for downstream scorer access.  To wire up execution-based feedback, subclass or replace `submit_sql` and connect it to the evaluation harness.

### Required `UserContext` fields

| Field | Type | Description |
|---|---|---|
| `user_simulator_prompt_folder` | `str` | Path to the Jinja prompt folder for the user simulator |
| `user_simulator_system_prompt` | `str \| None` | System prompt template name (optional) |
| `user_simulator_user_prompt` | `str` | User prompt template name |
| `template_params` | `dict` | Template variables (e.g. `user_query`, `interaction_history`); mutated at runtime |

---

## Generic Stubs (`tool_placeholder.py`)

| Registry name | Function | Description |
|---|---|---|
| `search` | `search(query)` | Stub web search |
| `get_weather` | `get_weather(location)` | Stub weather lookup |
| `sql_query_tool` | `sql_query_tool(query)` | Stub SQL execution |
| `db_query_tool` | `db_query_tool(query)` | Stub DB execution |
| `user` | `ask_user(question, runtime)` | Legacy ask-user tool (registered as `"user"`; prefer `ask_user` for new configs) |

---

## Adding a New Tool

1. Create a `@tool`-decorated function in this directory (new file or existing module).
2. Decorate with `@tool_registry.register(name="my_tool")` **above** `@tool`.
3. Import from `available_tools/__init__.py`.
4. List `my_tool` under `tool_names` in the YAML config.

If the tool needs runtime context (database path, user simulator config, etc.), add a `runtime: ToolRuntime[UserContext]` parameter — LangGraph injects it automatically.

---

## See Also

- [`predictors/README.md`](../README.md) — how predictors use tools and how `LangChainAgentFactory` binds them
- [`eval/interfaces.py`](../../interfaces.py) — `UserContext` model definition
- [`eval/registry.py`](../../registry.py) — `tool_registry` implementation
