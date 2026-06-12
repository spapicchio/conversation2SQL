from conversation2sql.eval_framework.agents.bird_baseline.tools.utils_db_execute import (
    _execute_query,
)

from conversation2sql.eval_framework.agents.bird_baseline.tools.tool_specs import (
    ToolSpec,
    format_cost,
    stamp_cost_in_descriptions,
)
from conversation2sql.eval_framework.agents.bird_baseline.tools.bird_interact_env_tools import (
    execute_sql,
    psql_console,
    create_python_udf,
    get_all_column_meanings,
    get_schema,
    get_table_names,
    get_table_schema,
    get_column_meaning,
    get_all_external_knowledge_names,
    get_knowledge_definition,
    get_all_knowledge_definitions,
    DB_TOOL_COSTS,
    DB_TOOL_SPECS,
    KNOWLEDGE_VISIBLE_FIELDS,
    ExecuteSQLResponse,
    UDFParameter,
    apply_column_comments_impl,
    create_python_udf_impl,
    execute_sql_impl,
    psql_console_impl,
    get_all_column_meanings_impl,
    get_all_external_knowledge_names_impl,
    get_all_knowledge_definitions_impl,
    get_column_meaning_impl,
    get_knowledge_definition_impl,
    get_schema_impl,
    get_table_names_impl,
    get_table_schema_impl,
)

from conversation2sql.eval_framework.agents.bird_baseline.tools.bird_interact_user_tools import (
    USER_TOOL_COSTS,
    USER_TOOL_SPECS,
    return_tool_ask_user,
    submit_sql,
    _extract_group_in_tag_pattern,
    ask_user_impl,
    stage_1_parse_action,
    stage_2_generator,
    submit_sql_impl,
)

# Single source of truth: per-tool cost + prompt summary. TOOL_COSTS is derived
# for the legacy callers (middleware, metrics) that only need the number.
TOOL_SPECS: dict[str, ToolSpec] = {**DB_TOOL_SPECS, **USER_TOOL_SPECS}
TOOL_COSTS: dict[str, float] = {name: spec.cost for name, spec in TOOL_SPECS.items()}

# Stamp each module-level tool's bird-coin cost onto its schema description from
# the spec, so the LLM sees the cost in the tool schema without it being
# hardcoded in the docstring. ask_user is not a module singleton (built per run
# by return_tool_ask_user) and is stamped inside that factory instead.
stamp_cost_in_descriptions(
    [
        execute_sql,
        psql_console,
        create_python_udf,
        get_schema,
        get_table_names,
        get_table_schema,
        get_all_column_meanings,
        get_column_meaning,
        get_all_external_knowledge_names,
        get_knowledge_definition,
        get_all_knowledge_definitions,
        submit_sql,
    ],
    TOOL_SPECS,
)

# Content prefixes of the messages LangChain's built-in limit middlewares inject
# when they fire (ToolCallLimitMiddleware in "continue" mode → a ToolMessage;
# ModelCallLimitMiddleware in "end" mode → a final AIMessage). These are NOT our
# own tool outputs — they mirror strings hard-coded inside LangChain — so they
# are a one-way *detection contract*: `extract_middleware_events` matches on them
# to know a limit fired. Changing a value here only changes the matcher; it does
# not change what LangChain emits, so it must stay in sync with the library.
TOOL_CALL_LIMIT_PREFIX = "Tool call limit exceeded."
MODEL_CALL_LIMIT_PREFIX = "Model call limits exceeded:"

__all__ = [
    # db execute helper
    "_execute_query",
    # env tools (LangGraph @tool wrappers)
    "execute_sql",
    "psql_console",
    "create_python_udf",
    "get_schema",
    "get_table_names",
    "get_table_schema",
    "get_all_column_meanings",
    "get_column_meaning",
    "get_all_external_knowledge_names",
    "get_knowledge_definition",
    "get_all_knowledge_definitions",
    # env tools (pure *_impl functions)
    "apply_column_comments_impl",
    "create_python_udf_impl",
    "execute_sql_impl",
    "psql_console_impl",
    "get_schema_impl",
    "get_table_names_impl",
    "get_table_schema_impl",
    "get_all_column_meanings_impl",
    "get_column_meaning_impl",
    "get_all_external_knowledge_names_impl",
    "get_knowledge_definition_impl",
    "get_all_knowledge_definitions_impl",
    # env tools (types / constants)
    "ExecuteSQLResponse",
    "UDFParameter",
    "KNOWLEDGE_VISIBLE_FIELDS",
    "DB_TOOL_COSTS",
    "DB_TOOL_SPECS",
    # user tools (LangGraph @tool wrappers)
    "return_tool_ask_user",
    "submit_sql",
    # user tools (pure functions)
    "_extract_group_in_tag_pattern",
    "ask_user_impl",
    "stage_1_parse_action",
    "stage_2_generator",
    "submit_sql_impl",
    # user tools (constants)
    "USER_TOOL_COSTS",
    "USER_TOOL_SPECS",
    # spec registry
    "ToolSpec",
    "format_cost",
    "stamp_cost_in_descriptions",
    # merged
    "TOOL_COSTS",
    "TOOL_SPECS",
    # built-in limit-middleware message prefixes (detection contract)
    "TOOL_CALL_LIMIT_PREFIX",
    "MODEL_CALL_LIMIT_PREFIX",
]
