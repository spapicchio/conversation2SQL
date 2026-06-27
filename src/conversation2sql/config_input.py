from typing import Literal

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Input Data for Pipeline
# ---------------------------------------------------------------------------
class ConfigPipeline(BaseModel):
    debug: bool = True
    output_folder: str = "results"
    baseline: Literal['no_tool', 'tools_only', 'tools_user', 'bird_full', 'deep_agent'] = 'bird_full'
    concurrency: int = 1
    num_iterations: int = Field(default=3, ge=1)  # repeat the dataset N times for statistical relevance
    resume: bool = False  # skip (instance_id, iteration) pairs already in output_folder/results_iter*.jsonl and run only the missing ones


# ---------------------------------------------------------------------------
# Input Data for DatasetReader
# ---------------------------------------------------------------------------

class ConfigReader(BaseModel):
    dataset_name_jsonl: str = 'data/bird_interact/bird-interact-full/bird_interact_data_GT.jsonl'
    dataset_path: str = 'data/bird_interact/bird-interact-full'
    filter_query_category: bool = True
    db_dsn_template: str = 'postgresql://root:123123@localhost:5432/{database}'  # DSN template; {database} is replaced with the per-sample database name
    user_patience_budget: int = 10
    make_data_ambiguous: bool = True
    database_schema_type: Literal['ddl', 'toon'] = 'ddl'  # Whether to use the original complex schema or a simplified version for better model understanding
    read_only_gt_kb: bool = False  # Whether to only include the tables/columns that are actually used in the GT SQL query when providing the schema to the model
    read_only_gt_tables: bool = False  # Whether to only include the tables that are actually used in the GT SQL query when providing the schema to the model   
    is_kb_linearized: bool = False  # Whether to linearize the KB schema into text or provide it in a structured format (e.g., JSON); linearization may be easier for LLMs to understand but less faithful to the original structure
    enable_table_schema_tools: bool = False  # When True the agent additionally gets get_table_names + get_table_schema (granular per-table DDL access, mirroring the KB name/definition tools). Off by default so the baseline keeps only the full-dump get_schema; flip on for ablations.
    enable_psql_console: bool = False  # Ablation: replace the DB tools (execute_sql/get_schema/get_table_*) with a single read-only psql terminal tool (psql_console). Mutually exclusive with enable_table_schema_tools.
    enable_psql_strict_inspection: bool = False  # Ablation (only meaningful with enable_psql_console): restrict psql_console to SQL + \h + the informational \d-family; \? lists only those. Off = legacy denylist behavior (full rollback).
    enable_python_udf: bool = False  # Ablation: add create_python_udf tool (plpython3u). Additive — compatible with all other DB-tool variants.

    # --- deep_agent baseline ablations (only meaningful when baseline='deep_agent') ---
    deep_enable_todos: bool = False  # add deepagents planning/write_todos middleware
    deep_enable_subagents: bool = False  # add deepagents subagents (task tool) middleware
    deep_enable_summarization: bool = False  # add deepagents/langchain SummarizationMiddleware
    deep_enable_fs_write: bool = False  # expose write_file/edit_file (default: read-only FS)
    deep_catalog_root: str = ''  # root holding <db>/tables/*.md catalogs for the deep_agent FS (e.g. data/bird_interact/catalog_bird_interact_lite); set per run

    @model_validator(mode="after")
    def _check_db_tool_ablation_exclusivity(self) -> "ConfigReader":
        if self.enable_psql_console and self.enable_table_schema_tools:
            raise ValueError(
                "enable_psql_console and enable_table_schema_tools are mutually "
                "exclusive; enable at most one DB-tool ablation."
            )
        return self
# ---------------------------------------------------------------------------
# Input Data for Predictor
# ---------------------------------------------------------------------------

class ConfigPredictor(BaseModel):
    model_name: str = 'qwen/qwen3.6-flash'  # The model name or path to be used for prediction, e.g., 'gpt-3.5-turbo', 'text-embedding-3-small', etc.
    model_provider: str = 'openrouter'  # The model provider, e.g., 'openai', 'azure', 'anthropic', etc.
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 20
    min_p: float = 0.0
    presence_penalty: float = 0.0
    repetition_penalty: float = 1.0   
    max_new_tokens: int = 2000
    reasoning_effort: str | None = None  # The level of reasoning effort for the model, e.g., 'low', 'medium', 'high'. This can be used to control how much intermediate reasoning the model generates before producing the final answer. The exact interpretation depends on the implementation in the ChatLiteLLM class.
    predictor_vllm_api_base: str | None = None  # If using vLLM API for prediction, the base URL of the API, e.g., 'http://localhost:8000/v1'
    enable_thinking: bool | None = None  # Whether to enable the "thinking" mode in the chat template, which allows the model to generate intermediate reasoning steps before the final answer
    request_timeout: float | None = 600.0  # Per-request timeout (s) for model calls; a stuck request fails fast instead of hanging a worker thread until the asyncio executor-join watchdog trips
    num_retries: int = 2  # How many times litellm retries a failed/timed-out request before giving up

class ConfigUserSimulator(BaseModel):
    model_name: str = 'gpt-3.5-turbo'  # The model name or path to be used for prediction, e.g., 'gpt-3.5-turbo', 'text-embedding-3-small', etc.
    model_provider: str = 'openai' # The model provider, e.g., 'openai', 'azure', 'anthropic', etc.
    temperature: float = 0.0
    max_new_tokens: int = 500
    top_p: float | None = None
    top_k: int | None= None
    presence_penalty: float | None = None
    repetition_penalty: float | None = None
    reasoning_effort: str | None = None  # The level of reasoning effort for the model, e.g., 'low', 'medium', 'high'. This can be used to control how much intermediate reasoning the model generates before producing the final answer. The exact interpretation depends on the implementation in the ChatLiteLLM class.
    user_simulator_vllm_api_base: str | None = None  # If using vLLM API for user simulation, the base URL of the API, e.g., 'http://localhost:8000/v1'
    enable_thinking: bool | None = None  # Whether to enable the "thinking" mode in the chat template, which allows the model to generate intermediate reasoning steps before the final answer
    request_timeout: float | None = 600.0  # Per-request timeout (s) for model calls; a stuck request fails fast instead of hanging a worker thread until the asyncio executor-join watchdog trips
    num_retries: int = 2  # How many times litellm retries a failed/timed-out request before giving up