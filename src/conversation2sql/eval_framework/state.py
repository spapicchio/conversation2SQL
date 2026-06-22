from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class FollowUpPayload(BaseModel):
    """Optional follow-up question/answer block for BIRD-Interact tasks.

    The benchmark may include a second-phase follow-up question that should only
    be asked after a successful first answer.
    """

    model_config = ConfigDict(extra="allow")
    query: str = ""
    sol_sql: str | list[str] = Field(default_factory=list)


class ExternalKnowledgeEntry(BaseModel):
    # Refer to dataset_readers/bird_interact.md for full description of these fields.
    id: int  # id: the integer id of the kb entry, which is the same as the id in the database kb jsonl file.
    knowledge: str  # a short name associated with the knowledge entry.
    description: str  # the description of the knowledge entry.
    definition: str  # the definition of the knowledge entry based on Mathematical formula or decision rule.
    type: str  # the type of the knowledge entry, which can be one of "calculation_knowledge", "domain_knowledge", "value_illustration".
    children_knowledge: list[int]  # list of IDs this entry depends on, or -1 if none


class ColumnMeaningEntry(BaseModel):
    column_meaning: str
    fields_meaning: dict[str, dict | str | list]  |  None = None


class TaskData(BaseModel):
    """Typed representation of one evaluation task loaded from JSONL.

    This model mirrors the core task fields used by the orchestrator.  Unknown
    fields are preserved (``extra='allow'``) so the runner can forward dataset-
    specific metadata to services without losing information.
    """

    model_config = ConfigDict(extra="allow")
    task_question: str # possible to be either amb_user_query or not_ambiguos_query depending on the modality
    instance_id: str
    selected_database: str
    amb_user_query: str
    sol_sql: list[str]
    not_ambiguos_query: str

    follow_up: FollowUpPayload | None = None

    task_budget: int
    db_dsn: str  # PostgreSQL DSN, e.g. "postgresql://root:123123@localhost:5432/mydb"

    database_engine: str = "postgresql"
    ddl_database_schema: str
    # External knowledge and column meanings loaded from the dataset
    # dicts: {ExternalKnowledgeEntry.knowledge: ExternalKnowledgeEntry}
    full_knowledge_base: dict[str, ExternalKnowledgeEntry] = Field(default_factory=dict)
    masked_agent_kb: dict[str, ExternalKnowledgeEntry] = Field(default_factory=dict)
    # When True the KB tools linearize output via agents/utils_kb_linearize.py.
    is_kb_linearized: bool = False
    # When True the agent gets the granular get_table_names / get_table_schema
    # tools in addition to the full-dump get_schema (off = baseline behavior).
    enable_table_schema_tools: bool = False
    # When True the agent gets the single read-only psql_console tool INSTEAD of
    # execute_sql/get_schema/get_table_* (ablation). Mutually exclusive with
    # enable_table_schema_tools (enforced in ConfigReader + run_agent_bird_baseline).
    enable_psql_console: bool = False
    # When True (and enable_psql_console), psql_console runs in strict inspection
    # mode: only SQL + \h + the informational \d-family are allowed and \? lists
    # only those (ablation, default off = legacy denylist behavior).
    enable_psql_strict_inspection: bool = False
    # When True the agent gets the create_python_udf tool (plpython3u ablation).
    # Additive — compatible with enable_psql_console and enable_table_schema_tools.
    enable_python_udf: bool = False
    gt_knowledge_base: dict[str, ExternalKnowledgeEntry] = Field(default_factory=dict)
    #  key = f"{db_name}|{req.table_name.lower()}|{req.column_name.lower()}"
    column_meanings: dict[str, ColumnMeaningEntry] = Field(default_factory=dict)
    user_query_ambiguity: dict  # the ambiguities injected into the user query.
    clean_up_sqls: list[
        str
    ]  # SQL queries to run after the test cases to revert any changes made to the database.
    preprocess_sql: list[
        str
    ]  # SQL queries to run before executing the solution or prediction
    test_cases: list[
        str
    ]  # possible string representing the python code to run as a unit test
    category: str  # the category of the task, which can be one of "Query", "Management"
    sql_query_conditions: dict  # dict containing "decimal": -1/1
    table_in_gt_sql: dict[str, list[str]] = Field(default_factory=dict)
    table_in_gt_sql_parse_error: str | None = None

    def has_follow_up_sql(self) -> bool:
        """Return ``True`` when a second-phase SQL answer is present.

        The source data can encode ``follow_up`` in slightly different shapes,
        so this helper normalizes those variants.
        """
        if not self.follow_up:
            return False

        fu_sql = self.follow_up.sol_sql

        if isinstance(fu_sql, str):
            return bool(fu_sql.strip())

        return bool(fu_sql)

    def follow_up_query(self) -> str:
        """Return follow-up query text (empty string when missing)."""
        return self.follow_up.query if self.follow_up else ""

    def follow_up_sql_list(self) -> list[str]:
        """Normalize follow-up SQL to ``list[str]`` for downstream execution."""

        fu_sql = self.follow_up.sol_sql if self.follow_up else []
        if isinstance(fu_sql, str):
            return [fu_sql] if fu_sql else []
        return [str(x) for x in fu_sql]


class EvaluationMetrics(BaseModel):
    """Aggregate metrics tracked while processing a full task set."""

    total_tasks: int
    total_reward: float
    average_reward: float
    phase1_rate: float
    phase2_rate: float
    phase1_count: int
    phase2_count: int


class EvaluationOutput(BaseModel):
    """On-disk JSON structure written by ``run_parallel_evaluation``."""

    mode: str
    metrics: EvaluationMetrics
    results: list[dict[str, Any]]
