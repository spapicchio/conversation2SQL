from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Input Data for DatasetReader
# ---------------------------------------------------------------------------

class ConfigReader(BaseModel):
    reader_name: str
    dataset_name: str
    dataset_kwargs: dict = Field(default_factory=dict)
    database_engine: str = 'postgresql'  # Database engine label surfaced in prompt templates (e.g. 'postgresql')
    db_dsn_template: str = 'postgresql://root:123123@localhost:5432/{database}'  # DSN template; {database} is replaced with the per-sample database name

    # Prompt related fields
    prompt_dir: str = 'prompts'  # Directory where Jinja prompt templates are stored
    system_prompt: str | None = None  # System prompt for chat models
    user_prompt: str = 'user.jinja'  # User prompt for chat models.
    # For non-chat templates, the system prompt and the user prompt are concatenated together to form the final prompt template.
    is_chat_template: bool = False  # Flag to indicate if the prompt template is for chat models

    # USER simulator's params
    user_simulator_prompt_folder: str = 'prompts/bird_interact_user_simulator'
    user_simulator_system_prompt: str | None = None
    user_simulator_user_prompt: str = 'simulator_base.jinja'

# ---------------------------------------------------------------------------
# Input Data for Predictor
# ---------------------------------------------------------------------------

class ConfigPredictor(BaseModel):
    predictor_name: str  # The class name of the predictor to be used, e.g., 'LLMPredictor', 'EmbeddingPredictor', etc.
    model_name: str  # The model name or path to be used for prediction, e.g., 'gpt-3.5-turbo', 'text-embedding-3-small', etc.
    model_provider: str  # The model provider, e.g., 'openai', 'huggingface', etc.
    tool_names: list[str] = Field(default_factory=list)  # List of tool names to be used by the predictor, e.g., ['calculator', 'search'], etc.
    temperature: float
    top_k: int
    top_p: float
    max_new_tokens: int
    user_patience_budget: int = 10


# ---------------------------------------------------------------------------
# Input Data for Scorer Name
# ---------------------------------------------------------------------------

class ConfigScorer(BaseModel):
    scorer_name: str
    run_score_in_parallel: bool = False
