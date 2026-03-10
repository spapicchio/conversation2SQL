from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Input Data for DatasetReader
# ---------------------------------------------------------------------------

class ConfigReader(BaseModel):
    reader_name: str
    dataset_name: str
    dataset_kwargs: dict = Field(default_factory=dict)
    database_engine: str = 'sqlite'  # Default to SQLite, can be overridden to use other databases like PostgreSQL, MySQL, etc.

    # Prompt related fields
    prompt_dir: str = 'prompts'  # Directory where Jinja prompt templates are stored
    system_prompt: str | None = None  # System prompt for chat models
    user_prompt: str = 'user.jinja'  # User prompt for chat models.
    # For non-chat templates, the system prompt and the user prompt are concatenated together to form the final prompt template.
    is_chat_template: bool = False  # Flag to indicate if the prompt template is for chat models


# ---------------------------------------------------------------------------
# Input Data for Predictor
# ---------------------------------------------------------------------------

class ConfigPredictor(BaseModel):
    predictor_name: str
    model_name: str
    model_revision: str | None = None
    run_pred_in_parallel: bool = False


# ---------------------------------------------------------------------------
# Input Data for Scorer Name
# ---------------------------------------------------------------------------

class ConfigScorer(BaseModel):
    scorer_name: str
    run_score_in_parallel: bool = False
