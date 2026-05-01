from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Input Data for Pipeline
# ---------------------------------------------------------------------------
class ConfigPipeline(BaseModel):
    debug: bool = True
    mode: str = 'a-interact'  # a-interact | c-interact | oracle 
    output_folder: str = "results"
    concurrency: int = Field(default=5, description="Number of parallel tasks to run")


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


# ---------------------------------------------------------------------------
# Input Data for Predictor
# ---------------------------------------------------------------------------

class ConfigPredictor(BaseModel):
    model_name: str = 'qwen/qwen3.6-flash'  # The model name or path to be used for prediction, e.g., 'gpt-3.5-turbo', 'text-embedding-3-small', etc.
    model_provider: str = 'openrouter'  # The model provider, e.g., 'openai', 'azure', 'anthropic', etc.
    temperature: float = 0.0
    top_p: float = 1
    max_new_tokens: int = 2000


class ConfigUserSimulator(BaseModel):
    model_name: str = 'gpt-3.5-turbo'  # The model name or path to be used for prediction, e.g., 'gpt-3.5-turbo', 'text-embedding-3-small', etc.
    model_provider: str = 'openai' # The model provider, e.g., 'openai', 'azure', 'anthropic', etc.
    temperature: float = 0.0
    max_new_tokens: int = 500
