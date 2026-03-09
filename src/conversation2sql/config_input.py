from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Input Data for DatasetReader
# ---------------------------------------------------------------------------

class ConfigReader(BaseModel):
    reader_name: str
    dataset_name: str
    dataset_revision: str | None = None


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


