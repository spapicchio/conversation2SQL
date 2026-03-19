from conversation2sql.eval import Sample, SampleWithPred, BasePredictor
from conversation2sql.eval.registry import predictor_registry


@predictor_registry.register
class PlaceholderPredictor(BasePredictor):
    """Returns a canned AIMessage for every input — no LLM call required."""

    def predict(self, samples: list[Sample]) -> list[SampleWithPred]:
        return [SampleWithPred(**sample.model_dump()) for sample in samples]
