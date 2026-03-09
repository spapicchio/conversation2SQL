from conversation2sql.eval import Sample, SampleWithPred
from conversation2sql.eval.registry import predictor_registry


@predictor_registry.register
class PlaceholderPredictor:
    """Returns a canned AIMessage for every input — no LLM call required."""

    def predict(self, samples: list[Sample]) -> list[SampleWithPred]:
        return [SampleWithPred(prediction='placeholder', **sample.model_dump()) for sample in samples]
