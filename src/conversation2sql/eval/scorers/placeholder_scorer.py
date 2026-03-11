from conversation2sql.eval import SampleWithPred, SampleWithPredScore, BaseScorer
from conversation2sql.eval.registry import scorer_registry


@scorer_registry.register
class PlaceholderScorer(BaseScorer):
    """Always marks predictions as incorrect (score=0) for demo purposes."""

    def score(self, predictions: list[SampleWithPred]) -> list[SampleWithPredScore]:
        return [SampleWithPredScore(score=0.0, **sample.model_dump()) for sample in predictions]
