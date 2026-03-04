"""Evaluation package for the BIRD-Interact framework."""

__all__: list[str] = [
    "BaseSuccessCriterion",
    "SuccessRate",
    "execute_sql",
    "load_or_download_dataset",
    "result_sets_match",
]


def __getattr__(name: str) -> object:
    """Lazily import public symbols to avoid heavyweight dependencies at import time."""
    if name in ("BaseSuccessCriterion", "SuccessRate", "execute_sql", "result_sets_match"):
        from src.evaluation.evaluation_metrics import (
            BaseSuccessCriterion,
            SuccessRate,
            execute_sql,
            result_sets_match,
        )
        return {
            "BaseSuccessCriterion": BaseSuccessCriterion,
            "SuccessRate": SuccessRate,
            "execute_sql": execute_sql,
            "result_sets_match": result_sets_match,
        }[name]

    if name == "load_or_download_dataset":
        from src.evaluation.read_dataset import load_or_download_dataset
        return load_or_download_dataset

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")