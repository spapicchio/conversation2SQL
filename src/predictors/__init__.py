"""Predictor protocol and implementations for tool-calling evaluation."""

__all__: list[str] = [
    "ToolCallingPredictor",
    "LiteLLMPredictor",
    "VLLMPredictor",
    "HuggingFacePredictor",
]


def __getattr__(name: str) -> object:
    """Lazily import predictor classes to avoid loading heavy dependencies eagerly."""
    if name == "ToolCallingPredictor":
        from src.predictors.protocol import ToolCallingPredictor
        return ToolCallingPredictor

    if name == "LiteLLMPredictor":
        from src.predictors.litellm_predictor import LiteLLMPredictor
        return LiteLLMPredictor

    if name == "VLLMPredictor":
        from src.predictors.vllm_predictor import VLLMPredictor
        return VLLMPredictor

    if name == "HuggingFacePredictor":
        from src.predictors.hf_predictor import HuggingFacePredictor
        return HuggingFacePredictor

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
