"""vLLM-backed predictor implementation.

Uses ``vllm.LLM`` for high-throughput offline inference.  Because vLLM
operates on raw text prompts, the tokenizer's ``apply_chat_template``
method is used to convert the standardized chat messages (and tool
definitions) into a single prompt string.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.predictors.parsing import parse_tool_call

if TYPE_CHECKING:
    from vllm import LLM, SamplingParams


class VLLMPredictor:
    """Predictor backed by the vLLM offline inference engine.

    Args:
        model_name: HuggingFace model identifier.
        llm: An existing :class:`vllm.LLM` instance to reuse.
            When provided, *model_name* is ignored.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens to generate.
    """

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
        llm: LLM | None = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> None:
        from vllm import LLM as _LLM
        from vllm import SamplingParams as _SamplingParams

        self._llm: _LLM = llm if llm is not None else _LLM(model=model_name)
        self._sampling_params: _SamplingParams = _SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
        )
        self._tokenizer = self._llm.get_tokenizer()

    async def generate(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Format messages via chat template and run vLLM generation.

        Args:
            messages: Chat messages in OpenAI format.
            tools: Tool definitions in OpenAI function-calling schema.

        Returns:
            Unified dict with either ``"type": "text"`` or
            ``"type": "tool_call"``.
        """
        prompt: str = self._tokenizer.apply_chat_template(
            messages,
            tools=tools if tools else None,
            tokenize=False,
            add_generation_prompt=True,
        )

        outputs = self._llm.generate([prompt], self._sampling_params)
        raw_text: str = outputs[0].outputs[0].text.strip()

        return parse_tool_call(raw_text)
