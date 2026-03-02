"""User simulator tool powered by a vLLM language model.

Simulates a human user clarifying an ambiguous Text-to-SQL request so that
the agent can iteratively refine its understanding before generating SQL.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.pipeline.tools.base import BaseTool

if TYPE_CHECKING:
    from vllm import LLM, SamplingParams

_DEFAULT_SYSTEM_PROMPT = (
    "You are a database user who asked a question that may be ambiguous. "
    "A SQL assistant is asking you a clarifying question. "
    "Answer concisely based on the provided context."
)


class UserSimulatorTool(BaseTool):
    """Simulate a human user responding to clarifying questions.

    The tool holds a reference to a :class:`vllm.LLM` engine used to
    generate the simulated user's responses.  An external engine can be
    injected at construction time to share resources with the rest of the
    pipeline; otherwise a new engine is created.
    """

    name: str = "user_simulator"
    description: str = (
        "Simulate a user response to a clarifying question about an "
        "ambiguous Text-to-SQL request."
    )

    def __init__(
        self,
        llm: LLM | None = None,
        model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
        system_prompt: str = _DEFAULT_SYSTEM_PROMPT,
        sampling_params: SamplingParams | None = None,
    ) -> None:
        """Initialise the user simulator.

        Args:
            llm: An existing :class:`vllm.LLM` instance to reuse.
                If ``None``, a new engine is created from *model_name*.
            model_name: HuggingFace model identifier used when *llm* is
                not provided.
            system_prompt: System-level instruction prepended to every
                generation request.
            sampling_params: vLLM sampling configuration.  Defaults to
                ``SamplingParams(temperature=0.7, max_tokens=256)``.
        """
        from vllm import LLM as _LLM
        from vllm import SamplingParams as _SamplingParams

        self._llm: LLM = llm if llm is not None else _LLM(model=model_name)
        self._system_prompt = system_prompt
        self._sampling_params = sampling_params or _SamplingParams(
            temperature=0.7, max_tokens=256
        )

    def execute(self, **kwargs: Any) -> str:
        """Generate a simulated user response.

        Keyword Args:
            clarifying_question (str): The question the agent is asking the
                user.  **Required.**
            context (str): Background context about the original question
                (e.g. schema, original utterance).  Defaults to ``""``.

        Returns:
            The simulated user's textual response.
        """
        clarifying_question: str = kwargs["clarifying_question"]
        context: str = kwargs.get("context", "")

        prompt = (
            f"{self._system_prompt}\n\n"
            f"Context: {context}\n\n"
            f"Clarifying question: {clarifying_question}\n\n"
            "Your response:"
        )

        outputs = self._llm.generate([prompt], self._sampling_params)
        return outputs[0].outputs[0].text.strip()
