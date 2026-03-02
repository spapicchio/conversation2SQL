"""Core orchestrator for the BIRD-Interact agent pipeline.

Combines Jinja2-based prompt rendering, vLLM inference, and a pluggable
tool system into a single :class:`AgentPipeline` that iteratively
generates SQL while resolving ambiguity through tool calls.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from jinja2 import Environment, FileSystemLoader

from src.pipeline.tools.base import BaseTool

if TYPE_CHECKING:
    from vllm import LLM, SamplingParams

_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


def load_prompt(template_path: str, **kwargs: Any) -> str:
    """Render a Jinja2 template from the ``src/prompts/`` directory.

    Args:
        template_path: Path to the template *relative* to the prompts
            directory (e.g. ``"agent_prompt.jinja"``).
        **kwargs: Variables passed into the Jinja2 template context.

    Returns:
        The rendered prompt string.
    """
    env = Environment(
        loader=FileSystemLoader(str(_PROMPTS_DIR)),
        keep_trailing_newline=True,
    )
    template = env.get_template(template_path)
    return template.render(**kwargs)


class AgentPipeline:
    """Agentic Text-to-SQL pipeline backed by vLLM.

    The pipeline renders a Jinja2 prompt, sends it to vLLM, and parses
    the model output to determine whether a tool should be called.  Tools
    are supplied as a list at construction time and can be added or removed
    at runtime via :meth:`register_tool` and :meth:`remove_tool`.

    Args:
        model_name: HuggingFace model identifier for vLLM.
        tools: Initial list of :class:`BaseTool` instances available to
            the agent.
        llm: An existing :class:`vllm.LLM` instance. When provided,
            *model_name* is ignored.
        sampling_params: vLLM sampling configuration.
        max_iterations: Maximum number of agent loop iterations.
    """

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
        tools: list[BaseTool] | None = None,
        llm: LLM | None = None,
        sampling_params: SamplingParams | None = None,
        max_iterations: int = 10,
    ) -> None:
        from vllm import LLM as _LLM
        from vllm import SamplingParams as _SamplingParams

        self._llm: _LLM = llm if llm is not None else _LLM(model=model_name)
        self._sampling_params = sampling_params or _SamplingParams(
            temperature=0.0, max_tokens=512
        )
        self._tools: dict[str, BaseTool] = {t.name: t for t in (tools or [])}
        self._max_iterations = max_iterations

    # -- Tool management ----------------------------------------------------

    def register_tool(self, tool: BaseTool) -> None:
        """Register a tool so the agent can invoke it.

        Args:
            tool: Tool instance to add.
        """
        self._tools[tool.name] = tool

    def remove_tool(self, tool_name: str) -> None:
        """Remove a previously registered tool by name.

        Args:
            tool_name: Name of the tool to remove.

        Raises:
            KeyError: If no tool with that name is registered.
        """
        del self._tools[tool_name]

    @property
    def tools(self) -> list[BaseTool]:
        """Return a snapshot of currently registered tools."""
        return list(self._tools.values())

    # -- Prompt helpers -----------------------------------------------------

    def _tool_descriptors(self) -> list[dict[str, str]]:
        """Serialise tool metadata for the prompt template."""
        return [
            {"name": t.name, "description": t.description}
            for t in self._tools.values()
        ]

    # -- Core loop ----------------------------------------------------------

    def run(
        self,
        user_question: str,
        schema: str,
        database_path: str,
        template_path: str = "agent_prompt.jinja",
    ) -> str:
        """Execute the agent loop until a final SQL answer is produced.

        Args:
            user_question: The natural-language question from the user.
            schema: Database schema description.
            database_path: Path to the SQLite database.
            template_path: Jinja2 template file name inside ``src/prompts/``.

        Returns:
            The final SQL query string.
        """
        conversation_history: list[dict[str, str]] = []

        for _ in range(self._max_iterations):
            prompt = load_prompt(
                template_path,
                user_question=user_question,
                schema=schema,
                tools=self._tool_descriptors(),
                conversation_history=conversation_history,
            )

            outputs = self._llm.generate([prompt], self._sampling_params)
            raw_response: str = outputs[0].outputs[0].text.strip()

            conversation_history.append(
                {"role": "assistant", "content": raw_response}
            )

            action = self._parse_response(raw_response)

            if action.get("action") == "final_answer":
                return action.get("sql", "")

            if action.get("action") == "tool_call":
                tool_name: str = action.get("tool_name", "")
                tool_input: dict[str, Any] = action.get("tool_input", {})

                if tool_name in self._tools:
                    tool_input.setdefault("database_path", database_path)
                    result = self._tools[tool_name].execute(**tool_input)
                    conversation_history.append(
                        {"role": "tool", "content": str(result)}
                    )
                else:
                    conversation_history.append(
                        {
                            "role": "tool",
                            "content": f"Error: tool '{tool_name}' not found.",
                        }
                    )

        return ""

    # -- Parsing ------------------------------------------------------------

    @staticmethod
    def _parse_response(text: str) -> dict[str, Any]:
        """Extract the JSON action block from the model's raw output.

        Args:
            text: Raw text generated by the LLM.

        Returns:
            Parsed dict with at least an ``"action"`` key, or an empty dict
            if parsing fails.
        """
        # Try to locate a JSON object in the text
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass
        return {}
