"""PromptFactory: Centralized Jinja2-based prompt template loader and renderer.

Usage:
    from conversation2sql.prompt_factory import PromptFactory

    factory = PromptFactory()
    system = factory.get_system_prompt("eval", schema="CREATE TABLE ...")
    user   = factory.get_user_prompt("eval", prompt="Show all users")
"""

from __future__ import annotations

from functools import cache
from pathlib import Path
from typing import Any

from frozendict import frozendict
from jinja2 import Environment, FileSystemLoader, StrictUndefined, TemplateNotFound
from jinja2 import meta as jinja2_meta
from pydantic import BaseModel

from conversation2sql.logger import get_logger

logger = get_logger(__name__)

# Default: <project_root>/prompts (3 parents up from src/conversation2sql/prompt_factory.py)
_DEFAULT_PROMPT_DIR = Path(__file__).resolve().parents[2] / "prompts"


class PromptFactory:
    """Load and render Jinja2 prompt templates from a directory tree.

    Expected directory layout::

        prompts/
        ├── base_system.jinja
        ├── partials/
        │   ├── few_shot.jinja
        │   └── output_format.jinja
        └── <task>/
            ├── system.jinja
            └── user.jinja

    Parameters
    ----------
    prompt_dir : str | Path | None
        Root directory containing the Jinja templates.
        Defaults to ``<project_root>/prompts``.
    """

    def __init__(self, prompt_dir: str | Path | None = None) -> None:
        self._prompt_dir = Path(prompt_dir) if prompt_dir else _DEFAULT_PROMPT_DIR
        if not self._prompt_dir.is_dir():
            raise FileNotFoundError(f"Prompt directory not found: {self._prompt_dir}")
        self._env = Environment(
            loader=FileSystemLoader(str(self._prompt_dir)),
            undefined=StrictUndefined,  # fail fast on missing vars
            keep_trailing_newline=True,
            trim_blocks=True,  # nicer whitespace
            lstrip_blocks=True,
            autoescape=False,  # we’re not rendering HTML
        )
        logger.debug(f"PromptFactory initialised with prompt_dir={self._prompt_dir}")

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    @cache
    def render_template(self, template_name: str, template_params: frozendict) -> str:
        """Render a template by relative path (e.g. ``"eval/user.jinja"``)."""
        try:
            template = self._env.get_template(template_name)
        except TemplateNotFound:
            raise FileNotFoundError(
                f"Template '{template_name}' not found in {self._prompt_dir}"
            )
        rendered = template.render(**template_params)
        logger.debug(f"Rendered template '{template_name}' ({len(rendered)} chars)")
        return rendered

    def get_system_prompt(self, task: str, **kwargs: Any) -> str:
        """Render ``<task>/system.jinja``."""
        return self.render_template(f"{task}/system.jinja", frozendict(**kwargs))

    def get_user_prompt(self, task: str, **kwargs: Any) -> str:
        """Render ``<task>/user.jinja``."""
        return self.render_template(f"{task}/user.jinja", frozendict(**kwargs))

    # ------------------------------------------------------------------
    # Optional: validated rendering via Pydantic
    # ------------------------------------------------------------------

    def render_validated(self, template_name: str, params: BaseModel) -> str:
        """Render a template using a Pydantic ``BaseModel`` instance.

        This ensures all required variables are present and correctly typed
        *before* the template is rendered.

        Parameters
        ----------
        template_name : str
            Relative path to the Jinja template.
        params : pydantic.BaseModel
            A Pydantic model whose fields map to template variables.
        """
        from pydantic import BaseModel  # deferred import to keep pydantic optional

        if not isinstance(params, BaseModel):
            raise TypeError(
                f"params must be a pydantic BaseModel, got {type(params).__name__}"
            )
        return self.render_template(template_name, frozendict(**params.model_dump()))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def get_template_variables(self, template_name: str) -> set[str]:
        """Return all undeclared variables required to render a template.

        Recursively follows ``{% extends %}`` and ``{% include %}`` directives
        so variables from parent/partial templates are included.

        Parameters
        ----------
        template_name : str
            Relative path to the Jinja template (e.g. ``"eval/user.jinja"``).
        """
        visited: set[str] = set()
        variables: set[str] = set()

        def _collect(name: str) -> None:
            if name in visited:
                return
            visited.add(name)
            try:
                source = self._env.loader.get_source(self._env, name)[0]  # pyrefly: ignore
            except TemplateNotFound:
                return
            ast = self._env.parse(source)
            variables.update(jinja2_meta.find_undeclared_variables(ast))
            for ref in jinja2_meta.find_referenced_templates(ast):
                if ref is not None:
                    _collect(ref)

        _collect(template_name)
        return variables

    def list_templates(self) -> list[str]:
        """Return a sorted list of all available template paths."""
        return sorted(self._env.loader.list_templates())  # pyrefly: ignore

    @property
    def prompt_dir(self) -> Path:
        return self._prompt_dir

    def __repr__(self) -> str:
        return f"PromptFactory(prompt_dir={self._prompt_dir!r})"


@cache
def get_cached_prompt_factory(folder: str | Path | None = None) -> PromptFactory:
    return PromptFactory(folder)
