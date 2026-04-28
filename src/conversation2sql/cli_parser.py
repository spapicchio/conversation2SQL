"""
PydanticParser — argument parser for Pydantic BaseModel configs.

Priority (lowest → highest): model defaults → env vars → YAML → CLI

Usage example:

```python
from pydantic import BaseModel, Field
from conversation2sql.my_parser import PydanticParser

class ScriptArgs(BaseModel):
    output_dir: str | None = Field(default="./outputs", description="Output directory")
    execution_timeout: int = Field(default=500, description="SQL timeout in seconds")

class DataArgs(BaseModel):
    dataset_name: str = Field(description="Dataset to load")

parser = PydanticParser([ScriptArgs, DataArgs], env_prefix="MYAPP_")
script_args, data_args = parser.parse_args_and_config()
```

When two models share a field name (e.g. both have ``model_name``), the CLI
argument is automatically prefixed with the model's section name derived from
its class name (``ConfigPredictor`` → ``predictor_model_name``).  The YAML
file is parsed section-by-section so each model's values are kept separate.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import types
from argparse import ArgumentDefaultsHelpFormatter, ArgumentTypeError
from pathlib import Path
from typing import Any, Union, get_args, get_origin

import yaml
from pydantic import BaseModel
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SENTINEL = object()  # marks "not provided on CLI"


def _string_to_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise ArgumentTypeError(
        f"Boolean value expected: got '{v}'. Use yes/no, true/false, 1/0."
    )


def _unwrap_optional(annotation: Any) -> tuple[Any, bool]:
    """Return (inner_type, is_optional). Handles Optional[X] and X | None."""
    origin = get_origin(annotation)
    is_union = origin is Union or (
            hasattr(types, "UnionType") and isinstance(annotation, types.UnionType)
    )
    if is_union:
        args = [a for a in get_args(annotation) if a is not type(None)]
        is_optional = type(None) in get_args(annotation)
        inner = args[0] if len(args) == 1 else Union[tuple(args)]
        return inner, is_optional
    return annotation, False


def _argparse_type(annotation: Any):
    """Map a Python type annotation to an argparse ``type=`` callable."""
    inner, _ = _unwrap_optional(annotation)
    if inner is bool:
        return _string_to_bool
    origin = get_origin(inner)
    if origin is list:
        item_args = get_args(inner)
        return item_args[0] if item_args else str
    if inner in (int, float, str, Path):
        return inner
    return str


def _is_list_type(annotation: Any) -> bool:
    inner, _ = _unwrap_optional(annotation)
    return get_origin(inner) is list


def _field_default(field_info: FieldInfo) -> Any:
    """Return the Pydantic field default, or _SENTINEL if required."""
    if field_info.is_required() or field_info.default is PydanticUndefined:
        return _SENTINEL
    if field_info.default is not None and field_info.default is not ...:
        return field_info.default
    if field_info.default_factory is not None:  # type: ignore[misc]
        return field_info.default_factory()  # type: ignore[misc]
    return _SENTINEL


def _camel_to_snake(name: str) -> str:
    s1 = re.sub(r'(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


def _derive_section_name(model_type: type[BaseModel]) -> str:
    """Derive a YAML section name from a Config model class (ConfigFoo → foo)."""
    name = model_type.__name__
    if name.startswith("Config"):
        name = name[len("Config"):]
    return _camel_to_snake(name)


# ---------------------------------------------------------------------------
# Main parser
# ---------------------------------------------------------------------------

class PydanticParser:
    """
    Argument parser for one or more Pydantic ``BaseModel`` subclasses.

    Args:
        model_types: Ordered list of Pydantic model classes to parse.
        env_prefix:  Prefix for environment-variable look-up.
                     Field ``foo_bar`` is read from ``{env_prefix}FOO_BAR``.
    """

    def __init__(
            self,
            model_types: list[type[BaseModel]] | type[BaseModel],
            env_prefix: str = "",
    ) -> None:
        if not isinstance(model_types, list):
            model_types = [model_types]
        self.model_types = model_types
        self.env_prefix = env_prefix
        self._section_names = [_derive_section_name(m) for m in model_types]
        self._ambiguous = self._compute_ambiguous_fields()

        self._parser = argparse.ArgumentParser(
            formatter_class=ArgumentDefaultsHelpFormatter
        )
        self._parser.add_argument(
            "--config",
            type=str,
            default=None,
            metavar="PATH",
            help="Path to a YAML config file (overrides env vars, overridden by CLI args).",
        )
        for model_type, section_name in zip(self.model_types, self._section_names):
            self._register_model(model_type, section_name)

    def _compute_ambiguous_fields(self) -> set[str]:
        """Field names that appear in more than one model."""
        seen: dict[str, int] = {}
        for model_type in self.model_types:
            for field_name in model_type.model_fields:
                seen[field_name] = seen.get(field_name, 0) + 1
        return {name for name, count in seen.items() if count > 1}

    def _cli_key(self, section_name: str, field_name: str) -> str:
        """CLI dest key — prefixed with section name when the field is ambiguous."""
        if field_name in self._ambiguous:
            return f"{section_name}_{field_name}"
        return field_name

    # ------------------------------------------------------------------
    # Argument registration
    # ------------------------------------------------------------------

    def _register_model(self, model_type: type[BaseModel], section_name: str) -> None:
        group = self._parser.add_argument_group(model_type.__name__)
        for field_name, field_info in model_type.model_fields.items():
            self._register_field(group, section_name, field_name, field_info)

    def _register_field(
            self,
            group: argparse._ArgumentGroup,
            section_name: str,
            name: str,
            field_info: FieldInfo,
    ) -> None:
        annotation = field_info.annotation
        default = _field_default(field_info)
        description = field_info.description or ""

        cli_name = self._cli_key(section_name, name)
        long_opts = [f"--{cli_name}"]
        if "_" in cli_name:
            long_opts.append(f"--{cli_name.replace('_', '-')}")

        kwargs: dict[str, Any] = {
            "dest": cli_name,
            "help": description,
            # Default to _SENTINEL so we can detect "was this actually supplied?"
            "default": _SENTINEL,
        }

        inner, is_optional = _unwrap_optional(annotation)

        if inner is bool:
            kwargs["type"] = _string_to_bool
            kwargs["nargs"] = "?"
            kwargs["const"] = True
            if default is not _SENTINEL:
                kwargs["metavar"] = str(default)
        elif _is_list_type(annotation):
            item_type = get_args(get_origin(inner) and inner or inner)
            kwargs["type"] = item_type[0] if item_type else str
            kwargs["nargs"] = "+"
        else:
            kwargs["type"] = _argparse_type(annotation)

        if default is not _SENTINEL:
            kwargs["metavar"] = kwargs.get("metavar", str(default))

        group.add_argument(*long_opts, **kwargs)

    # ------------------------------------------------------------------
    # Source readers
    # ------------------------------------------------------------------

    def _read_env(self) -> dict[tuple[int, str], Any]:
        """Return {(model_index, field_name): value} from environment variables."""
        result: dict[tuple[int, str], Any] = {}
        for idx, (model_type, section_name) in enumerate(zip(self.model_types, self._section_names)):
            for field_name, field_info in model_type.model_fields.items():
                cli_name = self._cli_key(section_name, field_name)
                # Try section-prefixed env var first, then plain field name
                for env_key in [
                    f"{self.env_prefix}{cli_name.upper()}",
                    f"{self.env_prefix}{field_name.upper()}",
                ]:
                    value = os.environ.get(env_key)
                    if value is not None:
                        result[(idx, field_name)] = self._cast_raw_value(value, field_info)
                        break
        return result

    def _read_yaml(self, path: str) -> dict[tuple[int, str], Any]:
        """Return {(model_index, field_name): value} parsed section-by-section."""
        content = yaml.safe_load(Path(path).read_text())
        if not isinstance(content, dict):
            raise ValueError(f"YAML config at '{path}' must be a mapping, got {type(content)}.")

        section_to_idx = {s: i for i, s in enumerate(self._section_names)}
        result: dict[tuple[int, str], Any] = {}

        for k, v in content.items():
            if isinstance(v, dict) and k in section_to_idx:
                idx = section_to_idx[k]
                model_type = self.model_types[idx]
                for field_name, field_value in v.items():
                    if field_name in model_type.model_fields:
                        result[(idx, field_name)] = field_value
            elif not isinstance(v, dict):
                # Flat field: assign to the first model that owns it
                for idx, model_type in enumerate(self.model_types):
                    if k in model_type.model_fields:
                        result[(idx, k)] = v
                        break

        return result

    # ------------------------------------------------------------------
    # Merging & construction
    # ------------------------------------------------------------------

    def _cast_raw_value(self, raw: str, field_info: FieldInfo) -> Any:
        """Cast a raw string (from env) to the field's annotated type."""
        annotation = field_info.annotation
        inner, _ = _unwrap_optional(annotation)
        if inner is bool:
            return _string_to_bool(raw)
        if _is_list_type(annotation):
            item_args = get_args(inner)
            item_type = item_args[0] if item_args else str
            return [item_type(v.strip()) for v in raw.split(",")]
        try:
            argparse_t = _argparse_type(annotation)
            return argparse_t(raw)
        except (ValueError, TypeError):
            return raw

    def _build_models(self, merged: dict[tuple[int, str], Any]) -> tuple[BaseModel, ...]:
        outputs = []
        for idx, model_type in enumerate(self.model_types):
            inputs = {field_name: value for (i, field_name), value in merged.items() if i == idx}
            outputs.append(model_type(**inputs))
        return tuple(outputs)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse_args_and_config(
            self,
            args: list[str] | None = None,
    ) -> tuple[Any, ...]:
        """
        Parse and merge configuration from env → yaml → CLI.

        Returns a tuple of instantiated Pydantic models in the same order as
        ``model_types``.
        """
        raw_args = list(args) if args is not None else sys.argv[1:]

        # merged uses (model_index, field_name) keys to avoid cross-model collision
        merged: dict[tuple[int, str], Any] = {}

        # --- Step 1: model defaults ----------------------------
        for idx, model_type in enumerate(self.model_types):
            for field_name, field_info in model_type.model_fields.items():
                default = _field_default(field_info)
                if default is not _SENTINEL:
                    merged[(idx, field_name)] = default

        # --- Step 2: env vars (override defaults) -------------------------
        merged.update(self._read_env())

        # --- Step 3: YAML file (override env) -----------------------------
        tmp_args = list(raw_args)
        if "--config" in tmp_args:
            idx = tmp_args.index("--config")
            tmp_args.pop(idx)
            config_path = tmp_args.pop(idx)
            merged.update(self._read_yaml(config_path))
            raw_args = tmp_args

        # --- Step 4: CLI args (highest priority) --------------------------
        namespace = self._parser.parse_args(raw_args)
        cli_dict = vars(namespace)
        cli_dict.pop("config", None)

        for i, (model_type, section_name) in enumerate(zip(self.model_types, self._section_names)):
            for field_name in model_type.model_fields:
                cli_name = self._cli_key(section_name, field_name)
                v = cli_dict.get(cli_name, _SENTINEL)
                if v is not _SENTINEL:
                    merged[(i, field_name)] = v

        # --- Step 5: build models -----------------------------------------
        return self._build_models(merged)
