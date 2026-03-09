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
"""
from __future__ import annotations

import argparse
import os
import sys
import types
from argparse import ArgumentDefaultsHelpFormatter, ArgumentTypeError
from pathlib import Path
from typing import Any, Union, get_args, get_origin

import yaml
from pydantic import BaseModel
from pydantic.fields import FieldInfo

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
    if field_info.default is not None and field_info.default is not ...:
        return field_info.default
    if field_info.default_factory is not None:  # type: ignore[misc]
        return field_info.default_factory()  # type: ignore[misc]
    return _SENTINEL


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
        for model_type in self.model_types:
            self._register_model(model_type)

    # ------------------------------------------------------------------
    # Argument registration
    # ------------------------------------------------------------------

    def _register_model(self, model_type: type[BaseModel]) -> None:
        group = self._parser.add_argument_group(model_type.__name__)
        for field_name, field_info in model_type.model_fields.items():
            self._register_field(group, field_name, field_info)

    def _register_field(
            self,
            group: argparse._ArgumentGroup,
            name: str,
            field_info: FieldInfo,
    ) -> None:
        annotation = field_info.annotation
        default = _field_default(field_info)
        description = field_info.description or ""

        long_opts = [f"--{name}"]
        if "_" in name:
            long_opts.append(f"--{name.replace('_', '-')}")

        kwargs: dict[str, Any] = {
            "dest": name,
            "help": description,
            # We default everything to _SENTINEL so we can detect "was this
            # actually supplied on the CLI?" when merging priorities.
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

    def _read_env(self) -> dict[str, str]:
        """Return {field_name: raw_string} from environment variables."""
        result: dict[str, str] = {}
        all_fields = self._all_field_names()
        for field_name in all_fields:
            env_key = f"{self.env_prefix}{field_name.upper()}"
            value = os.environ.get(env_key)
            if value is not None:
                result[field_name] = value
        return result

    def _read_yaml(self, path: str) -> dict[str, Any]:
        content = yaml.safe_load(Path(path).read_text())
        if not isinstance(content, dict):
            raise ValueError(f"YAML config at '{path}' must be a mapping, got {type(content)}.")
        # Flatten nested sections: if a value is a dict and its key is not a
        # known field name, treat it as a named section and merge its contents.
        known = set(self._all_field_names())
        flat: dict[str, Any] = {}
        for k, v in content.items():
            if isinstance(v, dict) and k not in known:
                flat.update(v)
            else:
                flat[k] = v
        return flat

    def _all_field_names(self) -> list[str]:
        names = []
        for model_type in self.model_types:
            names.extend(model_type.model_fields.keys())
        return names

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

    def _build_models(self, merged: dict[str, Any]) -> tuple[BaseModel, ...]:
        outputs = []
        for model_type in self.model_types:
            keys = set(model_type.model_fields.keys())
            inputs = {k: v for k, v in merged.items() if k in keys}
            outputs.append(model_type(**inputs))
        return tuple(outputs)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse_args_and_config(
            self,
            args: list[str] | None = None,
    ) -> tuple[BaseModel, ...]:
        """
        Parse and merge configuration from env → yaml → CLI.

        Returns a tuple of instantiated Pydantic models in the same order as
        ``model_types``.
        """
        raw_args = list(args) if args is not None else sys.argv[1:]

        # --- Step 1: start with model defaults ----------------------------
        merged: dict[str, Any] = {}
        for model_type in self.model_types:
            for field_name, field_info in model_type.model_fields.items():
                default = _field_default(field_info)
                if default is not _SENTINEL:
                    merged[field_name] = default

        # --- Step 2: env vars (override defaults) -------------------------
        for field_name, raw_value in self._read_env().items():
            for model_type in self.model_types:
                if field_name in model_type.model_fields:
                    merged[field_name] = self._cast_raw_value(
                        raw_value, model_type.model_fields[field_name]
                    )
                    break

        # --- Step 3: YAML file (override env) -----------------------------
        # Extract --config from raw_args without consuming anything else yet.
        yaml_values: dict[str, Any] = {}
        tmp_args = list(raw_args)
        if "--config" in tmp_args:
            idx = tmp_args.index("--config")
            tmp_args.pop(idx)
            config_path = tmp_args.pop(idx)
            yaml_values = self._read_yaml(config_path)
            # Remove --config from the args that argparse will see
            raw_args = tmp_args

        for k, v in yaml_values.items():
            if k in self._all_field_names():
                merged[k] = v

        # --- Step 4: CLI args (highest priority) --------------------------
        namespace = self._parser.parse_args(raw_args)
        cli_dict = vars(namespace)
        cli_dict.pop("config", None)  # already handled above

        for k, v in cli_dict.items():
            if v is not _SENTINEL:
                merged[k] = v

        # --- Step 5: build models -----------------------------------------
        return self._build_models(merged)
