"""Named presets for eval model profiles and dataset variants.

Each preset resolves to a list of CLI-flag strings keyed by the exact dest
names PydanticParser expects (predictor sampling fields are ambiguous across
ConfigPredictor/ConfigUserSimulator so they carry the `predictor_` prefix; the
reader schema flags are unique so they use bare names). Returning flags — not a
dict — lets cli.py splice presets straight into the argv it hands PydanticParser,
where the CLI layer wins over YAML.
"""
from __future__ import annotations

# --- Model profiles --------------------------------------------------------
# Sampling params differ by thinking mode for qwen; gemma is identical either way.
# `default_thinking` is used when the caller does not pass enable_thinking.
MODEL_PROFILES: dict[str, dict] = {
    "qwen35": {
        "predictor_model_name": "Qwen/Qwen3.5-9B",
        "default_thinking": True,
        "thinking": {
            "predictor_temperature": "0.6",
            "predictor_top_p": "0.95",
            "predictor_top_k": "20",
            "predictor_presence_penalty": "0.0",
            "predictor_repetition_penalty": "1.0",
        },
        "non_thinking": {
            "predictor_temperature": "1.0",
            "predictor_top_p": "0.95",
            "predictor_top_k": "20",
            "predictor_presence_penalty": "1.5",
            "predictor_repetition_penalty": "1.0",
        },
    },
    "gemma4": {
        "predictor_model_name": "google/gemma-4-26B-A4B-it",
        "default_thinking": False,
        "thinking": {
            "predictor_temperature": "1.0",
            "predictor_top_p": "0.95",
            "predictor_top_k": "64",
            "predictor_presence_penalty": "0.0",
            "predictor_repetition_penalty": "1.0",
        },
        "non_thinking": {
            "predictor_temperature": "1.0",
            "predictor_top_p": "0.95",
            "predictor_top_k": "64",
            "predictor_presence_penalty": "0.0",
            "predictor_repetition_penalty": "1.0",
        },
    },
}

# --- Dataset variants ------------------------------------------------------
# Maps each variant to the 4 reader schema flags. Values are "true"/"false"
# strings so PydanticParser's bool coercion handles them.
VARIANTS: dict[str, dict[str, str]] = {
    "all_db_all_kb": {"database_schema_type": "ddl", "read_only_gt_tables": "false", "read_only_gt_kb": "false", "is_kb_linearized": "false"},
    "all_db_all_kb_linearized": {"database_schema_type": "ddl", "read_only_gt_tables": "false", "read_only_gt_kb": "false", "is_kb_linearized": "true"},
    "all_db_toon_all_kb": {"database_schema_type": "toon", "read_only_gt_tables": "false", "read_only_gt_kb": "false", "is_kb_linearized": "false"},
    "all_db_toon_all_kb_linearized": {"database_schema_type": "toon", "read_only_gt_tables": "false", "read_only_gt_kb": "false", "is_kb_linearized": "true"},
    "gt_db_all_kb_linearized": {"database_schema_type": "ddl", "read_only_gt_tables": "true", "read_only_gt_kb": "false", "is_kb_linearized": "true"},
    "gt_db_gt_kb_linearized": {"database_schema_type": "ddl", "read_only_gt_tables": "true", "read_only_gt_kb": "true", "is_kb_linearized": "true"},
    "gt_db_gt_kb": {"database_schema_type": "ddl", "read_only_gt_tables": "true", "read_only_gt_kb": "true", "is_kb_linearized": "false"},
}

# Order in which reader flags are emitted (stable output for tests/readability).
_VARIANT_FLAG_ORDER = (
    "database_schema_type",
    "read_only_gt_tables",
    "read_only_gt_kb",
    "is_kb_linearized",
)


def _dict_to_flags(values: dict[str, str], order: tuple[str, ...] | None = None) -> list[str]:
    keys = order if order is not None else tuple(values)
    flags: list[str] = []
    for key in keys:
        flags.extend([f"--{key}", values[key]])
    return flags


def resolve_variant(name: str) -> list[str]:
    """Return CLI flags for a dataset variant, e.g. ['--database_schema_type', 'ddl', ...]."""
    if name not in VARIANTS:
        raise ValueError(
            f"Unknown variant {name!r}; valid: {', '.join(sorted(VARIANTS))}"
        )
    return _dict_to_flags(VARIANTS[name], _VARIANT_FLAG_ORDER)


def resolve_profile(name: str, enable_thinking: bool | None) -> list[str]:
    """Return CLI flags for a model profile's predictor sampling params.

    When ``enable_thinking`` is None, the profile's ``default_thinking`` is used.
    """
    if name not in MODEL_PROFILES:
        raise ValueError(
            f"Unknown model profile {name!r}; valid: {', '.join(sorted(MODEL_PROFILES))}"
        )
    prof = MODEL_PROFILES[name]
    think = prof["default_thinking"] if enable_thinking is None else enable_thinking
    sampling = prof["thinking" if think else "non_thinking"]
    flags = ["--predictor_model_name", prof["predictor_model_name"]]
    flags.extend(_dict_to_flags(sampling))
    return flags


def expand_presets(
    model_profile: str | None,
    variant: str | None,
    enable_thinking: bool | None,
) -> list[str]:
    """Compose profile + variant flags. Either selector may be None."""
    flags: list[str] = []
    if model_profile is not None:
        flags.extend(resolve_profile(model_profile, enable_thinking))
    if variant is not None:
        flags.extend(resolve_variant(variant))
    return flags
