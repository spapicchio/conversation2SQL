"""Named presets for eval model profiles and dataset variants.

Each preset resolves to a list of CLI-flag strings keyed by the exact dest
names PydanticParser expects (predictor sampling fields are ambiguous across
ConfigPredictor/ConfigUserSimulator so they carry the `predictor_` prefix; the
reader schema flags are unique so they use bare names). Returning flags — not a
dict — lets cli.py splice presets straight into the argv it hands PydanticParser,
where the CLI layer wins over YAML.

This module is also the single source of truth for the vLLM **server** launch
config (model name, context length, `vllm serve` flags). `bash_scripts/
eval_payload.sh` shells out to the standalone ``server-config`` CLI below to get
those instead of duplicating a bash ``case``. It deliberately has no third-party
imports, so the bash side can run it with the system ``python3`` (no venv/`uv`).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# --- Model profiles --------------------------------------------------------
# Sampling params differ by thinking mode for qwen; gemma is identical either way.
# `default_thinking` is used when the caller does not pass enable_thinking.
# `max_model_len` and `server` drive the vLLM server launch (see resolve_server_args).
MODEL_PROFILES: dict[str, dict] = {
    "qwen35": {
        # https://huggingface.co/Qwen/Qwen3.5-9B
        "predictor_model_name": "Qwen/Qwen3.5-9B",
        "default_thinking": True,
        "max_model_len": 50000,
        "server": {
            "reasoning_parser": "qwen3",
            "language_model_only": True,
            # Tool calling is only wired up for the tool baselines; see
            # _TOOL_BASELINES / resolve_server_args.
            "tool_call_parser": "qwen3_coder",
        },
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
        # https://huggingface.co/google/gemma-4-26B-A4B-it
        "predictor_model_name": "google/gemma-4-26B-A4B-it",
        "default_thinking": False,
        "max_model_len": 32000,
        "server": {
            "reasoning_parser": "gemma4",
            # Path is relative to BASE_WORK; resolve_server_args joins it.
            "chat_template": "bash_scripts/utils/tool_chat_template_gemma4.jinja",
            "limit_mm_per_prompt": {"image": 0, "audio": 0},
        },
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
    "all_db_all_kb": {
        "database_schema_type": "ddl",
        "read_only_gt_tables": "false",
        "read_only_gt_kb": "false",
        "is_kb_linearized": "false",
    },
    "all_db_all_kb_linearized": {
        "database_schema_type": "ddl",
        "read_only_gt_tables": "false",
        "read_only_gt_kb": "false",
        "is_kb_linearized": "true",
    },
    "all_db_toon_all_kb": {
        "database_schema_type": "toon",
        "read_only_gt_tables": "false",
        "read_only_gt_kb": "false",
        "is_kb_linearized": "false",
    },
    "all_db_toon_all_kb_linearized": {
        "database_schema_type": "toon",
        "read_only_gt_tables": "false",
        "read_only_gt_kb": "false",
        "is_kb_linearized": "true",
    },
    "gt_db_all_kb_linearized": {
        "database_schema_type": "ddl",
        "read_only_gt_tables": "true",
        "read_only_gt_kb": "false",
        "is_kb_linearized": "true",
    },
    "gt_db_gt_kb_linearized": {
        "database_schema_type": "ddl",
        "read_only_gt_tables": "true",
        "read_only_gt_kb": "true",
        "is_kb_linearized": "true",
    },
    "gt_db_gt_kb": {
        "database_schema_type": "ddl",
        "read_only_gt_tables": "true",
        "read_only_gt_kb": "true",
        "is_kb_linearized": "false",
    },
}

# Order in which reader flags are emitted (stable output for tests/readability).
_VARIANT_FLAG_ORDER = (
    "database_schema_type",
    "read_only_gt_tables",
    "read_only_gt_kb",
    "is_kb_linearized",
)


def _dict_to_flags(
    values: dict[str, str], order: tuple[str, ...] | None = None
) -> list[str]:
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


# --- vLLM server launch config ---------------------------------------------
# Used by bash_scripts/eval_payload.sh via the `server-config` CLI below.

# Baselines that exercise the agent's tool calling (everything but `no_tool`).
# For these the server must be told to parse tool calls (vLLM disables tool
# calling by default). See justfile's baseline table.
_TOOL_BASELINES = frozenset({"tools_only", "tools_user", "bird_full"})


def baseline_uses_tools(baseline: str) -> bool:
    """True when the baseline drives the tool-calling agent (not `no_tool`)."""
    return baseline in _TOOL_BASELINES


def _require_profile(name: str) -> dict:
    if name not in MODEL_PROFILES:
        raise ValueError(
            f"Unknown model profile {name!r}; valid: {', '.join(sorted(MODEL_PROFILES))}"
        )
    return MODEL_PROFILES[name]


def profile_for_model_name(model_name: str) -> str:
    """Reverse-map a predictor model name to its profile key.

    Model names are unique across MODEL_PROFILES. Used by `recover-config` to
    relaunch the right vLLM server for a snapshot that only records the model
    name, not the profile.
    """
    for name, prof in MODEL_PROFILES.items():
        if prof["predictor_model_name"] == model_name:
            return name
    raise ValueError(
        f"No model profile for model_name {model_name!r}; "
        f"known: {', '.join(sorted(p['predictor_model_name'] for p in MODEL_PROFILES.values()))}"
    )


def resolve_effective_thinking(name: str, enable_thinking: bool | None) -> bool:
    """Return the thinking mode, falling back to the profile default when None."""
    prof = _require_profile(name)
    return prof["default_thinking"] if enable_thinking is None else enable_thinking


def resolve_server_args(
    name: str,
    enable_thinking: bool | None,
    tp: int,
    dp: int,
    base_work: str,
    baseline: str = "no_tool",
) -> list[str]:
    """Return the `vllm serve` flags for a model profile (excluding the model
    name and --max-model-len, which the caller passes separately).

    Flag order mirrors the previous hand-written bash `case` so existing launch
    commands are byte-for-byte unchanged. When ``baseline`` drives the
    tool-calling agent and the profile declares a ``tool_call_parser``, the
    tool-calling flags are appended.
    """
    prof = _require_profile(name)
    server = prof["server"]
    think = resolve_effective_thinking(name, enable_thinking)

    args: list[str] = [
        "--tensor-parallel-size", str(tp),
        "--data-parallel-size", str(dp),
        "--reasoning-parser", server["reasoning_parser"],
    ]
    if "chat_template" in server:
        args += ["--chat-template", os.path.join(base_work, server["chat_template"])]
    args += ["--default-chat-template-kwargs", json.dumps({"enable_thinking": think})]
    if server.get("language_model_only"):
        args += ["--language-model-only"]
    if "limit_mm_per_prompt" in server:
        args += ["--limit-mm-per-prompt", json.dumps(server["limit_mm_per_prompt"])]
    if baseline_uses_tools(baseline) and "tool_call_parser" in server:
        args += ["--enable-auto-tool-choice", "--tool-call-parser", server["tool_call_parser"]]
    return args


def _str_to_bool(value: str) -> bool | None:
    """Parse a thinking flag; empty string means 'use the profile default'."""
    if value.strip() == "":
        return None
    return value.strip().lower() in ("1", "true", "t", "yes", "y")


def _cmd_server_config(args: argparse.Namespace) -> None:
    """Emit NUL-delimited fields for bash `mapfile -d ''` consumption:

        model_name \\0 max_model_len \\0 enable_thinking \\0 <vllm serve args...>

    enable_thinking is the *resolved* value so bash forwards a concrete
    --predictor_enable_thinking that matches the server's chat-template kwargs.
    """
    prof = _require_profile(args.model_profile)
    think = resolve_effective_thinking(args.model_profile, _str_to_bool(args.enable_thinking))
    server_args = resolve_server_args(
        args.model_profile, think, args.tp, args.dp, args.base_work, args.baseline
    )
    fields = [
        prof["predictor_model_name"],
        str(prof["max_model_len"]),
        "true" if think else "false",
        *server_args,
    ]
    sys.stdout.write("\0".join(fields))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="presets", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sc = sub.add_parser(
        "server-config",
        help="Emit NUL-delimited model name, max-model-len, thinking, and vllm serve args.",
    )
    sc.add_argument("--model-profile", required=True)
    sc.add_argument(
        "--baseline", default="no_tool",
        help="eval baseline; tool baselines add the tool-calling serve flags.",
    )
    sc.add_argument(
        "--enable-thinking", default="",
        help="true/false; empty uses the profile's default_thinking.",
    )
    sc.add_argument("--tp", type=int, default=1, help="tensor-parallel-size")
    sc.add_argument("--dp", type=int, default=1, help="data-parallel-size")
    sc.add_argument("--base-work", default=os.environ.get("BASE_WORK", ""))
    sc.set_defaults(func=_cmd_server_config)

    args = parser.parse_args(argv)
    try:
        args.func(args)
    except ValueError as exc:  # unknown profile/variant — no traceback needed
        sys.stderr.write(f"presets: {exc}\n")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
