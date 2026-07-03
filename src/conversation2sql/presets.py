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
import logging
import os
import sys

# NOTE: keep this module free of third-party / intra-package imports so the bash
# side can run it standalone with the system ``python3`` (no venv/`uv`). Use the
# stdlib logger here rather than ``conversation2sql.logger`` (which pulls in
# loguru and requires the package to be importable).
_logger = logging.getLogger(__name__)

# Fraction of a profile's `max_model_len` reserved for the model's completion.
# The predictor's `max_new_tokens` (the request's `max_completion_tokens`) is
# derived from this in `resolve_profile`, so `max_model_len` stays the single
# knob: vLLM rejects any request whose `max_completion_tokens > max_model_len`.
#  max_completion_tokens = max_model_len * _COMPLETION_RATIO
# A profile may override this with its own `completion_ratio`.
_COMPLETION_RATIO = (
    0.25  # consider that the promp is the bottleneck and reaches ~24k for long runs
)

# --- Model profiles --------------------------------------------------------
# Sampling params differ by thinking mode for qwen; gemma is identical either way.
# `default_thinking` is used when the caller does not pass enable_thinking.
# `max_model_len` and `server` drive the vLLM server launch (see resolve_server_args).
# `max_model_len` also drives the predictor's `max_new_tokens` (see _COMPLETION_RATIO).
MODEL_PROFILES: dict[str, dict] = {
    "qwen35": {
        # https://huggingface.co/Qwen/Qwen3.5-9B
        "predictor_model_name": "Qwen/Qwen3.5-9B",
        "default_thinking": True,
        "max_model_len": 64_000,  # Total context, PROMPT + comp;
        "server": {
            "reasoning_parser": "qwen3",
            "language_model_only": True,
            # Patched copy of the model's default chat template. The stock template
            # injects an empty <think></think> into historical assistant turns once
            # reasoning is stripped from history, which makes the model stop closing
            # </think> on the current turn and bury its tool call inside the open
            # think block (qwen3 reasoning parser swallows it → finish_reason=stop,
            # no tool call). The patched template only emits the think wrapper when
            # reasoning_content is non-empty, so multi-turn tool calling works in
            # thinking mode. Path is relative to BASE_WORK; resolve_server_args joins it.
            "chat_template": "bash_scripts/utils/tool_chat_template_qwen35.jinja",
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
    "gemma4-12B": {
        # https://huggingface.co/google/gemma-4-12B-it
        "predictor_model_name": "google/gemma-4-12B-it",
        "default_thinking": True,
        "max_model_len": 64_000,  # Total context, PROMPT + comp;
        # https://docs.vllm.ai/projects/recipes/en/latest/Google/Gemma4.html
        "server": {
            "reasoning_parser": "gemma4",
            # Path is relative to BASE_WORK; resolve_server_args joins it.
            "chat_template": "bash_scripts/utils/tool_chat_template_gemma4.jinja",
            "limit_mm_per_prompt": {"image": 0, "audio": 0},
            # Tool calling is only wired up for the tool baselines; see
            # _TOOL_BASELINES / resolve_server_args.
            "tool_call_parser": "gemma4",
        },
        "thinking": {
            "predictor_temperature": "1.0",
            "predictor_top_p": "0.95",
            "predictor_top_k": "64",
        },
    },
    "gemma4-26B-A4B": {
        # https://huggingface.co/google/gemma-4-26B-A4B-it
        # MoE variant: ~26B total params, ~4B active. Same gemma4 server family
        # (reasoning/tool parsers, chat template) and sampling as gemma4-12B.
        "predictor_model_name": "google/gemma-4-26B-A4B-it",
        "default_thinking": True,
        "max_model_len": 32000,
        # https://docs.vllm.ai/projects/recipes/en/latest/Google/Gemma4.html
        "server": {
            "reasoning_parser": "gemma4",
            # Path is relative to BASE_WORK; resolve_server_args joins it.
            "chat_template": "bash_scripts/utils/tool_chat_template_gemma4.jinja",
            "limit_mm_per_prompt": {"image": 0, "audio": 0},
            # Tool calling is only wired up for the tool baselines; see
            # _TOOL_BASELINES / resolve_server_args.
            "tool_call_parser": "gemma4",
        },
        "thinking": {
            "predictor_temperature": "1.0",
            "predictor_top_p": "0.95",
            "predictor_top_k": "64",
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


def resolve_profile(
    name: str, enable_thinking: bool | None, baseline: str = "no_tool"
) -> list[str]:
    """Return CLI flags for a model profile's predictor sampling params.

    The thinking mode is resolved via ``resolve_effective_thinking`` (explicit
    flag wins; tool baselines default to non-thinking). The resolved value is
    emitted as ``--predictor_enable_thinking`` so the predictor config stays in
    sync with the sampling block — both come from the same decision here.
    """
    if name not in MODEL_PROFILES:
        raise ValueError(
            f"Unknown model profile {name!r}; valid: {', '.join(sorted(MODEL_PROFILES))}"
        )
    prof = MODEL_PROFILES[name]
    think = resolve_effective_thinking(name, enable_thinking, baseline)
    sampling = prof["thinking" if think else "non_thinking"]
    # Derive the completion budget from the server context length so the two
    # stay in sync from a single edit to `max_model_len`.
    ratio = prof.get("completion_ratio", _COMPLETION_RATIO)
    max_new_tokens = round(prof["max_model_len"] * ratio)
    # CHECK WHEN LAUNCHING: `max_new_tokens` only actually bounds generation on
    # the hosted-provider path (it is sent as `max_completion_tokens`). On the
    # local vLLM path (`--predictor_vllm_api_base` set, i.e. every `just eval`
    # run) it is *not* enforced: `utils_create_model` omits it and vLLM lets the
    # model fill `max_model_len - prompt_tokens`. So the effective completion
    # budget is dynamic, not this number. See agents/utils.py:create model.
    _logger.warning(
        "Profile %r: derived max_new_tokens=%d (= max_model_len %d * ratio %.2f). "
        "This bounds generation ONLY on hosted providers; on the local vLLM path "
        "generation is capped dynamically at max_model_len - prompt_tokens.",
        name,
        max_new_tokens,
        prof["max_model_len"],
        ratio,
    )
    flags = [
        "--predictor_model_name",
        prof["predictor_model_name"],
        "--predictor_enable_thinking",
        "true" if think else "false",
        "--predictor_max_new_tokens",
        str(max_new_tokens),
    ]
    flags.extend(_dict_to_flags(sampling))
    return flags


def expand_presets(
    model_profile: str | None,
    variant: str | None,
    enable_thinking: bool | None,
    baseline: str = "no_tool",
) -> list[str]:
    """Compose profile + variant flags. Either selector may be None."""
    flags: list[str] = []
    if model_profile is not None:
        flags.extend(resolve_profile(model_profile, enable_thinking, baseline))
    if variant is not None:
        flags.extend(resolve_variant(variant))
    return flags


# --- vLLM server launch config ---------------------------------------------
# Used by bash_scripts/eval_payload.sh via the `server-config` CLI below.

# Baselines that exercise the agent's tool calling (everything but `no_tool`).
# For these the server must be told to parse tool calls (vLLM disables tool
# calling by default). See justfile's baseline table.
_TOOL_BASELINES = frozenset(
    {"tools_only", "tools_user", "bird_full", "deep_agent", "maintenance_agent"}
)


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


def resolve_effective_thinking(
    name: str, enable_thinking: bool | None, baseline: str = "no_tool"
) -> bool:
    """Return the thinking mode for a profile.

    An explicit ``enable_thinking`` always wins. When it is None we fall back to
    a default: tool baselines default to **non-thinking**, every other baseline
    uses the profile's ``default_thinking``.
    """
    prof = _require_profile(name)
    if enable_thinking is not None:
        return enable_thinking
    return prof["default_thinking"]


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
    # TODO: this function may be too model specific
    prof = _require_profile(name)
    server = prof["server"]
    think = resolve_effective_thinking(name, enable_thinking)

    args: list[str] = [
        "--tensor-parallel-size",
        str(tp),
        "--data-parallel-size",
        str(dp),
        "--reasoning-parser",
        server["reasoning_parser"],
    ]
    if "chat_template" in server:
        args += ["--chat-template", os.path.join(base_work, server["chat_template"])]

    if "qwen" in name.lower() and not think:
        # for Qwen profiles, the defualt chat template must be add only when False
        args += [
            "--default-chat-template-kwargs",
            json.dumps({"enable_thinking": think}),
        ]

    if server.get("language_model_only"):
        args += ["--language-model-only"]

    if "limit_mm_per_prompt" in server:
        args += ["--limit-mm-per-prompt", json.dumps(server["limit_mm_per_prompt"])]

    if baseline_uses_tools(baseline) and "tool_call_parser" in server:
        args += [
            "--enable-auto-tool-choice",
            "--tool-call-parser",
            server["tool_call_parser"],
        ]

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
    think = resolve_effective_thinking(
        args.model_profile, _str_to_bool(args.enable_thinking), args.baseline
    )
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


def _cmd_variant_config(args: argparse.Namespace) -> None:
    """Emit NUL-delimited variant flag *values* (no flag names) in _VARIANT_FLAG_ORDER:

        database_schema_type \\0 read_only_gt_tables \\0 read_only_gt_kb \\0 is_kb_linearized

    Used by build_run_slug in utils_evaluate.sh so the run-directory slug reflects
    the variant. The schema flags travel to the Python pipeline as CLI args (via
    --variant), so the slug must read them from here too — presets.py stays the
    single source of truth rather than the bash side re-deriving them from env vars.
    """
    if args.variant not in VARIANTS:
        raise ValueError(
            f"Unknown variant {args.variant!r}; valid: {', '.join(sorted(VARIANTS))}"
        )
    values = VARIANTS[args.variant]
    sys.stdout.write("\0".join(values[key] for key in _VARIANT_FLAG_ORDER))


def _cmd_recover_config(args: argparse.Namespace) -> None:
    """Emit NUL-delimited fields for a resume launch, read from a run's snapshot:

        provider \\0 model_name \\0 max_model_len \\0 enable_thinking \\0 <vllm serve args...>

    provider comes first so bash decides whether to start a local vLLM server.
    yaml is imported lazily: this command runs under the venv (recover_payload.sh
    sources the venv first), while the module itself stays third-party-free so the
    system python3 can run `server-config`.
    """
    import yaml  # lazy: keep module import third-party-free for system python3

    config_path = os.path.join(args.run_dir, "config.yaml")
    with open(config_path, encoding="utf-8") as f:
        snapshot = yaml.safe_load(f)

    predictor = snapshot["predictor"]
    provider = predictor["model_provider"]
    model_name = predictor["model_name"]
    enable_thinking = predictor.get("enable_thinking")
    baseline = snapshot.get("pipeline", {}).get("baseline", "no_tool")

    profile = profile_for_model_name(model_name)
    think = resolve_effective_thinking(profile, enable_thinking, baseline)
    server_args = resolve_server_args(
        profile, think, args.tp, args.dp, args.base_work, baseline
    )
    prof = _require_profile(profile)
    fields = [
        provider,
        model_name,
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
        "--baseline",
        default="no_tool",
        help="eval baseline; tool baselines add the tool-calling serve flags.",
    )
    sc.add_argument(
        "--enable-thinking",
        default="",
        help="true/false; empty uses the profile's default_thinking.",
    )
    sc.add_argument("--tp", type=int, default=1, help="tensor-parallel-size")
    sc.add_argument("--dp", type=int, default=1, help="data-parallel-size")
    sc.add_argument("--base-work", default=os.environ.get("BASE_WORK", ""))
    sc.set_defaults(func=_cmd_server_config)

    vc = sub.add_parser(
        "variant-config",
        help="Emit NUL-delimited schema/gt/linearized values for a dataset variant.",
    )
    vc.add_argument("--variant", required=True)
    vc.set_defaults(func=_cmd_variant_config)

    rc = sub.add_parser(
        "recover-config",
        help="Emit provider + vllm serve config read from a run's config.yaml snapshot.",
    )
    rc.add_argument(
        "--run-dir", required=True, help="Run directory containing config.yaml."
    )
    rc.add_argument("--tp", type=int, default=1, help="tensor-parallel-size")
    rc.add_argument("--dp", type=int, default=1, help="data-parallel-size")
    rc.add_argument("--base-work", default=os.environ.get("BASE_WORK", ""))
    rc.set_defaults(func=_cmd_recover_config)

    args = parser.parse_args(argv)
    try:
        args.func(args)
    except ValueError as exc:  # unknown profile/variant — no traceback needed
        sys.stderr.write(f"presets: {exc}\n")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
