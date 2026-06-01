import pytest

from conversation2sql.presets import (
    baseline_uses_tools,
    expand_presets,
    resolve_effective_thinking,
    resolve_profile,
    resolve_server_args,
    resolve_variant,
)


def test_resolve_variant_flags():
    assert resolve_variant("gt_db_gt_kb_linearized") == [
        "--database_schema_type", "ddl",
        "--read_only_gt_tables", "true",
        "--read_only_gt_kb", "true",
        "--is_kb_linearized", "true",
    ]


def test_resolve_variant_toon():
    flags = resolve_variant("all_db_toon_all_kb")
    assert "--database_schema_type" in flags
    assert flags[flags.index("--database_schema_type") + 1] == "toon"
    assert flags[flags.index("--read_only_gt_tables") + 1] == "false"


def test_resolve_variant_unknown_lists_valid_keys():
    with pytest.raises(ValueError) as exc:
        resolve_variant("nope")
    assert "all_db_all_kb" in str(exc.value)


def test_resolve_profile_qwen_thinking():
    flags = resolve_profile("qwen35", enable_thinking=True)
    assert flags[flags.index("--predictor_model_name") + 1] == "Qwen/Qwen3.5-9B"
    assert flags[flags.index("--predictor_temperature") + 1] == "0.6"
    assert flags[flags.index("--predictor_presence_penalty") + 1] == "0.0"


def test_resolve_profile_qwen_non_thinking():
    flags = resolve_profile("qwen35", enable_thinking=False)
    assert flags[flags.index("--predictor_temperature") + 1] == "1.0"
    assert flags[flags.index("--predictor_presence_penalty") + 1] == "1.5"


def test_resolve_profile_default_thinking_per_model():
    qwen = resolve_profile("qwen35", enable_thinking=None)
    assert qwen[qwen.index("--predictor_temperature") + 1] == "0.6"
    gemma = resolve_profile("gemma4", enable_thinking=None)
    assert gemma[gemma.index("--predictor_top_k") + 1] == "64"


def test_resolve_profile_unknown_lists_valid_keys():
    with pytest.raises(ValueError) as exc:
        resolve_profile("nope", enable_thinking=True)
    assert "qwen35" in str(exc.value)


def test_expand_presets_composes_profile_then_variant():
    flags = expand_presets("qwen35", "all_db_all_kb", enable_thinking=True)
    assert flags.index("--predictor_model_name") < flags.index("--database_schema_type")


def test_expand_presets_allows_none_selectors():
    assert expand_presets(None, "all_db_all_kb", enable_thinking=None) == resolve_variant(
        "all_db_all_kb"
    )
    assert expand_presets(None, None, enable_thinking=None) == []


# --- vLLM server config ----------------------------------------------------


def test_resolve_effective_thinking_uses_profile_default():
    assert resolve_effective_thinking("qwen35", None) is True
    assert resolve_effective_thinking("gemma4", None) is False
    assert resolve_effective_thinking("qwen35", False) is False


def test_resolve_server_args_qwen_thinking():
    args = resolve_server_args("qwen35", enable_thinking=True, tp=1, dp=1, base_work="/bw")
    assert args == [
        "--tensor-parallel-size", "1",
        "--data-parallel-size", "1",
        "--reasoning-parser", "qwen3",
        "--default-chat-template-kwargs", '{"enable_thinking": true}',
        "--language-model-only",
    ]


def test_resolve_server_args_qwen_non_thinking_flips_json():
    args = resolve_server_args("qwen35", enable_thinking=False, tp=2, dp=1, base_work="/bw")
    assert args[args.index("--tensor-parallel-size") + 1] == "2"
    assert args[args.index("--default-chat-template-kwargs") + 1] == '{"enable_thinking": false}'


def test_resolve_server_args_gemma_joins_chat_template_and_limits_mm():
    args = resolve_server_args("gemma4", enable_thinking=None, tp=1, dp=1, base_work="/bw")
    assert args[args.index("--chat-template") + 1] == (
        "/bw/bash_scripts/utils/tool_chat_template_gemma4.jinja"
    )
    assert args[args.index("--limit-mm-per-prompt") + 1] == '{"image": 0, "audio": 0}'
    assert "--language-model-only" not in args


def test_resolve_server_args_unknown_profile_lists_valid_keys():
    with pytest.raises(ValueError) as exc:
        resolve_server_args("nope", enable_thinking=None, tp=1, dp=1, base_work="/bw")
    assert "qwen35" in str(exc.value)


def test_baseline_uses_tools():
    assert baseline_uses_tools("no_tool") is False
    assert baseline_uses_tools("tools_only") is True
    assert baseline_uses_tools("tools_user") is True
    assert baseline_uses_tools("bird_full") is True


def test_resolve_server_args_qwen_tool_baseline_adds_tool_calling_flags():
    args = resolve_server_args(
        "qwen35", enable_thinking=True, tp=1, dp=1, base_work="/bw", baseline="tools_user"
    )
    assert args[-3:] == ["--enable-auto-tool-choice", "--tool-call-parser", "qwen3_coder"]


def test_resolve_server_args_qwen_no_tool_omits_tool_calling_flags():
    args = resolve_server_args(
        "qwen35", enable_thinking=True, tp=1, dp=1, base_work="/bw", baseline="no_tool"
    )
    assert "--enable-auto-tool-choice" not in args
    assert "--tool-call-parser" not in args


def test_resolve_server_args_gemma_tool_baseline_has_no_parser_so_no_flags():
    # gemma4 declares no tool_call_parser, so tool baselines don't add flags.
    args = resolve_server_args(
        "gemma4", enable_thinking=None, tp=1, dp=1, base_work="/bw", baseline="bird_full"
    )
    assert "--enable-auto-tool-choice" not in args


# --- profile_for_model_name ------------------------------------------------


def test_profile_for_model_name_roundtrips_all_profiles():
    from conversation2sql.presets import MODEL_PROFILES, profile_for_model_name

    for name, prof in MODEL_PROFILES.items():
        assert profile_for_model_name(prof["predictor_model_name"]) == name


def test_profile_for_model_name_unknown_raises():
    import pytest
    from conversation2sql.presets import profile_for_model_name

    with pytest.raises(ValueError):
        profile_for_model_name("no/such-model")
