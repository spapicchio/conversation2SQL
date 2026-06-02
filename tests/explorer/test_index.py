from pathlib import Path

import pytest

from explorer.index import render_args, COLUMNS


def _cfg(**over):
    cfg = {
        "pipeline": {"baseline": "no_tool", "num_iterations": 9},
        "reader": {
            "database_schema_type": "ddl",
            "is_kb_linearized": True,
            "read_only_gt_tables": True,
            "read_only_gt_kb": True,
            "make_data_ambiguous": False,
        },
        "predictor": {
            "model_name": "Qwen/Qwen3.5-9B",
            "model_provider": "hosted_vllm",
            "temperature": 0.6,
            "top_p": 0.95,
            "enable_thinking": True,
        },
        "user_simulator": {"model_name": "gpt-5.4-mini-2026-03-17"},
    }
    for section, vals in over.items():
        cfg[section] = {**cfg[section], **vals}
    return cfg


class TestRenderArgs:
    def test_full_string(self):
        assert render_args(_cfg()) == (
            "--schema-type ddl --kb-linearized --gt-db --gt-kb "
            "--num-iterations 9 --temperature 0.6 --top-p 0.95 --thinking "
            "--provider hosted_vllm --user-sim gpt-5.4-mini-2026-03-17"
        )

    def test_false_bools_omitted(self):
        s = render_args(_cfg(reader={"read_only_gt_kb": False}, predictor={"enable_thinking": False}))
        assert "--gt-kb" not in s
        assert "--thinking" not in s
        assert "--gt-db" in s

    def test_one_token_difference(self):
        a = render_args(_cfg())
        b = render_args(_cfg(reader={"read_only_gt_kb": False}))
        assert a.replace(" --gt-kb", "") == b

    def test_ambiguous_flag_appears(self):
        assert "--ambiguous" in render_args(_cfg(reader={"make_data_ambiguous": True}))

    def test_columns_constant_order(self):
        assert COLUMNS[:7] == ["run_dir", "date", "time", "status", "baseline", "model", "args"]
        assert COLUMNS[-1] == "Notes"
