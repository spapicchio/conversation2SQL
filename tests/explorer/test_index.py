import csv
import json
from pathlib import Path

import yaml

from explorer.index import render_args, COLUMNS, derive_status, append_stub, reconcile, _parse_time


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

    def test_none_top_p_renders_empty_not_None_string(self):
        # predictor top_p=None must not produce the literal string "None"
        cfg = _cfg(predictor={"top_p": None})
        result = render_args(cfg)
        assert "--top-p None" not in result
        assert "--top-p " in result  # flag is present with empty value

    def test_zero_temperature_renders_zero_not_empty(self):
        # temperature=0.0 is valid; must not be blanked by `x or ""`
        cfg = _cfg(predictor={"temperature": 0.0})
        result = render_args(cfg)
        assert "--temperature 0.0" in result


def _write_config(run_dir: Path, cfg: dict) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


class TestParseTime:
    def test_new_layout_slash(self):
        assert _parse_time("08-48-35/slug") == "08-48-35"

    def test_old_flat_layout_double_underscore(self):
        assert _parse_time("09_17_54__qwen-ddl") == "09_17_54"


class TestDeriveStatus:
    def test_error_when_dir_suffixed(self, tmp_path):
        d = tmp_path / "run__error"
        d.mkdir()
        assert derive_status(d, _cfg(), n_present=0) == "error"

    def test_error_when_error_jsonl_present(self, tmp_path):
        d = tmp_path / "run"
        d.mkdir()
        (d / "results_error.jsonl").write_text("{}\n")
        assert derive_status(d, _cfg(), n_present=3) == "error"

    def test_done_when_all_iterations_present(self, tmp_path):
        d = tmp_path / "run"
        d.mkdir()
        assert derive_status(d, _cfg(pipeline={"num_iterations": 3}), n_present=3) == "done"

    def test_partial_when_some_iterations(self, tmp_path):
        d = tmp_path / "run"
        d.mkdir()
        assert derive_status(d, _cfg(pipeline={"num_iterations": 9}), n_present=4) == "partial"

    def test_temp_zero_collapses_expected_to_one(self, tmp_path):
        d = tmp_path / "run"
        d.mkdir()
        cfg = _cfg(pipeline={"num_iterations": 9}, predictor={"temperature": 0.0})
        assert derive_status(d, cfg, n_present=1) == "done"


class TestAppendStub:
    def test_appends_running_row_with_config_columns(self, tmp_path):
        results = tmp_path / "results"
        run = results / "2026-06-02" / "08-48-35" / "no_tool__Qwen__ddl__iter9"
        _write_config(run, _cfg())
        csv_path = tmp_path / "experiments.csv"

        append_stub(run, csv_path=csv_path, results_root=results)

        rows = list(csv.DictReader(csv_path.open()))
        assert len(rows) == 1
        row = rows[0]
        assert row["run_dir"] == "2026-06-02/08-48-35/no_tool__Qwen__ddl__iter9"
        assert row["date"] == "2026-06-02"
        assert row["time"] == "08-48-35"
        assert row["status"] == "running"
        assert row["baseline"] == "no_tool"
        assert row["model"] == "Qwen/Qwen3.5-9B"
        assert "--schema-type ddl" in row["args"]
        assert row["accuracy"] == ""
        assert row["Notes"] == ""

    def test_idempotent(self, tmp_path):
        results = tmp_path / "results"
        run = results / "2026-06-02" / "08-48-35" / "slug"
        _write_config(run, _cfg())
        csv_path = tmp_path / "experiments.csv"

        append_stub(run, csv_path=csv_path, results_root=results)
        append_stub(run, csv_path=csv_path, results_root=results)

        rows = list(csv.DictReader(csv_path.open()))
        assert len(rows) == 1

    def test_no_config_yaml_writes_running_row(self, tmp_path):
        results = tmp_path / "results"
        run = results / "2026-06-02" / "10-00-00" / "no_config_run"
        run.mkdir(parents=True, exist_ok=True)
        csv_path = tmp_path / "experiments.csv"

        append_stub(run, csv_path=csv_path, results_root=results)

        rows = list(csv.DictReader(csv_path.open()))
        assert len(rows) == 1
        assert rows[0]["status"] == "running"
        assert rows[0]["baseline"] == ""


def _make_record(execution_accuracy=False, instance_id="t1", database="mydb"):
    return {
        "execution_accuracy": execution_accuracy,
        "instance_id": instance_id,
        "selected_database": database,
        "mean_prompt_tokens": 100,
        "mean_completion_tokens": 50,
        "total_cost": 0.01,
        "updated_user_patience": 3,
        "tool_calls_in_order": [],
        "messages": [],
        "iteration": 0,
    }


def _make_run(results: Path, date: str, time: str, slug: str, cfg: dict, records: list[dict]) -> Path:
    run = results / date / time / slug
    _write_config(run, cfg)
    (run / "results_iter0.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in records), encoding="utf-8"
    )
    return run


class TestReconcile:
    def test_builds_row_with_metrics(self, tmp_path):
        results = tmp_path / "results"
        _make_run(
            results, "2026-06-02", "08-48-35", "no_tool__Qwen__ddl__iter1",
            _cfg(pipeline={"num_iterations": 1}, predictor={"temperature": 0.0}),
            [_make_record(execution_accuracy=True), _make_record(execution_accuracy=False, instance_id="t2")],
        )
        csv_path = tmp_path / "experiments.csv"

        reconcile(results_root=results, csv_path=csv_path)

        rows = list(csv.DictReader(csv_path.open()))
        assert len(rows) == 1
        row = rows[0]
        assert row["run_dir"] == "2026-06-02/08-48-35/no_tool__Qwen__ddl__iter1"
        assert row["status"] == "done"
        assert row["n_instances"] == "2"
        assert row["accuracy"] == "0.5"
        assert "--schema-type ddl" in row["args"]

    def test_preserves_notes_across_reconcile(self, tmp_path):
        results = tmp_path / "results"
        _make_run(
            results, "2026-06-02", "08-48-35", "slug",
            _cfg(pipeline={"num_iterations": 1}, predictor={"temperature": 0.0}),
            [_make_record(execution_accuracy=True)],
        )
        csv_path = tmp_path / "experiments.csv"
        reconcile(results_root=results, csv_path=csv_path)

        rows = list(csv.DictReader(csv_path.open()))
        rows[0]["Notes"] = "best ablation so far"
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

        reconcile(results_root=results, csv_path=csv_path)

        out = list(csv.DictReader(csv_path.open()))
        assert out[0]["Notes"] == "best ablation so far"

    def test_keeps_running_stub_not_rediscovered(self, tmp_path):
        results = tmp_path / "results"
        run = results / "2026-06-02" / "09-00-00" / "pending"
        _write_config(run, _cfg())
        csv_path = tmp_path / "experiments.csv"
        append_stub(run, csv_path=csv_path, results_root=results)
        _make_run(
            results, "2026-06-02", "08-00-00", "done_slug",
            _cfg(pipeline={"num_iterations": 1}, predictor={"temperature": 0.0}),
            [_make_record(execution_accuracy=True)],
        )

        reconcile(results_root=results, csv_path=csv_path)

        out = {r["run_dir"]: r for r in csv.DictReader(csv_path.open())}
        assert "2026-06-02/09-00-00/pending" in out
        assert out["2026-06-02/09-00-00/pending"]["status"] == "running"
        assert "2026-06-02/08-00-00/done_slug" in out


class TestCli:
    def test_reconcile_subcommand(self, tmp_path):
        results = tmp_path / "results"
        _make_run(
            results, "2026-06-02", "08-48-35", "slug",
            _cfg(pipeline={"num_iterations": 1}, predictor={"temperature": 0.0}),
            [_make_record(execution_accuracy=True)],
        )
        csv_path = tmp_path / "experiments.csv"
        from explorer.index import main
        main(["reconcile", "--results-root", str(results), "--csv", str(csv_path)])
        rows = list(csv.DictReader(csv_path.open()))
        assert rows and rows[0]["run_dir"] == "2026-06-02/08-48-35/slug"

    def test_append_subcommand(self, tmp_path):
        results = tmp_path / "results"
        run = results / "2026-06-02" / "08-48-35" / "slug"
        _write_config(run, _cfg())
        csv_path = tmp_path / "experiments.csv"
        from explorer.index import main
        main(["append", str(run), "--csv", str(csv_path), "--results-root", str(results)])
        rows = list(csv.DictReader(csv_path.open()))
        assert rows[0]["status"] == "running"
