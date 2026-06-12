from __future__ import annotations

from explorer.charts import conversation_length_box
from explorer.charts import disambiguate_run_labels
from explorer.colors import stable_color


def test_unique_slugs_are_shortened():
    labels = [
        "2026-06-01 / 10-00-00/tools_only__qwen__ddl",
        "2026-06-02 / 11-00-00/bird_full__gemma__ddl",
    ]
    out = disambiguate_run_labels(labels)
    assert out[labels[0]] == "tools_only__qwen__ddl"
    assert out[labels[1]] == "bird_full__gemma__ddl"


def test_colliding_slugs_keep_full_labels():
    # The same variant slug run on two dates is the most common A/B comparison;
    # shortening both to the slug would silently merge them into one box.
    labels = [
        "2026-06-01 / 10-00-00/tools_only__qwen__ddl",
        "2026-06-02 / 11-00-00/tools_only__qwen__ddl",
    ]
    out = disambiguate_run_labels(labels)
    assert out[labels[0]] == labels[0]
    assert out[labels[1]] == labels[1]
    assert len(set(out.values())) == 2


def test_old_flat_layout_label_shortens_to_its_leaf():
    labels = ["2026-05-14 / 09_17_54__qwen-ddl"]
    out = disambiguate_run_labels(labels)
    assert out[labels[0]] == "09_17_54__qwen-ddl"


def test_conversation_length_box_one_trace_per_nonempty_series():
    fig = conversation_length_box(
        {"run_a": [3.0, 5.0, 8.0], "run_b": [2.0, 4.0]}, "Model calls"
    )
    assert len(fig.data) == 2
    assert [t.name for t in fig.data] == ["run_a", "run_b"]
    # Raw values are handed to go.Box (Plotly computes the quartiles).
    assert list(fig.data[0].y) == [3.0, 5.0, 8.0]


def test_conversation_length_box_skips_empty_series():
    # A no-tool baseline contributes no budget-spent values -> no box for it.
    fig = conversation_length_box(
        {"run_a": [3.0, 5.0], "no_tool": []}, "Budget spent"
    )
    assert [t.name for t in fig.data] == ["run_a"]


def test_outcome_color_passed_label():
    assert stable_color("Passed", kind="outcome") == "#2ca02c"


def test_outcome_color_checkmark_label():
    # compare.py uses "runA ✓" — the ✓ suffix triggers green
    assert stable_color("runA ✓", kind="outcome") == "#2ca02c"


def test_outcome_color_failed_label():
    assert stable_color("Failed", kind="outcome") == "#d62728"


def test_outcome_color_cross_label():
    assert stable_color("runA ✗", kind="outcome") == "#d62728"


def test_outcome_color_unknown_falls_back_to_hash():
    # Any label that isn't pass/fail should still return *something* (not raise).
    color = stable_color("some other label", kind="outcome")
    assert color.startswith("#") and len(color) == 7
