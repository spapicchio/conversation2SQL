from __future__ import annotations

from explorer.charts import disambiguate_run_labels


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
