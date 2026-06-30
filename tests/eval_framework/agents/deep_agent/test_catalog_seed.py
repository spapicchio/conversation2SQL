from pathlib import Path

import pytest

from conversation2sql.eval_framework.agents.deep_agent import catalog_seed


def _make_catalog(root: Path, db: str) -> None:
    tables = root / db / "tables"
    tables.mkdir(parents=True)
    (tables / "users.md").write_text("# users\n")
    (tables / "_foreign_key_constraints.md").write_text("no fks\n")
    (root / db / "database_overview.md").write_text("overview\n")


def test_materialize_copies_tables_overview_and_renders_masked_kb(task_data, tmp_path):
    _make_catalog(tmp_path, "mydb")
    task_data.deep_catalog_root = str(tmp_path)
    task_data.selected_database = "mydb"

    out = catalog_seed.materialize_catalog_dir(task_data)

    assert (out / "tables" / "users.md").read_text() == "# users\n"
    assert (out / "tables" / "_foreign_key_constraints.md").exists()
    assert (out / "database_overview.md").read_text() == "overview\n"
    # one KB file per surviving (masked) node, none for absent names
    assert (out / "knowledge_base" / "active_user.md").exists()
    assert (out / "knowledge_base" / "revenue.md").exists()
    assert not (out / "knowledge_base" / "secret_kb.md").exists()


def test_materialize_missing_tables_dir_raises(task_data, tmp_path):
    (tmp_path / "mydb").mkdir()  # db dir exists but no tables/
    task_data.deep_catalog_root = str(tmp_path)
    task_data.selected_database = "mydb"
    with pytest.raises(FileNotFoundError):
        catalog_seed.materialize_catalog_dir(task_data)


def test_materialize_skips_overview_when_absent(task_data, tmp_path):
    tables = tmp_path / "mydb" / "tables"
    tables.mkdir(parents=True)
    (tables / "users.md").write_text("# users\n")
    task_data.deep_catalog_root = str(tmp_path)
    task_data.selected_database = "mydb"
    out = catalog_seed.materialize_catalog_dir(task_data)
    assert not (out / "database_overview.md").exists()
    assert (out / "tables" / "users.md").exists()


def test_deep_tool_costs_three_tools():
    costs = catalog_seed.deep_tool_costs()
    assert set(costs) == {"bash", "submit_sql", "ask_user"}
    assert costs["bash"] == 1.0
