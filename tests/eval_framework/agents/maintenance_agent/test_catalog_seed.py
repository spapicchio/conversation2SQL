from pathlib import Path

import pytest

from conversation2sql.eval_framework.agents.maintenance_agent import catalog_seed


def _make_catalog(root: Path, db: str) -> None:
    tables = root / db / "tables"
    tables.mkdir(parents=True)
    (tables / "users.md").write_text("# users\n")
    (tables / "_foreign_key_constraints.md").write_text("no fks\n")
    (root / db / "database_overview.md").write_text("overview\n")


def test_materialize_nests_catalog_under_docs(task_data, tmp_path):
    _make_catalog(tmp_path, "mydb")
    task_data.deep_catalog_root = str(tmp_path)
    task_data.selected_database = "mydb"

    out = catalog_seed.materialize_maintenance_workspace(task_data)

    assert (out / "docs" / "database_overview.md").read_text().startswith("overview\n")
    assert (out / "docs" / "tables" / "users.md").read_text() == "# users\n"
    assert (out / "docs" / "knowledge_base" / "active_user.md").exists()
    # nothing left at the old flat deep_agent locations
    assert not (out / "database_overview.md").exists()
    assert not (out / "tables").exists()
    assert not (out / "knowledge_base").exists()


def test_materialize_skips_docs_overview_when_absent(task_data, tmp_path):
    tables = tmp_path / "mydb" / "tables"
    tables.mkdir(parents=True)
    (tables / "users.md").write_text("# users\n")
    task_data.deep_catalog_root = str(tmp_path)
    task_data.selected_database = "mydb"

    out = catalog_seed.materialize_maintenance_workspace(task_data)

    assert not (out / "docs" / "database_overview.md").exists()
    assert (out / "docs" / "tables" / "users.md").exists()


def test_materialize_writes_issue_with_task_question_and_empty_comments(task_data, tmp_path):
    _make_catalog(tmp_path, "mydb")
    task_data.deep_catalog_root = str(tmp_path)
    task_data.selected_database = "mydb"
    task_data.task_question = "How many active users do we have?"

    out = catalog_seed.materialize_maintenance_workspace(task_data)

    issue = (out / "ISSUE.md").read_text()
    assert issue == "# Issue\n\nHow many active users do we have?\n\n## Comments\n"


def test_materialize_writes_empty_stub_query(task_data, tmp_path):
    _make_catalog(tmp_path, "mydb")
    task_data.deep_catalog_root = str(tmp_path)
    task_data.selected_database = "mydb"

    out = catalog_seed.materialize_maintenance_workspace(task_data)

    stub = (out / "queries" / "answer.sql").read_text()
    assert stub == "-- TODO: replace this stub with your SQL query.\n"


def test_materialize_writes_test_contract_reference(task_data, tmp_path):
    _make_catalog(tmp_path, "mydb")
    task_data.deep_catalog_root = str(tmp_path)
    task_data.selected_database = "mydb"

    out = catalog_seed.materialize_maintenance_workspace(task_data)

    contract = (out / "tests" / "test_contract.py").read_text()
    assert "def test_query_is_written" in contract
    assert "def test_query_parses" in contract


def test_materialize_missing_tables_dir_raises(task_data, tmp_path):
    (tmp_path / "mydb").mkdir()  # db dir exists but no tables/
    task_data.deep_catalog_root = str(tmp_path)
    task_data.selected_database = "mydb"
    with pytest.raises(FileNotFoundError):
        catalog_seed.materialize_maintenance_workspace(task_data)


def test_maintenance_tool_costs_five_tools():
    costs = catalog_seed.maintenance_tool_costs()
    assert set(costs) == {"bash", "write_query", "comment_on_issue", "run_tests", "submit"}
    assert costs["bash"] == 1.0
    assert costs["write_query"] == 1.0
    assert costs["run_tests"] == 1.0
