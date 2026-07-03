from pathlib import Path

import pytest

from conversation2sql.eval_framework.agents.deep_agent import catalog_seed
from conversation2sql.eval_framework.state import ExternalKnowledgeEntry


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
    # one KB file per surviving (masked) node, none for absent names
    assert (out / "knowledge_base" / "active_user.md").exists()
    assert (out / "knowledge_base" / "revenue.md").exists()
    assert not (out / "knowledge_base" / "secret_kb.md").exists()


def test_materialize_appends_kb_overview_to_database_overview(task_data, tmp_path):
    """database_overview.md keeps the static disk content and gains a masked KB index."""
    _make_catalog(tmp_path, "mydb")
    task_data.deep_catalog_root = str(tmp_path)
    task_data.selected_database = "mydb"

    out = catalog_seed.materialize_catalog_dir(task_data)

    overview = (out / "database_overview.md").read_text()
    assert overview.startswith("overview\n")
    assert "## Knowledge Base" in overview
    assert "active_user" in overview
    assert "knowledge_base/active_user.md" in overview
    assert "revenue" in overview
    assert "knowledge_base/revenue.md" in overview
    # a KB entry not present in masked_agent_kb never leaks into the index
    assert "secret_kb" not in overview


def test_materialize_strips_unmasked_disk_kb_section_before_replacing(task_data, tmp_path):
    """generate_catalog.py may write an unmasked KB section to disk; the
    per-task copy must replace it with the masked one, never leak both."""
    _make_catalog(tmp_path, "mydb")
    (tmp_path / "mydb" / "database_overview.md").write_text(
        "overview\n\n## Knowledge Base\n- **secret_kb** (`knowledge_base/secret_kb.md`): masked-out entry\n"
    )
    task_data.deep_catalog_root = str(tmp_path)
    task_data.selected_database = "mydb"

    out = catalog_seed.materialize_catalog_dir(task_data)

    overview = (out / "database_overview.md").read_text()
    assert overview.count("## Knowledge Base") == 1
    assert "secret_kb" not in overview
    assert "active_user" in overview


def test_materialize_overview_unchanged_when_kb_empty(task_data, tmp_path):
    """No Knowledge Base section is appended when the task's masked KB is empty."""
    _make_catalog(tmp_path, "mydb")
    task_data.deep_catalog_root = str(tmp_path)
    task_data.selected_database = "mydb"
    task_data.masked_agent_kb = {}

    out = catalog_seed.materialize_catalog_dir(task_data)

    assert (out / "database_overview.md").read_text() == "overview\n"


def test_materialize_slugifies_kb_filenames_with_spaces_and_parens(
    task_data, tmp_path
):
    _make_catalog(tmp_path, "mydb")
    task_data.deep_catalog_root = str(tmp_path)
    task_data.selected_database = "mydb"
    task_data.masked_agent_kb = {
        "Coherent Information Pattern (CIP)": ExternalKnowledgeEntry(
            id=1,
            knowledge="Coherent Information Pattern (CIP)",
            description="",
            definition="",
            type="domain_knowledge",
            children_knowledge=[],
        )
    }

    out = catalog_seed.materialize_catalog_dir(task_data)

    kb_files = list((out / "knowledge_base").iterdir())
    assert [f.name for f in kb_files] == ["coherent_information_pattern_cip.md"]
    assert "Coherent Information Pattern (CIP)" in kb_files[0].read_text()


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
