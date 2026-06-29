import pytest

from conversation2sql.eval_framework.state import ExternalKnowledgeEntry, TaskData
from conversation2sql.eval_framework.agents.deep_agent.filesystem_seed import (
    build_db_filesystem,
)


def _write_catalog(root, db="mydb"):
    """Create a minimal on-disk tables catalog under <root>/<db>/tables/."""
    tables = root / db / "tables"
    tables.mkdir(parents=True)
    (tables / "users.md").write_text("# table: users\n", encoding="utf-8")
    (tables / "_foreign_key_constraints.md").write_text(
        "# constraints: mydb\n", encoding="utf-8"
    )
    return root


def test_tables_are_loaded_verbatim_from_disk(tmp_path, make_minimal_task_kwargs):
    _write_catalog(tmp_path)
    task = TaskData(
        **make_minimal_task_kwargs(
            selected_database="mydb",
            deep_catalog_root=str(tmp_path),
            masked_agent_kb={},
        )
    )
    files = build_db_filesystem(task)
    assert files["/db/tables/users.md"]["content"] == "# table: users\n"
    assert files["/db/tables/users.md"]["encoding"] == "utf-8"
    assert "/db/tables/_foreign_key_constraints.md" in files


def test_missing_catalog_dir_raises(tmp_path, make_minimal_task_kwargs):
    task = TaskData(
        **make_minimal_task_kwargs(
            selected_database="absent_db",
            deep_catalog_root=str(tmp_path),
            masked_agent_kb={},
        )
    )
    with pytest.raises(FileNotFoundError, match="absent_db"):
        build_db_filesystem(task)


def test_db_dir_without_tables_subdir_raises(tmp_path, make_minimal_task_kwargs):
    # Catalog dir exists but the tables/ subdir was never generated (partial run).
    (tmp_path / "mydb").mkdir(parents=True)
    task = TaskData(
        **make_minimal_task_kwargs(
            selected_database="mydb",
            deep_catalog_root=str(tmp_path),
            masked_agent_kb={},
        )
    )
    with pytest.raises(FileNotFoundError, match="tables"):
        build_db_filesystem(task)


def _kb_pair():
    """Two nodes: 'Score (SC)' depends on 'Base (BS)' (SC.children = [BS.id])."""
    return {
        "Score (SC)": ExternalKnowledgeEntry(
            id=1,
            knowledge="Score (SC)",
            description="A score",
            definition="SC = BS * 2",
            type="calculation_knowledge",
            children_knowledge=[2],
        ),
        "Base (BS)": ExternalKnowledgeEntry(
            id=2,
            knowledge="Base (BS)",
            description="A base value",
            definition="BS = 10",
            type="domain_knowledge",
            children_knowledge=[-1],
        ),
    }


def test_kb_files_only_for_nodes_in_masked_kb(tmp_path, make_minimal_task_kwargs):
    _write_catalog(tmp_path)
    full = _kb_pair()
    masked = {"Score (SC)": full["Score (SC)"]}  # 'Base (BS)' masked out
    task = TaskData(
        **make_minimal_task_kwargs(
            selected_database="mydb",
            deep_catalog_root=str(tmp_path),
            masked_agent_kb=masked,
        )
    )
    files = build_db_filesystem(task)
    assert "/db/knowledge_base/Score (SC).md" in files
    assert "/db/knowledge_base/Base (BS).md" not in files


def test_dangling_edge_to_masked_node_is_stripped(tmp_path, make_minimal_task_kwargs):
    _write_catalog(tmp_path)
    full = _kb_pair()
    masked = {"Score (SC)": full["Score (SC)"]}  # prerequisite 'Base (BS)' masked
    task = TaskData(
        **make_minimal_task_kwargs(
            selected_database="mydb",
            deep_catalog_root=str(tmp_path),
            masked_agent_kb=masked,
        )
    )
    files = build_db_filesystem(task)
    content = files["/db/knowledge_base/Score (SC).md"]["content"]
    assert "needs" not in content  # edge to the masked 'BS' node is gone


def test_edge_present_when_prerequisite_not_masked(tmp_path, make_minimal_task_kwargs):
    _write_catalog(tmp_path)
    task = TaskData(
        **make_minimal_task_kwargs(
            selected_database="mydb",
            deep_catalog_root=str(tmp_path),
            masked_agent_kb=_kb_pair(),  # both nodes present
        )
    )
    files = build_db_filesystem(task)
    content = files["/db/knowledge_base/Score (SC).md"]["content"]
    assert '"Score (SC)" needs "Base (BS)"' in content


def test_empty_kb_seeds_no_knowledge_base_files(tmp_path, make_minimal_task_kwargs):
    _write_catalog(tmp_path)
    task = TaskData(
        **make_minimal_task_kwargs(
            selected_database="mydb",
            deep_catalog_root=str(tmp_path),
            masked_agent_kb={},
        )
    )
    files = build_db_filesystem(task)
    assert not any(p.startswith("/db/knowledge_base/") for p in files)
