import pytest

from conversation2sql.eval_framework.state import TaskData
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
