from conversation2sql.eval_framework.agents.deep_agent.filesystem_seed import (
    build_db_filesystem,
    DB_FS_PATHS,
)


def test_builds_exactly_three_db_files(task_data):
    files = build_db_filesystem(task_data)
    assert set(files) == set(DB_FS_PATHS)
    assert set(DB_FS_PATHS) == {
        "/db/schema.sql",
        "/db/column_meanings.md",
        "/db/knowledge_base.md",
    }


def test_schema_file_contains_raw_ddl(task_data):
    files = build_db_filesystem(task_data)
    schema = files["/db/schema.sql"]
    assert schema["encoding"] == "utf-8"
    assert task_data.ddl_database_schema.strip()[:20] in schema["content"]


def test_empty_kb_yields_none_sentinel(task_data):
    task_data.masked_agent_kb = {}
    files = build_db_filesystem(task_data)
    assert files["/db/knowledge_base.md"]["content"].strip() == "(none)"


def test_empty_column_meanings_yields_none_sentinel(task_data):
    task_data.column_meanings = {}
    files = build_db_filesystem(task_data)
    assert files["/db/column_meanings.md"]["content"].strip() == "(none)"
