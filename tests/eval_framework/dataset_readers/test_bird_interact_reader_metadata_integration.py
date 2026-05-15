import json

from conversation2sql.eval_framework.dataset_readers.bird_interact_reader import (
    load_bird_interact_as_tasks,
)


def _build_dataset(tmp_path, sol_sql):
    dataset_path = tmp_path / "bird_interact"
    db_name = "testdb"
    db_dir = dataset_path / db_name
    db_dir.mkdir(parents=True)

    (db_dir / f"{db_name}_schema.txt").write_text(
        "CREATE TABLE users (id int, email text);\n"
        "CREATE TABLE orders (id int, user_id int, amount int);\n"
    )
    column_meanings = {
        f"{db_name}|users|id": "id",
        f"{db_name}|users|email": "email",
        f"{db_name}|orders|id": "id",
        f"{db_name}|orders|user_id": "user id",
        f"{db_name}|orders|amount": "amount",
    }
    (db_dir / f"{db_name}_column_meaning_base.json").write_text(
        json.dumps(column_meanings)
    )
    (db_dir / f"{db_name}_kb.jsonl").write_text("")

    line = {
        "instance_id": "test_instance_1",
        "selected_database": db_name,
        "category": "Query",
        "amb_user_query": "List user emails",
        "query": "List user emails",
        "sol_sql": sol_sql,
        "follow_up": {"query": "", "sol_sql": []},
        "user_query_ambiguity": {"critical_ambiguity": []},
        "knowledge_ambiguity": [],
        "external_knowledge": [],
        "preprocess_sql": [],
        "test_cases": [],
        "clean_up_sqls": [],
        "conditions": {"order": False},
    }

    dataset_jsonl = dataset_path / "bird_interact_data_GT.jsonl"
    dataset_jsonl.write_text(json.dumps(line) + "\n")

    return dataset_path, dataset_jsonl


def _load_samples(dataset_path, dataset_jsonl, make_data_ambiguous=True):
    return load_bird_interact_as_tasks(
        dataset_path=dataset_path,
        dataset_name_jsonl=str(dataset_jsonl),
        filter_query_category=True,
        db_dsn_template="postgresql://root:123123@localhost:5432/{database}",
        user_patience_budget=1,
        make_data_ambiguous=make_data_ambiguous,
    )


def test_reader_adds_table_in_gt_sql(tmp_path):
    sql = [
        "SELECT u.id, o.amount "
        "FROM users u JOIN orders o ON u.id = o.user_id"
    ]
    dataset_path, dataset_jsonl = _build_dataset(tmp_path, sql)

    samples = _load_samples(dataset_path, dataset_jsonl)

    assert len(samples) == 1
    assert samples[0].table_in_gt_sql == {
        "users": ["id"],
        "orders": ["amount", "user_id"],
    }
    assert samples[0].table_in_gt_sql_parse_error is None


def test_reader_parse_error_marker(tmp_path):
    dataset_path, dataset_jsonl = _build_dataset(tmp_path, ["SELECT FROM"])

    samples = _load_samples(dataset_path, dataset_jsonl)

    assert len(samples) == 1
    assert samples[0].table_in_gt_sql == {}
    assert samples[0].table_in_gt_sql_parse_error


def test_reader_metadata_with_no_ambiguity_flag(tmp_path):
    sql = ["SELECT users.email FROM users"]
    dataset_path, dataset_jsonl = _build_dataset(tmp_path, sql)

    samples = _load_samples(dataset_path, dataset_jsonl, make_data_ambiguous=False)

    assert len(samples) == 1
    assert samples[0].table_in_gt_sql == {"users": ["email"]}
