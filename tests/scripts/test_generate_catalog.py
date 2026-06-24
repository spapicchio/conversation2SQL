from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from extract_ddl import Column  # noqa: E402
from generate_catalog import _enums_used_by_table  # noqa: E402


def _col(name: str, data_type: str) -> Column:
    return Column(
        name=name,
        data_type=data_type,
        nullable=True,
        default=None,
        is_pk=False,
        is_unique=False,
    )


def test_enums_used_by_table_filters_and_preserves_order():
    columns = [
        _col("id", "integer"),
        _col("status", "account_status"),
        _col("tags", "label_kind[]"),
    ]
    all_enums = [
        ("label_kind", ["a", "b"]),
        ("account_status", ["active", "closed"]),
        ("unused_enum", ["x"]),
    ]
    result = _enums_used_by_table(columns, all_enums)
    assert result == [
        ("label_kind", ["a", "b"]),
        ("account_status", ["active", "closed"]),
    ]


def test_enums_used_by_table_empty_when_none_used():
    columns = [_col("id", "integer")]
    all_enums = [("account_status", ["active", "closed"])]
    assert _enums_used_by_table(columns, all_enums) == []


import json  # noqa: E402

from generate_catalog import load_column_meanings  # noqa: E402


def _write_meaning_file(tmp_path: Path, db_name: str, payload: dict) -> Path:
    db_dir = tmp_path / db_name
    db_dir.mkdir(parents=True)
    (db_dir / f"{db_name}_column_meaning_base.json").write_text(json.dumps(payload))
    return tmp_path


def test_load_column_meanings_groups_by_table(tmp_path):
    dataset_path = _write_meaning_file(
        tmp_path,
        "mydb",
        {
            "mydb|customers|email": "Login email, unique",
            "mydb|customers|id": "Surrogate PK",
            "mydb|orders|customer_id": "FK to customers",
        },
    )
    result = load_column_meanings(dataset_path, "mydb")
    assert result == {
        "customers": {"email": "Login email, unique", "id": "Surrogate PK"},
        "orders": {"customer_id": "FK to customers"},
    }


def test_load_column_meanings_missing_file_returns_empty(tmp_path):
    assert load_column_meanings(tmp_path, "absent_db") == {}
