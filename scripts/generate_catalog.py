"""Generate a per-table Markdown schema catalog for one Postgres database.

Run with `uv run python scripts/generate_catalog.py --help` for CLI options.
"""

from __future__ import annotations

import logging
from pathlib import Path

from conversation2sql.eval_framework.dataset_readers.bird_interact_reader import (
    _get_column_meanings,
)
from extract_ddl import Column

logger = logging.getLogger("generate_catalog")


def _enums_used_by_table(
    columns: list[Column],
    all_enums: list[tuple[str, list[str]]],
) -> list[tuple[str, list[str]]]:
    enum_names = {name for name, _ in all_enums}
    used: set[str] = set()
    for col in columns:
        base = col.data_type.strip().removesuffix("[]").strip().strip('"')
        if base in enum_names:
            used.add(base)
    return [(name, labels) for name, labels in all_enums if name in used]


def load_column_meanings(
    dataset_path: Path, db_name: str
) -> dict[str, dict[str, str]]:
    """Load `<db>_column_meaning_base.json` into {table: {column: meaning}}.

    Keys in the source file are lowercased `db|table|column`. Missing file
    yields an empty mapping (catalog still renders, with empty descriptions).
    """
    try:
        raw = _get_column_meanings(dataset_path, db_name)
    except FileNotFoundError:
        logger.warning(
            "no column-meaning file for %s under %s; descriptions will be empty",
            db_name,
            dataset_path,
        )
        return {}
    out: dict[str, dict[str, str]] = {}
    prefix = f"{db_name.lower()}|"
    for key, entry in raw.items():
        if not key.startswith(prefix):
            continue
        parts = key.split("|")
        if len(parts) != 3:
            continue
        _, table, column = parts
        out.setdefault(table, {})[column] = entry.column_meaning
    return out
