"""Generate a per-table Markdown schema catalog for one Postgres database.

Run with `uv run python scripts/generate_catalog.py --help` for CLI options.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from urllib.parse import urlparse

import psycopg2
from psycopg2.extensions import connection as PgConnection

from extract_ddl import (
    Table,
    Column,
    _render_table_ddl,
    fetch_enums,
    fetch_tables,
    load_table,
    open_readonly,
)

logger = logging.getLogger("generate_catalog")


def _enums_used_by_table(
    columns: list[Column],
    all_enums: list[tuple[str, list[str]]],
) -> list[tuple[str, list[str]]]:
    enum_names = {name for name, _ in all_enums}
    used: set[str] = set()
    for col in columns:
        base = col.data_type.replace("[]", "").strip().strip('"')
        if base in enum_names:
            used.add(base)
    return [(name, labels) for name, labels in all_enums if name in used]
