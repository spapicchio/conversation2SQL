"""Generate a per-table Markdown schema catalog for one Postgres database.

Run with `uv run python scripts/generate_catalog.py --help` for CLI options.
"""

from __future__ import annotations

import logging
from pathlib import Path

from conversation2sql.eval_framework.dataset_readers.bird_interact_reader import (
    _get_column_meanings,
)
from extract_ddl import Column, Table, _render_table_ddl

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


def _md_cell(text: str) -> str:
    """Sanitize a value for a Markdown table cell (escape pipes, flatten newlines)."""
    return text.replace("|", "\\|").replace("\r", " ").replace("\n", " ").strip()


def render_table_markdown(
    table: Table,
    enums: list[tuple[str, list[str]]],
    column_meanings: dict[str, str],
    referenced_by: list[str],
) -> str:
    lines: list[str] = []
    lines.append(f"# table: {table.name}")
    lines.append("")
    lines.append("## Description")
    lines.append("")  # intentional empty placeholder

    lines.append("## DDL")
    lines.append("```sql")
    for ename, labels in enums:
        quoted = ", ".join(f"'{lbl}'" for lbl in labels)
        lines.append(f'CREATE TYPE "{ename}" AS ENUM ({quoted});')
    if enums:
        lines.append("")
    lines.append(_render_table_ddl(table, None))
    lines.append("```")
    lines.append("")

    lines.append("## Columns")
    lines.append("| column | type | description |")
    lines.append("| --- | --- | --- |")
    for col in table.columns:
        name = _md_cell(col.name)
        data_type = _md_cell(col.data_type)
        desc = _md_cell(column_meanings.get(col.name.lower(), ""))
        lines.append(f"| {name} | {data_type} | {desc} |")

    fk_lines = [
        f"- {fk.column} -> {fk.ref_table}({fk.ref_column})"
        for fk in table.foreign_keys
    ]
    ref_lines = [f"- referenced by: {ref}" for ref in referenced_by]
    if fk_lines or ref_lines:
        lines.append("")
        lines.append("## Foreign keys")
        lines.extend(fk_lines)
        lines.extend(ref_lines)

    lines.append("")
    return "\n".join(lines)
