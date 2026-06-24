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

from conversation2sql.eval_framework.dataset_readers.bird_interact_reader import (
    _get_column_meanings,
)
from extract_ddl import Column, Table, _render_table_ddl, fetch_enums, fetch_tables, load_table, open_readonly

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


def fetch_referenced_by(conn: PgConnection, schema: str, table: str) -> list[str]:
    """Tables/columns that hold a foreign key pointing AT ``table``."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT c2.relname AS referencing_table,
                   a.attname  AS referencing_column
            FROM pg_constraint con
            JOIN pg_class c1     ON c1.oid = con.confrelid
            JOIN pg_class c2     ON c2.oid = con.conrelid
            JOIN pg_namespace n  ON n.oid = c1.relnamespace
            JOIN pg_attribute a  ON a.attrelid = con.conrelid
                                AND a.attnum = ANY(con.conkey)
            WHERE con.contype = 'f'
              AND c1.relname = %s
              AND n.nspname = %s
            ORDER BY referencing_table, referencing_column
            """,
            (table, schema),
        )
        return [f"{row[0]}({row[1]})" for row in cur.fetchall()]


def generate_catalog_for_db(
    database: str,
    output_dir: Path,
    db_dsn_template: str,
    dataset_path: Path,
    schema: str = "public",
    only_table: str | None = None,
) -> int:
    dsn = db_dsn_template.format(database=database)
    parsed = urlparse(dsn)
    conn = open_readonly(
        host=parsed.hostname or "localhost",
        port=parsed.port or 5432,
        user=parsed.username or "root",
        password=parsed.password or "",
        dbname=database,
    )
    try:
        meanings = load_column_meanings(dataset_path, database)
        all_enums = fetch_enums(conn, schema)
        table_names = [only_table] if only_table else fetch_tables(conn, schema)

        db_out = output_dir / database
        db_out.mkdir(parents=True, exist_ok=True)

        written = 0
        for name in table_names:
            try:
                # Pass an empty DSN so load_table skips fetch_examples: the
                # catalog has no example-rows section, and fetch_examples in
                # extract_ddl currently mis-calls _format_result.
                table = load_table(conn, schema, name, "")
                enums = _enums_used_by_table(table.columns, all_enums)
                referenced_by = fetch_referenced_by(conn, schema, name)
                per_table = meanings.get(name.lower(), {})
                md = render_table_markdown(table, enums, per_table, referenced_by)
            except psycopg2.Error as exc:
                logger.warning("skipping table %s: %s", name, exc)
                continue
            (db_out / f"{name}.md").write_text(md, encoding="utf-8")
            written += 1
        logger.info("wrote %d table file(s) under %s", written, db_out)
        return written
    finally:
        conn.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate a per-table Markdown schema catalog for one database."
    )
    p.add_argument("--database", required=True, help="Postgres database name")
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--table", default=None, help="single table (default: all tables)")
    p.add_argument(
        "--db-dsn-template",
        default="postgresql://root:123123@localhost:5432/{database}",
        help="DSN with a {database} placeholder; use :5433 for the full dataset",
    )
    p.add_argument(
        "--dataset-path",
        type=Path,
        default=Path("data/bird_interact/bird-interact-lite"),
        help="locates <db>_column_meaning_base.json for column descriptions",
    )
    p.add_argument("--schema", default="public")
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    try:
        generate_catalog_for_db(
            database=args.database,
            output_dir=args.output_dir,
            db_dsn_template=args.db_dsn_template,
            dataset_path=args.dataset_path,
            schema=args.schema,
            only_table=args.table,
        )
    except psycopg2.Error as exc:
        logger.error("connection/extraction failed for %s: %s", args.database, exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
