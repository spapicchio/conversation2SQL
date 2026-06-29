"""Generate a per-table Markdown schema catalog for one Postgres database.

For a given database this writes one Markdown file per table under
``<output-dir>/<database>/<table>.md``, each containing an (empty) description
placeholder, the table DDL (``CREATE TYPE`` enums it uses + ``CREATE TABLE``),
a Columns table joining live Postgres types with descriptions from the
dataset's ``<db>_column_meaning_base.json``, and a Foreign keys section.

For a whole-database run it also writes ``<output-dir>/<database>/_constraints.md``,
a single file listing every primary-key and foreign-key constraint in the db.

Examples
--------
Every benchmark database under ``--dataset-path`` — omit ``--database``::

    uv run python scripts/generate_catalog.py --output-dir catalogs

All tables of a single database on the lite container (:5432)::

    uv run python scripts/generate_catalog.py --database alien --output-dir catalogs

A single table, with verbose logging::

    uv run python scripts/generate_catalog.py \
        --database alien --table signals --output-dir catalogs -v

A database on the full container (:5433) — swap the port in the DSN template::

    uv run python scripts/generate_catalog.py \
        --database crypto --output-dir catalogs \
        --db-dsn-template "postgresql://root:123123@localhost:5433/{database}"

Point at a different dataset location for column descriptions::

    uv run python scripts/generate_catalog.py \
        --database alien --output-dir catalogs \
        --dataset-path data/bird_interact/bird-interact-full

Run with ``--help`` for the full list of options.
"""

from __future__ import annotations
from conversation2sql.eval_framework.agents.utils_kb_linearize import (
    linearize_prerequisites,
)
from conversation2sql.eval_framework.dataset_readers.bird_interact_reader import (
    _get_external_knowledge,
)

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
from extract_ddl import (
    Column,
    Table,
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
        base = col.data_type.strip().removesuffix("[]").strip().strip('"')
        if base in enum_names:
            used.add(base)
    return [(name, labels) for name, labels in all_enums if name in used]


def load_column_meanings(dataset_path: Path, db_name: str) -> dict[str, dict[str, str]]:
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
    # descriptions=None: column descriptions live in the ## Columns table
    # below, so the DDL block stays a clean CREATE TABLE.
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
        f"- {fk.column} -> {fk.ref_table}({fk.ref_column})" for fk in table.foreign_keys
    ]
    ref_lines = [f"- referenced by: {ref}" for ref in referenced_by]
    if fk_lines or ref_lines:
        lines.append("")
        lines.append("## Foreign keys")
        lines.extend(fk_lines)
        lines.extend(ref_lines)

    lines.append("")
    return "\n".join(lines)


def _primary_key_columns(table: Table) -> list[str]:
    """Primary-key column names for a table (composite or single-column)."""
    if table.composite_pk:
        return list(table.composite_pk)
    return [col.name for col in table.columns if col.is_pk]


def render_constraints_markdown(database: str, tables: list[Table]) -> str:
    """Render a single Markdown file listing every PK/FK constraint in the db.

    One ``## Primary keys`` table (table -> key columns) followed by one
    ``## Foreign keys`` table (table.column -> referenced table.column, plus the
    ON DELETE action). Tables are listed in the order they were loaded.
    """
    lines: list[str] = []
    lines.append(f"# constraints: {database}")
    lines.append("")

    lines.append("## Primary keys")
    lines.append("| table | columns |")
    lines.append("| --- | --- |")
    for table in tables:
        pk_cols = _primary_key_columns(table)
        if not pk_cols:
            continue
        cols = _md_cell(", ".join(pk_cols))
        lines.append(f"| {_md_cell(table.name)} | {cols} |")
    lines.append("")

    lines.append("## Foreign keys")
    lines.append("| table | column | references |")
    lines.append("| --- | --- | --- |")
    for table in tables:
        for fk in table.foreign_keys:
            ref = _md_cell(f"{fk.ref_table}({fk.ref_column})")
            lines.append(f"| {_md_cell(table.name)} | {_md_cell(fk.column)} | {ref} |")
    lines.append("")
    return "\n".join(lines)


def fetch_databases(dataset_path: Path) -> list[str]:
    """List benchmark databases under ``dataset_path``, sorted alphabetically.

    A database is a subdirectory holding ``<name>_column_meaning_base.json``
    (the per-DB marker BIRD-Interact ships); this skips ``.git`` and any stray
    directories.
    """
    return sorted(
        d.name
        for d in dataset_path.iterdir()
        if d.is_dir() and (d / f"{d.name}_column_meaning_base.json").is_file()
    )


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


def _generate_tables_for_db(dataset_path, database, db_out, conn, schema, only_table):
    meanings = load_column_meanings(dataset_path, database)
    all_enums = fetch_enums(conn, schema)
    table_names = [only_table] if only_table else fetch_tables(conn, schema)
    written = 0
    loaded: list[Table] = []
    for name in table_names:
        try:
            # Pass an empty DSN so load_table skips fetch_examples: the
            # catalog has no example-rows section, and fetch_examples in
            # extract_ddl currently mis-calls _format_result.
            table = load_table(conn, schema, name, "")
            enums = _enums_used_by_table(table.columns, all_enums)
            referenced_by = fetch_referenced_by(conn, schema, name)
            # meanings are keyed by lowercased table name; Postgres reports
            # unquoted identifiers lowercased, which is the BIRD-Interact norm.
            per_table = meanings.get(name.lower(), {})
            md = render_table_markdown(table, enums, per_table, referenced_by)
        except Exception:
            # Any per-table failure (DB error, introspection quirk, render
            # bug) skips that table and continues, so one bad table cannot
            # abort an unattended full-database run.
            logger.warning("skipping table %s", name, exc_info=True)
            continue
        (db_out / f"{name}.md").write_text(md, encoding="utf-8")
        loaded.append(table)
        written += 1
    logger.info("wrote %d table file(s) under %s", written, db_out)

    # Db-level constraints file: only meaningful for a whole-database run,
    # since a single-table run cannot list cross-table PK/FK relationships.
    if only_table is None and loaded:
        constraints_md = render_constraints_markdown(database, loaded)
        (db_out / "_foreign_key_constraints.md").write_text(
            constraints_md, encoding="utf-8"
        )
        logger.info("wrote constraints file %s", db_out / "_foreign_key_constraints.md")
    return written


def _generate_kb_for_db(dataset_path, database, kb_out):
    """
    Generate a knowledge base for a database.
    """

    external_kb = _get_external_knowledge(dataset_path, database)
    # print(external_kb)
    written = 0
    for kb_name in external_kb:
        kb_file = kb_out / f"{kb_name}.md"
        kb_linearize_content = linearize_prerequisites(kb_name, external_kb)
        kb_file.write_text(kb_linearize_content, encoding="utf-8")
        written += 1

    return written


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
    db_out = output_dir / database / "tables"
    db_out.mkdir(parents=True, exist_ok=True)
    kb_out = output_dir / database / "knowledge_base"
    kb_out.mkdir(parents=True, exist_ok=True)
    try:
        written_tbl = _generate_tables_for_db(
            dataset_path, database, db_out, conn, schema, only_table
        )
        written_kb = _generate_kb_for_db(dataset_path, database, kb_out)
        logger.info(
            f"catalog generation for {database} complete: {written_tbl} table(s), {written_kb} knowledge base(s)",
        )
    except Exception:
        logger.error(f"catalog generation failed for {database}", exc_info=True)
    finally:
        conn.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate a per-table Markdown schema catalog for one database."
    )
    p.add_argument(
        "--database",
        default=None,
        help="Postgres database name (default: every database under --dataset-path)",
    )
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
    if args.table and not args.database:
        logger.error("--table requires --database (cannot target one table across all databases)")
        return 1
    try:
        databases = (
            [args.database]
            if args.database
            else fetch_databases(args.dataset_path)
        )
        if not args.database:
            logger.info("no --database given; generating catalog for %d database(s): %s",
                        len(databases), ", ".join(databases))
        for database in databases:
            generate_catalog_for_db(
                database=database,
                output_dir=args.output_dir,
                db_dsn_template=args.db_dsn_template,
                dataset_path=args.dataset_path,
                schema=args.schema,
                only_table=args.table,
            )
    except (psycopg2.Error, OSError, ValueError) as exc:
        # psycopg2.Error: connection/extraction; OSError: output write/path;
        # ValueError: malformed meaning JSON (json.JSONDecodeError). Report
        # cleanly with a non-zero exit instead of a raw traceback.
        logger.error("catalog generation failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
