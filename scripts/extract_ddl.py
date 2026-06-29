"""Extract PostgreSQL schema as classical DDL or TOON format.

Run with `uv run python scripts/extract_ddl.py --help` for CLI options.
"""

from __future__ import annotations
from conversation2sql.eval_framework.agents.bird_baseline.tools.utils_db_execute import (
    _format_result,
)


import argparse
import fnmatch
import logging
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import psycopg2
from psycopg2 import sql
from psycopg2.extensions import connection as PgConnection

logger = logging.getLogger("extract_ddl")

STATEMENT_TIMEOUT_MS = 5000
TRUNCATE_LEN = 1000

SENSITIVE_PATTERNS: list[tuple[str, str]] = [
    ("password*", "<password>"),
    ("*_hash", "<hash>"),
    ("*token*", "<token>"),
    ("*secret*", "<secret>"),
    ("*api_key*", "<api_key>"),
    ("ssn", "<ssn>"),
    ("*credit_card*", "<credit_card>"),
]


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class Column:
    name: str
    data_type: str
    nullable: bool
    default: str | None
    is_pk: bool
    is_unique: bool


@dataclass
class ForeignKey:
    column: str
    ref_table: str
    ref_column: str
    on_delete: str


@dataclass
class Index:
    name: str
    columns: list[str]
    unique: bool
    definition: str


@dataclass
class CheckConstraint:
    name: str
    expression: str


@dataclass
class Table:
    schema: str
    examples_str: str
    name: str
    columns: list[Column]
    foreign_keys: list[ForeignKey]
    indexes: list[Index]
    checks: list[CheckConstraint]
    composite_pk: list[str]


# ---------------------------------------------------------------------------
# Sensitive-column detection
# ---------------------------------------------------------------------------


def sensitive_placeholder(col_name: str) -> str | None:
    name = col_name.lower()
    for pattern, placeholder in SENSITIVE_PATTERNS:
        if fnmatch.fnmatchcase(name, pattern):
            return placeholder
    return None


# ---------------------------------------------------------------------------
# Schema introspection
# ---------------------------------------------------------------------------


def fetch_tables(conn: PgConnection, schema: str) -> list[str]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = %s AND table_type = 'BASE TABLE'
            ORDER BY table_name
            """,
            (schema,),
        )
        return [row[0] for row in cur.fetchall()]


def fetch_columns(conn: PgConnection, schema: str, table: str) -> list[Column]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT a.attname,
                   format_type(a.atttypid, a.atttypmod),
                   NOT a.attnotnull AS nullable,
                   pg_get_expr(d.adbin, d.adrelid) AS default_expr,
                   a.attnum
            FROM pg_attribute a
            LEFT JOIN pg_attrdef d
                   ON d.adrelid = a.attrelid AND d.adnum = a.attnum
            WHERE a.attrelid = %s::regclass
              AND a.attnum > 0
              AND NOT a.attisdropped
            ORDER BY a.attnum
            """,
            (f'"{schema}"."{table}"',),
        )
        rows = cur.fetchall()

        cur.execute(
            """
            SELECT a.attname,
                   con.contype
            FROM pg_constraint con
            JOIN pg_attribute a
              ON a.attrelid = con.conrelid AND a.attnum = ANY(con.conkey)
            WHERE con.conrelid = %s::regclass
              AND array_length(con.conkey, 1) = 1
              AND con.contype IN ('p', 'u')
            """,
            (f'"{schema}"."{table}"',),
        )
        flag_map: dict[str, set[str]] = {}
        for col_name, contype in cur.fetchall():
            flag_map.setdefault(col_name, set()).add(contype)

    columns = []
    for name, dtype, nullable, default, _ in rows:
        flags = flag_map.get(name, set())
        columns.append(
            Column(
                name=name,
                data_type=dtype,
                nullable=bool(nullable),
                default=default,
                is_pk="p" in flags,
                is_unique="u" in flags,
            )
        )
    return columns


def fetch_composite_pk(conn: PgConnection, schema: str, table: str) -> list[str]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT a.attname
            FROM pg_constraint con
            JOIN pg_attribute a
              ON a.attrelid = con.conrelid AND a.attnum = ANY(con.conkey)
            WHERE con.conrelid = %s::regclass
              AND con.contype = 'p'
              AND array_length(con.conkey, 1) > 1
            ORDER BY array_position(con.conkey, a.attnum)
            """,
            (f'"{schema}"."{table}"',),
        )
        return [row[0] for row in cur.fetchall()]


def fetch_foreign_keys(conn: PgConnection, schema: str, table: str) -> list[ForeignKey]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT con.conname,
                   pg_get_constraintdef(con.oid),
                   con.confdeltype
            FROM pg_constraint con
            WHERE con.conrelid = %s::regclass AND con.contype = 'f'
            ORDER BY con.conname
            """,
            (f'"{schema}"."{table}"',),
        )
        results = []
        for _, defstr, deltype in cur.fetchall():
            parsed = _parse_fk_def(defstr)
            if parsed is None:
                continue
            col, ref_table, ref_col = parsed
            results.append(
                ForeignKey(
                    column=col,
                    ref_table=ref_table,
                    ref_column=ref_col,
                    on_delete=_action_for(deltype),
                )
            )
        return results


_FK_RE = re.compile(
    r"FOREIGN KEY \(([^)]+)\) REFERENCES ([^\s(]+)\s*\(([^)]+)\)",
    re.IGNORECASE,
)


def _parse_fk_def(defstr: str) -> tuple[str, str, str] | None:
    m = _FK_RE.search(defstr)
    if not m:
        return None
    col = m.group(1).split(",")[0].strip().strip('"')
    ref_table = m.group(2).strip().strip('"')
    ref_col = m.group(3).split(",")[0].strip().strip('"')
    return col, ref_table, ref_col


def _action_for(code: str) -> str:
    return {
        "a": "NO ACTION",
        "r": "RESTRICT",
        "c": "CASCADE",
        "n": "SET NULL",
        "d": "SET DEFAULT",
    }.get(code, "NO ACTION")


def fetch_indexes(conn: PgConnection, schema: str, table: str) -> list[Index]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT i.relname,
                   ix.indisunique,
                   ix.indisprimary,
                   pg_get_indexdef(ix.indexrelid),
                   array(
                       SELECT a.attname
                       FROM unnest(ix.indkey) WITH ORDINALITY AS k(attnum, ord)
                       JOIN pg_attribute a
                         ON a.attrelid = ix.indrelid AND a.attnum = k.attnum
                       ORDER BY k.ord
                   )
            FROM pg_index ix
            JOIN pg_class i ON i.oid = ix.indexrelid
            WHERE ix.indrelid = %s::regclass
            ORDER BY i.relname
            """,
            (f'"{schema}"."{table}"',),
        )
        results = []

        for name, is_unique, is_primary, defstr, cols in cur.fetchall():
            if is_primary:
                continue

            # Regex to find "ON schema.table" and replace with "ON table"
            # Matches "ON ", then any non-whitespace characters (the schema),
            # followed by a dot, and captures the table name.
            clean_def = re.sub(r"( ON\s+)[^.\s]+\.", r"\1", defstr, flags=re.IGNORECASE)

            results.append(
                Index(
                    name=name,
                    columns=list(cols),
                    unique=bool(is_unique),
                    definition=clean_def,
                )
            )
        return results


def fetch_checks(conn: PgConnection, schema: str, table: str) -> list[CheckConstraint]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT con.conname, pg_get_constraintdef(con.oid)
            FROM pg_constraint con
            WHERE con.conrelid = %s::regclass AND con.contype = 'c'
            ORDER BY con.conname
            """,
            (f'"{schema}"."{table}"',),
        )
        return [
            CheckConstraint(name=name, expression=_strip_check_prefix(defstr))
            for name, defstr in cur.fetchall()
        ]


def _strip_check_prefix(defstr: str) -> str:
    s = defstr.strip()
    if s.upper().startswith("CHECK "):
        s = s[6:].strip()
    if s.startswith("(") and s.endswith(")"):
        s = s[1:-1].strip()
    return s


def fetch_table_comment(conn: PgConnection, schema: str, table: str) -> str | None:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT obj_description(c.oid, 'pg_class')
            FROM pg_class c
            JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE n.nspname = %s AND c.relname = %s
            """,
            (schema, table),
        )
        row = cur.fetchone()
        return row[0] if row else None


def fetch_column_comments(
    conn: PgConnection, schema: str, table: str
) -> dict[str, str]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT a.attname, col_description(a.attrelid, a.attnum)
            FROM pg_attribute a
            JOIN pg_class c ON c.oid = a.attrelid
            JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE n.nspname = %s AND c.relname = %s
              AND a.attnum > 0 AND NOT a.attisdropped
            """,
            (schema, table),
        )
        return {name: desc for name, desc in cur.fetchall() if desc}


def fetch_enums(conn: PgConnection, schema: str) -> list[tuple[str, list[str]]]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT t.typname,
                   array_agg(e.enumlabel ORDER BY e.enumsortorder)
            FROM pg_type t
            JOIN pg_enum e ON e.enumtypid = t.oid
            JOIN pg_namespace n ON n.oid = t.typnamespace
            WHERE n.nspname = %s
            GROUP BY t.typname
            ORDER BY t.typname
            """,
            (schema,),
        )
        return [(name, list(labels)) for name, labels in cur.fetchall()]


def fetch_examples(table: str, db_dsn: str) -> list[str]:
    conn = psycopg2.connect(db_dsn, cursor_factory=psycopg2.extras.RealDictCursor)
    conn.set_session(readonly=True, autocommit=True)
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT *
            FROM "{table}"
            LIMIT 3
            """,
        )
        rows = cur.fetchmany(3)
        desc = cur.description
        return _format_result(rows, desc, max_characters=TRUNCATE_LEN)


def load_table(conn: PgConnection, schema: str, name: str, db_dsn: str) -> Table:
    return Table(
        schema=schema,
        name=name,
        examples_str=fetch_examples(name, db_dsn) if db_dsn else "",
        columns=fetch_columns(conn, schema, name),
        foreign_keys=fetch_foreign_keys(conn, schema, name),
        indexes=fetch_indexes(conn, schema, name),
        checks=fetch_checks(conn, schema, name),
        composite_pk=fetch_composite_pk(conn, schema, name),
    )


# ---------------------------------------------------------------------------
# Classical DDL output
# ---------------------------------------------------------------------------


def render_classical_ddl(
    conn: PgConnection,
    schema: str,
    include_examples: bool = False,
    db_dsn: str = "",
    column_descriptions: dict[tuple[str, str], str] | None = None,
) -> str:
    out: list[str] = []
    out.append(f"-- PostgreSQL schema dump for schema: {schema}")
    out.append("")

    enums = fetch_enums(conn, schema)
    for ename, labels in enums:
        quoted = ", ".join(f"'{lbl}'" for lbl in labels)
        out.append(f'CREATE TYPE "{ename}" AS ENUM ({quoted});')
    if enums:
        out.append("")

    table_names = fetch_tables(conn, schema)

    tables = [load_table(conn, schema, t, db_dsn) for t in table_names]

    for table in tables:
        col_comments = fetch_column_comments(conn, schema, table.name)
        descriptions: dict[str, str] = dict(col_comments)
        if column_descriptions:
            for col in table.columns:
                if col.name in descriptions:
                    continue
                desc = column_descriptions.get((table.name.lower(), col.name.lower()))
                if desc:
                    descriptions[col.name] = desc

        out.append(_render_table_ddl(table, descriptions or None))
        if include_examples and table.examples_str:
            out.append("First 3 rows:")
            out.append(table.examples_str)
            out.append("...")

        comment = fetch_table_comment(conn, schema, table.name)
        if comment:
            out.append(f'COMMENT ON TABLE "{table.name}" IS {_sql_string(comment)};')
        out.append("")

    out.append("-- Indexes")
    for table in tables:
        for idx in table.indexes:
            out.append(f"{idx.definition};")
    out.append("")

    out.append("-- Foreign keys")
    for table in tables:
        for fk in table.foreign_keys:
            cname = f"fk_{table.name}_{fk.column}"
            out.append(
                f'ALTER TABLE "{table.name}" '
                f'ADD CONSTRAINT "{cname}" '
                f'FOREIGN KEY ("{fk.column}") '
                f'REFERENCES "{fk.ref_table}" ("{fk.ref_column}") '
                f"ON DELETE {fk.on_delete};"
            )
    out.append("")
    return "\n".join(out)


def _render_table_ddl(
    table: Table,
    descriptions: dict[str, str] | None = None,
    include_foreign_keys: bool = False,
) -> str:
    lines: list[str] = []
    lines.append(f'CREATE TABLE "{table.name}" (')

    rows: list[tuple[str, str | None]] = []
    for col in table.columns:
        bits = [f'    "{col.name}"', col.data_type]
        if col.default is not None:
            bits.append(f"DEFAULT {col.default}")
        if not col.nullable:
            bits.append("NOT NULL")
        if col.is_pk and not table.composite_pk:
            bits.append("PRIMARY KEY")
        if col.is_unique:
            bits.append("UNIQUE")
        desc = descriptions.get(col.name) if descriptions else None
        rows.append((" ".join(bits), desc))

    if table.composite_pk:
        cols = ", ".join(f'"{c}"' for c in table.composite_pk)
        rows.append((f"    PRIMARY KEY ({cols})", None))

    for chk in table.checks:
        rows.append((f'    CONSTRAINT "{chk.name}" CHECK ({chk.expression})', None))

    if include_foreign_keys:
        for fk in table.foreign_keys:
            rows.append(
                (
                    f'    FOREIGN KEY ("{fk.column}") '
                    f'REFERENCES "{fk.ref_table}" ("{fk.ref_column}")',
                    None,
                )
            )

    rendered: list[str] = []
    last = len(rows) - 1
    for i, (sql_part, desc) in enumerate(rows):
        suffix = "," if i < last else ""
        line = f"{sql_part}{suffix}"
        if desc:
            clean = desc.replace("\r", " ").replace("\n", " ")
            line = f"{line}  -- {clean}"
        rendered.append(line)
    lines.append("\n".join(rendered))
    lines.append(");")

    return "\n".join(lines)


def _sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


# ---------------------------------------------------------------------------
# Example selection (TOON mode)
# ---------------------------------------------------------------------------


_NUMERIC_TYPES = {
    "smallint",
    "integer",
    "bigint",
    "real",
    "double precision",
    "numeric",
    "smallserial",
    "serial",
    "bigserial",
}
_DATE_TYPES = {
    "date",
    "time without time zone",
    "time with time zone",
    "timestamp without time zone",
    "timestamp with time zone",
}
_BOOL_TYPES = {"boolean"}
_JSON_TYPES = {"json", "jsonb"}
_UUID_TYPES = {"uuid"}


def _base_type(data_type: str) -> str:
    s = data_type.lower()
    s = re.sub(r"\(.*?\)", "", s).strip()
    if s.endswith("[]"):
        return "array"
    return s


def _is_array(data_type: str) -> bool:
    return data_type.strip().endswith("[]")


def _qual(schema: str, table: str, col: str) -> sql.SQL:
    return sql.SQL("{}.{}.{} ").format(
        sql.Identifier(schema), sql.Identifier(table), sql.Identifier(col)
    )


def _qident(*parts: str) -> sql.SQL:
    return sql.SQL(".").join(sql.Identifier(p) for p in parts)


def _exec(conn: PgConnection, query: sql.Composable) -> list[tuple]:
    with conn.cursor() as cur:
        cur.execute(query)
        return cur.fetchall()


def _has_null(conn: PgConnection, schema: str, table: str, col: str) -> bool:
    rows = _exec(
        conn,
        sql.SQL("SELECT EXISTS (SELECT 1 FROM {} WHERE {} IS NULL)").format(
            _qident(schema, table), sql.Identifier(col)
        ),
    )
    return bool(rows[0][0])


def _distinct_count(conn: PgConnection, schema: str, table: str, col: str) -> int:
    rows = _exec(
        conn,
        sql.SQL("SELECT COUNT(DISTINCT {}) FROM {}").format(
            sql.Identifier(col), _qident(schema, table)
        ),
    )
    return int(rows[0][0])


def _low_card_values(
    conn: PgConnection, schema: str, table: str, col: str, limit: int = 10
) -> list[Any]:
    rows = _exec(
        conn,
        sql.SQL(
            "SELECT DISTINCT {col} FROM {tbl} WHERE {col} IS NOT NULL "
            "ORDER BY {col} LIMIT {lim}"
        ).format(
            col=sql.Identifier(col),
            tbl=_qident(schema, table),
            lim=sql.Literal(limit),
        ),
    )
    return [r[0] for r in rows]


def _numeric_summary(
    conn: PgConnection, schema: str, table: str, col: str
) -> list[Any]:
    rows = _exec(
        conn,
        sql.SQL(
            "SELECT MIN({col}), "
            "PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY {col}), "
            "MAX({col}) FROM {tbl}"
        ).format(col=sql.Identifier(col), tbl=_qident(schema, table)),
    )
    if not rows or rows[0][0] is None:
        return []
    return [v for v in rows[0] if v is not None]


def _date_minmax(conn: PgConnection, schema: str, table: str, col: str) -> list[Any]:
    rows = _exec(
        conn,
        sql.SQL("SELECT MIN({col}), MAX({col}) FROM {tbl}").format(
            col=sql.Identifier(col), tbl=_qident(schema, table)
        ),
    )
    if not rows or rows[0][0] is None:
        return []
    return [v for v in rows[0] if v is not None]


def _top_text(
    conn: PgConnection, schema: str, table: str, col: str, limit: int = 3
) -> list[Any]:
    rows = _exec(
        conn,
        sql.SQL(
            "SELECT {col} FROM {tbl} WHERE {col} IS NOT NULL "
            "GROUP BY {col} ORDER BY COUNT(*) DESC LIMIT {lim}"
        ).format(
            col=sql.Identifier(col),
            tbl=_qident(schema, table),
            lim=sql.Literal(limit),
        ),
    )
    return [r[0] for r in rows]


def _random_text(
    conn: PgConnection, schema: str, table: str, col: str, limit: int = 3
) -> list[Any]:
    rows = _exec(
        conn,
        sql.SQL(
            "SELECT {col} FROM ("
            "  SELECT {col} FROM {tbl} WHERE {col} IS NOT NULL LIMIT 5000"
            ") s ORDER BY random() LIMIT {lim}"
        ).format(
            col=sql.Identifier(col),
            tbl=_qident(schema, table),
            lim=sql.Literal(limit),
        ),
    )
    return [r[0] for r in rows]


def _fk_samples(
    conn: PgConnection, schema: str, fk: ForeignKey, limit: int = 3
) -> list[Any]:
    rows = _exec(
        conn,
        sql.SQL(
            "SELECT {col} FROM {tbl} WHERE {col} IS NOT NULL ORDER BY {col} LIMIT {lim}"
        ).format(
            col=sql.Identifier(fk.ref_column),
            tbl=_qident(schema, fk.ref_table),
            lim=sql.Literal(limit),
        ),
    )
    return [r[0] for r in rows]


def _one_row(conn: PgConnection, schema: str, table: str, col: str) -> list[Any]:
    rows = _exec(
        conn,
        sql.SQL("SELECT {col} FROM {tbl} WHERE {col} IS NOT NULL LIMIT 1").format(
            col=sql.Identifier(col), tbl=_qident(schema, table)
        ),
    )
    return [r[0] for r in rows]


def select_examples(
    conn: PgConnection,
    schema: str,
    table: Table,
    col: Column,
    fk_by_col: dict[str, ForeignKey],
) -> list[str]:
    placeholder = sensitive_placeholder(col.name)
    if placeholder is not None:
        return [placeholder]

    base = _base_type(col.data_type)
    is_array = _is_array(col.data_type)

    try:
        cardinality = _distinct_count(conn, schema, table.name, col.name)
    except Exception as exc:
        logger.warning(
            "cardinality probe failed for %s.%s.%s: %s",
            schema,
            table.name,
            col.name,
            exc,
        )
        return []

    if cardinality == 0:
        return []

    examples: list[Any] = []
    try:
        if cardinality <= 8 and not is_array:
            examples = _low_card_values(conn, schema, table.name, col.name)
        elif col.name in fk_by_col:
            examples = _fk_samples(conn, schema, fk_by_col[col.name])
        elif base in _BOOL_TYPES:
            examples = [True, False]
        elif base in _NUMERIC_TYPES:
            examples = _numeric_summary(conn, schema, table.name, col.name)
        elif base in _DATE_TYPES:
            examples = _date_minmax(conn, schema, table.name, col.name)
        elif base in _JSON_TYPES or base in _UUID_TYPES or is_array:
            examples = _one_row(conn, schema, table.name, col.name)
        else:
            examples = _top_text(conn, schema, table.name, col.name)
            # if cardinality > 50:
            #     examples = _top_text(conn, schema, table.name, col.name)
            # else:
            #     examples = _random_text(conn, schema, table.name, col.name)
    except Exception as exc:
        logger.warning(
            "example probe failed for %s.%s.%s: %s",
            schema,
            table.name,
            col.name,
            exc,
        )
        return []

    rendered = [_render_example(v) for v in examples if v is not None]

    if col.nullable:
        try:
            if _has_null(conn, schema, table.name, col.name):
                rendered = ["NULL", *rendered]
        except Exception as exc:
            logger.warning(
                "null probe failed for %s.%s.%s: %s",
                schema,
                table.name,
                col.name,
                exc,
            )

    return rendered[:5]


def _render_example(val: Any) -> str:
    if isinstance(val, bool):
        return "true" if val else "false"
    if isinstance(val, datetime):
        return val.isoformat()
    s = str(val)
    s = s.replace("\r", " ").replace("\n", " ")
    if len(s) > TRUNCATE_LEN:
        s = s[:TRUNCATE_LEN] + "..."
    return s


# ---------------------------------------------------------------------------
# TOON output
# ---------------------------------------------------------------------------


def _csv_escape(value: str) -> str:
    if any(c in value for c in (",", "|", '"', "\n")):
        return '"' + value.replace('"', '""') + '"'
    return value


def _csv_escape_field(value: str) -> str:
    if any(c in value for c in (",", '"', "\n")):
        return '"' + value.replace('"', '""') + '"'
    return value


def _format_examples(examples: list[str]) -> str:
    if not examples:
        return ""
    parts = [_csv_escape(e) if e != "NULL" else "NULL" for e in examples]
    joined = "|".join(parts)
    return _csv_escape_field(joined)


def render_toon(
    conn: PgConnection,
    schema: str,
    db_name: str,
    include_examples: bool,
    db_dsn: str,
    column_descriptions: dict[tuple[str, str], str] | None = None,
) -> tuple[str, int, int]:
    table_names = fetch_tables(conn, schema)
    tables = [load_table(conn, schema, t, db_dsn) for t in table_names]

    include_descriptions = column_descriptions is not None
    columns = ["name", "type", "nullable", "pk"]
    if include_examples:
        columns.append("examples")

    if include_descriptions:
        columns.append("description")

    column_header = "{" + ",".join(columns) + "}"

    out: list[str] = []
    out.append("# TOON format — reading guide")
    out.append(f"# columns row fields: {column_header[1:-1]}")
    if include_examples:
        out.append("# examples field (pipe-separated, up to 5 values):")
        out.append("#   boolean          → true|false")
        out.append("#   numeric          → min|median|max")
        out.append("#   date/time        → min|max")
        out.append("#   low-cardinality (≤8 distinct) → all distinct values")
        out.append("#   FK column        → sampled values from referenced table")
        out.append("#   json/uuid/array  → one sample value")
        out.append("#   text → up to top-3 most frequent values")
        out.append("#   NULL prefix      → column contains null values")
    if include_descriptions:
        out.append("# description field → human-authored column meaning (may be empty)")

    out.append("")
    out.append(f"database: {db_name}")
    out.append("dialect: postgresql")

    out.append("")
    out.append(f"tables[{len(tables)}]:")

    total_columns = 0
    total_examples = 0

    for table in tables:
        out.append(f"  - name: {table.name}")
        if table.composite_pk:
            cols = ",".join(table.composite_pk)
            out.append(f"    primary_key{{columns}}: {cols}")

        fk_by_col = {fk.column: fk for fk in table.foreign_keys}

        out.append(f"    columns[{len(table.columns)}]{column_header}:")
        for col in table.columns:
            total_columns += 1
            examples: list[str] = []
            if include_examples:
                examples = select_examples(conn, schema, table, col, fk_by_col)
                total_examples += len(examples)
            row_fields = [
                _csv_escape_field(col.name),
                _csv_escape_field(col.data_type),
                "true" if col.nullable else "false",
                "true" if col.is_pk else "false",
            ]
            if include_examples:
                row_fields.append(_format_examples(examples))
            
            if include_descriptions:
                desc = column_descriptions.get(
                    (table.name.lower(), col.name.lower()), ""
                )
                row_fields.append(_csv_escape_field(desc))
            row = ",".join(row_fields)
            out.append(f"      {row}")

        if table.foreign_keys:
            out.append(
                f"    foreign_keys[{len(table.foreign_keys)}]"
                f"{{column,ref_table,ref_column,on_delete}}:"
            )
            for fk in table.foreign_keys:
                row = ",".join(
                    [
                        _csv_escape_field(fk.column),
                        _csv_escape_field(fk.ref_table),
                        _csv_escape_field(fk.ref_column),
                        _csv_escape_field(fk.on_delete),
                    ]
                )
                out.append(f"      {row}")

        if table.indexes:
            out.append(f"    indexes[{len(table.indexes)}]{{name,columns,unique}}:")
            for idx in table.indexes:
                cols = ";".join(idx.columns)
                row = ",".join(
                    [
                        _csv_escape_field(idx.name),
                        _csv_escape_field(cols),
                        "true" if idx.unique else "false",
                    ]
                )
                out.append(f"      {row}")

        if table.checks:
            out.append(f"    checks[{len(table.checks)}]{{name,expression}}:")
            for chk in table.checks:
                row = ",".join(
                    [
                        _csv_escape_field(chk.name),
                        _csv_escape_field(chk.expression),
                    ]
                )
                out.append(f"      {row}")

    out.append("")
    return "\n".join(out), total_columns, total_examples


def _clean_default(default: str | None) -> str:
    if default is None:
        return ""
    s = default.strip()
    m = re.match(r"^'(.*)'::[a-zA-Z_][\w ]*$", s)
    if m:
        return m.group(1)
    return s


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def open_readonly(
    host: str, port: int, user: str, password: str, dbname: str
) -> PgConnection:
    conn = psycopg2.connect(
        host=host,
        port=port,
        user=user,
        password=password,
        dbname=dbname,
    )
    conn.set_session(readonly=True, autocommit=False)
    with conn.cursor() as cur:
        cur.execute(f"SET statement_timeout = {STATEMENT_TIMEOUT_MS}")
    return conn


def process_database(
    host: str,
    port: int,
    user: str,
    password: str,
    dbname: str,
    schema: str,
    output_dir: Path,
    toon: bool,
    include_examples: bool,
) -> None:
    out_path = output_dir / f"{dbname}.txt"

    db_dsn = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
    logger.info("processing database %s (schema: %s), dsn: %s", dbname, schema, db_dsn)

    try:
        conn = open_readonly(host, port, user, password, dbname)
    except psycopg2.Error as exc:
        logger.error("[%s] connection failed: %s", dbname, exc)
        return

    try:
        if toon:
            content, n_cols, n_examples = render_toon(
                conn, schema, dbname, include_examples
            )
            n_tables = content.count("  - name: ")
            print(
                f"{dbname}: {n_tables} tables, {n_cols} columns, "
                f"{n_examples} examples extracted"
            )
        else:
            content = render_classical_ddl(conn, schema, include_examples, db_dsn)
            n_tables = content.count("CREATE TABLE ")
            print(f"{dbname}: {n_tables} tables (classical DDL)")
    except psycopg2.Error as exc:
        logger.error("[%s] extraction failed: %s", dbname, exc)
        conn.close()
        return
    finally:
        try:
            conn.close()
        except Exception:
            pass

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(content, encoding="utf-8")
    logger.info("wrote %s", out_path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract PostgreSQL DDL or TOON-formatted schema."
    )
    p.add_argument("--host", required=True)
    p.add_argument("--port", type=int, default=5432)
    p.add_argument("--user", required=True)
    p.add_argument("--password", default="123123")
    p.add_argument(
        "--databases",
        required=True,
        help="comma-separated database names",
    )
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--toon", action="store_true")
    p.add_argument("--schema", default="public")
    grp = p.add_mutually_exclusive_group()
    grp.add_argument("--include-examples", dest="examples", action="store_true")
    grp.add_argument("--no-examples", dest="examples", action="store_false")
    p.set_defaults(examples=None)
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if args.examples is None:
        include_examples = args.toon
    else:
        include_examples = args.examples

    if include_examples and not args.toon:
        logger.warning("--include-examples has no effect without --toon")

    dbs = [d.strip() for d in args.databases.split(",") if d.strip()]
    if not dbs:
        print("no databases specified", file=sys.stderr)
        return 2

    for db in dbs:
        process_database(
            host=args.host,
            port=args.port,
            user=args.user,
            password=args.password,
            dbname=db,
            schema=args.schema,
            output_dir=args.output_dir,
            toon=args.toon,
            include_examples=include_examples,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
