from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

from sqlglot import exp, parse
from sqlglot.optimizer.qualify_columns import qualify_columns

UNKNOWN_BUCKET = "unknown"


@dataclass
class Scope:
    alias_to_source: dict[str, str]
    source_columns: dict[str, set[str]]

    def resolve_source(self, name: str) -> str | None:
        return self.alias_to_source.get(name, name)


def extract_table_in_gt_sql(
    sql: str,
    table_to_columns: dict[str, set[str]],
    ddl_schema: str | None = None,
    dialect: str = "postgres",
) -> tuple[dict[str, list[str]], str | None]:
    normalized_tables = _normalize_table_to_columns(table_to_columns)
    if ddl_schema:
        ddl_tables = _table_columns_from_ddl(ddl_schema, dialect)
        normalized_tables = _merge_table_columns(normalized_tables, ddl_tables)
    schema_map = _schema_map_from_columns(normalized_tables)
    usage: dict[str, set[str]] = defaultdict(set)
    cte_columns: dict[str, set[str]] = {}

    try:
        expressions = parse(sql, read=dialect)
    except Exception as exc:
        return {}, str(exc)

    try:
        for expression in expressions:
            qualified = _qualify_expression(expression, schema_map, dialect)
            _analyze_expression(qualified, normalized_tables, usage, cte_columns)
    except Exception as exc:
        return {}, str(exc)

    return _finalize_usage(usage), None


def _normalize_table_to_columns(
    table_to_columns: dict[str, set[str]],
) -> dict[str, set[str]]:
    normalized: dict[str, set[str]] = {}
    for table, columns in table_to_columns.items():
        if not table:
            continue
        table_name = table.lower()
        normalized[table_name] = {col.lower() for col in columns}
    return normalized


def _table_columns_from_ddl(
    ddl_schema: str,
    dialect: str,
) -> dict[str, set[str]]:
    try:
        expressions = parse(ddl_schema, read=dialect)
    except Exception:
        return {}

    tables: dict[str, set[str]] = {}
    for expression in expressions:
        if not isinstance(expression, exp.Create):
            continue
        schema_expr = expression.this
        if not isinstance(schema_expr, exp.Schema):
            continue
        table_expr = schema_expr.this
        if not isinstance(table_expr, exp.Table):
            continue
        table_name = _safe_lower(table_expr.name)
        if not table_name:
            continue
        for column_def in schema_expr.expressions or []:
            if not isinstance(column_def, exp.ColumnDef):
                continue
            column_name = _safe_lower(column_def.this)
            if column_name:
                tables.setdefault(table_name, set()).add(column_name)
    return tables


def _merge_table_columns(
    base: dict[str, set[str]],
    extra: dict[str, set[str]],
) -> dict[str, set[str]]:
    merged: dict[str, set[str]] = {table: set(cols) for table, cols in base.items()}
    for table, cols in extra.items():
        merged.setdefault(table, set()).update(cols)
    return merged


def _schema_map_from_columns(
    table_columns: dict[str, set[str]],
) -> dict[str, dict[str, str]]:
    return {
        table: {column: "unknown" for column in columns}
        for table, columns in table_columns.items()
    }


def _qualify_expression(
    expression: exp.Expression,
    schema_map: dict[str, dict[str, str]],
    dialect: str,
) -> exp.Expression:
    try:
        return qualify_columns(
            expression,
            schema=schema_map,
            allow_partial_qualification=True,
            expand_stars=False,
            dialect=dialect,
        )
    except Exception:
        return expression


def _finalize_usage(usage: dict[str, set[str]]) -> dict[str, list[str]]:
    return {key: sorted(values) for key, values in usage.items() if values}


def _analyze_expression(
    expression: exp.Expression | None,
    table_to_columns: dict[str, set[str]],
    usage: dict[str, set[str]],
    cte_columns: dict[str, set[str]],
) -> None:
    if expression is None:
        return

    if isinstance(expression, exp.With):
        for cte in expression.expressions:
            _process_cte(cte, table_to_columns, usage, cte_columns)
        _analyze_expression(expression.this, table_to_columns, usage, cte_columns)
        return

    if isinstance(expression, exp.CTE):
        _process_cte(expression, table_to_columns, usage, cte_columns)
        return

    if isinstance(expression, exp.Subquery):
        _analyze_expression(expression.this, table_to_columns, usage, cte_columns)
        return

    if isinstance(expression, exp.Select):
        _analyze_select(expression, table_to_columns, usage, cte_columns)
        return

    if isinstance(expression, (exp.Union, exp.Intersect, exp.Except)):
        _analyze_expression(expression.left, table_to_columns, usage, cte_columns)
        _analyze_expression(expression.right, table_to_columns, usage, cte_columns)
        return

    for child in expression.iter_expressions():
        _analyze_expression(child, table_to_columns, usage, cte_columns)


def _process_cte(
    cte: exp.CTE,
    table_to_columns: dict[str, set[str]],
    usage: dict[str, set[str]],
    cte_columns: dict[str, set[str]],
) -> None:
    cte_name = _safe_lower(cte.alias_or_name)
    if not cte_name:
        return

    _analyze_expression(cte.this, table_to_columns, usage, cte_columns)
    cte_columns[cte_name] = _collect_output_columns(
        cte.this, table_to_columns, cte_columns
    )


def _collect_output_columns(
    expression: exp.Expression | None,
    table_to_columns: dict[str, set[str]],
    cte_columns: dict[str, set[str]],
) -> set[str]:
    if expression is None:
        return set()

    if isinstance(expression, exp.Subquery):
        return _collect_output_columns(expression.this, table_to_columns, cte_columns)

    if isinstance(expression, exp.With):
        for cte in expression.expressions:
            _process_cte(cte, table_to_columns, defaultdict(set), cte_columns)
        return _collect_output_columns(expression.this, table_to_columns, cte_columns)

    if isinstance(expression, exp.Select):
        with_expr = expression.args.get("with_")
        if with_expr:
            for cte in with_expr.expressions:
                _process_cte(cte, table_to_columns, defaultdict(set), cte_columns)
        scope = _build_scope(expression, table_to_columns, cte_columns)
        return _select_output_columns(expression, scope, cte_columns)

    if isinstance(expression, (exp.Union, exp.Intersect, exp.Except)):
        left = _collect_output_columns(expression.left, table_to_columns, cte_columns)
        right = _collect_output_columns(expression.right, table_to_columns, cte_columns)
        return left | right

    return set()


def _build_scope(
    select: exp.Select,
    table_to_columns: dict[str, set[str]],
    cte_columns: dict[str, set[str]],
) -> Scope:
    alias_to_source: dict[str, str] = {}
    source_columns: dict[str, set[str]] = {}

    for source in _iter_from_sources(select):
        if isinstance(source, exp.Table):
            source_name = _safe_lower(source.name)
            if not source_name:
                continue
            alias = _safe_lower(source.alias_or_name) or source_name
            alias_to_source[alias] = source_name
            alias_to_source[source_name] = source_name
            if source_name in cte_columns:
                source_columns[source_name] = set(cte_columns[source_name])
            elif source_name in table_to_columns:
                source_columns[source_name] = set(table_to_columns[source_name])
        elif isinstance(source, exp.Subquery):
            alias = _safe_lower(source.alias_or_name)
            if alias:
                alias_to_source[alias] = alias

    return Scope(alias_to_source=alias_to_source, source_columns=source_columns)


def _iter_from_sources(select: exp.Select) -> Iterable[exp.Expression]:
    from_expr = select.args.get("from_")
    if from_expr:
        if from_expr.expressions:
            for expr in from_expr.expressions:
                yield expr
        elif from_expr.this is not None:
            yield from_expr.this
    for join in select.args.get("joins") or []:
        if join.this is not None:
            yield join.this


def _analyze_select(
    select: exp.Select,
    table_to_columns: dict[str, set[str]],
    usage: dict[str, set[str]],
    cte_columns: dict[str, set[str]],
) -> None:
    with_expr = select.args.get("with_")
    if with_expr:
        for cte in with_expr.expressions:
            _process_cte(cte, table_to_columns, usage, cte_columns)

    scope = _build_scope(select, table_to_columns, cte_columns)

    for projection in select.expressions:
        if _is_star_projection(projection):
            _record_star_usage(projection, scope, usage, cte_columns)

    join_hints = _collect_join_column_hints(select, scope, cte_columns)

    for column in _iter_columns(select):
        if _is_star_projection(column):
            continue
        _record_column_usage(column, scope, usage, cte_columns, join_hints)

    for join in select.args.get("joins") or []:
        using = join.args.get("using")
        if using is None:
            continue
        if isinstance(using, list):
            identifiers = using
        elif hasattr(using, "expressions"):
            identifiers = using.expressions
        else:
            identifiers = [using]
        for identifier in identifiers:
            name = _safe_lower(getattr(identifier, "name", None))
            if name:
                _record_unqualified_column(name, scope, usage, cte_columns, join_hints)

    for subquery in select.find_all(exp.Subquery):
        _analyze_expression(subquery.this, table_to_columns, usage, cte_columns)


def _iter_columns(select: exp.Select) -> Iterable[exp.Column]:
    for expression in select.iter_expressions():
        yield from _iter_columns_inner(expression)


def _iter_columns_inner(expression: exp.Expression | None) -> Iterable[exp.Column]:
    if expression is None:
        return
    if isinstance(
        expression,
        (
            exp.Subquery,
            exp.Select,
            exp.With,
            exp.CTE,
            exp.Union,
            exp.Intersect,
            exp.Except,
        ),
    ):
        return
    if isinstance(expression, exp.Column):
        yield expression
    for child in expression.iter_expressions():
        yield from _iter_columns_inner(child)


def _is_star_projection(expression: exp.Expression) -> bool:
    if isinstance(expression, exp.Star):
        return True
    if isinstance(expression, exp.Column) and expression.is_star:
        return True
    if isinstance(expression, exp.Alias):
        alias_this = expression.this
        if isinstance(alias_this, exp.Star):
            return True
        if isinstance(alias_this, exp.Column) and alias_this.is_star:
            return True
    return False


def _record_star_usage(
    expression: exp.Expression,
    scope: Scope,
    usage: dict[str, set[str]],
    cte_columns: dict[str, set[str]],
) -> None:
    table_name = _star_table_name(expression)
    if table_name:
        source = scope.resolve_source(table_name)
        if source is None:
            return
        bucket = _bucket_name(source, cte_columns)
        for col in scope.source_columns.get(source, set()):
            usage[bucket].add(col)
        return

    for source, columns in scope.source_columns.items():
        bucket = _bucket_name(source, cte_columns)
        for col in columns:
            usage[bucket].add(col)


def _record_column_usage(
    column: exp.Column,
    scope: Scope,
    usage: dict[str, set[str]],
    cte_columns: dict[str, set[str]],
    join_hints: dict[str, set[str]] | None = None,
) -> None:
    column_name = _safe_lower(column.name)
    if not column_name:
        return

    table_name = _safe_lower(column.table)
    if table_name:
        source = scope.resolve_source(table_name) or table_name
        if source in scope.source_columns or source in cte_columns:
            bucket = _bucket_name(source, cte_columns)
        else:
            bucket = UNKNOWN_BUCKET
        usage[bucket].add(column_name)
        return

    _record_unqualified_column(column_name, scope, usage, cte_columns, join_hints)


def _record_unqualified_column(
    column_name: str,
    scope: Scope,
    usage: dict[str, set[str]],
    cte_columns: dict[str, set[str]],
    join_hints: dict[str, set[str]] | None = None,
) -> None:
    candidates = []
    for source, columns in scope.source_columns.items():
        if column_name in columns:
            candidates.append(source)

    if len(candidates) == 1:
        bucket = _bucket_name(candidates[0], cte_columns)
        usage[bucket].add(column_name)
        return

    if candidates and join_hints:
        hinted = [
            source
            for source in candidates
            if source in join_hints.get(column_name, set())
        ]
        if len(hinted) == 1:
            bucket = _bucket_name(hinted[0], cte_columns)
            usage[bucket].add(column_name)
            return
        if len(hinted) > 1:
            for source in hinted:
                bucket = _bucket_name(source, cte_columns)
                usage[bucket].add(column_name)
            return

    usage[UNKNOWN_BUCKET].add(column_name)


def _collect_join_column_hints(
    select: exp.Select,
    scope: Scope,
    cte_columns: dict[str, set[str]],
) -> dict[str, set[str]]:
    hints: dict[str, set[str]] = defaultdict(set)
    for join in select.args.get("joins") or []:
        on_expr = join.args.get("on")
        if on_expr is not None:
            for col in on_expr.find_all(exp.Column):
                column_name = _safe_lower(col.name)
                table_name = _safe_lower(col.table)
                if not column_name or not table_name:
                    continue
                source = scope.resolve_source(table_name) or table_name
                if source in scope.source_columns or source in cte_columns:
                    hints[column_name].add(source)

        using = join.args.get("using")
        if using is None:
            continue
        if isinstance(using, list):
            identifiers = using
        elif hasattr(using, "expressions"):
            identifiers = using.expressions
        else:
            identifiers = [using]
        for identifier in identifiers:
            name = _safe_lower(getattr(identifier, "name", None))
            if not name:
                continue
            for source, columns in scope.source_columns.items():
                if name in columns:
                    hints[name].add(source)

    return hints


def _select_output_columns(
    select: exp.Select,
    scope: Scope,
    cte_columns: dict[str, set[str]],
) -> set[str]:
    output: set[str] = set()
    for projection in select.expressions:
        if _is_star_projection(projection):
            output |= _expand_star_columns(projection, scope)
            continue

        alias = _safe_lower(getattr(projection, "alias_or_name", None))
        if alias:
            output.add(alias)
            continue

        if isinstance(projection, exp.Alias):
            alias = _safe_lower(getattr(projection, "alias", None))
            if alias:
                output.add(alias)
            continue

        if isinstance(projection, exp.Column):
            name = _safe_lower(projection.name)
            if name:
                output.add(name)

    return output


def _expand_star_columns(expression: exp.Expression, scope: Scope) -> set[str]:
    table_name = _star_table_name(expression)
    if table_name:
        source = scope.resolve_source(table_name)
        if source is None:
            return set()
        return set(scope.source_columns.get(source, set()))

    columns: set[str] = set()
    for source_columns in scope.source_columns.values():
        columns |= source_columns
    return columns


def _star_table_name(expression: exp.Expression) -> str | None:
    if isinstance(expression, exp.Star):
        return _safe_lower(getattr(expression, "this", None))

    if isinstance(expression, exp.Alias):
        return _star_table_name(expression.this)

    if isinstance(expression, exp.Column) and expression.is_star:
        return _safe_lower(expression.table)

    return None


def _bucket_name(source: str, cte_columns: dict[str, set[str]]) -> str:
    source_name = _safe_lower(source)
    if not source_name:
        return UNKNOWN_BUCKET
    if source_name in cte_columns:
        return f"cte_{source_name}"
    return source_name


def _safe_lower(value: str | exp.Expression | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, exp.Identifier):
        return value.name.lower()
    if isinstance(value, exp.Expression):
        name = getattr(value, "name", None)
        if name:
            return str(name).lower()
    return str(value).lower()
