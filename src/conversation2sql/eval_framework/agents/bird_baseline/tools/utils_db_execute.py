import json
from datetime import date, datetime
from decimal import Decimal, ROUND_HALF_UP
from typing import Any

import psycopg2
import psycopg2.extensions
import psycopg2.extras
from psycopg2.extensions import Column
from psycopg2.extras import RealDictRow


def _connect(db_dsn: str) -> psycopg2.extensions.connection:
    conn = psycopg2.connect(db_dsn, cursor_factory=psycopg2.extras.RealDictCursor)
    conn.set_session(readonly=True, autocommit=True)
    return conn
    

def _execute_query(query: str, db_dsn: str) -> tuple[list[RealDictRow], tuple[Column]]:
    conn = _connect(db_dsn)
    cursor = conn.cursor()
    # set timeout to prevent hanging if the agent generates a bad query
    #  https://github.com/bird-bench/BIRD-Interact/blob/451fe2c3518ee1cf908d8139e2913483bd519381/BIRD-Interact-ADK/shared/db_utils.py#L50
    cursor.execute("SET statement_timeout = '60s';")
    try:
        cursor.execute(query)
        conn.commit()
        lower_q = query.strip().lower()
        if lower_q.startswith("select") or lower_q.startswith("with"):
            rows = cursor.fetchmany(10_000 + 1)
            result = rows[:10_000]
        else:
            try:
                result = cursor.fetchall()

            except psycopg2.ProgrammingError:
                result = None

        desc = cursor.description
        return result, desc # pyrefly: ignore

    except psycopg2.DatabaseError as e:
        conn.rollback()
        raise e

    finally:
        cursor.close()
        conn.close()


def process_decimals_recursive(item, decimal_places: int):
    """
    - quantizer = Decimal(1).scaleb(-decimal_places) — builds the rounding step as an exact Decimal (e.g., 0.01 for 2 places).
    - Decimal branch — uses .quantize(..., ROUND_HALF_UP) for exact rounding (psycopg2 returns NUMERIC columns as Decimal).
    - float branch — falls back to builtin round() since floats can't use Decimal.quantize.
    - list/tuple branch — recurses and rebuilds with the same container type via type(item)(...).
    - dict branch — recurses into values; keys are left as-is.
    - Fallthrough — anything else (str, None, dates, bool) passes through unchanged.
    """

    # Build the rounding step as a Decimal (e.g. decimal_places=2 -> Decimal("0.01")) once
    # per top-level call. Using Decimal here keeps the precision exact for `quantize` below;
    # the inner recursion reuses this quantizer rather than rebuilding it at every level.
    quantizer = Decimal(1).scaleb(-decimal_places)

    def _process(node):
        if isinstance(node, Decimal):
            # psycopg2 returns NUMERIC columns as Decimal — quantize for exact, banker-safe rounding.
            return node.quantize(quantizer, rounding=ROUND_HALF_UP)
        elif isinstance(node, float):
            # Plain floats can't use Decimal.quantize; fall back to builtin round().
            return round(node, decimal_places)
        elif isinstance(node, (list, tuple)):
            # Recurse into sequences and rebuild with the same container type (list stays list, tuple stays tuple).
            return type(node)(_process(x) for x in node)
        elif isinstance(node, dict):
            # Recurse into dict values; keys are left untouched since they aren't numeric payload.
            return {k: _process(v) for k, v in node.items()}
        # Non-numeric, non-container values (str, None, bool, dates, ...) pass through unchanged.
        return node

    return _process(item)


def preprocess_results(
        results: list[RealDictRow],
        cursor_desc: tuple[Column, ...],
        decimal_places: int = 2,
) -> list[tuple[Any, ...]]:
    cols = [desc[0] for desc in cursor_desc]

    processed = []
    for row in results:
        processed_row = []
        for col in cols:
            value = row[col]
            if isinstance(value, (date, datetime)):
                processed_row.append(value.strftime("%Y-%m-%d"))
            else:
                pi = process_decimals_recursive(value, decimal_places)
                if isinstance(pi, (dict, list)):
                    processed_row.append(json.dumps(pi, sort_keys=True))
                else:
                    processed_row.append(pi)
        processed.append(tuple(processed_row))
    return processed


def _format_cell(value: Any, max_characters: int) -> str:
    """Render one cell for the text table.

    Container values (JSON/array/composite columns come back as ``list``/``dict``)
    are serialised as *compact* JSON rather than Python ``repr`` so the agent sees
    valid, deterministic, double-quoted JSON (``{"aoi":0.0146324,...}``) instead of
    single-quoted ``repr`` padding — shorter and parseable. Full numeric precision
    is preserved on purpose: ``execute_sql`` is an inspection tool, so rounding
    (which lives in ``preprocess_results`` for the submit-time comparison) would
    hide values the agent needs to verify its query.
    """
    if isinstance(value, (dict, list)):
        s = json.dumps(value, default=str, separators=(",", ":"), sort_keys=True)
    else:
        s = str(value)
    return s[:max_characters]


def _format_result(result: list, cursor_desc: tuple[Column, ...], max_characters=100) -> str:
    """
    Output:

    sitekey | sitelabel
    SP9227 | Solar Plant West Davidport
    SP6740 | Solar Plant Dillonmouth
    SP7738 | Solar Plant North Xavier
    SP7778 | Solar Plant East Alexandriaborough
    SP9784 | Solar Plant East Jake
    SP6230 | Solar Plant Gatesview
    SP6166 | Solar Plant Jacksonport
    SP9766 | Solar Plant Evanmouth
    SP1937 | Solar Plant Brittanybury
    SP6929 | Solar Plant Lake Kathrynburgh

    result = [RealDictRow({'sitekey': 'SP9227', 'sitelabel': 'Solar Plant West Davidport'})]

    cursor_desc = (Column(name='sitekey', type_code=25), Column(name='sitelabel', type_code=25))
    """

    if result is None:
        return "Query executed successfully."

    if len(result) == 0:
        return "Query executed, empty result set."

    cols = [desc[0] for desc in cursor_desc]
    header = " | ".join(cols)

    # take the first 100 rows to avoid overwhelming the output, and truncate each cell to max_characters chars
    rows = [
        " | ".join(_format_cell(row[col], max_characters) for col in cols)
        for row in result[:100]
    ]

    # No separator rule: it carries no information for the model and a dash line
    # padded to row width wastes the limited budget enforced downstream.
    return "\n".join([header, *rows])
