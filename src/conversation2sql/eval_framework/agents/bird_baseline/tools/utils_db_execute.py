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


def _execute_query(query: str, db_dsn: str) -> Any:
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
        return result, desc

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

    # Build the rounding step as a Decimal (e.g. decimal_places=2 -> Decimal("0.01")).
    # Using Decimal here keeps the precision exact, which matters for `quantize` below.
    quantizer = Decimal(1).scaleb(-decimal_places)
    if isinstance(item, Decimal):
        # psycopg2 returns NUMERIC columns as Decimal — quantize for exact, banker-safe rounding.
        return item.quantize(quantizer, rounding=ROUND_HALF_UP)
    elif isinstance(item, float):
        # Plain floats can't use Decimal.quantize; fall back to builtin round().
        return round(item, decimal_places)
    elif isinstance(item, (list, tuple)):
        # Recurse into sequences and rebuild with the same container type (list stays list, tuple stays tuple).
        return type(item)(process_decimals_recursive(x, decimal_places) for x in item)
    elif isinstance(item, dict):
        # Recurse into dict values; keys are left untouched since they aren't numeric payload.
        return {
            k: process_decimals_recursive(v, decimal_places) for k, v in item.items()
        }
    # Non-numeric, non-container values (str, None, bool, dates, ...) pass through unchanged.
    return item


def preprocess_results(
        results: list[RealDictRow] | None,
        cursor_desc: tuple[Column],
        decimal_places: int = 2,
):
    if results is None:
        return None
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


def _format_result(result: list, cursor_desc: tuple[Column]) -> str:
    """
    Output:

    sitekey | sitelabel
    -------------------
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

    # result = preprocess_results(result, cursor_desc)
    if result is None:
        return "Query executed successfully."

    if len(result) == 0:
        return "Query executed, empty result set."

    cols = [desc[0] for desc in cursor_desc]
    header = " | ".join(cols)

    # take the first 100 rows to avoid overwhelming the output, and truncate each cell to 100 chars
    rows = [" | ".join(
        str(row[col])[:100] for col in cols
    )
        for row in result[:100]
    ]

    separator = "-" * min(max(len(header), *(len(r) for r in rows)), 200)
    return "\n".join([header, separator, *rows])


if __name__ == "__main__":
    sql = 'SELECT ROUND(CAST(om."mttrh" / (om."mtbfh" + om."mttrh") AS numeric), 4)\nFROM operational_metrics om\nJOIN plant_record pr ON om."snapops" = pr."snapkey"\nJOIN plants p ON pr."sitetie" = p."sitekey"\nWHERE LOWER(p."sitelabel") = \'solar plant west davidport\'\nLIMIT 1;'
    db_dsn = "postgresql://root:123123@localhost:5433/solar_panel"
    exec_query, cur = _execute_query(sql, db_dsn)
    # formatted = _format_result(exec_query, cur)
    print(exec_query[0][cur[0][0]] is None)  # print the value of the first column in the first row
