import json
from datetime import date, datetime
from decimal import Decimal, ROUND_HALF_UP
from typing import Any

import psycopg2
import psycopg2.extensions

from conversation2sql.eval_framework.agent.tools.utils import MAX_RESULT_LENGTH


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


def _format_result(result, cursor_desc=None) -> str:
    if result is None:
        return "Query executed successfully."

    if not isinstance(result, list):
        return str(result)

    if not result:
        return "Query executed, empty result set."

    lines = []

    if cursor_desc:
        cols = [desc[0] for desc in cursor_desc]
        lines.append(" | ".join(cols))
        lines.append("-" * min(len(lines[0]), 200))

    for row in result[:100]:
        cells = [str(c)[:100] for c in row]
        lines.append(" | ".join(cells))

    text = "\n".join(lines)
    words = text.split()

    if len(words) > MAX_RESULT_LENGTH:
        text = " ".join(words[:MAX_RESULT_LENGTH]) + "..."

    return text


def process_decimals_recursive(item, decimal_places: int):
    quantizer = Decimal(1).scaleb(-decimal_places)
    if isinstance(item, Decimal):
        return item.quantize(quantizer, rounding=ROUND_HALF_UP)
    elif isinstance(item, float):
        return round(item, decimal_places)
    elif isinstance(item, (list, tuple)):
        return type(item)(process_decimals_recursive(x, decimal_places) for x in item)
    elif isinstance(item, dict):
        return {k: process_decimals_recursive(v, decimal_places) for k, v in item.items()}
    return item


def preprocess_results(results, decimal_places: int = 2):
    if results is None:
        return []
    processed = []
    for row in results:
        processed_row = []
        for item in row:
            if isinstance(item, (date, datetime)):
                processed_row.append(item.strftime("%Y-%m-%d"))
            else:
                pi = process_decimals_recursive(item, decimal_places)
                if isinstance(pi, (dict, list)):
                    processed_row.append(json.dumps(pi, sort_keys=True))
                else:
                    processed_row.append(pi)
        processed.append(tuple(processed_row))
    return processed
