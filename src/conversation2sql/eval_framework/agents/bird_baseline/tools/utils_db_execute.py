import json
from datetime import date, datetime
from decimal import Decimal, ROUND_HALF_UP
from typing import Any

import psycopg2
import psycopg2.extensions
import psycopg2.extras
from psycopg2.extensions import Column
from psycopg2.extras import RealDictRow

# Result truncation is row-based, not character-based: the agent issues many SQL
# calls per task, so we show only the first few *whole* rows and tell it how many
# more existed. Width is intentionally left unbounded so execute_sql and
# psql_console output stay comparable (neither caps cell/row width).
MAX_RESULT_ROWS = 3

# _execute_query fetches at most this many rows; when a result hits the cap the
# true total is unknown, so the "more rows" note reports it as "<limit>+".
RESULT_FETCH_LIMIT = 10_000


def _more_rows_note(total_str: str) -> str:
    """One-line note appended when a result is truncated to ``MAX_RESULT_ROWS``.

    States that the query *succeeded* (so the agent does not mistake the cut for
    a failure and waste bird-coins re-running it) and how to see more.
    ``total_str`` is the caller-formatted total (an exact count, or
    ``"<limit>+"`` when the fetch cap was hit).
    """
    return (
        f"\n... [showing first {MAX_RESULT_ROWS} of {total_str} rows; "
        "the query ran successfully — add a LIMIT or select fewer columns to see more]"
    )


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
            rows = cursor.fetchmany(RESULT_FETCH_LIMIT + 1)
            result = rows[:RESULT_FETCH_LIMIT]
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


def _format_cell(value: Any) -> str:
    """Render one cell for the text table.

    Container values (JSON/array/composite columns come back as ``list``/``dict``)
    are serialised as *compact* JSON rather than Python ``repr`` so the agent sees
    valid, deterministic, double-quoted JSON (``{"aoi":0.0146324,...}``) instead of
    single-quoted ``repr`` padding — shorter and parseable. Full numeric precision
    is preserved on purpose: ``execute_sql`` is an inspection tool, so rounding
    (which lives in ``preprocess_results`` for the submit-time comparison) would
    hide values the agent needs to verify its query. The cell is **not** width-capped
    (see ``MAX_RESULT_ROWS``): truncation is by whole rows, not characters.
    """
    if isinstance(value, (dict, list)):
        return json.dumps(value, default=str, separators=(",", ":"), sort_keys=True)
    return str(value)


def _format_result(result: list, cursor_desc: tuple[Column, ...], max_rows: int = MAX_RESULT_ROWS) -> str:
    """Render the result set as a GitHub-flavored markdown table.

    Output:

    | sitekey | sitelabel |
    | --- | --- |
    | SP9227 | Solar Plant West Davidport |
    | SP6740 | Solar Plant Dillonmouth |
    | SP7738 | Solar Plant North Xavier |
    | SP7778 | Solar Plant East Alexandriaborough |
    | SP9784 | Solar Plant East Jake |
    | SP6230 | Solar Plant Gatesview |
    | SP6166 | Solar Plant Jacksonport |
    | SP9766 | Solar Plant Evanmouth |
    | SP1937 | Solar Plant Brittanybury |
    | SP6929 | Solar Plant Lake Kathrynburgh |

    result = [RealDictRow({'sitekey': 'SP9227', 'sitelabel': 'Solar Plant West Davidport'})]

    cursor_desc = (Column(name='sitekey', type_code=25), Column(name='sitelabel', type_code=25))
    """

    if result is None:
        return "Query executed successfully."

    if len(result) == 0:
        return "Query executed, empty result set."

    cols = [desc[0] for desc in cursor_desc]
    header = "| " + " | ".join(cols) + " |"
    separator = "| " + " | ".join("---" for _ in cols) + " |"

    # Keep only the first max_rows whole rows so repeated calls don't flood the
    # agent's context; a note below states how many rows really matched.
    rows = [
        "| " + " | ".join(_format_cell(row[col]) for col in cols) + " |"
        for row in result[:max_rows]
    ]
    table = "\n".join([header, separator, *rows])

    if len(result) > max_rows:
        total = len(result)
        total_str = f"{RESULT_FETCH_LIMIT}+" if total >= RESULT_FETCH_LIMIT else str(total)
        table += _more_rows_note(total_str)
    return table
