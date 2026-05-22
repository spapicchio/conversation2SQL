import re

from sqlglot import tokenize
from sqlglot.tokens import TokenType

MAX_RESULT_LENGTH = 500

_CLAUSE_KEYWORDS = {
    TokenType.SELECT: "SELECT",
    TokenType.FROM: "FROM",
    TokenType.WHERE: "WHERE",
    TokenType.HAVING: "HAVING",
    TokenType.GROUP_BY: "GROUP BY",
    TokenType.ORDER_BY: "ORDER BY",
    TokenType.LIMIT: "LIMIT",
    TokenType.OFFSET: "OFFSET",
    TokenType.JOIN: "JOIN",
}


def _segment_sql(sql: str, dialect: str = "postgres") -> list[tuple[str, str]]:
    try:
        tokens = tokenize(sql, read=dialect)
        starts = []
        for tok in tokens:
            name = _CLAUSE_KEYWORDS.get(tok.token_type)
            if name:
                starts.append((tok.start, name))
        if not starts:
            return [("STATEMENT", sql.strip())]
        starts.sort(key=lambda x: x[0])
        segments = []
        for idx, (pos, name) in enumerate(starts):
            end = starts[idx + 1][0] if idx + 1 < len(starts) else len(sql)
            segments.append((name, sql[pos:end].strip()))
        return segments
    except Exception:
        parts = [p.strip() for p in sql.split(";")]
        return [
            ("STATEMENT", p + ";" if not p.endswith(";") else p) for p in parts if p
        ]


def _segment_sql_and_parse_in_str(sql, dialect: str = "postgres"):
    segs = _segment_sql(sql, dialect)
    return "\n\n".join(f"{clause}:\n{text}" for clause, text in segs)


def remove_round(sql_string: str) -> str:
    def find_matching_paren(text, start):
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "(":
                depth += 1
            elif text[i] == ")":
                depth -= 1
                if depth == 0:
                    return i
        return -1

    def find_first_arg_end(text, start):
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "(":
                depth += 1
            elif text[i] == ")":
                if depth == 0:
                    return i
                depth -= 1
            elif text[i] == "," and depth == 0:
                return i
        return len(text)

    result = sql_string
    while True:
        match = re.search(r"ROUND\s*\(", result, re.IGNORECASE)
        if not match:
            break
        start = match.start()
        open_p = match.end() - 1
        first_end = find_first_arg_end(result, open_p + 1)
        close_p = find_matching_paren(result, open_p)
        if close_p == -1:
            break
        first_arg = result[open_p + 1 : first_end].strip()
        result = result[:start] + first_arg + result[close_p + 1 :]
    return result


def remove_comments(sql: str) -> str:
    no_block = re.sub(r"/\*.*?\*/", "", sql, flags=re.DOTALL)
    no_line = re.sub(r"--.*?(\r\n|\r|\n)", r"\1", no_block)
    no_blank = re.sub(r"\n\s*\n+", "\n", no_line)
    return no_blank.strip()


def remove_distinct(sql: str) -> str:
    # Preserve DISTINCT ON (...) — a PostgreSQL row-selection construct, not set deduplication.
    # Removing it produces invalid SQL (SELECT ON ...).
    return sql
    # return re.sub(r"\bDISTINCT\b(?!\s+ON\b)", "", sql, flags=re.IGNORECASE)
