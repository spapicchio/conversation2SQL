import re

_FENCED_SQL_RE = re.compile(
    r"```(?:sql)?\s*\n(.*?)\n```",
    re.IGNORECASE | re.DOTALL,
)

_FENCE_OPEN_RE = re.compile(r"```(?:sql)?\s*\n", re.IGNORECASE)

_SQL_KEYWORD_RE = re.compile(
    r"^[ \t]*(SELECT|WITH|INSERT|UPDATE|DELETE|CREATE|DROP|ALTER|EXPLAIN)\b",
    re.IGNORECASE | re.MULTILINE,
)


def _extract_closed_fence(text: str) -> str | None:
    matches = _FENCED_SQL_RE.findall(text)
    for block in reversed(matches):
        stripped = block.strip()
        if stripped:
            return stripped
    return None


def _extract_unclosed_fence(text: str) -> str | None:
    openings = list(_FENCE_OPEN_RE.finditer(text))
    for m in reversed(openings):
        after = text[m.end() :]
        if "```" in after:
            continue  # this opener has a closer — handled by _extract_closed_fence
        semi_pos = after.find(";")
        candidate = after[: semi_pos + 1].strip() if semi_pos != -1 else after.strip()
        return candidate if candidate else None
    return None


def _extract_raw_sql(text: str) -> str | None:
    matches = list(_SQL_KEYWORD_RE.finditer(text))
    if not matches:
        return None
    first = matches[0]
    after = text[first.start():]
    # Don't cross into a later code fence (e.g. an LLM "let me retry" block).
    fence_pos = after.find("```")
    region = after[:fence_pos] if fence_pos != -1 else after
    # rfind so CTEs (WITH … SELECT … ;) are captured whole, not just the tail.
    semi_pos = region.rfind(";")
    candidate = region[:semi_pos + 1].strip() if semi_pos != -1 else region.strip()
    return candidate if candidate else None


def extract_sql_from_response(text: str) -> str | None:
    return (
        _extract_closed_fence(text)
        or _extract_unclosed_fence(text)
        or _extract_raw_sql(text)
    )
