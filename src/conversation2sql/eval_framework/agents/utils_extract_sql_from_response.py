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


if __name__ == "__main__":
    # Example usage
    response = """
    Here is the SQL query you requested:

    ```sql
    WITH observatory_data AS (
    SELECT 
        o.observstation,
        o.atmostransparency as atmostransparency,
        o.humidityrate,
        o.windspeedms,
        o.lunardistdeg,
        o.equipstatus,
        o.calibrstatus,
        o.coolsysstatus,
        t.telescregistry as teles_id,
        t.pointaccarc
        FROM observatories o
    JOIN telescopes t ON o.observstation = t.observstation
),
calculated_metrics AS (
    SELECT 
        *,
        -- Calculate AOI
        atmostransparency * (1 - humidityrate / 100.0) * (1 - 0.02 * windspeedms) as aoi,
        -- Calculate LIF
        (1 - lunardistdeg / 180.0) * (1 - atmostransparency) as lif,
        -- Calculate OQF
        atmostransparency * (1 - humidityrate / 100.0) * (1 - 0.02 * windspeedms) * 
        (1 - (1 - lunardistdeg / 180.0) * (1 - atmostransparency)) *
        CASE WHEN pointaccarc < 2 THEN 1 ELSE 2.0 / pointaccarc END as oqf,
        -- Identify equipment problems
        CASE 
            WHEN equipstatus != 'Operational' THEN CONCAT('\n', equipstatus || ' is not Operational')
            ELSE ''
        END ||
        CASE 
            WHEN calibrstatus != 'Current' THEN COALESCE(
                CASE WHEN equipped_problems != '' THEN ',' ELSE '' END, ''
            ) || '\n', calibrstatus || ' is not Current')
            ELSE ''
        END ||
        CASE 
            WHEN coolsysstatus != 'Normal' THEN COALESCE(
                CASE WHEN equipped_problems != '' THEN ',' ELSE '' END, ''
            ) || '\n', coolsysstatus || ' is not Normal')
            ELSE ''
        END as equipped_problems_detail
        FROM observatories o
    WHERE true  -- Placeholder for the full query structure
)
SELECT DISTINCT t.observation_quality_factor, t.observatory_name, t.itel, etc
FROM telescope_behavior_status t JOINDEMONZ_etc
WHERE NOT IN (AII) ETC
ORDER BY cte.observation_quality FACT DESC;


Okay, I'll start over with a cleaner approach:```sql
    """

    sql = extract_sql_from_response(response)
    print(sql)