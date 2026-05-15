# extract_sql_from_response Robustness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the single closed-fence regex in `extract_sql_from_response` with a priority chain of three strategies that also recover SQL from unclosed fences and raw keyword-started lines.

**Architecture:** Three private functions (`_extract_closed_fence`, `_extract_unclosed_fence`, `_extract_raw_sql`) are tried in order; the public function returns the first non-`None` non-empty result. All code stays in `baseline_model.py`. Public API and call sites are unchanged.

**Tech Stack:** Python 3.12, `re` stdlib, `pytest` via `uv run pytest`

---

## File Map

- Modify: `src/conversation2sql/eval_framework/agents/no_tool_baseline/baseline_model.py`
- Modify: `tests/eval_framework/agents/test_no_tool_baseline.py`

---

### Task 1: Refactor closed-fence logic into `_extract_closed_fence`

Move the existing regex + loop into a private function and wire the dispatcher. All existing tests must continue to pass — no new behaviour yet.

**Files:**
- Modify: `src/conversation2sql/eval_framework/agents/no_tool_baseline/baseline_model.py`

- [ ] **Step 1: Run the existing tests to establish a green baseline**

```bash
uv run pytest tests/eval_framework/agents/test_no_tool_baseline.py::TestExtractSqlFromResponse -v
```

Expected: all 7 tests PASS.

- [ ] **Step 2: Refactor `baseline_model.py`**

Replace the current `_FENCED_SQL_RE` + `extract_sql_from_response` block with:

```python
_FENCED_SQL_RE = re.compile(
    r"```(?:sql)?\s*\n?(.*?)\n?```",
    re.IGNORECASE | re.DOTALL,
)


def _extract_closed_fence(text: str) -> str | None:
    matches = _FENCED_SQL_RE.findall(text)
    for block in reversed(matches):
        stripped = block.strip()
        if stripped:
            return stripped
    return None


def extract_sql_from_response(text: str) -> str | None:
    return _extract_closed_fence(text)
```

- [ ] **Step 3: Run existing tests — must still be green**

```bash
uv run pytest tests/eval_framework/agents/test_no_tool_baseline.py::TestExtractSqlFromResponse -v
```

Expected: all 7 tests PASS.

- [ ] **Step 4: Commit**

```bash
git add src/conversation2sql/eval_framework/agents/no_tool_baseline/baseline_model.py
git commit -m "refactor: extract _extract_closed_fence from extract_sql_from_response"
```

---

### Task 2: Add `_extract_unclosed_fence` with TDD

**Files:**
- Modify: `tests/eval_framework/agents/test_no_tool_baseline.py`
- Modify: `src/conversation2sql/eval_framework/agents/no_tool_baseline/baseline_model.py`

- [ ] **Step 1: Write failing tests**

Append to `TestExtractSqlFromResponse` in `tests/eval_framework/agents/test_no_tool_baseline.py`:

```python
def test_unclosed_fence_with_semicolon(self):
    text = "```sql\nSELECT 1;"
    assert extract_sql_from_response(text) == "SELECT 1;"

def test_unclosed_fence_without_semicolon(self):
    text = "```sql\nSELECT 1"
    assert extract_sql_from_response(text) == "SELECT 1"

def test_unclosed_fence_stops_at_first_semicolon(self):
    text = "```sql\nSELECT 1;\nSELECT 2"
    assert extract_sql_from_response(text) == "SELECT 1;"

def test_closed_fence_takes_priority_over_unclosed(self):
    # closed block first, then an unclosed block — closed wins
    text = "```sql\nSELECT 1;\n```\n```sql\nSELECT 2;"
    assert extract_sql_from_response(text) == "SELECT 1;"
```

- [ ] **Step 2: Run to confirm they fail**

```bash
uv run pytest tests/eval_framework/agents/test_no_tool_baseline.py::TestExtractSqlFromResponse -v -k "unclosed"
```

Expected: the 3 unclosed-fence tests FAIL (return `None`). `test_closed_fence_takes_priority_over_unclosed` may already PASS — that's fine, the closed-fence strategy covers it.

- [ ] **Step 3: Add `_FENCE_OPEN_RE` and `_extract_unclosed_fence` to `baseline_model.py`**

Add after `_FENCED_SQL_RE`:

```python
_FENCE_OPEN_RE = re.compile(r"```(?:sql)?\s*\n", re.IGNORECASE)
```

Add after `_extract_closed_fence`:

```python
def _extract_unclosed_fence(text: str) -> str | None:
    openings = list(_FENCE_OPEN_RE.finditer(text))
    for m in reversed(openings):
        after = text[m.end():]
        if "```" in after:
            continue  # this opener has a closer — handled by _extract_closed_fence
        semi_pos = after.find(";")
        candidate = after[: semi_pos + 1].strip() if semi_pos != -1 else after.strip()
        return candidate if candidate else None
    return None
```

Update the dispatcher:

```python
def extract_sql_from_response(text: str) -> str | None:
    return (
        _extract_closed_fence(text)
        or _extract_unclosed_fence(text)
    )
```

- [ ] **Step 4: Run all tests — must be green**

```bash
uv run pytest tests/eval_framework/agents/test_no_tool_baseline.py::TestExtractSqlFromResponse -v
```

Expected: all 11 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/eval_framework/agents/test_no_tool_baseline.py \
        src/conversation2sql/eval_framework/agents/no_tool_baseline/baseline_model.py
git commit -m "feat: add _extract_unclosed_fence strategy to extract_sql_from_response"
```

---

### Task 3: Add `_extract_raw_sql` with TDD

**Files:**
- Modify: `tests/eval_framework/agents/test_no_tool_baseline.py`
- Modify: `src/conversation2sql/eval_framework/agents/no_tool_baseline/baseline_model.py`

- [ ] **Step 1: Write failing tests**

Append to `TestExtractSqlFromResponse`:

```python
def test_raw_sql_select_starts_line(self):
    text = "The answer is:\nSELECT 1;"
    assert extract_sql_from_response(text) == "SELECT 1;"

def test_raw_sql_with_keyword(self):
    text = "Reasoning...\nWITH cte AS (SELECT 1) SELECT * FROM cte;"
    assert extract_sql_from_response(text) == "WITH cte AS (SELECT 1) SELECT * FROM cte;"

def test_raw_sql_last_keyword_wins(self):
    text = "First try:\nSELECT 0;\nBetter answer:\nSELECT 1;"
    assert extract_sql_from_response(text) == "SELECT 1;"

def test_raw_sql_no_semicolon(self):
    text = "Reasoning prose\nSELECT * FROM t"
    assert extract_sql_from_response(text) == "SELECT * FROM t"

def test_raw_sql_keyword_inside_word_not_matched(self):
    # SELECTING never starts a line — should return None
    text = "I am SELECTING rows from the table"
    assert extract_sql_from_response(text) is None

def test_closed_fence_takes_priority_over_raw_sql(self):
    text = "```sql\nSELECT 1;\n```\nYou could also do SELECT 2;"
    assert extract_sql_from_response(text) == "SELECT 1;"
```

- [ ] **Step 2: Run to confirm they fail**

```bash
uv run pytest tests/eval_framework/agents/test_no_tool_baseline.py::TestExtractSqlFromResponse -v -k "raw_sql or priority_over_raw"
```

Expected: the 5 `raw_sql_*` tests FAIL (return `None`); `test_closed_fence_takes_priority_over_raw_sql` may already PASS (closed fence wins) — that's fine.

- [ ] **Step 3: Add `_SQL_KEYWORD_RE` and `_extract_raw_sql` to `baseline_model.py`**

Add after `_FENCE_OPEN_RE`:

```python
_SQL_KEYWORD_RE = re.compile(
    r"^[ \t]*(SELECT|WITH|INSERT|UPDATE|DELETE|CREATE|DROP|ALTER|EXPLAIN)\b",
    re.IGNORECASE | re.MULTILINE,
)
```

Add after `_extract_unclosed_fence`:

```python
def _extract_raw_sql(text: str) -> str | None:
    matches = list(_SQL_KEYWORD_RE.finditer(text))
    if not matches:
        return None
    last = matches[-1]
    after = text[last.start() :]
    semi_pos = after.find(";")
    candidate = after[: semi_pos + 1].strip() if semi_pos != -1 else after.strip()
    return candidate if candidate else None
```

Update the dispatcher:

```python
def extract_sql_from_response(text: str) -> str | None:
    return (
        _extract_closed_fence(text)
        or _extract_unclosed_fence(text)
        or _extract_raw_sql(text)
    )
```

- [ ] **Step 4: Run the full test suite**

```bash
uv run pytest tests/eval_framework/agents/test_no_tool_baseline.py -v
```

Expected: all tests PASS (the 7 original + 4 unclosed + 6 raw SQL = 17 total).

- [ ] **Step 5: Run the broader test suite to catch regressions**

```bash
uv run pytest tests/ -v
```

Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add tests/eval_framework/agents/test_no_tool_baseline.py \
        src/conversation2sql/eval_framework/agents/no_tool_baseline/baseline_model.py
git commit -m "feat: add _extract_raw_sql strategy to extract_sql_from_response"
```
