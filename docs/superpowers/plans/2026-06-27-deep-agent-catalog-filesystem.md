# Deep-agent catalog-backed filesystem — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the deep_agent's three monolithic `/db` seed files with the per-table Markdown catalog from `scripts/generate_catalog.py`, mounting only the sample's database folder and re-rendering the KB per task from `masked_agent_kb` for faithful masking.

**Architecture:** A new `deep_catalog_root` config is threaded `ConfigReader → load_bird_interact_as_tasks → TaskData`. `build_db_filesystem(task)` reads `<deep_catalog_root>/<selected_database>/tables/*.md` from disk verbatim and re-renders `/db/knowledge_base/<node>.md` for each node in `task.masked_agent_kb` via `linearize_prerequisites`. The deepagents `StateBackend` filesystem and tool set are unchanged.

**Tech Stack:** Python 3.12, Pydantic, LangChain `deepagents` FilesystemMiddleware, pytest (`asyncio_mode=auto`), `uv`.

## Global Constraints

- Run all Python/tests through `uv run` (e.g. `uv run pytest`, `uv run pyrefly check`).
- `deep_catalog_root` is a per-run field: add it to `ConfigReader` only. Do **not** add it to `configs/eval_pipeline_config.yaml` (a static-YAML entry would shadow the CLI flag).
- KB filenames may contain `:`, `<`, spaces (e.g. `FalsePosProb: <0.01.md`) — never sanitize node names; the file stem must equal the `knowledge` name verbatim.
- Reuse `linearize_prerequisites` from `agents/utils_kb_linearize.py`; do not reimplement KB rendering.
- Full reference: `docs/superpowers/specs/2026-06-27-deep-agent-catalog-filesystem-design.md`.

---

### Task 1: Plumb `deep_catalog_root` through config → reader → TaskData

**Files:**
- Modify: `src/conversation2sql/config_input.py:42` (after the `deep_enable_*` block in `ConfigReader`)
- Modify: `src/conversation2sql/eval_framework/state.py:81` (after the `deep_enable_*` fields in `TaskData`)
- Modify: `src/conversation2sql/eval_framework/dataset_readers/bird_interact_reader.py:260` (signature) and `:437` (TaskData construction)
- Test: `tests/eval_framework/agents/deep_agent/test_config_flags.py`

**Interfaces:**
- Produces: `ConfigReader.deep_catalog_root: str` (default `""`), `TaskData.deep_catalog_root: str` (default `""`), and `load_bird_interact_as_tasks(..., deep_catalog_root: str = "")` forwarding it into every `TaskData`.

- [ ] **Step 1: Write the failing test**

Add to `tests/eval_framework/agents/deep_agent/test_config_flags.py`:

```python
def test_config_reader_has_deep_catalog_root_default_empty():
    cfg = ConfigReader()
    assert cfg.deep_catalog_root == ""


def test_taskdata_carries_deep_catalog_root(make_minimal_task_kwargs):
    task = TaskData(**make_minimal_task_kwargs(deep_catalog_root="/some/root"))
    assert task.deep_catalog_root == "/some/root"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/eval_framework/agents/deep_agent/test_config_flags.py -v`
Expected: FAIL — `ConfigReader` has no attribute `deep_catalog_root` / `TaskData` rejects the kwarg.

- [ ] **Step 3: Add the field to `ConfigReader`**

In `src/conversation2sql/config_input.py`, immediately after the `deep_enable_fs_write` line (`:42`):

```python
    deep_catalog_root: str = ''  # root holding <db>/tables/*.md catalogs for the deep_agent FS (e.g. data/bird_interact/catalog_bird_interact_lite); set per run
```

- [ ] **Step 4: Add the field to `TaskData`**

In `src/conversation2sql/eval_framework/state.py`, immediately after the `deep_enable_fs_write: bool = False` line (`:81`):

```python
    # Root dir holding the per-database catalog (<db>/tables/*.md) the deep_agent
    # mounts as its /db filesystem; '' means unset (build_db_filesystem will raise).
    deep_catalog_root: str = ""
```

- [ ] **Step 5: Thread it through the reader**

In `src/conversation2sql/eval_framework/dataset_readers/bird_interact_reader.py`, add the parameter to `load_bird_interact_as_tasks` (after `deep_enable_fs_write: bool = False,` at `:260`):

```python
    deep_catalog_root: str = "",
```

And in the `TaskData(...)` construction (after `deep_enable_fs_write=deep_enable_fs_write,` at `:437`):

```python
                deep_catalog_root=deep_catalog_root,
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/eval_framework/agents/deep_agent/test_config_flags.py -v`
Expected: PASS (all tests in the file).

- [ ] **Step 7: Commit**

```bash
git add src/conversation2sql/config_input.py src/conversation2sql/eval_framework/state.py src/conversation2sql/eval_framework/dataset_readers/bird_interact_reader.py tests/eval_framework/agents/deep_agent/test_config_flags.py
git commit -m "feat(deep_agent): add deep_catalog_root config threaded to TaskData"
```

---

### Task 2: `build_db_filesystem` loads the tables catalog from disk

**Files:**
- Modify: `src/conversation2sql/eval_framework/agents/deep_agent/filesystem_seed.py` (full rewrite of the seed logic; keep `FS_TOOL_COSTS` / `deep_tool_costs` unchanged)
- Test: `tests/eval_framework/agents/deep_agent/test_filesystem_seed.py` (full rewrite)

**Interfaces:**
- Consumes: `TaskData.deep_catalog_root`, `TaskData.selected_database` (from Task 1).
- Produces: `build_db_filesystem(task) -> dict[str, FileData]` seeding `/db/tables/<file>.md` from `<deep_catalog_root>/<selected_database>/tables/*.md`; raises `FileNotFoundError` when that DB dir is absent. (KB seeding is added in Task 3.) Removes the old `DB_FS_PATHS` export and the in-memory schema/column/KB renderers.

- [ ] **Step 1: Write the failing tests**

Replace the entire contents of `tests/eval_framework/agents/deep_agent/test_filesystem_seed.py` with:

```python
import pytest

from conversation2sql.eval_framework.state import TaskData
from conversation2sql.eval_framework.agents.deep_agent.filesystem_seed import (
    build_db_filesystem,
)


def _write_catalog(root, db="mydb"):
    """Create a minimal on-disk tables catalog under <root>/<db>/tables/."""
    tables = root / db / "tables"
    tables.mkdir(parents=True)
    (tables / "users.md").write_text("# table: users\n", encoding="utf-8")
    (tables / "_foreign_key_constraints.md").write_text(
        "# constraints: mydb\n", encoding="utf-8"
    )
    return root


def test_tables_are_loaded_verbatim_from_disk(tmp_path, make_minimal_task_kwargs):
    _write_catalog(tmp_path)
    task = TaskData(
        **make_minimal_task_kwargs(
            selected_database="mydb",
            deep_catalog_root=str(tmp_path),
            masked_agent_kb={},
        )
    )
    files = build_db_filesystem(task)
    assert files["/db/tables/users.md"]["content"] == "# table: users\n"
    assert files["/db/tables/users.md"]["encoding"] == "utf-8"
    assert "/db/tables/_foreign_key_constraints.md" in files


def test_missing_catalog_dir_raises(tmp_path, make_minimal_task_kwargs):
    task = TaskData(
        **make_minimal_task_kwargs(
            selected_database="absent_db",
            deep_catalog_root=str(tmp_path),
            masked_agent_kb={},
        )
    )
    with pytest.raises(FileNotFoundError, match="absent_db"):
        build_db_filesystem(task)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/eval_framework/agents/deep_agent/test_filesystem_seed.py -v`
Expected: FAIL — old `build_db_filesystem` ignores `deep_catalog_root` and returns the three legacy paths (and the missing-dir test does not raise).

- [ ] **Step 3: Rewrite `filesystem_seed.py`**

Replace the entire contents of `src/conversation2sql/eval_framework/agents/deep_agent/filesystem_seed.py` with:

```python
"""Seed the deep agent's /db virtual filesystem from the on-disk catalog.

The deep_agent reads a per-database Markdown catalog (produced by
``scripts/generate_catalog.py``) through the deepagents FilesystemMiddleware.
Only the catalog folder of the task's own database is mounted, under /db:

  /db/tables/<table>.md                  (read from disk, as-is)
  /db/tables/_foreign_key_constraints.md (read from disk, as-is)
  /db/knowledge_base/<node>.md           (re-rendered per task, added in Task 3)
"""
from __future__ import annotations

from pathlib import Path

from deepagents.middleware.filesystem import FileData

from conversation2sql.eval_framework.agents.bird_baseline.tools import TOOL_COSTS
from conversation2sql.eval_framework.state import TaskData


def _text_file(content: str) -> FileData:
    return {"content": content, "encoding": "utf-8"}


def _catalog_db_dir(task: TaskData) -> Path:
    return Path(task.deep_catalog_root) / task.selected_database


def _seed_tables(db_dir: Path) -> dict[str, FileData]:
    """Mirror every <db_dir>/tables/*.md file at /db/tables/<name>, verbatim."""
    files: dict[str, FileData] = {}
    for path in sorted((db_dir / "tables").glob("*.md")):
        files[f"/db/tables/{path.name}"] = _text_file(path.read_text(encoding="utf-8"))
    return files


def build_db_filesystem(task: TaskData) -> dict[str, FileData]:
    """Render the /db filesystem for one task from its database's catalog folder."""
    db_dir = _catalog_db_dir(task)
    if not db_dir.is_dir():
        raise FileNotFoundError(
            f"deep_agent catalog not found for database '{task.selected_database}' "
            f"at {db_dir}. Generate it with scripts/generate_catalog.py "
            f"(--database {task.selected_database} --output-dir {task.deep_catalog_root})."
        )
    return _seed_tables(db_dir)


# Bird-coin costs for the deepagents filesystem read/write tools. Reads are cheap
# (static DB info); writes are only present under the deep_enable_fs_write ablation.
FS_TOOL_COSTS: dict[str, float] = {
    "ls": 0.5,
    "read_file": 0.5,
    "glob": 0.5,
    "grep": 0.5,
    "write_file": 0.5,
    "edit_file": 0.5,
}

_FS_READ_TOOLS = ("ls", "read_file", "glob", "grep")
_FS_WRITE_TOOLS = ("write_file", "edit_file")
# Reused tools the deep_agent always has; pull their costs from the shared table.
_REUSED_TOOLS = ("execute_sql", "ask_user", "submit_sql")


def deep_tool_costs(*, enable_fs_write: bool) -> dict[str, float]:
    """Cost map the patience middleware consults for the deep_agent's tool set."""
    names = list(_FS_READ_TOOLS)
    if enable_fs_write:
        names += list(_FS_WRITE_TOOLS)
    costs = {n: FS_TOOL_COSTS[n] for n in names}
    for n in _REUSED_TOOLS:
        costs[n] = TOOL_COSTS[n]
    return costs
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/eval_framework/agents/deep_agent/test_filesystem_seed.py tests/eval_framework/agents/deep_agent/test_state_and_costs.py -v`
Expected: PASS (seed tests pass; `test_state_and_costs.py` still passes — `FS_TOOL_COSTS`/`deep_tool_costs` unchanged).

- [ ] **Step 5: Commit**

```bash
git add src/conversation2sql/eval_framework/agents/deep_agent/filesystem_seed.py tests/eval_framework/agents/deep_agent/test_filesystem_seed.py
git commit -m "feat(deep_agent): seed /db/tables from on-disk catalog folder"
```

---

### Task 3: Re-render the masked KB into `/db/knowledge_base/`

**Files:**
- Modify: `src/conversation2sql/eval_framework/agents/deep_agent/filesystem_seed.py`
- Test: `tests/eval_framework/agents/deep_agent/test_filesystem_seed.py`

**Interfaces:**
- Consumes: `linearize_prerequisites(name, masked_agent_kb)` from `agents/utils_kb_linearize.py`; `TaskData.masked_agent_kb`.
- Produces: `build_db_filesystem` additionally seeds `/db/knowledge_base/<name>.md` for each `name` in `task.masked_agent_kb`; masked prerequisites and their dangling edges are absent.

- [ ] **Step 1: Write the failing tests**

Append to `tests/eval_framework/agents/deep_agent/test_filesystem_seed.py`:

```python
from conversation2sql.eval_framework.state import ExternalKnowledgeEntry


def _kb_pair():
    """Two nodes: 'Score (SC)' depends on 'Base (BS)' (SC.children = [BS.id])."""
    return {
        "Score (SC)": ExternalKnowledgeEntry(
            id=1,
            knowledge="Score (SC)",
            description="A score",
            definition="SC = BS * 2",
            type="calculation_knowledge",
            children_knowledge=[2],
        ),
        "Base (BS)": ExternalKnowledgeEntry(
            id=2,
            knowledge="Base (BS)",
            description="A base value",
            definition="BS = 10",
            type="domain_knowledge",
            children_knowledge=[-1],
        ),
    }


def test_kb_files_only_for_nodes_in_masked_kb(tmp_path, make_minimal_task_kwargs):
    _write_catalog(tmp_path)
    full = _kb_pair()
    masked = {"Score (SC)": full["Score (SC)"]}  # 'Base (BS)' masked out
    task = TaskData(
        **make_minimal_task_kwargs(
            selected_database="mydb",
            deep_catalog_root=str(tmp_path),
            masked_agent_kb=masked,
        )
    )
    files = build_db_filesystem(task)
    assert "/db/knowledge_base/Score (SC).md" in files
    assert "/db/knowledge_base/Base (BS).md" not in files


def test_dangling_edge_to_masked_node_is_stripped(tmp_path, make_minimal_task_kwargs):
    _write_catalog(tmp_path)
    full = _kb_pair()
    masked = {"Score (SC)": full["Score (SC)"]}  # prerequisite 'Base (BS)' masked
    task = TaskData(
        **make_minimal_task_kwargs(
            selected_database="mydb",
            deep_catalog_root=str(tmp_path),
            masked_agent_kb=masked,
        )
    )
    files = build_db_filesystem(task)
    content = files["/db/knowledge_base/Score (SC).md"]["content"]
    assert "needs" not in content  # edge to the masked 'BS' node is gone


def test_edge_present_when_prerequisite_not_masked(tmp_path, make_minimal_task_kwargs):
    _write_catalog(tmp_path)
    task = TaskData(
        **make_minimal_task_kwargs(
            selected_database="mydb",
            deep_catalog_root=str(tmp_path),
            masked_agent_kb=_kb_pair(),  # both nodes present
        )
    )
    files = build_db_filesystem(task)
    content = files["/db/knowledge_base/Score (SC).md"]["content"]
    assert '"SC" needs "BS"' in content


def test_empty_kb_seeds_no_knowledge_base_files(tmp_path, make_minimal_task_kwargs):
    _write_catalog(tmp_path)
    task = TaskData(
        **make_minimal_task_kwargs(
            selected_database="mydb",
            deep_catalog_root=str(tmp_path),
            masked_agent_kb={},
        )
    )
    files = build_db_filesystem(task)
    assert not any(p.startswith("/db/knowledge_base/") for p in files)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/eval_framework/agents/deep_agent/test_filesystem_seed.py -v`
Expected: FAIL — `build_db_filesystem` does not yet seed any `/db/knowledge_base/` files.

- [ ] **Step 3: Add KB seeding to `filesystem_seed.py`**

Add the import near the top of `src/conversation2sql/eval_framework/agents/deep_agent/filesystem_seed.py` (below the existing imports):

```python
from conversation2sql.eval_framework.agents.utils_kb_linearize import (
    linearize_prerequisites,
)
```

Add this helper after `_seed_tables`:

```python
def _seed_knowledge_base(task: TaskData) -> dict[str, FileData]:
    """One /db/knowledge_base/<node>.md per node, re-rendered from masked_agent_kb.

    Using the (already masked) KB means masked prerequisites are never collected
    and their edges never render — faithful per-sample masking, no leak.
    """
    files: dict[str, FileData] = {}
    for name in task.masked_agent_kb:
        content = linearize_prerequisites(name, task.masked_agent_kb)
        files[f"/db/knowledge_base/{name}.md"] = _text_file(content)
    return files
```

Update `build_db_filesystem` to merge KB files in (replace its final `return _seed_tables(db_dir)` line):

```python
    files = _seed_tables(db_dir)
    files.update(_seed_knowledge_base(task))
    return files
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/eval_framework/agents/deep_agent/test_filesystem_seed.py -v`
Expected: PASS (all seed tests, including the new KB tests).

- [ ] **Step 5: Commit**

```bash
git add src/conversation2sql/eval_framework/agents/deep_agent/filesystem_seed.py tests/eval_framework/agents/deep_agent/test_filesystem_seed.py
git commit -m "feat(deep_agent): re-render masked KB into /db/knowledge_base"
```

---

### Task 4: Update the system prompt to advertise the new layout

**Files:**
- Modify: `src/conversation2sql/eval_framework/agents/deep_agent/prompts.py:6-40` (the `_DEEP_AGENT_SYSTEM` template)
- Test: `tests/eval_framework/agents/deep_agent/test_prompts.py:18-24`

**Interfaces:**
- Consumes: nothing new.
- Produces: a system prompt that names `/db/tables/` and `/db/knowledge_base/` instead of the three fixed file paths.

- [ ] **Step 1: Update the failing test**

Replace `test_prompt_mentions_db_filesystem_paths` in `tests/eval_framework/agents/deep_agent/test_prompts.py` with:

```python
def test_prompt_mentions_db_filesystem_paths():
    msgs = _render()
    system = msgs[0]["content"]
    assert "/db/tables/" in system
    assert "/db/knowledge_base/" in system
    assert "read_file" in system
    # No get_schema-style tools are advertised, and no legacy monolithic paths.
    assert "get_schema" not in system
    assert "/db/schema.sql" not in system
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/eval_framework/agents/deep_agent/test_prompts.py -v`
Expected: FAIL — the prompt still says `/db/schema.sql` and lacks `/db/tables/`.

- [ ] **Step 3: Update the prompt template**

In `src/conversation2sql/eval_framework/agents/deep_agent/prompts.py`, replace the filesystem-description block and the first strategy bullet inside `_DEEP_AGENT_SYSTEM`. Change:

```
You explore the database NOT through dedicated schema tools, but through a virtual
filesystem under /db. Use the filesystem tools (ls, read_file, grep, glob) to read:
- /db/schema.sql          — the full DDL (tables, columns, foreign keys)
- /db/column_meanings.md  — natural-language meaning of each column
- /db/knowledge_base.md   — external knowledge definitions relevant to the task
```

to:

```
You explore the database NOT through dedicated schema tools, but through a virtual
filesystem under /db. Use the filesystem tools (ls, read_file, grep, glob) to read:
- /db/tables/            — one Markdown file per table (DDL, columns with descriptions,
                           foreign keys); _foreign_key_constraints.md lists every PK/FK.
- /db/knowledge_base/    — one Markdown file per external-knowledge entry for the task.
```

And change the first Strategy bullet:

```
- Start by reading /db/schema.sql and the column meanings to understand the data.
```

to:

```
- Start with `ls /db/tables` and read the relevant table files to understand the data.
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/eval_framework/agents/deep_agent/test_prompts.py -v`
Expected: PASS (all three prompt tests).

- [ ] **Step 5: Run the full suite**

Run: `uv run pytest tests/`
Expected: PASS. (If any deep_agent integration test seeds a fake catalog, confirm it sets `deep_catalog_root`; fix by pointing it at a tmp catalog dir built like `_write_catalog`.)

- [ ] **Step 6: Type-check**

Run: `uv run pyrefly check`
Expected: no new errors in the modified files.

- [ ] **Step 7: Commit**

```bash
git add src/conversation2sql/eval_framework/agents/deep_agent/prompts.py tests/eval_framework/agents/deep_agent/test_prompts.py
git commit -m "feat(deep_agent): advertise /db/tables and /db/knowledge_base in prompt"
```

---

## Self-Review

**Spec coverage:**
- FS layout `/db/tables/*` + `/db/knowledge_base/*`, old 3 files removed → Task 2 (tables), Task 3 (KB), Task 4 (prompt).
- Tables read verbatim from disk + missing-dir `FileNotFoundError` → Task 2.
- KB re-rendered from `masked_agent_kb` via `linearize_prerequisites`, masked nodes + dangling edges dropped, empty KB → no files → Task 3.
- `deep_catalog_root` on `ConfigReader` → reader → `TaskData`, not in static YAML → Task 1 (+ Global Constraints).
- Prompt advertises new layout → Task 4.
- `deep_tool_costs`/FS tools unchanged → preserved verbatim in Task 2's rewrite; `test_state_and_costs.py` left green (Task 2 Step 4).
- Caveats (`read_only_gt_tables` not applied; live-introspected DDL) are design-level notes with no code action — intentionally no task.

**Placeholder scan:** No TBD/TODO/"handle edge cases"; every code step shows full content.

**Type consistency:** `deep_catalog_root: str` is consistent across `ConfigReader`, `TaskData`, and the reader param. `build_db_filesystem(task) -> dict[str, FileData]` signature is unchanged for `agent_code.py`'s caller. `_seed_tables`/`_seed_knowledge_base`/`_catalog_db_dir`/`_text_file` are defined in Task 2/3 and used consistently. `FS_TOOL_COSTS`/`deep_tool_costs` names preserved for `test_state_and_costs.py` and `agent_code.py`.
