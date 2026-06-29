# Single full/lite dataset switch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the four hand-synced full/lite path/DSN fields in `ConfigReader` with one `dataset_variant` switch that drives all of them.

**Architecture:** Add a settable `dataset_variant: Literal['lite','full']` plus a small static base config (`data_root`, `db_host`, `db_user`, `db_password`). Turn `dataset_path`, `dataset_name_jsonl`, `deep_catalog_root`, and `db_dsn_template` into read-only Pydantic `@computed_field`s derived from those knobs. Computed fields stay in `model_dump()`, so the `load_bird_interact_as_tasks(**config_reader.model_dump())` splat keeps working unchanged.

**Tech Stack:** Python 3.12, Pydantic v2, `uv run pytest`, `uv run pyrefly`.

## Global Constraints

- Run all Python/pytest via `uv run` (project venv).
- `db_dsn_template` must keep the literal `{database}` placeholder the reader formats with `.format(database=db_name)` — in an f-string write it as `{{database}}`.
- Postgres port mapping is fixed: `lite -> 5432`, `full -> 5433` (per `.claude/CLAUDE.md`).
- Do NOT add per-run dataset/path fields back to `configs/eval_pipeline_config.yaml` as raw values — the switch lives there as `dataset_variant`.
- Preserve the existing `_check_db_tool_ablation_exclusivity` validator and every existing non-path `ConfigReader` field (`filter_query_category`, `user_patience_budget`, `make_data_ambiguous`, `database_schema_type`, `read_only_gt_kb`, `read_only_gt_tables`, `is_kb_linearized`, `enable_table_schema_tools`, `enable_psql_console`, `enable_psql_strict_inspection`, `enable_python_udf`, `deep_enable_todos`, `deep_enable_subagents`, `deep_enable_summarization`, `deep_enable_fs_write`).

---

## File Structure

- `src/conversation2sql/config_input.py` — refactor `ConfigReader`: add switch + base knobs, convert 4 fields to computed fields, add module-level `_VARIANT_DB_PORT`.
- `tests/test_config_input.py` — add variant-switch coverage.
- `tests/eval_framework/agents/deep_agent/test_config_flags.py` — repurpose the stale `deep_catalog_root == ""` test (currently failing) to assert the derived lite catalog path.
- `configs/eval_pipeline_config.yaml` — swap the 3 raw reader keys for `dataset_variant: 'lite'`; fix the commented full example.
- `src/conversation2sql/CLAUDE.md` — document the new switch in the config-models / gotchas notes.

---

## Task 1: Refactor `ConfigReader` to a single `dataset_variant` switch

**Files:**
- Modify: `src/conversation2sql/config_input.py:1-52` (imports + `ConfigReader`)
- Test: `tests/test_config_input.py`
- Modify: `tests/eval_framework/agents/deep_agent/test_config_flags.py:24-26`

**Interfaces:**
- Consumes: nothing (leaf change).
- Produces: `ConfigReader` with settable fields `dataset_variant: Literal['lite','full']` (default `'lite'`), `data_root: str` (default `'data/bird_interact'`), `db_host: str` (default `'localhost'`), `db_user: str` (default `'root'`), `db_password: str` (default `'123123'`); and read-only computed `str` properties `dataset_path`, `dataset_name_jsonl`, `deep_catalog_root`, `db_dsn_template`. All four computed values appear in `ConfigReader().model_dump()`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_config_input.py`:

```python
def test_dataset_variant_default_is_lite():
    cfg = ConfigReader()
    assert cfg.dataset_variant == "lite"
    assert cfg.dataset_path == "data/bird_interact/bird-interact-lite"
    assert (
        cfg.dataset_name_jsonl
        == "data/bird_interact/bird-interact-lite/bird_interact_data_GT.jsonl"
    )
    assert cfg.deep_catalog_root == "data/bird_interact/catalog_bird_interact_lite"
    assert cfg.db_dsn_template == "postgresql://root:123123@localhost:5432/{database}"


def test_dataset_variant_full_moves_all_paths_and_port():
    cfg = ConfigReader(dataset_variant="full")
    assert cfg.dataset_path == "data/bird_interact/bird-interact-full"
    assert (
        cfg.dataset_name_jsonl
        == "data/bird_interact/bird-interact-full/bird_interact_data_GT.jsonl"
    )
    assert cfg.deep_catalog_root == "data/bird_interact/catalog_bird_interact_full"
    assert cfg.db_dsn_template == "postgresql://root:123123@localhost:5433/{database}"


def test_db_host_credentials_override_flow_into_dsn():
    cfg = ConfigReader(
        dataset_variant="full", db_host="slurm-node-1", db_user="u", db_password="p"
    )
    assert cfg.db_dsn_template == "postgresql://u:p@slurm-node-1:5433/{database}"


def test_data_root_override_repaths_everything():
    cfg = ConfigReader(data_root="/custom/data")
    assert cfg.dataset_path == "/custom/data/bird-interact-lite"
    assert cfg.deep_catalog_root == "/custom/data/catalog_bird_interact_lite"


def test_computed_path_fields_present_in_model_dump():
    dumped = ConfigReader().model_dump()
    for key in ("dataset_path", "dataset_name_jsonl", "deep_catalog_root", "db_dsn_template"):
        assert key in dumped, f"{key} missing from model_dump()"


def test_dataset_variant_rejects_unknown_value():
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ConfigReader(dataset_variant="medium")
```

Also update the stale, currently-failing test in `tests/eval_framework/agents/deep_agent/test_config_flags.py` — replace lines 24-26:

```python
def test_config_reader_deep_catalog_root_default_is_lite_catalog():
    cfg = ConfigReader()
    assert cfg.deep_catalog_root == "data/bird_interact/catalog_bird_interact_lite"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_config_input.py -q`
Expected: FAIL — the new tests error because `dataset_variant`/`data_root`/`db_host` are not yet fields (e.g. `ValidationError: Extra inputs are not permitted` for `dataset_variant`, and `AttributeError`/missing-default mismatches for the computed values).

- [ ] **Step 3: Implement the refactor**

In `src/conversation2sql/config_input.py`, change the import line:

```python
from pydantic import BaseModel, Field, computed_field, model_validator
```

Add a module-level constant just above `class ConfigReader` (after the section banner comment, near line 20):

```python
# Postgres port per dataset variant (lite -> 5432, full -> 5433); see .claude/CLAUDE.md
_VARIANT_DB_PORT: dict[str, int] = {"lite": 5432, "full": 5433}
```

Replace the body of `ConfigReader` (the current lines 22-52, i.e. the four raw fields through the validator) with:

```python
class ConfigReader(BaseModel):
    # --- the single full/lite switch + static base knobs it derives from ---
    dataset_variant: Literal["lite", "full"] = "lite"  # one switch: drives all dataset paths, the deep_agent catalog root, and the Postgres port
    data_root: str = "data/bird_interact"  # base dir holding the bird-interact-* and catalog_bird_interact_* trees
    db_host: str = "localhost"  # Postgres host; override for SLURM. The port is derived from dataset_variant, not set here.
    db_user: str = "root"
    db_password: str = "123123"

    filter_query_category: bool = True
    user_patience_budget: int = 10
    make_data_ambiguous: bool = True
    database_schema_type: Literal['ddl', 'toon'] = 'ddl'  # Whether to use the original complex schema or a simplified version for better model understanding
    read_only_gt_kb: bool = False  # Whether to only include the tables/columns that are actually used in the GT SQL query when providing the schema to the model
    read_only_gt_tables: bool = False  # Whether to only include the tables that are actually used in the GT SQL query when providing the schema to the model
    is_kb_linearized: bool = False  # Whether to linearize the KB schema into text or provide it in a structured format (e.g., JSON); linearization may be easier for LLMs to understand but less faithful to the original structure
    enable_table_schema_tools: bool = False  # When True the agent additionally gets get_table_names + get_table_schema (granular per-table DDL access, mirroring the KB name/definition tools). Off by default so the baseline keeps only the full-dump get_schema; flip on for ablations.
    enable_psql_console: bool = False  # Ablation: replace the DB tools (execute_sql/get_schema/get_table_*) with a single read-only psql terminal tool (psql_console). Mutually exclusive with enable_table_schema_tools.
    enable_psql_strict_inspection: bool = False  # Ablation (only meaningful with enable_psql_console): restrict psql_console to SQL + \h + the informational \d-family; \? lists only those. Off = legacy denylist behavior (full rollback).
    enable_python_udf: bool = False  # Ablation: add create_python_udf tool (plpython3u). Additive — compatible with all other DB-tool variants.

    # --- deep_agent baseline ablations (only meaningful when baseline='deep_agent') ---
    deep_enable_todos: bool = False  # add deepagents planning/write_todos middleware
    deep_enable_subagents: bool = False  # add deepagents subagents (task tool) middleware
    deep_enable_summarization: bool = False  # add deepagents/langchain SummarizationMiddleware
    deep_enable_fs_write: bool = False  # expose write_file/edit_file (default: read-only FS)

    # --- derived, read-only: all four driven by dataset_variant + the base knobs ---
    @computed_field  # type: ignore[prop-decorator]
    @property
    def dataset_path(self) -> str:
        return f"{self.data_root}/bird-interact-{self.dataset_variant}"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def dataset_name_jsonl(self) -> str:
        return f"{self.dataset_path}/bird_interact_data_GT.jsonl"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def deep_catalog_root(self) -> str:
        return f"{self.data_root}/catalog_bird_interact_{self.dataset_variant}"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def db_dsn_template(self) -> str:
        port = _VARIANT_DB_PORT[self.dataset_variant]
        return f"postgresql://{self.db_user}:{self.db_password}@{self.db_host}:{port}/{{database}}"

    @model_validator(mode="after")
    def _check_db_tool_ablation_exclusivity(self) -> "ConfigReader":
        if self.enable_psql_console and self.enable_table_schema_tools:
            raise ValueError(
                "enable_psql_console and enable_table_schema_tools are mutually "
                "exclusive; enable at most one DB-tool ablation."
            )
        return self
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_config_input.py tests/eval_framework/agents/deep_agent/test_config_flags.py -q`
Expected: PASS (all, including the repurposed `test_config_reader_deep_catalog_root_default_is_lite_catalog`).

- [ ] **Step 5: Commit**

```bash
git add src/conversation2sql/config_input.py tests/test_config_input.py tests/eval_framework/agents/deep_agent/test_config_flags.py
git commit -m "feat(config): single dataset_variant switch drives all full/lite paths

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: Update YAML config and docs, verify full suite

**Files:**
- Modify: `configs/eval_pipeline_config.yaml:11-16` and the commented example block (~33-45)
- Modify: `src/conversation2sql/CLAUDE.md`

**Interfaces:**
- Consumes: `ConfigReader.dataset_variant` from Task 1.
- Produces: no code interface; static config + docs aligned with the switch.

- [ ] **Step 1: Update `configs/eval_pipeline_config.yaml`**

Replace the active `reader:` block keys:

```yaml
reader:
  dataset_variant: 'lite'   # 'lite' (Postgres port 5432) or 'full' (5433); drives dataset_path, dataset_name_jsonl, deep_catalog_root, and the DSN port
  filter_query_category: true
  user_patience_budget: 12
  # db_host/db_user/db_password default to localhost/root/123123 — override db_host for SLURM
```

In the commented "Example full:" block lower in the file, replace the three reader lines (`dataset_name_jsonl`, `dataset_path`, `db_dsn_template`) with a single:

```yaml
#   dataset_variant: lite
```

- [ ] **Step 2: Update `src/conversation2sql/CLAUDE.md`**

Under "## Key files", change the `config_input.py` bullet to note the switch, and add a bullet under "## Gotchas":

```markdown
- `ConfigReader` exposes one `dataset_variant: Literal['lite','full']` switch. `dataset_path`, `dataset_name_jsonl`, `deep_catalog_root`, and `db_dsn_template` are read-only `@computed_field`s derived from it (+ `data_root`, `db_host`, `db_user`, `db_password`). They are not settable via CLI/YAML; flip the variant instead. The lite/full Postgres port (5432/5433) is derived from the variant; host/creds stay overridable for SLURM.
```

- [ ] **Step 3: Run the full suite + type check**

Run: `uv run pytest tests/ -q && uv run pyrefly check`
Expected: PASS / clean. (Watch `tests/eval_framework/test_main_pipe_workflow.py::test_saved_snapshot_roundtrips_through_parser` — the snapshot now serializes computed keys, which `_read_yaml` ignores on reparse; the test should still pass.)

- [ ] **Step 4: Manual sanity check**

Run: `uv run python -c "from conversation2sql.config_input import ConfigReader as C; print(C(dataset_variant='full').db_dsn_template); print(C(dataset_variant='full').dataset_path)"`
Expected output:
```
postgresql://root:123123@localhost:5433/{database}
data/bird_interact/bird-interact-full
```

- [ ] **Step 5: Commit**

```bash
git add configs/eval_pipeline_config.yaml src/conversation2sql/CLAUDE.md
git commit -m "docs(config): point eval YAML + CLAUDE.md at dataset_variant switch

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage:**
- Switch as sole source of truth → Task 1 (computed fields).
- Variant picks port, host/creds configurable → Task 1 (`db_dsn_template` + `db_host/user/password`, `test_db_host_credentials_override_flow_into_dsn`).
- `model_dump()` splat keeps working → Task 1 (`test_computed_path_fields_present_in_model_dump`).
- YAML swap → Task 2 Step 1. Docs → Task 2 Step 2. Fix inconsistent defaults → covered (all derive from one variant).
- Out-of-scope items (no `catalog_bird_interact_full` generation, reader `"full" in` sniff untouched) → respected; reader unchanged.

**Placeholder scan:** none — all steps carry real code/commands.

**Type consistency:** `dataset_variant` (`Literal['lite','full']`), `data_root`/`db_host`/`db_user`/`db_password` (`str`), and the four computed `-> str` properties are named identically across the refactor, tests, YAML, and docs.
