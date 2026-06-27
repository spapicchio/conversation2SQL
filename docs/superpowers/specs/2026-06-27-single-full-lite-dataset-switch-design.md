# Single full/lite dataset switch in `ConfigReader`

**Date:** 2026-06-27
**Status:** Approved — ready for planning

## Problem

The choice between the BIRD-Interact **full** and **lite** datasets is currently
spread across four independent `ConfigReader` fields that must be kept mutually
consistent by hand:

| Field | full value | lite value |
|---|---|---|
| `dataset_path` | `.../bird-interact-full` | `.../bird-interact-lite` |
| `dataset_name_jsonl` | `.../bird-interact-full/bird_interact_data_GT.jsonl` | `.../bird-interact-lite/...` |
| `deep_catalog_root` | `.../catalog_bird_interact_full` | `.../catalog_bird_interact_lite` |
| `db_dsn_template` | port `5433` | port `5432` |

The committed defaults are already **inconsistent**: `dataset_path` /
`dataset_name_jsonl` point at `bird-interact-full`, while `db_dsn_template`
(port `5432`) and `deep_catalog_root` (`catalog_bird_interact_lite`) point at
lite. A single switch removes this whole class of mismatch.

## Goal

Expose **one switch**, `dataset_variant: Literal['lite', 'full']`, that drives
all four connected values. Setting it (e.g. `--dataset_variant full`) moves the
dataset paths, the deep_agent catalog root, and the Postgres port together.

## Design

### Settable fields (the knobs)

Replace the four raw path/DSN fields with a variant switch plus a small base
config that does **not** change between full and lite:

```python
dataset_variant: Literal['lite', 'full'] = 'lite'   # the one switch
data_root: str = 'data/bird_interact'               # base dir for dataset paths
db_host: str = 'localhost'                            # env-specific (SLURM override)
db_user: str = 'root'
db_password: str = '123123'
```

`db_host` / `db_user` / `db_password` stay overridable because the Postgres host
and credentials are environment-specific (local devcontainer vs. SLURM) and are
orthogonal to the full/lite choice. Only the **port** depends on the variant.

### Derived read-only fields (`@computed_field`)

The four previously-raw fields become Pydantic `@computed_field` properties.
Pydantic includes computed fields in `model_dump()`, so the existing
`load_bird_interact_as_tasks(**config_reader.model_dump())` splat keeps
receiving them by the same names — no reader-signature change required.

```python
_VARIANT_DB_PORT = {'lite': 5432, 'full': 5433}

@computed_field
@property
def dataset_path(self) -> str:
    return f"{self.data_root}/bird-interact-{self.dataset_variant}"

@computed_field
@property
def dataset_name_jsonl(self) -> str:
    return f"{self.dataset_path}/bird_interact_data_GT.jsonl"

@computed_field
@property
def deep_catalog_root(self) -> str:
    return f"{self.data_root}/catalog_bird_interact_{self.dataset_variant}"

@computed_field
@property
def db_dsn_template(self) -> str:
    port = _VARIANT_DB_PORT[self.dataset_variant]
    return f"postgresql://{self.db_user}:{self.db_password}@{self.db_host}:{port}/{{database}}"
```

Note the doubled `{{database}}` in the f-string so the emitted template keeps the
single `{database}` placeholder the reader formats with
`db_dsn_template.format(database=db_name)`.

The existing `_check_db_tool_ablation_exclusivity` validator is unchanged.

### Why this is safe (no CLI / preset breakage)

- `PydanticParser` iterates `model_type.model_fields` only (`cli_parser.py`
  lines 233, 260, 294). Computed fields are **not** in `model_fields`, so they
  generate no CLI flags, are never parsed from env/YAML, and are never passed to
  the model constructor.
- `_read_yaml` (line 260) silently ignores YAML keys not in `model_fields`, so a
  stale `dataset_path:` key would be dropped rather than error — but we remove
  them anyway for clarity.
- Nothing in `bash_scripts/`, `presets.py`, or `cli.py` sets these four fields as
  flags. The only place that sets them is `configs/eval_pipeline_config.yaml`.

## Touch points

1. **`src/conversation2sql/config_input.py`** — the refactor above.
2. **`configs/eval_pipeline_config.yaml`** — replace the `dataset_name_jsonl`,
   `dataset_path`, `db_dsn_template` keys (and commented variants) with
   `dataset_variant: lite`. Leave the static defaults for host/user/password
   unless an override is needed.
3. **`src/conversation2sql/CLAUDE.md`** — update the `config_input.py` field-list
   note to mention the new variant switch and computed fields.
4. **Tests** — update any test that constructs `ConfigReader` with, or asserts
   on, the old `dataset_path` / `dataset_name_jsonl` / `db_dsn_template` /
   `deep_catalog_root` raw fields. Add coverage that flipping `dataset_variant`
   moves all four computed values and the DB port together, and that
   `db_host`/`db_user`/`db_password` overrides flow into the DSN.

## Out of scope

- Generating a `catalog_bird_interact_full` directory (it does not exist yet; the
  computed path simply points there for when it is generated).
- Refactoring the reader's `if "full" in dataset_name_jsonl` sniff
  (`bird_interact_reader.py:342`) — it keeps working because the computed
  `dataset_name_jsonl` contains the variant string.

## Verification

`uv run pytest tests/` passes; `uv run pyrefly check` is clean.
Manual sanity: `ConfigReader(dataset_variant='full').db_dsn_template` ends in
`:5433/{database}` and `.dataset_path` ends in `bird-interact-full`.
