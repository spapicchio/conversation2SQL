# Deep-agent catalog-backed filesystem — design

**Date:** 2026-06-27
**Component:** `src/conversation2sql/eval_framework/agents/deep_agent/`

## Goal

Replace the deep_agent's three monolithic in-memory seed files
(`/db/schema.sql`, `/db/column_meanings.md`, `/db/knowledge_base.md`) with the
richer, navigable Markdown catalog produced by `scripts/generate_catalog.py`.
Per task, the virtual filesystem exposes **only the catalog folder of that
sample's database**, with the knowledge base masked per-sample so the agent
never sees knowledge that `make_data_ambiguous` is supposed to hide.

## Current state

`deep_agent/filesystem_seed.py::build_db_filesystem(task)` renders three files
in memory from `TaskData`:

| Path | Source |
|------|--------|
| `/db/schema.sql` | `task.ddl_database_schema` |
| `/db/column_meanings.md` | `task.column_meanings` |
| `/db/knowledge_base.md` | `task.masked_agent_kb` |

These are seeded into the deepagents `StateBackend` filesystem and read via the
FS tools (`ls`/`read_file`/`grep`/`glob`).

## Catalog structure (input)

`generate_catalog.py` writes, per database, under a chosen output root:

```
<root>/<db>/tables/<table>.md                  # DDL (+enums) + Columns table (type ⨝ description) + FKs + referenced-by
<root>/<db>/tables/_foreign_key_constraints.md # all PK/FK constraints in the db
<root>/<db>/knowledge_base/<knowledge>.md      # one file per KB node
```

The user generates these into two roots they pass per run:
`data/bird_interact/catalog_bird_interact_lite` and
`data/bird_interact/catalog_bird_interact_full`. An example exists at
`data/bird_interact/catalog_bird_interact_lite/alien`.

Key facts established from the example + code:

- A table file already contains DDL **and** per-column descriptions, so it fully
  replaces both the old `schema.sql` and `column_meanings.md`.
- A KB file contains **only its own node**: `# <knowledge>`, `description`,
  `definition`, and a prerequisite-edge block listing edges **by token** (e.g.
  `- "CCS" needs "SNQI"`). Prerequisite *definitions* are not inlined — so
  dropping a node's file removes that node's knowledge from the FS.
- A KB file's stem equals the node's `knowledge` name, which is exactly the key
  of `TaskData.masked_agent_kb` / `full_knowledge_base`
  (`_get_external_knowledge` keys by `entry["knowledge"]`; the catalog writes
  `f"{kb_name}.md"` over the same dict).
- Edge labels are **tokens** (`_extract_token`: the trailing parenthesized
  acronym, else the whole name). Tokens are **not unique** — the `alien` example
  has two distinct nodes whose token is `BFR`. So masked-ness cannot be decided
  reliably from a token alone.

## Design

### FS layout (per task)

`build_db_filesystem(task)` mounts only `<deep_catalog_root>/<selected_database>/`
under `/db`:

```
/db/tables/<table>.md                  ← read from disk, as-is
/db/tables/_foreign_key_constraints.md ← read from disk, as-is
/db/knowledge_base/<node>.md           ← re-rendered per task (see KB section)
```

The three old monolithic seed files are removed.

### Tables (read from disk)

For every `*.md` under `<deep_catalog_root>/<selected_database>/tables/`, read
the file text and seed it at `/db/tables/<filename>` as a `FileData`
(`{"content": text, "encoding": "utf-8"}`). Files are loaded verbatim; no
filtering.

If `<deep_catalog_root>/<selected_database>/` does not exist, raise
`FileNotFoundError` with a message naming the missing path and pointing at
`scripts/generate_catalog.py`.

> **Caveat (recorded):** `read_only_gt_tables` does **not** apply to the catalog
> FS — the catalog is introspected from the live DB and contains all tables,
> not the optionally GT-filtered dataset schema. Acceptable for deep_agent (a
> separate baseline). DDL/types also come from live introspection and may differ
> cosmetically from `ddl_database_schema`.

### Knowledge base (re-rendered per task, masking-aware) — option (b)

The on-disk KB files are produced by
`linearize_prerequisites(name, full_kb)`. At load time we instead call
`linearize_prerequisites(name, task.masked_agent_kb)` for each `name` in
`task.masked_agent_kb`, seeding the result at `/db/knowledge_base/<name>.md`.

Why re-render instead of reading the disk folder:

- **Identical format.** `linearize_prerequisites` is the exact function the
  catalog generator uses, so the rendered file is byte-identical to the on-disk
  file whenever none of the node's prerequisites are masked.
- **Faithful masking (option b).** When a prerequisite *is* masked, it is absent
  from `masked_agent_kb`, so `linearize_prerequisites` (a) never collects it and
  (b) never emits its edge — the dangling edge to a masked node disappears
  automatically. No token guessing, no leak of the masked node's name or
  definition.
- **Uniform across `make_data_ambiguous`.** When `False`,
  `masked_agent_kb == full_knowledge_base`, so the rendered KB is identical to
  the on-disk `knowledge_base/` folder. When `True`, masked nodes and their
  dangling edges are gone.

Empty `masked_agent_kb` → no `/db/knowledge_base/` files are seeded.

The generation step still produces the on-disk `knowledge_base/` folder; at
runtime it is a reference artifact only (the `tables/` trees are what get read).

### Wiring

- **`ConfigReader`** (`config_input.py`): add
  `deep_catalog_root: str = ''` (or a sensible default path), documented like
  the existing `deep_enable_*` flags.
- **`load_bird_interact_as_tasks`** (`bird_interact_reader.py`): add a
  `deep_catalog_root` parameter, thread it into each `TaskData` exactly like the
  `deep_enable_*` flags.
- **`TaskData`** (`state.py`): add `deep_catalog_root: str = ''`.
- **`build_db_filesystem`** (`filesystem_seed.py`): rewritten per above. Old
  `DB_FS_PATHS`, `_render_column_meanings`, `_render_knowledge_base`,
  `_text_file` helpers removed/replaced as needed.
- **`prompts.py`**: the system prompt advertises the new layout — explore
  `/db/tables/` (schema, columns, foreign keys) and `/db/knowledge_base/`
  (external knowledge) via `ls`/`glob`/`grep`/`read_file` — instead of the three
  fixed paths.
- **`deep_tool_costs` / FS tool set**: unchanged (still
  `ls`/`read_file`/`glob`/`grep`, plus writes under `deep_enable_fs_write`).

### Config plumbing note

`deep_catalog_root` is a per-run value the user sets to the lite or full catalog
root. It follows the same flag-over-YAML precedence as the other deep flags
(see `presets.py` / `cli_parser.py`); the static
`configs/eval_pipeline_config.yaml` must not re-declare it.

## Testing

- Seed test: build a small temp catalog (`<root>/<db>/tables/a.md`,
  `tables/_foreign_key_constraints.md`, and a `masked_agent_kb` with a masked
  prerequisite). Assert:
  - `/db/tables/*` mirror the on-disk files verbatim.
  - `/db/knowledge_base/<name>.md` exists only for nodes in `masked_agent_kb`.
  - A surviving node whose prerequisite is masked has **no** dangling
    `needs "<masked-token>"` edge in its rendered file.
  - `make_data_ambiguous=False` (`masked_agent_kb == full_knowledge_base`)
    reproduces the on-disk KB content.
- Missing-catalog test: `selected_database` with no catalog dir → `FileNotFoundError`.
- Run `uv run pytest tests/` after implementation (per repo convention).

## Out of scope

- Tuning subagents/todos/summarization middleware (existing flags unchanged).
- Applying `read_only_gt_tables` filtering to the catalog FS.
- Changing `generate_catalog.py`'s output format.
