# Merge knowledge-base overview into database_overview.md — design

**Date:** 2026-07-01
**Component:** `src/conversation2sql/eval_framework/agents/deep_agent/`, `src/conversation2sql/eval_framework/agents/utils_kb_linearize.py`

## Goal

The deep_agent's catalog directory currently gives the agent no way to discover
what external-knowledge (KB) terms exist for a task without blindly listing
`knowledge_base/` first: filenames there are shell-safe slugs (e.g.
`eer_rank.md`), not the human-readable term names, so the agent cannot tell
from the listing alone which file defines which term without opening several.

Add an index of every surviving KB entry — name, exact filename, one-line
description — merged into the existing `database_overview.md`, so a single
`cat database_overview.md` gives the agent both the table map (already there)
and the KB map, with no extra tool call.

## Current state

- `scripts/generate_catalog.py::render_database_overview_markdown` writes a
  static `<catalog_root>/<db>/database_overview.md` per database at
  catalog-generation time: a `## Description` (hand-curated,
  `DATABASE_DESCRIPTIONS`) and a `## Tables` bullet list (hand-curated,
  `TABLE_DESCRIPTIONS`). This file does not vary per task.
- `deep_agent/catalog_seed.py::materialize_catalog_dir(task)` copies that file
  verbatim into the per-task temp dir if present, silently skipping it if
  absent.
- The KB, by contrast, **does** vary per task: `materialize_catalog_dir` never
  reads the on-disk `knowledge_base/` folder. It re-renders one file per
  surviving `task.masked_agent_kb` entry via
  `utils_kb_linearize.linearize_prerequisites`, keyed by a slug filename from
  `utils_kb_linearize.build_kb_filenames`. Masked entries and their dangling
  edges are never emitted (faithful per-sample masking — see
  `2026-05-21-kb-linearization-agent-side-design.md`).

Because the KB overview must reflect this same per-task masking, it cannot be
pre-rendered once into the on-disk `database_overview.md` (that file is
DB-level, not task-level) — it must be computed at the same point the
individual KB files are: inside `materialize_catalog_dir`, from
`task.masked_agent_kb`.

## Design

### New helper: `utils_kb_linearize.build_kb_overview`

```python
def build_kb_overview(
    masked_agent_kb: dict[str, ExternalKnowledgeEntry],
    filenames: dict[str, str],
) -> str:
    ...
```

- Iterates `masked_agent_kb` in dict order (same order `materialize_catalog_dir`
  already iterates to write the per-entry files).
- One line per entry: `- **{entry.knowledge}** (\`knowledge_base/{filenames[name]}.md\`): {entry.description}`.
- Returns `""` when `masked_agent_kb` is empty (no section to add — mirrors the
  existing "silently skip when absent" convention already used for
  `database_overview.md` itself).
- `filenames` is passed in rather than recomputed, since callers
  (`catalog_seed.py`) already build it once via `build_kb_filenames` to name
  the per-entry files — reuse, don't recompute.

### `catalog_seed.materialize_catalog_dir` changes

After the existing copy-if-present step for `database_overview.md`, append the
KB overview section (blank line + `## Knowledge Base` heading + the entries
from `build_kb_overview`) to that same file, **only if** the section is
non-empty **and** the base file exists on disk (unchanged skip-when-absent
behavior — no new file is fabricated purely to hold the KB section).

Net effect on the per-task temp dir's `database_overview.md`:

```
# database: robot

## Description
...

## Tables
- **actuation_data**: ...
...

## Knowledge Base
- **EER Rank** (`knowledge_base/eer_rank.md`): Ranks the Energy Efficiency Ratio...
- **Energy Efficiency Ratio (EER)** (`knowledge_base/energy_efficiency_ratio_eer.md`): ...
```

### `prompts.py` changes

Update the catalog-layout bullets and the "explore the catalog" strategy tip so
the agent is told `database_overview.md` also lists KB terms with their exact
filenames, and drop the `ls knowledge_base/` discovery step (no longer
needed — the filename is already known from the overview):

- Bullet list: `database_overview.md` bullet gains "...and an index of every
  external-knowledge entry (with its exact filename under `knowledge_base/`)".
- Strategy tip: replace "run `ls knowledge_base/` and `cat` the exact filename
  it lists rather than guessing" with "look up the exact filename in
  `database_overview.md`'s Knowledge Base index, then `cat
  knowledge_base/<filename>.md`".

### Addendum (same day): `generate_catalog.py` also renders the index

Follow-up request: the on-disk `database_overview.md` (the one a human opens
directly, e.g. under `data/bird_interact/catalog_bird_interact_lite/<db>/`)
should show the KB index too, not just the runtime copy. So
`render_database_overview_markdown` gained an optional `external_kb` param —
when given (the full, unmasked KB via `_get_external_knowledge`), it appends
the same `## Knowledge Base` section using `build_kb_overview` +
`build_kb_filenames`. `generate_catalog_for_db` now passes it in.

This reintroduces the leak risk flagged as the reason to reject "option B"
above: the disk file now *does* carry an unmasked KB section.
`materialize_catalog_dir` handles it by **stripping** that section before
appending its own masked one, rather than copying-then-appending:
`base, _, _ = text.partition("\n## Knowledge Base")` keeps everything before
the marker (table content, present regardless of the split matching), then
the masked section is appended fresh. No section ever survives from disk —
either there wasn't one (`partition` finds nothing, `base` is the whole file)
or there was one and it's discarded outright.

### Out of scope

- No dependency-edge / DAG information in the overview (that's what individual
  `knowledge_base/<file>.md` files already show via
  `linearize_prerequisites`); the overview is a flat index only.
- No change to `knowledge_base/<file>.md` rendering itself.

## Testing

- `tests/eval_framework/agents/test_utils_kb_linearize.py`: new tests for
  `build_kb_overview` — one line per entry with correct name/filename/
  description; empty KB → `""`.
- `tests/eval_framework/agents/deep_agent/test_catalog_seed.py`: update
  `test_materialize_copies_tables_overview_and_renders_masked_kb` (the file is
  no longer copied byte-for-byte verbatim — it now also contains the `##
  Knowledge Base` section) and add a case asserting a masked-out KB entry does
  not appear in the merged overview. `test_materialize_skips_overview_when_absent`
  keeps asserting no file is fabricated when the disk file is absent, even if
  `masked_agent_kb` is non-empty.
- `tests/eval_framework/agents/deep_agent/test_prompts.py`: assert the new
  wording (mentions the KB index in `database_overview.md`) renders.
- `uv run pytest tests/` after implementation (repo convention).

## Docs

Update `deep_agent/README.md` and `deep_agent/CLAUDE.md` catalog tables: the
`database_overview.md` row gains "and an index of KB entries (name → exact
filename → description), re-merged per task from `masked_agent_kb`" alongside
its existing "copied if present" note.
