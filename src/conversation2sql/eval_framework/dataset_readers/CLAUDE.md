# dataset_readers

Loads and prepares BIRD-Interact tasks from JSONL + per-DB asset files.

## Entry function

`load_bird_interact_as_tasks(dataset_path, dataset_name_jsonl, filter_query_category, db_dsn_template, user_patience_budget, make_data_ambiguous)` returns `list[TaskData]`.

## Per-DB asset structure

For each `db_name` the reader expects under `dataset_path/`:
- `{db_name}/{db_name}_{database_schema_type}.txt` — schema file; `database_schema_type` defaults to `"ddl"`, so the default file is `{db_name}_ddl.txt`
- `{db_name}/{db_name}_column_meaning_base.json` — column meanings
- `{db_name}/{db_name}_kb.jsonl` — external knowledge base

Schema, column meanings, and KB are loaded with `@cache` (keyed on `(dataset_path, db_name)`) — they are read once per DB per process.

## Budget formula

`task_budget = 6 + 2 * m_amb + 2 * user_patience_budget`

where `m_amb = len(critical_ambiguity) + len(knowledge_ambiguity)` for each task.

## make_data_ambiguous flag

- `True` (default): agent sees the ambiguous query and a masked KB (entries involved in KB ambiguity are deleted).
- `False`: agent sees the unambiguous query and the full KB. Useful for ablations.

## Skipped instances

Twelve hard-coded `instance_id` values are always skipped (`skipped_instance_id` set in `load_bird_interact_as_tasks` — 6 from DB FULL, 6 from DB LITE) due to known dataset issues. Do not remove them without verifying those tasks are fixed upstream.

## Full dataset (bird-interact-full)

When `"full"` appears in `dataset_name_jsonl`, `_load_not_ambig_query_from_livesqlbench()` is called to join unambiguous queries from `data/livesqlbench-base-full-v1/livesqlbench_data.jsonl`. That file must exist on the host data mount.

