# conversation2sql (top-level package)

Entry point for all shared infrastructure: config models, CLI parser, and logging.

## Key files

- `config_input.py` — four Pydantic config models: `ConfigPipeline`, `ConfigReader`, `ConfigPredictor`, `ConfigUserSimulator`. All pipeline parameters live here. `ConfigReader` uses `dataset_variant` (`'lite'`/`'full'`) as the single switch for dataset paths and DB port (computed fields); see Gotchas below.
- `cli_parser.py` — `PydanticParser` merges defaults → env vars → YAML (`--config`) → CLI flags. Section names derived from class names (`ConfigPredictor` → `predictor`). Fields unique across configs work with bare names (`--debug`, `--mode`); shared fields are auto-prefixed.
- `logger.py` — thin wrapper around the standard library logger; call `get_logger(__name__)` in every module.

## Gotchas

- Config priority is **defaults → env vars → YAML → CLI**. CLI flags always win.
- Section-prefixed CLI flags look like `--predictor_model_name`, not `--model_name`. Check `PydanticParser` if a flag isn't being picked up.
- `presets.py` maps `--model-profile`/`--variant` names to CLI flags; `cli.py run` expands and **prepends** them so explicit `--extra`/passthrough overrides win (argparse keeps the last occurrence of a repeated flag).
- The eval bash flow sends all per-run params as CLI flags (top priority), so the now-static `configs/eval_pipeline_config.yaml` never shadows them. Do not re-add a per-run field (baseline, predictor sampling, reader schema flags, output_folder, vllm api-base) to that YAML — it would silently override the flag.
- `sft_dataset_generation/` is a placeholder package with only `__init__.py` — nothing functional there yet.
- `ConfigReader` exposes one `dataset_variant: Literal['lite','full']` switch. `dataset_path`, `dataset_name_jsonl`, `deep_catalog_root`, and `db_dsn_template` are read-only `@computed_field`s derived from it (+ `data_root`, `db_host`, `db_user`, `db_password`). They are not settable via CLI/YAML; flip the variant instead. The lite/full Postgres port (5432/5433) is derived from the variant; host/creds stay overridable for SLURM.
