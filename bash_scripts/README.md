# bash_scripts

Scripts for launching evaluation experiments locally (tmux) or on a SLURM cluster.

The entry point is the [`justfile`](../justfile) at the repo root. Recipes set a
couple of `EVAL_*` environment variables and hand a single payload script
(`eval_payload.sh`) to `submit_and_log.sh`, which runs it in a detached **tmux**
session locally or submits it with **sbatch** under SLURM. The same payload file
runs both ways — switching to SLURM is one environment variable.

## Prerequisites

- **`just`** — install the command runner (single static binary):
  ```bash
  curl --proto '=https' --tlsv1.2 -sSf https://just.systems/install.sh | bash -s -- --to /usr/local/bin
  ```
  (or `cargo install just`, or your distro package). Verify with `just --version`.
- A `.env` file at the repo root with secrets (`OPENAI_API_KEY`, `WANDB_API_KEY`, …).
  It is sourced by `evaluate.sh`.
- `uv` for the Python venv (see the top-level `CLAUDE.md`).

## Quick start

```bash
just                              # list recipes
just variants                     # list the valid variant keys
just eval all_db_all_kb           # local run: qwen35, GPU 1
just eval all_db_toon_all_kb 0,1  # local run on GPUs 0,1
just eval gt_db_gt_kb 1 gemma4    # gemma4 profile instead of qwen35
just dry all_db_all_kb            # print the resolved commands, launch nothing

# back-to-back local runs (waits for each tmux session before the next)
just sequential all_db_all_kb all_db_toon_all_kb
```

`just eval` prints a tmux session id; attach with `tmux attach -t <id>`. Logs are
tee'd to `results/<date>/<time>/tmux_log/{all,warning,error}.log` and the vLLM
server log to `tmux_log/vllm.log`.

### `eval` arguments

```
just eval VARIANT [gpus=1] [model=qwen35] [debug=false]
```

- **VARIANT** — eval condition; one of the keys from `just variants`.
- **gpus** — value for `CUDA_VISIBLE_DEVICES` (e.g. `1`, `0,1`).
- **model** — model profile: `qwen35` or `gemma4`.
- **debug** — `true` limits the run to a few tasks.

## The two axes

A run is a **model profile** × a **variant**. `eval_payload.sh`'s `case "$MODEL"`
now only builds the **`vllm serve` args** (model name, context length, server
flags). The **predictor sampling params** and the **variant→schema-flag** mapping
live in `src/conversation2sql/presets.py` and reach the pipeline as CLI flags via
`--model-profile` / `--variant` (expanded in `cli.py run`). Ad-hoc overrides go
through `just eval --variant <v> --extra "<flags>"`.

**Model profiles** set the model name, context length, sampling params, and vLLM
server flags:

| profile  | model                       | max-len | thinking |
|----------|-----------------------------|---------|----------|
| `qwen35` | `Qwen/Qwen3.5-9B`           | 50000   | true     |
| `gemma4` | `google/gemma-4-26B-A4B-it` | 32000   | false    |

**Variants** set the four `run_suite` condition flags:

| variant                          | schema_type | gt_tables | gt_kb | kb_linearized |
|----------------------------------|-------------|-----------|-------|---------------|
| `all_db_all_kb`                  | ddl         | false     | false | false         |
| `all_db_all_kb_linearized`       | ddl         | false     | false | true          |
| `all_db_toon_all_kb`             | toon        | false     | false | false         |
| `all_db_toon_all_kb_linearized`  | toon        | false     | false | true          |
| `gt_db_all_kb_linearized`        | ddl         | true      | false | true          |
| `gt_db_gt_kb_linearized`         | ddl         | true      | true  | true          |
| `gt_db_gt_kb`                    | ddl         | true      | true  | false         |

## Directory layout

```
bash_scripts/
├── eval_payload.sh        # The single SLURM-submittable payload (carries #SBATCH headers).
│                          #   Resolves EVAL_MODEL + EVAL_VARIANT, starts vLLM, runs run_suite.
├── evaluate.sh            # Sourced library: global exports + build_run_slug() + run_suite().
├── submit_and_log.sh      # Dispatcher: tmux locally, or sbatch when given a 2nd (job-name) arg.
├── evaluation_scripts/
│   └── gemma4/
│       └── tool_chat_template_gemma4.jinja   # referenced by the gemma4 profile
├── slurm/                 # Standalone multi-GPU / multi-node training scripts (separate workflow).
└── utils/
    ├── utils.sh                  # log_section, setup_idris, cp_files
    ├── vllm_server.sh            # start_vllm_server() + EXIT-trap cleanup
    ├── slurm_job_requeue.sh      # USR1 trap for SLURM preemption requeue
    ├── get_num_generations.py    # computes num_generations from GPU/batch config
    └── get_model_path_hf_cache.py# resolves a model id to its HF cache path
```

## How a run flows

```
just eval VARIANT ...                       (justfile, repo root)
  └─ exports EVAL_MODEL / EVAL_VARIANT / CUDA_VISIBLE_DEVICES / DEBUG
     └─ submit_and_log.sh eval_payload.sh [job-name]
          ├─ no job-name  → tmux session  (local)
          └─ job-name     → sbatch        (SLURM)
               └─ eval_payload.sh
                    ├─ source evaluate.sh        (exports + run_suite)
                    ├─ source utils/vllm_server.sh (start_vllm_server)
                    ├─ start a vLLM server on a free port
                    ├─ run_suite → `uv run conv2sql run …`
                    └─ copy results to $WORK   (SLURM only)
```

`submit_and_log.sh` forwards `EVAL_MODEL`, `EVAL_VARIANT`, `EVAL_BASELINE`,
`ENABLE_THINKING`, and `DEBUG` into the tmux session, and uses `sbatch --export=ALL`
so the same variables reach a SLURM job.

## How to switch to SLURM

The payload is already SLURM-ready — the only difference between a local and a
cluster run is the launcher.

1. **Flip the runner.** Set `RUNNER=slurm` so `submit_and_log.sh` takes its
   `sbatch` branch instead of tmux:
   ```bash
   RUNNER=slurm just eval all_db_all_kb
   RUNNER=slurm just sequential all_db_all_kb all_db_toon_all_kb   # submits all; sbatch queues them
   ```
   (Locally `sequential` waits for each tmux session; under SLURM it just submits.)

2. **Set your allocation.** The `#SBATCH` directives live at the top of
   `eval_payload.sh` (`-A`, `-C`, `--gpus-per-node`, `--qos`, `--time`, …). Edit
   them once for your project, or override per-submission on the command line —
   `sbatch` CLI flags win over the in-file `#SBATCH` directives. To override
   without editing the file, set `SBATCH_ARGS` style flags by adjusting the
   `sbatch` call in `submit_and_log.sh`, or just keep the headers correct.

3. **Environment reaches the job.** `submit_and_log.sh` submits with
   `sbatch --export=ALL`, so the `EVAL_*` variables exported by the recipe are
   visible inside the job. On clusters that restrict env propagation, confirm the
   variables survive (`env | grep EVAL_` early in the job log).

4. **Paths / offline mode.** Under SLURM, `submit_and_log.sh` derives `BASE_WORK`
   from `$SCRATCH` and calls `setup_idris` (offline HF/W&B, multi-node addressing).
   Results are copied to `$WORK/evaluation_results` after the run when `$WORK` is set.

5. **Multi-node training** lives separately under `slurm/` (`train_grpo*.sh`); it
   is its own workflow and is not driven by the justfile.

## Extending

- **New ablation knob** — add the field to `config_input.py`; use it ad hoc via
  `just eval --variant <v> --extra "--my_flag val"`. No bash/justfile edits.
- **New variant** — add an entry to `VARIANTS` in `src/conversation2sql/presets.py`
  (the four reader flags) and list the key in the `variants` recipe.
- **New model profile** — add an entry to `MODEL_PROFILES` in `presets.py`
  (sampling params) AND a `case` arm under "Model profile" in `eval_payload.sh`
  for the `vllm serve` args (`MODEL_NAME`, `MAX_MODEL_LEN`, `SERVER_ARGS`), then
  run `just eval --variant <v> --model <profile>`.
- **Check before launching** — `just dry <variant> [model]` prints the exact
  `vllm serve` and `run_suite` commands without starting anything.
