# bash_scripts

Utility scripts for launching experiments locally (via tmux) or on a SLURM cluster.

## Directory layout

```
bash_scripts/
├── submit_and_log.sh          # Entry point — wraps any job script for local or SLURM execution
├── evaluate.sh                # Shared evaluation library: defines run_suite() used by evaluation_scripts/
├── grpo.sh                    # GRPO RL training (local)
├── sft.sh                     # SFT training (SLURM)
├── evaluation_scripts/        # Per-model evaluation scripts; each sources evaluate.sh and calls run_suite()
│   └── qwen_3.5_local.sh      # Qwen3.5-9B: spins up a vLLM server and runs the no_tool baseline
├── slurm/                     # SLURM-specific multi-GPU / multi-node training scripts
└── utils/
    ├── utils.sh                   # Shared functions: log_section, launch_vllm, cp_files
    ├── slurm_job_requeue.sh       # USR1 trap for SLURM preemption requeue
    ├── utils_clenup_vllm_if_crash.sh  # EXIT/ERR trap to kill vLLM on crash
    ├── get_num_generations.py     # Computes num_generations from GPU/batch config
    └── get_model_path_hf_cache.py # Resolves a model ID to its HF cache path
```

## Prerequisites

- `BASE_WORK` is set automatically by `submit_and_log.sh`: from `${SCRATCH}/conversation2SQL` when running under SLURM, or derived from the script's own path otherwise. You do not need to set it manually.
- A `.env` file at `${BASE_WORK}/.env` with secrets (e.g. `WANDB_API_KEY`, API keys).

## How to launch an experiment

### Local (tmux)

```bash
bash bash_scripts/submit_and_log.sh bash_scripts/evaluation_scripts/qwen_3.5_local.sh
```

`submit_and_log.sh` will:
1. Generate a short job ID from the current timestamp (`FAKE_JOB_ID`).
2. Copy the job script to `bash_scripts/launched/<date>/`.
3. Spawn a detached tmux session named after the job ID.
4. Tee all output to `tmux_log/<date>/<id>/all.log`, with separate `warning.log` and `error.log` filters.

Attach to the running session:

```bash
tmux attach -t <FAKE_JOB_ID>
```

### SLURM

Pass a second argument (the SLURM job name) to trigger `sbatch`:

```bash
bash bash_scripts/submit_and_log.sh bash_scripts/evaluation_scripts/qwen_3.5_local.sh my_job_name
```

The script will `sbatch` the copied job file, symlink SLURM's `.out` log into `tmux_log/`, and record the submission in `log_sbatch.log`.

## How evaluation scripts work

Each script under `evaluation_scripts/` follows the same pattern:

1. Source `evaluate.sh` (defines the `run_suite` helper and shared env).
2. Start a vLLM server in the background on a free port.
3. Wait up to 60 s for the server to become healthy.
4. Call `run_suite <baseline> <predictor_api_base> <user_simulator_api_base> <temperature> <top_p> <top_k> <presence_penalty> <repetition_penalty> <enable_thinking>`.

`run_suite` invokes `uv run conv2sql run` with the config at `${BASE_WORK}/config/config_evaluate.yaml` and the provided sampling parameters.

### Adding a new model

1. Copy an existing script from `evaluation_scripts/` as a template (e.g. `qwen_3.5_local.sh`).
2. Set `MODEL_NAME`, `MAX_MODEL_LEN`, `ENABLE_THINKING`, and the corresponding sampling parameters.
3. Adjust `--tensor-parallel-size` / `--data-parallel-size` and `CUDA_VISIBLE_DEVICES` to match your GPU allocation.
4. Run via `submit_and_log.sh` (local) or submit with `sbatch`.

## Key environment variables

| Variable | Default | Description |
|---|---|---|
| `BASE_WORK` | auto-detected by `submit_and_log.sh` | Project root (set from `SCRATCH` or script path) |
| `SCRATCH` | — | HPC scratch dir (set by SLURM env) |
| `WORK` | — | HPC long-term storage dir (optional; enables result copy-out) |
| `CUDA_VISIBLE_DEVICES` | `1,2` | GPUs for the job (local runs) |
| `MY_SLURM_JOB_ID` | auto | Injected by `submit_and_log.sh`; used as folder/run name |

## Adding a non-evaluation experiment (training, etc.)

1. Copy an existing script (`grpo.sh`, `sft.sh`) as a template.
2. Set the `#SBATCH` headers if you plan to use SLURM.
3. Source `${BASE_WORK}/bash_scripts/utils/utils.sh` for `log_section` and related helpers.
4. Launch via `submit_and_log.sh` (local tmux) or with a second argument for `sbatch`.
