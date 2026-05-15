# conversation2SQL

# Installation

We are using the base image provided by [VeRL](https://verl.readthedocs.io/en/latest/start/install.html#install-from-docker-image): https://hub.docker.com/r/vllm/vllm-openai.

It already contains VLLM and Flash-Attn:

```bash
python3 -c "import verl; print(verl.__version__); print(verl.__file__)"
>> 0.8.0.dev
>> /workspaces/conversation2SQL/verl/verl/__init__.py

python3 -c "import flash_attn; print(flash_attn.__version__); print(flash_attn.__file__)"
>> 2.8.1
>> /usr/local/lib/python3.12/dist-packages/flash_attn/__init__.py
```

Please Note that all the system packages are managed in the devcontainer.json files.

Note that verl is installed as a git submodule/clone, 
if you want to update verl you can ```git pull``` inside verl.

```bash
flashinfer show-config
# flashinfer configurations with JIT and CUBIN installed 
>>  === Version Info ===
>> FlashInfer version: 0.6.3
>> flashinfer-cubin version: 0.6.3
>> flashinfer-jit-cache version: 0.6.3+cu129
>> === Torch Version Info ===
>> Torch version: 2.9.1+cu129
>> CUDA runtime available: Yes
```

Okay now we can create an UV venv based on the system-site-packages:

```bash
# Create the venv based on the system site packages directory
uv venv --system-site-packages
# install custom packages
uv sync 
# activate env
source .venv/bin/activate
```

# Running Experiments

## Evaluation pipeline

The pipeline evaluates an LLM agent on the [BIRD-Interact](https://bird-bench.github.io/) text-to-SQL benchmark. It requires two PostgreSQL containers (defined in `.devcontainer/docker-compose.yml`):

- `localhost:5432` — BIRD-Interact **lite** dataset
- `localhost:5433` — BIRD-Interact **full** dataset

Both use credentials `root / 123123`.

Run with:

```bash
uv run python main.py --config configs/eval_pipeline_config.yaml
```

Config priority is **defaults → env vars → YAML → CLI flags**. CLI flags always win. Unique field names can be used bare (`--debug`, `--baseline`); shared fields use section prefixes (`--predictor_model_name`).

## Ablation baselines

Four ablation baselines are selectable via the `baseline` config field (`ConfigPipeline.baseline`). Each isolates a different component of the full agent:

| `baseline` | Query | DB tools | `ask_user` | Purpose |
|---|---|---|---|---|
| `no_tool` | clean | — | — | Lower bound: classical text-to-SQL, single-shot |
| `tools_only` | clean | ✅ | — | Value of agentic DB exploration alone |
| `tools_user` | clean | ✅ | ✅ | Effect of `ask_user` on an unambiguous query |
| `bird_full` | ambiguous | ✅ | ✅ | Full BIRD-Interact agent (paper baseline) |

Run a specific baseline via `--baseline`:

```bash
# No-tool lower bound (single-shot text-to-SQL)
uv run python main.py --config configs/eval_pipeline_config.yaml --baseline no_tool

# Agentic with DB tools only (no user interaction)
uv run python main.py --config configs/eval_pipeline_config.yaml --baseline tools_only

# Agentic with DB tools + ask_user (clean query)
uv run python main.py --config configs/eval_pipeline_config.yaml --baseline tools_user

# Full BIRD-Interact agent with ambiguous query (default)
uv run python main.py --config configs/eval_pipeline_config.yaml --baseline bird_full
```

`make_data_ambiguous` is automatically overridden by the chosen baseline (only `bird_full` uses ambiguous queries); a warning is logged if the YAML sets it inconsistently.

## Configuration

Edit `configs/eval_pipeline_config.yaml` or pass CLI flags to override:

```bash
# Switch model and run one task (debug=true stops after first task)
uv run python main.py \
    --config configs/eval_pipeline_config.yaml \
    --baseline tools_only \
    --predictor_model_name gpt-4o \
    --predictor_model_provider openai \
    --debug true

# Full run with more patience budget
uv run python main.py \
    --config configs/eval_pipeline_config.yaml \
    --baseline bird_full \
    --reader_user_patience_budget 10 \
    --debug false
```

## CLI Commands

The `conv2sql` command-line interface provides a convenient way to run experiments and inspect results.

### Run experiments

```bash
# Run with default config
uv run conv2sql run

# Run with custom config
uv run conv2sql run --config configs/eval_pipeline_config.yaml

# Run with config + CLI overrides
uv run conv2sql run --config configs/eval_pipeline_config.yaml --baseline no_tool --debug true
```

The `run` subcommand forwards all arguments to the underlying `PydanticParser`, so any config option can be overridden via CLI flags (see **Configuration** section above for examples).

### Inspect results

View all past experiment runs as a summary table:

```bash
# Show all results
uv run conv2sql results

# Show results from a custom directory
uv run conv2sql results --dir /path/to/results
```

Drill into a specific run to see per-task metrics:

```bash
# Show per-task details for a specific run
uv run conv2sql results --run bird_full/2026_05_07/12_00_00

# From a custom results directory
uv run conv2sql results --dir /custom/path --run no_tool/2026_05_07/09_00_00
```

The summary table shows:
- **baseline**: ablation baseline used
- **date**: YYYY_MM_DD (UTC)
- **time**: HH_MM_SS (UTC)
- **model**: LLM model name
- **tasks**: number of tasks in the run
- **exec_acc**: mean execution accuracy (0–1)
- **avg_cost**: mean cost in USD per task

The drill-down table shows per-task metrics:
- **instance_id**: task identifier
- **exec_accuracy**: 1 if SQL correct, 0 otherwise
- **total_cost**: cost in USD for this task
- **submit_msg**: first 80 chars of the final `submit_sql` message

### Classify turn traces

Use the `classify-turns` command to label every AI turn in one or more `results_smaller.jsonl` files and write an enriched JSONL file with `turn_classifications` appended to each record:

Classification uses two label sets:

| Level | Label | Meaning |
|---|---|---|
| L2 | `SQL_SUBMISSION` | Any `submit_sql` call present |
| L2 | `USER_INTERACTION` | Any `ask_user` call, no `submit_sql` |
| L2 | `DB_EXPLORATION` | Only DB-type tools |
| L2 | `KNOWLEDGE_LOOKUP` | Only KB-type tools |
| L2 | `MIXED` | Tools span more than one non-priority category |
| L2 | `TEXT_ONLY` | No tools, non-empty content |
| L2 | `NO_ACTION` | No tools, empty content |
| L2 | `UNKNOWN` | A tool is present but missing from the YAML config |
| L1 | `ANSWER_ATTEMPT` | Direct attempt to answer |
| L1 | `REVISION` | Correcting a prior failed submit |
| L1 | `CLARIFICATION` | Asking the user for missing information |
| L1 | `INTERROGATION` | Probing the user for details |
| L1 | `ASSUMPTION` | Making a unilateral assumption |
| L1 | `CONFIRMATION` | Confirming the user's prior answer |
| L1 | `DISCUSSION` | General reasoning or back-and-forth |
| L1 | `HEDGING` | Uncertain or cautious phrasing |
| L1 | `REFUSAL` | Refusing to answer or act |
| L1 | `MISSING` | No clear semantic label could be assigned |

```bash
# Classify a single run trace
uv run conv2sql classify-turns results/bird_full/2026_05_07/12_00_00/results_smaller.jsonl

# Classify multiple traces into a custom output file
uv run conv2sql classify-turns \
  results/bird_full/2026_05_07/*/results_smaller.jsonl \
  --output results/classified.jsonl \
  --model openai/gpt-4o-mini \
  --tool-categories configs/turn_classifier/tool_categories.yaml
```

If you do not pass `--output`, the command writes `results_classified.jsonl` next to the single input file, or to the current directory when classifying multiple inputs. After writing the enriched JSONL, it prints Rich summary tables for the L2 categories, L1 categories, confidence levels, and a per-instance breakdown.

## Evaluate with bash script

`bash_scripts/evaluate.sh` is a convenience wrapper for running the evaluation pipeline — either **locally via tmux** or on a **SLURM cluster** (H100 GPUs).

### Local run (tmux)

When executed outside SLURM, the script automatically:

1. Creates a timestamped log directory under `logs/local/<YYYY-MM-DD>/<HH-MM-SS>/`.
2. Launches each dataset evaluation in a **detached tmux session** so the terminal can be closed without interrupting the run.

```bash
# Set required env vars, then run
MODEL_NAME=my-model \
  MAX_NEW_TOKENS=2048 \
  MAX_MOD_LENGTH=8192 \
  TENSOR_PARALLEL_SIZE=2 \
  USER_PROMPT_NAME=my_prompt.jinja \
  SYSTEM_PROMPT_NAME=system.jinja \
  ENABLE_THINKING_MODE=false \
  RUN_ONLY_PREDICTIONS=false \
  bash bash_scripts/evaluate.sh

# Attach to a running session
tmux attach -t eval_20260513_120000_12345_<label>

# List all active sessions
tmux ls
```

### SLURM run

Submit as a SLURM batch job (H100 partition):

```bash
sbatch bash_scripts/evaluate.sh
```

SLURM output is written to `./logs/rl/<job_id>.out`. After the job completes, results are copied to `$WORK/evaluation_results/` if `$WORK` is set.

### Configuration

The following environment variables control the run (export them or prepend to the command):

| Variable | Default | Description |
|---|---|---|
| `MODEL_NAME` | *(required)* | Model path or HuggingFace id |
| `CUDA_VISIBLE_DEVICES` | `1,2` | GPU indices to use |
| `GREEDY_TEMP` | `0.0` | Sampling temperature |
| `GREEDY_TOP_P` | `1.0` | Top-p nucleus sampling |
| `GREEDY_TOP_K` | `-1` | Top-k (disabled by default) |
| `GREEDY_REP_PENALTY` | `1.0` | Repetition penalty |
| `GREEDY_NUM_SAMPLES` | `1` | Number of completions per prompt |
| `MAX_NEW_TOKENS` | — | Maximum tokens to generate |
| `MAX_MOD_LENGTH` | — | Maximum model context length |
| `TENSOR_PARALLEL_SIZE` | — | Number of GPUs for tensor parallelism |
| `USER_PROMPT_NAME` | — | Jinja prompt template name |
| `SYSTEM_PROMPT_NAME` | — | System prompt template name |
| `ENABLE_THINKING_MODE` | — | Enable chain-of-thought thinking |
| `RUN_ONLY_PREDICTIONS` | — | Skip evaluation, only generate predictions |

## Output

Results are written as JSONL under a per-baseline subfolder:

```
results/
  <baseline>/
    <YYYY_MM_DD>/
      <HH_MM_SS>/
        results.jsonl        # full record per task
        results_smaller.jsonl  # smaller projection
        config.yaml          # exact config that ran
```

Each record includes `execution_accuracy`, `tool_calls_in_order`, token/cost metrics, and the full message history.

## Tests

```bash
# Unit and mocked tests (fast, no DB required)
uv run pytest tests/ --ignore=tests/eval_framework/integration

# Integration smoke tests (requires live Postgres + API keys)
uv run pytest tests/eval_framework/integration/ -m integration

# Type checking
uv run pyrefly check
```
