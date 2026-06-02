# conversation2SQL task runner.
#
# Front door over bash_scripts/. Recipes set the EVAL_* env and dispatch through
# bash_scripts/submit_and_log.sh, which runs the payload in a detached tmux
# session locally, or submits it with sbatch under SLURM.
#
# Quick reference:
#   just --usage eval
#   just --list                                                                       # list available recipes
#   just eval --variant all_db_all_kb                                                 # local, qwen35, GPU 1, hosted_vllm
#   just eval --variant all_db_all_kb --model qwen35                                  # explicit model
#   just eval --variant gt_db_gt_kb --model gemma4 --gpus 0,1                        # gemma4, two GPUs
#   just eval --variant all_db_all_kb --model qwen35 --provider openai               # qwen35 profile, OpenAI provider
#   just eval --variant all_db_all_kb --model qwen35 --provider openrouter           # via OpenRouter
#   just eval --variant all_db_all_kb --model qwen35 --provider together_ai          # via Together AI
#   just eval --variant all_db_all_kb --num-iterations 5                             # 5 statistical iterations
#   RUNNER=slurm just eval --variant all_db_all_kb                                   # submit via sbatch instead of local tmux
#   just sequential all_db_all_kb all_db_toon_all_kb
#   just sequential --model qwen35 --provider openrouter all_db_all_kb               # sequential with OpenRouter provider
#   just dry --variant all_db_all_kb                                                  # print resolved commands without launching

# Use bash with -u (error on unset vars) and -c (read from string) for all recipes.
set shell := ["bash", "-uc"]

# ── Runtime selector ──────────────────────────────────────────────────────────
# "local"  → runs inside a detached tmux session on this machine.
# "slurm"  → submits the job with sbatch (for HPC clusters).
# Override for a single invocation:  RUNNER=slurm just eval ...
runner := env_var_or_default("RUNNER", "local")

# The dispatcher that wraps every eval launch.
# submit_and_log.sh decides whether to open a tmux session or call sbatch,
# then forwards the actual work to eval_payload.sh.
dispatch := "bash bash_scripts/submit_and_log.sh bash_scripts/eval_payload.sh"
# Same dispatcher, but runs the resume/recover payload instead of eval.
dispatch_recover := "bash bash_scripts/submit_and_log.sh bash_scripts/recover_payload.sh"



# ── Recipes ───────────────────────────────────────────────────────────────────

# List available recipes (default when you run bare `just`).
default:
    @just --list

# Sync the project venv from the lockfile (run once after cloning or after uv.lock changes).
setup:
    uv sync --frozen

# Print the valid EVAL_VARIANT keys.
# Each variant controls which DB subset and knowledge-base combination is used:
#   all_db           → use every database in the benchmark
#   gt_db            → use only the "ground-truth" databases
#   all_kb           → inject the full external knowledge base
#   gt_kb            → inject only the ground-truth KB entries
#   toon_all_kb      → apply cartoon/simplified KB
#   linearized       → represent the KB as linearized text instead of structured JSON
variants:
    @printf '%s\n' \
      all_db_all_kb \
      all_db_all_kb_linearized \
      all_db_toon_all_kb \
      all_db_toon_all_kb_linearized \
      gt_db_all_kb_linearized \
      gt_db_gt_kb_linearized \
      gt_db_gt_kb

# ── eval ──────────────────────────────────────────────────────────────────────
# Run ONE evaluation variant.
#
# Parameters (all have defaults — only --variant is required):
#   --variant        — one of the keys printed by `just variants`  (e.g. all_db_all_kb)
#   --model          — "qwen35" or "gemma4"                        (default: qwen35)
#                      Selects sampling params and server args; model name can be overridden
#                      at the shell level with EVAL_MODEL_NAME= if needed for external APIs.
#   --gpus           — comma-separated CUDA device IDs             (default: "1" → device 1)
#                      e.g.  --gpus 0,1  to use two GPUs
#                      Ignored (but harmless) when provider != hosted_vllm.
#   --debug          — "true" to enable debug logging              (default: false)
#   --provider       — LiteLLM provider for the predictor          (default: hosted_vllm)
#                      hosted_vllm  → start a local vLLM server (needs GPU)
#                      openai       → use OpenAI API  (export OPENAI_API_KEY first)
#                      openrouter   → use OpenRouter  (export OPENROUTER_API_KEY first)
#                      together_ai  → use Together AI (export TOGETHER_API_KEY first)
#   --baseline       — evaluation mode passed to run_suite         (default: no_tool)
#                      no_tool | tools_only | tools_user | bird_full
#   --concurrency    — tasks processed concurrently by the Python  (default: 16)
#                      pipeline; raise to saturate the vLLM server
#   --num-iterations — repeat the dataset N times for statistical  (default: 1)
#                      relevance; collapses to 1 when predictor
#                      temperature=0 (deterministic runs add no info)
#
# Under RUNNER=local  the job runs in a new detached tmux session.
# Under RUNNER=slurm  sbatch receives a job name of "eval_<model>_<variant>_<provider>".

#   ┌────────────┬─────────────────────┬─────────────────────────┬──────────┬─────────────────┐
#   │  baseline  │ make_data_ambiguous │         runner          │ user-sim │ enable_ask_user │
#   ├────────────┼─────────────────────┼─────────────────────────┼──────────┼─────────────────┤
#   │ no_tool    │ False               │ run_baseline_no_tool    │ False    │ —               │
#   ├────────────┼─────────────────────┼─────────────────────────┼──────────┼─────────────────┤
#   │ tools_only │ False               │ run_agent_bird_baseline │ False    │ False           │
#   ├────────────┼─────────────────────┼─────────────────────────┼──────────┼─────────────────┤
#   │ tools_user │ False               │ run_agent_bird_baseline │ True     │ True            │
#   ├────────────┼─────────────────────┼─────────────────────────┼──────────┼─────────────────┤
#   │ bird_full  │ True                │ run_agent_bird_baseline │ True     │ True            │
#   └────────────┴─────────────────────┴─────────────────────────┴──────────┴─────────────────┘

[arg("variant", long="variant", help="one of the keys printed by `just variants` (e.g. all_db_all_kb_linearized)")]
[arg("model", long="model", help="model profile: qwen35 or gemma4 (default: qwen35)")]
[arg("gpus", long="gpus", help="comma-separated CUDA device IDs (default: 1)")]
[arg("debug", long="debug", help="true to enable debug logging (default: false)")]
[arg("provider", long="provider", help="LiteLLM provider: hosted_vllm | openai | openrouter | together_ai (default: hosted_vllm)")]
[arg("baseline", long="baseline", help="evaluation mode: no_tool | tools_only | tools_user | bird_full (default: no_tool)")]
[arg("concurrency", long="concurrency", help="tasks processed concurrently by the Python pipeline (default: 16)")]
[arg("num_iterations", long="num-iterations", help="repeat dataset N times for statistical relevance (default 1); collapses to 1 when predictor temperature=0")]
[arg("extra", long="extra", help="extra flags forwarded verbatim to `conv2sql run`, quoted (e.g. --extra \"--predictor_top_p 0.8\")")]
eval variant="all_db_all_kb_linearized" model="qwen35" gpus="1" debug="false" provider="hosted_vllm" baseline="no_tool" concurrency="16" num_iterations="1" extra="":
    #!/usr/bin/env bash
    set -Eeuo pipefail
    # Export env vars read by eval_payload.sh and the Python pipeline.
    export MODEL="{{model}}"
    export VARIANT="{{variant}}"
    export BASELINE="{{baseline}}"
    export PREDICTOR_MODEL_PROVIDER="{{provider}}"
    export CONCURRENCY="{{concurrency}}"
    export NUM_ITERATIONS="{{num_iterations}}"
    export CUDA_VISIBLE_DEVICES="{{gpus}}"   # which GPU(s) the vLLM server may use
    export DEBUG="{{debug}}"
    export EXTRA="{{extra}}"                 # ad-hoc flags forwarded to conv2sql run
    if [ "{{runner}}" = "slurm" ]; then
        # Pass a human-readable job name to sbatch so it appears in squeue output.
        {{dispatch}} "eval_{{model}}_{{variant}}_{{provider}}"
    else
        # Local path: submit_and_log.sh opens a tmux session; no job name needed.
        {{dispatch}}
    fi

# ── recover ────────────────────────────────────────────────────────────────────
# Re-run only the missing/errored (instance_id, iteration) pairs of a previous run,
# appending results into that same run directory. Replays the run's own config.yaml
# snapshot, so the original model/variant/baseline settings are reused automatically.
#
#   just recover results/2026_06_01/13_55_45__no_tool__Qwen3.5-9B__ddl
#
# run_dir must be the leaf directory that contains config.yaml (and results_iter*.jsonl).
[arg("run_dir", help="run directory to resume (the dir holding config.yaml)")]
recover run_dir:
    #!/usr/bin/env bash
    set -Eeuo pipefail
    if [ ! -f "{{run_dir}}/config.yaml" ]; then
        echo "[recover] No config.yaml in '{{run_dir}}' — pass the leaf run dir." >&2
        exit 1
    fi
    export RESUME_DIR="{{run_dir}}"
    if [ "{{runner}}" = "slurm" ]; then
        {{dispatch_recover}} "recover_$(basename '{{run_dir}}')"
    else
        {{dispatch_recover}}
    fi

# ── dry ───────────────────────────────────────────────────────────────────────
# Print the vLLM server command + run_suite command that `eval` would execute,
# without actually launching anything. Useful for inspecting the resolved config.
[arg("variant", long="variant", help="one of the keys printed by `just variants` (e.g. all_db_all_kb_linearized)")]
[arg("model", long="model", help="model profile: qwen35 or gemma4 (default: qwen35)")]
[arg("provider", long="provider", help="LiteLLM provider: hosted_vllm | openai | openrouter | together_ai (default: hosted_vllm)")]
[arg("baseline", long="baseline", help="evaluation mode: no_tool | tools_only | tools_user | bird_full (default: no_tool)")]
[arg("concurrency", long="concurrency", help="tasks processed concurrently by the Python pipeline (default: 16)")]
[arg("num_iterations", long="num-iterations", help="repeat dataset N times for statistical relevance (default 1)")]
[arg("extra", long="extra", help="extra flags forwarded verbatim to `conv2sql run`")]
dry variant="all_db_all_kb_linearized" model="qwen35" provider="hosted_vllm" baseline="no_tool" concurrency="16" num_iterations="1" extra="":
    DRY_RUN=1 MODEL="{{model}}" VARIANT="{{variant}}" BASELINE="{{baseline}}" PREDICTOR_MODEL_PROVIDER="{{provider}}" CONCURRENCY="{{concurrency}}" NUM_ITERATIONS="{{num_iterations}}" EXTRA="{{extra}}" bash bash_scripts/eval_payload.sh

# ── sequential ────────────────────────────────────────────────────────────────
# Run several variants one after another (waits for each to finish before starting
# the next when RUNNER=local; under SLURM all jobs are queued immediately).
#
# Parameters (all apply to every variant in the list):
#   --model          — "qwen35" or "gemma4"               (default: qwen35)
#   --gpus           — comma-separated CUDA device IDs    (default: "1")
#   --provider       — LiteLLM provider (see `eval`)      (default: hosted_vllm)
#   --baseline       — evaluation mode (see `eval`)       (default: no_tool)
#   --concurrency    — pipeline concurrency (see `eval`)  (default: 16)
#   --num-iterations — iterations per variant (see `eval`) (default: 1)
#   variants         — zero or more variant names (variadic, must come last)
#
# Usage examples:
#   just sequential all_db_all_kb all_db_toon_all_kb                                        # defaults
#   just sequential                                                                          # default pair
#   just sequential --model gemma4 --gpus 0,1 --provider hosted_vllm all_db_all_kb         # explicit flags
#   just sequential --model qwen35 --provider openrouter --concurrency 8 all_db_all_kb gt_db_gt_kb
#   just sequential --num-iterations 5 all_db_all_kb all_db_toon_all_kb                    # 5 iterations
[arg("model", long="model", help="model profile: qwen35 or gemma4 (default: qwen35)")]
[arg("gpus", long="gpus", help="comma-separated CUDA device IDs (default: 1)")]
[arg("provider", long="provider", help="LiteLLM provider: hosted_vllm | openai | openrouter | together_ai (default: hosted_vllm)")]
[arg("baseline", long="baseline", help="evaluation mode: no_tool | tools_only | tools_user | bird_full (default: no_tool)")]
[arg("concurrency", long="concurrency", help="tasks processed concurrently by the Python pipeline (default: 16)")]
[arg("num_iterations", long="num-iterations", help="iterations per variant (default 1); see eval --num-iterations")]
[arg("extra", long="extra", help="extra flags forwarded verbatim to each variant's `conv2sql run`")]
sequential model="qwen35" gpus="1" provider="hosted_vllm" baseline="no_tool" concurrency="16" num_iterations="1" extra="" *variants:
    #!/usr/bin/env bash
    set -Eeuo pipefail
    # Expand the variadic just parameter into a bash array.
    variants=({{variants}})
    # Default pair when called with no arguments.
    if [ "${#variants[@]}" -eq 0 ]; then
        variants=(all_db_all_kb all_db_toon_all_kb)
    fi
    failed=()   # collects names of variants that failed to launch
    for variant in "${variants[@]}"; do
        echo "============================================================"
        echo "[sequential] Launching: ${variant} (model={{model}}, gpus={{gpus}}, provider={{provider}}, baseline={{baseline}}, concurrency={{concurrency}}, num_iterations={{num_iterations}})"
        echo "============================================================"
        # Delegate to the single-variant `eval` recipe; capture combined stdout+stderr.
        # Named-arg syntax avoids positional coupling with debug (kept at its default).
        if ! out=$(just eval variant="${variant}" model="{{model}}" gpus="{{gpus}}" provider="{{provider}}" baseline="{{baseline}}" concurrency="{{concurrency}}" --num-iterations "{{num_iterations}}" --extra "{{extra}}" 2>&1); then
            echo "$out"
            echo "[sequential] ERROR launching ${variant} — skipping."
            failed+=("${variant}")
            continue
        fi
        echo "$out"
        if [ "{{runner}}" = "slurm" ]; then
            continue   # sbatch has queued the job; the scheduler handles ordering
        fi
        # Local mode: extract the tmux session name printed by submit_and_log.sh
        # so we can poll for its exit before starting the next variant.
        session=$(echo "$out" | grep 'Generated fake job ID' | awk '{print $NF}')
        if [ -z "${session}" ]; then
            echo "[sequential] ERROR: could not extract tmux session id — continuing."
            failed+=("${variant}")
            continue
        fi
        echo "[sequential] Waiting for tmux session '${session}' (attach: tmux attach -t ${session})"
        # Poll every 3 s until the tmux session disappears (job finished or crashed).
        while tmux has-session -t "${session}" 2>/dev/null; do
            sleep 3
        done
        echo "[sequential] Session '${session}' exited — next."
    done
    echo "============================================================"
    if [ "${#failed[@]}" -gt 0 ]; then
        echo "[sequential] Failed: ${failed[*]}"
        exit 1
    fi
    echo "[sequential] All variants finished."
