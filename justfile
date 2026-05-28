# conversation2SQL task runner.
#
# Front door over bash_scripts/. Recipes set the EVAL_* env and dispatch through
# bash_scripts/submit_and_log.sh, which runs the payload in a detached tmux
# session locally, or submits it with sbatch under SLURM.
#
# Quick reference:
#   just --list                           # list available recipes
#   just eval all_db_all_kb                              # local, qwen35, GPU 1, hosted_vllm
#   just eval all_db_all_kb qwen35                       # explicit model
#   just eval gt_db_gt_kb gemma4 0,1                     # gemma4, two GPUs
#   just eval all_db_all_kb qwen35 1 false openai        # qwen35 profile, OpenAI provider
#   just eval all_db_all_kb qwen35 1 false openrouter    # via OpenRouter
#   just eval all_db_all_kb qwen35 1 false together_ai   # via Together AI
#   RUNNER=slurm just eval all_db_all_kb                 # submit via sbatch instead of local tmux
#   just sequential all_db_all_kb all_db_toon_all_kb
#   just sequential qwen35 1 openrouter all_db_all_kb    # sequential with OpenRouter provider
#   just dry all_db_all_kb                               # print resolved commands without launching

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
# Parameters (all have defaults — only `variant` is required):
#   variant   — one of the keys printed by `just variants`  (e.g. all_db_all_kb)
#   model     — "qwen35" or "gemma4"                        (default: qwen35)
#               Selects sampling params and server args; model name can be overridden
#               at the shell level with EVAL_MODEL_NAME= if needed for external APIs.
#   gpus      — comma-separated CUDA device IDs             (default: "1" → device 1)
#               e.g.  gpus="0,1"  to use two GPUs
#               Ignored (but harmless) when provider != hosted_vllm.
#   debug     — "true" to enable debug logging              (default: false)
#   provider    — LiteLLM provider for the predictor          (default: hosted_vllm)
#                 hosted_vllm  → start a local vLLM server (needs GPU)
#                 openai       → use OpenAI API  (export OPENAI_API_KEY first)
#                 openrouter   → use OpenRouter  (export OPENROUTER_API_KEY first)
#                 together_ai  → use Together AI (export TOGETHER_API_KEY first)
#   baseline    — evaluation mode passed to run_suite         (default: no_tool)
#                 no_tool | tools_only | tools_user | bird_full
#   concurrency — tasks processed concurrently by the Python  (default: 16)
#                 pipeline; raise to saturate the vLLM server
#
# Under RUNNER=local  the job runs in a new detached tmux session.
# Under RUNNER=slurm  sbatch receives a job name of "eval_<model>_<variant>_<provider>".
eval variant model="qwen35" gpus="1" debug="false" provider="hosted_vllm" baseline="no_tool" concurrency="16":
    #!/usr/bin/env bash
    set -Eeuo pipefail
    # Export env vars read by eval_payload.sh and the Python pipeline.
    export EVAL_MODEL="{{model}}"
    export EVAL_VARIANT="{{variant}}"
    export EVAL_PROVIDER="{{provider}}"
    export EVAL_BASELINE="{{baseline}}"
    export EVAL_CONCURRENCY="{{concurrency}}"
    export CUDA_VISIBLE_DEVICES="{{gpus}}"   # which GPU(s) the vLLM server may use
    export DEBUG="{{debug}}"
    if [ "{{runner}}" = "slurm" ]; then
        # Pass a human-readable job name to sbatch so it appears in squeue output.
        {{dispatch}} "eval_{{model}}_{{variant}}_{{provider}}"
    else
        # Local path: submit_and_log.sh opens a tmux session; no job name needed.
        {{dispatch}}
    fi

# ── dry ───────────────────────────────────────────────────────────────────────
# Print the vLLM server command + run_suite command that `eval` would execute,
# without actually launching anything. Useful for inspecting the resolved config.
dry variant model="qwen35" provider="hosted_vllm" baseline="no_tool" concurrency="16":
    DRY_RUN=1 EVAL_MODEL="{{model}}" EVAL_VARIANT="{{variant}}" EVAL_PROVIDER="{{provider}}" EVAL_BASELINE="{{baseline}}" EVAL_CONCURRENCY="{{concurrency}}" bash bash_scripts/eval_payload.sh

# ── sequential ────────────────────────────────────────────────────────────────
# Run several variants one after another (waits for each to finish before starting
# the next when RUNNER=local; under SLURM all jobs are queued immediately).
#
# Parameters (all apply to every variant in the list):
#   model       — "qwen35" or "gemma4"               (default: qwen35)
#   gpus        — comma-separated CUDA device IDs    (default: "1")
#   provider    — LiteLLM provider (see `eval`)      (default: hosted_vllm)
#   baseline    — evaluation mode (see `eval`)       (default: no_tool)
#   concurrency — pipeline concurrency (see `eval`)  (default: 16)
#   variants    — zero or more variant names (variadic, must come last)
#
# Usage examples:
#   just sequential all_db_all_kb all_db_toon_all_kb                              # defaults
#   just sequential                                                                # default pair
#   just sequential gemma4 0,1 hosted_vllm no_tool 16 all_db_all_kb              # explicit all
#   just sequential qwen35 1 openrouter no_tool 8 all_db_all_kb gt_db_gt_kb      # OpenRouter, 8 workers
#
# NOTE: model, gpus, provider, baseline, concurrency are positional and must appear
# before the variant names. The *variants variadic captures all remaining words.
sequential model="qwen35" gpus="1" provider="hosted_vllm" baseline="no_tool" concurrency="16" *variants:
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
        echo "[sequential] Launching: ${variant} (model={{model}}, gpus={{gpus}}, provider={{provider}}, baseline={{baseline}}, concurrency={{concurrency}})"
        echo "============================================================"
        # Delegate to the single-variant `eval` recipe; capture combined stdout+stderr.
        # Named-arg syntax avoids positional coupling with debug (kept at its default).
        if ! out=$(just eval variant="${variant}" model="{{model}}" gpus="{{gpus}}" provider="{{provider}}" baseline="{{baseline}}" concurrency="{{concurrency}}" 2>&1); then
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
