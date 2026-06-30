#!/bin/bash
#SBATCH -A wjx@h100
#SBATCH -C h100
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --output=./logs/rl/%j.out
#SBATCH --nodes=1
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --time=05:00:00
#SBATCH --cpus-per-task=100
#SBATCH --signal=B:USR1@30   # USR1 30 s before wall-time so a requeue handler can checkpoint
#SBATCH --open-mode=append

# ---------------------------------------------------------------------------
# Single SLURM-submittable evaluation payload.
#
# Input axes (set via environment variables, all have defaults):
#   MODEL                    – model profile: qwen35 | gemma4-12B        (default: qwen35)
#   VARIANT                  – eval condition key (run 'just variants')  (required)
#   BASELINE                 – evaluation mode: no_tool | …             (default: no_tool)
#   PREDICTOR_MODEL_PROVIDER – LiteLLM provider                         (default: hosted_vllm)
#   CONCURRENCY              – concurrent tasks in the Python pipeline  (default: 16)
#   NUM_ITERATIONS           – repeat dataset N times                   (default: 1)
#   DEBUG                    – debug logging                            (default: false)
#   ENABLE_THINKING          – true|false; blank uses the profile default
#   TP                       – vLLM tensor-parallel size                (default: 1)
#   DP                       – vLLM data-parallel size                  (default: #GPUs in CUDA_VISIBLE_DEVICES)
#
# The model NAME, context length, thinking mode and vLLM server flags are
# resolved by src/conversation2sql/presets.py (the single source of truth).
# Per-run sampling/schema params are forwarded to `conv2sql run` as CLI flags
# (via RUN_ARGS + the --model-profile/--variant presets), not env vars.
#
# Launch via `just eval ...`, which sets these and dispatches through
# submit_and_log.sh.  Set DRY_RUN=1 to print the resolved config and exit.
# ---------------------------------------------------------------------------

set -Eeuo pipefail

export BASE_WORK="${BASE_WORK:-/workspaces/conversation2SQL}"
export MY_SLURM_JOB_ID="${MY_SLURM_JOB_ID:-local}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

MODEL="${MODEL:-qwen35}"
VARIANT="${VARIANT:?set VARIANT (run 'just variants' to list valid keys)}"
BASELINE="${BASELINE:-no_tool}"
PREDICTOR_MODEL_PROVIDER="${PREDICTOR_MODEL_PROVIDER:-hosted_vllm}"
CONCURRENCY="${CONCURRENCY:-16}"
NUM_ITERATIONS="${NUM_ITERATIONS:-1}"
DEBUG="${DEBUG:-false}"
ENABLE_THINKING="${ENABLE_THINKING:-}"   # blank → use the profile's default_thinking

# Count the GPUs handed to us via --gpus (CUDA_VISIBLE_DEVICES is a comma-separated
# device list, e.g. "0,1"). Used as the default data-parallel size below.
IFS=',' read -ra _GPU_ARR <<< "${CUDA_VISIBLE_DEVICES}"
_GPU_COUNT="${#_GPU_ARR[@]}"

TP="${TP:-1}"                            # vLLM tensor-parallel-size
DP="${DP:-${_GPU_COUNT}}"                # vLLM data-parallel-size (defaults to #GPUs)


# ---------------------------------------------------------------------------
# Model profile -> model name, context length, thinking mode, vLLM server flags.
#
# Resolved by the Python preset (single source of truth, presets.py) instead of
# a bash case. presets.py has no third-party imports, so the system python3 runs
# it fast and hermetically — no venv/uv, works in DRY_RUN too. The resolver emits
# NUL-delimited fields so JSON args (chat-template-kwargs) survive intact:
#   model_name \0 max_model_len \0 enable_thinking \0 <vllm serve args...>
# ---------------------------------------------------------------------------
mapfile -d '' _SERVER_CONFIG < <(
  python3 "${BASE_WORK}/src/conversation2sql/presets.py" server-config \
    --model-profile "${MODEL}" \
    --baseline "${BASELINE}" \
    --enable-thinking "${ENABLE_THINKING}" \
    --tp "${TP}" --dp "${DP}" \
    --base-work "${BASE_WORK}"
)
if (( ${#_SERVER_CONFIG[@]} < 4 )) || [ -z "${_SERVER_CONFIG[0]}" ]; then
  echo "[eval_payload] Failed to resolve server config for MODEL='${MODEL}' (run 'just variants' for valid profiles)" >&2
  exit 1
fi
MODEL_NAME="${_SERVER_CONFIG[0]}"
MAX_MODEL_LEN="${_SERVER_CONFIG[1]}"
ENABLE_THINKING="${_SERVER_CONFIG[2]}"          # resolved "true"/"false"
SERVER_ARGS=("${_SERVER_CONFIG[@]:3}")

# Only the model NAME is still needed bash-side (build_run_slug + vllm serve).
# All predictor sampling params come from the Python --model-profile preset.
export PREDICTOR_MODEL_NAME="${MODEL_NAME}"


# ---------------------------------------------------------------------------
# Build the CLI args forwarded to `conv2sql run`. Everything per-run goes
# through the CLI (highest-priority layer) so the static YAML cannot shadow it.
# The model profile + variant expand to predictor/reader flags inside Python.
# ---------------------------------------------------------------------------
RUN_ARGS=(
  --model-profile "${MODEL}"
  --variant "${VARIANT}"
  --baseline "${BASELINE}"
  --predictor_model_provider "${PREDICTOR_MODEL_PROVIDER}"
  --predictor_enable_thinking "${ENABLE_THINKING}"
  --concurrency "${CONCURRENCY}"
  --num-iterations "${NUM_ITERATIONS}"
  --debug "${DEBUG}"
)
# Ad-hoc ablation overrides from `just ... --extra "..."`. Word-split on spaces;
# values containing spaces are out of scope.
if [ -n "${EXTRA:-}" ]; then
  read -ra _EXTRA_ARR <<< "${EXTRA}"
  RUN_ARGS+=("${_EXTRA_ARR[@]}")
fi


# ---------------------------------------------------------------------------
# Dry run: print the resolved config and exit before any heavy setup.
# ---------------------------------------------------------------------------
if [ "${DRY_RUN:-0}" = "1" ]; then
  echo "[DRY-RUN] model=${MODEL} variant=${VARIANT} baseline=${BASELINE} provider=${PREDICTOR_MODEL_PROVIDER} concurrency=${CONCURRENCY} num_iterations=${NUM_ITERATIONS} gpus=${CUDA_VISIBLE_DEVICES}"
  printf '[DRY-RUN] vllm serve %q --max-model-len %q' "${MODEL_NAME}" "${MAX_MODEL_LEN}"
  printf ' %q' "${SERVER_ARGS[@]}"; printf '\n'
  printf '[DRY-RUN] conv2sql run --config configs/eval_pipeline_config.yaml'
  printf ' %s' "${RUN_ARGS[@]}"; printf '\n'
  exit 0
fi


# ---------------------------------------------------------------------------
# Real run: bring up the environment + a vLLM server, then evaluate.
# ---------------------------------------------------------------------------
source "${BASE_WORK}/bash_scripts/utils/utils_evaluate.sh"          # exports + run_suite()

log_section "Starting evaluation: model=${MODEL} variant=${VARIANT} provider=${PREDICTOR_MODEL_PROVIDER}" "${MY_SLURM_JOB_ID}"

if [ "${PREDICTOR_MODEL_PROVIDER}" = "hosted_vllm" ]; then
  # Local vLLM server: source the helper (registers cleanup trap) then start it.
  source "${BASE_WORK}/bash_scripts/utils/vllm_server.sh"
  start_vllm_server "$MODEL_NAME" "$MAX_MODEL_LEN" "${SERVER_ARGS[@]}"
  # PREDICTOR_VLLM_API_BASE is exported by vllm_server.sh after the server is up.
else
  # External API: unset api-base vars so Python gets None (not "").
  unset PREDICTOR_VLLM_API_BASE       || true
  unset USER_SIMULATOR_VLLM_API_BASE  || true
fi

run_suite "${RUN_ARGS[@]}"


# ---------------------------------------------------------------------------
# Post-run: copy results to WORK (Jean Zay persistent storage), SLURM only.
# ---------------------------------------------------------------------------
if [[ -n "${WORK:-}" ]]; then
  log_section "Moving results into WORK: ${WORK}" "${MY_SLURM_JOB_ID}"
  cp_files "${WORK}/evaluation_results" "${RESULTS_ROOT}" "${MY_SLURM_JOB_ID}"
else
  log_section "WORK is not set; skipping result copy-out" "${MY_SLURM_JOB_ID}"
fi
