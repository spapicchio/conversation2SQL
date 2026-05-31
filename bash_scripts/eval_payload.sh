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
#   MODEL                    – model profile: qwen35 | gemma4           (default: qwen35)
#   VARIANT                  – eval condition key (run 'just variants')  (required)
#   BASELINE                 – evaluation mode: no_tool | …             (default: no_tool)
#   PREDICTOR_MODEL_PROVIDER – LiteLLM provider                         (default: hosted_vllm)
#   CONCURRENCY              – concurrent tasks in the Python pipeline  (default: 16)
#   NUM_ITERATIONS           – repeat dataset N times                   (default: 1)
#   DEBUG                    – debug logging                            (default: false)
#
# After resolving the model profile and variant, all sampling params and
# schema flags are exported as Python-compatible env vars so PydanticParser
# reads them directly — no CLI flag translation needed in run_suite.
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


# ---------------------------------------------------------------------------
# Model profile -> model name, context length, sampling, vLLM server flags.
# ---------------------------------------------------------------------------
case "$MODEL" in
  qwen35)
    # https://huggingface.co/Qwen/Qwen3.5-9B
    # Thinking (coding):  temp=0.6 top_p=0.95 top_k=20 presence=0.0
    # Non-thinking (gen): temp=1.0 top_p=0.95 top_k=20 presence=1.5
    MODEL_NAME="Qwen/Qwen3.5-9B"
    MAX_MODEL_LEN=50000
    ENABLE_THINKING="${ENABLE_THINKING:-true}"
    if [ "$ENABLE_THINKING" = true ]; then
      TEMPERATURE=0.6; TOP_P=0.95; TOP_K=20; PRESENCE_PENALTY=0.0; REPETITION_PENALTY=1.0
      DEFAULT_PARAMS='{"enable_thinking": true}'
    else
      TEMPERATURE=1.0; TOP_P=0.95; TOP_K=20; PRESENCE_PENALTY=1.5; REPETITION_PENALTY=1.0
      DEFAULT_PARAMS='{"enable_thinking": false}'
    fi
    SERVER_ARGS=(
      --tensor-parallel-size 1
      --data-parallel-size 1
      --reasoning-parser qwen3
      --default-chat-template-kwargs "$DEFAULT_PARAMS"
      --language-model-only
    )
    ;;
  gemma4)
    # https://huggingface.co/google/gemma-4-26B-A4B-it
    MODEL_NAME="google/gemma-4-26B-A4B-it"
    MAX_MODEL_LEN=32000
    ENABLE_THINKING="${ENABLE_THINKING:-false}"
    TEMPERATURE=1.0; TOP_P=0.95; TOP_K=64; PRESENCE_PENALTY=0.0; REPETITION_PENALTY=1.0
    if [ "$ENABLE_THINKING" = true ]; then
      DEFAULT_PARAMS='{"enable_thinking": true}'
    else
      DEFAULT_PARAMS='{"enable_thinking": false}'
    fi
    SERVER_ARGS=(
      --tensor-parallel-size 1
      --data-parallel-size 1
      --reasoning-parser gemma4
      --chat-template "${BASE_WORK}/bash_scripts/utils/tool_chat_template_gemma4.jinja"
      --default-chat-template-kwargs "$DEFAULT_PARAMS"
      --limit-mm-per-prompt '{"image": 0, "audio": 0}'
    )
    ;;
  *)
    echo "[eval_payload] Unknown MODEL='$MODEL' (expected: qwen35 | gemma4)" >&2
    exit 1
    ;;
esac

# Only the model NAME is still needed bash-side (build_run_slug + vllm serve).
# All predictor sampling params now come from the Python --model-profile preset.
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
