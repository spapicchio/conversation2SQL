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
# Three orthogonal axes are selected via environment variables:
#   EVAL_MODEL    – model profile: qwen35 | gemma4            (default: qwen35)
#   EVAL_VARIANT  – eval condition (the 4 run_suite flags)    (required)
#   EVAL_PROVIDER – LiteLLM provider passed as --predictor_model_provider
#                   hosted_vllm (default) | openai | openrouter | together_ai
#                   When NOT hosted_vllm the local vLLM server is skipped entirely;
#                   make sure the matching API key env var is exported beforehand
#                   (e.g. OPENAI_API_KEY, OPENROUTER_API_KEY, TOGETHER_API_KEY).
#
# Launch through `just eval ...`, which sets these and dispatches via
# submit_and_log.sh (tmux locally, sbatch under SLURM). Runnable standalone too.
#
# Set DRY_RUN=1 to print the resolved vLLM + run_suite commands and exit without
# starting a server — used to verify the variant->flag mapping without GPUs.
# ---------------------------------------------------------------------------

set -Eeuo pipefail

export BASE_WORK="${BASE_WORK:-/workspaces/conversation2SQL}"
export MY_SLURM_JOB_ID="${MY_SLURM_JOB_ID:-local}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

EVAL_MODEL="${EVAL_MODEL:-qwen35}"
EVAL_VARIANT="${EVAL_VARIANT:?set EVAL_VARIANT (run 'just variants' to list valid keys)}"
EVAL_BASELINE="${EVAL_BASELINE:-no_tool}"
# LiteLLM provider string forwarded to --predictor_model_provider.
# hosted_vllm → start a local vLLM server; anything else → skip server, use external API.
EVAL_PROVIDER="${EVAL_PROVIDER:-hosted_vllm}"
# Number of tasks processed concurrently by the Python pipeline.
EVAL_CONCURRENCY="${EVAL_CONCURRENCY:-16}"
DEBUG="${DEBUG:-false}"


# ---------------------------------------------------------------------------
# Model profile -> model name, context length, sampling, vLLM server flags.
# ---------------------------------------------------------------------------
case "$EVAL_MODEL" in
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
    echo "[eval_payload] Unknown EVAL_MODEL='$EVAL_MODEL' (expected: qwen35 | gemma4)" >&2
    exit 1
    ;;
esac


# ---------------------------------------------------------------------------
# Variant -> the 4 run_suite condition flags.
#   *_toon_*       -> --database_schema_type toon (else ddl)
#   gt_db_*        -> --read_only_gt_tables true   (else false)
#   *_gt_kb*       -> --read_only_gt_kb true        (else false)
#   *_linearized   -> --is_kb_linearized true       (else false)
# ---------------------------------------------------------------------------
case "$EVAL_VARIANT" in
  all_db_all_kb)                 SCHEMA_TYPE=ddl;  GT_DB=false; GT_KB=false; IS_LIN=false ;;
  all_db_all_kb_linearized)      SCHEMA_TYPE=ddl;  GT_DB=false; GT_KB=false; IS_LIN=true  ;;
  all_db_toon_all_kb)            SCHEMA_TYPE=toon; GT_DB=false; GT_KB=false; IS_LIN=false ;;
  all_db_toon_all_kb_linearized) SCHEMA_TYPE=toon; GT_DB=false; GT_KB=false; IS_LIN=true  ;;
  gt_db_all_kb_linearized)       SCHEMA_TYPE=ddl;  GT_DB=true;  GT_KB=false; IS_LIN=true  ;;
  gt_db_gt_kb_linearized)        SCHEMA_TYPE=ddl;  GT_DB=true;  GT_KB=true;  IS_LIN=true  ;;
  gt_db_gt_kb)                   SCHEMA_TYPE=ddl;  GT_DB=true;  GT_KB=true;  IS_LIN=false ;;
  *)
    echo "[eval_payload] Unknown EVAL_VARIANT='$EVAL_VARIANT'." >&2
    echo "Valid: all_db_all_kb all_db_all_kb_linearized all_db_toon_all_kb all_db_toon_all_kb_linearized gt_db_all_kb_linearized gt_db_gt_kb_linearized gt_db_gt_kb" >&2
    exit 1
    ;;
esac


# ---------------------------------------------------------------------------
# Assemble the run_suite flags (everything after baseline + the two api bases).
# ---------------------------------------------------------------------------
RUN_SUITE_ARGS=(
  --predictor_model_provider "${EVAL_PROVIDER}"
  --predictor_model_name "${MODEL_NAME}"
  --predictor_temperature "${TEMPERATURE}"
  --predictor_top_p "${TOP_P}"
  --predictor_top_k "${TOP_K}"
  --predictor_presence_penalty "${PRESENCE_PENALTY}"
  --predictor_repetition_penalty "${REPETITION_PENALTY}"
  --predictor_enable_thinking "${ENABLE_THINKING}"
  --database_schema_type "${SCHEMA_TYPE}"
  --make_data_ambiguous false
  --read_only_gt_tables "${GT_DB}"
  --read_only_gt_kb "${GT_KB}"
  --is_kb_linearized "${IS_LIN}"
  --concurrency "${EVAL_CONCURRENCY}"
  --debug "${DEBUG}"
)


# ---------------------------------------------------------------------------
# Dry run: print the resolved commands and exit before any heavy setup.
# ---------------------------------------------------------------------------
if [ "${DRY_RUN:-0}" = "1" ]; then
  echo "[DRY-RUN] model=${EVAL_MODEL} variant=${EVAL_VARIANT} baseline=${EVAL_BASELINE} provider=${EVAL_PROVIDER} concurrency=${EVAL_CONCURRENCY} gpus=${CUDA_VISIBLE_DEVICES}"
  printf '[DRY-RUN] vllm serve %q --max-model-len %q' "${MODEL_NAME}" "${MAX_MODEL_LEN}"
  printf ' %q' "${SERVER_ARGS[@]}"; printf '\n'
  printf '[DRY-RUN] run_suite %q' "${EVAL_BASELINE}"
  printf ' %q' "${RUN_SUITE_ARGS[@]}"; printf '\n'
  exit 0
fi


# ---------------------------------------------------------------------------
# Real run: bring up the environment + a vLLM server, then evaluate.
# ---------------------------------------------------------------------------
source "${BASE_WORK}/bash_scripts/utils/utils_evaluate.sh"          # exports + run_suite()

log_section "Starting evaluation: model=${EVAL_MODEL} variant=${EVAL_VARIANT} provider=${EVAL_PROVIDER}" "${MY_SLURM_JOB_ID}"

if [ "${EVAL_PROVIDER}" = "hosted_vllm" ]; then
  # Local vLLM server: source the helper (registers cleanup trap) then start it.
  source "${BASE_WORK}/bash_scripts/utils/vllm_server.sh"
  start_vllm_server "$MODEL_NAME" "$MAX_MODEL_LEN" "${SERVER_ARGS[@]}"
else
  # External API (openai / openrouter / together_ai / …): no local server needed.
  # The Python pipeline uses EVAL_PROVIDER + the model name via LiteLLM directly.
  # Ensure the matching API key is exported before calling this script, e.g.:
  #   export OPENAI_API_KEY=sk-...
  #   export OPENROUTER_API_KEY=sk-...
  #   export TOGETHER_API_KEY=...
  PREDICTOR_VLLM_API_BASE=""
  USER_SIMULATOR_VLLM_API_BASE=""
fi

run_suite "$EVAL_BASELINE" \
  "$PREDICTOR_VLLM_API_BASE" \
  "$USER_SIMULATOR_VLLM_API_BASE" \
  "${RUN_SUITE_ARGS[@]}"


# ---------------------------------------------------------------------------
# Post-run: copy results to WORK (Jean Zay persistent storage), SLURM only.
# ---------------------------------------------------------------------------
if [[ -n "${WORK:-}" ]]; then
  log_section "Moving results into WORK: ${WORK}" "${MY_SLURM_JOB_ID}"
  cp_files "${WORK}/evaluation_results" "${RESULTS_ROOT}" "${MY_SLURM_JOB_ID}"
else
  log_section "WORK is not set; skipping result copy-out" "${MY_SLURM_JOB_ID}"
fi
