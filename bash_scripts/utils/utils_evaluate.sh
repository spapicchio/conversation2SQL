#!/bin/bash
# Shared evaluation library. SOURCED by bash_scripts/eval_payload.sh — not run or
# submitted directly. SLURM #SBATCH headers live in eval_payload.sh, not here.
#
# Provides: global path/env exports, build_run_slug(), run_suite().

# --- robust shell settings ---
# -E: ERR trap inherited by functions/subshells
# -e: exit on error  -u: error on unset vars  -o pipefail: catch pipe failures
set -Eeuo pipefail


# ---------------------------------------------------------------------------
# Paths & global exports
# ---------------------------------------------------------------------------
export HOME_WORK="/workspaces"
export BASE_WORK="${HOME_WORK}/conversation2SQL"
export BASE_WORK_DATA="${BASE_WORK}/data"
export BASE_WORK_MODEL="${BASE_WORK}/model_trained"
export WANDB_PROJECT='conversation2SQL'
export HF_HOME="/hf_cache"   # shared HuggingFace cache visible inside the container

# Single source of truth for results output root — used by run_suite and the explorer.
export RESULTS_ROOT="${BASE_WORK}/results"

# Allow the caller to override which GPUs are used (e.g. CUDA_VISIBLE_DEVICES=0,1 ./evaluate.sh)
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2}"

source "${BASE_WORK}/.env"                        # API keys (OPENAI_API_KEY, WANDB_API_KEY, …)
source "${BASE_WORK}/bash_scripts/utils/utils.sh" # shared helpers: cp_files, setup_idris, …


# ---------------------------------------------------------------------------
# Python environment
# ---------------------------------------------------------------------------
uv sync --frozen                              # ensure lockfile-exact deps are installed
source "${BASE_WORK}/.venv/bin/activate"      # activate the project venv

cd "${BASE_WORK}"                           # ensure we're in the repo root for relative paths

NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l || echo 0)
log_section "Visible ${NUM_GPUS} GPUs." "${MY_SLURM_JOB_ID:-}"


# ---------------------------------------------------------------------------
# Shared configuration
# ---------------------------------------------------------------------------
export OMP_NUM_THREADS=50   # limit OpenMP threads to avoid CPU oversubscription


# ---------------------------------------------------------------------------
# build_run_slug
#
# Prints a short identifier for the run, e.g.:
#   no_tool__Qwen3.5-9B__ddl__lin__gt-db
#
# Reads from the Python-compatible env vars set by eval_payload.sh:
#   BASELINE, PREDICTOR_MODEL_NAME, DATABASE_SCHEMA_TYPE,
#   IS_KB_LINEARIZED, READ_ONLY_GT_TABLES, READ_ONLY_GT_KB
# ---------------------------------------------------------------------------
build_run_slug() {
  local baseline="${BASELINE:-no_tool}"
  local model_name="${PREDICTOR_MODEL_NAME:-}"
  local schema_type="${DATABASE_SCHEMA_TYPE:-ddl}"
  local is_lin="${IS_KB_LINEARIZED:-false}"
  local gt_db="${READ_ONLY_GT_TABLES:-false}"
  local gt_kb="${READ_ONLY_GT_KB:-false}"

  # Take only the last component of the model path (e.g. "Qwen/Qwen3.5-9B" → "Qwen3.5-9B"),
  # then replace any character that is not alphanumeric / dot / underscore / hyphen with "-",
  # and strip any trailing hyphens that might result.
  local model_slug
  model_slug=$(basename "${model_name}" | sed 's/[^A-Za-z0-9._-]/-/g; s/-*$//')

  local slug="${baseline}__${model_slug}__${schema_type}"
  [[ "${is_lin}" == "true" ]] && slug="${slug}__lin"
  [[ "${gt_db}"  == "true" ]] && slug="${slug}__gt-db"
  [[ "${gt_kb}"  == "true" ]] && slug="${slug}__gt-kb"

  echo "${slug}"
}


# ---------------------------------------------------------------------------
# run_suite
#
# Reads all run parameters from env vars (set by eval_payload.sh) so no
# CLI flag translation is needed.  Creates a dated output directory under
# RESULTS_ROOT, exports OUTPUT_FOLDER, then launches the Python pipeline.
#
# Directory layout:
#   RESULTS_ROOT/<YYYY_MM_DD>/<HH_MM_SS>__<slug>/
# ---------------------------------------------------------------------------
run_suite() {
  local slug
  slug=$(build_run_slug)

  # Build and create the output directory for this specific run.
  # When launched via submit_and_log.sh, DEST_DIR is already set to the
  # timestamped directory that also holds tmux_log/ — use it directly so
  # results and logs are co-located.  Fall back to a fresh dated dir otherwise.
  local run_dir
  if [[ -n "${DEST_DIR:-}" ]]; then
    run_dir="${DEST_DIR}/${slug}"
  else
    local date_dir time_tag
    date_dir="$(date +%Y_%m_%d)"
    time_tag="$(date +%H_%M_%S)"
    run_dir="${RESULTS_ROOT}/${date_dir}/${time_tag}__${slug}"
  fi
  mkdir -p "${run_dir}"

  log_section "Running suite: ${BASELINE:-no_tool} → ${run_dir}" "${MY_SLURM_JOB_ID:-}"

  # Per-run params arrive as CLI flags (passed through from eval_payload's
  # RUN_ARGS via "$@"). output_folder and the vLLM api-base are computed here,
  # so pass them as flags too — this is also what stops the static YAML from
  # shadowing them.
  local extra_flags=("$@")
  extra_flags+=(--output_folder "${run_dir}")
  if [ -n "${PREDICTOR_VLLM_API_BASE:-}" ]; then
    extra_flags+=(--predictor_vllm_api_base "${PREDICTOR_VLLM_API_BASE}")
  fi
  if [ -n "${USER_SIMULATOR_VLLM_API_BASE:-}" ]; then
    extra_flags+=(--user_simulator_vllm_api_base "${USER_SIMULATOR_VLLM_API_BASE}")
  fi

  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  TOKENIZERS_PARALLELISM=true \
  VLLM_WORKER_MULTIPROC_METHOD=spawn \
  uv run conv2sql run \
    --config "${BASE_WORK}/configs/eval_pipeline_config.yaml" \
    "${extra_flags[@]}"

  log_section "=== Done ${BASELINE:-no_tool} ===" "${MY_SLURM_JOB_ID:-}"
}
