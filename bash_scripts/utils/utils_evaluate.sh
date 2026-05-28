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
# build_run_slug <baseline> "$@"   (remaining flags forwarded from run_suite)
#
# Prints a short identifier for the run, e.g.:
#   no_tool__Qwen3.5-9B__ddl__lin__gt-db
#
# Components (separated by "__"):
#   1. baseline name        – what evaluation mode is used
#   2. model basename       – last path component of --predictor_model_name,
#                             with any characters outside [A-Za-z0-9._-] replaced by "-"
#   3. schema type          – value of --database_schema_type (default: ddl)
#   4. optional suffixes    – "lin"   if --is_kb_linearized true
#                             "gt-db" if --read_only_gt_tables true
#                             "gt-kb" if --read_only_gt_kb     true
# ---------------------------------------------------------------------------
build_run_slug() {
  local baseline="$1"
  shift

  # Walk the remaining flag-value pairs and capture the ones we care about.
  # We scan by index so we can safely skip the value after each matched flag.
  local model_name="" schema_type="ddl" is_lin="false" gt_db="false" gt_kb="false"
  local args=("$@")
  local i
  for ((i = 0; i < ${#args[@]}; i++)); do
    case "${args[$i]}" in
      --predictor_model_name)  model_name="${args[$((i+1))]}";  i=$((i+1)) ;;
      --database_schema_type)  schema_type="${args[$((i+1))]}"; i=$((i+1)) ;;
      --is_kb_linearized)      is_lin="${args[$((i+1))]}";      i=$((i+1)) ;;
      --read_only_gt_tables)   gt_db="${args[$((i+1))]}";       i=$((i+1)) ;;
      --read_only_gt_kb)       gt_kb="${args[$((i+1))]}";       i=$((i+1)) ;;
    esac
  done

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
# run_suite <baseline> <predictor_api_base> <user_sim_api_base> [extra flags...]
#
# Creates a dated output directory under RESULTS_ROOT, then launches the
# evaluation pipeline pointing at it.  Directory layout:
#   RESULTS_ROOT/<YYYY_MM_DD>/<HH_MM_SS>__<slug>/
# ---------------------------------------------------------------------------
run_suite() {
  local baseline="$1"
  local predictor_vllm_api_base="$2"
  local user_simulator_vllm_api_base="$3"
  shift 3

  # Derive the human-readable slug from the flags that follow.
  local slug
  slug=$(build_run_slug "${baseline}" "$@")

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

  log_section "Running suite: ${baseline} → ${run_dir}" "${MY_SLURM_JOB_ID:-}"

  local launcher=(
    uv run conv2sql run
    --config "${BASE_WORK}/configs/eval_pipeline_config.yaml"
    --baseline "${baseline}"
    --predictor_vllm_api_base "${predictor_vllm_api_base}"
    --user_simulator_vllm_api_base "${user_simulator_vllm_api_base}"
    --output_folder "${run_dir}"   # Python writes results.jsonl directly here
  )

  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  TOKENIZERS_PARALLELISM=true \
  VLLM_WORKER_MULTIPROC_METHOD=spawn \
  "${launcher[@]}" \
  "$@"   # forward all remaining flags (model params, schema type, debug, …) to Python

  log_section "=== Done ${baseline} ===" "${MY_SLURM_JOB_ID:-}"
}
