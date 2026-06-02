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
#   no_tool__Qwen3.5-9B__ddl__lin__gt-db__iter5
#
# baseline + model name come from env vars (BASELINE, PREDICTOR_MODEL_NAME); the
# schema/gt/linearized parts come from the VARIANT via presets.py — the single
# source of truth. (The schema flags travel to Python as CLI args, not env vars,
# so the slug must query presets directly or it would always fall back to "ddl".)
# The trailing __iter<N> records NUM_ITERATIONS so the requested pass count is
# visible from the directory name (count results_iter*.jsonl to confirm).
# ---------------------------------------------------------------------------
build_run_slug() {
  local baseline="${BASELINE:-no_tool}"
  local model_name="${PREDICTOR_MODEL_NAME:-}"
  local num_iterations="${NUM_ITERATIONS:-1}"

  # Resolve the variant's four schema flag values (NUL-delimited, in the order:
  # database_schema_type, read_only_gt_tables, read_only_gt_kb, is_kb_linearized).
  local _vc=()
  mapfile -d '' _vc < <(
    python3 "${BASE_WORK}/src/conversation2sql/presets.py" variant-config \
      --variant "${VARIANT:-}" 2>/dev/null
  )
  local schema_type="${_vc[0]:-ddl}"
  local gt_db="${_vc[1]:-false}"
  local gt_kb="${_vc[2]:-false}"
  local is_lin="${_vc[3]:-false}"

  # Take only the last component of the model path (e.g. "Qwen/Qwen3.5-9B" → "Qwen3.5-9B"),
  # then replace any character that is not alphanumeric / dot / underscore / hyphen with "-",
  # and strip any trailing hyphens that might result.
  local model_slug
  model_slug=$(basename "${model_name}" | sed 's/[^A-Za-z0-9._-]/-/g; s/-*$//')

  local slug="${baseline}__${model_slug}__${schema_type}"
  [[ "${is_lin}" == "true" ]] && slug="${slug}__lin"
  [[ "${gt_db}"  == "true" ]] && slug="${slug}__gt-db"
  [[ "${gt_kb}"  == "true" ]] && slug="${slug}__gt-kb"
  slug="${slug}__iter${num_iterations}"

  echo "${slug}"
}


# ---------------------------------------------------------------------------
# has_unresolved_errors <run_dir>
#
# Exit 0 (true) when the run still has *unresolved* errors, i.e. an entry in
# results_error.jsonl that has NOT since been superseded by a successful result.
# results_error.jsonl is append-only and is not cleared on --resume, so a stale
# error line for a pair that was later re-run successfully (and thus appears in
# results_iter*.jsonl) does NOT count. A whole-run error record (no instance_id)
# always counts as unresolved. Exit 1 (false) when nothing remains broken.
#
# This is the single source of truth for the __error suffix: both the fresh
# eval run (run_suite) and the recover path use it, so the suffix is added when
# errors remain and dropped once a recover resolves them.
# ---------------------------------------------------------------------------
has_unresolved_errors() {
  local run_dir="$1"
  [[ -s "${run_dir}/results_error.jsonl" ]] || return 1
  python3 - "${run_dir}" <<'PY'
import glob, json, os, re, sys

run_dir = sys.argv[1]

# Successful (instance_id, iteration) pairs already written to disk. The
# iteration comes from the filename (matches the pipeline's _load_completed_pairs).
completed = set()
for path in glob.glob(os.path.join(run_dir, "results_iter*.jsonl")):
    m = re.search(r"results_iter(\d+)\.jsonl$", os.path.basename(path))
    if not m:
        continue
    iteration = int(m.group(1))
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            iid = rec.get("instance_id")
            if iid is not None:
                completed.add((iid, iteration))

# An error is unresolved when its pair has no successful result. A record with
# no instance_id/iteration is a whole-run failure -> always unresolved.
with open(os.path.join(run_dir, "results_error.jsonl"), encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            sys.exit(0)  # garbled error line -> treat as unresolved
        iid, iteration = rec.get("instance_id"), rec.get("iteration")
        if iid is None or iteration is None or (iid, iteration) not in completed:
            sys.exit(0)
sys.exit(1)
PY
}


# ---------------------------------------------------------------------------
# apply_error_suffix <run_dir>
#
# Reconcile the run directory's __error suffix with its current state and print
# the (possibly renamed) directory. Strips any existing suffix to find the base
# name, then re-adds __error only when has_unresolved_errors says so. Renames in
# place when the name changes; a no-op otherwise.
# ---------------------------------------------------------------------------
apply_error_suffix() {
  local run_dir="$1"
  local base="${run_dir%__error}"   # canonical name without the suffix
  local final="${base}"
  has_unresolved_errors "${run_dir}" && final="${base}__error"
  if [[ "${final}" != "${run_dir}" ]]; then
    mv "${run_dir}" "${final}"
  fi
  echo "${final}"
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

  # If the pipeline left any unresolved errors, mark the run by renaming its
  # directory with an __error suffix so failed runs are obvious from a listing.
  local final_dir
  final_dir=$(apply_error_suffix "${run_dir}")
  if [[ "${final_dir}" != "${run_dir}" ]]; then
    run_dir="${final_dir}"
    log_section "Errors detected → renamed run dir to ${run_dir}" "${MY_SLURM_JOB_ID:-}"
  fi

  log_section "=== Done ${BASELINE:-no_tool} ===" "${MY_SLURM_JOB_ID:-}"

  # Best-effort: refresh the experiments index so this run's metrics + final
  # status land in experiments.csv. Never fail the run on an index error.
  ( cd "${BASE_WORK}" && uv run python -m explorer.index reconcile ) \
    || log_section "experiments index reconcile failed (non-fatal)" "${MY_SLURM_JOB_ID:-}"
}
