#!/bin/bash
#SBATCH -A vno@h100
#SBATCH -C h100
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --output=./logs/rl/%j.out
#SBATCH --nodes=1
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --time=05:00:00
#SBATCH --cpus-per-task=100
#SBATCH --signal=B:USR1@30   # send USR1 30 s before wall-time so the requeue handler can checkpoint
#SBATCH --open-mode=append

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
# run_suite <baseline> <vllm_server_host> <vllm_server_port>
# ---------------------------------------------------------------------------
run_suite() {

  local baseline="$1"
  local predictor_vllm_api_base="$2"
  local user_simulator_vllm_api_base="$3"
  shift 3

  log_section "Running suite: ${baseline}" "${MY_SLURM_JOB_ID:-}"
  local launcher=(
    uv run conv2sql run
    --config "${BASE_WORK}/configs/eval_pipeline_config.yaml"
    --baseline "${baseline}"
    --predictor_vllm_api_base "${predictor_vllm_api_base}"
    --user_simulator_vllm_api_base "${user_simulator_vllm_api_base}"
    
  )

  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  TOKENIZERS_PARALLELISM=true \
  VLLM_WORKER_MULTIPROC_METHOD=spawn \
  "${launcher[@]}" \
  "$@"
  
  log_section "=== Done ${baseline} ===" "${MY_SLURM_JOB_ID:-}"
}


# ---------------------------------------------------------------------------
# Post-run: copy results to WORK (Jean Zay persistent storage), SLURM only
# ---------------------------------------------------------------------------
if [[ -n "${WORK:-}" ]]; then
    log_section "Moving file into WORK: ${WORK}" "${MY_SLURM_JOB_ID:-}"
    OUTPUT_DIR="${BASE_WORK}/results/"
    DEST="${WORK}/evaluation_results"
    cp_files "${DEST}" "${OUTPUT_DIR}" "${MY_SLURM_JOB_ID:-}"
else
    log_section "WORK is not set; skipping move" "${MY_SLURM_JOB_ID:-}"
fi
