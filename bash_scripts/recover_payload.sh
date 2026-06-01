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
#SBATCH --open-mode=append

# ---------------------------------------------------------------------------
# Resume/recover payload: re-run only the missing (instance_id, iteration)
# pairs of a previous eval run, appending into that same run directory.
#
# Inputs (env vars):
#   RESUME_DIR  – the run directory to resume (must contain config.yaml)  (required)
#   TP / DP     – vLLM tensor/data-parallel size                          (default: 1)
#
# It replays the run's own config.yaml snapshot via --config and overrides only
# output_folder, --resume, and (for hosted_vllm) a fresh vLLM api-base. The model
# server profile is reverse-mapped from the snapshot by presets.py recover-config.
#
# Launch via `just recover <run_dir>`. Set DRY_RUN=1 to print and exit (the
# dry-run path stays hermetic: it does NOT activate the venv or resolve the
# server config, so it is safe to run without uv/GPU).
# ---------------------------------------------------------------------------

set -Eeuo pipefail

export BASE_WORK="${BASE_WORK:-/workspaces/conversation2SQL}"
export MY_SLURM_JOB_ID="${MY_SLURM_JOB_ID:-local}"
RESUME_DIR="${RESUME_DIR:?set RESUME_DIR to the run directory to resume (must contain config.yaml)}"
TP="${TP:-1}"
DP="${DP:-1}"

if [ ! -f "${RESUME_DIR}/config.yaml" ]; then
  echo "[recover_payload] No config.yaml in RESUME_DIR='${RESUME_DIR}'" >&2
  exit 1
fi

RUN_ARGS=(
  --config "${RESUME_DIR}/config.yaml"
  --output_folder "${RESUME_DIR}"
  --resume true
)

# Dry run: print the base resume command and exit BEFORE any heavy setup
# (venv activation, server-config resolution, vLLM launch).
if [ "${DRY_RUN:-0}" = "1" ]; then
  echo "[DRY-RUN] recover resume_dir=${RESUME_DIR}"
  printf '[DRY-RUN] conv2sql run'; printf ' %s' "${RUN_ARGS[@]}"; printf '\n'
  exit 0
fi

# Real run: bring up the venv + shared exports (uv sync, activate .venv, source
# .env). This also means `recover-config` below runs under the venv (PyYAML).
source "${BASE_WORK}/bash_scripts/utils/utils_evaluate.sh"

# Snapshot -> provider + model + max-model-len + thinking + vllm serve args.
mapfile -d '' _RC < <(
  python "${BASE_WORK}/src/conversation2sql/presets.py" recover-config \
    --run-dir "${RESUME_DIR}" \
    --tp "${TP}" --dp "${DP}" \
    --base-work "${BASE_WORK}"
)
if (( ${#_RC[@]} < 5 )) || [ -z "${_RC[0]}" ]; then
  echo "[recover_payload] Failed to resolve recover config from ${RESUME_DIR}/config.yaml" >&2
  exit 1
fi
PROVIDER="${_RC[0]}"
MODEL_NAME="${_RC[1]}"
MAX_MODEL_LEN="${_RC[2]}"
ENABLE_THINKING="${_RC[3]}"
SERVER_ARGS=("${_RC[@]:4}")
export PREDICTOR_MODEL_NAME="${MODEL_NAME}"

if [ "${PROVIDER}" = "hosted_vllm" ]; then
  source "${BASE_WORK}/bash_scripts/utils/vllm_server.sh"
  start_vllm_server "$MODEL_NAME" "$MAX_MODEL_LEN" "${SERVER_ARGS[@]}"
  RUN_ARGS+=(--predictor_vllm_api_base "${PREDICTOR_VLLM_API_BASE}")
  if [ -n "${USER_SIMULATOR_VLLM_API_BASE:-}" ]; then
    RUN_ARGS+=(--user_simulator_vllm_api_base "${USER_SIMULATOR_VLLM_API_BASE}")
  fi
else
  unset PREDICTOR_VLLM_API_BASE      || true
  unset USER_SIMULATOR_VLLM_API_BASE || true
fi

uv run conv2sql run "${RUN_ARGS[@]}"
