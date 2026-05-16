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
#SBATCH --signal=B:USR1@30
#SBATCH --open-mode=append

set -Eeuo pipefail

export BASE_WORK=/workspaces/conversation2SQL
export MY_SLURM_JOB_ID="${MY_SLURM_JOB_ID:-local}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

source "${BASE_WORK}/bash_scripts/evaluate.sh"
source "${BASE_WORK}/bash_scripts/utils/vllm_server.sh"

log_section "Starting evaluation script" "${MY_SLURM_JOB_ID}"

# https://huggingface.co/Qwen/Qwen3.5-9B
# Thinking mode (coding):  temperature=0.6, top_p=0.95, top_k=20, presence_penalty=0.0
# Non-thinking (general):  temperature=1.0, top_p=0.95, top_k=20, presence_penalty=1.5
MODEL_NAME="Qwen/Qwen3.5-9B"
MAX_MODEL_LEN=32000
ENABLE_THINKING=true


if [ "$ENABLE_THINKING" = true ]; then
    TEMPERATURE=0.6
    TOP_P=0.95
    TOP_K=20
    PRESENCE_PENALTY=0.0
    REPETITION_PENALTY=1.0
    DEFAULT_PARAMS='{"enable_thinking": true}'
else
    TEMPERATURE=1.0
    TOP_P=0.95
    TOP_K=20
    PRESENCE_PENALTY=1.5
    REPETITION_PENALTY=1.0
    DEFAULT_PARAMS='{"enable_thinking": false}'
fi

OUTPUT_DIR="${DEST_DIR}"
DEBUG=false

start_vllm_server "$MODEL_NAME" "$MAX_MODEL_LEN" \
    --tensor-parallel-size 1 \
    --data-parallel-size 1 \
    --reasoning-parser qwen3 \
    --default-chat-template-kwargs "$DEFAULT_PARAMS" \
    --language-model-only

run_suite "no_tool" \
    "$PREDICTOR_VLLM_API_BASE" \
    "$USER_SIMULATOR_VLLM_API_BASE" \
    --predictor_model_provider "hosted_vllm" \
    --predictor_model_name "${MODEL_NAME}" \
    --predictor_temperature "${TEMPERATURE}" \
    --predictor_top_p "${TOP_P}" \
    --predictor_top_k "${TOP_K}" \
    --predictor_presence_penalty "${PRESENCE_PENALTY}" \
    --predictor_repetition_penalty "${REPETITION_PENALTY}" \
    --predictor_enable_thinking "${ENABLE_THINKING}" \
    --database_schema_type "toon" \
    --make_data_ambiguous false \
    --read_only_gt_tables false \
    --read_only_gt_kb false \
    --is_kb_linearized false \
    --output_folder "${OUTPUT_DIR}" \
    --debug $DEBUG
