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

# https://huggingface.co/google/gemma-4-26B-A4B-it

MODEL_NAME="google/gemma-4-26B-A4B-it"
MAX_MODEL_LEN=32000
ENABLE_THINKING=false


TEMPERATURE=1.0
TOP_P=0.95
TOP_K=64

if [ "$ENABLE_THINKING" = true ]; then
    DEFAULT_PARAMS='{"enable_thinking": true}'
else
    DEFAULT_PARAMS='{"enable_thinking": false}'
fi


OUTPUT_DIR="${BASE_WORK}/results"
DEBUG=true

start_vllm_server "$MODEL_NAME" "$MAX_MODEL_LEN" \
    --tensor-parallel-size 1 \
    --data-parallel-size 1 \
    --reasoning-parser gemma4 \
    --chat-template ./tool_chat_template_gemma4.jinja \
    --default-chat-template-kwargs "$DEFAULT_PARAMS" \
    --limit-mm-per-prompt '{"image": 0, "audio": 0}' # Text only

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
    --database_schema_type "ddl" \
    --make_data_ambiguous false \
    --read_only_gt_tables true \
    --read_only_gt_kb true \
    --output_folder "${OUTPUT_DIR}" \
    --debug $DEBUG
