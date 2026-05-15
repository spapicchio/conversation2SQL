#!/bin/bash

if [ -z "$BASE_WORK" ]; then
  echo "BASE_WORK not set. must be set before sourcing utils.sh"
  exit 1
fi
LOGFILE="${BASE_WORK}/log_sbatch.log"
echo "loading utils.sh with LOGFILE: ${LOGFILE}"


log_section() {
  local msg="$1"
  local job_name="$2"

  local timestamp
  timestamp=$(date "+%Y-%m-%d %H:%M:%S")
  {
    echo "--------------------------------------------------"
    echo "[$timestamp][$job_name] $msg"
    echo "--------------------------------------------------"
  } >>"$LOGFILE"

  echo "[$timestamp][$job_name] $msg"
}

function setup_idris {
  # Set some env variable for running on compute nodes without internet access
  export GIT_PYTHON_REFRESH=quiet
  export HF_DATASETS_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  export HF_HUB_OFFLINE=1
  export WANDB_MODE=offline

  NODELIST=($(scontrol show hostnames $SLURM_JOB_NODELIST))
  export MASTER_ADDR=${NODELIST[0]} # First node for main process
  export MASTER_PORT=6000

  # setup one node for vLLM
  WORLD_SIZE=$(($SLURM_NNODES * SLURM_GPUS_PER_NODE))
  export TRAIN_NODES=("${NODELIST[@]:0:$((SLURM_NNODES - 1))}")  # Last node used for vLLM
  export TRAIN_NODES_STR=$(NODES="${TRAIN_NODES[*]}" python -c 'import os; print(",".join(os.getenv("NODES").split()))')

  export VLLM_NODE=${NODELIST[-1]} # Last node
  export WORLD_SIZE=$((WORLD_SIZE - SLURM_GPUS_PER_NODE)) # exclude vLLM node
  export NUM_NODES=$((SLURM_NNODES - 1)) # exclude vLLM node
}

function cp_files {
  local dest="$1"
  local src_file="$2"
  local job_id="$3"
  log_section "From '${dest}' to '${src_file}'" "${job_id}"
  mkdir -p "${dest}"
  cp -r "${src_file}" "${dest}"
}
