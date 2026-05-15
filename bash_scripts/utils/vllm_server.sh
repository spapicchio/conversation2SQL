#!/bin/bash
# Helper for launching a local vLLM inference server during evaluation.
# Source this file AFTER evaluate.sh (which sets BASE_WORK and log_section).
#
# Sourcing registers the cleanup trap from utils_clenup_vllm_if_crash.sh.
#
# Usage:
#   source "${BASE_WORK}/bash_scripts/utils/vllm_server.sh"
#
#   start_vllm_server MODEL_NAME MAX_MODEL_LEN ENABLE_THINKING [extra vllm serve args...]
#
# After the call the following variables are set in the caller's scope:
#   VLLM_SERVER_PORT            — port the server is listening on
#   PREDICTOR_VLLM_API_BASE     — http://127.0.0.1:<port>/v1
#   USER_SIMULATOR_VLLM_API_BASE — same value (both roles share one server)
#   TEMPERATURE, TOP_P, TOP_K, PRESENCE_PENALTY, REPETITION_PENALTY
#   VLLM_PID, VLLM_PGID         — consumed by the cleanup trap

# kill vLLM on exit or error
VLLM_PID=""
cleanup() {
    if [ -n "$VLLM_PID" ] && kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "Killing VLLM server (PID $VLLM_PID)..."
        kill "$VLLM_PID"
        wait "$VLLM_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT


start_vllm_server() {
    local model_name="$1"
    local max_model_len="$2"
    shift 2
    # All remaining arguments are forwarded verbatim to `vllm serve`.
    
    # ---- pick a free ephemeral port ----
    VLLM_SERVER_PORT=$(uv run python -c "
import socket
s = socket.socket()
s.bind(('', 0))
print(s.getsockname()[1])
s.close()
")
    local host=127.0.0.1
    
    PREDICTOR_VLLM_API_BASE="http://${host}:${VLLM_SERVER_PORT}/v1"
    USER_SIMULATOR_VLLM_API_BASE="http://${host}:${VLLM_SERVER_PORT}/v1"


    log_section "Starting VLLM server: ${model_name} on port ${VLLM_SERVER_PORT}" "${MY_SLURM_JOB_ID:-}"

    # Launch in its own process group so the cleanup trap can kill all workers.
    setsid uv run vllm serve "$model_name" \
        --port "$VLLM_SERVER_PORT" \
        --max-model-len "$max_model_len" \
        --uvicorn-log-level warning \
        "$@" &

    VLLM_PID=$!
    VLLM_PGID=$(ps -o pgid= "$VLLM_PID" 2>/dev/null | tr -d ' ' || echo "$VLLM_PID")

    # Wait up to 120 s for the server to respond.
    timeout 120 bash -c "
        until curl -s http://${host}:${VLLM_SERVER_PORT}/health > /dev/null 2>&1; do
            sleep 5
        done
    " || { echo "[ERROR] VLLM server did not start within 120 s"; exit 1; }

    log_section "VLLM server is up on port ${VLLM_SERVER_PORT}" "${MY_SLURM_JOB_ID:-}"
}
