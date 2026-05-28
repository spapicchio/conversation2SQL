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
        echo "Killing VLLM server (PID $VLLM_PID, PGID ${VLLM_PGID:-$VLLM_PID})..."
        # Kill the entire process group created by setsid so worker subprocesses
        # (tensor-parallel workers, async engine) don't outlive the launcher.
        if [ -n "${VLLM_PGID:-}" ]; then
            kill -- -"$VLLM_PGID" 2>/dev/null || kill "$VLLM_PID"
        else
            kill "$VLLM_PID"
        fi
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


    local vllm_log="${DEST_DIR:+${DEST_DIR}/tmux_log/vllm.log}"
    log_section "Starting VLLM server: ${model_name} on port ${VLLM_SERVER_PORT}" "${MY_SLURM_JOB_ID:-}"
    [ -n "$vllm_log" ] && echo "[vllm_server] vLLM log → ${vllm_log}"

    # Launch in its own process group so the cleanup trap can kill all workers.
    # --enable-prefix-caching  if prompts share a system prompt — reduces KV recalculation
    if [ -n "$vllm_log" ]; then
        setsid uv run vllm serve "$model_name" \
            --port "$VLLM_SERVER_PORT" \
            --max-model-len "$max_model_len" \
            --uvicorn-log-level warning \
            --max-num-seqs 64 \
            --enable-prefix-caching \
            "$@" >> "$vllm_log" 2>&1 &
    else
        setsid uv run vllm serve "$model_name" \
            --port "$VLLM_SERVER_PORT" \
            --max-model-len "$max_model_len" \
            --uvicorn-log-level warning \
            --max-num-seqs 64 \
            --enable-prefix-caching \
            "$@" &
    fi

    VLLM_PID=$!
    VLLM_PGID=$(ps -o pgid= "$VLLM_PID" 2>/dev/null | tr -d ' ' || echo "$VLLM_PID")

    # Wait up to 120 s for the server to respond.
    # Wait for the server to become healthy, failing fast if the process dies or the
    # deadline (VLLM_START_TIMEOUT seconds, default 1800) is exceeded.
    VLLM_START_TIMEOUT=600
    local deadline=$(( $(date +%s) + ${VLLM_START_TIMEOUT:-1800} ))
    while ! curl -s "http://${host}:${VLLM_SERVER_PORT}/health" > /dev/null 2>&1; do
        if ! kill -0 "$VLLM_PID" 2>/dev/null; then
            echo "[ERROR] VLLM server process (PID $VLLM_PID) died before becoming healthy"
            exit 1
        fi
        if [ "$(date +%s)" -ge "$deadline" ]; then
            echo "[ERROR] VLLM server did not start within ${VLLM_START_TIMEOUT:-1800} s — killing"
            exit 1
        fi
        sleep 5
    done

    log_section "VLLM server is up on port ${VLLM_SERVER_PORT}" "${MY_SLURM_JOB_ID:-}"
}
