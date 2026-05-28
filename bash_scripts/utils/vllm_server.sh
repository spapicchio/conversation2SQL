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
#   VLLM_PID                    — consumed by the cleanup trap

# kill vLLM on exit or error
VLLM_PID=""
cleanup() {
    [ -n "$VLLM_PID" ] || return 0
    # Resolve the process group at kill time. Capturing it at launch races with
    # setsid() taking effect and can grab the wrong group, leaving vLLM alive.
    # setsid put vLLM in its own session, so killing the group reaps uv + every
    # vLLM worker. Falls back to the bare PID if the group can't be resolved.
    local pgid=$(ps -o pgid= "$VLLM_PID" 2>/dev/null | tr -d ' ')
    if [ -n "$pgid" ]; then
        echo "Killing VLLM process group (PGID $pgid)..."
        kill -9 -- "-${pgid}" 2>/dev/null || true
    else
        echo "Killing VLLM server (PID $VLLM_PID)..."
        kill -9 "$VLLM_PID" 2>/dev/null || true
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

    # Send vLLM's logs to a file rather than letting it inherit our stdout/stderr.
    # Under submit_and_log.sh those fds are the input of a `tee` pipeline that
    # drives the tmux session; a backgrounded vLLM holding them open keeps the
    # pipeline (and thus the tmux session) alive even after this script exits on
    # a startup timeout. Redirecting detaches it so exit tears everything down.
    local vllm_log="${VLLM_LOG:-${DEST_DIR:-${BASE_WORK:-/tmp}}/vllm_${VLLM_SERVER_PORT}.log}"
    log_section "VLLM server logs -> ${vllm_log}" "${MY_SLURM_JOB_ID:-}"

    # Launch in its own process group so the cleanup trap can kill all workers.
    # --enable-prefix-caching  if prompts share a system prompt — reduces KV recalculation
    setsid uv run vllm serve "$model_name" \
        --port "$VLLM_SERVER_PORT" \
        --max-model-len "$max_model_len" \
        --uvicorn-log-level warning \
        --max-num-seqs 64 \
        --enable-prefix-caching \
        "$@" > "$vllm_log" 2>&1 &

    VLLM_PID=$!

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



# What was wrong

#   When vLLM fails to start within VLLM_START_TIMEOUT, start_vllm_server calls exit 1. The
#    eval script exits — but the tmux session never ends, so run_all_sequential.sh waits on
#    it forever.

#   Two coupled root causes, both verified with dependency-free reproductions:

#   1. vLLM inherited the tmux pipe. submit_and_log.sh:64-75 runs the eval script as … | 
#   tee | tee | tee — that pipeline is the tmux session, and it only ends when every writer
#    to the pipe closes it. vLLM was launched (vllm_server.sh:58-64) with no output 
#   redirection, so it inherited the script's stdout/stderr and kept the pipe open after
#   the script exited → tmux session lingers forever. (Repro: inherited daemon → pipeline
#   hung at timeout 6 / exit 124; redirected daemon → exit 0 in 0s.)
#   2. Cleanup killed the wrong group. VLLM_PGID was captured by ps immediately after
#   setsid &, racing setsid() — so it often grabbed the wrong group and left vLLM (and its
#   GPU memory) alive.

#   The fix (bash_scripts/utils/vllm_server.sh)

#   - vLLM's stdout/stderr now go to a per-run log file
#   (${DEST_DIR:-$BASE_WORK}/vllm_<port>.log, overridable via VLLM_LOG) instead of the
#   inherited pipe. So exit always tears the pipeline (and tmux session) down, regardless
#   of vLLM's state.
#   - cleanup() resolves the process group at kill time via ps, eliminating the launch-time
#    race. Verified under the eval script's non-interactive (job-control-off) semantics:
#   pgid == pid, group kill → process DEAD.

#   Net effect: on a startup timeout, vLLM is reaped, the eval script exits, the tmux
#   session ends, and run_all_sequential.sh moves on.
