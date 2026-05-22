#!/bin/bash
# Run all evaluation scripts sequentially via submit_and_log.sh.
# Each job gets its own tmux session + timestamped results folder (handled by submit_and_log.sh).
# This script waits for each tmux session to exit before launching the next.
#
# Usage:
#   bash bash_scripts/run_all_sequential.sh [CUDA_DEVICES]
#
#   CUDA_DEVICES  Optional. GPU IDs to use (default: 1). Passed to every eval script.
#                 Examples: 0   or   0,1   or   2,3
#                 Alternatively: CUDA_VISIBLE_DEVICES=0,1 bash run_all_sequential.sh
#

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Accept CUDA devices as first positional arg; fall back to env var, then default to 1.
export CUDA_VISIBLE_DEVICES="${1:-${CUDA_VISIBLE_DEVICES:-1}}"
echo "[SEQUENTIAL] Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

EVAL_SCRIPTS=(
    # "${SCRIPT_DIR}/evaluation_scripts/qwen35/qwen_3.5_local_all_db_all_kb.sh"
    "${SCRIPT_DIR}/evaluation_scripts/qwen35/qwen_3.5_local_all_db_all_kb_linearized.sh"
    # "${SCRIPT_DIR}/evaluation_scripts/qwen35/qwen_3.5_local_all_db_toon_all_kb.sh"
    # "${SCRIPT_DIR}/evaluation_scripts/qwen35/qwen_3.5_local_all_db_toon_all_kb_linearized.sh"
    # "${SCRIPT_DIR}/evaluation_scripts/qwen35/qwen_3.5_local_gt_db_all_kb_linearized.sh"
    # "${SCRIPT_DIR}/evaluation_scripts/qwen35/qwen_3.5_local_gt_db_gt_kb_linearized.sh"
)

POLL_INTERVAL=300   # seconds between tmux session presence checks

FAILED=()

for script in "${EVAL_SCRIPTS[@]}"; do
    echo ""
    echo "============================================================"
    echo "[SEQUENTIAL] Launching: $(basename "$script")"
    echo "============================================================"

    # Run submit_and_log.sh, capture output to extract the tmux session name,
    # and echo it so the user can follow progress.
    if LAUNCH_OUTPUT=$(bash "${SCRIPT_DIR}/submit_and_log.sh" "$script" 2>&1); then
        echo "$LAUNCH_OUTPUT"
    else
        echo "$LAUNCH_OUTPUT"
        echo "[SEQUENTIAL] ERROR: submit_and_log.sh failed for $(basename "$script") — skipping."
        FAILED+=("$(basename "$script")")
        continue
    fi

    # submit_and_log.sh prints a line like:
    #   [SUBMIT_AND_LOG] Generated fake job ID for date 1234567890: abcdef12
    # The tmux session name equals that 8-char hex ID.
    SESSION_ID=$(echo "$LAUNCH_OUTPUT" | grep 'Generated fake job ID' | awk '{print $NF}')

    if [[ -z "${SESSION_ID}" ]]; then
        echo "[SEQUENTIAL] ERROR: could not extract session ID — cannot wait; continuing immediately."
        FAILED+=("$(basename "$script")")
        continue
    fi

    echo "[SEQUENTIAL] Waiting for tmux session '${SESSION_ID}' to finish (polling every ${POLL_INTERVAL}s)..."
    echo "[SEQUENTIAL] Attach with:  tmux attach -t ${SESSION_ID}"

    while tmux has-session -t "${SESSION_ID}" 2>/dev/null; do
        sleep "${POLL_INTERVAL}"
    done

    echo "[SEQUENTIAL] Session '${SESSION_ID}' exited — moving to next experiment."
done

echo ""
echo "============================================================"
echo "[SEQUENTIAL] All experiments finished."
if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "[SEQUENTIAL] Failed scripts:"
    for f in "${FAILED[@]}"; do
        echo "  - $f"
    done
else
    echo "[SEQUENTIAL] No failures detected."
fi
echo "============================================================"
