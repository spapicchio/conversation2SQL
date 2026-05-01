#!/usr/bin/env bash
# Launch a Python experiment inside a detached tmux session.
#
# Usage:
#   ./launch_experiments.sh <python_script> [extra args forwarded to python]
#
# The script:
#   * Creates results/YYYY-MM-DD/HH-MM/<script_name>/
#   * Copies the --config file (or configs/eval_pipeline_config.yaml by default) into that folder
#   * Runs `uv run python <python_script> [args]` inside a detached tmux session, with stdout+stderr tee'd to run.log in the result folder

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <python_script> [args...]" >&2
    exit 1
fi

PYTHON_SCRIPT="$1"
shift
EXTRA_ARGS=("$@")

if [[ ! -f "$PYTHON_SCRIPT" ]]; then
    echo "Error: python script '$PYTHON_SCRIPT' does not exist." >&2
    exit 1
fi

# Locate repo root: this script lives at <repo>/src/conversation2sql/.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

CONFIG_PATH="$REPO_ROOT/configs/eval_pipeline_config.yaml"

if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "Error: config '$CONFIG_PATH' does not exist." >&2
    exit 1
fi

DATE_DIR="$(date +%Y-%m-%d)"
TIME_DIR="$(date +%H-%M-%S)"
SCRIPT_NAME="$(basename "$PYTHON_SCRIPT" .py)"
RESULT_DIR="$REPO_ROOT/results/$DATE_DIR/$TIME_DIR/$SCRIPT_NAME"
mkdir -p "$RESULT_DIR"

LOG_FILE="$RESULT_DIR/run.log"
SESSION_NAME="exp_${SCRIPT_NAME}_$(date +%Y%m%d_%H%M%S)_$$"

# Build the command. `printf %q` keeps spaces / quotes safe when re-parsed
# by tmux's shell.
CMD="cd $(printf '%q' "$REPO_ROOT") && uv run python $(printf '%q' "$PYTHON_SCRIPT") --config $(printf '%q' "$CONFIG_PATH") --output_folder $(printf '%q' "${RESULT_DIR}")"
for arg in "${EXTRA_ARGS[@]}"; do
    CMD+=" $(printf '%q' "$arg")"
done

CMD+=" "

CMD+=" 2>&1 | tee $(printf '%q' "$LOG_FILE")"

tmux new-session -d -s "$SESSION_NAME" "$CMD"

echo "Launched experiment in tmux."
echo "  command : $CMD"
echo "  session : $SESSION_NAME"
echo "  results : $RESULT_DIR"
echo "  log     : $LOG_FILE"
echo "Attach with: tmux attach -t $SESSION_NAME"
