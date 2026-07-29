#!/usr/bin/env bash
# Run AgentDyn benchmark with AgentGuard as the defense.
#
# Prerequisites:
#   - conda activate agentguard
#   - Set AGENTGUARD_MODEL (required)
#   - Optionally set AGENTGUARD_BASE_URL, AGENTGUARD_API_KEY, etc.
#
# Usage:
#   # Utility test (no attack)
#   bash util_scripts/run_agentguard_benchmark.sh --model gpt-4o-2024-08-06 -s shopping
#
#   # Security test (with attack)
#   bash util_scripts/run_agentguard_benchmark.sh --model gpt-4o-2024-08-06 -s shopping \
#       --attack important_instructions
#
#   # Single task (debug)
#   bash util_scripts/run_agentguard_benchmark.sh --model gpt-4o-2024-08-06 -s shopping \
#       -ut user_task_0 --attack important_instructions -it injection_task_0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
AGENTDYN_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
AGENTGUARD_SRC="$(cd "$SCRIPT_DIR/../../../src" && pwd)"

# Add AgentDyn local src + AgentGuard src to PYTHONPATH
# (AgentDyn local src takes priority over pip-installed agentdojo)
AGENTDYN_SRC="${AGENTDYN_ROOT}/src"
export PYTHONPATH="${AGENTDYN_SRC}:${AGENTGUARD_SRC}:${PYTHONPATH:-}"

# Default env vars (can be overridden before calling this script)
export AGENTGUARD_PROMPT_NAME="${AGENTGUARD_PROMPT_NAME:-general}"
export AGENTGUARD_RESPONSE_PARSER="${AGENTGUARD_RESPONSE_PARSER:-strict}"
export AGENTGUARD_BLOCKING_MODE="${AGENTGUARD_BLOCKING_MODE:-continue}"
export AGENTGUARD_CONFIDENCE_THRESHOLD="${AGENTGUARD_CONFIDENCE_THRESHOLD:-0.5}"

# Validate required variable
if [ -z "${AGENTGUARD_MODEL:-}" ]; then
    echo "ERROR: AGENTGUARD_MODEL is required. Set it before running this script."
    echo "Example: export AGENTGUARD_MODEL=gpt-4o-mini"
    exit 1
fi

echo "=== AgentGuard Defense Benchmark ==="
echo "Guard model:    ${AGENTGUARD_MODEL}"
echo "Prompt:         ${AGENTGUARD_PROMPT_NAME}"
echo "Blocking mode:  ${AGENTGUARD_BLOCKING_MODE}"
echo "Threshold:      ${AGENTGUARD_CONFIDENCE_THRESHOLD}"
echo "===================================="

cd "$AGENTDYN_ROOT"
python -m agentdojo.scripts.benchmark --defense agentguard "$@"
