#!/usr/bin/env bash
# Shortcut: runs the baseline experiment (clean training, no bonuses).
set -xeuo pipefail
export EXP=baseline
exec "$(dirname "$0")/run_qwen2_5vl7b_grpo.sh"
