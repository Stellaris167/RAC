#!/usr/bin/env bash
# Shortcut: runs the ranking experiment (clean training, rank bonus only).
set -xeuo pipefail
export EXP=ranking
exec "$(dirname "$0")/run_qwen2_5vl7b_grpo.sh"
