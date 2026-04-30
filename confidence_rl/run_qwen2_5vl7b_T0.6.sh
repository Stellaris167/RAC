#!/usr/bin/env bash
# Shortcut: runs the pair_T0.6 experiment (severity 3 corruption pairs).
set -xeuo pipefail
export EXP=pair_T0.6
exec "$(dirname "$0")/run_qwen2_5vl7b_grpo.sh"
