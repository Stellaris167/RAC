#!/usr/bin/env bash
# Shortcut: runs the pair_T0.8 experiment (severity 4 corruption pairs).
set -xeuo pipefail
export EXP=pair_T0.8
exec "$(dirname "$0")/run_qwen2_5vl7b_grpo.sh"
