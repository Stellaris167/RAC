#!/usr/bin/env bash
# Shortcut: runs the pair_T0.4 experiment (severity 2 corruption pairs).
set -xeuo pipefail
export EXP=pair_T0.4
exec "$(dirname "$0")/run_qwen2_5vl7b_grpo.sh"
