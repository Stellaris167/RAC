#!/usr/bin/env bash
# Shortcut: runs the pair_T1.0 experiment (severity 5 corruption pairs).
set -xeuo pipefail
export EXP=pair_T1.0
exec "$(dirname "$0")/run_qwen2_5vl7b_grpo.sh"
