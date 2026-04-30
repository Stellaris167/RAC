#!/usr/bin/env bash
# Shortcut: runs the pair_T0.2 experiment (severity 1 corruption pairs).
# Equivalent to: EXP=pair_T0.2 bash run_qwen2_5vl7b_grpo.sh
set -xeuo pipefail

export EXP=pair_T0.2
exec "$(dirname "$0")/run_qwen2_5vl7b_grpo.sh"
