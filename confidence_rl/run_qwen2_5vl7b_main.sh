#!/usr/bin/env bash
# Shortcut: runs the pair_main experiment
# (noisy mix = 50% T0.2 + 40% T0.4 + 10% T0.6).
# Equivalent to: EXP=pair_main bash run_qwen2_5vl7b_grpo.sh
set -xeuo pipefail

export EXP=pair_main
exec "$(dirname "$0")/run_qwen2_5vl7b_grpo.sh"
