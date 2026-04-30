#!/usr/bin/env bash
# Build all corrupted datasets for confidence RL experiments.
#
# Creates:
#   Training: 5 pair parquets (train_pair_T0.2..T1.0.parquet)
#   Test: 30 corrupted test sets (6 benchmarks × 5 severity levels)
#
# Usage:
#   bash verl/tools/run_build_all_datasets.sh                    # build all
#   bash verl/tools/run_build_all_datasets.sh --mode train_pairs # train only
#   bash verl/tools/run_build_all_datasets.sh --mode test_corrupt # test only
#   bash verl/tools/run_build_all_datasets.sh --mode verify      # verify only
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON_BIN=${PYTHON_BIN:-/share/home/cuipeng/miniconda3/envs/mllm_v2/bin/python}
DATA_ROOT=${DATA_ROOT:-/share/home/cuipeng/cuipeng_a100/yangboyao/datasets/SCS_data}
MODE=${1:---mode}
MODE_VAL=${2:-all}

echo "============================================================"
echo "Build corrupted datasets"
echo "  Python:    $PYTHON_BIN"
echo "  Data root: $DATA_ROOT"
echo "  Mode:      $MODE_VAL"
echo "============================================================"

export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"

"$PYTHON_BIN" "$SCRIPT_DIR/build_corrupted_datasets.py" \
    --data_root "$DATA_ROOT" \
    --mode "$MODE_VAL" \
    --seed 42

echo ""
echo "[done] All corrupted datasets built successfully."
