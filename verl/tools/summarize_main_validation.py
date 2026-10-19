"""Read-only audit of the original RAC Qwen3.5 comparison artifacts."""

import argparse
import json
from pathlib import Path

import numpy as np

from verl.trainer.ppo.metric_utils import _binary_ece


def read_rows(path):
    # An active writer may not have finished the final JSONL record yet.
    return [json.loads(line) for line in path.read_text().splitlines(keepends=True) if line.endswith("\n")]


def binary_auroc(labels, scores):
    labels = np.asarray(labels, dtype=float)
    scores = np.asarray(scores, dtype=float)
    positive = labels == 1
    negative = labels == 0
    if not positive.any() or not negative.any():
        return None

    order = np.argsort(scores, kind="stable")
    ranks = np.empty(len(scores), dtype=float)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and scores[order[end]] == scores[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2
        start = end
    n_positive = int(positive.sum())
    n_negative = int(negative.sum())
    return float((ranks[positive].sum() - n_positive * (n_positive + 1) / 2) / (n_positive * n_negative))


def summarize_rows(rows):
    acc = np.asarray([r["acc"] for r in rows], dtype=float)
    conf = np.asarray([r["confidence"] for r in rows], dtype=float)
    valid = np.isfinite(acc) & (acc >= 0) & (acc <= 1)
    calibrated = valid & np.isfinite(conf) & (conf >= 0) & (conf <= 1)
    auc_valid = calibrated & np.isin(acc, [0, 1])
    wrong = calibrated & (acc == 0)
    c, a = conf[calibrated], acc[calibrated]
    return {
        "n": len(rows), "accuracy_valid": int(valid.sum()), "calibration_valid": int(calibrated.sum()),
        "skipped_fraction": float((~valid).mean()) if len(rows) else None,
        "accuracy": float(acc[valid].mean()) if valid.any() else None,
        "ece": _binary_ece(c, a) if len(c) else None,
        "brier": float(((c-a)**2).mean()) if len(c) else None,
        "auroc": binary_auroc(acc[auc_valid], conf[auc_valid]),
        "mean_confidence": float(c.mean()) if len(c) else None,
        "confidence_one_fraction": float((c == 1).mean()) if len(c) else None,
        "wrong_count": int(wrong.sum()),
        "wrong_confidence_one_fraction": float((conf[wrong] == 1).mean()) if wrong.any() else None,
        "format_rate": float(np.mean([r["format_ok"] for r in rows])) if rows else None,
    }


def summarize_validation_directory(path, expected_rows):
    result = {}
    for dump_path in sorted(path.glob("*.jsonl"), key=lambda item: int(item.stem)):
        rows = read_rows(dump_path)
        if len(rows) != expected_rows:
            result[dump_path.stem] = {"pending_rows": len(rows), "expected_rows": expected_rows}
            continue
        result[dump_path.stem] = {
            "overall": summarize_rows(rows),
            "clean": summarize_rows([row for row in rows if not row["is_noisy"]]),
            "noisy": summarize_rows([row for row in rows if row["is_noisy"]]),
        }
    return result


def _latest_training_metrics(metric_path):
    metrics = read_rows(metric_path) if metric_path.exists() else []
    training = [row for row in metrics if "training/global_step" in row["data"]]
    latest = training[-1] if training else None
    keys = (
        "calibration/mean_accuracy",
        "calibration/mean_confidence",
        "format/rate",
        "accuracy/skipped_count",
        "actor/grad_norm",
        "actor/ppo_kl",
        "response_length/mean",
        "response_length/clip_ratio",
        "timing_s/step",
    )
    return {
        "latest_step": latest["step"] if latest else 0,
        "training": {key: latest["data"].get(key) for key in keys} if latest else {},
    }


def summarize_run(root, variant, expected_rows, run_name=None):
    name = run_name or f"qwen3.5-4b-sft20-rl-main_{variant}-densemask"
    metric_path = root / "logs/mlp_metrics" / f"{name}.jsonl"
    result = _latest_training_metrics(metric_path)
    result["validation"] = summarize_validation_directory(root / "logs/val_dumps" / name, expected_rows)
    return result


def summarize_run_directory(run_dir, expected_rows):
    result = _latest_training_metrics(run_dir / "metrics.jsonl")
    result["validation"] = summarize_validation_directory(run_dir / "validation", expected_rows)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/home/yby/RAC"))
    parser.add_argument("--expected-rows", type=int, default=768)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--run-name", action="append", help="Exact run name; repeat for comparisons")
    parser.add_argument("--run-dir", type=Path, action="append", help="Run directory containing metrics.jsonl and validation/")
    args = parser.parse_args()
    if args.run_dir:
        result = {path.name: summarize_run_directory(path, args.expected_rows) for path in args.run_dir}
    elif args.run_name:
        result = {name: summarize_run(args.root, name, args.expected_rows, run_name=name) for name in args.run_name}
    else:
        result = {v: summarize_run(args.root, v, args.expected_rows) for v in ("baseline", "pair_rank", "pair_rank_hinge")}
    serialized = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized)
    print(serialized)


if __name__ == "__main__":
    main()
