"""Cross-fit post-hoc maps for frozen, verbalized confidence predictions."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit
from sklearn.model_selection import StratifiedKFold

from verl.tools.summarize_main_validation import read_rows
from verl.trainer.ppo.metric_utils import _binary_ece


EPSILON = 1e-4


def clipped_logit(confidence):
    confidence = np.clip(np.asarray(confidence, dtype=float), EPSILON, 1 - EPSILON)
    return np.log(confidence) - np.log1p(-confidence)


def binary_nll(logits, labels):
    return float(np.mean(np.logaddexp(0, logits) - labels * logits))


def fit_temperature(confidence, labels):
    logits = clipped_logit(confidence)
    result = minimize_scalar(
        lambda log_temperature: binary_nll(logits / np.exp(log_temperature), labels),
        bounds=(-6, 6),
        method="bounded",
    )
    if not result.success:
        raise RuntimeError(result.message)
    return float(np.exp(result.x))


def fit_platt(confidence, labels):
    logits = clipped_logit(confidence)
    result = minimize(
        lambda params: binary_nll(params[0] * logits + params[1], labels),
        x0=np.array([1.0, 0.0]),
        method="L-BFGS-B",
        bounds=((1e-6, 20.0), (-20.0, 20.0)),
    )
    if not result.success:
        raise RuntimeError(result.message)
    return [float(value) for value in result.x]


def load_metadata(path):
    columns = ("dataset_name", "sample_id", "question", "answer", "is_noisy")
    rows = pq.read_table(path, columns=list(columns)).to_pylist()
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        key = tuple(row[column] for column in columns[:-1])
        groups[key].append(index)
    for key, indices in groups.items():
        if len(indices) != 2 or {rows[index]["is_noisy"] for index in indices} != {0, 1}:
            raise ValueError(f"Expected one clean/noisy pair for {key}")
    return rows, groups


def paired_folds(rows, groups, n_splits=5, seed=42):
    keys = list(groups)
    strata = [key[0] for key in keys]
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_ids = np.empty(len(rows), dtype=int)
    for fold, (_, heldout) in enumerate(splitter.split(keys, strata)):
        for group_index in heldout:
            fold_ids[groups[keys[group_index]]] = fold
    return fold_ids


def load_predictions(path, metadata):
    rows = read_rows(path)
    if len(rows) != len(metadata):
        raise ValueError(f"{path}: expected {len(metadata)} rows, got {len(rows)}")
    for index, (prediction, sample) in enumerate(zip(rows, metadata, strict=True)):
        if (
            sample["question"].strip() not in prediction["input"]
            or str(sample["answer"]) != str(prediction["gts"])
            or int(sample["is_noisy"]) != int(prediction["is_noisy"])
        ):
            raise ValueError(f"{path}: parquet/output alignment failed at row {index}")
        if not prediction["accuracy_valid"] or not prediction["confidence_valid"]:
            raise ValueError(f"{path}: invalid accuracy/confidence at row {index}")
    labels = np.asarray([row["acc"] for row in rows], dtype=float)
    confidence = np.asarray([row["confidence"] for row in rows], dtype=float)
    if not np.isin(labels, [0, 1]).all() or not ((confidence >= 0) & (confidence <= 1)).all():
        raise ValueError(f"{path}: labels or confidence outside their expected range")
    return labels, confidence


def crossfit(confidence, labels, fold_ids):
    logits = clipped_logit(confidence)
    calibrated = {
        "raw": confidence.copy(),
        "prevalence": np.empty_like(confidence),
        "temperature": np.empty_like(confidence),
        "platt": np.empty_like(confidence),
    }
    parameters = []
    for fold in np.unique(fold_ids):
        fit = fold_ids != fold
        heldout = ~fit
        prevalence = float(labels[fit].mean())
        temperature = fit_temperature(confidence[fit], labels[fit])
        slope, intercept = fit_platt(confidence[fit], labels[fit])
        calibrated["prevalence"][heldout] = prevalence
        calibrated["temperature"][heldout] = expit(logits[heldout] / temperature)
        calibrated["platt"][heldout] = expit(slope * logits[heldout] + intercept)
        parameters.append({
            "fold": int(fold),
            "fit_rows": int(fit.sum()),
            "test_rows": int(heldout.sum()),
            "prevalence": prevalence,
            "temperature": temperature,
            "platt_slope": slope,
            "platt_intercept": intercept,
        })
    return calibrated, parameters


def summarize(confidence, labels):
    clipped = np.clip(confidence, EPSILON, 1 - EPSILON)
    return {
        "n": int(len(labels)),
        "accuracy": float(labels.mean()),
        "ece": _binary_ece(confidence, labels),
        "brier": float(np.mean((confidence - labels) ** 2)),
        "nll": float(-np.mean(labels * np.log(clipped) + (1 - labels) * np.log1p(-clipped))),
        "mean_confidence": float(confidence.mean()),
    }


def run(eval_path, run_paths, n_splits=5, seed=42):
    metadata, groups = load_metadata(eval_path)
    fold_ids = paired_folds(metadata, groups, n_splits=n_splits, seed=seed)
    noisy = np.asarray([row["is_noisy"] for row in metadata], dtype=bool)
    result = {
        "method": "paired-stratified-crossfit",
        "folds": n_splits,
        "seed": seed,
        "groups": len(groups),
        "rows": len(metadata),
        "confidence_clip_for_fitting_and_nll": EPSILON,
        "runs": {},
    }
    oof = []
    for name, path in run_paths.items():
        labels, confidence = load_predictions(path, metadata)
        calibrated, parameters = crossfit(confidence, labels, fold_ids)
        result["runs"][name] = {
            "source": str(path),
            "fold_parameters": parameters,
            "metrics": {
                method: {
                    split: summarize(values[mask], labels[mask])
                    for split, mask in (("overall", np.ones(len(labels), dtype=bool)), ("clean", ~noisy), ("corrupted", noisy))
                }
                for method, values in calibrated.items()
            },
        }
        for index, sample in enumerate(metadata):
            oof.append({
                "run": name,
                "dataset_name": sample["dataset_name"],
                "sample_id": sample["sample_id"],
                "question": sample["question"],
                "is_noisy": bool(sample["is_noisy"]),
                "fold": int(fold_ids[index]),
                "correct": int(labels[index]),
                "raw_confidence": float(confidence[index]),
                "prevalence_confidence": float(calibrated["prevalence"][index]),
                "temperature_confidence": float(calibrated["temperature"][index]),
                "platt_confidence": float(calibrated["platt"][index]),
            })
    return result, oof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-parquet", type=Path, required=True)
    parser.add_argument("--run", action="append", required=True, help="NAME=validation JSONL")
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_paths = dict(item.split("=", 1) for item in args.run)
    result, oof = run(args.eval_parquet, {name: Path(path) for name, path in run_paths.items()}, args.folds, args.seed)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_prefix.with_name(args.output_prefix.name + "_summary.json")
    oof_path = args.output_prefix.with_name(args.output_prefix.name + "_oof.jsonl")
    summary_path.write_text(json.dumps(result, indent=2) + "\n")
    oof_path.write_text("\n".join(json.dumps(row) for row in oof) + "\n")
    print(f"Wrote {summary_path} and {oof_path}")


if __name__ == "__main__":
    main()
