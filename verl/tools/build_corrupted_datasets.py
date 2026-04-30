#!/usr/bin/env python3
"""Build ImageNet-C corrupted training/test datasets for confidence RL.

Creates corrupted versions of images at 5 severity levels (T0.2–T1.0).
For each sample, ONE corruption method is randomly chosen from the 8 available.

Training mode (--mode train_pairs):
  Reads the clean train.parquet, produces 5 pair parquets:
    train_pair_T0.2.parquet, ..., train_pair_T1.0.parquet
  Each pair parquet has 2× rows: clean + corrupted with same pair_id.

Main training set mode (--mode train_main):
    Reads pair datasets and builds train_pair_main.parquet with:
        - clean rows unchanged
        - noisy rows mixed as 50% T0.2 + 40% T0.4 + 10% T0.6

Test mode (--mode test_corrupt):
  For each benchmark × severity, produces a corrupted test.parquet:
    {benchmark}_test_processed_T{label}/test.parquet

All mode (--mode all):
  Runs both training and test dataset construction.

Usage:
  python -m verl.tools.build_corrupted_datasets --data_root .../SCS_data --mode all
  python -m verl.tools.build_corrupted_datasets --data_root .../SCS_data --mode train_pairs
    python -m verl.tools.build_corrupted_datasets --data_root .../SCS_data --mode train_main
  python -m verl.tools.build_corrupted_datasets --data_root .../SCS_data --mode test_corrupt --benchmarks m3cot mmmu
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from io import BytesIO
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

# Add project root to path so we can import sibling module
sys.path.insert(0, str(Path(__file__).resolve().parent))
from imagenetc_corruptions import (
    CORRUPTION_NAMES,
    SEVERITY_TO_T,
    T_TO_SEVERITY,
    apply_random_corruption,
)

# ============================================================
# Constants
# ============================================================
BENCHMARKS = {
    "m3cot": "m3cot_test_processed",
    "mathverse": "mathverse_test_processed",
    "mathvision": "mathvision_test_processed",
    "mmmu": "mmmu_test_processed",
    "scienceqa": "scienceqa_test_processed",
    "wemath": "we_math_test_processed",
}

MAIN_T_MIX: tuple[tuple[str, float], ...] = (
    ("T0.2", 0.5),
    ("T0.4", 0.4),
    ("T0.6", 0.1),
)


# ============================================================
# Image I/O helpers
# ============================================================
def _strip_file_scheme(uri: str) -> str:
    if uri.startswith("file://"):
        return uri[len("file://"):]
    return uri


def _to_file_uri(path: Path) -> str:
    return f"file://{path.resolve()}"


def _load_image_from_uri(uri: str) -> Image.Image:
    """Load a PIL Image from a file:// URI or path."""
    path = Path(_strip_file_scheme(uri))
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {uri} → {path}")
    return Image.open(path).convert("RGB")


def _save_image(img: Image.Image, path: Path) -> str:
    """Save PIL Image to path, return file:// URI."""
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, format="PNG")
    return _to_file_uri(path)


# ============================================================
# Message JSON manipulation
# ============================================================
def _parse_message(raw: str | list) -> list[dict]:
    if isinstance(raw, str):
        return json.loads(raw)
    return raw


def _dump_message(msg_obj: list[dict], was_string: bool) -> str | list:
    if was_string:
        return json.dumps(msg_obj, ensure_ascii=False)
    return msg_obj


def _extract_image_uris_from_message(msg_obj: list[dict]) -> list[str]:
    """Find all image URIs in the message structure."""
    uris = []
    for msg in msg_obj:
        content = msg.get("content")
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image" and "image" in item:
                    uris.append(item["image"])
    return uris


def _replace_image_uris_in_message(msg_obj: list[dict], old_to_new: dict[str, str]) -> list[dict]:
    """Replace image URIs in message structure according to mapping."""
    for msg in msg_obj:
        content = msg.get("content")
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image" and "image" in item:
                    old = item["image"]
                    if old in old_to_new:
                        item["image"] = old_to_new[old]
    return msg_obj


# ============================================================
# Core corruption logic for a single row
# ============================================================
def corrupt_row_images(
    row: dict,
    severity: int,
    image_output_dir: Path,
    rng: np.random.Generator,
    message_cols: tuple[str, ...] = ("message_qwenvl", "message_internvl"),
) -> tuple[dict, str]:
    """Create a corrupted copy of a row.

    Returns (corrupted_row, corruption_name_used).
    """
    new_row = copy.deepcopy(row)
    corruption_name_applied = None

    # Find all image URIs across message columns
    all_uris = set()
    for col in message_cols:
        if col in new_row and new_row[col]:
            msg_obj = _parse_message(new_row[col])
            all_uris.update(_extract_image_uris_from_message(msg_obj))

    # Also check image_uris column
    if "image_uris" in new_row:
        img_uris = new_row["image_uris"]
        if isinstance(img_uris, list):
            all_uris.update(img_uris)

    if not all_uris:
        return new_row, "none"

    # Corrupt each unique image and build URI mapping
    uri_mapping = {}
    for uri in all_uris:
        try:
            img = _load_image_from_uri(uri)
        except FileNotFoundError:
            continue

        corrupted_img, c_name = apply_random_corruption(img, severity=severity, rng=rng)
        corruption_name_applied = c_name

        # Build output path: image_output_dir / {original_relative_structure}
        original_path = Path(_strip_file_scheme(uri))
        # Use sample_id and image filename for unique path
        sid = new_row.get("sample_id", original_path.parent.name)
        img_name = original_path.name
        out_path = image_output_dir / str(sid) / img_name
        new_uri = _save_image(corrupted_img, out_path)
        uri_mapping[uri] = new_uri

    # Update message columns with new URIs
    for col in message_cols:
        if col in new_row and new_row[col]:
            was_string = isinstance(new_row[col], str)
            msg_obj = _parse_message(new_row[col])
            msg_obj = _replace_image_uris_in_message(msg_obj, uri_mapping)
            new_row[col] = _dump_message(msg_obj, was_string)

    # Update image_uris list
    if "image_uris" in new_row and isinstance(new_row["image_uris"], list):
        new_row["image_uris"] = [uri_mapping.get(u, u) for u in new_row["image_uris"]]

    return new_row, corruption_name_applied or "none"


# ============================================================
# Build training pair parquets
# ============================================================
def build_train_pairs(data_root: Path, seed: int = 42):
    """Build 5 pair parquets for training (one per severity level)."""
    train_parquet = data_root / "train" / "train.parquet"
    if not train_parquet.exists():
        print(f"[ERROR] Train parquet not found: {train_parquet}")
        return

    table = pq.read_table(train_parquet)
    rows = table.to_pylist()
    n = len(rows)
    print(f"[train] Loaded {n} clean training samples")

    for severity in range(1, 6):
        t_label = SEVERITY_TO_T[severity]
        out_path = data_root / "train" / f"train_pair_{t_label}.parquet"
        img_dir = data_root / "train" / f"images_{t_label}"

        rng = np.random.default_rng(seed + severity)
        pair_rows = []
        noise_counts = {}

        t0 = time.time()
        for i, row in enumerate(rows):
            # Clean row
            clean_row = copy.deepcopy(row)
            clean_row["pair_id"] = i
            clean_row["view"] = "clean"
            clean_row["noise_type"] = "clean"
            clean_row["corruption_level"] = 0.0
            clean_row["severity"] = 0
            clean_row["is_noisy"] = False
            pair_rows.append(clean_row)

            # Corrupted row
            noisy_row, c_name = corrupt_row_images(
                row, severity=severity, image_output_dir=img_dir, rng=rng
            )
            noisy_row["pair_id"] = i
            noisy_row["view"] = "noisy"
            noisy_row["noise_type"] = c_name
            noisy_row["corruption_level"] = severity * 0.2
            noisy_row["severity"] = severity
            noisy_row["is_noisy"] = True
            pair_rows.append(noisy_row)
            noise_counts[c_name] = noise_counts.get(c_name, 0) + 1

            if (i + 1) % 2000 == 0:
                elapsed = time.time() - t0
                print(f"  [{t_label}] {i + 1}/{n} ({elapsed:.1f}s)")

        pq.write_table(pa.Table.from_pylist(pair_rows), out_path)
        elapsed = time.time() - t0
        print(
            f"[train] {t_label}: wrote {len(pair_rows)} rows to {out_path.name} "
            f"({elapsed:.1f}s, noise={noise_counts})"
        )


def _split_pair_rows(rows: list[dict], source_name: str) -> tuple[dict[int, dict], dict[int, dict]]:
    clean_by_pair: dict[int, dict] = {}
    noisy_by_pair: dict[int, dict] = {}

    for row in rows:
        if "pair_id" not in row or "view" not in row:
            raise ValueError(f"{source_name} is missing pair_id/view columns required for pair data")

        pair_id = int(row["pair_id"])
        view = row["view"]
        if view == "clean":
            clean_by_pair[pair_id] = row
        elif view == "noisy":
            noisy_by_pair[pair_id] = row

    return clean_by_pair, noisy_by_pair


def _compute_mix_counts(total: int, ratios: list[float]) -> list[int]:
    raw = [total * ratio for ratio in ratios]
    counts = [int(value) for value in raw]
    remainder = total - sum(counts)
    if remainder > 0:
        order = sorted(range(len(raw)), key=lambda idx: raw[idx] - counts[idx], reverse=True)
        for idx in order[:remainder]:
            counts[idx] += 1
    return counts


def build_train_main(data_root: Path, seed: int = 42):
    """Build train_pair_main.parquet: clean unchanged + mixed noisy (50/40/10)."""
    train_dir = data_root / "train"
    out_path = train_dir / "train_pair_main.parquet"

    clean_reference: dict[int, dict] | None = None
    noisy_sources: dict[str, dict[int, dict]] = {}

    for t_label, _ in MAIN_T_MIX:
        src_path = train_dir / f"train_pair_{t_label}.parquet"
        if not src_path.exists():
            print(f"[ERROR] Required source pair parquet not found: {src_path}")
            return

        rows = pq.read_table(src_path).to_pylist()
        clean_rows, noisy_rows = _split_pair_rows(rows, src_path.name)
        if clean_reference is None:
            clean_reference = clean_rows
        else:
            if set(clean_reference.keys()) != set(clean_rows.keys()):
                print(f"[ERROR] Pair-id mismatch between source files and {src_path.name}")
                return

        if clean_reference is not None and set(clean_reference.keys()) != set(noisy_rows.keys()):
            print(f"[ERROR] Missing noisy rows for some pair_ids in {src_path.name}")
            return
        noisy_sources[t_label] = noisy_rows

    if not clean_reference:
        print("[ERROR] Empty clean reference rows for main train set build")
        return

    pair_ids = sorted(clean_reference.keys())
    total_pairs = len(pair_ids)

    shuffled_ids = list(pair_ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(shuffled_ids)

    ratios = [ratio for _, ratio in MAIN_T_MIX]
    counts = _compute_mix_counts(total_pairs, ratios)

    assignment: dict[int, str] = {}
    start = 0
    for (t_label, _), count in zip(MAIN_T_MIX, counts):
        end = start + count
        for pair_id in shuffled_ids[start:end]:
            assignment[pair_id] = t_label
        start = end

    if len(assignment) != total_pairs:
        print("[ERROR] Failed to assign all pair_ids when building main train set")
        return

    pair_rows = []
    for pair_id in pair_ids:
        pair_rows.append(copy.deepcopy(clean_reference[pair_id]))
        src_t = assignment[pair_id]
        pair_rows.append(copy.deepcopy(noisy_sources[src_t][pair_id]))

    pq.write_table(pa.Table.from_pylist(pair_rows), out_path)

    mix_summary = {t_label: counts[idx] for idx, (t_label, _) in enumerate(MAIN_T_MIX)}
    print(
        f"[train_main] wrote {len(pair_rows)} rows ({total_pairs} clean + {total_pairs} noisy) "
        f"to {out_path.name}, noisy_mix={mix_summary}"
    )


# ============================================================
# Build corrupted test parquets
# ============================================================
def build_test_corrupt(
    data_root: Path,
    benchmarks: list[str] | None = None,
    seed: int = 42,
):
    """Build corrupted test sets: 6 benchmarks × 5 severity levels = 30 files."""
    if benchmarks is None:
        benchmarks = list(BENCHMARKS.keys())

    for bm_name in benchmarks:
        if bm_name not in BENCHMARKS:
            print(f"[skip] Unknown benchmark: {bm_name}")
            continue

        src_dir_name = BENCHMARKS[bm_name]
        src_parquet = data_root / src_dir_name / "test.parquet"
        if not src_parquet.exists():
            print(f"[skip] {src_parquet} not found")
            continue

        table = pq.read_table(src_parquet)
        rows = table.to_pylist()
        n = len(rows)
        print(f"[test] {bm_name}: loaded {n} clean test samples")

        for severity in range(1, 6):
            t_label = SEVERITY_TO_T[severity]
            out_dir = data_root / f"{src_dir_name}_{t_label}"
            out_parquet = out_dir / "test.parquet"
            img_dir = data_root / f"test_images_{t_label}" / bm_name

            rng = np.random.default_rng(seed + severity * 100 + hash(bm_name) % 1000)
            corrupt_rows = []
            noise_counts = {}

            t0 = time.time()
            for i, row in enumerate(rows):
                c_row, c_name = corrupt_row_images(
                    row, severity=severity, image_output_dir=img_dir, rng=rng
                )
                c_row["noise_type"] = c_name
                c_row["corruption_level"] = severity * 0.2
                c_row["severity"] = severity
                corrupt_rows.append(c_row)
                noise_counts[c_name] = noise_counts.get(c_name, 0) + 1

                if (i + 1) % 1000 == 0:
                    elapsed = time.time() - t0
                    print(f"  [{bm_name}/{t_label}] {i + 1}/{n} ({elapsed:.1f}s)")

            out_dir.mkdir(parents=True, exist_ok=True)
            pq.write_table(pa.Table.from_pylist(corrupt_rows), out_parquet)
            elapsed = time.time() - t0
            print(
                f"[test] {bm_name}/{t_label}: wrote {len(corrupt_rows)} rows "
                f"({elapsed:.1f}s, noise={noise_counts})"
            )


# ============================================================
# Verify dataset
# ============================================================
def verify_datasets(data_root: Path):
    """Quick verification: check file existence and row counts."""
    print("\n" + "=" * 60)
    print("DATASET VERIFICATION")
    print("=" * 60)

    # Training
    clean = data_root / "train" / "train.parquet"
    if clean.exists():
        n = len(pq.read_table(clean))
        print(f"\n[train] clean: {n} rows ✓")
    else:
        print(f"\n[train] clean: MISSING ✗")

    for sev in range(1, 6):
        t = SEVERITY_TO_T[sev]
        p = data_root / "train" / f"train_pair_{t}.parquet"
        if p.exists():
            n = len(pq.read_table(p))
            print(f"[train] pair_{t}: {n} rows ✓")
        else:
            print(f"[train] pair_{t}: MISSING ✗")

    main_p = data_root / "train" / "train_pair_main.parquet"
    if main_p.exists():
        n = len(pq.read_table(main_p))
        print(f"[train] pair_main: {n} rows ✓")
    else:
        print("[train] pair_main: MISSING ✗")

    # Test
    for bm_name, src_dir_name in BENCHMARKS.items():
        clean_p = data_root / src_dir_name / "test.parquet"
        if clean_p.exists():
            n = len(pq.read_table(clean_p))
            print(f"\n[test] {bm_name}/clean: {n} rows ✓")
        else:
            print(f"\n[test] {bm_name}/clean: MISSING ✗")

        for sev in range(1, 6):
            t = SEVERITY_TO_T[sev]
            p = data_root / f"{src_dir_name}_{t}" / "test.parquet"
            if p.exists():
                n = len(pq.read_table(p))
                print(f"[test] {bm_name}/{t}: {n} rows ✓")
            else:
                print(f"[test] {bm_name}/{t}: MISSING ✗")


# ============================================================
# CLI
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Build ImageNet-C corrupted datasets for confidence RL"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="/share/home/cuipeng/cuipeng_a100/yangboyao/datasets/SCS_data",
    )
    parser.add_argument(
        "--mode",
        choices=["train_pairs", "train_main", "test_corrupt", "all", "verify"],
        default="all",
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=None,
        help="Benchmarks to process (test mode). Default: all 6.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    if not data_root.exists():
        print(f"[ERROR] Data root not found: {data_root}")
        return

    if args.mode in ("train_pairs", "all"):
        print("\n" + "=" * 60)
        print("BUILDING TRAINING PAIR DATASETS")
        print("=" * 60)
        build_train_pairs(data_root, seed=args.seed)

    if args.mode in ("train_main", "all"):
        print("\n" + "=" * 60)
        print("BUILDING MAIN TRAINING PAIR DATASET")
        print("=" * 60)
        build_train_main(data_root, seed=args.seed)

    if args.mode in ("test_corrupt", "all"):
        print("\n" + "=" * 60)
        print("BUILDING CORRUPTED TEST DATASETS")
        print("=" * 60)
        build_test_corrupt(data_root, benchmarks=args.benchmarks, seed=args.seed)

    if args.mode == "verify" or args.mode == "all":
        verify_datasets(data_root)


if __name__ == "__main__":
    main()
