#!/usr/bin/env python3
"""Build ImageNet-C corrupted training/test datasets for confidence RL.

Creates corrupted versions of images at 5 severity levels (T0.2-T1.0).
For each sample, one corruption method is randomly chosen from the 8 available.

Training mode (--mode train_pairs):
  Reads the clean train.parquet, produces 5 pair parquets:
    train_pair_T0.2.parquet, ..., train_pair_T1.0.parquet
  Each pair parquet has 2x rows: clean + corrupted with same pair_id.

Main training set mode (--mode train_main):
  Reads pair datasets and builds train_pair_main.parquet with:
    - clean rows unchanged
    - noisy rows mixed as 50% T0.2 + 40% T0.4 + 10% T0.6

Test mode (--mode test_corrupt):
  For each benchmark x severity, produces a corrupted test.parquet:
    {benchmark}_test_processed_T{label}/test.parquet

All mode (--mode all):
  Runs both training and test dataset construction.

The public release expects a local RAC working tree under --data_root.
Optional source download mode materializes the six public benchmark snapshots
into a separate directory for local preprocessing and provenance tracking.

Examples:
    python -m verl.tools.build_corrupted_datasets --data_root ./rac_data --mode all
  python -m verl.tools.build_corrupted_datasets \
        --mode download_sources \
        --source_download_root ./upstream_sources
  python -m verl.tools.build_corrupted_datasets \
        --data_root ./rac_data \
        --download_sources \
        --source_download_root ./upstream_sources \
    --mode test_corrupt --benchmarks m3cot mmmu
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import time
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

BENCHMARKS = {
    "m3cot": "m3cot_test_processed",
    "mathverse": "mathverse_test_processed",
    "mathvision": "mathvision_test_processed",
    "mmmu": "mmmu_test_processed",
    "scienceqa": "scienceqa_test_processed",
    "wemath": "we_math_test_processed",
}

SOURCE_DATASET_ALIASES = {
    "we-math": "wemath",
    "we_math": "wemath",
}

SOURCE_DATASETS = {
    "m3cot": {
        "repo_id": "LightChen233/M3CoT",
        "url": "https://huggingface.co/datasets/LightChen233/M3CoT",
    },
    "mathverse": {
        "repo_id": "AI4Math/MathVerse",
        "url": "https://huggingface.co/datasets/AI4Math/MathVerse",
    },
    "mathvision": {
        "repo_id": "mathvision-bench/MathVision",
        "url": "https://huggingface.co/datasets/mathvision-bench/MathVision",
    },
    "mmmu": {
        "repo_id": "MMMU/MMMU",
        "url": "https://huggingface.co/datasets/MMMU/MMMU",
    },
    "scienceqa": {
        "repo_id": "derek-thomas/ScienceQA",
        "url": "https://huggingface.co/datasets/derek-thomas/ScienceQA",
    },
    "wemath": {
        "repo_id": "We-Math/We-Math",
        "url": "https://huggingface.co/datasets/We-Math/We-Math",
    },
}

SEVERITY_TO_T = {1: "T0.2", 2: "T0.4", 3: "T0.6", 4: "T0.8", 5: "T1.0"}

MAIN_T_MIX: tuple[tuple[str, float], ...] = (
    ("T0.2", 0.5),
    ("T0.4", 0.4),
    ("T0.6", 0.1),
)


def _apply_random_corruption(img: Image.Image, severity: int, rng: np.random.Generator):
    try:
        from .imagenetc_corruptions import apply_random_corruption
    except ImportError as exc:
        if __package__:
            raise ImportError(
                "Image corruption dependencies are unavailable. Install scipy and opencv-python-headless "
                "to use train_pairs/test_corrupt modes."
            ) from exc
        try:
            from imagenetc_corruptions import apply_random_corruption
        except ImportError as bare_exc:
            raise ImportError(
                "Image corruption dependencies are unavailable. Install scipy and opencv-python-headless "
                "to use train_pairs/test_corrupt modes."
            ) from bare_exc

    return apply_random_corruption(img, severity=severity, rng=rng)


def _strip_file_scheme(uri: str) -> str:
    if uri.startswith("file://"):
        return uri[len("file://"):]
    return uri


def _to_file_uri(path: Path) -> str:
    return f"file://{path.resolve()}"


def _parse_hf_dataset_source(value: str, default_revision: str) -> tuple[str, str]:
    text = value.strip().rstrip("/")
    if text.startswith(("http://", "https://")):
        prefix = "https://huggingface.co/datasets/"
        if not text.startswith(prefix):
            raise ValueError(f"Unsupported dataset URL: {value}")
        remainder = text[len(prefix):]
        parts = remainder.split("/")
        if len(parts) < 2:
            raise ValueError(f"Invalid dataset URL: {value}")
        repo_id = "/".join(parts[:2])
        revision = default_revision
        if len(parts) >= 4 and parts[2] in {"tree", "resolve"}:
            revision = parts[3]
        return repo_id, revision

    if text.count("/") != 1:
        raise ValueError(f"Invalid Hugging Face dataset repo id: {value}")
    return text, default_revision


def materialize_hf_dataset_snapshot(
    repo_id: str,
    revision: str,
    target_dir: Path,
    hf_cache_dir: str | None = None,
    force_refresh: bool = False,
) -> Path:
    from huggingface_hub import snapshot_download

    snapshot_dir = Path(
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            cache_dir=hf_cache_dir,
        )
    )
    target_dir = target_dir.expanduser()

    if force_refresh and target_dir.exists():
        shutil.rmtree(target_dir)

    if target_dir.exists():
        print(f"[hf] Using existing local dataset root: {target_dir}")
        return target_dir.resolve()

    target_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(snapshot_dir, target_dir, symlinks=True)
    print(f"[hf] Materialized {repo_id}@{revision} to {target_dir}")
    return target_dir.resolve()


def resolve_data_root(
    data_root: str | None,
    hf_dataset_repo: str | None,
    hf_dataset_revision: str,
    hf_cache_dir: str | None,
    force_refresh_source: bool,
) -> Path:
    if hf_dataset_repo:
        repo_id, revision = _parse_hf_dataset_source(hf_dataset_repo, hf_dataset_revision)
        local_root = Path(data_root).expanduser() if data_root else Path.cwd() / repo_id.rsplit("/", 1)[-1]
        return materialize_hf_dataset_snapshot(
            repo_id=repo_id,
            revision=revision,
            target_dir=local_root,
            hf_cache_dir=hf_cache_dir,
            force_refresh=force_refresh_source,
        )

    if not data_root:
        raise ValueError(
            "Either --data_root or --hf_dataset_repo must be provided for generation modes. "
            "Use --mode download_sources to fetch the public benchmark snapshots only."
        )

    local_root = Path(data_root).expanduser()
    if not local_root.exists():
        raise FileNotFoundError(f"Data root not found: {local_root}")
    return local_root.resolve()


def resolve_source_dataset_names(dataset_names: list[str] | None) -> list[str]:
    selected = dataset_names or list(SOURCE_DATASETS.keys())
    normalized: list[str] = []
    unknown: list[str] = []

    for name in selected:
        normalized_name = SOURCE_DATASET_ALIASES.get(name, name)
        if normalized_name not in SOURCE_DATASETS:
            unknown.append(name)
            continue
        if normalized_name not in normalized:
            normalized.append(normalized_name)

    if unknown:
        supported = ", ".join(sorted(SOURCE_DATASETS))
        raise ValueError(f"Unknown source datasets: {', '.join(unknown)}. Supported: {supported}")

    return normalized


def resolve_source_download_root(data_root: str | None, source_download_root: str | None) -> Path:
    if source_download_root:
        return Path(source_download_root).expanduser()
    if data_root:
        return Path(data_root).expanduser().parent / "upstream_sources"
    return Path.cwd() / "upstream_sources"


def download_source_datasets(
    source_download_root: Path,
    dataset_names: list[str] | None = None,
    hf_cache_dir: str | None = None,
    force_refresh: bool = False,
) -> dict[str, Path]:
    materialized: dict[str, Path] = {}
    for name in resolve_source_dataset_names(dataset_names):
        source = SOURCE_DATASETS[name]
        materialized[name] = materialize_hf_dataset_snapshot(
            repo_id=source["repo_id"],
            revision="main",
            target_dir=source_download_root / name,
            hf_cache_dir=hf_cache_dir,
            force_refresh=force_refresh,
        )

    print(f"[sources] Materialized {len(materialized)} source snapshot(s) under {source_download_root}")
    return materialized

def _load_image_from_uri(uri: str, source_root: Path | None = None) -> Image.Image:
    """Load a PIL image from a file:// URI, HTTP(S) URL, or path."""
    parsed = urlparse(uri)
    if parsed.scheme in {"http", "https"}:
        with urlopen(uri) as response:
            return Image.open(BytesIO(response.read())).convert("RGB")

    path = Path(_strip_file_scheme(uri))
    if not path.is_absolute() and source_root is not None:
        path = source_root / path
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {uri} -> {path}")
    return Image.open(path).convert("RGB")

def _save_image(img: Image.Image, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, format="PNG")
    return _to_file_uri(path)

def _parse_message(raw: str | list) -> list[dict]:
    if isinstance(raw, str):
        return json.loads(raw)
    return raw

def _dump_message(msg_obj: list[dict], was_string: bool) -> str | list:
    if was_string:
        return json.dumps(msg_obj, ensure_ascii=False)
    return msg_obj

def _extract_image_uris_from_message(msg_obj: list[dict]) -> list[str]:
    uris = []
    for msg in msg_obj:
        content = msg.get("content")
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image" and "image" in item:
                    uris.append(item["image"])
    return uris


def _replace_image_uris_in_message(msg_obj: list[dict], old_to_new: dict[str, str]) -> list[dict]:
    for msg in msg_obj:
        content = msg.get("content")
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image" and "image" in item:
                    old = item["image"]
                    if old in old_to_new:
                        item["image"] = old_to_new[old]
    return msg_obj


def corrupt_row_images(
    row: dict,
    severity: int,
    image_output_dir: Path,
    rng: np.random.Generator,
    source_root: Path | None = None,
    message_cols: tuple[str, ...] = ("message_qwenvl", "message_internvl"),
) -> tuple[dict, str]:
    """Create a corrupted copy of a row.

    Returns (corrupted_row, corruption_name_used).
    """
    new_row = copy.deepcopy(row)
    corruption_name_applied = None

    all_uris = set()
    for col in message_cols:
        if col in new_row and new_row[col]:
            msg_obj = _parse_message(new_row[col])
            all_uris.update(_extract_image_uris_from_message(msg_obj))

    if "image_uris" in new_row and isinstance(new_row["image_uris"], list):
        all_uris.update(new_row["image_uris"])

    if not all_uris:
        return new_row, "none"

    uri_mapping = {}
    for uri in all_uris:
        try:
            img = _load_image_from_uri(uri, source_root=source_root)
        except FileNotFoundError:
            continue

        corrupted_img, c_name = _apply_random_corruption(img, severity=severity, rng=rng)
        corruption_name_applied = c_name

        original_path = Path(_strip_file_scheme(uri))
        sample_id = new_row.get("sample_id", original_path.parent.name)
        image_name = original_path.name
        out_path = image_output_dir / str(sample_id) / image_name
        new_uri = _save_image(corrupted_img, out_path)
        uri_mapping[uri] = new_uri

    for col in message_cols:
        if col in new_row and new_row[col]:
            was_string = isinstance(new_row[col], str)
            msg_obj = _parse_message(new_row[col])
            msg_obj = _replace_image_uris_in_message(msg_obj, uri_mapping)
            new_row[col] = _dump_message(msg_obj, was_string)

    if "image_uris" in new_row and isinstance(new_row["image_uris"], list):
        new_row["image_uris"] = [uri_mapping.get(uri, uri) for uri in new_row["image_uris"]]

    return new_row, corruption_name_applied or "none"


def build_train_pairs(data_root: Path, seed: int = 42):
    """Build 5 pair parquets for training (one per severity level)."""
    train_parquet = data_root / "train" / "train.parquet"
    if not train_parquet.exists():
        print(
            f"[ERROR] Train parquet not found: {train_parquet}. "
            "Prepare the RAC working tree under --data_root before generating corrupted pairs."
        )
        return

    table = pq.read_table(train_parquet)
    rows = table.to_pylist()
    n_rows = len(rows)
    print(f"[train] Loaded {n_rows} clean training samples")

    for severity in range(1, 6):
        t_label = SEVERITY_TO_T[severity]
        out_path = data_root / "train" / f"train_pair_{t_label}.parquet"
        img_dir = data_root / "train" / f"images_{t_label}"

        rng = np.random.default_rng(seed + severity)
        pair_rows = []
        noise_counts = {}

        t0 = time.time()
        for i, row in enumerate(rows):
            clean_row = copy.deepcopy(row)
            clean_row["pair_id"] = i
            clean_row["view"] = "clean"
            clean_row["noise_type"] = "clean"
            clean_row["corruption_level"] = 0.0
            clean_row["severity"] = 0
            clean_row["is_noisy"] = False
            pair_rows.append(clean_row)

            noisy_row, c_name = corrupt_row_images(
                row,
                severity=severity,
                image_output_dir=img_dir,
                rng=rng,
                source_root=data_root,
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
                print(f"  [{t_label}] {i + 1}/{n_rows} ({elapsed:.1f}s)")

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
        elif set(clean_reference.keys()) != set(clean_rows.keys()):
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


def _stable_hash_offset(text: str) -> int:
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 1000


def build_test_corrupt(data_root: Path, benchmarks: list[str] | None = None, seed: int = 42):
    """Build corrupted test sets: 6 benchmarks x 5 severity levels = 30 files."""
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
        n_rows = len(rows)
        print(f"[test] {bm_name}: loaded {n_rows} clean test samples")

        for severity in range(1, 6):
            t_label = SEVERITY_TO_T[severity]
            out_dir = data_root / f"{src_dir_name}_{t_label}"
            out_parquet = out_dir / "test.parquet"
            img_dir = data_root / f"test_images_{t_label}" / bm_name

            rng = np.random.default_rng(seed + severity * 100 + _stable_hash_offset(bm_name))
            corrupt_rows = []
            noise_counts = {}

            t0 = time.time()
            for i, row in enumerate(rows):
                c_row, c_name = corrupt_row_images(
                    row,
                    severity=severity,
                    image_output_dir=img_dir,
                    rng=rng,
                    source_root=data_root,
                )
                c_row["noise_type"] = c_name
                c_row["corruption_level"] = severity * 0.2
                c_row["severity"] = severity
                corrupt_rows.append(c_row)
                noise_counts[c_name] = noise_counts.get(c_name, 0) + 1

                if (i + 1) % 1000 == 0:
                    elapsed = time.time() - t0
                    print(f"  [{bm_name}/{t_label}] {i + 1}/{n_rows} ({elapsed:.1f}s)")

            out_dir.mkdir(parents=True, exist_ok=True)
            pq.write_table(pa.Table.from_pylist(corrupt_rows), out_parquet)
            elapsed = time.time() - t0
            print(
                f"[test] {bm_name}/{t_label}: wrote {len(corrupt_rows)} rows "
                f"({elapsed:.1f}s, noise={noise_counts})"
            )


def verify_datasets(data_root: Path):
    """Quick verification: check file existence and row counts."""
    print("\n" + "=" * 60)
    print("DATASET VERIFICATION")
    print("=" * 60)

    clean = data_root / "train" / "train.parquet"
    if clean.exists():
        n_rows = len(pq.read_table(clean))
        print(f"\n[train] clean: {n_rows} rows")
    else:
        print("\n[train] clean: MISSING")

    for severity in range(1, 6):
        t_label = SEVERITY_TO_T[severity]
        path = data_root / "train" / f"train_pair_{t_label}.parquet"
        if path.exists():
            n_rows = len(pq.read_table(path))
            print(f"[train] pair_{t_label}: {n_rows} rows")
        else:
            print(f"[train] pair_{t_label}: MISSING")

    main_path = data_root / "train" / "train_pair_main.parquet"
    if main_path.exists():
        n_rows = len(pq.read_table(main_path))
        print(f"[train] pair_main: {n_rows} rows")
    else:
        print("[train] pair_main: MISSING")

    for bm_name, src_dir_name in BENCHMARKS.items():
        clean_path = data_root / src_dir_name / "test.parquet"
        if clean_path.exists():
            n_rows = len(pq.read_table(clean_path))
            print(f"\n[test] {bm_name}/clean: {n_rows} rows")
        else:
            print(f"\n[test] {bm_name}/clean: MISSING")

        for severity in range(1, 6):
            t_label = SEVERITY_TO_T[severity]
            path = data_root / f"{src_dir_name}_{t_label}" / "test.parquet"
            if path.exists():
                n_rows = len(pq.read_table(path))
                print(f"[test] {bm_name}/{t_label}: {n_rows} rows")
            else:
                print(f"[test] {bm_name}/{t_label}: MISSING")


def main():
    parser = argparse.ArgumentParser(description="Build ImageNet-C corrupted datasets for confidence RL")
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Local RAC dataset working tree containing train/train.parquet and *_test_processed/test.parquet.",
    )
    parser.add_argument(
        "--hf_dataset_repo",
        type=str,
        default=None,
        help="Optional private prepared working-tree snapshot repo id or dataset URL to materialize locally.",
    )
    parser.add_argument(
        "--hf_dataset_revision",
        type=str,
        default="main",
        help="Revision used when materializing --hf_dataset_repo.",
    )
    parser.add_argument(
        "--hf_cache_dir",
        type=str,
        default=None,
        help="Optional Hugging Face cache dir used for snapshot downloads.",
    )
    parser.add_argument(
        "--force_refresh_source",
        action="store_true",
        help="Re-materialize downloaded snapshots even if the local target directory already exists.",
    )
    parser.add_argument(
        "--download_sources",
        action="store_true",
        help="Download the six public benchmark source datasets before running generation steps.",
    )
    parser.add_argument(
        "--source_download_root",
        type=str,
        default=None,
        help="Target directory used by --download_sources or --mode download_sources.",
    )
    parser.add_argument(
        "--source_datasets",
        nargs="+",
        default=None,
        help="Subset of public source datasets to download. Default: all 6 benchmark sources.",
    )
    parser.add_argument(
        "--mode",
        choices=["download_sources", "train_pairs", "train_main", "test_corrupt", "all", "verify"],
        default="all",
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=None,
        help="Benchmarks to process in test mode. Default: all 6.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.download_sources or args.mode == "download_sources":
        try:
            download_source_datasets(
                source_download_root=resolve_source_download_root(args.data_root, args.source_download_root),
                dataset_names=args.source_datasets,
                hf_cache_dir=args.hf_cache_dir,
                force_refresh=args.force_refresh_source,
            )
        except (ImportError, ValueError) as exc:
            print(f"[ERROR] {exc}")
            return

        if args.mode == "download_sources":
            return

    try:
        data_root = resolve_data_root(
            data_root=args.data_root,
            hf_dataset_repo=args.hf_dataset_repo,
            hf_dataset_revision=args.hf_dataset_revision,
            hf_cache_dir=args.hf_cache_dir,
            force_refresh_source=args.force_refresh_source,
        )
    except (FileNotFoundError, ImportError, ValueError) as exc:
        print(f"[ERROR] {exc}")
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

    if args.mode in {"verify", "all"}:
        verify_datasets(data_root)


if __name__ == "__main__":
    main()