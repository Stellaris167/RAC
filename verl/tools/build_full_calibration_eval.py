import argparse
from pathlib import Path

import numpy as np
import pandas as pd


BENCHMARKS = ("m3cot", "mathverse", "mathvision", "mmmu", "scienceqa", "we_math")
SEVERITIES = (
    (0.0, "", 0),
    (0.2, "_T0.2", 1),
    (0.4, "_T0.4", 2),
    (0.6, "_T0.6", 3),
    (0.8, "_T0.8", 4),
    (1.0, "_T1.0", 5),
)


def build_full_evaluation(
    data_root: Path,
    sample_fraction: float = 1.0,
    seed: int = 42,
) -> pd.DataFrame:
    if not 0 < sample_fraction <= 1:
        raise ValueError(f"sample_fraction must be in (0, 1], got {sample_fraction}")

    frames = []
    keys = ["sample_id", "question", "answer"]
    rng = np.random.default_rng(seed)
    for benchmark in BENCHMARKS:
        clean = pd.read_parquet(data_root / f"{benchmark}_test_processed" / "test.parquet")
        selected = np.arange(len(clean))
        if sample_fraction < 1:
            sample_size = max(1, int(len(clean) * sample_fraction))
            selected = np.sort(rng.choice(len(clean), size=sample_size, replace=False))
        for corruption_level, suffix, severity in SEVERITIES:
            source = clean if not suffix else pd.read_parquet(
                data_root / f"{benchmark}_test_processed{suffix}" / "test.parquet"
            )
            assert len(source) == len(clean), f"{benchmark}{suffix}: row-count mismatch"
            assert clean[keys].equals(source[keys]), f"{benchmark}{suffix}: rows are not aligned"
            frame = source.iloc[selected].copy()
            frame["corruption_level"] = corruption_level
            frame["is_noisy"] = int(corruption_level > 0)
            frame["severity"] = severity
            if corruption_level == 0:
                frame["noise_type"] = "clean"
            frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-fraction", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    frame = build_full_evaluation(
        args.data_root,
        sample_fraction=args.sample_fraction,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.output, index=False)
    print(f"Wrote {len(frame)} full clean/corrupted evaluation rows to {args.output}")


if __name__ == "__main__":
    main()
