import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--samples', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    root = Path(args.data_root)
    frames = []
    for benchmark in ('m3cot', 'mathverse', 'mathvision', 'mmmu', 'scienceqa', 'we_math'):
        clean = pd.read_parquet(root / f'{benchmark}_test_processed/test.parquet')
        noisy = pd.read_parquet(root / f'{benchmark}_test_processed_T0.6/test.parquet')
        # WE-Math reuses sample IDs across subquestions; match the aligned rows.
        keys = ['sample_id', 'question', 'answer']
        assert clean[keys].equals(noisy[keys]), f'{benchmark}: clean/corrupted rows are not aligned'
        selected = clean.sample(n=args.samples, random_state=args.seed).copy()
        paired = noisy.iloc[selected.index].copy()
        selected['corruption_level'] = 0.0
        selected['is_noisy'] = 0
        paired['corruption_level'] = 0.6
        paired['is_noisy'] = 1
        frames.extend([selected, paired])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.concat(frames, ignore_index=True)
    frame.to_parquet(output, index=False)
    print(f'Wrote {len(frame)} matched clean/T0.6 evaluation rows to {output}')


if __name__ == '__main__':
    main()
