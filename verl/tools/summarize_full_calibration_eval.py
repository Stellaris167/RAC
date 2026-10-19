"""Summarize the paper-standard 6-benchmark, 6-severity evaluation."""

import argparse
import json
from pathlib import Path

import numpy as np

from verl.tools.summarize_main_validation import read_rows, summarize_rows


BENCHMARKS = ("m3cot", "mathverse", "mathvision", "mmmu", "scienceqa", "wemath")
CONDITIONS = ("clean", "T0.2", "T0.4", "T0.6", "T0.8", "T1.0")
BANDS = {
    "clean": ("clean",),
    "mild": ("T0.2", "T0.4"),
    "severe": ("T0.6", "T0.8", "T1.0"),
}
PAPER_METRICS = ("accuracy", "ece", "brier")


def split_data_source(data_source: str) -> tuple[str, str]:
    benchmark, separator, condition = data_source.rpartition("@")
    if not separator or not benchmark or condition not in CONDITIONS:
        raise ValueError(f"Unexpected data_source: {data_source!r}")
    return benchmark, condition


def _macro_average(cell_summaries, conditions):
    selected = [
        cell_summaries[benchmark][condition]
        for benchmark in BENCHMARKS
        for condition in conditions
    ]
    return {
        metric: float(np.mean([cell[metric] for cell in selected]))
        for metric in PAPER_METRICS
    }


def summarize_full_evaluation(rows):
    grouped = {(benchmark, condition): [] for benchmark in BENCHMARKS for condition in CONDITIONS}
    for row in rows:
        benchmark, condition = split_data_source(row["data_source"])
        key = (benchmark, condition)
        if key not in grouped:
            raise ValueError(f"Unexpected benchmark-condition cell: {key}")
        grouped[key].append(row)

    missing = [key for key, cell_rows in grouped.items() if not cell_rows]
    if missing:
        raise ValueError(f"Missing evaluation cells: {missing}")

    cells = {
        benchmark: {
            condition: summarize_rows(grouped[(benchmark, condition)])
            for condition in CONDITIONS
        }
        for benchmark in BENCHMARKS
    }
    bands = {name: _macro_average(cells, conditions) for name, conditions in BANDS.items()}
    bands["avg"] = {
        metric: float(np.mean([bands[name][metric] for name in BANDS]))
        for metric in PAPER_METRICS
    }
    return {
        "rows": len(rows),
        "cells": cells,
        "bands": bands,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expected-rows", type=int)
    args = parser.parse_args()

    rows = read_rows(args.input)
    if args.expected_rows is None:
        if len(rows) not in {63_660, 10_596}:
            raise ValueError(f"Expected 63660 or 10596 rows, found {len(rows)} in {args.input}")
    elif len(rows) != args.expected_rows:
        raise ValueError(f"Expected {args.expected_rows} rows, found {len(rows)} in {args.input}")
    result = summarize_full_evaluation(rows)
    serialized = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized)
    print(serialized)


if __name__ == "__main__":
    main()
