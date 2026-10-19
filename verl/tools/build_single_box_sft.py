from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import pandas as pd

from verl.trainer.reward_fn.confidence_dataset import (
    _rewrite_local_media_paths,
    inject_confidence_prompt,
)

CONFIDENCE_PLACEHOLDERS = tuple(f"{value / 10:.1f}" for value in range(11))


def format_sft_answer(answer: Any) -> str:
    if answer is None or pd.isna(answer):
        return ""
    text = str(answer).strip()
    return text.upper() if len(text) == 1 and text.isalpha() else text


def build_assistant_message(answer: Any, confidence_placeholder: str | None = None) -> dict[str, Any]:
    answer_text = format_sft_answer(answer)
    if confidence_placeholder is not None:
        return {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": f"\\boxed{{answer={answer_text}; confidence={confidence_placeholder}}}",
                }
            ],
            "loss_mask_text": confidence_placeholder,
        }
    return {
        "role": "assistant",
        "partial": True,
        "content": [
            {
                "type": "text",
                "text": f"\\boxed{{answer={answer_text}; confidence=",
            }
        ],
    }


def load_prompt_messages(raw: Any) -> list[dict]:
    if isinstance(raw, str):
        try:
            messages = json.loads(raw)
        except json.JSONDecodeError:
            messages = [{"role": "user", "content": raw}]
    elif isinstance(raw, list):
        messages = raw
    else:
        messages = [{"role": "user", "content": str(raw)}]
    return copy.deepcopy(messages)


def strip_file_scheme_from_media(messages: list[dict]) -> list[dict]:
    def _strip(value):
        if isinstance(value, dict):
            return {
                key: value_item.removeprefix("file://")
                if key in {"image", "video"} and isinstance(value_item, str)
                else _strip(value_item)
                for key, value_item in value.items()
            }
        if isinstance(value, list):
            return [_strip(item) for item in value]
        return value

    return _strip(messages)


def build_sft_dataframe(
    input_path: str | Path,
    *,
    prompt_key: str = "message_qwenvl",
    local_data_root: str | None = None,
    max_samples: int = -1,
    response_schema: str = "boxed_sft",
    full_structure_target: bool = False,
) -> pd.DataFrame:
    dataframe = pd.read_parquet(input_path)
    if max_samples > 0:
        dataframe = dataframe.head(max_samples)

    rows: list[dict[str, Any]] = []
    for _, row in dataframe.iterrows():
        messages = load_prompt_messages(row.get(prompt_key, ""))
        messages = _rewrite_local_media_paths(messages, local_data_root)
        messages = strip_file_scheme_from_media(messages)
        messages = inject_confidence_prompt(messages, response_schema=response_schema)
        confidence_placeholder = CONFIDENCE_PLACEHOLDERS[len(rows) % len(CONFIDENCE_PLACEHOLDERS)]
        messages.append(
            build_assistant_message(
                row.get("answer"),
                confidence_placeholder if full_structure_target else None,
            )
        )

        item = {
            "messages": messages,
            "answer": format_sft_answer(row.get("answer")),
        }
        for key in ("dataset_name", "sample_id", "source_task"):
            if key in row and not pd.isna(row[key]):
                item[key] = row[key]
        rows.append(item)

    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build single-box SFT data for RAC format warm-start.")
    parser.add_argument("--input", required=True, help="Input RAC training parquet.")
    parser.add_argument("--output", required=True, help="Output SFT parquet with a messages column.")
    parser.add_argument("--prompt-key", default="message_qwenvl", help="Prompt column in the input parquet.")
    parser.add_argument("--local-data-root", default=None, help="Local SCS_data root used to rewrite media paths.")
    parser.add_argument("--max-samples", type=int, default=-1, help="Optional cap for smoke data generation.")
    parser.add_argument("--response-schema", choices=("boxed_sft", "boxed_direct"), default="boxed_sft")
    parser.add_argument("--full-structure-target", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe = build_sft_dataframe(
        args.input,
        prompt_key=args.prompt_key,
        local_data_root=args.local_data_root,
        max_samples=args.max_samples,
        response_schema=args.response_schema,
        full_structure_target=args.full_structure_target,
    )
    dataframe.to_parquet(output_path, index=False)
    print(f"Wrote {len(dataframe)} SFT rows to {output_path}")


if __name__ == "__main__":
    main()
