"""Custom dataset for multimodal confidence RL training.

Reads parquet with columns: message_qwenvl/message_internvl, answer, dataset_name, etc.
Converts to verl format: raw_prompt, data_source, reward_model, extra_info.

Supports both Qwen-VL and InternVL message formats via prompt_col config.
"""
from __future__ import annotations

import copy
import glob
import json
import logging
import os
import re
from typing import Optional

import datasets
import numpy as np
import torch
from omegaconf import DictConfig, ListConfig
from PIL import Image
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

from verl.utils.dataset.rl_dataset import RLHFDataset

logger = logging.getLogger(__name__)

RESPONSE_SCHEMA_INSTRUCTION = (
    "\nPlease reason step by step and follow this exact response schema:\n"
    "<think>\n"
    "your reasoning\n"
    "</think>\n"
    "<answer>your final answer</answer>\n"
    "<confidence>your confidence</confidence>\n"
    "If options are provided, put exactly one uppercase option letter inside <answer> with no explanation or punctuation.\n"
    "Otherwise, put only the final value or short expression inside <answer>.\n"
    "Inside <confidence>, output exactly one decimal number between 0.0 and 1.0.\n"
    "Start your response with <think> and do not output any text before <think> or after </confidence>."
)

_SCHEMA_BLOCK_RE = re.compile(
    r"\n?Follow this exact response schema:\s*<think>.*?Start your response with <think> and do not output any text before <think> or after </confidence>\.?,?",
    re.IGNORECASE | re.DOTALL,
)

_LEGACY_PROMPT_PREFIX_RE = re.compile(
    r"^\s*Read the question and use the image if provided\.\s*",
    re.IGNORECASE,
)

_LEGACY_TRAILING_ANSWER_RE = re.compile(r"\nAnswer:\s*$", re.IGNORECASE)

_LEGACY_SCHEMA_SNIPPETS = (
    "Think step by step inside <think>...</think>.",
    "Then give your final answer inside <answer>...</answer>.",
    "If options are provided, answer with only the option letter inside <answer>.",
    "Otherwise, answer with only the final value or short expression inside <answer>.",
    "Also state your confidence (0-100) inside <confidence>...</confidence>.",
    "Also state your confidence as a decimal between 0.0 and 1.0 inside <confidence>...</confidence>.",
)

_LEADING_MM_PLACEHOLDERS_RE = re.compile(r"^(?:\s*(?:<image>|<video>)\s*)+")


def _normalize_response_schema(text: str) -> str:
    normalized = _LEGACY_PROMPT_PREFIX_RE.sub("", text)
    normalized = _SCHEMA_BLOCK_RE.sub("", normalized)
    for snippet in _LEGACY_SCHEMA_SNIPPETS:
        normalized = normalized.replace(snippet, "")
    normalized = _LEGACY_TRAILING_ANSWER_RE.sub("", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    normalized = normalized.lstrip()
    normalized = normalized.rstrip()
    if RESPONSE_SCHEMA_INSTRUCTION not in normalized:
        normalized += RESPONSE_SCHEMA_INSTRUCTION
    return normalized


def _strip_redundant_multimodal_placeholders(messages: list[dict]) -> list[dict]:
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            continue

        has_explicit_mm_item = any(
            isinstance(item, dict) and item.get("type") in {"image", "video"}
            for item in content
        )
        if not has_explicit_mm_item:
            continue

        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text", "")
                cleaned = _LEADING_MM_PLACEHOLDERS_RE.sub("", text)
                if cleaned != text:
                    item["text"] = cleaned.lstrip("\n")

    return messages


def inject_confidence_prompt(messages: list[dict]) -> list[dict]:
    messages = _strip_redundant_multimodal_placeholders(messages)
    for msg in reversed(messages):
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        if isinstance(content, list):
            for item in reversed(content):
                if isinstance(item, dict) and item.get("type") == "text":
                    item["text"] = _normalize_response_schema(item.get("text", ""))
                    break
        elif isinstance(content, str):
            msg["content"] = _normalize_response_schema(content)
        break
    return messages


def _format_condition_label(corruption_level) -> str:
    if corruption_level is None:
        return "clean"
    if isinstance(corruption_level, str):
        text = corruption_level.strip()
        if not text:
            return "clean"
        if text.lower() == "clean":
            return "clean"
        if text.startswith("T"):
            return text
        try:
            corruption_level = float(text)
        except ValueError:
            return text
    try:
        return f"T{float(corruption_level):.1f}"
    except (TypeError, ValueError):
        return str(corruption_level)


class ConfidenceRLDataset(Dataset):
    ANSWER_COL = "answer"
    SOURCE_COL = "dataset_name"
    MC_COL = "is_multiple_choice"
    META_KEYS = ("pair_id", "view", "noise_type", "noise_level", "corruption_level", "severity", "is_noisy")

    def __init__(
        self,
        data_files: str | list[str],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
        max_samples: int = -1,
    ):
        if not isinstance(data_files, (list, ListConfig)):
            data_files = [data_files]
        self.data_files = list(data_files)
        self.original_data_files = list(data_files)
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.max_samples = max_samples
        self.inject_confidence = config.get("inject_confidence", True)
        self.prompt_col = config.get("prompt_key", "message_qwenvl")
        self.cache_dir = os.path.expanduser(config.get("cache_dir", "~/.cache/verl/rlhf"))
        self._download()
        self._read()

    def _download(self):
        from verl.utils.fs import copy_to_local

        for index, data_file in enumerate(self.data_files):
            self.data_files[index] = copy_to_local(src=data_file, cache_dir=self.cache_dir)

    @staticmethod
    def _expand_path(path: str) -> list[str]:
        if os.path.isdir(path):
            found = sorted(glob.glob(os.path.join(path, "**", "*.parquet"), recursive=True))
            if not found:
                found = sorted(glob.glob(os.path.join(path, "**", "*.jsonl"), recursive=True))
            return found if found else [path]
        if "*" in path or "?" in path:
            return sorted(glob.glob(path, recursive=True))
        return [path]

    def _read(self):
        resolved = []
        for data_file in self.data_files:
            resolved.extend(self._expand_path(data_file))
        frames = []
        for data_file in resolved:
            if data_file.endswith(".parquet"):
                dataset = datasets.load_dataset("parquet", data_files=data_file)["train"]
            elif data_file.endswith(".json") or data_file.endswith(".jsonl"):
                dataset = datasets.load_dataset("json", data_files=data_file)["train"]
            else:
                raise ValueError(f"Unsupported format: {data_file}")
            frames.append(dataset)
        self.dataframe = datasets.concatenate_datasets(frames)
        total = len(self.dataframe)
        logger.info("Loaded %d samples from %d file(s)", total, len(self.data_files))
        if 0 < self.max_samples < total:
            rng = np.random.default_rng(42)
            indices = rng.choice(total, size=self.max_samples, replace=False)
            self.dataframe = self.dataframe.select(indices.tolist())
            logger.info("Sub-sampled to %d", self.max_samples)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, item: int) -> dict:
        row = self.dataframe[item]
        raw = row.get(self.prompt_col) or row.get("prompt", "[]")
        if isinstance(raw, str):
            try:
                messages = json.loads(raw)
            except json.JSONDecodeError:
                messages = [{"role": "user", "content": raw}]
        elif isinstance(raw, list):
            messages = raw
        else:
            messages = [{"role": "user", "content": str(raw)}]

        messages = copy.deepcopy(messages)
        if self.inject_confidence:
            messages = inject_confidence_prompt(messages)
        base_data_source = str(row.get(self.SOURCE_COL, "unknown"))
        ground_truth = row.get(self.ANSWER_COL, "")
        extra_info = {}
        if isinstance(row.get("extra_info"), dict):
            extra_info = dict(row["extra_info"])
        extra_info["is_multiple_choice"] = bool(row.get(self.MC_COL, True))
        for key in self.META_KEYS:
            value = row.get(key)
            if value is not None:
                extra_info[key] = value
        choices = row.get("choices")
        if choices is not None:
            extra_info["choices"] = choices
        extra_info.setdefault("base_data_source", base_data_source)
        condition_label = _format_condition_label(extra_info.get("corruption_level"))
        extra_info.setdefault("eval_condition", condition_label)
        split_data_source = self.config.get("split_data_source_by_corruption", True)
        data_source = f"{base_data_source}@{condition_label}" if split_data_source else base_data_source
        return {
            "raw_prompt": messages,
            "data_source": data_source,
            "reward_model": {"ground_truth": ground_truth, "style": "rule"},
            "extra_info": extra_info,
            "dummy_tensor": torch.tensor([0], dtype=torch.uint8),
            "index": extra_info.get("index", 0),
            "tools_kwargs": {},
            "interaction_kwargs": {},
        }

    def resume_dataset_state(self):
        self._download()
        self._read()

    @classmethod
    async def process_vision_info(
        cls,
        messages: list[dict],
        image_patch_size,
        config: DictConfig,
    ) -> tuple[list[Image.Image], list[tuple[torch.Tensor, dict]]]:
        return await RLHFDataset.process_vision_info(
            messages=messages,
            image_patch_size=image_patch_size,
            config=config,
        )