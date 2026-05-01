# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Callable

from omegaconf import DictConfig
from transformers import AutoTokenizer

from verl import DataProto
from verl.utils.ray_utils import get_event_loop

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _json_safe(value: Any):
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            return repr(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _render_prompt_text(raw_prompt: Any) -> str:
    if isinstance(raw_prompt, str):
        return raw_prompt
    if not isinstance(raw_prompt, list):
        return repr(raw_prompt)

    rendered_messages = []
    for message in raw_prompt:
        if not isinstance(message, dict):
            rendered_messages.append(repr(message))
            continue
        role = message.get("role", "unknown")
        content = message.get("content", "")
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            parts = []
            for item in content:
                if not isinstance(item, dict):
                    parts.append(repr(item))
                    continue
                item_type = item.get("type")
                if item_type == "text":
                    parts.append(str(item.get("text", "")))
                elif item_type == "image":
                    parts.append("<image>")
                elif item_type == "video":
                    parts.append("<video>")
                else:
                    parts.append(repr(item))
            text = "".join(parts)
        else:
            text = repr(content)
        rendered_messages.append(f"[{role}]\n{text}")
    return "\n\n".join(rendered_messages)


RawRewardFn = Callable[..., Any] | None


class RewardManagerBase(ABC):
    _class_initialized = False

    def __init__(self, config: DictConfig, tokenizer: AutoTokenizer, compute_score: RawRewardFn):
        """Initialize reward manager.

        Args:
            config (DictConfig): YAML config.
            tokenizer (AutoTokenizer): Tokenizer for tokenize messages.
        """
        self.config = config
        self.tokenizer = tokenizer
        self.compute_score = compute_score
        self.loop = get_event_loop()
        reward_manager_cfg = config.reward.reward_manager
        self.dump_pairs = bool(reward_manager_cfg.get("dump_pairs", False))
        self.dump_max_samples_per_worker = int(reward_manager_cfg.get("dump_max_samples_per_worker", 0) or 0)
        self.dump_pairs_count = 0
        self.dump_file_path = None
        if self.dump_pairs and self.dump_max_samples_per_worker != 0:
            dump_dir = reward_manager_cfg.get("dump_dir") or os.path.join(os.getcwd(), "reward_pair_dumps")
            dump_dir = os.path.abspath(os.path.expanduser(dump_dir))
            os.makedirs(dump_dir, exist_ok=True)
            self.dump_file_path = os.path.join(dump_dir, f"reward_loop_worker_{os.getpid()}.txt")
        self.init_class(config, tokenizer)

    def maybe_dump_prompt_response(
        self,
        *,
        data_source: str,
        ground_truth: Any,
        extra_info: dict,
        raw_prompt: Any,
        response_str: str,
        reward_extra_info: dict,
        response_token_ids: Any = None,
    ) -> None:
        if not self.dump_pairs or self.dump_file_path is None:
            return
        if self.dump_max_samples_per_worker > 0 and self.dump_pairs_count >= self.dump_max_samples_per_worker:
            return

        prompt_text = _render_prompt_text(raw_prompt)
        payload = {
            "pid": os.getpid(),
            "sample_index": self.dump_pairs_count,
            "data_source": data_source,
            "ground_truth": _json_safe(ground_truth),
            "extra_info": _json_safe(extra_info),
            "reward_extra_info": _json_safe(reward_extra_info),
            "prompt_text": prompt_text,
            "raw_prompt": _json_safe(raw_prompt),
            "response": response_str,
            "response_token_ids": _json_safe(response_token_ids),
        }
        with open(self.dump_file_path, "a", encoding="utf-8") as fout:
            fout.write(f"===== SAMPLE {self.dump_pairs_count} =====\n")
            fout.write(json.dumps(payload, ensure_ascii=False, indent=2))
            fout.write("\n\n")
        self.dump_pairs_count += 1

    @classmethod
    def init_class(cls, config: DictConfig, tokenizer: AutoTokenizer):
        """Initialize class state shared across all instances."""
        if cls._class_initialized:
            return
        cls._class_initialized = True

    @abstractmethod
    async def run_single(self, data: DataProto):
        raise NotImplementedError
