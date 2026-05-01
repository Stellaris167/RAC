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
"""Utils for tokenization."""

import os
import types
import warnings

__all__ = ["bind_processor_tokens_to_model", "hf_tokenizer", "hf_processor", "normalize_token_ids"]


def _maybe_enable_fix_mistral_regex(name_or_path, kwargs: dict):
    """Enable the tokenizer regex fix for known affected Mistral-family tokenizers."""

    if "fix_mistral_regex" in kwargs:
        return kwargs

    if isinstance(name_or_path, str):
        normalized_name = os.path.normpath(name_or_path).lower()
        if any(marker in normalized_name for marker in ("mistral", "internvl3.5", "internvl3_5")):
            kwargs["fix_mistral_regex"] = True

    return kwargs


def bind_processor_tokens_to_model(model, processor):
    """Bind processor-derived special token ids onto multimodal model instances."""

    image_context_token_id = getattr(processor, "image_context_token_id", None)
    if model is not None and image_context_token_id is not None:
        setattr(model, "img_context_token_id", image_context_token_id)

    return model


def _build_internvl_processor(name_or_path, config, kwargs):
    from transformers import AutoTokenizer

    from .internvl_processor import InternVLProcessor

    tokenizer_kwargs = _maybe_enable_fix_mistral_regex(name_or_path, dict(kwargs))
    tokenizer_kwargs["use_fast"] = False
    tokenizer = AutoTokenizer.from_pretrained(name_or_path, **tokenizer_kwargs)
    set_pad_token_id(tokenizer)
    return InternVLProcessor(tokenizer=tokenizer, config=config)


def normalize_token_ids(tokenized_output) -> list[int]:
    """Normalize tokenizer outputs into a flat ``list[int]``.

    This handles Transformers 4/5 differences where ``apply_chat_template(tokenize=True)``
    may return either ``list[int]`` or a ``BatchEncoding``/mapping with ``input_ids``.
    """

    token_ids = tokenized_output
    if isinstance(tokenized_output, dict):
        if "input_ids" in tokenized_output:
            token_ids = tokenized_output["input_ids"]
    elif hasattr(tokenized_output, "input_ids"):
        token_ids = tokenized_output.input_ids

    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()

    if isinstance(token_ids, tuple):
        token_ids = list(token_ids)

    if isinstance(token_ids, list) and len(token_ids) == 1 and isinstance(token_ids[0], list | tuple):
        token_ids = list(token_ids[0])

    if not isinstance(token_ids, list):
        raise TypeError(f"token_ids must be list-like token ids, got {type(token_ids).__name__}: {token_ids!r}")

    normalized_ids = []
    for idx, token_id in enumerate(token_ids):
        if hasattr(token_id, "item"):
            token_id = token_id.item()
        try:
            normalized_ids.append(int(token_id))
        except (TypeError, ValueError) as e:
            raise TypeError(f"token_id must be int-convertible, got {type(token_id).__name__}: {token_id!r}") from e
    return normalized_ids


def set_pad_token_id(tokenizer):
    """Set pad_token_id to eos_token_id if it is None.

    Args:
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to be set.

    """
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        warnings.warn(f"tokenizer.pad_token_id is None. Now set to {tokenizer.eos_token_id}", stacklevel=1)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        warnings.warn(f"tokenizer.pad_token is None. Now set to {tokenizer.eos_token}", stacklevel=1)


def hf_tokenizer(name_or_path, correct_pad_token=True, correct_gemma2=True, **kwargs):
    """Create a huggingface pretrained tokenizer which correctness handles eos and pad tokens.

    Args:

        name (str): The name of the tokenizer.
        correct_pad_token (bool): Whether to correct the pad token id.
        correct_gemma2 (bool): Whether to correct the gemma2 tokenizer.

    Returns:

        transformers.PreTrainedTokenizer: The pretrained tokenizer.

    """
    from transformers import AutoTokenizer

    if correct_gemma2 and isinstance(name_or_path, str) and "gemma-2-2b-it" in name_or_path:
        # the EOS token in gemma2 is ambiguious, which may worsen RL performance.
        # https://huggingface.co/google/gemma-2-2b-it/commit/17a01657f5c87135bcdd0ec7abb4b2dece04408a
        warnings.warn(
            "Found gemma-2-2b-it tokenizer. Set eos_token and eos_token_id to <end_of_turn> and 107.", stacklevel=1
        )
        kwargs["eos_token"] = "<end_of_turn>"
        kwargs["eos_token_id"] = 107
    tokenizer_kwargs = _maybe_enable_fix_mistral_regex(name_or_path, dict(kwargs))
    tokenizer = AutoTokenizer.from_pretrained(name_or_path, **tokenizer_kwargs)
    if correct_pad_token:
        set_pad_token_id(tokenizer)
    return tokenizer


def hf_processor(name_or_path, **kwargs):
    """Create a huggingface processor to process multimodal data.

    Args:
        name_or_path (str): The name of the processor.

    Returns:
        Optional[transformers.ProcessorMixin]: The pretrained multimodal processor.
        Returns ``None`` for text-only models (including AutoProcessor fallbacks to
        tokenizer backends such as ``TokenizersBackend``).
    """
    from transformers import AutoConfig, AutoProcessor, PreTrainedTokenizerBase

    config = None
    try:
        config = AutoConfig.from_pretrained(name_or_path, **kwargs)
        processor_kwargs = _maybe_enable_fix_mistral_regex(name_or_path, dict(kwargs))
        processor = AutoProcessor.from_pretrained(name_or_path, **processor_kwargs)
        # In newer transformers, AutoProcessor may legitimately fall back to a
        # tokenizer backend (e.g. TokenizersBackend) for text-only models.
        # Treat it as "no multimodal processor" and let callers use hf_tokenizer.
        if isinstance(processor, PreTrainedTokenizerBase):
            model_type = getattr(config, "model_type", None)
            if model_type in {"internvl_chat", "internvl"}:
                return _build_internvl_processor(name_or_path, config, kwargs)
            return None

        # Bind vlm model's get_rope_index method to processor
        processor.config = config
        rope_index_impl = None
        vision_position_impl = None
        match processor.__class__.__name__:
            case "Qwen2VLProcessor":
                from verl.models.transformers.qwen2_vl import get_rope_index as rope_index_impl
            case "Qwen2_5_VLProcessor":
                from verl.models.transformers.qwen2_vl import get_rope_index as rope_index_impl
            case "Qwen3VLProcessor":
                from verl.models.transformers.qwen3_vl import get_rope_index as rope_index_impl
            case "Glm4vImageProcessor":
                from verl.models.transformers.glm4v import get_rope_index as rope_index_impl
            case "MllamaProcessor":
                pass  # MllamaProcessor and MllamaModel doesn't have get_rope_index property
            case _:
                model_type = getattr(config, "model_type", None)
                if model_type in {"internvl_chat", "internvl"}:
                    return _build_internvl_processor(name_or_path, config, kwargs)
                raise ValueError(f"Unsupported processor type: {processor.__class__.__name__}")

        if rope_index_impl is not None:
            processor.get_rope_index = types.MethodType(rope_index_impl, processor)
        if vision_position_impl is not None:
            processor.get_vision_position_ids = types.MethodType(vision_position_impl, processor)
    except Exception as e:
        model_type = getattr(config, "model_type", None)
        if model_type in {"internvl_chat", "internvl"}:
            processor = _build_internvl_processor(name_or_path, config, kwargs)
            warnings.warn(
                f"AutoProcessor failed for InternVL ({e}); falling back to InternVLProcessor.",
                stacklevel=1,
            )
        else:
            processor = None
            # TODO(haibin.lin): try-catch should be removed after adding transformer version req to setup.py to avoid
            # silent failure
            warnings.warn(f"Failed to create processor: {e}. This may affect multimodal processing", stacklevel=1)
    # Avoid load tokenizer, see:
    # https://github.com/huggingface/transformers/blob/v4.49.0/src/transformers/models/auto/processing_auto.py#L344
    if processor is not None and "Processor" not in processor.__class__.__name__:
        processor = None
    return processor
