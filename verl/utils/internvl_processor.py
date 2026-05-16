from __future__ import annotations

import json
import math
import os
import re
from collections.abc import Sequence

import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import BatchFeature
from transformers import ProcessorMixin


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMG_START_TOKEN = "<img>"
IMG_END_TOKEN = "</img>"
IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
IMAGE_PLACEHOLDER = "<image>"
_INTERNVL_IMAGE_TOKEN_BLOCK_RE = re.compile(
    rf"{re.escape(IMG_START_TOKEN)}(?:{re.escape(IMG_CONTEXT_TOKEN)})+{re.escape(IMG_END_TOKEN)}"
)


class _InternVLImageProcessorProxy:
    def __init__(self, patch_size: int):
        self.patch_size = patch_size


def bind_internvl_img_context_token_id(model, tokenizer=None, processor=None):
    token_id = getattr(processor, "image_context_token_id", None)
    if token_id is None and tokenizer is not None:
        token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    if token_id is None or token_id < 0:
        raise ValueError("Failed to resolve InternVL <IMG_CONTEXT> token id from tokenizer/processor.")

    pending = [model]
    visited = set()
    while pending:
        current = pending.pop()
        if current is None or id(current) in visited:
            continue
        visited.add(id(current))

        model_type = getattr(getattr(current, "config", None), "model_type", None)
        if model_type in {"internvl_chat", "internvl"}:
            current.img_context_token_id = int(token_id)

        for attr in ("model", "module", "base_model", "language_model", "pretrained_model"):
            pending.append(getattr(current, attr, None))


def _build_transform(input_size: int):
    return T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def _find_closest_aspect_ratio(
    aspect_ratio: float,
    target_ratios: list[tuple[int, int]],
    width: int,
    height: int,
    image_size: int,
) -> tuple[int, int]:
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(
    image: Image.Image,
    min_num: int = 1,
    max_num: int = 12,
    image_size: int = 448,
    use_thumbnail: bool = False,
) -> list[Image.Image]:
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if min_num <= i * j <= max_num
    )
    target_ratios = sorted(target_ratios, key=lambda ratio: ratio[0] * ratio[1])
    target_aspect_ratio = _find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for block_idx in range(blocks):
        box = (
            (block_idx % (target_width // image_size)) * image_size,
            (block_idx // (target_width // image_size)) * image_size,
            ((block_idx % (target_width // image_size)) + 1) * image_size,
            ((block_idx // (target_width // image_size)) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))

    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images


def expand_internvl_image_tokens(prompt: str, num_patches_list: Sequence[int], num_image_token: int) -> str:
    expanded_prompt = prompt
    for num_patches in num_patches_list:
        image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * (num_image_token * int(num_patches)) + IMG_END_TOKEN
        if IMAGE_PLACEHOLDER not in expanded_prompt:
            expanded_prompt = f"{IMAGE_PLACEHOLDER}\n{expanded_prompt}"
        expanded_prompt = expanded_prompt.replace(IMAGE_PLACEHOLDER, image_tokens, 1)
    return expanded_prompt


def collapse_internvl_image_tokens(prompt: str) -> str:
    return _INTERNVL_IMAGE_TOKEN_BLOCK_RE.sub(IMAGE_PLACEHOLDER, prompt)


def build_internvl_vllm_prompt_text(tokenizer, prompt_ids: Sequence[int]) -> str:
    decoded_prompt = tokenizer.decode(
        prompt_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    return collapse_internvl_image_tokens(decoded_prompt)


class InternVLProcessor(ProcessorMixin):
    attributes = ["tokenizer"]

    def __init__(self, tokenizer, config):
        self.tokenizer = tokenizer
        self.config = config

        image_size = getattr(config, "force_image_size", None)
        if image_size is None:
            image_size = getattr(config.vision_config, "image_size", 448)
        self.image_size = int(image_size)
        self.patch_size = int(getattr(config.vision_config, "patch_size", 14))
        self.downsample_ratio = float(getattr(config, "downsample_ratio", 0.5))
        self.num_image_token = int((self.image_size // self.patch_size) ** 2 * (self.downsample_ratio**2))
        self.min_dynamic_patch = int(getattr(config, "min_dynamic_patch", 1))
        self.max_dynamic_patch = int(getattr(config, "max_dynamic_patch", 6))
        self.use_thumbnail = bool(getattr(config, "use_thumbnail", False))
        self.chat_template = getattr(tokenizer, "chat_template", None)
        self.audio_tokenizer = None
        self.image_processor = _InternVLImageProcessorProxy(patch_size=self.patch_size)
        self._transform = _build_transform(self.image_size)
        self.image_token_id = self.tokenizer.convert_tokens_to_ids(IMG_START_TOKEN)
        self.image_end_token_id = self.tokenizer.convert_tokens_to_ids(IMG_END_TOKEN)
        self.image_context_token_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.video_token_id = None
        self.vision_start_token_id = None
        self.visual_token_ids = {
            token_id
            for token_id in (self.image_token_id, self.image_context_token_id, self.image_end_token_id)
            if token_id is not None and token_id >= 0
        }

    def to_dict(self, *args, **kwargs) -> dict:
        return {
            "processor_class": self.__class__.__name__,
            "image_processor_type": "InternVLImageProcessorProxy",
            "patch_size": self.patch_size,
            "size": {"height": self.image_size, "width": self.image_size},
            "min_dynamic_patch": self.min_dynamic_patch,
            "max_dynamic_patch": self.max_dynamic_patch,
            "use_thumbnail": self.use_thumbnail,
            "downsample_ratio": self.downsample_ratio,
            "image_mean": list(IMAGENET_MEAN),
            "image_std": list(IMAGENET_STD),
        }

    def save_pretrained(self, save_directory, push_to_hub: bool = False, **kwargs):
        if push_to_hub:
            raise NotImplementedError("InternVLProcessor.save_pretrained does not support push_to_hub.")

        os.makedirs(save_directory, exist_ok=True)
        save_jinja_files = kwargs.get("save_jinja_files", True)

        if hasattr(self.tokenizer, "_set_processor_class"):
            self.tokenizer._set_processor_class(self.__class__.__name__)
        self.tokenizer.save_pretrained(save_directory, save_jinja_files=save_jinja_files)

        processor_config_path = os.path.join(save_directory, "processor_config.json")
        with open(processor_config_path, "w", encoding="utf-8") as writer:
            json.dump(self.to_dict(), writer, ensure_ascii=False, indent=2, sort_keys=True)
            writer.write("\n")

        if self.chat_template is not None and save_jinja_files and isinstance(self.chat_template, str):
            chat_template_path = os.path.join(save_directory, "chat_template.jinja")
            with open(chat_template_path, "w", encoding="utf-8") as writer:
                writer.write(self.chat_template)

        return [processor_config_path]

    @property
    def model_input_names(self) -> list[str]:
        return ["input_ids", "attention_mask", "pixel_values", "image_flags"]

    def apply_chat_template(self, *args, **kwargs):
        return self.tokenizer.apply_chat_template(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def convert_tokens_to_ids(self, *args, **kwargs):
        return self.tokenizer.convert_tokens_to_ids(*args, **kwargs)

    def _preprocess_images(self, images: Sequence[Image.Image] | None) -> tuple[torch.Tensor | None, list[int]]:
        if not images:
            return None, []

        tiled_tensors: list[torch.Tensor] = []
        num_patches_list: list[int] = []
        for image in images:
            processed_images = dynamic_preprocess(
                image,
                min_num=self.min_dynamic_patch,
                max_num=self.max_dynamic_patch,
                image_size=self.image_size,
                use_thumbnail=self.use_thumbnail,
            )
            num_patches_list.append(len(processed_images))
            tiled_tensors.extend(self._transform(tile) for tile in processed_images)

        return torch.stack(tiled_tensors), num_patches_list

    def __call__(
        self,
        text,
        images=None,
        videos=None,
        return_tensors: str | None = None,
        **kwargs,
    ):
        if videos:
            raise NotImplementedError("InternVLProcessor does not support videos in this RL pipeline yet.")

        if isinstance(text, str):
            texts = [text]
        else:
            texts = list(text)

        if len(texts) != 1:
            raise ValueError(f"InternVLProcessor expects a single prompt per call, got {len(texts)}")

        pixel_values, num_patches_list = self._preprocess_images(images)
        prompt = texts[0]
        if num_patches_list:
            prompt = expand_internvl_image_tokens(prompt, num_patches_list, self.num_image_token)

        # Agent-loop prompt budgeting truncates InternVL prompts after tokenization
        # with visual-run-aware logic. Avoid the tokenizer's premature length warning
        # for over-budget expanded prompts here so we do not emit a misleading
        # "will result in indexing errors" message before that truncation happens.
        tokenized = self.tokenizer(prompt, return_tensors=return_tensors or "pt", verbose=False)
        model_inputs = dict(tokenized)
        if pixel_values is not None:
            model_inputs["pixel_values"] = pixel_values
            model_inputs["image_flags"] = torch.ones((pixel_values.shape[0], 1), dtype=torch.long)
        return BatchFeature(data=model_inputs, tensor_type=return_tensors or "pt")