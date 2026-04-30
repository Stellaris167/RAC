from __future__ import annotations

import math
import types
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


class InternVLProcessor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer"]

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
        self.image_processor = types.SimpleNamespace(patch_size=self.patch_size)
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

        tokenized = self.tokenizer(prompt, return_tensors=return_tensors or "pt")
        model_inputs = dict(tokenized)
        if pixel_values is not None:
            model_inputs["pixel_values"] = pixel_values
            model_inputs["image_flags"] = torch.ones((pixel_values.shape[0], 1), dtype=torch.long)
        return BatchFeature(data=model_inputs, tensor_type=return_tensors or "pt")