"""ImageNet-C corruption functions adapted for variable-size images.

Reference: Hendrycks & Dietterich, "Benchmarking Neural Network Robustness to
Common Corruptions and Perturbations", ICLR 2019.

Kept 8 corruption types safe for math/science images:
  gaussian_noise, shot_noise, impulse_noise, fog,
  brightness, contrast, elastic_transform, jpeg_compression

Severity: 1 (mild) to 5 (severe), mapped to T labels:
  severity 1 = T0.2, severity 2 = T0.4, ..., severity 5 = T1.0
"""

from __future__ import annotations

import warnings
from io import BytesIO

import cv2
import numpy as np
from PIL import Image as PILImage
from scipy.ndimage import gaussian_filter, map_coordinates

warnings.simplefilter("ignore", UserWarning)

SEVERITY_TO_T = {1: "T0.2", 2: "T0.4", 3: "T0.6", 4: "T0.8", 5: "T1.0"}
T_TO_SEVERITY = {value: key for key, value in SEVERITY_TO_T.items()}

CORRUPTION_NAMES = [
    "gaussian_noise",
    "shot_noise",
    "impulse_noise",
    "fog",
    "brightness",
    "contrast",
    "elastic_transform",
    "jpeg_compression",
]


def _ensure_rng(rng: np.random.Generator | None) -> np.random.Generator:
    return rng if rng is not None else np.random.default_rng()


def _next_power_of_2(n: int) -> int:
    power = 1
    while power < n:
        power *= 2
    return power


def plasma_fractal(mapsize: int = 256, wibbledecay: float = 3.0, rng: np.random.Generator | None = None):
    """Diamond-square heightmap, returns (mapsize, mapsize) in [0,1]."""
    assert (mapsize & (mapsize - 1)) == 0
    rng = _ensure_rng(rng)
    heightmap = np.empty((mapsize, mapsize), dtype=np.float64)
    heightmap[0, 0] = 0
    step = mapsize
    wibble = 100.0

    def wibbled_mean(area):
        return area / 4 + wibble * rng.uniform(-wibble, wibble, area.shape)

    def fill_squares():
        corner_refs = heightmap[0:mapsize:step, 0:mapsize:step]
        square_acc = corner_refs + np.roll(corner_refs, shift=-1, axis=0)
        square_acc += np.roll(square_acc, shift=-1, axis=1)
        heightmap[step // 2:mapsize:step, step // 2:mapsize:step] = wibbled_mean(square_acc)

    def fill_diamonds():
        diamond_refs = heightmap[step // 2:mapsize:step, step // 2:mapsize:step]
        upper_left = heightmap[0:mapsize:step, 0:mapsize:step]
        left_down = diamond_refs + np.roll(diamond_refs, 1, axis=0)
        left_up = upper_left + np.roll(upper_left, -1, axis=1)
        left_total = left_down + left_up
        heightmap[0:mapsize:step, step // 2:mapsize:step] = wibbled_mean(left_total)
        top_down = diamond_refs + np.roll(diamond_refs, 1, axis=1)
        top_up = upper_left + np.roll(upper_left, -1, axis=0)
        top_total = top_down + top_up
        heightmap[step // 2:mapsize:step, 0:mapsize:step] = wibbled_mean(top_total)

    while step >= 2:
        fill_squares()
        fill_diamonds()
        step //= 2
        wibble /= wibbledecay

    heightmap -= heightmap.min()
    max_value = heightmap.max()
    if max_value > 0:
        heightmap /= max_value
    return heightmap


def gaussian_noise(x, severity: int = 1, rng: np.random.Generator | None = None):
    rng = _ensure_rng(rng)
    scale = [0.08, 0.12, 0.18, 0.26, 0.38][severity - 1]
    array = np.array(x) / 255.0
    out = np.clip(array + rng.normal(size=array.shape, scale=scale), 0, 1) * 255
    return PILImage.fromarray(out.astype(np.uint8))


def shot_noise(x, severity: int = 1, rng: np.random.Generator | None = None):
    rng = _ensure_rng(rng)
    scale = [60, 25, 12, 5, 3][severity - 1]
    array = np.array(x) / 255.0
    out = np.clip(rng.poisson(array * scale) / float(scale), 0, 1) * 255
    return PILImage.fromarray(out.astype(np.uint8))


def impulse_noise(x, severity: int = 1, rng: np.random.Generator | None = None):
    rng = _ensure_rng(rng)
    scale = [0.03, 0.06, 0.09, 0.17, 0.27][severity - 1]
    array = np.array(x).copy()
    height, width, _ = array.shape
    n_pixels = int(height * width * scale)
    ys = rng.integers(0, height, n_pixels)
    xs = rng.integers(0, width, n_pixels)
    salt = rng.random(n_pixels) > 0.5
    array[ys[salt], xs[salt]] = 255
    array[ys[~salt], xs[~salt]] = 0
    return PILImage.fromarray(array)


def fog(x, severity: int = 1, rng: np.random.Generator | None = None):
    rng = _ensure_rng(rng)
    scale = [(1.5, 2), (2.0, 2), (2.5, 1.7), (2.5, 1.5), (3.0, 1.4)][severity - 1]
    array = np.array(x) / 255.0
    height, width = array.shape[:2]
    mapsize = max(_next_power_of_2(max(height, width)), 4)
    fractal = plasma_fractal(mapsize=mapsize, wibbledecay=scale[1], rng=rng)[:height, :width]
    array = array + scale[0] * fractal[..., np.newaxis]
    max_value = array.max()
    out = np.clip(array * max_value / (max_value + scale[0]), 0, 1) * 255
    return PILImage.fromarray(out.astype(np.uint8))


def brightness(x, severity: int = 1, rng: np.random.Generator | None = None):
    del rng
    scale = [0.1, 0.2, 0.3, 0.4, 0.5][severity - 1]
    array = np.array(x).astype(np.float32)
    hsv = cv2.cvtColor(array, cv2.COLOR_RGB2HSV)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] + scale * 255, 0, 255)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return PILImage.fromarray(rgb.astype(np.uint8))


def contrast(x, severity: int = 1, rng: np.random.Generator | None = None):
    del rng
    scale = [0.4, 0.3, 0.2, 0.1, 0.05][severity - 1]
    array = np.array(x) / 255.0
    means = np.mean(array, axis=(0, 1), keepdims=True)
    out = np.clip((array - means) * scale + means, 0, 1) * 255
    return PILImage.fromarray(out.astype(np.uint8))


def elastic_transform(x, severity: int = 1, rng: np.random.Generator | None = None):
    rng = _ensure_rng(rng)
    ref_size = 244.0
    image = np.array(x, dtype=np.float32) / 255.0
    shape = image.shape
    spatial_shape = shape[:2]
    scale = min(spatial_shape) / ref_size
    raw_config = [
        (ref_size * 2, ref_size * 0.7, ref_size * 0.1),
        (ref_size * 2, ref_size * 0.08, ref_size * 0.2),
        (ref_size * 0.05, ref_size * 0.01, ref_size * 0.02),
        (ref_size * 0.07, ref_size * 0.01, ref_size * 0.02),
        (ref_size * 0.12, ref_size * 0.01, ref_size * 0.02),
    ][severity - 1]
    config = (raw_config[0] * scale, raw_config[1] * scale, raw_config[2] * scale)

    center = np.float32(spatial_shape) // 2
    square_size = min(spatial_shape) // 3
    points_1 = np.float32([
        center + square_size,
        [center[0] + square_size, center[1] - square_size],
        center - square_size,
    ])
    points_2 = points_1 + rng.uniform(-config[2], config[2], size=points_1.shape).astype(np.float32)
    affine = cv2.getAffineTransform(points_1, points_2)
    image = cv2.warpAffine(image, affine, spatial_shape[::-1], borderMode=cv2.BORDER_REFLECT_101)

    sigma = max(config[1], 0.01)
    dx = gaussian_filter(rng.uniform(-1, 1, size=shape[:2]), sigma, mode="reflect", truncate=3) * config[0]
    dy = gaussian_filter(rng.uniform(-1, 1, size=shape[:2]), sigma, mode="reflect", truncate=3) * config[0]
    dx = dx.astype(np.float32)[..., np.newaxis]
    dy = dy.astype(np.float32)[..., np.newaxis]

    yy, xx, zz = np.meshgrid(
        np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing="ij"
    )
    indices = (
        np.reshape(yy + dy, (-1, 1)),
        np.reshape(xx + dx, (-1, 1)),
        np.reshape(zz, (-1, 1)),
    )
    out = np.clip(map_coordinates(image, indices, order=1, mode="reflect").reshape(shape), 0, 1) * 255
    return PILImage.fromarray(out.astype(np.uint8))


def jpeg_compression(x, severity: int = 1, rng: np.random.Generator | None = None):
    del rng
    quality = [25, 18, 15, 10, 7][severity - 1]
    buffer = BytesIO()
    x.save(buffer, "JPEG", quality=quality)
    buffer.seek(0)
    return PILImage.open(buffer).convert("RGB")


CORRUPTION_FUNCTIONS = {
    "gaussian_noise": gaussian_noise,
    "shot_noise": shot_noise,
    "impulse_noise": impulse_noise,
    "fog": fog,
    "brightness": brightness,
    "contrast": contrast,
    "elastic_transform": elastic_transform,
    "jpeg_compression": jpeg_compression,
}


def apply_corruption(image, corruption_name: str, severity: int = 1, rng: np.random.Generator | None = None):
    """Apply a named corruption at given severity to a PIL image."""
    if corruption_name not in CORRUPTION_FUNCTIONS:
        raise ValueError(f"Unknown corruption: {corruption_name}")
    if severity < 1 or severity > 5:
        raise ValueError(f"Severity must be 1-5, got {severity}")
    image = image.convert("RGB")
    return CORRUPTION_FUNCTIONS[corruption_name](image, severity, rng=rng)


def apply_random_corruption(image, severity: int = 1, rng: np.random.Generator | None = None):
    """Apply a randomly chosen corruption. Returns (image, name)."""
    rng = _ensure_rng(rng)
    name = rng.choice(CORRUPTION_NAMES)
    return apply_corruption(image, name, severity, rng=rng), name