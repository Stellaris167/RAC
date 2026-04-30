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
T_TO_SEVERITY = {v: k for k, v in SEVERITY_TO_T.items()}

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


def _next_power_of_2(n):
    p = 1
    while p < n:
        p = p * 2
    return p


def plasma_fractal(mapsize=256, wibbledecay=3.0):
    """Diamond-square heightmap, returns (mapsize, mapsize) in [0,1]."""
    assert (mapsize & (mapsize - 1)) == 0
    m = np.empty((mapsize, mapsize), dtype=np.float64)
    m[0, 0] = 0
    step = mapsize
    wib = 100.0

    def wm(a):
        return a / 4 + wib * np.random.uniform(-wib, wib, a.shape)

    def fill_sq():
        cr = m[0:mapsize:step, 0:mapsize:step]
        sa = cr + np.roll(cr, shift=-1, axis=0)
        sa += np.roll(sa, shift=-1, axis=1)
        m[step // 2:mapsize:step, step // 2:mapsize:step] = wm(sa)

    def fill_dm():
        dr = m[step // 2:mapsize:step, step // 2:mapsize:step]
        ul = m[0:mapsize:step, 0:mapsize:step]
        ld = dr + np.roll(dr, 1, axis=0)
        lu = ul + np.roll(ul, -1, axis=1)
        lt = ld + lu
        m[0:mapsize:step, step // 2:mapsize:step] = wm(lt)
        td = dr + np.roll(dr, 1, axis=1)
        tu = ul + np.roll(ul, -1, axis=0)
        tt = td + tu
        m[step // 2:mapsize:step, 0:mapsize:step] = wm(tt)

    while step >= 2:
        fill_sq()
        fill_dm()
        step //= 2
        wib /= wibbledecay

    m -= m.min()
    mx = m.max()
    if mx > 0:
        m /= mx
    return m


# ---- corruption functions (PIL in, PIL out) ----

def gaussian_noise(x, severity=1):
    c = [0.08, 0.12, 0.18, 0.26, 0.38][severity - 1]
    a = np.array(x) / 255.0
    out = np.clip(a + np.random.normal(size=a.shape, scale=c), 0, 1) * 255
    return PILImage.fromarray(out.astype(np.uint8))


def shot_noise(x, severity=1):
    c = [60, 25, 12, 5, 3][severity - 1]
    a = np.array(x) / 255.0
    out = np.clip(np.random.poisson(a * c) / float(c), 0, 1) * 255
    return PILImage.fromarray(out.astype(np.uint8))


def impulse_noise(x, severity=1):
    c = [0.03, 0.06, 0.09, 0.17, 0.27][severity - 1]
    a = np.array(x).copy()
    h, w, ch = a.shape
    n_pixels = int(h * w * c)
    # salt
    ys = np.random.randint(0, h, n_pixels)
    xs = np.random.randint(0, w, n_pixels)
    salt = np.random.random(n_pixels) > 0.5
    a[ys[salt], xs[salt]] = 255
    a[ys[~salt], xs[~salt]] = 0
    return PILImage.fromarray(a)


def fog(x, severity=1):
    c = [(1.5, 2), (2.0, 2), (2.5, 1.7), (2.5, 1.5), (3.0, 1.4)][severity - 1]
    a = np.array(x) / 255.0
    h, w = a.shape[:2]
    ms = max(_next_power_of_2(max(h, w)), 4)
    fl = plasma_fractal(mapsize=ms, wibbledecay=c[1])[:h, :w]
    a = a + c[0] * fl[..., np.newaxis]
    mx = a.max()
    out = np.clip(a * mx / (mx + c[0]), 0, 1) * 255
    return PILImage.fromarray(out.astype(np.uint8))


def brightness(x, severity=1):
    c = [0.1, 0.2, 0.3, 0.4, 0.5][severity - 1]
    a = np.array(x).astype(np.float32)
    hsv = cv2.cvtColor(a, cv2.COLOR_RGB2HSV)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] + c * 255, 0, 255)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return PILImage.fromarray(rgb.astype(np.uint8))


def contrast(x, severity=1):
    c = [0.4, 0.3, 0.2, 0.1, 0.05][severity - 1]
    a = np.array(x) / 255.0
    means = np.mean(a, axis=(0, 1), keepdims=True)
    out = np.clip((a - means) * c + means, 0, 1) * 255
    return PILImage.fromarray(out.astype(np.uint8))


def elastic_transform(x, severity=1):
    REF = 244.0
    img = np.array(x, dtype=np.float32) / 255.0
    shape = img.shape
    ss = shape[:2]
    s = min(ss) / REF
    c_raw = [
        (REF * 2, REF * 0.7, REF * 0.1),
        (REF * 2, REF * 0.08, REF * 0.2),
        (REF * 0.05, REF * 0.01, REF * 0.02),
        (REF * 0.07, REF * 0.01, REF * 0.02),
        (REF * 0.12, REF * 0.01, REF * 0.02),
    ][severity - 1]
    c = (c_raw[0] * s, c_raw[1] * s, c_raw[2] * s)

    csq = np.float32(ss) // 2
    sqsz = min(ss) // 3
    pts1 = np.float32([
        csq + sqsz,
        [csq[0] + sqsz, csq[1] - sqsz],
        csq - sqsz,
    ])
    pts2 = pts1 + np.random.uniform(-c[2], c[2], size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    img = cv2.warpAffine(img, M, ss[::-1], borderMode=cv2.BORDER_REFLECT_101)

    sigma_val = max(c[1], 0.01)
    dx = (gaussian_filter(np.random.uniform(-1, 1, size=shape[:2]),
                          sigma_val, mode="reflect", truncate=3) * c[0]).astype(np.float32)
    dy = (gaussian_filter(np.random.uniform(-1, 1, size=shape[:2]),
                          sigma_val, mode="reflect", truncate=3) * c[0]).astype(np.float32)
    dx = dx[..., np.newaxis]
    dy = dy[..., np.newaxis]

    yy, xx, zz = np.meshgrid(
        np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing="ij"
    )
    indices = (
        np.reshape(yy + dy, (-1, 1)),
        np.reshape(xx + dx, (-1, 1)),
        np.reshape(zz, (-1, 1)),
    )
    out = np.clip(
        map_coordinates(img, indices, order=1, mode="reflect").reshape(shape), 0, 1
    ) * 255
    return PILImage.fromarray(out.astype(np.uint8))


def jpeg_compression(x, severity=1):
    c = [25, 18, 15, 10, 7][severity - 1]
    buf = BytesIO()
    x.save(buf, "JPEG", quality=c)
    buf.seek(0)
    return PILImage.open(buf).convert("RGB")


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


def apply_corruption(image, corruption_name, severity=1):
    """Apply a named corruption at given severity to a PIL image."""
    if corruption_name not in CORRUPTION_FUNCTIONS:
        raise ValueError(f"Unknown corruption: {corruption_name}")
    if severity < 1 or severity > 5:
        raise ValueError(f"Severity must be 1-5, got {severity}")
    image = image.convert("RGB")
    return CORRUPTION_FUNCTIONS[corruption_name](image, severity)


def apply_random_corruption(image, severity=1, rng=None):
    """Apply a randomly chosen corruption. Returns (image, name)."""
    if rng is None:
        rng = np.random.default_rng()
    name = rng.choice(CORRUPTION_NAMES)
    return apply_corruption(image, name, severity), name
