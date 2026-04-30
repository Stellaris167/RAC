"""Build noise-augmented datasets from a clean parquet.

Modes: clean (pass-through), noisy (add noise), pair (clean+noisy pairs)

Usage:
  python -m verl.tools.build_noise_dataset --input_parquet .../train.parquet --mode pair --output .../train_pair.parquet
"""
from __future__ import annotations
import argparse
import io
import json
import os
import random
import numpy as np
import pandas as pd
from PIL import Image


def add_gaussian_noise(img: Image.Image, severity: float = 0.3) -> Image.Image:
    arr = np.array(img, dtype=np.float32)
    noise = np.random.normal(0, severity * 255, arr.shape)
    return Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))


def add_salt_pepper(img: Image.Image, severity: float = 0.05) -> Image.Image:
    arr = np.array(img)
    n_salt = int(severity * arr.size / 2)
    coords_salt = [np.random.randint(0, max(1, d), n_salt) for d in arr.shape]
    coords_pepper = [np.random.randint(0, max(1, d), n_salt) for d in arr.shape]
    arr[tuple(coords_salt)] = 255
    arr[tuple(coords_pepper)] = 0
    return Image.fromarray(arr)


def add_jpeg_compression(img: Image.Image, severity: float = 0.7) -> Image.Image:
    quality = max(1, int((1 - severity) * 95))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def add_blur(img: Image.Image, severity: float = 0.5) -> Image.Image:
    from PIL import ImageFilter
    return img.filter(ImageFilter.GaussianBlur(radius=max(1, int(severity * 10))))


NOISE_FNS = {
    "gaussian": add_gaussian_noise,
    "salt_pepper": add_salt_pepper,
    "jpeg": add_jpeg_compression,
    "blur": add_blur,
}


def load_image(uri: str) -> Image.Image:
    return Image.open(uri.replace("file://", "")).convert("RGB")


def save_image(img: Image.Image, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path, format="PNG")


def build_noisy(df, output_dir, noise_type="gaussian", severity=0.3):
    rows = []
    noisy_img_dir = os.path.join(output_dir, "images_noisy")
    fn = NOISE_FNS[noise_type]
    for idx, row in df.iterrows():
        new_row = dict(row)
        new_row.update({"is_noisy": True, "noise_type": noise_type, "noise_level": severity, "corruption_level": severity})
        image_uris = json.loads(row["image_uris"]) if isinstance(row["image_uris"], str) else row["image_uris"]
        new_uris = []
        for i, uri in enumerate(image_uris):
            try:
                img = load_image(uri)
                noisy_img = fn(img, severity)
                fname = f"{row['sample_id']}_{noise_type}_{i}.png"
                out_path = os.path.join(noisy_img_dir, fname)
                save_image(noisy_img, out_path)
                new_uris.append(f"file://{out_path}")
            except Exception as e:
                print(f"[warn] skip image {uri}: {e}")
                new_uris.append(uri)
        new_row["image_uris"] = json.dumps(new_uris) if isinstance(row["image_uris"], str) else new_uris
        msg = json.loads(row["message_qwenvl"]) if isinstance(row["message_qwenvl"], str) else row["message_qwenvl"]
        uri_idx = 0
        for m in msg:
            if isinstance(m.get("content"), list):
                for item in m["content"]:
                    if item.get("type") == "image" and uri_idx < len(new_uris):
                        item["image"] = new_uris[uri_idx]
                        uri_idx += 1
        new_row["message_qwenvl"] = json.dumps(msg) if isinstance(row["message_qwenvl"], str) else msg
        rows.append(new_row)
    return pd.DataFrame(rows)


def build_pair(df, output_dir, noise_type="gaussian", severity=0.3):
    noisy_df = build_noisy(df, output_dir, noise_type, severity)
    pair_rows = []
    for i, (_, row) in enumerate(df.iterrows()):
        clean = dict(row)
        clean.update({"is_noisy": False, "noise_type": "none", "noise_level": 0.0, "corruption_level": 0.0, "pair_id": i})
        noisy = dict(noisy_df.iloc[i])
        noisy["pair_id"] = i
        pair_rows.extend([clean, noisy])
    return pd.DataFrame(pair_rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_parquet", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=["clean", "noisy", "pair"], default="noisy")
    parser.add_argument("--noise_type", default="gaussian", choices=list(NOISE_FNS.keys()))
    parser.add_argument("--severity", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    df = pd.read_parquet(args.input_parquet)
    print(f"Loaded {len(df)} samples")
    output_dir = os.path.dirname(os.path.abspath(args.output))
    if args.mode == "clean":
        out_df = df
    elif args.mode == "noisy":
        out_df = build_noisy(df, output_dir, args.noise_type, args.severity)
    else:
        out_df = build_pair(df, output_dir, args.noise_type, args.severity)
    out_df.to_parquet(args.output, index=False)
    print(f"Wrote {len(out_df)} samples to {args.output}")


if __name__ == "__main__":
    main()
