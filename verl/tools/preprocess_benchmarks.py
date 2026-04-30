"""Preprocess test benchmarks: filter MC answers, extract images, build messages.

Reads raw HF parquets from *_test/ directories, outputs standardized parquets
to *_test_processed/ directories with message_qwenvl, message_internvl, dataset_name,
answer, is_multiple_choice columns.

Usage:
  python -m verl.tools.preprocess_benchmarks --data_root .../SCS_data
"""
from __future__ import annotations
import argparse
import ast
import json
import os
import re
from io import BytesIO
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

LETTER_OPTIONS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
VALID_OPTION_LETTERS = set("ABCDEFGHI")

BENCHMARKS = {
    "m3cot": ("m3cot_test", "m3cot_test_processed"),
    "mathverse": ("mathverse_test", "mathverse_test_processed"),
    "mathvision": ("mathvision_test", "mathvision_test_processed"),
    "mmmu": ("mmmu_test", "mmmu_test_processed"),
    "scienceqa": ("scienceqa_test", "scienceqa_test_processed"),
    "wemath": ("we_math_test", "we_math_test_processed"),
}


def option_letter(index: int) -> str | None:
    return LETTER_OPTIONS[index] if 0 <= index < len(LETTER_OPTIONS) else None


def answer_to_option(answer) -> str | None:
    """Convert raw answer to a single option letter (A-I) or None.
    Accepts: single letter A-I, integer index 0-25, short strings like "(A)".
    Rejects: LaTeX, math symbols ($=+\\{}[]), digits, commas, semicolons, slashes.
    """
    if answer is None:
        return None
    if isinstance(answer, (int, float)):
        i = int(answer)
        if 0 <= i < 26:
            letter = chr(ord("A") + i)
            return letter if letter in VALID_OPTION_LETTERS else None
        return None
    s = str(answer).strip().upper()
    if not s:
        return None
    # Reject if contains digits, math symbols, or LaTeX
    if re.search(r"[0-9$=+\\{}\[\]]", s):
        return None
    if re.search(r"[,;/]", s):
        return None
    # Single letter
    if len(s) == 1 and s in VALID_OPTION_LETTERS:
        return s
    # Short strings like "(A)", "A.", "A)", "option A"
    if len(s) <= 10:
        letters = re.findall(r"\b([A-I])\b", s)
        if len(set(letters)) == 1:
            return letters[0]
    return None


def extract_image_bytes(value) -> bytes | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        return value
    if isinstance(value, dict) and isinstance(value.get("bytes"), (bytes, bytearray)):
        return bytes(value["bytes"])
    return None


def save_image(value, path: Path) -> str | None:
    raw = extract_image_bytes(value)
    if raw is None:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        Image.open(BytesIO(raw)).convert("RGB").save(path, format="PNG")
    return path.resolve().as_uri()


def format_choices(choices: list[str]) -> str:
    if not choices:
        return ""
    lines = ["Options:"]
    for i, c in enumerate(choices):
        lines.append(f"{option_letter(i)}. {c}")
    return "\n".join(lines)


def build_prompt(question: str, choices: list[str] | None = None,
                 context: str | None = None, hint: str | None = None) -> str:
    parts = []
    if hint and hint.strip():
        parts.append(f"Hint: {hint.strip()}")
    if context and context.strip():
        parts.append(f"Context: {context.strip()}")
    parts.append(f"Question: {question.strip()}")
    if choices:
        parts.append(format_choices(choices))
    return "\n".join(parts)


def build_message(prompt_text: str, image_uris: list[str], placeholders: bool = False) -> str:
    text = prompt_text
    if placeholders and image_uris:
        text = "".join("<image>\n" for _ in image_uris) + text
    content = [{"type": "image", "image": u} for u in image_uris]
    content.append({"type": "text", "text": text})
    return json.dumps([{"role": "user", "content": content}], ensure_ascii=False)


def make_record(dataset_name, sample_id, question, answer, image_uris,
                choices=None, context=None, hint=None):
    prompt = build_prompt(question, choices, context, hint)
    return {
        "dataset_name": dataset_name,
        "sample_id": sample_id,
        "question": question,
        "choices": choices or [],
        "answer": answer,
        "image_uris": image_uris,
        "is_multiple_choice": bool(choices),
        "message_qwenvl": build_message(prompt, image_uris, placeholders=False),
        "message_internvl": build_message(prompt, image_uris, placeholders=True),
    }


def parse_options_str(options_str: str) -> list[str]:
    if not options_str or options_str.strip() == "":
        return []
    # Try Python literal first (handles single-quoted lists from HF datasets)
    try:
        opts = ast.literal_eval(options_str)
        if isinstance(opts, (list, tuple)):
            return [str(o).strip() for o in opts if str(o).strip()]
    except (ValueError, SyntaxError):
        pass
    try:
        opts = json.loads(options_str)
        if isinstance(opts, list):
            return [str(o).strip() for o in opts if str(o).strip()]
    except (json.JSONDecodeError, TypeError):
        pass
    parts = re.split(r"\n|;\s*", options_str.strip())
    cleaned = [re.sub(r"^[A-Z][.)]\s*", "", p).strip() for p in parts if p.strip()]
    return cleaned if len(cleaned) > 1 else []


def process_m3cot(src_dir: Path, dst_dir: Path, img_root: Path):
    records = []
    for pf in sorted(src_dir.glob("**/*.parquet")):
        for i, row in enumerate(pq.read_table(pf).to_pylist()):
            choices = [str(c).strip() for c in (row.get("choices") or [])]
            ans = answer_to_option(row.get("answer"))
            if not ans:
                continue
            sid = str(row.get("id") or f"m3cot-{pf.stem}-{i}")
            uri = save_image(row.get("image"), img_root / "m3cot" / sid / "image.png")
            records.append(make_record("m3cot", sid, str(row.get("question", "")),
                                       ans, [uri] if uri else [], choices,
                                       context=str(row.get("context", "") or "")))
    _write(records, dst_dir)
    return len(records)


def _extract_mc_choices(text: str) -> list[str]:
    """Extract MC choices from MathVerse query text.

    Handles formats: '(A) ...' and 'A:...' (with Choices: header).
    """
    # Format: A:value or A: value (MathVerse style)
    pattern1 = re.findall(r"(?:^|\n)\s*([A-F])\s*:\s*(.+)", text)
    if len(pattern1) >= 2:
        return [v.strip() for _, v in pattern1]
    # Format: (A) value
    pattern2 = re.findall(r"\(([A-F])\)\s*([^(]+)", text)
    if len(pattern2) >= 2:
        return [v.strip().rstrip("\n") for _, v in pattern2]
    return []


def process_mathverse(src_dir: Path, dst_dir: Path, img_root: Path):
    records = []
    for pf in sorted(src_dir.glob("**/*.parquet")):
        # Skip text-only variants (no images)
        if "text_only" in pf.name:
            continue
        for i, row in enumerate(pq.read_table(pf).to_pylist()):
            if row.get("question_type") != "multi-choice":
                continue
            ans = answer_to_option(row.get("answer"))
            if not ans:
                continue
            sid = str(row.get("sample_index") or f"mathverse-{i}")
            uri = save_image(row.get("image"), img_root / "mathverse" / sid / "image.png")
            q = str(row.get("query_cot") or row.get("question", ""))
            choices = _extract_mc_choices(q)
            records.append(make_record("mathverse", sid, q, ans, [uri] if uri else [],
                                       choices=choices or None))
    _write(records, dst_dir)
    return len(records)


def process_scienceqa(src_dir: Path, dst_dir: Path, img_root: Path):
    records = []
    # Only use test split (not train/validation)
    test_files = sorted(src_dir.glob("**/test-*.parquet"))
    if not test_files:
        test_files = sorted(src_dir.glob("**/*.parquet"))
    for pf in test_files:
        for i, row in enumerate(pq.read_table(pf).to_pylist()):
            # Only keep questions with images (multimodal evaluation)
            if extract_image_bytes(row.get("image")) is None:
                continue
            choices = [str(c).strip() for c in (row.get("choices") or [])]
            if not choices:
                continue
            ans = answer_to_option(row.get("answer"))
            if not ans:
                continue
            sid = f"scienceqa-{pf.stem}-{i}"
            uri = save_image(row.get("image"), img_root / "scienceqa" / sid / "image.png")
            records.append(make_record("scienceqa", sid, str(row.get("question", "")),
                                       ans, [uri] if uri else [], choices,
                                       hint=str(row.get("hint", "") or "")))
    _write(records, dst_dir)
    return len(records)


def process_wemath(src_dir: Path, dst_dir: Path, img_root: Path):
    records = []
    for pf in sorted(src_dir.glob("**/*.parquet")):
        for i, row in enumerate(pq.read_table(pf).to_pylist()):
            opts_raw = row.get("option") or ""
            choices = parse_options_str(str(opts_raw)) if opts_raw else []
            ans = answer_to_option(row.get("answer"))
            if not ans:
                continue
            sid = str(row.get("ID") or f"wemath-{i}")
            uri = save_image(row.get("image_path"), img_root / "wemath" / sid / "image.png")
            records.append(make_record("wemath", sid, str(row.get("question", "")),
                                       ans, [uri] if uri else [], choices or None))
    _write(records, dst_dir)
    return len(records)


def process_mathvision(src_dir: Path, dst_dir: Path, img_root: Path):
    records = []
    # Only use test split (skip testmini) for consistent benchmark size
    test_files = sorted(src_dir.glob("**/test-*.parquet"))
    if not test_files:
        test_files = sorted(src_dir.glob("**/*.parquet"))
    for pf in test_files:
        for i, row in enumerate(pq.read_table(pf).to_pylist()):
            opts_raw = row.get("options") or []
            if isinstance(opts_raw, str):
                choices = parse_options_str(opts_raw)
            else:
                choices = [str(o).strip() for o in opts_raw if str(o).strip()]
            ans = answer_to_option(row.get("answer"))
            if not ans:
                continue
            sid = str(row.get("id") or f"mathvision-{i}")
            img_val = row.get("decoded_image") or row.get("image")
            uri = save_image(img_val, img_root / "mathvision" / sid / "image.png")
            records.append(make_record("mathvision", sid, str(row.get("question", "")),
                                       ans, [uri] if uri else [], choices or None))
    _write(records, dst_dir)
    return len(records)


def process_mmmu(src_dir: Path, dst_dir: Path, img_root: Path):
    records = []
    # MMMU has dev/validation/test splits per category.
    # Use validation split only (consistent with SCS benchmark).
    for cat_dir in sorted(src_dir.iterdir()):
        if not cat_dir.is_dir() or cat_dir.name.startswith("."):
            continue
        # Prefer validation parquets, fall back to test
        val_files = sorted(cat_dir.glob("validation-*.parquet"))
        test_files = sorted(cat_dir.glob("test-*.parquet"))
        parquet_files = val_files if val_files else test_files
        if not parquet_files:
            continue
        subject = cat_dir.name
        for pf in parquet_files:
            for i, row in enumerate(pq.read_table(pf).to_pylist()):
                opts_raw = row.get("options") or []
                if isinstance(opts_raw, str):
                    choices = parse_options_str(opts_raw)
                else:
                    choices = [str(o).strip() for o in opts_raw if str(o).strip()]
                ans = answer_to_option(row.get("answer"))
                if not ans:
                    continue
                sid = str(row.get("id") or f"mmmu-{subject}-{i}")
                uris = []
                for k in range(1, 8):
                    val = row.get(f"image_{k}")
                    u = save_image(val, img_root / "mmmu" / sid / f"image_{k}.png")
                    if u:
                        uris.append(u)
                q = str(row.get("question", ""))
                q = re.sub(r"<image\s*\d*>", "", q).strip()
                records.append(make_record("mmmu", sid, q, ans, uris, choices or None))
    _write(records, dst_dir)
    return len(records)


def _write(records: list[dict], dst_dir: Path):
    dst_dir.mkdir(parents=True, exist_ok=True)
    if not records:
        print(f"  WARNING: 0 records for {dst_dir}")
        return
    pq.write_table(pa.Table.from_pylist(records), dst_dir / "test.parquet")


PROCESSORS = {
    "m3cot": process_m3cot,
    "mathverse": process_mathverse,
    "mathvision": process_mathvision,
    "mmmu": process_mmmu,
    "scienceqa": process_scienceqa,
    "wemath": process_wemath,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root",
                        default="/share/home/cuipeng/cuipeng_a100/yangboyao/datasets/SCS_data")
    parser.add_argument("--benchmarks", nargs="+", default=list(BENCHMARKS.keys()))
    parser.add_argument("--image_root", default=None,
                        help="Directory for extracted images. Defaults to data_root/test_images")
    args = parser.parse_args()
    data_root = Path(args.data_root)
    img_root = Path(args.image_root) if args.image_root else data_root / "test_images"
    for bm in args.benchmarks:
        if bm not in BENCHMARKS:
            print(f"[skip] unknown benchmark: {bm}")
            continue
        src_name, dst_name = BENCHMARKS[bm]
        src_dir = data_root / src_name
        dst_dir = data_root / dst_name
        if not src_dir.is_dir():
            print(f"[skip] {src_dir} not found")
            continue
        print(f"Processing {bm}: {src_dir} -> {dst_dir}")
        n = PROCESSORS[bm](src_dir, dst_dir, img_root)
        print(f"  -> {n} records")
    print("Done.")


if __name__ == "__main__":
    main()
