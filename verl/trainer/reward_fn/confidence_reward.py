"""Per-sample reward with answer + confidence extraction.

Extracts <answer>, <confidence>, validates <think> format.
Returns dict with rich per-sample metrics for case study analysis:
    score, acc, reward_acc, confidence, format_ok, is_multiple_choice,
    think_length, answer_text, confidence_raw, response_length
"""
import re
import math
import json

_RE_ANSWER = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)
_RE_CONFIDENCE = re.compile(r"<confidence>\s*(.*?)\s*</confidence>", re.DOTALL | re.IGNORECASE)
_RE_THINK = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
_RE_OPTION_LETTER = re.compile(r"^([A-Za-z])")
_RE_OPTION_TOKEN = re.compile(r"\b([A-Za-z])\b")
_RE_STRICT_OPTION_PAYLOAD = re.compile(r"^\s*\(?\s*([A-Za-z])\s*[\.)]?\s*\)?\s*$")
_RE_XML_TAG = re.compile(r"</?[^>]+>")
_RE_CONFIDENCE_RANGE = re.compile(
    r"([0-9]+(?:\.[0-9]+)?\s*%?)\s*(?:to|[-~–—]|and)\s*([0-9]+(?:\.[0-9]+)?\s*%?)",
    re.IGNORECASE,
)
_RE_PLAIN_CONFIDENCE = re.compile(
    r"(?:confidence|conf|certainty|probability)\s*[:：=\-]?\s*([0-9]+(?:\.[0-9]+)?\s*%?)",
    re.IGNORECASE,
)
_RE_PLAIN_ANSWER = re.compile(
    r"(?:final\s+answer|answer|final|option|choice)\s*[:：=\-]?\s*(.+)",
    re.IGNORECASE | re.DOTALL,
)


def _parse_float_like(value: str | None) -> float | None:
    if value is None:
        return None

    cleaned = str(value).strip().replace(",", "")
    if not cleaned:
        return None

    range_match = _RE_CONFIDENCE_RANGE.search(cleaned)
    if range_match:
        lower = _parse_float_like(range_match.group(1))
        upper = _parse_float_like(range_match.group(2))
        if lower is not None and upper is not None:
            return (lower + upper) / 2.0

    if cleaned.endswith("%"):
        cleaned = cleaned[:-1].strip()
        if not cleaned:
            return None
        try:
            return float(cleaned) / 100.0
        except ValueError:
            return None

    try:
        return float(cleaned)
    except ValueError:
        return None


def _extract_confidence_raw(text: str) -> float | None:
    m = _RE_CONFIDENCE.search(text)
    if m:
        return _parse_float_like(m.group(1))

    plain = _RE_PLAIN_CONFIDENCE.search(text)
    if plain:
        return _parse_float_like(plain.group(1))
    return None


def extract_answer_tag(text: str):
    m = _RE_ANSWER.search(text)
    return m.group(1).strip() if m else None


def _answer_payload_format_ok(answer_text: str | None, is_multiple_choice: bool) -> bool:
    if answer_text is None:
        return False
    answer_text = str(answer_text).strip()
    if not answer_text:
        return False
    if is_multiple_choice:
        return _RE_STRICT_OPTION_PAYLOAD.match(answer_text) is not None
    return True


def _strip_xml_sections(text: str) -> str:
    text = _RE_XML_TAG.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def _extract_plain_answer(text: str, is_multiple_choice: bool, choices=None) -> str | None:
    cleaned = _strip_xml_sections(text)
    if not cleaned:
        return None

    matched = _RE_PLAIN_ANSWER.search(cleaned)
    if matched:
        cleaned = matched.group(1).strip()

    if is_multiple_choice:
        option = _extract_option_letter(cleaned)
        if option:
            return option

        choice_map = _parse_choices(choices)
        cleaned_lower = cleaned.lower()
        matching_labels = [label for label, value in choice_map.items() if value and value.lower() in cleaned_lower]
        if len(matching_labels) == 1:
            return matching_labels[0]

    return cleaned or None


def extract_answer(text: str, is_multiple_choice: bool, choices=None) -> tuple[str | None, str]:
    tagged = extract_answer_tag(text)
    if tagged is not None:
        return tagged, "tag"

    plain = _extract_plain_answer(text, is_multiple_choice=is_multiple_choice, choices=choices)
    if plain is not None:
        return plain, "plain"
    return None, "missing"


def extract_confidence_tag(text: str) -> float:
    raw_conf = _extract_confidence_raw(text)
    if raw_conf is None:
        return -1.0

    v = raw_conf
    if v > 1.0:
        v /= 100.0
    return max(0.0, min(1.0, v))


def _find_last_match(pattern: re.Pattern, text: str):
    last = None
    for m in pattern.finditer(text):
        last = m
    return last


def _trailing_text_after_confidence(text: str) -> str:
    last_conf = _find_last_match(_RE_CONFIDENCE, text)
    if last_conf is None:
        return ""
    return text[last_conf.end():].strip()


def _schema_compliance_score(
    text: str,
    *,
    is_multiple_choice: bool = False,
    choices=None,
) -> float:
    """Return a soft schema score in [0, 1] for format-failed samples."""
    answer_m = _RE_ANSWER.search(text)
    conf_m = _RE_CONFIDENCE.search(text)
    think_m = _RE_THINK.search(text)

    has_think = 1.0 if think_m is not None else 0.0
    has_answer = 1.0 if answer_m is not None else 0.0
    has_conf = 1.0 if conf_m is not None else 0.0
    answer_payload_ok = 1.0 if _answer_payload_format_ok(extract_answer_tag(text), is_multiple_choice=is_multiple_choice) else 0.0
    conf_value_ok = 1.0 if _extract_confidence_raw(text) is not None else 0.0
    order_ok = 1.0 if (answer_m is not None and conf_m is not None and answer_m.start() <= conf_m.start()) else 0.0
    no_tail_after_conf = 1.0 if (conf_m is not None and _trailing_text_after_confidence(text) == "") else 0.0

    soft = (
        0.20 * has_think
        + 0.20 * has_answer
        + 0.20 * has_conf
        + 0.15 * answer_payload_ok
        + 0.10 * conf_value_ok
        + 0.10 * order_ok
        + 0.05 * no_tail_after_conf
    )
    return float(max(0.0, min(1.0, soft)))


def check_format(text: str, is_multiple_choice: bool = False, choices=None) -> bool:
    answer_m = _RE_ANSWER.search(text)
    conf_m = _RE_CONFIDENCE.search(text)
    if answer_m is None or conf_m is None:
        return False
    if _extract_confidence_raw(text) is None:
        return False
    if not _answer_payload_format_ok(extract_answer_tag(text), is_multiple_choice=is_multiple_choice):
        return False
    if answer_m.start() > conf_m.start():
        return False
    # Strict format: no non-whitespace content after the final </confidence>
    return _trailing_text_after_confidence(text) == ""


def _normalise(s: str) -> str:
    s = str(s).strip()
    lowered = s.lower()
    for pfx in ("answer:", "the answer is", "the correct answer is"):
        if lowered.startswith(pfx):
            s = s[len(pfx):].strip()
            lowered = s.lower()
    for pfx in ("option", "choice"):
        if lowered.startswith(pfx):
            s = s[len(pfx):].lstrip(" :.-)")
            lowered = s.lower()
    return s


def _extract_option_letter(text: str) -> str | None:
    cleaned = _normalise(text)
    match = _RE_OPTION_LETTER.match(cleaned.strip())
    if match:
        return match.group(1).upper()
    match = _RE_OPTION_TOKEN.search(cleaned)
    if match:
        return match.group(1).upper()
    return None


def _parse_choices(choices) -> dict[str, str]:
    if choices is None:
        return {}
    parsed = choices
    if isinstance(choices, str):
        text = choices.strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return {}
    if isinstance(parsed, dict):
        return {str(key).strip().upper(): str(value).strip() for key, value in parsed.items()}
    if isinstance(parsed, (list, tuple)):
        return {chr(ord("A") + idx): str(value).strip() for idx, value in enumerate(parsed)}
    return {}


def match_option(pred: str, gt: str, choices=None) -> bool:
    p = _extract_option_letter(pred)
    g = _extract_option_letter(gt)
    if p and g:
        return p == g

    pred_norm = _normalise(pred).strip().lower()
    gt_norm = _normalise(gt).strip().lower()
    if pred_norm == gt_norm:
        return True

    choice_map = _parse_choices(choices)
    if g and g in choice_map:
        return pred_norm == choice_map[g].strip().lower()
    return False


def _extract_reasoning_text(text: str) -> tuple[str, bool]:
    think_match = _RE_THINK.search(text)
    if think_match:
        return think_match.group(1).strip(), True

    answer_match = _RE_ANSWER.search(text)
    prefix = text[: answer_match.start()] if answer_match else text
    prefix = _RE_XML_TAG.sub(" ", prefix)
    prefix = re.sub(r"\s+", " ", prefix).strip()
    return prefix, False


def _resolve_format_coef(extra_info: dict, kwargs: dict) -> float:
    base_coef = kwargs.get("format_reward_coef", extra_info.get("format_reward_coef", 0.3))
    warmup_steps = int(kwargs.get("format_reward_warmup_steps", extra_info.get("format_reward_warmup_steps", 10)) or 0)
    warmup_start = float(kwargs.get("format_reward_warmup_start", extra_info.get("format_reward_warmup_start", 1.0)))
    global_step = kwargs.get("global_steps", extra_info.get("global_steps"))

    try:
        base_coef = float(base_coef)
    except (TypeError, ValueError):
        base_coef = 0.3

    if warmup_steps <= 0 or global_step is None:
        return base_coef

    try:
        progress = min(max(float(global_step), 0.0), float(warmup_steps)) / float(warmup_steps)
    except (TypeError, ValueError, ZeroDivisionError):
        return base_coef
    return warmup_start + (base_coef - warmup_start) * progress


def match_freeform(pred: str, gt: str) -> bool:
    pn, gn = _normalise(pred), _normalise(gt)
    if pn.lower() == gn.lower():
        return True
    try:
        pv = float(pn.replace(",", ""))
        gv = float(gn.replace(",", ""))
        return math.isclose(pv, gv, rel_tol=1e-6, abs_tol=1e-8)
    except (ValueError, TypeError):
        pass
    return False


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict | None = None,
    **kwargs,
) -> dict:
    extra_info = extra_info or {}
    is_mc = bool(extra_info.get("is_multiple_choice", True))
    choices = extra_info.get("choices")
    fmt_coef = _resolve_format_coef(extra_info, kwargs)
    tail_penalty_coef = float(kwargs.get("tail_text_penalty_coef", extra_info.get("tail_text_penalty_coef", 0.2)))
    schema_bonus_coef = float(kwargs.get("schema_bonus_coef", extra_info.get("schema_bonus_coef", 0.0)))
    schema_bonus_cap_ratio = float(kwargs.get("schema_bonus_cap_ratio", extra_info.get("schema_bonus_cap_ratio", 0.8)))
    pred, answer_source = extract_answer(solution_str, is_multiple_choice=is_mc, choices=choices)
    conf = extract_confidence_tag(solution_str)
    answer_tag_payload = extract_answer_tag(solution_str)
    answer_format_ok = 1.0 if _answer_payload_format_ok(answer_tag_payload, is_multiple_choice=is_mc) else 0.0
    fmt_ok = 1.0 if check_format(solution_str, is_multiple_choice=is_mc, choices=choices) else 0.0
    trailing_text = _trailing_text_after_confidence(solution_str)
    trailing_text_len = len(trailing_text)

    # --- think block analysis for case study ---
    think_text, has_explicit_think = _extract_reasoning_text(solution_str)
    think_length = len(think_text)  # char count of reasoning
    think_word_count = len(think_text.split()) if think_text else 0

    # --- confidence analysis ---
    confidence_raw = _extract_confidence_raw(solution_str)
    if confidence_raw is None:
        confidence_raw = -1.0

    acc = 0.0
    if pred is not None and ground_truth is not None:
        gt = str(ground_truth)
        acc = 1.0 if (match_option(pred, gt, choices) if is_mc else match_freeform(pred, gt)) else 0.0

    # Do not grant positive task reward unless the response satisfies the
    # strict answer/confidence schema. Keeping raw acc separately is useful for
    # diagnostics, but reward shaping should follow the formatted contract.
    reward_acc = acc if fmt_ok > 0.5 else 0.0

    # Format penalty: subtract fmt_coef when format requirements not met
    tail_penalty = 0.0
    if trailing_text_len > 0:
        tail_penalty = tail_penalty_coef * min(1.0, trailing_text_len / 128.0)

    schema_soft = 1.0 if fmt_ok > 0.5 else _schema_compliance_score(
        solution_str,
        is_multiple_choice=is_mc,
        choices=choices,
    )
    schema_bonus = 0.0
    if fmt_ok <= 0.5 and schema_bonus_coef > 0.0:
        schema_bonus = schema_bonus_coef * schema_soft
        cap_ratio = max(0.0, min(1.0, schema_bonus_cap_ratio))
        schema_bonus = min(schema_bonus, fmt_coef * cap_ratio)

    score = reward_acc - fmt_coef * (1.0 - fmt_ok) - tail_penalty + schema_bonus

    # --- calibration gap for this sample ---
    calib_gap = abs(conf - acc) if conf >= 0 else -1.0
    # overconfident: wrong but high conf; underconfident: correct but low conf
    is_overconfident = 1.0 if (acc < 0.5 and conf > 0.5) else 0.0
    is_underconfident = 1.0 if (acc > 0.5 and conf < 0.5) else 0.0

    # keep raw score for diagnostics, but provide a normalized [0,1] score
    raw_score = float(score)
    # allow callers to override normalization bounds via kwargs or extra_info
    score_min = kwargs.get("score_min", extra_info.get("score_min", -1.0))
    score_max = kwargs.get("score_max", extra_info.get("score_max", 1.0))
    try:
        score_min = float(score_min)
    except (TypeError, ValueError):
        score_min = -1.0
    try:
        score_max = float(score_max)
    except (TypeError, ValueError):
        score_max = 1.0

    if score_max <= score_min:
        score_norm = 0.5
    else:
        clamped = max(min(raw_score, score_max), score_min)
        score_norm = (clamped - score_min) / (score_max - score_min)
        score_norm = float(max(0.0, min(1.0, score_norm)))

    return {
        # ``score`` is normalized to [0,1] by default; raw value kept in ``score_raw``
        "score": score_norm,
        "score_raw": raw_score,
        "score_norm": score_norm,
        "acc": acc,
        "reward_acc": reward_acc,
        "confidence": conf,
        "format_ok": fmt_ok,
        "answer_format_ok": answer_format_ok,
        "format_penalty_coef": fmt_coef,
        "schema_soft": schema_soft,
        "schema_bonus": schema_bonus,
        "tail_text_penalty": tail_penalty,
        "tail_text_length": float(trailing_text_len),
        "is_multiple_choice": float(is_mc),
        # --- new per-sample metrics for case study ---
        "think_length": float(think_length),
        "think_word_count": float(think_word_count),
        "has_explicit_think": float(has_explicit_think),
        "response_length": float(len(solution_str)),
        "answer_text": pred or "",
        "answer_source": answer_source,
        "confidence_raw": confidence_raw,
        "calibration_gap": calib_gap,
        "is_overconfident": is_overconfident,
        "is_underconfident": is_underconfident,
    }
