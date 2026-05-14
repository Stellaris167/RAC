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
# from . import gsm8k, math, prime_math, prime_code

import json
import math
import re

from verl.utils.import_utils import deprecated


_ANSWER_TAG_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
_PLAIN_ANSWER_RE = re.compile(r"(?:final\s+answer|answer|option|choice)\s*[:：=\-]?\s*(.+)", re.IGNORECASE | re.DOTALL)
_XML_TAG_RE = re.compile(r"</?[^>]+>")
_STRICT_OPTION_RE = re.compile(r"^\s*\(?\s*([A-Za-z])\s*[\.)]?\s*\)?\s*$")
_LEADING_OPTION_RE = re.compile(r"^\s*\(?\s*([A-Za-z])\b")
_OPTION_TOKEN_RE = re.compile(r"\b([A-Za-z])\b")
_DATA_SOURCE_ALIASES = {"we-math": "wemath", "we_math": "wemath"}
_BENCHMARK_DATA_SOURCES = {"geometry3k", "m3cot", "mathverse", "mathvision", "mmmu", "scienceqa", "wemath"}


def _normalize_reward_data_source(data_source) -> str:
    normalized = str(data_source).strip()
    if "@" in normalized:
        normalized = normalized.split("@", 1)[0].strip()
    return _DATA_SOURCE_ALIASES.get(normalized, normalized)


def _parse_choices(choices) -> dict[str, str]:
    if choices is None:
        return {}

    parsed = choices.tolist() if hasattr(choices, "tolist") else choices
    if isinstance(parsed, str):
        text = parsed.strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return {}

    if isinstance(parsed, dict):
        return {str(key).strip().upper(): str(value).strip() for key, value in parsed.items()}
    if isinstance(parsed, (list, tuple)):
        return {chr(ord("A") + index): str(value).strip() for index, value in enumerate(parsed)}
    return {}


def _normalize_answer_text(text) -> str:
    return " ".join(str(text).strip().lower().split())


def _freeform_answers_match(predicted, expected) -> bool:
    predicted_norm = _normalize_answer_text(predicted) if predicted is not None else ""
    expected_norm = _normalize_answer_text(expected) if expected is not None else ""
    if not predicted_norm or not expected_norm:
        return False
    if predicted_norm == expected_norm:
        return True

    try:
        return math.isclose(float(predicted_norm), float(expected_norm), rel_tol=1e-6, abs_tol=1e-8)
    except (TypeError, ValueError):
        return False


def _extract_answer_candidate(text) -> str | None:
    if text is None:
        return None

    raw_text = str(text)
    tagged = _ANSWER_TAG_RE.search(raw_text)
    if tagged is not None:
        return tagged.group(1).strip()

    cleaned = _XML_TAG_RE.sub(" ", raw_text)
    matches = list(_PLAIN_ANSWER_RE.finditer(cleaned))
    if matches:
        return matches[-1].group(1).strip()

    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    if lines:
        return lines[-1]
    cleaned = cleaned.strip()
    return cleaned or None


def _extract_option_letter(text) -> str | None:
    if text is None:
        return None

    cleaned = str(text).strip()
    strict = _STRICT_OPTION_RE.match(cleaned)
    if strict:
        return strict.group(1).upper()

    leading = _LEADING_OPTION_RE.match(cleaned)
    if leading:
        return leading.group(1).upper()

    if len(cleaned) <= 8:
        token = _OPTION_TOKEN_RE.search(cleaned)
        if token:
            return token.group(1).upper()
    return None


def _is_benchmark_data_source(data_source: str, extra_info=None) -> bool:
    extra_info = extra_info or {}
    base_data_source = extra_info.get("base_data_source")
    if base_data_source is not None and _normalize_reward_data_source(base_data_source) in _BENCHMARK_DATA_SOURCES:
        return True
    return data_source in _BENCHMARK_DATA_SOURCES


def _score_benchmark_data_source(solution_str, ground_truth, extra_info=None) -> float:
    extra_info = extra_info or {}
    choices = _parse_choices(extra_info.get("choices"))
    is_multiple_choice = bool(extra_info.get("is_multiple_choice", bool(choices)))

    predicted = _extract_answer_candidate(solution_str)
    expected = _extract_answer_candidate(ground_truth) or str(ground_truth).strip()

    if not is_multiple_choice:
        return 1.0 if _freeform_answers_match(predicted, expected) else 0.0

    predicted_label = _extract_option_letter(predicted)
    expected_label = _extract_option_letter(expected)
    if predicted_label is not None and expected_label is not None:
        return 1.0 if predicted_label == expected_label else 0.0

    predicted_norm = _normalize_answer_text(predicted) if predicted is not None else ""
    expected_norm = _normalize_answer_text(expected)
    if predicted_norm and predicted_norm == expected_norm:
        return 1.0

    if expected_label is not None and expected_label in choices:
        expected_choice = _normalize_answer_text(choices[expected_label])
        if predicted_norm and predicted_norm == expected_choice:
            return 1.0

    if predicted_label is not None and predicted_label in choices and expected_norm:
        predicted_choice = _normalize_answer_text(choices[predicted_label])
        if predicted_choice == expected_norm:
            return 1.0

    if _freeform_answers_match(predicted, expected):
        return 1.0
    return 0.0


def default_compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    sandbox_fusion_url=None,
    concurrent_semaphore=None,
    memory_limit_mb=None,
    **kwargs,
):
    """Compute the score for a given solution based on the data source.

    Args:
        data_source (str): The source dataset identifier which determines the scoring method.
        solution_str (str): The solution string to be evaluated.
        ground_truth (str): The ground truth answer for comparison.
        extra_info (dict, optional): Additional information that might be needed for scoring. Defaults to None.

    Returns:
        float: The computed score as a floating point number. If the result is a dictionary,
               it returns the dictionary instead.

    Raises:
        NotImplementedError: If the reward function is not implemented for the given data source.
    """
    data_source = _normalize_reward_data_source(data_source)

    if data_source == "openai/gsm8k":
        from . import gsm8k

        res = gsm8k.compute_score(solution_str, ground_truth)
    elif data_source in ["lighteval/MATH", "DigitalLearningGmbH/MATH-lighteval", "HuggingFaceH4/MATH-500"]:
        from . import math_reward

        res = math_reward.compute_score(solution_str, ground_truth)
    elif data_source in ["math_dapo", "math", "math_dapo_reasoning"] or data_source.startswith("aime"):
        from . import math_dapo

        res = math_dapo.compute_score(solution_str, ground_truth)
    elif _is_benchmark_data_source(data_source, extra_info=extra_info):
        res = _score_benchmark_data_source(solution_str, ground_truth, extra_info=extra_info)
    elif data_source in [
        "numina_aops_forum",
        "numina_synthetic_math",
        "numina_amc_aime",
        "numina_synthetic_amc",
        "numina_cn_k12",
        "numina_olympiads",
    ]:
        from . import prime_math

        res = prime_math.compute_score(solution_str, ground_truth)
    elif data_source in ["codecontests", "apps", "codeforces", "taco"]:
        if sandbox_fusion_url:
            from . import sandbox_fusion

            res = sandbox_fusion.compute_score(
                sandbox_fusion_url, concurrent_semaphore, memory_limit_mb, solution_str, ground_truth, continuous=True
            )
        else:
            from . import prime_code

            res = prime_code.compute_score(solution_str, ground_truth, continuous=True)
    elif data_source in ["hiyouga/geometry3k"]:
        from . import geo3k

        res = geo3k.compute_score(solution_str, ground_truth)
    elif data_source in [
        "searchR1_nq",
        "searchR1_triviaqa",
        "searchR1_popqa",
        "searchR1_hotpotqa",
        "searchR1_2wikimultihopqa",
        "searchR1_musique",
        "searchR1_bamboogle",
    ]:
        from . import search_r1_like_qa_em

        res = search_r1_like_qa_em.compute_score(solution_str, ground_truth)

    else:
        raise NotImplementedError(f"Reward function is not implemented for {data_source=}")

    if isinstance(res, dict):
        return res
    elif isinstance(res, int | float | bool):
        return float(res)
    else:
        return float(res[0])


def default_compute_score_image(
    data_source,
    solution_image,
    ground_truth,
    extra_info=None,
    sandbox_fusion_url=None,
    concurrent_semaphore=None,
    memory_limit_mb=None,
    **kwargs,
):
    """Compute the score for a given solution based on the data source.

    Args:
        data_source (str): The source dataset identifier which determines the scoring method.
        solution_image (Image.Image or torch.Tensor): The solution image to be evaluated.
        ground_truth (str): The ground truth answer for comparison.
        extra_info (dict, optional): Additional information that might be needed for scoring. Defaults to None.

    Returns:
        float: The computed score as a floating point number. If the result is a dictionary,
               it returns the dictionary instead.

    Raises:
        NotImplementedError: If the reward function is not implemented for the given data source.
    """
    if data_source == "jpeg_compressibility":
        from . import jpeg_compressibility

        res = jpeg_compressibility.compute_score(solution_image)

    else:
        raise NotImplementedError(f"Reward function is not implemented for {data_source=}")

    if isinstance(res, dict):
        return res
    elif isinstance(res, int | float | bool):
        return float(res)
    else:
        return float(res[0])


@deprecated("verl.utils.reward_score.default_compute_score")
def _default_compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    sandbox_fusion_url=None,
    concurrent_semaphore=None,
    memory_limit_mb=None,
):
    """
    Legacy function API to be deprecated. Please use `default_compute_score` instead.
    """
    return default_compute_score(
        data_source, solution_str, ground_truth, extra_info, sandbox_fusion_url, concurrent_semaphore, memory_limit_mb
    )


def get_default_compute_score(reward_name: str | None):
    """Get the default compute_score function based on the reward manager type."""
    if reward_name == "visual":
        return default_compute_score_image
    else:
        return default_compute_score


__all__ = ["default_compute_score"]
