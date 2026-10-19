"""Brace-aware boxed fields and numerical confidence spans in the original text."""

import re
from dataclasses import dataclass


@dataclass
class Box:
    start: int
    end: int
    payload_start: int
    payload: str
    closed: bool = True


def last_box(text, allow_unclosed=False):
    box = None
    for match in re.finditer(r"\\?boxed\s*\{", text, re.IGNORECASE):
        depth = 1
        for index in range(match.end(), len(text)):
            if text[index] in "{}" and (index == 0 or text[index - 1] != "\\"):
                depth += 1 if text[index] == "{" else -1
            if depth == 0:
                box = Box(match.start(), index + 1, match.end(), text[match.end():index])
                break
        else:
            if allow_unclosed:
                box = Box(match.start(), len(text), match.end(), text[match.end():], closed=False)
    return box


def _plain_with_offsets(text, offset):
    # Remove presentation commands, retaining a mapping back to source characters.
    ignored = set()
    for match in re.finditer(r"\\[a-zA-Z]+\s*|\\[,;! ]|[{}]", text):
        ignored.update(range(match.start(), match.end()))
    indices = [i for i in range(len(text)) if i not in ignored]
    return "".join(text[i] for i in indices), [offset + i for i in indices]


def box_answer(box):
    depth = 0
    end = len(box.payload)
    for i, char in enumerate(box.payload):
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
        elif char == ";" and depth == 0:
            end = i
            break
    answer = box.payload[:end]
    # Also support the existing comma-separated/plain field spelling.
    answer = re.split(r",?\s*\b(?:confidence|conf|certainty|probability)\s*[:：=]", answer, maxsplit=1, flags=re.I)[0]
    answer = re.sub(r"^\s*(?:answer|ans|choice|option)\s*[:：=]\s*", "", answer, flags=re.I)
    answer = answer.strip().strip(";,")
    if re.fullmatch(r"(?:answer|ans|choice|option)\s*[:=]?", answer, re.I):
        return None
    return answer or None


def _number(text, offset):
    # Repeating decimals and symbolic expressions require semantic interpretation.
    if re.search(r"\\(?:overline|frac|sqrt)\b", text):
        return None
    plain, offsets = _plain_with_offsets(text, offset)
    match = re.fullmatch(r"\s*((?:\d+(?:\s*\.\s*\d+)?|\.\s*\d+)\s*%?)\s*", plain)
    if not match:
        return None
    start, end = match.span(1)
    while plain[end - 1].isspace():
        end -= 1
    value = re.sub(r"\s+", "", plain[start:end])
    number = float(value.rstrip("%")) / (100 if value.endswith("%") else 1)
    return number, (offsets[start], offsets[end - 1] + 1)


def confidence_value_span(text):
    box = last_box(text, allow_unclosed=True)
    if box is not None:
        plain, offsets = _plain_with_offsets(box.payload, box.payload_start)
        field = re.search(r"\b(?:confidence|conf|certainty|probability)\s*[:：=]\s*", plain, re.I)
        if field is None or field.end() == len(offsets):
            return None
        start = offsets[field.end()]
        return _number(text[start:box.end - int(box.closed)], start)
    tagged = re.search(r"</answer>\s*<confidence>(.*?)</confidence>", text, re.I | re.S)
    if tagged:
        return _number(tagged[1], tagged.start(1))
    return None
