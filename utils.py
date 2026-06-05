import re
import json
import logging
from typing import Optional, Union
from constants import NPK_LEVELS

logger = logging.getLogger(__name__)

def _normalize_text(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    cleaned = value.strip().lower()
    return cleaned if cleaned else None


def _normalize_npk(value: Optional[Union[str, float, int]]) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        lowered = value.strip().lower()
        aliases = {
            "low": "low",
            "l": "low",
            "medium": "medium",
            "med": "medium",
            "m": "medium",
            "high": "high",
            "h": "high",
        }
        if lowered in aliases:
            return aliases[lowered]
        try:
            numeric = float(lowered)
        except ValueError:
            return None
    else:
        numeric = float(value)

    if numeric < 40:
        return "low"
    if numeric < 80:
        return "medium"
    return "high"


def _score_numeric(value: float, lower: float, upper: float) -> float:
    if lower <= value <= upper:
        return 1.0

    spread = max(upper - lower, 1.0)
    if value < lower:
        distance = lower - value
    else:
        distance = value - upper

    penalty = distance / spread
    return max(0.0, 1.0 - penalty)


def _score_npk(actual: Optional[str], expected: str) -> float:
    if actual is None:
        return 0.4
    if actual == expected:
        return 1.0

    ai = NPK_LEVELS.index(actual)
    ei = NPK_LEVELS.index(expected)
    diff = abs(ai - ei)
    if diff == 1:
        return 0.65
    return 0.3


def _clean_response_text(text):
    # Some models may return internal reasoning inside <think> tags.
    while "<think>" in text and "</think>" in text:
        start = text.find("<think>")
        end = text.find("</think>", start)
        if end == -1:
            break
        text = text[:start] + text[end + len("</think>"):]
    return text.strip()


def _extract_generated_text(result):
    if isinstance(result, dict):
        choices = result.get("choices")
        if isinstance(choices, list) and len(choices) > 0:
            message = choices[0].get("message", {})
            return _clean_response_text((message.get("content") or ""))
    return ""


def _parse_json_response(text: str) -> Optional[dict]:
    if not text:
        return None
    # 1. Try to extract from a ```...``` block first
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)(?:```|$)", text)
    if fence_match:
        candidate = fence_match.group(1).strip()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
    # 2. Try the raw text as-is
    cleaned = text.strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    # 3. Scan for the first {...} block
    brace_match = re.search(r"\{[\s\S]*\}", cleaned)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass
    return None
