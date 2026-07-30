"""Side-effect-free answer parsing shared by Math-10K validation and evaluation.

The numeric rule intentionally matches the repository's historical math
evaluator: remove thousands separators, take the last decimal-looking number,
and compare it to the reference with an absolute tolerance of 1e-3.
"""

from __future__ import annotations

import math
import re
from typing import Any, Optional


NUMERIC_EXACT_TOLERANCE = 1e-3
_NUMBER_RE = re.compile(r"-?\d+\.?\d*")


def extract_last_number(text: Any) -> Optional[float]:
    """Return the last finite decimal number in *text*, or ``None``."""
    matches = _NUMBER_RE.findall(str(text).replace(",", ""))
    if not matches:
        return None
    try:
        value = float(matches[-1])
    except (TypeError, ValueError, OverflowError):
        return None
    return value if math.isfinite(value) else None


def parse_reference_number(value: Any) -> Optional[float]:
    """Parse a Math-10K reference answer without accepting NaN or infinity."""
    try:
        parsed = float(str(value).replace(",", "").strip())
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if math.isfinite(parsed) else None


def numeric_exact_match(
    prediction_text: Any,
    reference: Any,
    *,
    tolerance: float = NUMERIC_EXACT_TOLERANCE,
) -> tuple[bool, Optional[float], Optional[float]]:
    """Return ``(correct, prediction, reference)`` under the legacy rule."""
    if not math.isfinite(float(tolerance)) or float(tolerance) < 0.0:
        raise ValueError("tolerance must be finite and non-negative")
    prediction = extract_last_number(prediction_text)
    target = parse_reference_number(reference)
    correct = (
        prediction is not None
        and target is not None
        and abs(prediction - target) <= float(tolerance)
    )
    return bool(correct), prediction, target


def response_from_decoded_text(text: Any) -> str:
    """Strip the prompt from a decoded prompt-plus-continuation sequence."""
    decoded = str(text)
    marker = "### Response:"
    if marker in decoded:
        return decoded.split(marker, 1)[1].strip()
    return decoded.strip()
