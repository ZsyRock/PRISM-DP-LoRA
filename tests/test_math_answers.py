from __future__ import annotations

import pytest

from prism_cli.math_answers import (
    extract_last_number,
    numeric_exact_match,
    parse_reference_number,
    response_from_decoded_text,
)


def test_extract_last_number_matches_legacy_math_evaluator() -> None:
    assert extract_last_number("work: 12,345.5; final answer = -7.25") == -7.25
    assert extract_last_number("no numeric answer") is None
    assert extract_last_number("inf") is None


def test_numeric_exact_match_uses_original_absolute_tolerance() -> None:
    correct, prediction, reference = numeric_exact_match(
        "The answer is 1.0009",
        "1.0",
    )
    assert correct is True
    assert prediction == pytest.approx(1.0009)
    assert reference == 1.0
    assert numeric_exact_match("1.0011", 1.0)[0] is False
    assert numeric_exact_match("not parsed", 1.0) == (False, None, 1.0)


def test_reference_and_response_helpers_are_side_effect_free() -> None:
    assert parse_reference_number("1,234") == 1234.0
    assert parse_reference_number("NaN") is None
    assert response_from_decoded_text("prompt\n### Response:\n42") == "42"
    assert response_from_decoded_text("42") == "42"


def test_negative_or_nonfinite_tolerance_is_rejected() -> None:
    with pytest.raises(ValueError, match="finite and non-negative"):
        numeric_exact_match("1", "1", tolerance=-1)
