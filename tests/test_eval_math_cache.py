from __future__ import annotations

import json
from pathlib import Path

import pytest

from prism_cli.eval_math import _safe_load


def _records() -> list[dict[str, object]]:
    records = []
    for index in range(4):
        prediction = index if index % 2 == 0 else index + 100
        records.append({
            'instruction': f'question-{index}',
            'input': f'context-{index}',
            'answer': index,
            'output_pred': f'The answer is {prediction}.',
            'pred': float(prediction),
            'flag': index % 2 == 0,
        })
    return records


def _write(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(json.dumps(records), encoding='utf-8')


def test_safe_load_accepts_only_complete_ordered_boolean_flag_cache(
    tmp_path: Path,
) -> None:
    path = tmp_path / 'predictions.json'
    expected = _records()
    _write(path, expected)

    accuracy, loaded = _safe_load(
        path,
        task='gsm8k',
        expected_records=expected,
    )

    assert accuracy == pytest.approx(0.5)
    assert loaded == expected


@pytest.mark.parametrize(
    'corruption',
    ['partial', 'wrong_order', 'flipped_bool', 'wrong_pred', 'wrong_output'],
)
def test_safe_load_rejects_incomplete_or_forged_cache(
    tmp_path: Path,
    corruption: str,
) -> None:
    path = tmp_path / 'predictions.json'
    expected = _records()
    observed = [dict(record) for record in expected]
    if corruption == 'partial':
        observed.pop()
    elif corruption == 'wrong_order':
        observed[0], observed[1] = observed[1], observed[0]
    elif corruption == 'flipped_bool':
        observed[0]['flag'] = not observed[0]['flag']
    elif corruption == 'wrong_pred':
        observed[0]['pred'] = 999.0
    elif corruption == 'wrong_output':
        observed[0]['output_pred'] = 'The answer is 999.'
    else:
        raise AssertionError(corruption)
    _write(path, observed)

    assert _safe_load(
        path,
        task='gsm8k',
        expected_records=expected,
    ) == (None, None)


def test_safe_load_uses_paper_aqua_letter_semantics(tmp_path: Path) -> None:
    path = tmp_path / 'AQuA.json'
    expected = [{
        'instruction': 'Choose one.',
        'input': '',
        'answer': 'C',
    }]
    observed = [{
        **expected[0],
        'output_pred': 'Rationale mentions Area. The answer is (C).',
        'pred': 'C',
        'flag': True,
    }]
    _write(path, observed)

    accuracy, loaded = _safe_load(
        path,
        task='AQuA',
        expected_records=expected,
    )

    assert accuracy == 1.0
    assert loaded == observed


def test_safe_load_requires_expected_cardinality(tmp_path: Path) -> None:
    path = tmp_path / 'predictions.json'
    _write(path, _records())
    with pytest.raises(ValueError, match='expected_rows'):
        _safe_load(path, task='gsm8k')
