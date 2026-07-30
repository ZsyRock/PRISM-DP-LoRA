from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from prism_cli.validation_math import (
    PUBLIC_MATH10K_NUMERIC_EXACT_METRIC,
    evaluate_public_math_numeric_exact,
)


class _Tokenizer:
    pad_token_id = 0
    eos_token_id = 9
    padding_side = "left"

    def __call__(self, prompts, **kwargs):
        del kwargs
        rows = [[1, len(prompt) % 7 + 1] for prompt in prompts]
        return {
            "input_ids": torch.tensor(rows),
            "attention_mask": torch.ones((len(rows), 2), dtype=torch.long),
        }

    def batch_decode(self, rows, **kwargs):
        del kwargs
        mapping = {4: "reasoning; answer 4", 7: "unparsed"}
        return [mapping[int(row[-1])] for row in rows]


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(1.0))
        self.training_modes = []

    def generate(self, input_ids, attention_mask, **kwargs):
        del attention_mask
        assert kwargs["do_sample"] is False
        self.training_modes.append(self.training)
        suffixes = torch.tensor([[4], [7]], device=input_ids.device)
        return SimpleNamespace(sequences=torch.cat([input_ids, suffixes], dim=1))


def test_public_numeric_generation_is_deterministic_auditable_and_state_safe(
    tmp_path,
) -> None:
    model = _Model()
    model.train()
    tokenizer = _Tokenizer()
    torch.manual_seed(123)
    before = torch.get_rng_state().clone()
    predictions_path = tmp_path / "predictions.json"
    metrics = evaluate_public_math_numeric_exact(
        model,
        tokenizer,
        [
            {
                "_source_index": 11,
                "instruction": "first",
                "input": "",
                "output": "4",
                "answer": "4.0",
            },
            {
                "_source_index": 12,
                "instruction": "second",
                "input": "",
                "output": "8",
                "answer": "8.0",
            },
        ],
        device=torch.device("cpu"),
        batch_size=2,
        max_input_length=32,
        max_new_tokens=8,
        num_beams=1,
        needs_text_token_type_ids=False,
        predictions_path=predictions_path,
    )
    assert model.training is True
    assert model.training_modes == [False]
    assert torch.equal(torch.get_rng_state(), before)
    assert metrics["selection_metric"] == PUBLIC_MATH10K_NUMERIC_EXACT_METRIC
    assert metrics["records"] == 2
    assert metrics["numeric_exact_correct"] == 1
    assert metrics["numeric_exact_accuracy"] == 0.5
    assert metrics["numeric_parse_failures"] == 1
    assert metrics["predictions_file_sha256"]
    assert predictions_path.exists()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("batch_size", 0),
        ("max_input_length", 0),
        ("max_new_tokens", 0),
        ("num_beams", 0),
    ],
)
def test_numeric_generation_rejects_invalid_controls(field, value) -> None:
    kwargs = {
        "batch_size": 2,
        "max_input_length": 32,
        "max_new_tokens": 8,
        "num_beams": 1,
    }
    kwargs[field] = value
    with pytest.raises(ValueError, match=field):
        evaluate_public_math_numeric_exact(
            _Model(),
            _Tokenizer(),
            [{"instruction": "q", "input": "", "answer": "1"}],
            device=torch.device("cpu"),
            needs_text_token_type_ids=False,
            **kwargs,
        )
