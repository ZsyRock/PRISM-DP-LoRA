"""Deterministic public Math-10K generation metrics for model selection."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import torch

from .experiment_identity import content_sha256
from .math_answers import (
    NUMERIC_EXACT_TOLERANCE,
    numeric_exact_match,
)
from .utils import (
    ensure_text_only_token_type_ids,
    generate_prompt,
    get_rng_state,
    set_rng_state,
    unwrap_for_save,
    write_json_atomic,
)


PUBLIC_MATH10K_NUMERIC_EXACT_METRIC = (
    "public_math10k_numeric_exact_match_accuracy"
)
PARSER_ID = "legacy_last_decimal_absolute_tolerance_v1"


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@torch.no_grad()
def evaluate_public_math_numeric_exact(
    model,
    tokenizer,
    records: Sequence[Mapping[str, Any]],
    *,
    device: torch.device,
    batch_size: int,
    max_input_length: int,
    max_new_tokens: int,
    num_beams: int,
    needs_text_token_type_ids: bool,
    predictions_path: Path | None = None,
) -> Dict[str, Any]:
    """Greedily decode a public holdout and compute legacy numeric exact match.

    This evaluator never samples, preserves caller RNG/model mode, and only
    receives the explicitly selected public validation records.  It does not
    know any task-test paths.
    """
    if not records:
        raise ValueError("public numeric validation requires at least one record")
    for name, value in {
        "batch_size": batch_size,
        "max_input_length": max_input_length,
        "max_new_tokens": max_new_tokens,
        "num_beams": num_beams,
    }.items():
        if int(value) <= 0:
            raise ValueError(f"{name} must be positive")

    evaluation_model = unwrap_for_save(model)
    was_training = bool(evaluation_model.training)
    rng_state = get_rng_state()
    evaluation_model.eval()
    predictions: list[dict[str, Any]] = []
    correct_count = 0
    parse_failures = 0
    try:
        for start in range(0, len(records), int(batch_size)):
            batch_records = records[start : start + int(batch_size)]
            prompts = [
                generate_prompt(
                    {
                        "instruction": str(record.get("instruction", "")),
                        "input": str(record.get("input", "")),
                        "output": "",
                    }
                )
                for record in batch_records
            ]
            tokenized = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=int(max_input_length),
            )
            inputs = {
                key: value.to(device)
                for key, value in tokenized.items()
                if isinstance(value, torch.Tensor)
            }
            inputs = ensure_text_only_token_type_ids(
                inputs,
                required=needs_text_token_type_ids,
            )
            if "input_ids" not in inputs:
                raise RuntimeError("validation tokenizer returned no input_ids")
            input_width = int(inputs["input_ids"].shape[1])
            generated = evaluation_model.generate(
                **inputs,
                do_sample=False,
                num_beams=int(num_beams),
                max_new_tokens=int(max_new_tokens),
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
            sequences = getattr(generated, "sequences", generated)
            if not isinstance(sequences, torch.Tensor) or sequences.ndim != 2:
                raise RuntimeError("model.generate returned no rank-2 token sequences")
            continuations = sequences[:, input_width:]
            decoded = tokenizer.batch_decode(
                continuations,
                skip_special_tokens=True,
            )
            if len(decoded) != len(batch_records):
                raise RuntimeError(
                    "numeric validation generation count does not match input count"
                )
            for offset, (record, output_text) in enumerate(
                zip(batch_records, decoded)
            ):
                if "answer" not in record:
                    raise ValueError(
                        "Math-10K public validation record is missing 'answer'"
                    )
                correct, prediction, reference = numeric_exact_match(
                    output_text,
                    record["answer"],
                )
                if reference is None:
                    raise ValueError(
                        "Math-10K public validation reference is not a finite number"
                    )
                correct_count += int(correct)
                parse_failures += int(prediction is None)
                source_index = record.get("_source_index")
                public_record = {
                    key: value
                    for key, value in record.items()
                    if key != "_source_index"
                }
                predictions.append(
                    {
                        "validation_position": start + offset,
                        "source_index": source_index,
                        "record_sha256": _canonical_sha256(public_record),
                        "output_pred": str(output_text),
                        "predicted_number": prediction,
                        "reference_number": reference,
                        "correct": bool(correct),
                        "parse_failure": prediction is None,
                    }
                )
    finally:
        if was_training:
            evaluation_model.train()
        set_rng_state(rng_state)

    predictions_sha = _canonical_sha256(predictions)
    if predictions_path is not None:
        write_json_atomic(Path(predictions_path), predictions)
        written_sha = content_sha256(Path(predictions_path))
    else:
        written_sha = None
    records_count = len(predictions)
    return {
        "selection_metric": PUBLIC_MATH10K_NUMERIC_EXACT_METRIC,
        "numeric_exact_accuracy": float(correct_count) / float(records_count),
        "numeric_exact_correct": int(correct_count),
        "numeric_parse_failures": int(parse_failures),
        "numeric_parse_failure_rate": float(parse_failures)
        / float(records_count),
        "records": int(records_count),
        "numeric_tolerance": float(NUMERIC_EXACT_TOLERANCE),
        "numeric_parser": PARSER_ID,
        "decoding": {
            "do_sample": False,
            "num_beams": int(num_beams),
            "max_new_tokens": int(max_new_tokens),
            "max_input_length": int(max_input_length),
            "padding_side": str(getattr(tokenizer, "padding_side", "unknown")),
        },
        "predictions_canonical_sha256": predictions_sha,
        "predictions_file_sha256": written_sha,
    }
