from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "llm_adapters_evaluate",
    ROOT / "LLM-Adapters" / "evaluate.py",
)
assert SPEC and SPEC.loader
evaluate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluate)


def test_math_evaluator_applies_deterministic_prefix_limit() -> None:
    args = SimpleNamespace(dataset="gsm8k", max_examples=3, sample_seed=1729)
    limited = evaluate.load_data(args)
    repeated = evaluate.load_data(args)
    full = evaluate.load_data(SimpleNamespace(dataset="gsm8k", max_examples=0, sample_seed=1729))
    assert len(limited) == 3
    assert limited == repeated
    assert all(record in full for record in limited)
    assert limited != full[:3]
