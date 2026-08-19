from __future__ import annotations
import hashlib
import importlib.util
import json
import math
import re
import random
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple
import pandas as pd
from .evaluation_identity import prepare_evaluation_cache
from .trainers import RunConfig, completed_adapter_status
from .utils import llm_adapters_dir
MATH_TASKS = ['gsm8k', 'AQuA', 'mawps', 'SVAMP']
EVAL_SUBSET_SEED = 1729
_NUMERIC_ANSWER_RE = re.compile(r'-?\d+\.?\d*')
_AQUA_STANDALONE_ANSWER_RE = re.compile(r'(?:^|[^A-Z])([A-E])(?:$|[^A-Z])')
_AQUA_FALLBACK_ANSWER_RE = re.compile(r'[A-E]')

def _task_dir_name(task: str) -> str:
    low = task.lower()
    if low == 'gsm8k':
        return 'gsm8k'
    if low == 'aqua':
        return 'AQuA'
    if low == 'svamp':
        return 'SVAMP'
    if low == 'mawps':
        return 'mawps'
    return task

def _record_identity(record: Mapping[str, Any]) -> tuple[Any, Any, Any]:
    return (
        record.get('instruction'),
        record.get('input'),
        record.get('answer'),
    )


def _deterministic_subset(records: Sequence[Mapping[str, Any]], limit: int) -> list[Mapping[str, Any]]:
    if int(limit) <= 0 or int(limit) >= len(records):
        return list(records)
    indices = sorted(random.Random(EVAL_SUBSET_SEED).sample(range(len(records)), int(limit)))
    return [records[index] for index in indices]


def _prediction_from_output(task: str, output: str) -> Any:
    """Mirror the historical evaluator's stored-prediction parser."""
    if task.casefold() == 'aqua':
        upper = output.upper()
        matches = _AQUA_STANDALONE_ANSWER_RE.findall(upper)
        if matches:
            return matches[-1]
        fallback = _AQUA_FALLBACK_ANSWER_RE.findall(upper)
        return fallback[-1] if fallback else ''
    matches = _NUMERIC_ANSWER_RE.findall(output.replace(',', ''))
    if not matches:
        return float('inf')
    try:
        return float(matches[-1])
    except (TypeError, ValueError, OverflowError):
        return float('inf')


def _prediction_and_flag_match(task: str, record: Mapping[str, Any]) -> bool:
    """Verify cached ``pred`` and ``flag`` from the immutable generated text."""
    output = record.get('output_pred')
    if not isinstance(output, str) or 'pred' not in record:
        return False
    parsed = _prediction_from_output(task, output)
    stored = record.get('pred')
    if task.casefold() == 'aqua':
        if not isinstance(stored, str) or stored != parsed:
            return False
        expected_flag = str(record.get('answer', '')).strip().upper() == parsed
    else:
        try:
            stored_number = float(stored)
        except (TypeError, ValueError, OverflowError):
            return False
        if math.isnan(stored_number) or stored_number != parsed:
            return False
        try:
            reference = float(record.get('answer'))
        except (TypeError, ValueError, OverflowError):
            reference = float('inf')
        expected_flag = abs(reference - parsed) <= 0.001
    return record.get('flag') is bool(expected_flag)


def _safe_load(
    path: Path,
    *,
    task: str,
    expected_rows: Optional[int] = None,
    expected_records: Optional[Sequence[Mapping[str, Any]]] = None,
):
    """Load only a complete, ordered prediction cache for the locked test rows."""
    if task.casefold() not in {name.casefold() for name in MATH_TASKS}:
        raise ValueError(f'unsupported math evaluation task: {task!r}')
    if expected_records is not None:
        if expected_rows is not None and int(expected_rows) != len(expected_records):
            raise ValueError('expected_rows disagrees with expected_records')
        expected_rows = len(expected_records)
    if expected_rows is None or int(expected_rows) <= 0:
        raise ValueError('a positive expected_rows or non-empty expected_records is required')
    try:
        with path.open('r', encoding='utf-8') as handle:
            data = json.load(handle)
        if not isinstance(data, list) or len(data) != int(expected_rows):
            return (None, None)
        if any(not isinstance(record, dict) for record in data):
            return (None, None)
        flags = [record.get('flag') for record in data]
        if any(not isinstance(flag, bool) for flag in flags):
            return (None, None)
        if expected_records is not None and any(
            _record_identity(observed) != _record_identity(expected)
            for observed, expected in zip(data, expected_records)
        ):
            return (None, None)
        if any(not _prediction_and_flag_match(task, record) for record in data):
            return (None, None)
        acc = sum(int(flag) for flag in flags) / len(data)
        return (acc, data)
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        pass
    return (None, None)

def evaluate_math10k(cfg: RunConfig, batch_size: int=64, num_beams: int=4, max_new_tokens: int=256, max_input_length: int=1024, fast_dev_run: int=0) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cfg.finalize()
    adapter_dir = Path(cfg.output_dir)
    adapter_status = completed_adapter_status(cfg)
    if adapter_status is None:
        raise RuntimeError(f'Adapter is not complete: {adapter_dir}')
    evaluation_revision = (
        adapter_status.get('resolved_model_revision') or cfg.model_revision
    )
    adapters = llm_adapters_dir(cfg.root)
    eval_py = adapters / 'evaluate.py'
    spec = importlib.util.spec_from_file_location('llm_adapters_math_evaluate', eval_py)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Could not load {eval_py}')
    llm_eval = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(llm_eval)
    result_dir = Path(cfg.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    expected_records_by_task: Dict[str, list[Mapping[str, Any]]] = {}
    test_asset_identity: Dict[str, Dict[str, Any]] = {}
    for task in MATH_TASKS:
        ds_name = _task_dir_name(task)
        test_path = adapters / 'dataset' / ds_name / 'test.json'
        encoded = test_path.read_bytes()
        records = json.loads(encoded)
        if not isinstance(records, list) or not records or any(
            not isinstance(record, dict) for record in records
        ):
            raise RuntimeError(f'Invalid math evaluation asset: {test_path}')
        if int(fast_dev_run) > 0:
            records = _deterministic_subset(records, int(fast_dev_run))
        expected_records_by_task[task] = records
        test_asset_identity[ds_name] = {
            'rows': len(records),
            'sha256': hashlib.sha256(encoded).hexdigest(),
        }
    prepare_evaluation_cache(
        result_dir,
        {
            'config_fingerprint': cfg.config_fingerprint,
            'dataset': cfg.dataset,
            'base_model': cfg.base_model,
            'requested_model_revision': cfg.model_revision,
            'resolved_model_revision': evaluation_revision,
            'tasks': MATH_TASKS,
            'batch_size': int(batch_size),
            'num_beams': int(num_beams),
            'max_new_tokens': int(max_new_tokens),
            'max_input_length': int(max_input_length),
            'fast_dev_run': int(fast_dev_run),
            'eval_subset_seed': EVAL_SUBSET_SEED if int(fast_dev_run) > 0 else None,
            'test_assets': test_asset_identity,
        },
        artifact_names=[f'{_task_dir_name(task)}.json' for task in MATH_TASKS],
        force=bool(cfg.force_eval),
    )
    try:
        lora_weights = str(adapter_dir.relative_to(adapters))
    except ValueError:
        lora_weights = str(adapter_dir)
    results: Dict[str, float] = {}
    rows = []
    for task in MATH_TASKS:
        ds_name = _task_dir_name(task)
        unique_json = result_dir / f'{ds_name}.json'
        expected_records = expected_records_by_task[task]
        acc, data = _safe_load(
            unique_json,
            task=task,
            expected_records=expected_records,
        )
        if acc is not None:
            print(f'[eval cache] {task}: {unique_json}')
            results[ds_name] = float(acc)
            rows.append({'dataset': ds_name, 'accuracy': float(acc), 'n': len(data), 'json': str(unique_json)})
            continue
        # The upstream evaluator rewrites from row zero.  Remove a partial or
        # forged cache first so an interrupted run can never be mistaken for a
        # complete result by another reader.
        if unique_json.exists():
            unique_json.unlink()
        sys.argv = ['evaluate.py', '--dataset', task, '--model', 'other', '--adapter', 'LoRA', '--base_model', cfg.base_model, '--lora_weights', lora_weights, '--output_file', str(unique_json), '--batch_size', str(int(batch_size)), '--num_beams', str(int(num_beams)), '--max_new_tokens', str(int(max_new_tokens)), '--max_input_length', str(int(max_input_length)), '--log_every', '0']
        if int(fast_dev_run) > 0:
            sys.argv.extend([
                '--max_examples', str(int(fast_dev_run)),
                '--sample_seed', str(EVAL_SUBSET_SEED),
            ])
        if evaluation_revision:
            sys.argv.extend(['--model_revision', str(evaluation_revision)])
        print('[eval]', ' '.join(sys.argv))
        llm_eval.main()
        if not unique_json.exists():
            raise FileNotFoundError(f'Expected eval output not found: {unique_json}')
        acc, data = _safe_load(
            unique_json,
            task=task,
            expected_records=expected_records,
        )
        if acc is None:
            raise RuntimeError(f'Could not parse eval output: {unique_json}')
        results[ds_name] = float(acc)
        rows.append({'dataset': ds_name, 'accuracy': float(acc), 'n': len(data), 'json': str(unique_json)})
    df = pd.DataFrame([results])
    df['Average'] = df.mean(axis=1)
    detail_df = pd.DataFrame(rows)
    df.to_csv(result_dir / 'summary.csv', index=False)
    detail_df.to_csv(result_dir / 'details.csv', index=False)
    print(df)
    print('Saved:', result_dir / 'summary.csv')
    return (df, detail_df)
