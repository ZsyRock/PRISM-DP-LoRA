from __future__ import annotations
import json
import hashlib
import re
from pathlib import Path
from typing import Dict, Tuple
import pandas as pd
try:
    import torch
except Exception:
    torch = None
from .trainers import RunConfig, completed_adapter_status
from .evaluation_identity import prepare_evaluation_cache
from .modeling import load_base_model
from .utils import build_prompt, build_tokenizer, cleanup_cuda, write_json_atomic
GLUE_TASKS = ['cola', 'sst2', 'mrpc', 'stsb', 'qqp', 'mnli', 'qnli', 'rte']
GLUE_ASSET_SCHEMA_VERSION = 1
EVAL_SUBSET_SEED = 1729


def _load_glue_asset_manifest(root: Path) -> dict:
    manifest_path = root / 'manifest.json'
    try:
        raw = manifest_path.read_bytes()
        manifest = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f'GLUE evaluation assets are unavailable or invalid: {manifest_path}') from exc
    if manifest.get('schema_version') != GLUE_ASSET_SCHEMA_VERSION:
        raise RuntimeError(f'Unsupported GLUE asset schema in {manifest_path}')
    if manifest.get('tasks') != GLUE_TASKS:
        raise RuntimeError(f'GLUE asset task order does not match evaluator: {manifest_path}')
    manifest['manifest_sha256'] = hashlib.sha256(raw).hexdigest()
    return manifest


def _load_glue_dataset(task: str, data_root: Path | None):
    if data_root is None:
        from datasets import load_dataset
        return load_dataset('nyu-mll/glue', task)
    from datasets import load_from_disk
    task_root = data_root / task
    if not task_root.is_dir():
        raise RuntimeError(f'Missing materialized GLUE task: {task_root}')
    return load_from_disk(str(task_root))


def _compute_metric(task: str, predictions, references) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
    if task == 'stsb':
        from scipy.stats import pearsonr, spearmanr
        pearson = float(pearsonr(predictions, references).statistic)
        spearman = float(spearmanr(predictions, references).statistic)
        return {'pearson': pearson, 'spearmanr': spearman}
    predictions = [int(value) for value in predictions]
    references = [int(value) for value in references]
    if task == 'cola':
        return {'matthews_correlation': float(matthews_corrcoef(references, predictions))}
    result = {'accuracy': float(accuracy_score(references, predictions))}
    if task in ('mrpc', 'qqp'):
        result['f1'] = float(f1_score(references, predictions, zero_division=0))
    return result

def glue_to_instruction_input(task: str, ex: dict):
    task = task.lower()
    if task == 'cola':
        return ('Task: CoLA (linguistic acceptability). Determine whether the sentence is grammatically acceptable. Answer with exactly one label: acceptable, unacceptable.', f"Sentence: {ex['sentence']}")
    if task == 'sst2':
        return ('Task: SST-2 (sentiment). Determine the sentiment of the sentence. Answer with exactly one label: positive, negative.', f"Sentence: {ex['sentence']}")
    if task == 'mrpc':
        return ('Task: MRPC (paraphrase). Determine whether the two sentences are semantically equivalent. Answer with exactly one label: equivalent, not_equivalent.', f"Sentence 1: {ex['sentence1']}\nSentence 2: {ex['sentence2']}")
    if task == 'stsb':
        return ('Task: STS-B (semantic textual similarity). Rate the semantic similarity of the two sentences on a scale from 0 to 5 (0 = completely different, 5 = equivalent in meaning). Answer with a single number (you may use one decimal place).', f"Sentence 1: {ex['sentence1']}\nSentence 2: {ex['sentence2']}")
    if task == 'qqp':
        return ('Task: QQP (duplicate questions). Determine whether the two questions are duplicates. Answer with exactly one label: duplicate, not_duplicate.', f"Question 1: {ex.get('question1') or ''}\nQuestion 2: {ex.get('question2') or ''}")
    if task == 'mnli':
        return ('Task: MNLI (natural language inference). Given a premise and a hypothesis, determine the relationship. Answer with exactly one label: entailment, neutral, contradiction.', f"Premise: {ex['premise']}\nHypothesis: {ex['hypothesis']}")
    if task == 'qnli':
        return ('Task: QNLI (question-answering NLI). Given a question and a sentence, determine whether the sentence entails the answer to the question. Answer with exactly one label: entailment, not_entailment.', f"Question: {ex['question']}\nSentence: {ex['sentence']}")
    if task == 'rte':
        return ('Task: RTE (recognizing textual entailment). Given a premise and a hypothesis, determine whether the premise entails the hypothesis. Answer with exactly one label: entailment, not_entailment.', f"Premise: {ex['sentence1']}\nHypothesis: {ex['sentence2']}")
    raise ValueError(task)

def _norm(s: str) -> str:
    s = (s or '').strip().lower().replace('\n', ' ')
    return re.sub('\\s+', ' ', s)

def parse_pred(task: str, text: str):
    task = task.lower()
    t = _norm(text)
    for prefix in ['answer:', 'label:', 'prediction:', 'pred:']:
        if t.startswith(prefix):
            t = t[len(prefix):].strip()
    if task == 'stsb':
        nums = re.findall('-?\\d+\\.?\\d*', t.replace(',', ''))
        if not nums:
            return (0.0, False)
        try:
            return (max(0.0, min(5.0, float(nums[0]))), True)
        except Exception:
            return (0.0, False)
    if task == 'mnli':
        if 'contradiction' in t:
            return (2, True)
        if 'neutral' in t:
            return (1, True)
        if 'entailment' in t:
            return (0, True)
        if t.startswith('yes'):
            return (0, False)
        if t.startswith('no'):
            return (2, False)
        return (0, False)
    if task == 'cola':
        if 'unacceptable' in t or 'not acceptable' in t:
            return (0, True)
        if 'acceptable' in t:
            return (1, True)
        if t.startswith('yes'):
            return (1, False)
        if t.startswith('no'):
            return (0, False)
        return (0, False)
    if task == 'sst2':
        if 'positive' in t:
            return (1, True)
        if 'negative' in t:
            return (0, True)
        return (0, False)
    if task == 'mrpc':
        if 'not_equivalent' in t or 'not equivalent' in t:
            return (0, True)
        if 'equivalent' in t:
            return (1, True)
        if t.startswith('yes'):
            return (1, False)
        if t.startswith('no'):
            return (0, False)
        return (0, False)
    if task == 'qqp':
        if 'not_duplicate' in t or 'not duplicate' in t:
            return (0, True)
        if 'duplicate' in t:
            return (1, True)
        if t.startswith('yes'):
            return (1, False)
        if t.startswith('no'):
            return (0, False)
        return (0, False)
    if task in ('qnli', 'rte'):
        if 'not_entailment' in t or 'not entailment' in t:
            return (1, True)
        if 'entailment' in t:
            return (0, True)
        if t.startswith('yes'):
            return (0, False)
        if t.startswith('no'):
            return (1, False)
        return (0, False)
    raise ValueError(task)

def _dtype():
    global torch
    if torch is None:
        import torch as _torch
        torch = _torch
    if torch.cuda.is_available():
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32

def load_eval_model(base_model_id: str, adapter_dir: Path, num_beams: int, revision=None):
    from peft import PeftModel
    from transformers import GenerationConfig
    dtype = _dtype()
    tokenizer = build_tokenizer(base_model_id, revision=revision)
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = 0
    model = load_base_model(
        base_model_id,
        revision=revision,
        torch_dtype=dtype,
        device_map='auto',
        trust_remote_code=True,
    )
    model.config.use_cache = False
    try:
        model = PeftModel.from_pretrained(
            model,
            str(adapter_dir),
            torch_dtype=dtype,
            device_map='auto',
        )
    except TypeError as exc:
        try:
            model = PeftModel.from_pretrained(
                model,
                str(adapter_dir),
                dtype=dtype,
                device_map='auto',
            )
        except TypeError:
            raise exc
    gen_cfg = GenerationConfig(do_sample=False, num_beams=max(1, int(num_beams)), pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    if hasattr(gen_cfg, 'remove_invalid_values'):
        gen_cfg.remove_invalid_values = True
    if hasattr(gen_cfg, 'renormalize_logits'):
        gen_cfg.renormalize_logits = True
    model.eval()
    return (model, tokenizer, gen_cfg)

def generate_labels(model, tokenizer, gen_cfg, prompts, max_input_length: int, max_new_tokens: int):
    global torch
    if torch is None:
        import torch as _torch
        torch = _torch
    inputs = tokenizer(prompts, return_tensors='pt', padding=True, truncation=True, max_length=int(max_input_length))
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(**inputs, generation_config=gen_cfg, max_new_tokens=int(max_new_tokens), return_dict_in_generate=True, output_scores=False)
    decoded = tokenizer.batch_decode(out.sequences, skip_special_tokens=True)
    preds = []
    for text in decoded:
        preds.append(text.split('### Response:', 1)[1].strip() if '### Response:' in text else text.strip())
    return preds

def _maybe_limit(ds, n):
    if n is None or int(n) <= 0:
        return ds
    return ds.shuffle(
        seed=EVAL_SUBSET_SEED,
        keep_in_memory=True,
    ).select(range(min(int(n), len(ds))))

def eval_task(model, tokenizer, gen_cfg, task: str, batch_size: int, max_input_length: int, max_new_tokens: int, fast_dev_run: int=0, split_name: str='validation', data_root: Path | None=None):
    from tqdm.auto import tqdm
    ds = _load_glue_dataset(task, data_root)
    split = ds[split_name].filter(
        lambda x: x['label'] != -1,
        keep_in_memory=True,
    )
    split = _maybe_limit(split, fast_dev_run)
    refs, prompts = ([], [])
    for ex in split:
        inst, inp = glue_to_instruction_input(task, ex)
        prompts.append(build_prompt(inst, inp))
        refs.append(ex['label'])
    pred_ids, parse_ok = ([], 0)
    for i in tqdm(range(0, len(prompts), int(batch_size)), desc=f'predict {task}:{split_name}'):
        outs = generate_labels(model, tokenizer, gen_cfg, prompts[i:i + int(batch_size)], max_input_length, max_new_tokens)
        for o in outs:
            pred, ok = parse_pred(task, o)
            pred_ids.append(pred)
            parse_ok += int(ok)
    if task == 'stsb':
        res = _compute_metric(task, [float(p) for p in pred_ids], [float(r) for r in refs])
        score = (float(res['pearson']) + float(res['spearmanr'])) / 2.0
    else:
        res = _compute_metric(task, pred_ids, refs)
        if task in ('mrpc', 'qqp'):
            score = (float(res['accuracy']) + float(res['f1'])) / 2.0
        elif task == 'cola':
            score = float(res.get('matthews_correlation', res.get('matthews', 0.0)))
        else:
            score = float(res['accuracy'])
    return {'task': task, 'score': float(score), 'parse_rate': float(parse_ok / max(1, len(pred_ids))), 'n': int(len(pred_ids)), 'metrics': {k: float(v) for k, v in res.items()}}

def eval_mnli(model, tokenizer, gen_cfg, batch_size: int, max_input_length: int, max_new_tokens: int, fast_dev_run: int=0, data_root: Path | None=None):
    from tqdm.auto import tqdm
    ds = _load_glue_dataset('mnli', data_root)
    details = {}
    for split_name in ['validation_matched', 'validation_mismatched']:
        split = ds[split_name].filter(
            lambda x: x['label'] != -1,
            keep_in_memory=True,
        )
        split = _maybe_limit(split, fast_dev_run)
        refs, prompts = ([], [])
        for ex in split:
            inst, inp = glue_to_instruction_input('mnli', ex)
            prompts.append(build_prompt(inst, inp))
            refs.append(ex['label'])
        pred_ids, parse_ok = ([], 0)
        for i in tqdm(range(0, len(prompts), int(batch_size)), desc=f'predict mnli:{split_name}'):
            outs = generate_labels(model, tokenizer, gen_cfg, prompts[i:i + int(batch_size)], max_input_length, max_new_tokens)
            for o in outs:
                pred, ok = parse_pred('mnli', o)
                pred_ids.append(pred)
                parse_ok += int(ok)
        res = _compute_metric('mnli', pred_ids, refs)
        details[split_name] = {'metrics': {k: float(v) for k, v in res.items()}, 'parse_rate': float(parse_ok / max(1, len(pred_ids))), 'n': int(len(pred_ids))}
    score = (details['validation_matched']['metrics']['accuracy'] + details['validation_mismatched']['metrics']['accuracy']) / 2.0
    return {'task': 'mnli', 'score': float(score), 'matched_acc': float(details['validation_matched']['metrics']['accuracy']), 'mismatched_acc': float(details['validation_mismatched']['metrics']['accuracy']), 'details': details}

def evaluate_glue8(cfg: RunConfig, batch_size: int=128, num_beams: int=1, max_new_tokens: int=8, max_input_length: int=384, fast_dev_run: int=0, data_root: Path | None=None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cfg.finalize()
    adapter_dir = Path(cfg.output_dir)
    adapter_status = completed_adapter_status(cfg)
    if adapter_status is None:
        raise RuntimeError(f'Adapter is not complete: {adapter_dir}')
    evaluation_revision = (
        adapter_status.get('resolved_model_revision') or cfg.model_revision
    )
    result_dir = Path(cfg.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    data_root = Path(data_root).resolve() if data_root is not None else None
    asset_identity = None
    if data_root is not None:
        manifest = _load_glue_asset_manifest(data_root)
        asset_identity = {
            'dataset_id': manifest['dataset_id'],
            'dataset_revision': manifest['dataset_revision'],
            'content_sha256': manifest['content_sha256'],
            'manifest_sha256': manifest['manifest_sha256'],
        }
    prepare_evaluation_cache(
        result_dir,
        {
            'config_fingerprint': cfg.config_fingerprint,
            'dataset': cfg.dataset,
            'base_model': cfg.base_model,
            'requested_model_revision': cfg.model_revision,
            'resolved_model_revision': evaluation_revision,
            'tasks': GLUE_TASKS,
            'batch_size': int(batch_size),
            'num_beams': int(num_beams),
            'max_new_tokens': int(max_new_tokens),
            'max_input_length': int(max_input_length),
            'fast_dev_run': int(fast_dev_run),
            'eval_subset_seed': EVAL_SUBSET_SEED if int(fast_dev_run) > 0 else None,
            'glue_eval_assets': asset_identity,
        },
        artifact_names=[f'{task}.json' for task in GLUE_TASKS],
        force=bool(cfg.force_eval),
    )
    cached, tasks_to_run = ({}, [])
    for task in GLUE_TASKS:
        p = result_dir / f'{task}.json'
        if p.exists():
            cached[task] = json.load(open(p, 'r', encoding='utf-8'))
        else:
            tasks_to_run.append(task)
    model = tokenizer = gen_cfg = None
    if tasks_to_run:
        model, tokenizer, gen_cfg = load_eval_model(
            cfg.base_model,
            adapter_dir,
            num_beams,
            revision=evaluation_revision,
        )
    task_rows = []
    for task in GLUE_TASKS:
        p = result_dir / f'{task}.json'
        if task in cached:
            rec = cached[task]
            print(f'[eval cache] {task}: {p}')
        else:
            rec = eval_mnli(model, tokenizer, gen_cfg, batch_size, max_input_length, max_new_tokens, fast_dev_run, data_root=data_root) if task == 'mnli' else eval_task(model, tokenizer, gen_cfg, task, batch_size, max_input_length, max_new_tokens, fast_dev_run, data_root=data_root)
            write_json_atomic(p, rec)
        task_rows.append(rec)
    if model is not None:
        del model, tokenizer, gen_cfg
        cleanup_cuda()
    score_row = {'method': cfg.method, 'privacy': cfg.privacy, 'base_model': cfg.base_model, 'lora_r': int(cfg.lora_r)}
    for rec in task_rows:
        score_row[rec['task']] = float(rec['score'])
    score_row['GLUE8_Avg'] = sum((score_row[t] for t in GLUE_TASKS)) / len(GLUE_TASKS)
    scores_df = pd.DataFrame([score_row])
    task_df = pd.DataFrame(task_rows)
    scores_df.to_csv(result_dir / 'summary.csv', index=False)
    task_df.to_csv(result_dir / 'details.csv', index=False)
    print(scores_df[['method', 'privacy', 'cola', 'sst2', 'mrpc', 'stsb', 'qqp', 'mnli', 'qnli', 'rte', 'GLUE8_Avg']])
    print('Saved:', result_dir / 'summary.csv')
    return (scores_df, task_df)
