from __future__ import annotations
import gc
import importlib.metadata
import json
import os
import platform
import random
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
import torch

def project_root() -> Path:
    return Path(__file__).resolve().parents[2]

def llm_adapters_dir(root: Optional[Path]=None) -> Path:
    root = project_root() if root is None else Path(root)
    return root / 'LLM-Adapters'

def tag_text(x: str) -> str:
    return str(x).replace('/', '_').replace(':', '_').replace(' ', '_')

def set_seed(seed: int=42) -> None:
    random.seed(seed)
    os.environ.setdefault('PYTHONHASHSEED', str(seed))
    try:
        import numpy as np
    except ImportError:
        pass
    else:
        np.random.seed(int(seed) % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def build_prompt(instruction: str, input_text: str, output: str='') -> str:
    data_point = {'instruction': instruction, 'input': input_text, 'output': output}
    return generate_prompt(data_point)

def generate_prompt(data_point: Dict[str, Any]) -> str:
    if data_point.get('input'):
        return f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. \n\n                ### Instruction:\n                {data_point['instruction']}\n                \n                ### Input:\n                {data_point['input']}\n                \n                ### Response:\n                {data_point['output']}"
    return f"Below is an instruction that describes a task. Write a response that appropriately completes the request.  \n\n                ### Instruction:\n                {data_point['instruction']}\n                \n                ### Response:\n                {data_point['output']}"

def build_tokenizer(model_id: str, *, revision: Optional[str]=None):
    from transformers import AutoTokenizer
    kwargs = {'trust_remote_code': True}
    if revision is not None:
        kwargs['revision'] = revision
    tok = AutoTokenizer.from_pretrained(model_id, **kwargs)
    tok.padding_side = 'left'
    if tok.pad_token_id is None:
        tok.pad_token_id = 0
    return tok

def tokenize_prompt(tokenizer, example: Dict[str, Any], cutoff_len: int, train_on_inputs: bool, base_model: str):
    full_prompt = generate_prompt(example)
    result = tokenizer(full_prompt, truncation=True, max_length=cutoff_len, padding=False, return_tensors=None)
    if result['input_ids'][-1] != tokenizer.eos_token_id and len(result['input_ids']) < cutoff_len:
        result['input_ids'].append(tokenizer.eos_token_id)
        if 'chatglm' not in base_model:
            result['attention_mask'].append(1)
    result['labels'] = result['input_ids'].copy()
    if not train_on_inputs:
        user_prompt = generate_prompt({**example, 'output': ''})
        user_tok = tokenizer(user_prompt, truncation=True, max_length=cutoff_len, padding=False, return_tensors=None)
        user_prompt_len = len(user_tok['input_ids'])
        result['labels'] = [-100] * user_prompt_len + result['labels'][user_prompt_len:]
    return result

def freeze_vision_tower_params(model) -> None:
    frozen = 0
    for name, p in model.named_parameters():
        if 'vision_tower' in name or '.vision_model.' in name:
            if p.requires_grad:
                p.requires_grad = False
                frozen += 1
    if frozen:
        print(f'[train] Froze {frozen} trainable vision-tower parameters for text-only training.')

def iter_microbatches(batch_dict: Dict[str, torch.Tensor], micro_bs: int):
    bs = batch_dict['input_ids'].shape[0]
    for start in range(0, bs, micro_bs):
        end = min(bs, start + micro_bs)
        yield {k: v[start:end] for k, v in batch_dict.items()}

def unwrap_for_save(model):
    return getattr(model, '_module', model)

def adapter_is_complete(path: Path) -> bool:
    path = Path(path)
    if not path.exists():
        return False
    return (path / 'adapter_config.json').exists() and any(((path / name).exists() for name in ['adapter_model.safetensors', 'adapter_model.bin']))

def status_path(output_dir: Path) -> Path:
    return Path(output_dir) / 'run_status.json'

def save_status(output_dir: Path, **payload: Any) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_json_atomic(status_path(out), payload)


def write_json_atomic(target: Path, payload: Any) -> None:
    target = Path(target)
    temporary = _new_atomic_temp_path(target)
    try:
        with open(temporary, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2, sort_keys=True, default=str)
            f.write('\n')
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, target)
        _fsync_directory(target.parent)
    finally:
        temporary.unlink(missing_ok=True)


def collect_runtime_metadata(root: Path) -> Dict[str, Any]:
    def git_value(*args: str) -> Optional[str]:
        try:
            result = subprocess.run(
                ['git', *args],
                cwd=str(root),
                check=True,
                capture_output=True,
                text=True,
            )
            return result.stdout.strip()
        except Exception:
            return None

    packages: Dict[str, Optional[str]] = {}
    for name in ('torch', 'transformers', 'peft', 'opacus', 'datasets', 'evaluate', 'accelerate'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    cuda = {
        'available': bool(torch.cuda.is_available()),
        'torch_cuda': getattr(torch.version, 'cuda', None),
        'device_count': int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        'devices': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        if torch.cuda.is_available()
        else [],
    }
    dirty = git_value('status', '--porcelain')
    return {
        'python': sys.version,
        'platform': platform.platform(),
        'packages': packages,
        'cuda': cuda,
        'git_sha': git_value('rev-parse', 'HEAD'),
        'git_branch': git_value('branch', '--show-current'),
        'git_dirty': bool(dirty) if dirty is not None else None,
    }

def checkpoint_file(output_dir: Path) -> Path:
    return Path(output_dir) / '_resume_checkpoint.pt'

def get_rng_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {
        'python': random.getstate(),
        'torch': torch.get_rng_state(),
    }
    try:
        import numpy as np
    except ImportError:
        pass
    else:
        state['numpy'] = np.random.get_state()
    if torch.cuda.is_available():
        state['cuda'] = torch.cuda.get_rng_state_all()
    return state

def set_rng_state(state: Optional[Dict[str, Any]]) -> None:
    if not state:
        return
    if 'python' in state:
        random.setstate(state['python'])
    if 'numpy' in state:
        try:
            import numpy as np
        except ImportError:
            pass
        else:
            np.random.set_state(state['numpy'])
    if 'torch' in state:
        torch.set_rng_state(state['torch'])
    if torch.cuda.is_available() and 'cuda' in state:
        torch.cuda.set_rng_state_all(state['cuda'])


def _new_atomic_temp_path(target: Path) -> Path:
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(
        prefix=f'.{target.name}.',
        suffix='.tmp',
        dir=str(target.parent),
    )
    os.close(fd)
    return Path(name)


def _fsync_file(path: Path) -> None:
    with open(path, 'rb') as f:
        os.fsync(f.fileno())


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, 'O_DIRECTORY'):
        flags |= os.O_DIRECTORY
    try:
        fd = os.open(str(path), flags)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        # Some filesystems do not support fsync on directory descriptors.
        pass
    finally:
        os.close(fd)

def trainable_state_dict_cpu(model) -> Dict[str, torch.Tensor]:
    model = unwrap_for_save(model)
    return {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}

def _canonical_parameter_name(name: str) -> str:
    while name.startswith('_module.'):
        name = name[len('_module.'):]
    return name


def load_trainable_state_dict(model, state: Dict[str, torch.Tensor]) -> int:
    named = dict(model.named_parameters())
    canonical_named = {_canonical_parameter_name(n): p for n, p in named.items()}
    loaded = 0
    missing: List[str] = []
    with torch.no_grad():
        for n, v in state.items():
            p = canonical_named.get(_canonical_parameter_name(n))
            if p is None:
                missing.append(n)
                continue
            if tuple(p.shape) != tuple(v.shape):
                raise ValueError(f'Checkpoint shape mismatch for {n}: saved={tuple(v.shape)} current={tuple(p.shape)}')
            p.copy_(v.to(device=p.device, dtype=p.dtype))
            loaded += 1
    if state and loaded == 0:
        raise ValueError('Checkpoint contained trainable parameters, but none matched the current model')
    if missing:
        print(f'[resume] warning: skipped {len(missing)} unmatched trainable tensors')
    print(f'[resume] restored {loaded} trainable tensors')
    return loaded

def save_resume_checkpoint(output_dir: Path, model, optimizer, update_steps: int, extra: Optional[Dict[str, Any]]=None) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {'update_steps': int(update_steps), 'trainable_state': trainable_state_dict_cpu(model), 'optimizer_state': optimizer.state_dict() if optimizer is not None else None, 'rng_state': get_rng_state(), 'extra': extra or {}}
    target = checkpoint_file(output_dir)
    temporary = _new_atomic_temp_path(target)
    try:
        torch.save(payload, temporary)
        _fsync_file(temporary)
        os.replace(temporary, target)
        _fsync_directory(target.parent)
    finally:
        temporary.unlink(missing_ok=True)

def load_resume_checkpoint_if_available(output_dir: Path) -> Optional[Dict[str, Any]]:
    path = checkpoint_file(output_dir)
    if not path.exists():
        return None
    print(f'[resume] loading {path}')
    # Resume checkpoints are locally generated, trusted training artifacts. Passing
    # weights_only=False explicitly is required for optimizer states on Torch 2.6+.
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError as exc:
        if 'weights_only' not in str(exc):
            raise
        # Compatibility with Torch releases predating the weights_only argument.
        return torch.load(path, map_location='cpu')


def truncate_jsonl_to_step(path: Path, step: int, *, step_key: str='step') -> int:
    """Atomically retain JSONL records whose integer ``step_key`` is at most ``step``.

    The helper is intended to run before opening an append-mode logger during
    checkpoint recovery. A malformed final line is treated as an interrupted
    write and removed; malformed non-final content raises instead of silently
    discarding an already-corrupt log. The return value is the number of lines
    removed. A missing file is a no-op.
    """
    path = Path(path)
    if not path.exists():
        return 0
    if not step_key:
        raise ValueError('step_key must be non-empty')
    max_step = int(step)
    if isinstance(step, float) and not step.is_integer():
        raise ValueError('step must be an integer')

    lines = path.read_text(encoding='utf-8').splitlines(keepends=True)
    retained: List[str] = []
    removed = 0
    for index, line in enumerate(lines):
        if not line.strip():
            removed += 1
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            has_later_content = any(later.strip() for later in lines[index + 1:])
            if has_later_content:
                raise ValueError(f'Malformed JSONL record at {path}:{index + 1}') from exc
            removed += 1
            break
        if not isinstance(record, dict) or step_key not in record:
            raise ValueError(f'JSONL record at {path}:{index + 1} has no {step_key!r} field')
        try:
            record_step = int(record[step_key])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f'JSONL record at {path}:{index + 1} has a non-integer {step_key!r}'
            ) from exc
        if isinstance(record[step_key], float) and not record[step_key].is_integer():
            raise ValueError(
                f'JSONL record at {path}:{index + 1} has a non-integer {step_key!r}'
            )
        if record_step <= max_step:
            retained.append(line if line.endswith('\n') else line + '\n')
        else:
            removed += 1

    temporary = _new_atomic_temp_path(path)
    try:
        with open(temporary, 'w', encoding='utf-8') as f:
            f.writelines(retained)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)
    return removed

class JsonlLogger:

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(self.path, 'a', encoding='utf-8')

    def log(self, rec: Dict[str, Any]) -> None:
        self.f.write(json.dumps(rec, ensure_ascii=False, default=str) + '\n')
        self.f.flush()

    def close(self) -> None:
        self.f.close()

def clean_output_dir(path: Path, force: bool) -> None:
    path = Path(path)
    if force and path.exists():
        print(f'[train] removing {path}')
        shutil.rmtree(path)
