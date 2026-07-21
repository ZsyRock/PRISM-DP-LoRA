from __future__ import annotations
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import torch
from tqdm.auto import tqdm
from .experiment_identity import FINGERPRINT_SCHEMA_VERSION, config_fingerprint, content_sha256, extract_status_fingerprint, git_worktree_identity, make_run_id
from .losses import forward_causal_lm_per_example_loss
from .modeling import (
    load_base_model,
    resolve_text_lora_target_modules,
    resolved_model_revision,
)
from .utils import JsonlLogger, adapter_is_complete, build_tokenizer, checkpoint_file, clean_output_dir, cleanup_cuda, collect_runtime_metadata, freeze_vision_tower_params, generate_prompt, iter_microbatches, llm_adapters_dir, load_resume_checkpoint_if_available, load_trainable_state_dict, save_resume_checkpoint, save_status, set_rng_state, set_seed, tokenize_prompt, truncate_jsonl_to_step, unwrap_for_save


CHECKPOINT_SCHEMA_VERSION = 2
TELEMETRY_SCHEMA_VERSION = 2
LOSS_DEFINITION = 'per_record_mean_of_nonignored_next_token_losses'

@dataclass
class RunConfig:
    dataset: str
    method: str
    privacy: str
    root: Path
    base_model: str = 'google/gemma-3-4b-pt'
    model_revision: str = 'main'
    seed: int = 42
    run_name: Optional[str] = None
    repeat_id: Optional[int] = None
    data_path: Optional[Path] = None
    output_dir: Optional[Path] = None
    result_dir: Optional[Path] = None
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: ['q_proj', 'k_proj', 'v_proj', 'up_proj', 'down_proj'])
    total_update_steps: Optional[int] = None
    batch_size: int = 64
    micro_batch_size: int = 4
    learning_rate: Optional[float] = None
    cutoff_len: Optional[int] = None
    train_on_inputs: Optional[bool] = None
    val_set_size: int = 0
    eval_step: int = 100
    save_step: int = 200
    dp_epsilon: float = 6.0
    dp_delta: float = 1e-05
    dp_max_grad_norm: float = 1.0
    dp_grad_sample_mode: str = 'functorch'
    dp_accountant: str = 'prv'
    dp_secure_mode: bool = False
    require_cuda: bool = False
    telemetry_mode: str = 'dp_safe'
    allow_non_private_telemetry: bool = False
    raw_hist_bins: int = 32
    raw_hist_max: float = 0.0
    slaclip_num_slots: int = 0
    slaclip_eta: float = 0.5
    slaclip_beta: float = 0.5
    slaclip_c_min: float = 0.1
    slaclip_c_max: float = 50.0
    force_train: bool = False
    force_eval: bool = False
    resume: bool = True
    checkpoint_every: int = 25
    run_train: bool = True
    run_eval: bool = True
    spectral_svd_device: str = 'cpu'
    spectral_oversample: int = 8
    spectral_n_iter: int = 2
    prism_floor_factor: float = 0.5
    prism_floor_mode: str = 'scalar'
    prism_cond_max: float = 10000.0
    prism_cond_strategy: str = 'raise_small'
    prism_lift_fix: str = 'both'
    prism_debias_second_moment: bool = False
    max_update_norm: float = 0.0
    data_content_sha256: Optional[str] = field(init=False, default=None)
    config_fingerprint: str = field(init=False, default='')
    run_id: str = field(init=False, default='')
    resolved_model_revision: Optional[str] = field(init=False, default=None)
    implementation_git_sha: Optional[str] = field(init=False, default=None)
    implementation_git_dirty: Optional[bool] = field(init=False, default=None)
    implementation_dirty_sha256: Optional[str] = field(init=False, default=None)

    def fingerprint_payload(self) -> Dict[str, Any]:
        """Return the path-independent fields that define one experiment."""
        return {
            'fingerprint_schema_version': FINGERPRINT_SCHEMA_VERSION,
            'implementation_git_sha': self.implementation_git_sha,
            'implementation_git_dirty': self.implementation_git_dirty,
            'implementation_dirty_sha256': self.implementation_dirty_sha256,
            'dataset': self.dataset,
            'method': self.method,
            'privacy': self.privacy,
            'base_model': self.base_model,
            'model_revision': self.model_revision,
            'seed': int(self.seed),
            'repeat_id': self.repeat_id,
            'data_content_sha256': self.data_content_sha256,
            'lora_r': int(self.lora_r),
            'lora_alpha': int(self.lora_alpha),
            'lora_dropout': float(self.lora_dropout),
            'target_modules': sorted(str(x) for x in self.target_modules),
            'total_update_steps': int(self.total_update_steps),
            'batch_size': int(self.batch_size),
            'micro_batch_size': int(self.micro_batch_size),
            'learning_rate': float(self.learning_rate),
            'cutoff_len': int(self.cutoff_len),
            'train_on_inputs': bool(self.train_on_inputs),
            'val_set_size': int(self.val_set_size),
            'eval_step': int(self.eval_step),
            'save_step': int(self.save_step),
            'dp_epsilon': float(self.dp_epsilon),
            'dp_delta': float(self.dp_delta),
            'dp_max_grad_norm': float(self.dp_max_grad_norm),
            'dp_grad_sample_mode': self.dp_grad_sample_mode,
            'dp_accountant': self.dp_accountant,
            'dp_secure_mode': bool(self.dp_secure_mode),
            'telemetry_mode': self.telemetry_mode,
            'raw_hist_bins': int(self.raw_hist_bins),
            'raw_hist_max': float(self.raw_hist_max),
            'slaclip_num_slots': int(self.slaclip_num_slots),
            'slaclip_eta': float(self.slaclip_eta),
            'slaclip_beta': float(self.slaclip_beta),
            'slaclip_c_min': float(self.slaclip_c_min),
            'slaclip_c_max': float(self.slaclip_c_max),
            'spectral_svd_device': self.spectral_svd_device,
            'spectral_oversample': int(self.spectral_oversample),
            'spectral_n_iter': int(self.spectral_n_iter),
            'prism_floor_factor': float(self.prism_floor_factor),
            'prism_floor_mode': self.prism_floor_mode,
            'prism_cond_max': float(self.prism_cond_max),
            'prism_cond_strategy': self.prism_cond_strategy,
            'prism_lift_fix': self.prism_lift_fix,
            'prism_debias_second_moment': bool(self.prism_debias_second_moment),
            'max_update_norm': float(self.max_update_norm),
        }

    def finalize(self) -> 'RunConfig':
        self.root = Path(self.root).resolve()
        self.dataset = self.dataset.lower().replace('-', '')
        if self.dataset in {'math', 'math10k'}:
            self.dataset = 'math10k'
        elif self.dataset in {'glue', 'glue8'}:
            self.dataset = 'glue8'
        else:
            raise ValueError('dataset must be one of: math10k, glue8')
        method = self.method.lower().replace('-', '_')
        if method in {'baseline', 'prism', 'fixed', 'fixed_prism'}:
            self.method = 'baseline'
        elif method in {'slaclip', 'slaclip_prism'}:
            self.method = 'slaclip'
        else:
            raise ValueError('method must be baseline or slaclip')
        self.privacy = self.privacy.lower().replace('_', '-')
        if self.privacy in {'non-dp', 'nondp', 'none'}:
            self.privacy = 'nondp'
        if self.privacy not in {'dp', 'nondp'}:
            raise ValueError('privacy must be dp or nondp')
        if self.method == 'slaclip' and self.privacy != 'dp':
            raise ValueError('SlaClip is a DP clipping controller; use --privacy dp')
        self.telemetry_mode = self.telemetry_mode.lower().replace('-', '_')
        if self.telemetry_mode not in {'dp_safe', 'research_raw'}:
            raise ValueError('telemetry_mode must be dp_safe or research_raw')
        if self.telemetry_mode == 'research_raw' and not self.allow_non_private_telemetry:
            raise ValueError(
                'research_raw exposes private training statistics. Re-run with '
                '--allow_non_private_telemetry only for trusted research analysis.'
            )
        if self.telemetry_mode == 'research_raw' and self.privacy != 'dp':
            raise ValueError('research_raw is intended for observing DP training; use --privacy dp')
        self.dp_accountant = self.dp_accountant.lower()
        if self.dp_accountant not in {'rdp', 'prv', 'gdp'}:
            raise ValueError('dp_accountant must be one of: rdp, prv, gdp')
        self.base_model = '' if self.base_model is None else str(self.base_model).strip()
        self.model_revision = '' if self.model_revision is None else str(self.model_revision).strip()
        if not self.base_model:
            raise ValueError('base_model must not be empty')
        if not self.model_revision:
            raise ValueError('model_revision must not be empty; use an immutable model commit for formal runs')
        adapters = llm_adapters_dir(self.root)
        if self.dataset == 'math10k':
            self.total_update_steps = 300 if self.total_update_steps is None else self.total_update_steps
            self.learning_rate = 0.0003 if self.learning_rate is None else self.learning_rate
            self.cutoff_len = 256 if self.cutoff_len is None else self.cutoff_len
            self.train_on_inputs = True if self.train_on_inputs is None else self.train_on_inputs
            self.val_set_size = int(self.val_set_size or 120)
            self.eval_step = int(self.eval_step or 10)
            self.save_step = int(self.save_step or 20)
            self.data_path = self.data_path or adapters / 'ft-training_set' / 'math_10k.json'
        else:
            self.total_update_steps = 500 if self.total_update_steps is None else self.total_update_steps
            self.learning_rate = 0.0002 if self.learning_rate is None else self.learning_rate
            self.cutoff_len = 384 if self.cutoff_len is None else self.cutoff_len
            self.train_on_inputs = False if self.train_on_inputs is None else self.train_on_inputs
            self.val_set_size = int(self.val_set_size or 1000)
            self.eval_step = int(self.eval_step or 100)
            self.save_step = int(self.save_step or 200)
            self.data_path = self.data_path or adapters / 'ft-training_set' / 'glue8_1250.json'
        self.data_path = Path(self.data_path)
        if self.target_modules is None:
            self.target_modules = []
        elif isinstance(self.target_modules, str):
            self.target_modules = [x.strip() for x in self.target_modules.split(',') if x.strip()]
        else:
            self.target_modules = [str(x).strip() for x in self.target_modules if str(x).strip()]
        if self.batch_size <= 0 or self.micro_batch_size <= 0:
            raise ValueError('batch_size and micro_batch_size must be positive')
        if int(self.total_update_steps) <= 0:
            raise ValueError('total_update_steps must be positive')
        if float(self.learning_rate) <= 0:
            raise ValueError('learning_rate must be positive')
        if int(self.cutoff_len) <= 0:
            raise ValueError('cutoff_len must be positive')
        if self.lora_r <= 0 or self.lora_alpha <= 0:
            raise ValueError('lora_r and lora_alpha must be positive')
        if not 0.0 <= float(self.lora_dropout) < 1.0:
            raise ValueError('lora_dropout must be in [0, 1)')
        if not self.target_modules:
            raise ValueError('target_modules must contain at least one module')
        if self.dp_epsilon <= 0 or not 0.0 < float(self.dp_delta) < 1.0:
            raise ValueError('require dp_epsilon > 0 and 0 < dp_delta < 1')
        if self.dp_max_grad_norm <= 0:
            raise ValueError('initial clip threshold must be positive')
        if int(self.slaclip_num_slots) < 0:
            raise ValueError('slaclip_num_slots must be >= 0 (0 selects it automatically)')
        if float(self.slaclip_eta) < 0:
            raise ValueError('slaclip_eta must be non-negative')
        if not 0.0 <= float(self.slaclip_beta) <= 1.0:
            raise ValueError('slaclip_beta must be in [0, 1]')
        if float(self.slaclip_c_min) <= 0 or float(self.slaclip_c_max) < float(self.slaclip_c_min):
            raise ValueError('require 0 < slaclip_c_min <= slaclip_c_max')
        if self.method == 'slaclip' and not (
            float(self.slaclip_c_min) <= float(self.dp_max_grad_norm) <= float(self.slaclip_c_max)
        ):
            raise ValueError('SlaClip initial C must satisfy slaclip_c_min <= C0 <= slaclip_c_max')
        if self.raw_hist_bins <= 0 or self.raw_hist_max < 0:
            raise ValueError('raw_hist_bins must be positive and raw_hist_max must be non-negative')
        if int(self.checkpoint_every) <= 0:
            raise ValueError('checkpoint_every must be positive')
        if self.repeat_id is not None and int(self.repeat_id) < 0:
            raise ValueError('repeat_id must be non-negative')
        if self.dp_secure_mode and self.privacy != 'dp':
            raise ValueError('dp_secure_mode is only valid with --privacy dp')

        self.data_content_sha256 = content_sha256(self.data_path)
        (
            self.implementation_git_sha,
            self.implementation_git_dirty,
            self.implementation_dirty_sha256,
        ) = git_worktree_identity(self.root)
        if self.implementation_git_dirty and not self.implementation_dirty_sha256:
            raise RuntimeError(
                'The Git worktree is dirty but its content hash could not be computed; '
                'commit/stash the changes before starting an experiment.'
            )
        self.config_fingerprint = config_fingerprint(self.fingerprint_payload())
        label = self.run_name or f'{self.dataset}_{self.method}_{self.privacy}'
        self.run_id = make_run_id(
            label=label,
            clip_threshold=float(self.dp_max_grad_norm),
            fingerprint=self.config_fingerprint,
            repeat_id=self.repeat_id,
        )
        self.output_dir = Path(self.output_dir) if self.output_dir is not None else adapters / 'trained_models' / self.run_id
        self.result_dir = Path(self.result_dir) if self.result_dir is not None else adapters / 'experiment' / self.run_id
        return self

    @property
    def log_jsonl(self) -> Path:
        return Path(self.output_dir) / 'train_log.jsonl'

    @property
    def raw_log_jsonl(self) -> Path:
        return Path(self.result_dir) / 'research_raw' / 'NON_PRIVATE_train_log.jsonl'


def validate_existing_run_identity(cfg: RunConfig) -> None:
    """Refuse to reuse an output directory belonging to another experiment."""
    output_dir = Path(cfg.output_dir)
    status_file = output_dir / 'run_status.json'
    has_adapter = adapter_is_complete(output_dir)
    has_checkpoint = checkpoint_file(output_dir).exists()
    has_log = cfg.log_jsonl.exists() and cfg.log_jsonl.stat().st_size > 0
    if not status_file.exists():
        if has_adapter or has_checkpoint or has_log:
            raise RuntimeError(
                f'Existing artifacts in {output_dir} have no verifiable config fingerprint. '
                'Choose a new --run_name/output directory instead of reusing them.'
            )
        return
    try:
        with open(status_file, 'r', encoding='utf-8') as handle:
            status = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f'Could not validate existing run status {status_file}: {exc}') from exc
    if not isinstance(status, dict):
        raise RuntimeError(f'Existing run status is not a JSON object: {status_file}')
    found = extract_status_fingerprint(status)
    if not found:
        raise RuntimeError(
            f'Existing run status {status_file} predates config fingerprints. '
            'Choose a new --run_name/output directory.'
        )
    if found != cfg.config_fingerprint:
        raise RuntimeError(
            'Existing run fingerprint does not match the requested experiment: '
            f'found={found}, requested={cfg.config_fingerprint}, output_dir={output_dir}'
        )

def _preimport_prism_training_stack() -> None:
    import datasets
    import opacus
    import peft
    import transformers
    from torch.utils.data import DataLoader
    from .optim import prism as _prism_module

def _ensure_paths(cfg: RunConfig) -> None:
    for name in ('WORLD_SIZE', 'SLURM_NTASKS'):
        raw = os.environ.get(name)
        if raw is None:
            continue
        try:
            value = int(raw)
        except ValueError as exc:
            raise RuntimeError(f'{name} must be an integer, got {raw!r}') from exc
        if value != 1:
            raise RuntimeError(
                f'{name}={value} is unsupported: this trainer requires exactly one process and one GPU'
            )
    if cfg.require_cuda and not torch.cuda.is_available():
        raise RuntimeError('This run requires CUDA, but torch.cuda.is_available() is false')
    if not cfg.data_path or not Path(cfg.data_path).exists():
        raise FileNotFoundError(f'Training data not found: {cfg.data_path}')
    Path(cfg.output_dir).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.result_dir).mkdir(parents=True, exist_ok=True)

def _device_and_dtype() -> Tuple[torch.device, torch.dtype, dict]:
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return (torch.device(f'cuda:{local_rank}'), dtype, {'': local_rank})
    return (torch.device('cpu'), torch.float32, {'': 'cpu'})

def _calibrate_noise_multiplier(cfg: RunConfig, sample_rate: float) -> float:
    from opacus.accountants.utils import get_noise_multiplier
    kwargs = dict(
        target_epsilon=float(cfg.dp_epsilon),
        target_delta=float(cfg.dp_delta),
        sample_rate=float(sample_rate),
        accountant=cfg.dp_accountant,
    )
    try:
        return float(get_noise_multiplier(**kwargs, steps=int(cfg.total_update_steps)))
    except TypeError:
        # Older Opacus versions expose epochs rather than exact steps.
        return float(
            get_noise_multiplier(
                **kwargs,
                epochs=float(cfg.total_update_steps) * float(sample_rate),
            )
        )


def _make_private(privacy_engine, model, optimizer, train_loader, cfg: RunConfig):
    sample_rate = 1.0 / float(len(train_loader))
    noise_multiplier = _calibrate_noise_multiplier(cfg, sample_rate)
    common = dict(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        noise_multiplier=noise_multiplier,
        max_grad_norm=cfg.dp_max_grad_norm,
    )
    try:
        out = privacy_engine.make_private(**common, grad_sample_mode=cfg.dp_grad_sample_mode)
        return (out, cfg.dp_grad_sample_mode, noise_multiplier, sample_rate)
    except TypeError:
        out = privacy_engine.make_private(**common)
        return (out, 'default', noise_multiplier, sample_rate)
    except Exception as e:
        print(f'[warn] make_private failed for grad_sample_mode={cfg.dp_grad_sample_mode}: {e!r}')
        out = privacy_engine.make_private(**common, grad_sample_mode='hooks')
        return (out, 'hooks', noise_multiplier, sample_rate)

def _step_accountant(privacy_engine, noise_multiplier: float, sample_rate: float) -> None:
    try:
        privacy_engine.accountant.step(noise_multiplier=noise_multiplier, sample_rate=sample_rate)
    except TypeError:
        privacy_engine.accountant.step(noise_multiplier, sample_rate)

def _make_loader(cfg: RunConfig, tokenizer):
    import transformers
    from datasets import load_dataset
    from torch.utils.data import DataLoader
    data = load_dataset('json', data_files=str(cfg.data_path)) if str(cfg.data_path).endswith('.json') else load_dataset(str(cfg.data_path))
    train_ds = data['train']

    def map_fn(ex):
        return tokenize_prompt(tokenizer, ex, cutoff_len=int(cfg.cutoff_len), train_on_inputs=bool(cfg.train_on_inputs), base_model=cfg.base_model)
    train_ds = train_ds.map(map_fn, remove_columns=train_ds.column_names, desc='Tokenizing')
    collator = transformers.DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors='pt', padding=True)
    loader_kwargs = dict(dataset=train_ds, batch_size=int(cfg.batch_size), shuffle=True, drop_last=False, collate_fn=collator)
    generator = torch.Generator()
    generator.manual_seed(cfg.seed)
    loader_kwargs['generator'] = generator
    loader_kwargs['num_workers'] = 0
    loader = DataLoader(**loader_kwargs)
    return (loader, train_ds)

def _build_lora_model(cfg: RunConfig):
    from peft import LoraConfig, get_peft_model
    device, dtype, device_map = _device_and_dtype()
    tokenizer = build_tokenizer(cfg.base_model, revision=cfg.model_revision)
    model = load_base_model(
        cfg.base_model,
        revision=cfg.model_revision,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
    )
    cfg.resolved_model_revision = resolved_model_revision(model) or cfg.model_revision
    model.config.use_cache = False
    text_config = getattr(model.config, 'text_config', None)
    if text_config is not None and hasattr(text_config, 'use_cache'):
        text_config.use_cache = False
    resolved_targets = resolve_text_lora_target_modules(model, cfg.target_modules)
    lora_cfg = LoraConfig(r=int(cfg.lora_r), lora_alpha=int(cfg.lora_alpha), target_modules=resolved_targets, lora_dropout=float(cfg.lora_dropout), bias='none', task_type='CAUSAL_LM')
    model = get_peft_model(model, lora_cfg)
    freeze_vision_tower_params(model)
    trainable_lora = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and ('lora_A' in name or 'lora_B' in name)
    ]
    if not trainable_lora:
        raise RuntimeError(f'No trainable LoRA parameters matched target_modules={cfg.target_modules!r}')
    leaked_vision = [
        name for name in trainable_lora if 'vision_tower' in name or '.vision_model.' in name
    ]
    if leaked_vision:
        raise RuntimeError(
            f'Text-only run left {len(leaked_vision)} trainable vision LoRA tensors; first={leaked_vision[0]}'
        )
    print(
        f'[model] class={type(model.get_base_model()).__name__ if hasattr(model, "get_base_model") else type(model).__name__} '
        f'revision={cfg.resolved_model_revision} text_lora_modules={len(resolved_targets)} '
        f'trainable_lora_tensors={len(trainable_lora)}'
    )
    return (model, tokenizer, device)

def _initialize_prism_factors(cfg: RunConfig, model) -> None:
    from .optim.prism import spectral_init_peft_model
    print('[init] spectral residual initialization')
    spectral_init_peft_model(model, svd_device=cfg.spectral_svd_device, svd_oversample=int(cfg.spectral_oversample), svd_n_iter=int(cfg.spectral_n_iter), verbose=True)

def _build_prism_optimizer(cfg: RunConfig, model):
    from .optim.prism import PRISM, get_paired_lora_parameters
    params = get_paired_lora_parameters(model)
    print(f'[optimizer] paired LoRA tensors = {len(params)} / modules = {len(params) // 2}')
    return PRISM(
        params,
        lr=float(cfg.learning_rate),
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=0.0,
        use_adaptive=True,
        dp_precond_floor_factor=float(cfg.prism_floor_factor),
        dp_floor_mode=cfg.prism_floor_mode,
        precond_cond_max=float(cfg.prism_cond_max),
        precond_cond_strategy=cfg.prism_cond_strategy,
        precond_update_mode='current',
        lift_gauge_fix=cfg.prism_lift_fix,
        gauge_fix_eps=1e-12,
        max_update_norm=float(cfg.max_update_norm),
        dp_debias_second_moment=bool(cfg.prism_debias_second_moment),
        clipping_method=cfg.method,
        slaclip_num_slots=int(cfg.slaclip_num_slots),
        slaclip_eta=float(cfg.slaclip_eta),
        slaclip_beta=float(cfg.slaclip_beta),
        slaclip_c_min=float(cfg.slaclip_c_min),
        slaclip_c_max=float(cfg.slaclip_c_max),
        telemetry_mode=cfg.telemetry_mode,
        raw_hist_bins=int(cfg.raw_hist_bins),
        raw_hist_max=float(cfg.raw_hist_max),
    )

def _rebase_prism_for_save(model) -> Dict[str, float]:
    from .optim.prism import spectral_rebase_adapter_inplace
    stats = spectral_rebase_adapter_inplace(model, adapter_name='default', verbose=True)
    if int(stats.get('spectral_rebase_modules', 0)) <= 0:
        raise RuntimeError(
            'Spectral rebase did not transform any adapter module; refusing to save an '
            'adapter that would be incompatible with the original base model.'
        )
    return stats

def _loader_generator(loader):
    candidates = (
        getattr(getattr(loader, 'batch_sampler', None), 'generator', None),
        getattr(getattr(loader, 'sampler', None), 'generator', None),
        getattr(loader, 'generator', None),
    )
    for candidate in candidates:
        if candidate is not None and hasattr(candidate, 'get_state') and hasattr(candidate, 'set_state'):
            return candidate
    return None


def _decouple_loader_worker_rng(loader, seed: int) -> None:
    """Keep DataLoader iterator seeding from advancing the sampling RNG.

    PyTorch consumes ``loader.generator`` whenever a new iterator is created,
    even with zero workers. Opacus also uses that generator for Poisson sampling
    by default. A resumed run necessarily creates a fresh iterator, so sharing
    those generators would shift the post-resume sample sequence by one draw.
    The batch sampler retains its sampling generator while the loader receives a
    separate worker/base-seed generator (workers are fixed to zero in this repo).
    """

    worker_generator = torch.Generator()
    worker_generator.manual_seed((int(seed) + 0x5EED5EED) % (2**63 - 1))
    loader.generator = worker_generator


def _loader_generator_state(loader) -> Optional[torch.Tensor]:
    generator = _loader_generator(loader)
    if generator is None:
        return None
    return generator.get_state().detach().cpu()


def _set_loader_generator_state(loader, state: Optional[torch.Tensor]) -> None:
    if state is None:
        return
    generator = _loader_generator(loader)
    if generator is None:
        raise RuntimeError('checkpoint contains a data-loader RNG state, but the loader has no generator')
    generator.set_state(state)


def _validate_checkpoint_identity(cfg: RunConfig, checkpoint: Dict[str, Any]) -> Dict[str, Any]:
    extra = checkpoint.get('extra')
    if not isinstance(extra, dict):
        raise RuntimeError('resume checkpoint has no structured metadata; start a new run directory')
    if int(extra.get('checkpoint_schema_version', -1)) != CHECKPOINT_SCHEMA_VERSION:
        raise RuntimeError(
            f"unsupported checkpoint schema={extra.get('checkpoint_schema_version')!r}; "
            f'expected {CHECKPOINT_SCHEMA_VERSION}'
        )
    saved_fingerprint = extra.get('config_fingerprint')
    if saved_fingerprint != cfg.config_fingerprint:
        raise RuntimeError(
            f'checkpoint config fingerprint mismatch: saved={saved_fingerprint!r}, '
            f'current={cfg.config_fingerprint!r}'
        )
    saved_loss = extra.get('loss_definition')
    if saved_loss != LOSS_DEFINITION:
        raise RuntimeError(
            f'checkpoint loss definition mismatch: saved={saved_loss!r}, current={LOSS_DEFINITION!r}'
        )
    step = int(checkpoint.get('update_steps', 0))
    if step < 0 or step >= int(cfg.total_update_steps):
        raise RuntimeError(
            f'checkpoint update_steps={step} is outside [0, {int(cfg.total_update_steps)})'
        )
    return extra


def _restore_if_possible(cfg: RunConfig, model, optimizer) -> Optional[Dict[str, Any]]:
    if not cfg.resume or cfg.force_train:
        return None
    ckpt = load_resume_checkpoint_if_available(Path(cfg.output_dir))
    if ckpt is None:
        return None
    _validate_checkpoint_identity(cfg, ckpt)
    load_trainable_state_dict(model, ckpt.get('trainable_state', {}))
    if optimizer is not None and ckpt.get('optimizer_state') is not None:
        try:
            if hasattr(optimizer, '_ensure_state'):
                optimizer._ensure_state()
            optimizer.load_state_dict(ckpt['optimizer_state'])
        except Exception as e:
            raise RuntimeError(f'optimizer state could not be restored: {e!r}') from e
    step = int(ckpt.get('update_steps', 0))
    print(f'[resume] starting at update step {step}')
    return ckpt


def _restore_runtime_state(
    cfg: RunConfig,
    checkpoint: Optional[Dict[str, Any]],
    *,
    train_loader,
    optimizer,
    privacy_engine,
    noise_multiplier: Optional[float],
    sample_rate: Optional[float],
    expected_batch_size: Optional[float],
) -> int:
    if checkpoint is None:
        return 0
    extra = _validate_checkpoint_identity(cfg, checkpoint)
    for key, current in (
        ('noise_multiplier', noise_multiplier),
        ('sample_rate', sample_rate),
        ('expected_batch_size', expected_batch_size),
    ):
        saved = extra.get(key)
        if saved is None and current is None:
            continue
        if saved is None or current is None or not math.isclose(
            float(saved), float(current), rel_tol=1e-10, abs_tol=1e-12
        ):
            raise RuntimeError(f'checkpoint {key}={saved!r} does not match current value={current!r}')
    saved_revision = extra.get('resolved_model_revision')
    if saved_revision and cfg.resolved_model_revision and saved_revision != cfg.resolved_model_revision:
        raise RuntimeError(
            f'checkpoint resolved model revision={saved_revision!r} does not match '
            f'loaded revision={cfg.resolved_model_revision!r}'
        )
    _set_loader_generator_state(train_loader, extra.get('data_loader_generator_state'))
    if hasattr(optimizer, 'set_dp_noise_generator_state'):
        optimizer.set_dp_noise_generator_state(extra.get('dp_noise_generator_state'))
    accountant_state = extra.get('accountant_state')
    if privacy_engine is not None:
        if accountant_state is None:
            raise RuntimeError('DP checkpoint is missing accountant_state')
        privacy_engine.accountant.load_state_dict(accountant_state)
    elif accountant_state is not None:
        raise RuntimeError('non-DP run cannot restore a DP accountant state')
    # Restore the global RNG only after model/loading/privacy setup has consumed
    # any initialization randomness.
    set_rng_state(checkpoint.get('rng_state'))
    return int(checkpoint.get('update_steps', 0))


def _checkpoint_extra(
    cfg: RunConfig,
    *,
    train_loader,
    optimizer,
    privacy_engine,
    noise_multiplier: Optional[float],
    sample_rate: Optional[float],
    expected_batch_size: Optional[float],
) -> Dict[str, Any]:
    accountant_state = None
    if privacy_engine is not None:
        accountant_state = privacy_engine.accountant.state_dict()
    noise_state = None
    if hasattr(optimizer, 'get_dp_noise_generator_state'):
        noise_state = optimizer.get_dp_noise_generator_state()
    return {
        'checkpoint_schema_version': CHECKPOINT_SCHEMA_VERSION,
        'config_fingerprint': cfg.config_fingerprint,
        'run_id': cfg.run_id,
        'loss_definition': LOSS_DEFINITION,
        'resolved_model_revision': cfg.resolved_model_revision,
        'noise_multiplier': noise_multiplier,
        'sample_rate': sample_rate,
        'expected_batch_size': expected_batch_size,
        'data_loader_generator_state': _loader_generator_state(train_loader),
        'dp_noise_generator_state': noise_state,
        'accountant_state': accountant_state,
    }


def _read_status(output_dir: Path) -> Optional[Dict[str, Any]]:
    path = Path(output_dir) / 'run_status.json'
    if not path.exists():
        return None
    try:
        with open(path, 'r', encoding='utf-8') as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f'Could not read run status {path}: {exc}') from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f'Run status must be a JSON object: {path}')
    return payload


def _status_common(cfg: RunConfig) -> Dict[str, Any]:
    return {
        'method': cfg.method,
        'privacy': cfg.privacy,
        'run_id': cfg.run_id,
        'config_fingerprint': cfg.config_fingerprint,
        'fingerprint_schema_version': FINGERPRINT_SCHEMA_VERSION,
        'data_content_sha256': cfg.data_content_sha256,
        'base_model': cfg.base_model,
        'model_revision': cfg.model_revision,
        'resolved_model_revision': cfg.resolved_model_revision,
        'telemetry_mode': cfg.telemetry_mode,
        'non_private_telemetry': cfg.telemetry_mode == 'research_raw',
        'loss_definition': LOSS_DEFINITION,
        'config': cfg.__dict__,
    }


def completed_adapter_status(cfg: RunConfig) -> Optional[Dict[str, Any]]:
    if not adapter_is_complete(Path(cfg.output_dir)):
        return None
    status = _read_status(Path(cfg.output_dir))
    if not (
        status
        and status.get('state') == 'completed'
        and extract_status_fingerprint(status) == cfg.config_fingerprint
    ):
        return None
    return status


def completed_adapter_is_usable(cfg: RunConfig) -> bool:
    return completed_adapter_status(cfg) is not None

def train_prism_manual(cfg: RunConfig) -> Path:
    validate_existing_run_identity(cfg)
    clean_output_dir(Path(cfg.output_dir), cfg.force_train)
    # Evaluation caches and raw summaries belong to the adapter being
    # restarted. Keeping them would allow a newly trained adapter to inherit
    # stale metrics or diagnostics from the previous attempt.
    clean_output_dir(Path(cfg.result_dir), cfg.force_train)
    if completed_adapter_is_usable(cfg) and (not cfg.force_train):
        completed = _read_status(Path(cfg.output_dir))
        if completed is not None:
            save_status(
                Path(cfg.result_dir),
                **completed,
                adapter_output_dir=str(cfg.output_dir),
            )
        checkpoint_file(Path(cfg.output_dir)).unlink(missing_ok=True)
        print('[train] completed adapter found; skipping:', cfg.output_dir)
        return Path(cfg.output_dir)
    existing_status = _read_status(Path(cfg.output_dir))
    existing_checkpoint = checkpoint_file(Path(cfg.output_dir)).exists()
    if existing_status is not None and not cfg.force_train:
        state = existing_status.get('state')
        if state == 'completed':
            raise RuntimeError(
                f'Run status is completed but adapter files are incomplete in {cfg.output_dir}'
            )
        main_log_has_records = cfg.log_jsonl.exists() and cfg.log_jsonl.stat().st_size > 0
        pristine_initialization = state == 'initializing' and not main_log_has_records
        if not existing_checkpoint and not pristine_initialization:
            raise RuntimeError(
                f'Incomplete run state={state!r} has no resume checkpoint in {cfg.output_dir}; '
                're-run with --force_train to restart this exact configuration'
            )
        if pristine_initialization and not existing_checkpoint:
            print('[resume] retrying an interrupted initialization with no completed updates')
    save_status(
        Path(cfg.output_dir),
        state='initializing',
        **_status_common(cfg),
        runtime=collect_runtime_metadata(cfg.root),
    )
    if cfg.dataset == 'math10k':
        _preimport_prism_training_stack()
        set_seed(cfg.seed)
        model, tokenizer, device = _build_lora_model(cfg)
        _initialize_prism_factors(cfg, model)
        train_loader, train_ds = _make_loader(cfg, tokenizer)
        optimizer = _build_prism_optimizer(cfg, model)
    else:
        set_seed(cfg.seed)
        model, tokenizer, device = _build_lora_model(cfg)
        _initialize_prism_factors(cfg, model)
        optimizer = _build_prism_optimizer(cfg, model)
        train_loader, train_ds = _make_loader(cfg, tokenizer)
    checkpoint = _restore_if_possible(cfg, model, optimizer)
    privacy_engine = None
    noise_multiplier = None
    sample_rate = None
    expected_batch_size = None
    used_mode = 'none'
    if cfg.privacy == 'dp':
        from opacus import PrivacyEngine
        privacy_engine = PrivacyEngine(accountant=cfg.dp_accountant, secure_mode=bool(cfg.dp_secure_mode))
        (model, dp_opt, train_loader), used_mode, calibrated_noise, calibrated_sample_rate = _make_private(
            privacy_engine, model, optimizer, train_loader, cfg
        )
        noise_multiplier = float(getattr(dp_opt, 'noise_multiplier', calibrated_noise))
        sample_rate = float(calibrated_sample_rate)
        # Opacus stores ``int(N * q)`` on its DPOptimizer.  PRISM performs the
        # mechanism itself and therefore keeps the mathematically exact
        # expected Poisson batch size N*q; truncating 63.99 to 63 would alter
        # the advertised batch-64 update and its absolute noise scale.
        expected_batch_size = float(len(train_ds)) * sample_rate
        base_opt = getattr(dp_opt, 'original_optimizer', optimizer)
        optimizer = base_opt
        if hasattr(optimizer, 'configure_dp_noise'):
            optimizer.configure_dp_noise(
                generator=getattr(dp_opt, 'generator', None),
                secure_mode=bool(cfg.dp_secure_mode),
            )
        print(
            f'[DP] accountant={cfg.dp_accountant} grad_sample_mode={used_mode} '
            f'noise_multiplier={noise_multiplier:.8g} sample_rate={sample_rate:.8g} '
            f'expected_batch_size={expected_batch_size:.8g}'
        )
    _decouple_loader_worker_rng(train_loader, cfg.seed)
    start_step = _restore_runtime_state(
        cfg,
        checkpoint,
        train_loader=train_loader,
        optimizer=optimizer,
        privacy_engine=privacy_engine,
        noise_multiplier=noise_multiplier,
        sample_rate=sample_rate,
        expected_batch_size=expected_batch_size,
    )
    if checkpoint is not None:
        removed_main = truncate_jsonl_to_step(cfg.log_jsonl, start_step)
        removed_raw = truncate_jsonl_to_step(cfg.raw_log_jsonl, start_step)
        print(f'[resume] truncated log records after step {start_step}: main={removed_main}, raw={removed_raw}')
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    logger = JsonlLogger(cfg.log_jsonl)
    raw_logger = None
    if cfg.telemetry_mode == 'research_raw':
        if cfg.force_train and cfg.raw_log_jsonl.exists():
            cfg.raw_log_jsonl.unlink()
        raw_logger = JsonlLogger(cfg.raw_log_jsonl)
        warning_path = cfg.raw_log_jsonl.parent / 'README_NON_PRIVATE.txt'
        warning_path.write_text(
            'NON-PRIVATE RESEARCH TELEMETRY\n\n'
            'Files in this directory contain exact statistics derived from private '
            'training examples. The model update still uses the configured DP mechanism, '
            'but these telemetry files are not a DP release and must not be published.\n',
            encoding='utf-8',
        )
        print(f'[privacy warning] research_raw telemetry enabled: {cfg.raw_log_jsonl}')
    if cfg.resume and checkpoint is None:
        # A step-zero checkpoint makes preemption before the first periodic
        # checkpoint recoverable without an inexact or manual restart.
        save_resume_checkpoint(
            Path(cfg.output_dir),
            model,
            optimizer,
            0,
            extra=_checkpoint_extra(
                cfg,
                train_loader=train_loader,
                optimizer=optimizer,
                privacy_engine=privacy_engine,
                noise_multiplier=noise_multiplier,
                sample_rate=sample_rate,
                expected_batch_size=expected_batch_size,
            ),
        )
    save_status(
        Path(cfg.output_dir),
        state='running',
        **_status_common(cfg),
        privacy_accounting={
            'accountant': cfg.dp_accountant,
            'secure_mode': bool(cfg.dp_secure_mode),
            'grad_sample_mode': used_mode,
            'target_epsilon': cfg.dp_epsilon,
            'target_delta': cfg.dp_delta,
            'noise_multiplier': noise_multiplier,
            'sample_rate': sample_rate,
            'expected_batch_size': expected_batch_size,
            'planned_update_steps': cfg.total_update_steps,
        },
        runtime=collect_runtime_metadata(cfg.root),
    )
    model.train()
    update_steps = int(start_step)
    pbar = tqdm(total=int(cfg.total_update_steps), initial=update_steps, desc=f'{cfg.method}/{cfg.privacy} updates')
    while update_steps < int(cfg.total_update_steps):
        for batch in train_loader:
            if update_steps >= int(cfg.total_update_steps):
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            loss_sum = 0.0
            token_sum = 0
            seen = 0
            if cfg.privacy == 'dp':
                assert expected_batch_size is not None
                assert noise_multiplier is not None
                optimizer.dp_begin(
                    max_grad_norm=float(cfg.dp_max_grad_norm),
                    expected_batch_size=float(expected_batch_size),
                    noise_multiplier=float(noise_multiplier),
                )
                for micro in iter_microbatches(batch, int(cfg.micro_batch_size)):
                    optimizer.zero_grad(set_to_none=True)
                    per_example_loss, supervised_tokens = forward_causal_lm_per_example_loss(model, micro)
                    micro_bs = int(micro['input_ids'].shape[0])
                    if micro_bs <= 0:
                        continue
                    loss = per_example_loss.mean()
                    loss_sum += float(per_example_loss.detach().sum().cpu().item())
                    seen += micro_bs
                    token_sum += int(supervised_tokens.detach().sum().cpu().item())
                    # The helper defines one token-normalized loss per record;
                    # Opacus then correctly reconstructs grad_sample from this
                    # record mean, independent of physical microbatch size.
                    loss.backward()
                    accumulated = optimizer.dp_accumulate()
                    if int(accumulated) != micro_bs:
                        raise RuntimeError(
                            f'PRISM accumulated {accumulated} samples for a microbatch of {micro_bs}'
                        )
                total_seen = optimizer.dp_finalize(noise_multiplier=float(noise_multiplier))
                _step_accountant(privacy_engine, noise_multiplier=float(noise_multiplier), sample_rate=float(sample_rate))
                seen = max(seen, int(total_seen))
            else:
                optimizer.zero_grad(set_to_none=True)
                micros = list(iter_microbatches(batch, int(cfg.micro_batch_size)))
                logical_batch_size = int(batch['input_ids'].shape[0])
                for micro in micros:
                    per_example_loss, supervised_tokens = forward_causal_lm_per_example_loss(model, micro)
                    micro_bs = int(micro['input_ids'].shape[0])
                    if micro_bs <= 0:
                        continue
                    loss = per_example_loss.mean()
                    loss_sum += float(per_example_loss.detach().sum().cpu().item())
                    seen += micro_bs
                    token_sum += int(supervised_tokens.detach().sum().cpu().item())
                    (loss * (float(micro_bs) / max(1, logical_batch_size))).backward()
                optimizer.step()
            update_steps += 1
            pbar.update(1)
            rec = {
                'telemetry_schema_version': TELEMETRY_SCHEMA_VERSION,
                'run_id': cfg.run_id,
                'config_fingerprint': cfg.config_fingerprint,
                'method': cfg.method,
                'privacy': cfg.privacy,
                'telemetry_mode': cfg.telemetry_mode,
                'step': update_steps,
                'base_model': cfg.base_model,
                'model_revision': cfg.model_revision,
                'resolved_model_revision': cfg.resolved_model_revision,
                'dataset': cfg.dataset,
                'lora_r': int(cfg.lora_r),
                'lr': float(cfg.learning_rate),
                'loss_definition': LOSS_DEFINITION,
                'spectral_oversample': int(cfg.spectral_oversample),
                'spectral_n_iter': int(cfg.spectral_n_iter),
                'lift_gauge_fix': cfg.prism_lift_fix,
                'prism_floor_factor': float(cfg.prism_floor_factor),
                'prism_floor_mode': cfg.prism_floor_mode,
                'prism_debias_second_moment': bool(cfg.prism_debias_second_moment),
            }
            if cfg.privacy != 'dp':
                rec.update({'loss_mean': loss_sum / max(1, seen), 'tokens': token_sum, 'batch_n': seen})
            safe_optimizer_log = getattr(optimizer, 'last_log', {}) or {}
            rec.update(safe_optimizer_log)
            if privacy_engine is not None:
                try:
                    rec['eps_spent'] = float(privacy_engine.get_epsilon(float(cfg.dp_delta)))
                except Exception:
                    pass
            logger.log(rec)
            if raw_logger is not None:
                raw_rec = {
                    'NON_PRIVATE_TELEMETRY': True,
                    'telemetry_schema_version': TELEMETRY_SCHEMA_VERSION,
                    'run_id': cfg.run_id,
                    'config_fingerprint': cfg.config_fingerprint,
                    'method': cfg.method,
                    'privacy': cfg.privacy,
                    'dataset': cfg.dataset,
                    'base_model': cfg.base_model,
                    'model_revision': cfg.model_revision,
                    'resolved_model_revision': cfg.resolved_model_revision,
                    'loss_definition': LOSS_DEFINITION,
                    'step': update_steps,
                    'loss_mean': loss_sum / max(1, seen),
                    'tokens': token_sum,
                    'batch_n': seen,
                }
                # Copy every DP-safe release/context field so the raw analysis
                # file remains self-contained and does not require a fragile
                # positional join with train_log.jsonl.
                raw_rec.update(safe_optimizer_log)
                if 'eps_spent' in rec:
                    raw_rec['eps_spent'] = rec['eps_spent']
                raw_rec.update(getattr(optimizer, 'last_raw_log', {}) or {})
                raw_logger.log(raw_rec)
            if update_steps % 10 == 0 or update_steps == int(cfg.total_update_steps):
                msg = f"[{cfg.method}/{cfg.privacy}] step={update_steps}"
                # Exact DP-training loss belongs only in the explicitly marked
                # raw artifact, never in an otherwise easy-to-share terminal or
                # Slurm log.
                if cfg.privacy != 'dp':
                    msg += f" loss={loss_sum / max(1, seen):.4f}"
                if 'dp_clip_threshold' in rec:
                    msg += f" C={rec['dp_clip_threshold']:.5g}"
                if 'eps_spent' in rec:
                    msg += f" eps≈{rec['eps_spent']:.3f}"
                print(msg)
            if cfg.resume and update_steps % int(cfg.checkpoint_every) == 0 and (update_steps < int(cfg.total_update_steps)):
                save_resume_checkpoint(
                    Path(cfg.output_dir),
                    model,
                    optimizer,
                    update_steps,
                    extra=_checkpoint_extra(
                        cfg,
                        train_loader=train_loader,
                        optimizer=optimizer,
                        privacy_engine=privacy_engine,
                        noise_multiplier=noise_multiplier,
                        sample_rate=sample_rate,
                        expected_batch_size=expected_batch_size,
                    ),
                )
    pbar.close()
    logger.close()
    if raw_logger is not None:
        raw_logger.close()
    to_save = unwrap_for_save(model)
    rebase_stats = _rebase_prism_for_save(to_save)
    to_save.save_pretrained(str(cfg.output_dir))
    epsilon_spent = None
    if privacy_engine is not None:
        epsilon_spent = float(privacy_engine.get_epsilon(float(cfg.dp_delta)))
    saved_rank = int(rebase_stats.get('spectral_rebase_new_rank') or cfg.lora_r)
    completed_status = {
        **_status_common(cfg),
        'state': 'completed',
        'update_steps': update_steps,
        'dataset': cfg.dataset,
        'training_lora_r': int(cfg.lora_r),
        'saved_adapter_r': saved_rank,
        'spectral_rebase': rebase_stats,
        'privacy_accounting': {
            'accountant': cfg.dp_accountant,
            'secure_mode': bool(cfg.dp_secure_mode),
            'grad_sample_mode': used_mode,
            'target_epsilon': cfg.dp_epsilon,
            'target_delta': cfg.dp_delta,
            'epsilon_spent': epsilon_spent,
            'noise_multiplier': noise_multiplier,
            'sample_rate': sample_rate,
            'expected_batch_size': expected_batch_size,
            'completed_update_steps': update_steps,
        },
        'runtime': collect_runtime_metadata(cfg.root),
    }
    save_status(Path(cfg.output_dir), **completed_status)
    save_status(
        Path(cfg.result_dir),
        **completed_status,
        adapter_output_dir=str(cfg.output_dir),
    )
    if checkpoint_file(Path(cfg.output_dir)).exists():
        checkpoint_file(Path(cfg.output_dir)).unlink()
    cleanup_cuda()
    print('Saved adapter to:', cfg.output_dir)
    return Path(cfg.output_dir)

def train(cfg: RunConfig) -> Path:
    cfg.finalize()
    _ensure_paths(cfg)
    print('Run config:')
    for k, v in sorted(cfg.__dict__.items()):
        print(f'  {k}: {v}')
    return train_prism_manual(cfg)
