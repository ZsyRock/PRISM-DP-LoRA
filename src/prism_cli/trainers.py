from __future__ import annotations
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch
from tqdm.auto import tqdm
from .utils import JsonlLogger, adapter_is_complete, build_tokenizer, checkpoint_file, clean_output_dir, cleanup_cuda, collect_runtime_metadata, freeze_vision_tower_params, generate_prompt, iter_microbatches, llm_adapters_dir, load_resume_checkpoint_if_available, load_trainable_state_dict, save_resume_checkpoint, save_status, set_rng_state, set_seed, tag_text, tokenize_prompt, unwrap_for_save

@dataclass
class RunConfig:
    dataset: str
    method: str
    privacy: str
    root: Path
    base_model: str = 'google/gemma-3-4b-pt'
    seed: int = 42
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
    dp_accountant: str = 'rdp'
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

    def finalize(self) -> 'RunConfig':
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
        if self.batch_size <= 0 or self.micro_batch_size <= 0:
            raise ValueError('batch_size and micro_batch_size must be positive')
        if self.dp_max_grad_norm <= 0:
            raise ValueError('dp_max_grad_norm must be positive')
        if self.raw_hist_bins <= 0:
            raise ValueError('raw_hist_bins must be positive')
        adapters = llm_adapters_dir(self.root)
        if self.dataset == 'math10k':
            self.total_update_steps = self.total_update_steps or 300
            self.learning_rate = self.learning_rate or 0.0003
            self.cutoff_len = self.cutoff_len or 256
            self.train_on_inputs = True if self.train_on_inputs is None else self.train_on_inputs
            self.val_set_size = int(self.val_set_size or 120)
            self.eval_step = int(self.eval_step or 10)
            self.save_step = int(self.save_step or 20)
            self.data_path = self.data_path or adapters / 'ft-training_set' / 'math_10k.json'
        else:
            self.total_update_steps = self.total_update_steps or 500
            self.learning_rate = self.learning_rate or 0.0002
            self.cutoff_len = self.cutoff_len or 384
            self.train_on_inputs = False if self.train_on_inputs is None else self.train_on_inputs
            self.val_set_size = int(self.val_set_size or 1000)
            self.eval_step = int(self.eval_step or 100)
            self.save_step = int(self.save_step or 200)
            self.data_path = self.data_path or adapters / 'ft-training_set' / 'glue8_1250.json'
        base_tag = tag_text(self.base_model)
        if self.privacy == 'dp':
            run_tag = f'{self.dataset}_{self.method}_dp_eps{self.dp_epsilon}_seed{self.seed}_r{self.lora_r}_{base_tag}'
        else:
            run_tag = f'{self.dataset}_{self.method}_nondp_seed{self.seed}_r{self.lora_r}_{base_tag}'
        self.output_dir = self.output_dir or adapters / 'trained_models' / run_tag
        self.result_dir = self.result_dir or adapters / 'experiment' / run_tag
        return self

    @property
    def log_jsonl(self) -> Path:
        return Path(self.output_dir) / 'train_log.jsonl'

    @property
    def raw_log_jsonl(self) -> Path:
        return Path(self.result_dir) / 'research_raw' / 'NON_PRIVATE_train_log.jsonl'

def _preimport_prism_training_stack() -> None:
    import datasets
    import opacus
    import peft
    import transformers
    from torch.utils.data import DataLoader
    from .optim import prism as _prism_module

def _ensure_paths(cfg: RunConfig) -> None:
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
    from transformers import AutoModelForCausalLM
    device, dtype, device_map = _device_and_dtype()
    tokenizer = build_tokenizer(cfg.base_model)
    model = AutoModelForCausalLM.from_pretrained(cfg.base_model, torch_dtype=dtype, device_map=device_map, trust_remote_code=True)
    model.config.use_cache = False
    lora_cfg = LoraConfig(r=int(cfg.lora_r), lora_alpha=int(cfg.lora_alpha), target_modules=list(cfg.target_modules), lora_dropout=float(cfg.lora_dropout), bias='none', task_type='CAUSAL_LM')
    model = get_peft_model(model, lora_cfg)
    freeze_vision_tower_params(model)
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
    try:
        return spectral_rebase_adapter_inplace(model, adapter_name='default', verbose=True)
    except Exception as e:
        print('[warn] spectral rebase for saving failed:', repr(e))
        return {'spectral_rebase_modules': 0.0, 'spectral_rebase_new_rank': 0.0}

def _restore_if_possible(cfg: RunConfig, model, optimizer) -> int:
    if not cfg.resume or cfg.force_train:
        return 0
    ckpt = load_resume_checkpoint_if_available(Path(cfg.output_dir))
    if ckpt is None:
        return 0
    load_trainable_state_dict(model, ckpt.get('trainable_state', {}))
    if optimizer is not None and ckpt.get('optimizer_state') is not None:
        try:
            if hasattr(optimizer, '_ensure_state'):
                optimizer._ensure_state()
            optimizer.load_state_dict(ckpt['optimizer_state'])
        except Exception as e:
            raise RuntimeError(f'optimizer state could not be restored: {e!r}') from e
    set_rng_state(ckpt.get('rng_state'))
    step = int(ckpt.get('update_steps', 0))
    print(f'[resume] starting at update step {step}')
    return step

def train_prism_manual(cfg: RunConfig) -> Path:
    clean_output_dir(Path(cfg.output_dir), cfg.force_train)
    if adapter_is_complete(Path(cfg.output_dir)) and (not cfg.force_train):
        print('[train] completed adapter found; skipping:', cfg.output_dir)
        return Path(cfg.output_dir)
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
    start_step = _restore_if_possible(cfg, model, optimizer)
    privacy_engine = None
    noise_multiplier = None
    sample_rate = None
    expected_batch_size = None
    if cfg.privacy == 'dp':
        from opacus import PrivacyEngine
        privacy_engine = PrivacyEngine(accountant=cfg.dp_accountant)
        (model, dp_opt, train_loader), used_mode, calibrated_noise, calibrated_sample_rate = _make_private(
            privacy_engine, model, optimizer, train_loader, cfg
        )
        noise_multiplier = float(getattr(dp_opt, 'noise_multiplier', calibrated_noise))
        sample_rate = float(calibrated_sample_rate)
        expected_batch_size = float(
            getattr(dp_opt, 'expected_batch_size', float(len(train_ds)) * sample_rate)
        )
        base_opt = getattr(dp_opt, 'original_optimizer', optimizer)
        optimizer = base_opt
        for _ in range(start_step):
            _step_accountant(privacy_engine, noise_multiplier, float(sample_rate))
        print(
            f'[DP] accountant={cfg.dp_accountant} grad_sample_mode={used_mode} '
            f'noise_multiplier={noise_multiplier:.8g} sample_rate={sample_rate:.8g} '
            f'expected_batch_size={expected_batch_size:.8g}'
        )
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
    save_status(
        Path(cfg.output_dir),
        state='running',
        method=cfg.method,
        privacy=cfg.privacy,
        telemetry_mode=cfg.telemetry_mode,
        non_private_telemetry=cfg.telemetry_mode == 'research_raw',
        config=cfg.__dict__,
        privacy_accounting={
            'accountant': cfg.dp_accountant,
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
                    out = model(**micro)
                    loss = out.loss if hasattr(out, 'loss') else out[0]
                    micro_bs = int(micro['input_ids'].shape[0])
                    loss_sum += float(loss.detach().cpu().item()) * micro_bs
                    seen += micro_bs
                    token_sum += int(micro.get('attention_mask', torch.ones_like(micro['input_ids'])).detach().sum().item())
                    # Opacus reconstructs per-example gradients under mean reduction.
                    # Multiplying by micro_bs here would change clipping with microbatch size.
                    loss.backward()
                    optimizer.dp_accumulate()
                total_seen = optimizer.dp_finalize(noise_multiplier=float(noise_multiplier))
                _step_accountant(privacy_engine, noise_multiplier=float(noise_multiplier), sample_rate=float(sample_rate))
                seen = max(seen, int(total_seen))
            else:
                optimizer.zero_grad(set_to_none=True)
                micros = list(iter_microbatches(batch, int(cfg.micro_batch_size)))
                logical_batch_size = int(batch['input_ids'].shape[0])
                for micro in micros:
                    out = model(**micro)
                    loss = out.loss if hasattr(out, 'loss') else out[0]
                    micro_bs = int(micro['input_ids'].shape[0])
                    loss_sum += float(loss.detach().cpu().item()) * micro_bs
                    seen += micro_bs
                    token_sum += int(micro.get('attention_mask', torch.ones_like(micro['input_ids'])).detach().sum().item())
                    (loss * (float(micro_bs) / max(1, logical_batch_size))).backward()
                optimizer.step()
            update_steps += 1
            pbar.update(1)
            rec = {'method': cfg.method, 'privacy': cfg.privacy, 'telemetry_mode': cfg.telemetry_mode, 'step': update_steps, 'base_model': cfg.base_model, 'dataset': cfg.dataset, 'lora_r': int(cfg.lora_r), 'lr': float(cfg.learning_rate), 'spectral_oversample': int(cfg.spectral_oversample), 'spectral_n_iter': int(cfg.spectral_n_iter), 'lift_gauge_fix': cfg.prism_lift_fix, 'prism_floor_factor': float(cfg.prism_floor_factor), 'prism_floor_mode': cfg.prism_floor_mode, 'prism_debias_second_moment': bool(cfg.prism_debias_second_moment)}
            if cfg.privacy != 'dp':
                rec.update({'loss_mean': loss_sum / max(1, seen), 'tokens': token_sum, 'batch_n': seen})
            rec.update(getattr(optimizer, 'last_log', {}) or {})
            if privacy_engine is not None:
                try:
                    rec['eps_spent'] = float(privacy_engine.get_epsilon(float(cfg.dp_delta)))
                except Exception:
                    pass
            logger.log(rec)
            if raw_logger is not None:
                raw_rec = {
                    'NON_PRIVATE_TELEMETRY': True,
                    'method': cfg.method,
                    'privacy': cfg.privacy,
                    'step': update_steps,
                    'loss_mean': loss_sum / max(1, seen),
                    'tokens': token_sum,
                    'batch_n': seen,
                }
                raw_rec.update(getattr(optimizer, 'last_raw_log', {}) or {})
                raw_logger.log(raw_rec)
            if update_steps % 10 == 0 or update_steps == int(cfg.total_update_steps):
                msg = f"[{cfg.method}/{cfg.privacy}] step={update_steps}"
                if cfg.privacy != 'dp' or cfg.telemetry_mode == 'research_raw':
                    msg += f" loss={loss_sum / max(1, seen):.4f}"
                if 'dp_clip_threshold' in rec:
                    msg += f" C={rec['dp_clip_threshold']:.5g}"
                if 'eps_spent' in rec:
                    msg += f" eps≈{rec['eps_spent']:.3f}"
                print(msg)
            if cfg.resume and update_steps % int(cfg.checkpoint_every) == 0 and (update_steps < int(cfg.total_update_steps)):
                save_resume_checkpoint(Path(cfg.output_dir), model, optimizer, update_steps)
    pbar.close()
    logger.close()
    if raw_logger is not None:
        raw_logger.close()
    to_save = unwrap_for_save(model)
    rebase_stats = _rebase_prism_for_save(to_save)
    to_save.save_pretrained(str(cfg.output_dir))
    if checkpoint_file(Path(cfg.output_dir)).exists():
        checkpoint_file(Path(cfg.output_dir)).unlink()
    epsilon_spent = None
    if privacy_engine is not None:
        epsilon_spent = float(privacy_engine.get_epsilon(float(cfg.dp_delta)))
    saved_rank = int(rebase_stats.get('spectral_rebase_new_rank') or cfg.lora_r)
    save_status(Path(cfg.output_dir), state='completed', method=cfg.method, privacy=cfg.privacy, telemetry_mode=cfg.telemetry_mode, non_private_telemetry=cfg.telemetry_mode == 'research_raw', update_steps=update_steps, dataset=cfg.dataset, base_model=cfg.base_model, training_lora_r=int(cfg.lora_r), saved_adapter_r=saved_rank, spectral_rebase=rebase_stats, config=cfg.__dict__, privacy_accounting={'accountant': cfg.dp_accountant, 'target_epsilon': cfg.dp_epsilon, 'target_delta': cfg.dp_delta, 'epsilon_spent': epsilon_spent, 'noise_multiplier': noise_multiplier, 'sample_rate': sample_rate, 'expected_batch_size': expected_batch_size, 'completed_update_steps': update_steps}, runtime=collect_runtime_metadata(cfg.root))
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
