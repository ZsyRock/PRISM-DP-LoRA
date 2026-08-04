from __future__ import annotations
import hashlib
import json
import math
import os
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import torch
from tqdm.auto import tqdm
from .experiment_identity import FINGERPRINT_SCHEMA_VERSION, config_fingerprint, content_sha256, extract_status_fingerprint, git_worktree_identity, make_run_id
from .losses import forward_causal_lm_per_example_loss
from .modeling import (
    is_multimodal_causal_lm_config,
    load_base_model,
    resolve_text_lora_target_modules,
    resolved_model_revision,
)
from .math_answers import parse_reference_number
from .utils import JsonlLogger, adapter_is_complete, build_tokenizer, checkpoint_file, clean_output_dir, cleanup_cuda, collect_runtime_metadata, ensure_text_only_token_type_ids, freeze_vision_tower_params, generate_prompt, get_rng_state, iter_microbatches, llm_adapters_dir, load_resume_checkpoint_if_available, load_trainable_state_dict, save_resume_checkpoint, save_status, set_rng_state, set_seed, tokenize_prompt, truncate_jsonl_to_step, unwrap_for_save, write_json_atomic
from .validation_math import evaluate_public_math_numeric_exact
from .slaclip import resolve_full_slaclip_target


CHECKPOINT_SCHEMA_VERSION = 4
TELEMETRY_SCHEMA_VERSION = 5
LOSS_DEFINITION = 'per_record_mean_of_nonignored_next_token_losses'
VALIDATION_SPLIT_SCHEMA_VERSION = 2
PROMPT_GROUP_NORMALIZATION_ID = (
    'instruction_input_nfkc_casefold_whitespace_collapse_v1'
)


def _normalize_prompt_group_field(value: Any) -> str:
    """Return the conservative, versioned prompt-group representation."""
    return ' '.join(unicodedata.normalize('NFKC', str(value)).casefold().split())


def _prompt_group_payload(record: Dict[str, Any]) -> bytes:
    """Canonicalize the two prompt fields without introducing join ambiguity."""
    return json.dumps(
        {
            'instruction': _normalize_prompt_group_field(record.get('instruction', '')),
            'input': _normalize_prompt_group_field(record.get('input', '')),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(',', ':'),
    ).encode('utf-8')

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
    validation_seed: int = 1729
    validation_batch_size: int = 8
    validation_eval_interval: int = 0
    validation_generate_numeric: bool = False
    validation_num_beams: int = 1
    validation_max_new_tokens: int = 128
    validation_max_input_length: int = 512
    protocol_stage: str = 'pilot'
    validation_data_is_public: bool = False
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
    slaclip_target_non_small_clip_fraction: Optional[float] = None
    slaclip_beta: Optional[float] = None
    slaclip_target_clip_fraction: Optional[float] = None
    slaclip_c_min: float = 0.1
    slaclip_c_max: float = 50.0
    clip_schedule_path: Optional[Path] = None
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
    clip_schedule_sha256: Optional[str] = field(init=False, default=None)
    clip_schedule_values: Optional[List[float]] = field(init=False, default=None)
    clip_schedule_metadata: Dict[str, Any] = field(init=False, default_factory=dict)
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
            'validation_seed': int(self.validation_seed),
            'validation_batch_size': int(self.validation_batch_size),
            'validation_eval_interval': int(self.validation_eval_interval),
            'validation_generate_numeric': bool(self.validation_generate_numeric),
            'validation_num_beams': int(self.validation_num_beams),
            'validation_max_new_tokens': int(self.validation_max_new_tokens),
            'validation_max_input_length': int(self.validation_max_input_length),
            'validation_split_schema_version': VALIDATION_SPLIT_SCHEMA_VERSION,
            'protocol_stage': self.protocol_stage,
            'validation_data_is_public': bool(self.validation_data_is_public),
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
            'slaclip_target_non_small_clip_fraction': (
                float(self.slaclip_target_non_small_clip_fraction)
                if self.method == 'slaclip'
                else None
            ),
            'slaclip_target_clip_fraction': (
                float(self.slaclip_target_clip_fraction)
                if self.method == 'slaclip_q'
                else None
            ),
            'slaclip_c_min': float(self.slaclip_c_min),
            'slaclip_c_max': float(self.slaclip_c_max),
            # Machine-specific paths never define experiment identity.  For a
            # deterministic replay, the exact schedule file bytes do.
            'clip_schedule_sha256': self.clip_schedule_sha256,
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
        elif method in {'slaclip_q', 'slaclipq', 'slaclip_q_prism'}:
            self.method = 'slaclip_q'
        elif method in {'replay', 'schedule_replay', 'deterministic_replay'}:
            self.method = 'replay'
        else:
            raise ValueError('method must be baseline, slaclip, slaclip_q, or replay')
        self.privacy = self.privacy.lower().replace('_', '-')
        if self.privacy in {'non-dp', 'nondp', 'none'}:
            self.privacy = 'nondp'
        if self.privacy not in {'dp', 'nondp'}:
            raise ValueError('privacy must be dp or nondp')
        if self.method in {'slaclip', 'slaclip_q'} and self.privacy != 'dp':
            raise ValueError('SlaClip is a DP clipping controller; use --privacy dp')
        if self.method == 'replay' and self.privacy != 'dp':
            raise ValueError('replay controls a DP clipping threshold; use --privacy dp')
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
            self.val_set_size = int(self.val_set_size)
            self.eval_step = int(self.eval_step or 10)
            self.save_step = int(self.save_step or 20)
            self.data_path = self.data_path or adapters / 'ft-training_set' / 'math_10k.json'
        else:
            self.total_update_steps = 500 if self.total_update_steps is None else self.total_update_steps
            self.learning_rate = 0.0002 if self.learning_rate is None else self.learning_rate
            self.cutoff_len = 384 if self.cutoff_len is None else self.cutoff_len
            self.train_on_inputs = False if self.train_on_inputs is None else self.train_on_inputs
            self.val_set_size = int(self.val_set_size)
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
        if int(self.val_set_size) < 0:
            raise ValueError('val_set_size must be non-negative')
        if int(self.validation_seed) < 0:
            raise ValueError('validation_seed must be non-negative')
        if int(self.validation_batch_size) <= 0:
            raise ValueError('validation_batch_size must be positive')
        if int(self.validation_eval_interval) < 0:
            raise ValueError('validation_eval_interval must be non-negative')
        for name in (
            'validation_num_beams',
            'validation_max_new_tokens',
            'validation_max_input_length',
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f'{name} must be positive')
        self.protocol_stage = str(self.protocol_stage).lower().replace('-', '_')
        if self.protocol_stage not in {'pilot', 'selection', 'final'}:
            raise ValueError('protocol_stage must be pilot, selection, or final')
        has_validation_holdout = int(self.val_set_size) > 0
        if has_validation_holdout and bool(self.run_eval):
            raise ValueError(
                'validation-holdout runs forbid task test evaluation; set run_eval=false'
            )
        if has_validation_holdout and not bool(self.validation_data_is_public):
            raise ValueError(
                'validation-holdout runs require explicit validation_data_is_public=true; '
                'exact validation metrics are otherwise a non-DP data-dependent release'
            )
        if int(self.validation_eval_interval) > 0 and not has_validation_holdout:
            raise ValueError(
                'validation_eval_interval requires val_set_size > 0'
            )
        if bool(self.validation_generate_numeric):
            if self.dataset != 'math10k':
                raise ValueError(
                    'validation_generate_numeric is only defined for Math-10K'
                )
            if not has_validation_holdout:
                raise ValueError(
                    'validation_generate_numeric requires val_set_size > 0'
                )
        if self.protocol_stage == 'selection':
            if not has_validation_holdout:
                raise ValueError('selection stage requires val_set_size > 0')
        if self.protocol_stage == 'final' and int(self.val_set_size) != 0:
            raise ValueError('final stage requires val_set_size=0 and full-data retraining')
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
        full_target_was_supplied = (
            self.slaclip_target_non_small_clip_fraction is not None
            or self.slaclip_beta is not None
        )
        q_target_was_supplied = self.slaclip_target_clip_fraction is not None
        if self.method == 'slaclip' and q_target_was_supplied:
            raise ValueError(
                'slaclip_target_clip_fraction is only valid for method=slaclip_q; '
                'full SlaClip uses slaclip_target_non_small_clip_fraction'
            )
        if self.method == 'slaclip_q' and full_target_was_supplied:
            raise ValueError(
                'slaclip_target_non_small_clip_fraction/slaclip_beta is only '
                'valid for method=slaclip'
            )
        resolved_full_target = resolve_full_slaclip_target(
            self.slaclip_target_non_small_clip_fraction,
            beta=self.slaclip_beta,
        )
        resolved_q_target = (
            0.99
            if self.slaclip_target_clip_fraction is None
            else float(self.slaclip_target_clip_fraction)
        )
        if not 0.0 <= resolved_q_target <= 1.0:
            raise ValueError('slaclip_target_clip_fraction must be in [0, 1]')
        if self.method == 'slaclip':
            self.slaclip_target_non_small_clip_fraction = resolved_full_target
            # Preserve the old status/config key as a normalized alias so
            # existing campaign analyzers can still read new artifacts.
            self.slaclip_beta = resolved_full_target
            self.slaclip_target_clip_fraction = None
        elif self.method == 'slaclip_q':
            self.slaclip_target_non_small_clip_fraction = None
            self.slaclip_beta = None
            self.slaclip_target_clip_fraction = resolved_q_target
        else:
            # Baseline/replay configs may intentionally carry the target for a
            # paired adaptive arm, but do not invent inactive defaults.
            self.slaclip_target_non_small_clip_fraction = (
                resolved_full_target if full_target_was_supplied else None
            )
            self.slaclip_beta = self.slaclip_target_non_small_clip_fraction
            self.slaclip_target_clip_fraction = (
                resolved_q_target if q_target_was_supplied else None
            )
        if float(self.slaclip_c_min) <= 0 or float(self.slaclip_c_max) < float(self.slaclip_c_min):
            raise ValueError('require 0 < slaclip_c_min <= slaclip_c_max')
        if self.method in {'slaclip', 'slaclip_q'} and not (
            float(self.slaclip_c_min) <= float(self.dp_max_grad_norm) <= float(self.slaclip_c_max)
        ):
            raise ValueError('SlaClip initial C must satisfy slaclip_c_min <= C0 <= slaclip_c_max')
        if self.method == 'replay':
            if self.clip_schedule_path is None:
                raise ValueError('replay requires --clip_schedule_path')
            schedule_path = Path(self.clip_schedule_path).expanduser().resolve()
            try:
                with schedule_path.open('r', encoding='utf-8') as handle:
                    schedule_payload = json.load(handle)
            except FileNotFoundError as exc:
                raise ValueError(f'clip schedule does not exist: {schedule_path}') from exc
            except json.JSONDecodeError as exc:
                raise ValueError(f'invalid clip schedule JSON {schedule_path}: {exc}') from exc
            if isinstance(schedule_payload, list):
                schedule_values = schedule_payload
                schedule_metadata: Dict[str, Any] = {}
            elif isinstance(schedule_payload, dict):
                schedule_values = schedule_payload.get('clip_thresholds')
                schedule_metadata = {
                    str(key): value
                    for key, value in schedule_payload.items()
                    if key != 'clip_thresholds'
                }
            else:
                schedule_values = None
                schedule_metadata = {}
            if not isinstance(schedule_values, list):
                raise ValueError(
                    'clip schedule JSON must be an array or an object containing '
                    'a clip_thresholds array'
                )
            if len(schedule_values) != int(self.total_update_steps):
                raise ValueError(
                    f'clip schedule length={len(schedule_values)} must equal '
                    f'total_update_steps={int(self.total_update_steps)}'
                )
            normalized_schedule: List[float] = []
            for index, value in enumerate(schedule_values):
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError(
                        f'clip schedule value at index {index} must be a finite positive number'
                    )
                number = float(value)
                if not math.isfinite(number) or number <= 0:
                    raise ValueError(
                        f'clip schedule value at index {index} must be finite and positive'
                    )
                normalized_schedule.append(number)
            if not math.isclose(
                normalized_schedule[0],
                float(self.dp_max_grad_norm),
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    'clip schedule first value must equal initial C: '
                    f'schedule[0]={normalized_schedule[0]!r}, '
                    f'initial_C={float(self.dp_max_grad_norm)!r}'
                )
            self.clip_schedule_path = schedule_path
            self.clip_schedule_values = normalized_schedule
            self.clip_schedule_metadata = schedule_metadata
            self.clip_schedule_sha256 = content_sha256(schedule_path)
            if not self.clip_schedule_sha256:
                raise RuntimeError(f'could not hash clip schedule: {schedule_path}')
        elif self.clip_schedule_path is not None:
            raise ValueError('clip_schedule_path is only valid with method=replay')
        else:
            self.clip_schedule_values = None
            self.clip_schedule_metadata = {}
            self.clip_schedule_sha256 = None
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

    def clip_threshold_for_step(self, zero_based_step: int) -> float:
        """Return the pre-committed clipping threshold for one update."""

        step = int(zero_based_step)
        if step < 0 or step >= int(self.total_update_steps):
            raise IndexError(
                f'update step {step} is outside [0, {int(self.total_update_steps)})'
            )
        if self.method != 'replay':
            return float(self.dp_max_grad_norm)
        if self.clip_schedule_values is None:
            raise RuntimeError('replay schedule was not loaded during finalize()')
        return float(self.clip_schedule_values[step])

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

def _deterministic_holdout_indices(
    records,
    holdout_size: int,
    seed: int,
    *,
    dataset: str,
    require_numeric_reference: bool = False,
) -> tuple[list[int], list[int], Dict[str, Any]]:
    """Return a stable stratified prompt-group split and an audit manifest."""
    total_rows = int(len(records))
    holdout_size = int(holdout_size)
    seed = int(seed)
    if total_rows <= 0:
        raise ValueError('training dataset must contain at least one row')
    if holdout_size < 0 or holdout_size >= total_rows:
        raise ValueError(
            f'val_set_size must satisfy 0 <= val_set_size < {total_rows}, got {holdout_size}'
        )
    if require_numeric_reference and str(dataset) != 'math10k':
        raise ValueError(
            'numeric-reference validation eligibility is only defined for Math-10K'
        )
    groups_by_stratum: Dict[str, Dict[str, list[int]]] = {}
    record_hashes: Dict[int, str] = {}
    numeric_reference_by_index: Dict[int, bool] = {}
    for index in range(total_rows):
        record = dict(records[index])
        instruction = str(record.get('instruction', ''))
        stratum = (
            _normalize_prompt_group_field(instruction)
            if str(dataset) == 'glue8'
            else str(dataset)
        )
        prompt_payload = _prompt_group_payload(record)
        prompt_hash = hashlib.sha256(prompt_payload).hexdigest()
        groups_by_stratum.setdefault(stratum, {}).setdefault(prompt_hash, []).append(index)
        record_payload = json.dumps(
            record,
            ensure_ascii=False,
            sort_keys=True,
            separators=(',', ':'),
            default=str,
        ).encode('utf-8')
        record_hashes[index] = hashlib.sha256(record_payload).hexdigest()
        if require_numeric_reference:
            numeric_reference_by_index[index] = (
                parse_reference_number(record.get('answer')) is not None
            )

    prompt_group_count = sum(len(groups) for groups in groups_by_stratum.values())
    duplicate_prompt_groups = [
        indices
        for groups in groups_by_stratum.values()
        for indices in groups.values()
        if len(indices) > 1
    ]

    eligible_groups_by_stratum = groups_by_stratum
    ineligible_group_count = 0
    if require_numeric_reference:
        eligible_groups_by_stratum = {}
        for stratum, groups in groups_by_stratum.items():
            eligible_groups = {
                prompt_hash: indices
                for prompt_hash, indices in groups.items()
                if all(numeric_reference_by_index[index] for index in indices)
            }
            eligible_groups_by_stratum[stratum] = eligible_groups
            ineligible_group_count += len(groups) - len(eligible_groups)

    stratum_sizes = {
        key: sum(len(group) for group in groups.values())
        for key, groups in eligible_groups_by_stratum.items()
    }
    eligible_rows = sum(stratum_sizes.values())
    if holdout_size > eligible_rows:
        qualifier = ' numeric-reference-eligible' if require_numeric_reference else ''
        raise ValueError(
            f'val_set_size={holdout_size} exceeds the {eligible_rows}'
            f'{qualifier} rows available for validation'
        )
    raw_targets = {
        key: float(holdout_size) * float(size) / float(eligible_rows)
        for key, size in stratum_sizes.items()
    } if eligible_rows else {key: 0.0 for key in stratum_sizes}
    stratum_targets = {key: int(math.floor(value)) for key, value in raw_targets.items()}
    remainder = holdout_size - sum(stratum_targets.values())
    for key in sorted(
        stratum_targets,
        key=lambda value: (-(raw_targets[value] - stratum_targets[value]), value),
    )[:remainder]:
        stratum_targets[key] += 1

    validation_indices: list[int] = []
    for stratum in sorted(eligible_groups_by_stratum):
        target = stratum_targets[stratum]
        ranked_groups = sorted(
            eligible_groups_by_stratum[stratum].items(),
            key=lambda item: (
                hashlib.sha256(f'{seed}:{stratum}:{item[0]}'.encode('utf-8')).hexdigest(),
                item[0],
            ),
        )
        selected: list[int] = []
        remaining = target
        for _group_hash, indices in ranked_groups:
            if len(indices) <= remaining:
                selected.extend(indices)
                remaining -= len(indices)
            if remaining == 0:
                break
        if remaining != 0:
            raise ValueError(
                f'could not construct an exact prompt-group validation split for stratum={stratum!r}; '
                f'target={target}, missing={remaining}'
            )
        validation_indices.extend(selected)
    validation_indices = sorted(validation_indices)
    validation_set = set(validation_indices)
    train_indices = [index for index in range(total_rows) if index not in validation_set]
    index_bytes = json.dumps(validation_indices, separators=(',', ':')).encode('utf-8')
    train_hash_bytes = json.dumps(
        sorted(record_hashes[index] for index in train_indices),
        separators=(',', ':'),
    ).encode('utf-8')
    validation_hash_bytes = json.dumps(
        sorted(record_hashes[index] for index in validation_indices),
        separators=(',', ':'),
    ).encode('utf-8')
    metadata = {
        'schema_version': VALIDATION_SPLIT_SCHEMA_VERSION,
        'algorithm': 'sha256_ranked_stratified_normalized_prompt_group_v2',
        'seed': seed,
        'source_rows': total_rows,
        'train_rows': len(train_indices),
        'validation_rows': len(validation_indices),
        'requested_validation_rows': holdout_size,
        'stratum_targets': stratum_targets,
        'validation_indices': validation_indices,
        'validation_indices_sha256': hashlib.sha256(index_bytes).hexdigest(),
        'train_record_hashes_sha256': hashlib.sha256(train_hash_bytes).hexdigest(),
        'validation_record_hashes_sha256': hashlib.sha256(validation_hash_bytes).hexdigest(),
        'prompt_group_identity': {
            'id': PROMPT_GROUP_NORMALIZATION_ID,
            'fields': ['instruction', 'input'],
            'unicode_normalization': 'NFKC',
            'case_normalization': 'casefold',
            'whitespace_rule': 'split then join with one ASCII space',
            'canonical_serialization': 'sorted-key compact UTF-8 JSON object',
        },
        'normalized_prompt_groups': prompt_group_count,
        'normalized_duplicate_prompt_groups': len(duplicate_prompt_groups),
        'normalized_duplicate_prompt_rows': sum(
            len(indices) for indices in duplicate_prompt_groups
        ),
    }
    if require_numeric_reference:
        metadata.update({
            'algorithm': (
                'sha256_ranked_stratified_numeric_reference_normalized_prompt_group_v3'
            ),
            'validation_eligibility': (
                'all_records_in_prompt_group_have_finite_numeric_answer'
            ),
            'validation_eligible_rows': eligible_rows,
            'validation_ineligible_rows': total_rows - eligible_rows,
            'validation_eligible_prompt_groups': sum(
                len(groups) for groups in eligible_groups_by_stratum.values()
            ),
            'validation_ineligible_prompt_groups': ineligible_group_count,
        })
    return train_indices, validation_indices, metadata


def _make_loader(cfg: RunConfig, tokenizer):
    import transformers
    from datasets import load_dataset
    from torch.utils.data import DataLoader
    data = load_dataset('json', data_files=str(cfg.data_path)) if str(cfg.data_path).endswith('.json') else load_dataset(str(cfg.data_path))
    source_ds = data['train']
    train_indices, validation_indices, split_metadata = _deterministic_holdout_indices(
        source_ds,
        int(cfg.val_set_size),
        int(cfg.validation_seed),
        dataset=cfg.dataset,
        require_numeric_reference=bool(
            cfg.dataset == 'math10k'
            and getattr(cfg, 'validation_generate_numeric', False)
        ),
    )
    split_metadata['source_content_sha256'] = cfg.data_content_sha256
    split_metadata['protocol_stage'] = cfg.protocol_stage
    split_metadata['validation_data_is_public'] = bool(cfg.validation_data_is_public)
    split_payload = json.dumps(
        split_metadata,
        ensure_ascii=False,
        sort_keys=True,
        separators=(',', ':'),
    ).encode('utf-8')
    split_metadata['manifest_sha256'] = hashlib.sha256(split_payload).hexdigest()
    train_ds = source_ds.select(train_indices)
    validation_ds = source_ds.select(validation_indices) if validation_indices else None
    validation_records = [
        {**dict(source_ds[index]), '_source_index': int(index)}
        for index in validation_indices
    ]

    def map_fn(ex):
        return tokenize_prompt(tokenizer, ex, cutoff_len=int(cfg.cutoff_len), train_on_inputs=bool(cfg.train_on_inputs), base_model=cfg.base_model)

    def validation_map_fn(ex):
        # Hyperparameter selection is preregistered on response-only loss so a
        # long prompt cannot dominate the utility signal used to choose beta/C.
        return tokenize_prompt(
            tokenizer,
            ex,
            cutoff_len=int(cfg.cutoff_len),
            train_on_inputs=False,
            base_model=cfg.base_model,
        )
    train_ds = train_ds.map(map_fn, remove_columns=train_ds.column_names, desc='Tokenizing')
    if validation_ds is not None:
        validation_ds = validation_ds.map(
            validation_map_fn,
            remove_columns=validation_ds.column_names,
            desc='Tokenizing validation holdout',
        )
    collator = transformers.DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors='pt', padding=True)
    loader_kwargs = dict(dataset=train_ds, batch_size=int(cfg.batch_size), shuffle=True, drop_last=False, collate_fn=collator)
    generator = torch.Generator()
    generator.manual_seed(cfg.seed)
    loader_kwargs['generator'] = generator
    loader_kwargs['num_workers'] = 0
    loader = DataLoader(**loader_kwargs)
    validation_loader = None
    if validation_ds is not None:
        validation_generator = torch.Generator()
        validation_generator.manual_seed((int(cfg.validation_seed) + 0x0A11CE) % (2**63 - 1))
        validation_loader = DataLoader(
            dataset=validation_ds,
            batch_size=int(cfg.validation_batch_size),
            shuffle=False,
            drop_last=False,
            collate_fn=collator,
            num_workers=0,
            generator=validation_generator,
        )
    return (
        loader,
        train_ds,
        validation_loader,
        validation_records,
        split_metadata,
    )


@torch.no_grad()
def _evaluate_validation_loss(
    model,
    validation_loader,
    *,
    device: torch.device,
    needs_text_token_type_ids: bool,
) -> Dict[str, Any]:
    """Evaluate the public selection holdout without touching task test sets."""
    evaluation_model = unwrap_for_save(model)
    was_training = bool(evaluation_model.training)
    rng_state = get_rng_state()
    evaluation_model.eval()
    loss_sum = 0.0
    token_loss_sum = 0.0
    record_count = 0
    supervised_tokens = 0
    try:
        for batch in validation_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            batch = ensure_text_only_token_type_ids(
                batch,
                required=needs_text_token_type_ids,
            )
            per_example_loss, token_counts = forward_causal_lm_per_example_loss(
                evaluation_model,
                batch,
            )
            zero_token_records = int(token_counts.eq(0).sum().detach().cpu().item())
            if zero_token_records:
                raise RuntimeError(
                    'response-only validation produced '
                    f'{zero_token_records} record(s) with no supervised tokens; '
                    'increase cutoff_len or revise the validation tokenization protocol'
                )
            loss_sum += float(per_example_loss.detach().sum().cpu().item())
            token_loss_sum += float(
                (per_example_loss.detach() * token_counts.detach()).sum().cpu().item()
            )
            record_count += int(per_example_loss.numel())
            supervised_tokens += int(token_counts.detach().sum().cpu().item())
    finally:
        if was_training:
            evaluation_model.train()
        set_rng_state(rng_state)
    if record_count <= 0:
        raise RuntimeError('validation holdout produced no evaluable records')
    mean_loss = loss_sum / float(record_count)
    token_mean_loss = token_loss_sum / float(max(1, supervised_tokens))
    return {
        'metric_schema_version': 1,
        'selection_metric': 'response_only_mean_per_record_causal_lm_loss',
        'loss_definition': 'response_only_per_record_mean_of_nonignored_next_token_losses',
        'loss_mean': float(mean_loss),
        'token_mean_loss': float(token_mean_loss),
        'token_perplexity': float(math.exp(min(token_mean_loss, 50.0))),
        'records': int(record_count),
        'supervised_tokens': int(supervised_tokens),
    }


def _validation_curve_steps(path: Path) -> set[int]:
    """Read already committed curve steps so checkpoint resume is idempotent."""
    path = Path(path)
    if not path.exists():
        return set()
    steps: set[int] = set()
    with path.open('r', encoding='utf-8') as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f'invalid validation curve JSON at {path}:{line_number}'
                ) from exc
            step = record.get('step') if isinstance(record, dict) else None
            if isinstance(step, bool) or not isinstance(step, int) or step < 0:
                raise RuntimeError(
                    f'invalid validation curve step at {path}:{line_number}'
                )
            if step in steps:
                raise RuntimeError(
                    f'duplicate validation curve step={step} in {path}'
                )
            steps.add(int(step))
    return steps


def _required_curve_steps_before_resume(
    *,
    start_step: int,
    total_update_steps: int,
    interval: int,
) -> set[int]:
    """Return curve points that must already exist at a saved checkpoint.

    Periodic curve points are written before their same-step checkpoint.  The
    final checkpoint is deliberately written before endpoint evaluation, so the
    final curve point is optional and can be regenerated after resume.
    """

    start = int(start_step)
    total = int(total_update_steps)
    cadence = int(interval)
    if start < 0 or total <= 0 or start > total or cadence <= 0:
        raise ValueError('invalid validation-curve resume bounds')
    required = {0}
    upper = start if start < total else total - 1
    required.update(range(cadence, upper + 1, cadence))
    return required


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
        # Replay is deliberately the baseline Gaussian mechanism.  Its only
        # difference is that the trainer supplies a pre-committed C_t before
        # each update; it must not construct or release Slack coordinates.
        clipping_method='baseline' if cfg.method == 'replay' else cfg.method,
        slaclip_num_slots=int(cfg.slaclip_num_slots),
        slaclip_eta=float(cfg.slaclip_eta),
        slaclip_target_non_small_clip_fraction=(
            cfg.slaclip_target_non_small_clip_fraction
        ),
        slaclip_target_clip_fraction=cfg.slaclip_target_clip_fraction,
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
    saved_schedule_sha = extra.get('clip_schedule_sha256')
    if saved_schedule_sha != cfg.clip_schedule_sha256:
        raise RuntimeError(
            'checkpoint clip schedule SHA256 mismatch: '
            f'saved={saved_schedule_sha!r}, current={cfg.clip_schedule_sha256!r}'
        )
    saved_loss = extra.get('loss_definition')
    if saved_loss != LOSS_DEFINITION:
        raise RuntimeError(
            f'checkpoint loss definition mismatch: saved={saved_loss!r}, current={LOSS_DEFINITION!r}'
        )
    step = int(checkpoint.get('update_steps', 0))
    # A checkpoint at exactly ``total_update_steps`` is intentional: it is
    # written after the final private update but before public endpoint
    # generation and artifact finalization.  Accept it so a preempted run can
    # skip training and repeat only those deterministic post-training steps.
    if step < 0 or step > int(cfg.total_update_steps):
        raise RuntimeError(
            f'checkpoint update_steps={step} is outside '
            f'[0, {int(cfg.total_update_steps)}]'
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
        'clip_schedule_sha256': cfg.clip_schedule_sha256,
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
    status = {
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
    if cfg.method == 'replay':
        schedule_privacy_class = cfg.clip_schedule_metadata.get(
            'schedule_privacy_class',
            cfg.clip_schedule_metadata.get(
                'source_privacy_class',
                'UNSPECIFIED_PRECOMMITTED_SCHEDULE',
            ),
        )
        status['clip_schedule'] = {
            'sha256': cfg.clip_schedule_sha256,
            'steps': len(cfg.clip_schedule_values or ()),
            'metadata': cfg.clip_schedule_metadata,
            'privacy_accounting_scope': (
                'training_run_epsilon_is_conditional_on_the_locked_schedule'
            ),
            'schedule_privacy_class': schedule_privacy_class,
            'end_to_end_privacy_requires_schedule_provenance_and_composition': True,
        }
    return status


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
        (
            train_loader,
            train_ds,
            validation_loader,
            validation_records,
            split_metadata,
        ) = _make_loader(cfg, tokenizer)
        optimizer = _build_prism_optimizer(cfg, model)
    else:
        set_seed(cfg.seed)
        model, tokenizer, device = _build_lora_model(cfg)
        _initialize_prism_factors(cfg, model)
        optimizer = _build_prism_optimizer(cfg, model)
        (
            train_loader,
            train_ds,
            validation_loader,
            validation_records,
            split_metadata,
        ) = _make_loader(cfg, tokenizer)
    needs_text_token_type_ids = is_multimodal_causal_lm_config(model.config)
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
    validation_dir = Path(cfg.result_dir) / 'validation'
    validation_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(validation_dir / 'split_manifest.json', split_metadata)
    validation_curve_path = validation_dir / 'validation_curve.jsonl'
    if checkpoint is not None and validation_curve_path.exists():
        removed_curve = truncate_jsonl_to_step(
            validation_curve_path,
            start_step,
        )
        print(
            '[resume] truncated validation curve records after '
            f'step {start_step}: removed={removed_curve}'
        )
    validation_curve_logger = None
    validation_curve_steps: set[int] = set()
    if validation_loader is not None and int(cfg.validation_eval_interval) > 0:
        validation_curve_steps = _validation_curve_steps(validation_curve_path)
        if checkpoint is not None and start_step > 0:
            required_curve_steps = _required_curve_steps_before_resume(
                start_step=start_step,
                total_update_steps=int(cfg.total_update_steps),
                interval=int(cfg.validation_eval_interval),
            )
            missing_curve_steps = sorted(
                required_curve_steps - validation_curve_steps
            )
            if missing_curve_steps:
                raise RuntimeError(
                    'resume checkpoint cannot reconstruct missing historical '
                    'validation curve steps: '
                    f'{missing_curve_steps}'
                )
        validation_curve_logger = JsonlLogger(validation_curve_path)
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
            'scope': (
                'conditional_on_locked_clip_schedule'
                if cfg.method == 'replay'
                else 'single_training_run'
            ),
            'target_epsilon': cfg.dp_epsilon,
            'target_delta': cfg.dp_delta,
            'noise_multiplier': noise_multiplier,
            'sample_rate': sample_rate,
            'expected_batch_size': expected_batch_size,
            'planned_update_steps': cfg.total_update_steps,
        },
        data_split=split_metadata,
        runtime=collect_runtime_metadata(cfg.root),
    )

    def record_validation_curve(step: int) -> Optional[Dict[str, Any]]:
        if validation_curve_logger is None or validation_loader is None:
            return None
        step = int(step)
        if step in validation_curve_steps:
            return None
        metrics = _evaluate_validation_loss(
            model,
            validation_loader,
            device=device,
            needs_text_token_type_ids=needs_text_token_type_ids,
        )
        curve_record = {
            **metrics,
            'NON_PRIVATE_SELECTION_METRIC': True,
            'PUBLIC_VALIDATION_DATA': True,
            'run_id': cfg.run_id,
            'config_fingerprint': cfg.config_fingerprint,
            'step': step,
            'planned_update_steps': int(cfg.total_update_steps),
            'validation_indices_sha256': split_metadata[
                'validation_indices_sha256'
            ],
            'validation_record_hashes_sha256': split_metadata[
                'validation_record_hashes_sha256'
            ],
            'manifest_sha256': split_metadata['manifest_sha256'],
        }
        validation_curve_logger.log(curve_record)
        validation_curve_steps.add(step)
        print(
            '[validation curve] '
            f"step={step} records={metrics['records']} "
            f"loss={metrics['loss_mean']:.8f}"
        )
        return metrics

    record_validation_curve(0)
    model.train()
    update_steps = int(start_step)
    pbar = tqdm(total=int(cfg.total_update_steps), initial=update_steps, desc=f'{cfg.method}/{cfg.privacy} updates')
    while update_steps < int(cfg.total_update_steps):
        for batch in train_loader:
            if update_steps >= int(cfg.total_update_steps):
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            batch = ensure_text_only_token_type_ids(
                batch,
                required=needs_text_token_type_ids,
            )
            loss_sum = 0.0
            token_sum = 0
            seen = 0
            if cfg.privacy == 'dp':
                assert expected_batch_size is not None
                assert noise_multiplier is not None
                scheduled_step = update_steps
                step_clip_threshold = cfg.clip_threshold_for_step(scheduled_step)
                optimizer.dp_begin(
                    max_grad_norm=step_clip_threshold,
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
            safe_optimizer_log = dict(getattr(optimizer, 'last_log', {}) or {})
            if cfg.method == 'replay':
                if cfg.clip_schedule_values is None:
                    raise RuntimeError('replay schedule disappeared after configuration')
                next_index = min(scheduled_step + 1, len(cfg.clip_schedule_values) - 1)
                # For replay, C_{t+1} comes from the immutable schedule, not a
                # private-data-dependent controller.
                safe_optimizer_log['dp_next_clip_threshold'] = float(
                    cfg.clip_schedule_values[next_index]
                )
                safe_optimizer_log['replay_schedule_index'] = int(scheduled_step)
                safe_optimizer_log['replay_clip_schedule_sha256'] = str(
                    cfg.clip_schedule_sha256
                )
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
            if (
                validation_curve_logger is not None
                and update_steps < int(cfg.total_update_steps)
                and update_steps % int(cfg.validation_eval_interval) == 0
            ):
                record_validation_curve(update_steps)
            # Keep the final pre-evaluation state as well.  If endpoint
            # generation or artifact writing is preempted, resume can repeat
            # only validation instead of the last training interval.
            if cfg.resume and (
                update_steps % int(cfg.checkpoint_every) == 0
                or update_steps == int(cfg.total_update_steps)
            ):
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
    validation_metrics = None
    if validation_loader is not None:
        validation_metrics = record_validation_curve(
            int(cfg.total_update_steps)
        )
        if validation_metrics is None:
            validation_metrics = _evaluate_validation_loss(
                model,
                validation_loader,
                device=device,
                needs_text_token_type_ids=needs_text_token_type_ids,
            )
        if bool(cfg.validation_generate_numeric):
            numeric_metrics = evaluate_public_math_numeric_exact(
                model,
                tokenizer,
                validation_records,
                device=device,
                batch_size=int(cfg.validation_batch_size),
                max_input_length=int(cfg.validation_max_input_length),
                max_new_tokens=int(cfg.validation_max_new_tokens),
                num_beams=int(cfg.validation_num_beams),
                needs_text_token_type_ids=needs_text_token_type_ids,
                predictions_path=(
                    validation_dir / 'public_numeric_predictions.json'
                ),
            )
            if int(numeric_metrics['records']) != int(
                validation_metrics['records']
            ):
                raise RuntimeError(
                    'numeric generation and response-loss validation row '
                    'counts do not match'
                )
            validation_metrics.update(numeric_metrics)
            validation_metrics['metric_schema_version'] = 2
        validation_metrics.update(split_metadata)
        validation_metrics['NON_PRIVATE_SELECTION_METRIC'] = True
        validation_metrics['PUBLIC_VALIDATION_DATA'] = True
        validation_path = validation_dir / 'validation_metrics.json'
        write_json_atomic(validation_path, validation_metrics)
        accuracy_message = ''
        if 'numeric_exact_accuracy' in validation_metrics:
            accuracy_message = (
                f" numeric_exact={validation_metrics['numeric_exact_accuracy']:.6f}"
                f" parse_failures={validation_metrics['numeric_parse_failures']}"
            )
        print(
            '[validation] '
            f"records={validation_metrics['records']} "
            f"loss={validation_metrics['loss_mean']:.8f} "
            f"{accuracy_message} "
            f"split_sha256={validation_metrics['validation_indices_sha256']}"
        )
    if validation_curve_logger is not None:
        validation_curve_logger.close()
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
        'data_split': split_metadata,
        'validation': validation_metrics,
        'privacy_accounting': {
            'accountant': cfg.dp_accountant,
            'secure_mode': bool(cfg.dp_secure_mode),
            'grad_sample_mode': used_mode,
            'scope': (
                'conditional_on_locked_clip_schedule'
                if cfg.method == 'replay'
                else 'single_training_run'
            ),
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
