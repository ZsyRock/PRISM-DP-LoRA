from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import fields
from pathlib import Path

import pytest

import train_eval
from prism_cli.experiment_identity import (
    FINGERPRINT_SCHEMA_VERSION,
    git_worktree_identity,
)
from prism_cli.trainers import (
    CHECKPOINT_SCHEMA_VERSION,
    LOSS_DEFINITION,
    RunConfig,
    _status_common,
    _validate_checkpoint_identity,
    validate_existing_run_identity,
)


def _write_math_data(root: Path, text: str = '[{"instruction":"x","input":"","output":"y"}]') -> Path:
    path = root / 'LLM-Adapters' / 'ft-training_set' / 'math_10k.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')
    return path


def _config(root: Path, **overrides) -> RunConfig:
    _write_math_data(root)
    values = dict(dataset='math10k', method='slaclip', privacy='dp', root=root)
    values.update(overrides)
    return RunConfig(**values).finalize()


def _write_schedule(path: Path, thresholds: list[float], **metadata) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {'clip_thresholds': thresholds, **metadata}
    path.write_text(json.dumps(payload, sort_keys=True), encoding='utf-8')
    return path


def test_json_defaults_are_overridden_by_explicit_cli(tmp_path: Path) -> None:
    config_file = tmp_path / 'run.json'
    config_file.write_text(
        json.dumps(
            {
                'schema_version': 1,
                'dataset': 'math10k',
                'batch_size': 128,
                'initial_clip_threshold': 0.5,
                'dp_accountant': 'rdp',
                'target_modules': ['q_proj', 'v_proj'],
                'resume': False,
            }
        ),
        encoding='utf-8',
    )
    args = train_eval.parse_cli_args(
        [
            '--config',
            str(config_file),
            '--batch_size',
            '32',
            '--initial_clip_threshold',
            '2.0',
            '--dp_accountant',
            'prv',
            '--resume',
        ]
    )
    assert args.dataset == 'math10k'
    assert args.batch_size == 32
    assert args.dp_max_grad_norm == pytest.approx(2.0)
    assert args.dp_accountant == 'prv'
    assert args.target_modules == ['q_proj', 'v_proj']
    assert args.resume is True


def test_legacy_clip_option_is_the_same_destination() -> None:
    args = train_eval.parse_cli_args(['--dataset', 'math10k', '--dp_max_grad_norm', '3.5'])
    assert args.dp_max_grad_norm == pytest.approx(3.5)
    assert args.dp_accountant == 'prv'


def test_slaclip_q_cli_keeps_requested_clipped_fraction_unambiguous() -> None:
    args = train_eval.parse_cli_args(
        [
            '--dataset',
            'math10k',
            '--method',
            'slaclip_q',
            '--slaclip_target_clip_fraction',
            '0.99',
            '--slaclip_c_min',
            '0.1',
            '--slaclip_c_max',
            '15',
        ]
    )
    assert args.method == 'slaclip_q'
    assert args.slaclip_target_clip_fraction == pytest.approx(0.99)
    assert args.slaclip_c_min == pytest.approx(0.1)
    assert args.slaclip_c_max == pytest.approx(15.0)


def test_full_slaclip_cli_exposes_conditional_target_and_legacy_alias() -> None:
    canonical = train_eval.parse_cli_args(
        [
            '--dataset',
            'math10k',
            '--method',
            'slaclip',
            '--slaclip_target_non_small_clip_fraction',
            '0.975',
        ]
    )
    assert canonical.slaclip_target_non_small_clip_fraction == pytest.approx(
        0.975
    )
    assert not hasattr(canonical, 'slaclip_beta')

    legacy = train_eval.parse_cli_args(
        [
            '--dataset',
            'math10k',
            '--method',
            'slaclip',
            '--slaclip_beta',
            '0.95',
        ]
    )
    assert legacy.slaclip_target_non_small_clip_fraction == pytest.approx(0.95)
    assert not hasattr(legacy, 'slaclip_beta')

    with pytest.raises(SystemExit):
        train_eval.parse_cli_args(
            [
                '--dataset',
                'math10k',
                '--slaclip_target_non_small_clip_fraction',
                '0.95',
                '--slaclip_beta',
                '0.5',
            ]
        )


def test_full_target_cli_overrides_legacy_json_default(tmp_path: Path) -> None:
    config_file = tmp_path / 'legacy.json'
    config_file.write_text(
        json.dumps(
            {
                'schema_version': 1,
                'dataset': 'math10k',
                'slaclip_beta': 0.5,
            }
        ),
        encoding='utf-8',
    )
    args = train_eval.parse_cli_args(
        [
            '--config',
            str(config_file),
            '--slaclip_target_non_small_clip_fraction',
            '0.875',
        ]
    )
    assert args.slaclip_target_non_small_clip_fraction == pytest.approx(0.875)


def test_replay_cli_accepts_schedule_path() -> None:
    args = train_eval.parse_cli_args(
        [
            '--dataset',
            'math10k',
            '--method',
            'replay',
            '--clip_schedule_path',
            '/tmp/locked-schedule.json',
        ]
    )
    assert args.method == 'replay'
    assert args.clip_schedule_path == Path('/tmp/locked-schedule.json')


def test_replay_schedule_is_validated_hashed_and_path_independent(tmp_path: Path) -> None:
    schedule_a = _write_schedule(
        tmp_path / 'schedule-a.json',
        [1.0, 1.25, 0.75],
        source_campaign='development-v1',
        source_arm='slaclip-beta99',
        source_seed=42,
        selection_rule='mean C_t by step over locked development seeds',
    )
    schedule_b = tmp_path / 'other-machine' / 'schedule-b.json'
    schedule_b.parent.mkdir(parents=True)
    schedule_b.write_bytes(schedule_a.read_bytes())
    first = _config(
        tmp_path / 'host-a',
        method='replay',
        total_update_steps=3,
        dp_max_grad_norm=1.0,
        clip_schedule_path=schedule_a,
    )
    second = _config(
        tmp_path / 'host-b',
        method='replay',
        total_update_steps=3,
        dp_max_grad_norm=1.0,
        clip_schedule_path=schedule_b,
    )

    expected_sha = hashlib.sha256(schedule_a.read_bytes()).hexdigest()
    assert first.clip_schedule_sha256 == expected_sha
    assert first.clip_schedule_values == [1.0, 1.25, 0.75]
    assert first.clip_schedule_metadata['source_campaign'] == 'development-v1'
    assert first.clip_threshold_for_step(1) == pytest.approx(1.25)
    assert first.fingerprint_payload()['clip_schedule_sha256'] == expected_sha
    assert first.config_fingerprint == second.config_fingerprint
    assert 'clip_schedule_path' not in first.fingerprint_payload()
    status = _status_common(first)
    assert status['clip_schedule'] == {
        'sha256': expected_sha,
        'steps': 3,
        'metadata': first.clip_schedule_metadata,
        'privacy_accounting_scope': (
            'training_run_epsilon_is_conditional_on_the_locked_schedule'
        ),
        'schedule_privacy_class': 'UNSPECIFIED_PRECOMMITTED_SCHEDULE',
        'end_to_end_privacy_requires_schedule_provenance_and_composition': True,
    }


@pytest.mark.parametrize(
    ('thresholds', 'steps', 'initial_c', 'message'),
    [
        ([1.0, 2.0], 3, 1.0, 'length'),
        ([1.0, 0.0, 2.0], 3, 1.0, 'finite and positive'),
        ([1.0, float('inf'), 2.0], 3, 1.0, 'finite and positive'),
        ([0.5, 1.0, 2.0], 3, 1.0, 'first value'),
    ],
)
def test_replay_schedule_rejects_invalid_values(
    tmp_path: Path,
    thresholds: list[float],
    steps: int,
    initial_c: float,
    message: str,
) -> None:
    schedule = _write_schedule(tmp_path / 'bad-schedule.json', thresholds)
    _write_math_data(tmp_path)
    with pytest.raises(ValueError, match=message):
        RunConfig(
            dataset='math10k',
            method='replay',
            privacy='dp',
            root=tmp_path,
            total_update_steps=steps,
            dp_max_grad_norm=initial_c,
            clip_schedule_path=schedule,
        ).finalize()


def test_replay_requires_schedule_and_schedule_is_replay_only(tmp_path: Path) -> None:
    _write_math_data(tmp_path)
    with pytest.raises(ValueError, match='requires --clip_schedule_path'):
        RunConfig(
            dataset='math10k',
            method='replay',
            privacy='dp',
            root=tmp_path,
            total_update_steps=1,
        ).finalize()
    schedule = _write_schedule(tmp_path / 'schedule.json', [1.0])
    with pytest.raises(ValueError, match='only valid with method=replay'):
        RunConfig(
            dataset='math10k',
            method='baseline',
            privacy='dp',
            root=tmp_path,
            total_update_steps=1,
            clip_schedule_path=schedule,
        ).finalize()


def test_checkpoint_identity_records_and_checks_schedule_sha(tmp_path: Path) -> None:
    schedule = _write_schedule(tmp_path / 'schedule.json', [1.0, 1.1])
    cfg = _config(
        tmp_path,
        method='replay',
        total_update_steps=2,
        clip_schedule_path=schedule,
    )
    checkpoint = {
        'update_steps': 0,
        'extra': {
            'checkpoint_schema_version': CHECKPOINT_SCHEMA_VERSION,
            'config_fingerprint': cfg.config_fingerprint,
            'clip_schedule_sha256': cfg.clip_schedule_sha256,
            'loss_definition': LOSS_DEFINITION,
        },
    }
    assert _validate_checkpoint_identity(cfg, checkpoint)[
        'clip_schedule_sha256'
    ] == cfg.clip_schedule_sha256
    checkpoint['extra']['clip_schedule_sha256'] = '0' * 64
    with pytest.raises(RuntimeError, match='clip schedule SHA256 mismatch'):
        _validate_checkpoint_identity(cfg, checkpoint)


def test_checkpoint_identity_accepts_final_step_for_post_training_resume(
    tmp_path: Path,
) -> None:
    schedule = _write_schedule(tmp_path / 'schedule.json', [1.0, 1.1])
    cfg = _config(
        tmp_path,
        method='replay',
        total_update_steps=2,
        clip_schedule_path=schedule,
    )
    checkpoint = {
        'update_steps': cfg.total_update_steps,
        'extra': {
            'checkpoint_schema_version': CHECKPOINT_SCHEMA_VERSION,
            'config_fingerprint': cfg.config_fingerprint,
            'clip_schedule_sha256': cfg.clip_schedule_sha256,
            'loss_definition': LOSS_DEFINITION,
        },
    }

    assert _validate_checkpoint_identity(cfg, checkpoint) is checkpoint['extra']
    checkpoint['update_steps'] = cfg.total_update_steps + 1
    with pytest.raises(RuntimeError, match=r'outside \[0, 2\]'):
        _validate_checkpoint_identity(cfg, checkpoint)


def test_unknown_json_field_is_rejected(tmp_path: Path) -> None:
    config_file = tmp_path / 'bad.json'
    config_file.write_text(json.dumps({'dataset': 'math10k', 'surprise': 1}), encoding='utf-8')
    with pytest.raises(SystemExit):
        train_eval.parse_cli_args(['--config', str(config_file)])


def test_fingerprint_is_stable_across_workspace_paths(tmp_path: Path) -> None:
    first = _config(tmp_path / 'host_a')
    second = _config(tmp_path / 'host_b')
    assert first.data_content_sha256 == second.data_content_sha256
    assert first.config_fingerprint == second.config_fingerprint
    assert first.run_id == second.run_id
    assert Path(first.output_dir).name == Path(second.output_dir).name


def test_dirty_worktree_identity_hashes_change_contents(tmp_path: Path) -> None:
    subprocess.run(['git', 'init', '-q', str(tmp_path)], check=True)
    source = tmp_path / 'source.py'
    source.write_text('value = 1\n', encoding='utf-8')
    subprocess.run(['git', '-C', str(tmp_path), 'add', 'source.py'], check=True)
    subprocess.run(
        [
            'git', '-C', str(tmp_path),
            '-c', 'user.name=PRISM Test',
            '-c', 'user.email=prism-test@example.invalid',
            'commit', '-q', '-m', 'initial',
        ],
        check=True,
    )
    commit, dirty, digest = git_worktree_identity(tmp_path)
    assert commit
    assert dirty is False
    assert digest is None

    source.write_text('value = 2\n', encoding='utf-8')
    _, dirty, first_digest = git_worktree_identity(tmp_path)
    source.write_text('value = 3\n', encoding='utf-8')
    _, _, second_digest = git_worktree_identity(tmp_path)
    assert dirty is True
    assert first_digest and second_digest and first_digest != second_digest


def test_training_changes_get_distinct_hashed_directories(tmp_path: Path) -> None:
    first = _config(tmp_path, dp_max_grad_norm=0.5)
    second = _config(tmp_path, dp_max_grad_norm=2.0)
    third = _config(tmp_path, dp_max_grad_norm=2.0, slaclip_eta=0.2)
    fourth = _config(tmp_path, method='slaclip_q', slaclip_target_clip_fraction=0.99)
    fifth = _config(tmp_path, method='slaclip_q', slaclip_target_clip_fraction=0.98)
    sixth = _config(
        tmp_path,
        slaclip_target_non_small_clip_fraction=0.75,
    )
    seventh = _config(
        tmp_path,
        slaclip_target_non_small_clip_fraction=0.95,
    )
    assert first.config_fingerprint != second.config_fingerprint
    assert second.config_fingerprint != third.config_fingerprint
    assert fourth.config_fingerprint != fifth.config_fingerprint
    assert sixth.config_fingerprint != seventh.config_fingerprint
    assert first.output_dir != second.output_dir
    assert '_C0p5_' in first.run_id
    assert '_C2_' in second.run_id


def test_full_target_aliases_normalize_to_one_identity(tmp_path: Path) -> None:
    canonical = _config(
        tmp_path,
        slaclip_target_non_small_clip_fraction=0.8,
    )
    legacy = _config(tmp_path, slaclip_beta=0.8)
    assert canonical.slaclip_target_non_small_clip_fraction == pytest.approx(0.8)
    assert canonical.slaclip_beta == pytest.approx(0.8)
    assert canonical.fingerprint_payload() == legacy.fingerprint_payload()
    assert canonical.config_fingerprint == legacy.config_fingerprint
    assert canonical.run_id == legacy.run_id
    assert canonical.fingerprint_payload()['fingerprint_schema_version'] == 7
    assert FINGERPRINT_SCHEMA_VERSION == 7
    assert 'slaclip_beta' not in canonical.fingerprint_payload()


@pytest.mark.parametrize(
    ('overrides', 'message'),
    [
        ({'checkpoint_every': 0}, 'checkpoint_every'),
        ({'slaclip_num_slots': -1}, 'slaclip_num_slots'),
        ({'slaclip_eta': -0.1}, 'slaclip_eta'),
        ({'slaclip_beta': 1.1}, 'slaclip_beta'),
        (
            {'slaclip_target_non_small_clip_fraction': 1.1},
            'slaclip_target_non_small_clip_fraction',
        ),
        ({'slaclip_target_clip_fraction': 1.1}, 'slaclip_target_clip_fraction'),
        ({'dp_max_grad_norm': 0.05}, 'initial C'),
    ],
)
def test_slaclip_and_checkpoint_ranges_are_validated(tmp_path: Path, overrides: dict, message: str) -> None:
    _write_math_data(tmp_path)
    with pytest.raises(ValueError, match=message):
        RunConfig(dataset='math10k', method='slaclip', privacy='dp', root=tmp_path, **overrides).finalize()


def test_full_slaclip_target_alias_conflict_fails_closed(tmp_path: Path) -> None:
    _write_math_data(tmp_path)
    with pytest.raises(ValueError, match='conflicts'):
        RunConfig(
            dataset='math10k',
            method='slaclip',
            privacy='dp',
            root=tmp_path,
            slaclip_target_non_small_clip_fraction=0.75,
            slaclip_beta=0.5,
        ).finalize()


def test_full_and_q_target_names_cannot_be_silently_crossed(tmp_path: Path) -> None:
    _write_math_data(tmp_path)
    with pytest.raises(ValueError, match='only valid for method=slaclip_q'):
        RunConfig(
            dataset='math10k',
            method='slaclip',
            privacy='dp',
            root=tmp_path,
            slaclip_target_clip_fraction=0.99,
        ).finalize()
    with pytest.raises(ValueError, match='only valid for method=slaclip'):
        RunConfig(
            dataset='math10k',
            method='slaclip_q',
            privacy='dp',
            root=tmp_path,
            slaclip_target_non_small_clip_fraction=0.5,
        ).finalize()
    with pytest.raises(ValueError, match='slaclip_target_clip_fraction'):
        RunConfig(
            dataset='math10k',
            method='slaclip_q',
            privacy='dp',
            root=tmp_path,
            slaclip_target_clip_fraction=1.1,
        ).finalize()


def test_existing_status_with_other_fingerprint_is_rejected(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    Path(cfg.output_dir).mkdir(parents=True)
    (Path(cfg.output_dir) / 'run_status.json').write_text(
        json.dumps({'config_fingerprint': '0' * 64}),
        encoding='utf-8',
    )
    with pytest.raises(RuntimeError, match='fingerprint does not match'):
        validate_existing_run_identity(cfg)


def test_existing_matching_status_is_accepted(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    Path(cfg.output_dir).mkdir(parents=True)
    (Path(cfg.output_dir) / 'run_status.json').write_text(
        json.dumps({'config_fingerprint': cfg.config_fingerprint}),
        encoding='utf-8',
    )
    validate_existing_run_identity(cfg)


def test_unverifiable_legacy_adapter_is_not_silently_reused(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    Path(cfg.output_dir).mkdir(parents=True)
    (Path(cfg.output_dir) / 'adapter_config.json').write_text('{}', encoding='utf-8')
    (Path(cfg.output_dir) / 'adapter_model.safetensors').write_bytes(b'x')
    with pytest.raises(RuntimeError, match='no verifiable config fingerprint'):
        validate_existing_run_identity(cfg)


def test_all_runconfig_cli_values_construct_cleanly(tmp_path: Path) -> None:
    args = train_eval.parse_cli_args(['--dataset', 'math10k'])
    names = {item.name for item in fields(RunConfig) if item.init and item.name != 'root'}
    kwargs = {name: getattr(args, name) for name in names if hasattr(args, name)}
    cfg = RunConfig(root=tmp_path, **kwargs).finalize()
    assert cfg.dp_accountant == 'prv'


def test_pair_script_rejects_shared_output_directory() -> None:
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        ['bash', str(root / 'scripts' / 'run_math10k_pair.sh'), '--output_dir', '/tmp/shared'],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert 'paired runs manage' in result.stderr
