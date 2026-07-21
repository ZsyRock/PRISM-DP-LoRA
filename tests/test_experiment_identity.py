from __future__ import annotations

import json
import subprocess
from dataclasses import fields
from pathlib import Path

import pytest

import train_eval
from prism_cli.experiment_identity import git_worktree_identity
from prism_cli.trainers import RunConfig, validate_existing_run_identity


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
    assert first.config_fingerprint != second.config_fingerprint
    assert second.config_fingerprint != third.config_fingerprint
    assert first.output_dir != second.output_dir
    assert '_C0p5_' in first.run_id
    assert '_C2_' in second.run_id


@pytest.mark.parametrize(
    ('overrides', 'message'),
    [
        ({'checkpoint_every': 0}, 'checkpoint_every'),
        ({'slaclip_num_slots': -1}, 'slaclip_num_slots'),
        ({'slaclip_eta': -0.1}, 'slaclip_eta'),
        ({'slaclip_beta': 1.1}, 'slaclip_beta'),
        ({'dp_max_grad_norm': 0.05}, 'initial C'),
    ],
)
def test_slaclip_and_checkpoint_ranges_are_validated(tmp_path: Path, overrides: dict, message: str) -> None:
    _write_math_data(tmp_path)
    with pytest.raises(ValueError, match=message):
        RunConfig(dataset='math10k', method='slaclip', privacy='dp', root=tmp_path, **overrides).finalize()


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
