from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import DataLoader

import train_eval
import prism_cli.trainers as trainers
from prism_cli.losses import causal_lm_per_example_loss
from prism_cli.trainers import (
    RunConfig,
    _deterministic_holdout_indices,
    _evaluate_validation_loss,
    _make_loader,
    _required_curve_steps_before_resume,
    _validation_curve_steps,
)


def _math_records(count: int) -> list[dict[str, str]]:
    return [
        {
            'instruction': f'question-{index}',
            'input': '',
            'output': f'answer-{index}',
        }
        for index in range(count)
    ]


class _TinyValidationLM(torch.nn.Module):
    def __init__(self, vocab_size: int = 11, hidden_size: int = 7) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, hidden_size)
        self.dropout = torch.nn.Dropout(p=0.5)
        self.projection = torch.nn.Linear(hidden_size, vocab_size, bias=False)
        self.forward_training_modes: list[bool] = []

    def forward(self, input_ids: torch.Tensor, attention_mask=None):
        del attention_mask
        self.forward_training_modes.append(bool(self.training))
        hidden = self.dropout(self.embedding(input_ids))
        return SimpleNamespace(logits=self.projection(hidden))


def test_holdout_indices_are_stable_disjoint_and_hashed() -> None:
    records = _math_records(100)
    first_train, first_validation, first_metadata = _deterministic_holdout_indices(
        records,
        17,
        1729,
        dataset='math10k',
    )
    second_train, second_validation, second_metadata = _deterministic_holdout_indices(
        records,
        17,
        1729,
        dataset='math10k',
    )
    assert first_train == second_train
    assert first_validation == second_validation
    assert first_metadata == second_metadata
    assert len(first_train) == 83
    assert len(first_validation) == 17
    assert set(first_train).isdisjoint(first_validation)
    assert sorted(first_train + first_validation) == list(range(100))
    assert first_metadata['algorithm'] == 'sha256_ranked_stratified_prompt_group_v1'
    assert first_metadata['validation_indices'] == first_validation
    for key in (
        'validation_indices_sha256',
        'train_record_hashes_sha256',
        'validation_record_hashes_sha256',
    ):
        assert len(first_metadata[key]) == 64


def test_holdout_seed_size_and_record_content_change_split_identity() -> None:
    records = _math_records(100)
    _, first, first_metadata = _deterministic_holdout_indices(
        records, 17, 1729, dataset='math10k'
    )
    _, second, second_metadata = _deterministic_holdout_indices(
        records, 17, 1730, dataset='math10k'
    )
    _, third, third_metadata = _deterministic_holdout_indices(
        records, 18, 1729, dataset='math10k'
    )
    changed_records = [{**record, 'output': record['output'] + '-changed'} for record in records]
    _, changed, changed_metadata = _deterministic_holdout_indices(
        changed_records, 17, 1729, dataset='math10k'
    )

    assert first != second
    assert first_metadata['validation_indices_sha256'] != second_metadata['validation_indices_sha256']
    assert first_metadata['validation_indices_sha256'] != third_metadata['validation_indices_sha256']
    assert len(third) == 18
    # Output content does not alter prompt-ranked membership, but it must alter
    # both auditable record-membership digests.
    assert changed == first
    assert changed_metadata['train_record_hashes_sha256'] != first_metadata['train_record_hashes_sha256']
    assert (
        changed_metadata['validation_record_hashes_sha256']
        != first_metadata['validation_record_hashes_sha256']
    )


def test_duplicate_prompts_never_cross_train_and_validation() -> None:
    records = _math_records(30)
    records[1] = {
        'instruction': records[0]['instruction'],
        'input': records[0]['input'],
        'output': 'a different answer for the same prompt',
    }
    duplicate_indices = {0, 1}
    for seed in range(20):
        train, validation, _ = _deterministic_holdout_indices(
            records,
            7,
            seed,
            dataset='math10k',
        )
        train_set = set(train)
        validation_set = set(validation)
        assert duplicate_indices <= train_set or duplicate_indices <= validation_set


def test_glue_holdout_is_stratified_by_task_instruction() -> None:
    records = [
        {
            'instruction': f'Task: task-{task}',
            'input': f'example-{task}-{index}',
            'output': str(index % 2),
        }
        for task in range(8)
        for index in range(10)
    ]
    _, validation, metadata = _deterministic_holdout_indices(
        records,
        16,
        1729,
        dataset='glue8',
    )
    validation_counts: dict[str, int] = {}
    for index in validation:
        task = records[index]['instruction']
        validation_counts[task] = validation_counts.get(task, 0) + 1
    assert set(metadata['stratum_targets'].values()) == {2}
    assert set(validation_counts.values()) == {2}
    assert len(validation_counts) == 8


@pytest.mark.parametrize(('rows', 'holdout'), [(0, 0), (10, -1), (10, 10), (10, 11)])
def test_invalid_holdout_sizes_are_rejected(rows: int, holdout: int) -> None:
    with pytest.raises(ValueError):
        _deterministic_holdout_indices(
            _math_records(rows),
            holdout,
            1729,
            dataset='math10k',
        )


def test_validation_loss_is_response_only_record_mean_and_token_mean() -> None:
    torch.manual_seed(123)
    model = _TinyValidationLM()
    records = []
    for offset in range(5):
        input_ids = (torch.arange(6) + offset).remainder(11)
        labels = input_ids.clone()
        labels[: 2 + (offset % 2)] = -100
        records.append({'input_ids': input_ids, 'labels': labels})
    loader = DataLoader(records, batch_size=2, shuffle=False)

    model.eval()
    with torch.no_grad():
        input_ids = torch.stack([record['input_ids'] for record in records])
        labels = torch.stack([record['labels'] for record in records])
        expected, counts = causal_lm_per_example_loss(model(input_ids).logits, labels)
    model.forward_training_modes.clear()
    model.train()
    for parameter in model.parameters():
        assert parameter.grad is None
    torch.manual_seed(991)
    rng_before = torch.get_rng_state().clone()

    metrics = _evaluate_validation_loss(
        model,
        loader,
        device=torch.device('cpu'),
        needs_text_token_type_ids=False,
    )

    expected_token_mean = float((expected * counts).sum() / counts.sum())
    assert model.training is True
    assert model.forward_training_modes and not any(model.forward_training_modes)
    assert torch.equal(torch.get_rng_state(), rng_before)
    assert all(parameter.grad is None for parameter in model.parameters())
    assert metrics['records'] == 5
    assert metrics['supervised_tokens'] == int(counts.sum())
    assert metrics['loss_mean'] == pytest.approx(float(expected.mean()), rel=1e-7)
    assert metrics['token_mean_loss'] == pytest.approx(expected_token_mean, rel=1e-7)
    assert metrics['token_perplexity'] == pytest.approx(torch.exp(torch.tensor(expected_token_mean)).item())
    assert metrics['selection_metric'] == 'response_only_mean_per_record_causal_lm_loss'
    assert metrics['loss_definition'].startswith('response_only_')


def test_validation_preserves_eval_mode() -> None:
    model = _TinyValidationLM()
    model.eval()
    loader = DataLoader(
        [{'input_ids': torch.tensor([1, 2, 3]), 'labels': torch.tensor([-100, 2, 3])}],
        batch_size=1,
    )
    _evaluate_validation_loss(
        model,
        loader,
        device=torch.device('cpu'),
        needs_text_token_type_ids=False,
    )
    assert model.training is False


def test_validation_rejects_records_without_response_tokens_and_restores_state() -> None:
    model = _TinyValidationLM()
    model.train()
    loader = DataLoader(
        [{'input_ids': torch.tensor([1, 2, 3]), 'labels': torch.tensor([-100, -100, -100])}],
        batch_size=1,
    )
    torch.manual_seed(313)
    rng_before = torch.get_rng_state().clone()
    with pytest.raises(RuntimeError, match='no supervised tokens'):
        _evaluate_validation_loss(
            model,
            loader,
            device=torch.device('cpu'),
            needs_text_token_type_ids=False,
        )
    assert model.training is True
    assert torch.equal(torch.get_rng_state(), rng_before)
    assert all(parameter.grad is None for parameter in model.parameters())


def test_loader_tokenizes_validation_response_only_and_hashes_manifest(
    tmp_path, monkeypatch
) -> None:
    records = _math_records(12)
    data_path = tmp_path / 'math.json'
    data_path.write_text(json.dumps(records), encoding='utf-8')

    def fake_tokenize_prompt(
        tokenizer,
        example,
        cutoff_len,
        train_on_inputs,
        base_model,
    ):
        del tokenizer, example, cutoff_len, base_model
        return {
            'input_ids': [1, 2, 3],
            'attention_mask': [1, 1, 1],
            'labels': [1, 2, 3] if train_on_inputs else [-100, 2, 3],
        }

    monkeypatch.setattr(trainers, 'tokenize_prompt', fake_tokenize_prompt)
    cfg = SimpleNamespace(
        data_path=data_path,
        val_set_size=3,
        validation_seed=1729,
        dataset='math10k',
        data_content_sha256='f' * 64,
        protocol_stage='selection',
        validation_data_is_public=True,
        cutoff_len=32,
        train_on_inputs=True,
        base_model='tiny',
        batch_size=4,
        validation_batch_size=2,
        seed=42,
    )
    (
        train_loader,
        train_ds,
        validation_loader,
        validation_records,
        metadata,
    ) = _make_loader(cfg, object())
    assert len(train_ds) == 9
    assert len(train_loader.dataset) == 9
    assert validation_loader is not None
    assert len(validation_loader.dataset) == 3
    assert all(row['labels'] == [1, 2, 3] for row in train_ds)
    assert all(row['labels'] == [-100, 2, 3] for row in validation_loader.dataset)
    assert [row['_source_index'] for row in validation_records] == metadata[
        'validation_indices'
    ]
    assert all('instruction' in row for row in validation_records)
    assert metadata['source_content_sha256'] == 'f' * 64
    assert metadata['protocol_stage'] == 'selection'
    assert len(metadata['manifest_sha256']) == 64

    # Training seeds control model/sampler randomness but must not alter the
    # fixed public holdout shared by all paired seeds.
    other_seed_cfg = SimpleNamespace(**{**vars(cfg), 'seed': 43})
    _, _, _, _, other_seed_metadata = _make_loader(other_seed_cfg, object())
    assert other_seed_metadata == metadata


def test_validation_curve_step_reader_is_strict_and_resume_safe(tmp_path) -> None:
    curve = tmp_path / 'validation_curve.jsonl'
    curve.write_text(
        '{"step":0,"loss_mean":1.0}\n'
        '{"step":50,"loss_mean":0.8}\n',
        encoding='utf-8',
    )
    assert _validation_curve_steps(curve) == {0, 50}
    curve.write_text('{"step":0}\n{"step":0}\n', encoding='utf-8')
    with pytest.raises(RuntimeError, match='duplicate'):
        _validation_curve_steps(curve)


def test_required_curve_steps_do_not_backfill_historical_models() -> None:
    assert _required_curve_steps_before_resume(
        start_step=0,
        total_update_steps=300,
        interval=50,
    ) == {0}
    assert _required_curve_steps_before_resume(
        start_step=75,
        total_update_steps=300,
        interval=50,
    ) == {0, 50}
    assert _required_curve_steps_before_resume(
        start_step=100,
        total_update_steps=300,
        interval=50,
    ) == {0, 50, 100}
    # The final checkpoint precedes endpoint evaluation, so step 300 may be
    # regenerated from that exact model state.
    assert _required_curve_steps_before_resume(
        start_step=300,
        total_update_steps=300,
        interval=50,
    ) == {0, 50, 100, 150, 200, 250}


def _write_config_data(tmp_path) -> object:
    data_path = tmp_path / 'math.json'
    data_path.write_text(json.dumps(_math_records(6)), encoding='utf-8')
    return data_path


def _config(tmp_path, **overrides) -> RunConfig:
    values = {
        'dataset': 'math10k',
        'method': 'baseline',
        'privacy': 'dp',
        'root': tmp_path,
        'data_path': _write_config_data(tmp_path),
    }
    values.update(overrides)
    return RunConfig(**values).finalize()


def test_selection_and_final_protocol_gates(tmp_path) -> None:
    selected = _config(
        tmp_path,
        protocol_stage='selection',
        val_set_size=2,
        run_eval=False,
        validation_data_is_public=True,
    )
    assert selected.protocol_stage == 'selection'

    with pytest.raises(ValueError, match='requires val_set_size'):
        _config(
            tmp_path,
            protocol_stage='selection',
            val_set_size=0,
            run_eval=False,
            validation_data_is_public=True,
        )
    with pytest.raises(ValueError, match='forbid.*test evaluation'):
        _config(
            tmp_path,
            protocol_stage='selection',
            val_set_size=2,
            run_eval=True,
            validation_data_is_public=True,
        )
    with pytest.raises(ValueError, match='validation_data_is_public'):
        _config(
            tmp_path,
            protocol_stage='selection',
            val_set_size=2,
            run_eval=False,
            validation_data_is_public=False,
        )
    with pytest.raises(ValueError, match='full-data retraining'):
        _config(
            tmp_path,
            protocol_stage='final',
            val_set_size=2,
            run_eval=False,
            validation_data_is_public=True,
        )
    final = _config(tmp_path, protocol_stage='final', val_set_size=0, run_eval=False)
    assert final.protocol_stage == 'final'
    assert final.val_set_size == 0


def test_any_holdout_requires_public_acknowledgement_and_no_test_eval(tmp_path) -> None:
    with pytest.raises(ValueError, match='validation_data_is_public'):
        _config(tmp_path, protocol_stage='pilot', val_set_size=2, run_eval=False)
    with pytest.raises(ValueError, match='forbid.*test evaluation'):
        _config(
            tmp_path,
            protocol_stage='pilot',
            val_set_size=2,
            run_eval=True,
            validation_data_is_public=True,
        )
    with pytest.raises(ValueError, match='requires val_set_size'):
        _config(
            tmp_path,
            validation_eval_interval=50,
            val_set_size=0,
            run_eval=False,
        )
    with pytest.raises(ValueError, match='requires val_set_size'):
        _config(
            tmp_path,
            validation_generate_numeric=True,
            val_set_size=0,
            run_eval=False,
        )


def test_validation_cli_and_fingerprint_are_explicit(tmp_path) -> None:
    args = train_eval.parse_cli_args(
        [
            '--dataset',
            'math10k',
            '--val_set_size',
            '500',
            '--validation_seed',
            '1729',
            '--validation_batch_size',
            '16',
            '--validation_eval_interval',
            '50',
            '--validation_generate_numeric',
            '--validation_num_beams',
            '1',
            '--validation_max_new_tokens',
            '128',
            '--validation_max_input_length',
            '512',
            '--protocol_stage',
            'selection',
            '--validation_data_is_public',
            '--run_eval',
            'false',
        ]
    )
    assert args.val_set_size == 500
    assert args.validation_seed == 1729
    assert args.validation_batch_size == 16
    assert args.validation_eval_interval == 50
    assert args.validation_generate_numeric is True
    assert args.validation_num_beams == 1
    assert args.validation_max_new_tokens == 128
    assert args.validation_max_input_length == 512
    assert args.protocol_stage == 'selection'
    assert args.validation_data_is_public is True
    assert args.run_eval is False

    final = _config(tmp_path, protocol_stage='final', val_set_size=0, run_eval=False)
    selected = _config(
        tmp_path,
        protocol_stage='selection',
        val_set_size=2,
        validation_seed=1730,
        validation_batch_size=16,
        validation_eval_interval=50,
        validation_generate_numeric=True,
        validation_num_beams=1,
        validation_max_new_tokens=128,
        validation_max_input_length=512,
        validation_data_is_public=True,
        run_eval=False,
    )
    assert final.val_set_size == 0
    assert final.config_fingerprint != selected.config_fingerprint
    payload = selected.fingerprint_payload()
    assert payload['validation_seed'] == 1730
    assert payload['validation_batch_size'] == 16
    assert payload['validation_eval_interval'] == 50
    assert payload['validation_generate_numeric'] is True
    assert payload['validation_num_beams'] == 1
    assert payload['validation_max_new_tokens'] == 128
    assert payload['validation_max_input_length'] == 512
    assert payload['protocol_stage'] == 'selection'
    assert payload['validation_data_is_public'] is True
