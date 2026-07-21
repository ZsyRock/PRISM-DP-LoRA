# PRISM-DP-LoRA with SlaClip

Research code for **PRISM: Gauge-Invariant Tangent-Space Differentially Private LoRA**, with a controlled comparison between:

- `baseline`: PRISM with a fixed clipping threshold;
- `slaclip`: the full SlaClip controller applied to the same PRISM tangent-space update.

SlaClip-Q is intentionally not part of this repository's experiment interface. The two supported methods use the same model, data order, Poisson sampling, optimizer, privacy accountant, noise multiplier, update count, and evaluation settings. Their intended algorithmic difference is whether the clipping threshold remains fixed or is updated from the DP Slack Indicator.

## Setup

Python 3.11 is recommended. Create and activate an isolated environment, then install:

```bash
python -m pip install -r requirements.txt
```

Development checks additionally require:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Gemma models may require accepting the model license and authenticating with Hugging Face. Formal training requires a CUDA GPU with sufficient memory; the unit tests are CPU-only.

## Fair baseline/SlaClip runs

Fixed-threshold baseline:

```bash
python train_eval.py \
  --dataset math10k \
  --method baseline \
  --privacy dp \
  --epsilon 6 \
  --delta 1e-5 \
  --dp_max_grad_norm 1.0 \
  --batch_size 64 \
  --micro_batch_size 4 \
  --seed 42 \
  --telemetry_mode dp_safe
```

SlaClip plus the same baseline:

```bash
python train_eval.py \
  --dataset math10k \
  --method slaclip \
  --privacy dp \
  --epsilon 6 \
  --delta 1e-5 \
  --dp_max_grad_norm 1.0 \
  --batch_size 64 \
  --micro_batch_size 4 \
  --seed 42 \
  --telemetry_mode dp_safe
```

`scripts/run_math10k_pair.sh` runs this pair from one shared argument list. `--slaclip_num_slots 0` (the default) selects `K` from the SlaClip paper's bound using expected batch size and noise multiplier. `--slaclip_eta`, `--slaclip_beta`, `--slaclip_c_min`, and `--slaclip_c_max` configure the full SlaClip controller.

The privacy noise multiplier is calibrated for the exact requested update count. A Poisson batch is normalized by Opacus's fixed expected batch size, not by its randomly realized size.

## Telemetry and the privacy boundary

`--telemetry_mode dp_safe` is the default. Its main log contains configuration, privacy accounting, the jointly noised Slack Indicator, threshold updates, and quantities obtained by post-processing DP releases. Exact training loss, exact clipping fraction, and raw gradient-norm statistics are excluded.

For trusted internal analysis only, an explicit research switch enables the requested “god-view” diagnostics:

```bash
python train_eval.py \
  --dataset math10k \
  --method slaclip \
  --privacy dp \
  --telemetry_mode research_raw \
  --allow_non_private_telemetry
```

This mode records exact per-step tangent-gradient norm histograms, quantiles, clipping fraction, clipping coefficients, raw clipped-signal norm, signal-to-noise ratio, token count, and training loss under:

```text
LLM-Adapters/experiment/<run>/research_raw/NON_PRIVATE_train_log.jsonl
```

The model update still executes the configured DP mechanism. However, the `research_raw` files themselves are direct functions of private examples and are **not differentially private releases**. Publishing them, or treating the entire model-plus-logs bundle as DP, would invalidate that end-to-end claim. The directory is separately marked and ignored by Git.

The raw histogram resolution (`--raw_hist_bins`) is independent of SlaClip's slack dimension `K`. By default, histogram edges use a fixed range from zero to four times the initial clipping threshold, plus an overflow count, so distributions remain comparable across steps.

## Outputs

```text
LLM-Adapters/
  ft-training_set/       # supplied training data
  trained_models/        # adapters, DP-safe log, status/config snapshot
  experiment/            # evaluation outputs and optional research_raw logs
```

The saved adapter rank can be twice the training rank because spectral residual rebasing is required to restore the original base model while preserving the learned low-rank update. `run_status.json` records both `training_lora_r` and `saved_adapter_r`.

GLUE evaluation retains each task's standard metrics: Matthews correlation for CoLA; accuracy for SST-2, QNLI, and RTE; accuracy/F1 for MRPC and QQP; Pearson/Spearman for STS-B; and matched/mismatched accuracy for MNLI. Math evaluation reports exact-answer accuracy.

## Reproducibility notes

- Checkpoints store unwrapped PEFT parameter names and accept the older Opacus `_module.` prefix when resuming.
- SlaClip controller state, current threshold, selected `K`, optimizer moments, RNG state, and completed update count are checkpointed.
- Training logs and status snapshots record the complete run configuration.
- Use at least three seeds for reported results. Compare both the paper/default fixed threshold and a validation-budget-matched fixed-threshold grid before making claims about adaptive clipping.

See [docs/experiment_protocol.md](docs/experiment_protocol.md) for the comparison and release checklist.

## Acknowledgements

This repository builds on code from the PRISM authors and the [LLM-Adapters](https://github.com/AGI-Edgerunners/LLM-Adapters) pipeline. Portions adapted from LLM-Adapters are distributed under the Apache-2.0 license; see `licenses/Apache-2.0.txt`.
