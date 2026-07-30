# PRISM-DP-LoRA with SlaClip controllers

This repository provides reproducible, separately labelled comparisons among:

- `baseline`: PRISM with a fixed clipping threshold `C`;
- `slaclip`: the camera-ready full SlaClip controller, whose global target is dynamically constructed from the jointly noised Slack Indicator and a predeclared conditional target fraction;
- `slaclip_q`: the paper's fixed-target SlaClip-Q ablation applied to PRISM.

For full SlaClip, let `rho=slaclip_target_non_small_clip_fraction`, let
`s_hat_1` denote the noisy near-threshold *unclipped* CDF proxy, and let
`z_t=s_hat_K/C_t` denote the paper's noisy, threshold-adjusted small-gradient
proxy. The controller uses
`gamma_t=Proj_[0,1](1-rho*(1-z_t))` and
`C_(t+1)=C_t*exp(eta*(gamma_t-s_hat_1))`, followed by the declared threshold
bounds. Thus `rho` is the target clipped fraction of the residual/non-small
mass after accounting for `z_t`; it is not a fixed global clipping rate and is
not guaranteed to equal the exact realized clipping fraction. `rho=0.5`
reproduces the paper's literal `1/2`, while `eta` is the threshold-update gain.
When `gamma_t-s_hat_1` is positive the controller increases `C_t`, tending to
reduce clipping; when it is negative the controller decreases `C_t`, tending
to increase clipping. The legacy name `slaclip_beta` refers to the same `rho`.

SlaClip-Q is different: it omits `s_hat_K` and tracks a fixed global
unclipped-CDF target. A requested SlaClip-Q global clipped fraction of `0.99`
therefore maps to the fixed target `gamma=0.01`. A full-SlaClip run with
`rho=0.99` remains full SlaClip because its global target still changes with
`z_t`; it must not be labelled SlaClip-Q-99. Neither controller's noisy proxy
target guarantees an exact achieved clipping fraction.

Every paired launcher keeps the model, data order, Poisson sampling, optimizer, privacy target, accountant, noise multiplier, update count, LoRA setup, and evaluation settings identical. Only the declared clipping controller differs.

## What is ready, and what still needs HPC validation

The repository-level unit suite and the synthetic Opacus/PRISM DP smoke path have been exercised on Ubuntu/CPU. They cover the fixed and adaptive threshold paths, per-sample gradients, DP noise and accounting, privacy-separated telemetry, stable experiment identities, atomic checkpoints, and checkpoint/resume equivalence.

That does **not** prove that the full gated `google/gemma-3-4b-pt` checkpoint fits a particular GPU, that the cluster's CUDA/PyTorch binaries are compatible, or that every Gemma 3 module works in that environment. Before a formal run, the HPC still needs:

1. the environment/data preflight;
2. the synthetic CUDA smoke test;
3. a two-step, one-GPU Gemma smoke run for both methods.

These are hardware and environment validation steps, not unfinished experiment interfaces. If they expose a cluster-specific incompatibility, the code can still be revised after migration; HPC is not an artificial boundary on future changes. See [the HPC runbook](docs/hpc_runbook.md) for the exact sequence.

## Environment

Python 3.11 is recommended. For a local installation:

```bash
python -m pip install -r requirements.txt
```

Development checks additionally require:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
python scripts/smoke_dp_path.py --device cpu --steps 2
```

The dependency files are layered as follows:

- `requirements-core.txt`: Transformers, PEFT, datasets, evaluation, and I/O;
- `requirements.txt`: runtime stack including bounded PyTorch and Opacus families;
- `requirements-dev.txt`: runtime stack plus test dependencies;
- `environment.yml`: reference Python 3.11 Conda environment.

Gemma 3 4B is a gated, multimodal checkpoint even when this project fine-tunes it on text. Accept the model terms, authenticate with Hugging Face, and pin a reviewed model commit for formal experiments. The loader selects a compatible multimodal/conditional-generation class and freezes the vision tower during text-only LoRA training. Transformers 4.50 or later is required for this architecture.

## Paper/default configurations

The tracked JSON configurations make the reproduction defaults explicit:

| Setting | Math-10K | GLUE8 |
|---|---:|---:|
| Config | `configs/math10k_paper.json` | `configs/glue8_paper.json` |
| Updates | 300 | 500 |
| Learning rate | `3e-4` | `2e-4` |
| Cutoff length | 256 | 384 |
| Train on inputs | yes | no |
| Expected batch / physical microbatch | 64 / 4 | 64 / 4 |
| LoRA rank / alpha / dropout | 16 / 16 / 0.05 | 16 / 16 / 0.05 |
| Initial or fixed threshold | 1.0 | 1.0 |

Both configs use `google/gemma-3-4b-pt`, target `q_proj,k_proj,v_proj,up_proj,down_proj`, request `epsilon=6` and `delta=1e-5`, and use the PRV accountant. `prv` is also the CLI default.

The default optimizer protocol is deliberately labelled the **PRISM public-code variant**: scalar second-moment floor `0.5` and no noise-moment debias, matching the authors' released training defaults. The paper text also describes a geometry-dependent additive floor/debias variant; that is a distinct optimizer ablation and must not be silently mixed into a clipping-controller comparison. This repository additionally uses a record-normalized, microbatch-invariant causal-LM loss. It matches the paper's per-record DP formulation but differs from the released code's batch-token reduction, so clipping trajectories from this hardened protocol should not be described as bitwise upstream-code reproduction.

The journal ablation configs `configs/math10k_slaclip_q99.json` and `configs/glue8_slaclip_q99.json` retain those task/model settings and predeclare `target_clip_fraction=0.99`, `C_min=0.1`, `C_max=15`, and the official SlaClip CLI gain `eta=0.2`. They also make the PRISM public-code optimizer variant explicit (`scalar` floor with factor `0.5`, no second-moment debias). Their research histogram range is fixed at 30 with 128 bins so an adaptive threshold above the original `4*C_0` range does not silently collapse into overflow.

For a formal run, override the moving `main` model revision with an immutable Hugging Face commit or a pinned local snapshot:

```bash
MODEL_REVISION="REPLACE_WITH_REVIEWED_HF_COMMIT_SHA"

bash scripts/run_math10k_pair.sh \
  --model_revision "$MODEL_REVISION" \
  --run_eval false
```

The pair script launches `baseline` and then `slaclip` from one shared argument list. The GLUE equivalent is `scripts/run_glue8_pair.sh`. Training-only jobs should use `--run_eval false` and perform generation/classification evaluation in a separate job.

For the fixed-target 99%-clipped ablation, use one paired process:

```bash
export PRISM_PAIR_RUN_ROOT="${SCRATCH:-/scratch/$USER}/runs/prism-dp-lora/pairs"
export PRISM_PAIR_RUN_ID="math10k-q99-seed42"
bash scripts/run_math10k_q99_pair.sh \
  --model_revision "$MODEL_REVISION" \
  --run_eval true
```

The GLUE equivalent is `scripts/run_glue8_q99_pair.sh`. Supplying both pair environment variables routes each arm's adapter, logs, telemetry, and results to separate directories below the declared absolute root; this is the portable scratch path. On Slurm, use the cluster wrapper generated by the HPC skill so both arms execute sequentially inside one allocation.

To run either method directly:

```bash
python train_eval.py \
  --config configs/math10k_paper.json \
  --method baseline \
  --model_revision "$MODEL_REVISION" \
  --initial_clip_threshold 1.0

python train_eval.py \
  --config configs/math10k_slaclip_q99.json \
  --method slaclip_q \
  --model_revision "$MODEL_REVISION" \
  --initial_clip_threshold 1.0

python train_eval.py \
  --config configs/math10k_paper.json \
  --method slaclip \
  --model_revision "$MODEL_REVISION" \
  --initial_clip_threshold 1.0 \
  --slaclip_target_non_small_clip_fraction 0.5
```

`--initial_clip_threshold` means fixed `C` for `baseline` and initial `C_0` for either adaptive controller. `--dp_max_grad_norm` remains only as a legacy alias. A custom initial-threshold experiment can therefore use, for example:

```bash
bash scripts/run_math10k_pair.sh \
  --model_revision "$MODEL_REVISION" \
  --initial_clip_threshold 2.0 \
  --run_eval false
```

For full SlaClip, `--slaclip_target_non_small_clip_fraction`,
`--slaclip_eta`, `--slaclip_c_min`, and `--slaclip_c_max` configure the
controller. The first option is `rho`, the target clipped fraction within the
noisy residual/non-small mass; `--slaclip_beta` is its legacy alias. It is not
a fixed global or exact achieved clipping rate. `eta`, not `rho`, is the
feedback/update gain. For SlaClip-Q, `--slaclip_target_clip_fraction 0.99` is
converted to the complementary fixed target-unclipped proxy `0.01`.
`--slaclip_num_slots 0` follows the journal-extension policy: expected batches
below 128 use `K=15`, while batches of 128 or more use the paper-bound formula
based on expected batch size and noise multiplier. Small-batch `K=15` remains
DP-valid but can exceed the paper's high-probability CDF-monotonicity bound, so
experiments must label it as the journal policy rather than a paper-bound
choice.

Journal hyperparameter selection uses a deterministic prompt-grouped public
holdout, never the task test sets or exact raw clipping telemetry.  Selection
runs must pass `--protocol_stage selection`, `--validation_data_is_public`, and
`--run_eval false` with a nonzero `--val_set_size`; configuration locking is
performed by `scripts/select_validation_candidates.py`.  Math-10K screens can
add `--validation_generate_numeric` so public-holdout numeric exact accuracy is
the primary utility signal and response-only loss is the deterministic
tie-break. `--validation_eval_interval 50` records the public response-loss
curve at steps `0, 50, ..., 300`; endpoint predictions, parse-failure counts,
decoding settings, and hashes are stored with the run. Once locked, formal
arms use fresh seeds, `--protocol_stage final`, and `--val_set_size 0` to retrain
on the complete training set before evaluation. In particular, a saturated
100% baseline clipping median is not mechanically equated with full SlaClip's
conditional `rho`; candidate `rho` values are predeclared and selected only
with the public validation protocol. See
[the experiment protocol](docs/experiment_protocol.md) for the leakage guards
and public-data privacy boundary.

The noise multiplier is calibrated for the requested update count. Each Poisson batch is normalized by Opacus's fixed expected batch size, not by the randomly realized batch size. Dynamic clipping changes the absolute noise scale with `C_t`, while the matched noise multiplier and accountant determine the same privacy schedule. The sensitivity statement uses Poisson subsampling and add/remove record adjacency; replace-one adjacency must not be substituted without changing the analysis.

For `method=replay`, the reported per-run accountant is conditional on the
locked clipping schedule. If that schedule was derived from earlier
private-data-dependent trajectories, compose their privacy costs with the replay
run before making any end-to-end claim. Replay is used here as a mechanistic
control; it is not a way to reset the privacy budget.

## Training loss and microbatch invariance

Training uses one causal-LM loss per record:

1. shift logits and labels by one token;
2. average cross-entropy over the non-ignored target tokens **within each record**;
3. average those per-record values across the physical batch.

In short: token mean within a record, then record mean across the batch. This matches Opacus's record-level `loss_reduction='mean'` contract, so per-record gradients and clipping decisions do not change merely because a logical batch is split into different physical microbatches. The definition is recorded in logs, statuses, and checkpoints and is checked on resume.

## DP-safe telemetry and research “god view”

`--telemetry_mode dp_safe` is the default. Its main log contains run identity, privacy accounting, clipping thresholds, values obtained by post-processing the DP release/model update, and—for either SlaClip arm only—the jointly noised Slack Indicator and controller updates. The baseline arm does not compute or release SlaClip slack coordinates. The log deliberately excludes exact DP-training loss, exact clipping fractions, raw norm distributions, the realized noise norm, and other decompositions that would reveal non-released training behavior.

For access-controlled training-dynamics research, the requested exact observer is enabled only with both flags:

```bash
bash scripts/run_math10k_analysis_pair.sh \
  --model_revision "$MODEL_REVISION" \
  --initial_clip_threshold 1.0 \
  --run_eval false
```

The analysis scripts are equivalent to passing:

```text
--telemetry_mode research_raw --allow_non_private_telemetry
```

The GLUE equivalent is `scripts/run_glue8_analysis_pair.sh`. Raw records are written to:

```text
<result_dir>/research_raw/NON_PRIVATE_train_log.jsonl
```

They include exact record-mean training loss and supervised-token count,
per-record tangent-gradient norm summaries/histograms, clipping fraction and
coefficients, clipped and unclipped signal norms, clipping-bias norm, realized
noise norm, signal-to-noise ratio, exact-vs-noisy Slack/CDF residuals, signal
cosines, bias/noise ratio, and a bias-squared-plus-noise-squared proxy. The
DP-safe controller record separately exposes the requested conditional `rho`,
the noisy pre-projection small-gradient and remaining-mass proxies, the dynamic
target before and after projection, the observed `s_hat_1` proxy, and the
controller error. These proxy fields are post-processing of the jointly noised
DP-safe Slack Indicator release; they are not `research_raw` measurements.
The raw file is self-contained and marked `NON_PRIVATE_TELEMETRY`.

For SlaClip-Q-99 dynamics, use `scripts/run_math10k_q99_analysis_pair.sh` (or
the GLUE counterpart). Its summary separates the requested global clipped-rate
label, noisy CDF proxy, fixed proxy target, controller error,
unbounded/bounded next threshold, and bound-hit flags. The label is a requested
proxy target, not a claim that the exact achieved rate is 99%.

To turn one raw trajectory into a step-wise CSV plus a compact JSON summary:

```bash
python scripts/summarize_telemetry.py \
  <result_dir>/research_raw/NON_PRIVATE_train_log.jsonl \
  --safe-log <output_dir>/train_log.jsonl
```

The generated files remain explicitly marked non-private and belong under the same access controls as the source log.

The optimizer and released adapter still follow the configured DP training mechanism while this observer is active. The observer file itself, however, is a direct function of private examples and is **not a DP release**. Keep it access-controlled, do not publish it, and do not claim that the model-plus-raw-log bundle is DP. If raw telemetry is used to choose a checkpoint, hyperparameter, seed, or model for release, that selection also needs a privacy analysis; “internal only” does not make data-dependent selection free.

See [the telemetry schema](docs/telemetry_schema.md) for field-level meanings and [the experiment protocol](docs/experiment_protocol.md) for the release boundary.

## Run identity, outputs, and exact resume

Every experiment receives a content-derived configuration fingerprint and a readable run ID containing the clipping threshold plus a short hash. The fingerprint covers the Git implementation commit plus a content hash when the worktree is dirty, dataset content hash, requested model revision, method, privacy parameters, optimizer/LoRA configuration, training schedule, and telemetry mode. Existing output directories with a different or unverifiable fingerprint are rejected instead of silently mixed. Formal runs should still start from a clean commit so another machine can reproduce the implementation directly.

The automatic dataset hash is safe to expose here only because the tracked benchmark data and their hashes are public auxiliary information. For a genuinely private dataset, do not publish the hash, fingerprint/run ID derived from it, or status/config snapshot without a separate release design. Interrupted resume checkpoints also contain sampler and RNG state and are controlled training state, not DP outputs; a completed run deletes its resume checkpoint.

Default artifacts are placed under:

```text
LLM-Adapters/
  ft-training_set/       # tracked training assets
  trained_models/<run>/  # adapter, DP-safe JSONL, status/config snapshot
  experiment/<run>/      # evaluation output and optional research_raw directory
```

Checkpoints are written atomically at step zero and after completed logical steps. A valid resume restores and validates trainable weights, PRISM optimizer and SlaClip state, current `C_t`, Poisson sampler state, data-loader generator, serializable DP-noise generator state, privacy accountant, global Python/NumPy/PyTorch RNG state, completed update count, loss definition, model revision, and configuration fingerprint. Logs are truncated back to the checkpoint step before appending, preventing duplicate future records. Bitwise resume equivalence is covered by a synthetic Opacus/PRISM test in the default reproducible research mode (`dp_secure_mode=false`). A secure CSPRNG may intentionally resume from fresh entropy when its generator is not serializable; the mechanism remains valid but the continuation is not expected to match an uninterrupted run bit for bit.

Use the same run identity and unchanged configuration to resume. `--force_train` intentionally restarts the exact configuration and clears both that run's model and result directories so stale evaluation or telemetry cannot survive; it should not be used to overwrite an unrelated run directory.

The saved adapter rank can be twice the training rank because spectral residual rebasing restores the original base model while preserving the learned low-rank update. `run_status.json` records `training_lora_r`, `saved_adapter_r`, the resolved model revision, privacy accounting, runtime metadata, and completion state. Evaluation accepts only a completed adapter whose fingerprint matches the requested experiment.

## Evaluation and reporting

Math evaluation reports exact-answer accuracy. GLUE evaluation retains the standard task metrics: Matthews correlation for CoLA; accuracy for SST-2, QNLI, and RTE; accuracy/F1 for MRPC and QQP; Pearson/Spearman for STS-B; and matched/mismatched accuracy for MNLI.

Evaluation reloads the resolved base-model revision recorded by training, even if the original request used a moving label such as `main`. Cached predictions are bound to an `evaluation_config.json` containing that resolved revision, the adapter fingerprint, task list, decoding parameters, and (for GLUE) `fast_dev_run`. Math predictions are written directly to each run's result directory, so concurrent baseline/SlaClip evaluations cannot collide. A mismatched or legacy cache is rejected; use `--force_eval` to rebuild it explicitly.

For claims about adaptive clipping, use at least three predeclared seeds and report mean and standard deviation. Keep all paired fields matched, use the same `C` as baseline's fixed threshold and SlaClip's `C_0`, and do not select settings from the test set. The detailed comparison and artifact checklist is in [docs/experiment_protocol.md](docs/experiment_protocol.md).

## Acknowledgements

This repository builds on the [PRISM paper](https://arxiv.org/abs/2606.00944), the full [SlaClip implementation](https://github.com/ZsyRock/SlaClip), and the [LLM-Adapters](https://github.com/AGI-Edgerunners/LLM-Adapters) pipeline. Portions adapted from LLM-Adapters are distributed under the Apache-2.0 license; see `licenses/Apache-2.0.txt`.
