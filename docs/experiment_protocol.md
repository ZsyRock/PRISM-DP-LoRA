# Experiment protocol

## Supported comparison

The primary comparison has exactly two method labels:

1. `baseline`: fixed-threshold PRISM;
2. `slaclip`: full SlaClip controlling the threshold of that same PRISM mechanism.

SlaClip-Q, fixed-`gamma`, and other adaptive-clipping variants are not supported method labels and must not appear in the primary tables. In the SlaClip arm, `--initial_clip_threshold` is `C_0`; in the baseline arm it is the fixed `C`. `--dp_max_grad_norm` is only a legacy alias.

For a paired run, keep every field identical except `method` and fields that only have an effect in the SlaClip arm (`slaclip_*`). Match at least:

- Git commit and clean worktree state;
- base model and immutable model revision/local snapshot;
- dataset file and content hash;
- seed and repeat identifier;
- update count, learning rate, cutoff, and target masking;
- LoRA rank, alpha, dropout, and target modules;
- expected batch size and physical microbatch size;
- accountant, epsilon, delta, and secure-mode setting;
- initial/fixed clipping threshold;
- evaluation prompts, decoding parameters, and data split;
- telemetry mode for the paired analysis.

Dynamic clipping necessarily changes the per-step absolute standard deviation because it is proportional to `noise_multiplier * C_t / expected_batch_size`. The noise multiplier, sampling schedule, and privacy accountant remain matched. This is the intended difference produced by adapting `C_t`, not an unmatched privacy target.

## Predeclare the comparison

### Paper/default comparison

Use the tracked `configs/math10k_paper.json` or `configs/glue8_paper.json`. These configurations use the PRV accountant, a fixed/initial threshold of 1.0, and the dataset-specific training schedule documented in the README. For a formal experiment:

- replace the moving model revision `main` with a reviewed Hugging Face commit or pinned local snapshot;
- use at least seeds 42, 43, and 44;
- report each seed plus mean and standard deviation;
- record any deviation from the tracked JSON before looking at results.

The paired launchers are:

```bash
bash scripts/run_math10k_pair.sh --model_revision "$MODEL_REVISION" --run_eval false
bash scripts/run_glue8_pair.sh --model_revision "$MODEL_REVISION" --run_eval false
```

They reject method, config, and output-path overrides so one shared argument list defines the pair. Distinct, fingerprinted output directories are generated automatically.

### Custom clipping-threshold comparison

Use the same predeclared value as baseline's fixed `C` and SlaClip's `C_0`:

```bash
bash scripts/run_math10k_pair.sh \
  --model_revision "$MODEL_REVISION" \
  --initial_clip_threshold 2.0 \
  --run_eval false
```

For SlaClip, predeclare `eta`, `beta`, `c_min`, `c_max`, and either `K` or automatic `K=0`. `beta` is the target used by the full controller and must not be described as a fixed target value for the clipping threshold.

### Validation-budget-matched comparison

If tuning is part of the research question:

- predeclare the candidate space, for example baseline `C in {0.25, 0.5, 1, 2, 4}`;
- give each method the same validation-driven selection budget;
- do not use exact raw telemetry from one arm to grant it extra tuning decisions;
- never choose configurations or stopping points using the test set;
- state whether the privacy cost of data-dependent selection is included in the final claim.

## Loss definition

The training objective is defined at the privacy unit (one record):

1. compute shifted next-token cross-entropy;
2. ignore labels equal to `-100`;
3. take the mean over valid target tokens within each record;
4. take the mean over records in the physical batch.

This token-mean-then-record-mean definition is invariant to how a logical batch is divided into physical microbatches and matches Opacus's record-mean gradient-sample convention. It is saved in logs, status files, and checkpoints. A resumed run with a different loss definition is rejected.

## Validation gates before formal training

The Ubuntu/CPU unit and synthetic DP paths validate repository logic, but they do not establish full Gemma/GPU compatibility. On the target HPC, complete the following gates from [the HPC runbook](hpc_runbook.md):

1. pin the Git commit and model revision;
2. build the Python 3.11 environment and pass `pip check`;
3. run `scripts/preflight_hpc.py` for package imports, data hashes, output paths, CUDA, and gated-model access as applicable;
4. run `scripts/smoke_dp_path.py` on CPU and CUDA;
5. run two update steps with the real Gemma checkpoint for both `baseline` and `slaclip`;
6. confirm both completed statuses, a fixed baseline `C`, a finite adaptive SlaClip trajectory, and expected telemetry fields;
7. only then submit the formal one-process/one-GPU arrays.

A synthetic smoke pass is necessary but does not prove GPU memory fit. If the Gemma smoke runs out of memory, reduce physical `micro_batch_size` first. Changing expected `batch_size`, update count, epsilon, delta, or clipping settings changes the experiment and must be treated as such.

## Diagnostic modes and the privacy boundary

Use `dp_safe` for release candidates. Use `research_raw` only on an access-controlled system when exact training-dynamics analysis is required.

The analysis launchers enable the required explicit acknowledgement for both paired arms:

```bash
bash scripts/run_math10k_analysis_pair.sh \
  --model_revision "$MODEL_REVISION" \
  --run_eval false

bash scripts/run_glue8_analysis_pair.sh \
  --model_revision "$MODEL_REVISION" \
  --run_eval false
```

In `research_raw`, the model update still uses clipping, Gaussian noise, and the selected accountant. This allows an internal “god-view” observer of exact loss, gradients, clipping, and realized-noise behavior while DP training proceeds. The observer artifact is not protected by that mechanism: it is a separate, non-DP output derived from private records.

Consequently:

- keep `<result_dir>/research_raw/` access-controlled;
- preserve `README_NON_PRIVATE.txt` and `NON_PRIVATE_TELEMETRY` markers;
- never upload raw JSONL files to GitHub or a public artifact store;
- never describe the model-plus-raw-log bundle as DP;
- do not publish exact losses, norm histograms, clipping fractions, or signal/noise decompositions from the raw file;
- if raw telemetry influences checkpoint, hyperparameter, seed, or model selection for release, account for that data-dependent selection or avoid making an end-to-end DP claim.

The raw files may be used for internal mechanistic analysis without entering reported accuracy. That intended use does not change their privacy classification. See [telemetry_schema.md](telemetry_schema.md) for the field-level boundary.

## Run identity and resume protocol

Every run records a canonical configuration fingerprint plus a readable run ID. The fingerprint includes the Git implementation commit (and a content hash for dirty changes), dataset content hash, requested model revision, method, schedule, LoRA/PRISM settings, privacy parameters, clipping/controller parameters, and telemetry mode. Do not manually reuse an output directory for a different fingerprint, and do not use a dirty worktree for a formal run.

Periodic checkpoints are atomic and occur only at completed logical-update boundaries. Resume restores and validates:

- trainable model state and PRISM optimizer moments;
- SlaClip controller state, selected `K`, and current threshold;
- Poisson sampler/data-loader state;
- serializable DP-noise generator state and global Python/NumPy/PyTorch RNG states;
- accountant state, noise multiplier, sample rate, and expected batch size;
- update count, loss definition, resolved model revision, and configuration fingerprint.

The main and raw JSONL files are truncated to the checkpoint step before appending. Exact resume equivalence is tested on a synthetic Opacus/PRISM SlaClip run with `dp_secure_mode=false`. A secure random-device generator can resume with fresh entropy when it has no serializable state; that is privacy-valid but not bitwise-identical to an uninterrupted run. For a cluster requeue or a fresh resume submission, reuse the same run identity and unchanged arguments. If an incomplete status has no valid checkpoint, investigate it or restart the exact configuration explicitly; do not splice artifacts by hand.

## Evaluation and reporting

Run evaluation separately from long training jobs using the completed adapter, the same fingerprinted output/result directories, and `--run_train false --run_eval true`. Evaluation refuses an adapter whose completion status or fingerprint does not match.

For each reported result retain:

- exact Git commit and repository clean/dirty state;
- Slurm job/array task ID and logs;
- Python, package, CUDA, driver, and GPU metadata;
- base model plus requested and resolved revision;
- dataset SHA-256;
- full run configuration and fingerprint;
- seed, completed update count, accountant, final epsilon/delta, noise multiplier, and sample rate;
- training LoRA rank and saved adapter rank;
- telemetry mode and raw-artifact access policy.

## Release checklist

Before copying or publishing a model/result bundle, verify:

- `run_status.json` says `state: completed`;
- the status fingerprint matches the intended experiment;
- `telemetry_mode` is `dp_safe` and `non_private_telemetry` is `false`;
- no `research_raw/`, `NON_PRIVATE_*.jsonl`, or copied raw terminal output is included;
- no exact private training loss, norm distribution, clipping fraction, realized noise, signal/noise decomposition, or supervised-token count is published;
- reported epsilon/delta correspond to the released model's completed update count and accountant;
- any data-dependent model or hyperparameter selection is covered by the stated privacy claim;
- training and saved adapter ranks are reported separately;
- the model revision and dataset hash are recorded.

Treat the DP adapter and any non-DP diagnostic log as separate artifacts with separate storage, access, retention, and release policies.
