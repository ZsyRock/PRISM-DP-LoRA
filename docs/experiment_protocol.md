# Experiment protocol

## Supported comparison

The official primary comparison has exactly two method labels:

1. `baseline`: fixed-threshold PRISM;
2. `slaclip`: full SlaClip controlling the threshold of that same PRISM mechanism.

The interface additionally supports `slaclip_q` for a separately labelled
fixed-target ablation. It must not be substituted for `slaclip` in a table
claiming to evaluate the camera-ready full controller. In either adaptive arm,
`--initial_clip_threshold` is `C_0`; in the baseline arm it is the fixed `C`.
`--dp_max_grad_norm` is only a legacy alias.

Full SlaClip has a tunable conditional target. Write
`rho=slaclip_target_non_small_clip_fraction`,
`z_t=s_hat_K/C_t`, and let `s_hat_1` be the noised near-threshold unclipped-CDF
proxy. Its update is

```text
gamma_t = Proj_[0,1](1 - rho * (1 - z_t))
C_next = clip(C_t * exp(eta * (gamma_t - s_hat_1)), c_min, c_max)
```

Here `z_t` is the paper's noisy, threshold-adjusted proxy for near-zero
small-gradient mass. Consequently, `rho` is the target clipped fraction within
the residual/non-small proxy mass `1-z_t`; the dynamic global clipped proxy is
`1-gamma_t` (equal to `rho*(1-z_t)` before projection). `rho=0.5` reproduces
the paper's literal `1/2`, and `slaclip_beta` is the legacy name for the same
quantity. `rho` is neither a fixed global clipping rate nor a promise about
exact `raw_clip_fraction`; `eta` is the threshold-update gain. A positive
`gamma_t-s_hat_1` increases `C_t` and tends to reduce clipping, while a negative
error decreases `C_t` and tends to increase clipping.

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

For an adaptive-arm mechanism comparison, also include a seed-matched fixed
control with `C=C_0`.  This control isolates threshold adaptation from the
initial threshold; it is distinct from the validation-tuned best-fixed arm used
in the campaign's primary comparison.

The privacy analysis assumes Opacus Poisson subsampling and add/remove record adjacency. Under this convention, the clipped tangent plus K slack coordinates have joint norm at most `C_t`; independently generated Gaussian blocks are distributionally the same joint release. `C_{t+1}` is post-processing used only at the next update. Do not reinterpret the same sensitivity statement under replace-one adjacency without revising the bound and accountant.

The intrinsic tangent-noise block uses the paper's exact full-rank thin-QR
sampler and triangular solves.  It does not damp Gram inverse square roots with
the optimizer `eps`: such damping would shrink covariance in some tangent
directions below the `noise_multiplier` assumed by the accountant.  The
DP aggregation uses the equivalent asymmetric QR-chart lift
`dA=(I-Q_A Q_A^T) g_A (B^T B)^-1`,
`dB=g_B (A^T A)^-1`, matching the support of the factorized Gaussian sampler.
It induces exactly the same intrinsic projected tangent and clipping norm as
the paper's symmetric half lift, while ensuring that the factor pair consumed
by adaptive post-processing is a deterministic lift of the already-noised
intrinsic release rather than retaining an unnoised, lift-dependent component.
The rank-`r` manifold analysis requires both LoRA factors to have full column rank.
Before per-record gradients are accumulated or a release/accountant step occurs,
training checks this condition using the configured relative Gram tolerance and
fails closed if it is not met.
A pseudoinverse/truncated continuation at a singular factor would be a different
lower-dimensional mechanism and must not be reported as the paper's rank-`r`
PRISM run without a separate query, sensitivity, and accounting analysis.

## Predeclare the comparison

### Paper/default comparison

Use the tracked `configs/math10k_paper.json` or `configs/glue8_paper.json`. These configurations use the PRV accountant, a fixed/initial threshold of 1.0, and the dataset-specific training schedule documented in the README. For a formal experiment:

The tracked defaults are the authors' public-code PRISM optimizer variant (`scalar` floor factor `0.5`, no second-moment debias) combined with this repository's record-normalized, microbatch-invariant loss. Keep those choices identical across fixed, full-SlaClip, and SlaClip-Q arms. A literal paper-geometry additive-floor/debias study is a separate optimizer ablation; do not change it in only one clipping arm or call the two protocols interchangeable.

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

For full SlaClip, predeclare
`slaclip_target_non_small_clip_fraction` (`rho`), `eta`, `c_min`, `c_max`, and
either `K` or automatic `K=0`. Use `rho=0.5` for the paper-default arm.
Sweeping `rho` retains the full controller because `s_hat_K` still determines
the dynamic global target; report other values as a target-conditioned
full-SlaClip extension. Do not describe `rho` as the exact or fixed global
clipping fraction. The legacy `slaclip_beta` spelling has the same semantics.

### Direct weighted conditional-target screen (September 2026)

The `glue-weighted-target-screen` profile implements the user-specified recipe:

```text
rho = 0.3 * mean(first 25 fixed-run clipping fractions)
    + 0.7 * mean(last 25 fixed-run clipping fractions)
u_t = normalized_noisy_slack_indicator[0]
z_t = normalized_noisy_slack_indicator[-1] / (C_t + 1e-6)
gamma_t = clip(1 - rho * (1 - z_t), 0, 1)
C_next = clip(C_t * exp(eta * (gamma_t - u_t)), C_min, C_max)
```

The weighted number is used directly as conditional rho: **no second division
by `(1-z)`**. For an early rate 100% and late rate 70%, rho is 0.79.
If the later small-gradient proxy is 0.20, the global clipped target proxy is
0.632 and the unclipped target proxy is 0.368. The first indicator coordinate
is already normalized and is not divided by C again. Both coordinates are
noisy proxies, and use the public expected Poisson batch size; neither is an
exact promise about a realized batch's hard clipping rate.

The four complete, full-length GLUE C=1 baselines provide one target each.
A second target per setting uses the same weighted recipe on a selected
fixed-C trajectory. All source identities and hashes are locked before
adaptive training. This differs intentionally from the earlier inverse-map
conditional-quantile recipe, which remains available for historical replay.

The single-A100, 80G host RAM, 24-hour job first expands the old C=15-limited
fixed scans: rank 8 at C={15,40,80}, epsilon-6/rank-16 at C={15,30,60}, seed47,
200 steps. The already-completed epsilon-6/rank-32 and epsilon-3/rank-16 scans
select C=15 and C=30, respectively. Selection uses public validation loss only.
Then all four settings run seed48, 150 steps each, with a selected-fixed
control and two Full-SlaClip target recipes (12 arms). Stage2 shares the same
800-record public holdout and the same per-setting noise multiplier, seed,
budget and initial C. It fixes eta=0.05, K=15, C_min=0.1, C_max=100; the wider
bound prevents the old C_max=15 from blocking adaptation around larger fixed C.

Only compare a 150-step adaptive arm against its 150-step fresh-seed fixed
control, never against the 200-step calibration arms. A C selected at 200 steps
is not guaranteed optimal at 150 steps. Boundary winners are flagged and
remain exploratory. These are candidate screens, not full-length paper
reproductions, task-accuracy results, or multi-seed confirmation. Positive
signals need a subsequent full-500-step, multi-seed GLUE evaluation with a
matched tuned-fixed comparator. The raw calibration and search are NON_PRIVATE;
the per-run accountant does not certify the complete data-dependent search.

### Fixed 99%-clipped SlaClip-Q ablation

The first Slack Indicator coordinate is a noisy, bin-averaged surrogate for the
*unclipped* CDF near `C_t`. SlaClip-Q ignores `s_hat_K`, so a requested fixed
global clipped fraction of 0.99 maps to the SlaClip-Q target `gamma=0.01`:

```text
C_next = clip(C_t * exp(eta * (0.01 - s_hat_1)), 0.1, 15)
```

Use the predeclared configs and paired launchers:

```bash
export PRISM_PAIR_RUN_ROOT="${SCRATCH:-/scratch/$USER}/runs/prism-dp-lora/pairs"
export PRISM_PAIR_RUN_ID="q99-seed42"
bash scripts/run_math10k_q99_pair.sh --model_revision "$MODEL_REVISION" --run_eval false
bash scripts/run_glue8_q99_pair.sh --model_revision "$MODEL_REVISION" --run_eval false
```

The Q99 configs use `eta=0.2`, matching the official SlaClip CLI
default/example. This tracks a noisy proxy and does not guarantee that exact
`raw_clip_fraction` is 0.99. Report `slack_unclipped_proxy`, its fixed 0.01
target, controller error, `C_t`, bound-hit rates, and access-controlled exact
clipping diagnostics across predeclared seeds. Keep both the paper-default full
SlaClip (`rho=0.5`) and any predeclared target-conditioned full-SlaClip arms
separate from SlaClip-Q.

### Validation-budget-matched comparison

For a release-oriented or confirmatory comparison in which the search space is
fixed independently of the private training data:

- predeclare the candidate space, for example baseline `C in {0.25, 0.5, 1, 2, 4}`;
- give each method the same validation-driven selection budget;
- do not use exact raw telemetry from one arm to grant it extra tuning decisions;
- never choose configurations or stopping points using the test set;
- state whether the privacy cost of data-dependent selection is included in the final claim.

The 4B training-dynamics campaign is deliberately a different, exploratory
protocol: it uses access-controlled exact `research_raw` statistics from a
fixed-threshold scan to construct a narrower full-SlaClip screen. It must not be
described as end-to-end DP or as a validation-budget-matched comparison. A
formal privacy claim requires either an independently predeclared grid or a new
independent confirmation whose privacy analysis composes every mechanism that
informed the grid and every released result.

Full SlaClip's target semantics are essential to this calibration. Let

```text
p*_t = rho * (1 - z_t),     z_t = s_hat_K / C_t
```

before projection. Here `p*_t` is the dynamic global clipped proxy, whereas
`rho` is conditional on the residual/non-small proxy mass. When the requested
number is a desired **global** clipping fraction, the exact fixed-run
`raw_clip_fraction` cannot be copied directly into `rho` while preserving that
global target; doing so would ignore `z_t`. The separate direct-weighted recipe
below deliberately defines the weighted number as **conditional rho**, so it
does not apply this inverse conversion. A saturated 100% fixed-run clipping median is also censored
and does not reveal how far the norms lie beyond `C`.

The exploratory campaign locks the following sequence before task-test
evaluation:

1. Run fixed PRISM at `C in {0.1, 0.5, 1, 1.5, 2, 3, 5, 15}`, all with training
   seed 42, the same deterministic public Math-10K holdout, 300 updates, and
   exact `NON_PRIVATE` `research_raw` telemetry. Record each step's clipping
   fraction, norm distribution, bias/noise diagnostics, and the telemetry-only
   `K=15` counterfactual small-gradient proxy. This counterfactual calculation
   does not add Slack coordinates to the fixed arm's DP query or alter its
   clipping, Gaussian noise, optimizer update, or accountant.
2. Lock `C_best` by public validation numeric exact-match, with public
   response-only validation loss as the deterministic tie-break. Independently
   lock `C_transition` as the smallest scanned `C` whose 10th percentile of the
   300 exact clipping fractions is below 0.99. If none meets that criterion but
   at least one observed step is below 100%, use `C=15` as an explicitly recorded
   fallback; if all eight trajectories are 100% throughout, fail closed instead
   of inventing a target grid.
3. Use `C_best` and `C_transition` as the two `C_0` values. If they coincide,
   replace the duplicate with the best distinct fixed candidate under the same
   public-validation ordering.
4. On the `C_transition` trajectory, form a narrow five-point global-target
   grid `p_i` between the clipped-fraction 10th and 90th percentiles, bounded to
   `[0.50, 0.99]` and widened to at least 0.04 when necessary. With
   `z_ref` equal to the median exact telemetry-only small-gradient proxy, map
   each point to the conditional full-SlaClip target
   `rho_i=clip(p_i/(1-z_ref), 0.50, 0.995)`. Require five finite, strictly
   increasing targets; record both `p_i` and `rho_i` plus their source hashes in
   the immutable calibration artifact. This median-proxy conversion is an
   exploratory calibration approximation, not an exact per-step inverse replay
   of the fixed trajectory.
5. Screen the Cartesian product of the two locked `C_0` values and five locked
   `rho` values. These ten full-SlaClip arms fix `eta=0.15`, `K=15`,
   `c_min=0.1`, and `c_max=15`; only `C_0` and `rho` vary.
6. Rank all fixed and SlaClip screen arms only with the common public holdout,
   then confirm the selected short list with the additional selection seeds in
   stage 2. Lock the selection record before any task-test inference.
7. Retrain the selected SlaClip and validation-tuned fixed control on the full
   training set with fresh, independent final seeds. Retain the replay and
   matched-noise controls required by the campaign, and never reuse seed 42 or a
   stage-2 seed as a fresh final seed.

The final control plan for this campaign must include the canonical
paper-default-controller anchor `rho=0.5`, `eta=0.2`, `C_0=1`, `K=15`,
`c_min=0.1`, and `c_max=15`. This anchors the literal `1/2` controller target
and the repository's official default gain while keeping the campaign bounds
and small-batch `K` explicit. It is a controller anchor, not a claim that this
modified task, loss, validation, and evaluation pipeline is a verbatim
reproduction of every original-paper experiment. It is a final control outside
the data-derived two-by-five SlaClip screen, whose gain is fixed at 0.15.

Every screen and selection stage reserves one deterministic, prompt-grouped
public holdout with a fixed `validation_seed` shared by every candidate and
training seed. Prompt groups use the versioned
`instruction_input_nfkc_casefold_whitespace_collapse_v1` identity: normalize
`instruction` and `input` separately with Unicode NFKC, `casefold`, and
split/join whitespace collapse, then compose the two fields as canonical JSON;
they are not grouped by raw bytes. Selection arms train with
`--protocol_stage selection`, `--validation_data_is_public`, and
`--run_eval false`.

For Math-10K numeric generation, only prompt groups for which every record has
a finite numeric `answer` are eligible for the public holdout. All ineligible
groups and every other unselected row, including nonnumeric-answer rows, remain
in training. Rank candidates first by deterministic numeric exact-match on that
holdout, then by response-only per-record causal-LM
validation loss as the declared tie-break. Exact raw clipping telemetry
constructs and audits the exploratory grid but is not a utility-ranking metric.
After the selection record is locked, final arms retrain with fresh seeds,
`--protocol_stage final`, and `--val_set_size 0`, then begin model inference on
the task test sets.

The split keeps all records with the same normalized `(instruction, input)`
prompt on the same side and stratifies GLUE8 by normalized task instruction.
Its source hash, membership digests, indices, normalization and selection
algorithm versions, and manifest hash are stored with every run.
Any nonzero holdout fails closed unless task-test evaluation is disabled and the
data is explicitly acknowledged as public auxiliary data.  This public-benchmark
workflow does not make exact validation release or validation-driven selection
free for a genuinely private dataset; such a deployment needs its own privacy
accounting or a separately public selection set. Nor does a public validation
set protect the fixed-scan `research_raw` artifact: the grid remains dependent
on exact private-training statistics.

These task test sets were accessed during earlier repository development, so
they must not be described as untouched, pristine, or newly held out. This
campaign guarantees only that its model evaluations start after the selection
record is locked. Prior access limits confirmatory interpretation even when the
current execution follows the lock mechanically.

Selection runs can additionally set `--validation_eval_interval 50` to write
response-only validation loss at steps `0, 50, ..., 300`, and
`--validation_generate_numeric` to write endpoint public-holdout predictions,
numeric exact accuracy, parse-failure counts, deterministic decoding settings,
and artifact hashes.  These are public-data, non-DP selection measurements and
remain distinct from exact `research_raw` training diagnostics.

A replay trajectory is an intervention, not automatically a fresh
single-budget DP result. In this campaign, each schedule is obtained from three
new full-data, no-test-evaluation SlaClip source seeds and is reported only as a
**full-data schedule-transfer control**. The downstream optimizer is the
fixed-threshold Gaussian mechanism conditional on that immutable schedule, but
the schedule still carries the privacy cost of its private-data-dependent source
runs. Any end-to-end provenance/accounting record must therefore include every
screen/selection training mechanism that informed the locked choice, all three
schedule-source mechanisms, and every released final run; a final run's
standalone accountant value is not the campaign-level privacy cost. Because the
campaign retains `research_raw` diagnostics, the complete artifact bundle is
explicitly **NON_PRIVATE** regardless of those per-run accountant values.

For this journal campaign, the sole primary endpoint is the fresh-seed paired
comparison of selected full SlaClip against the validation-tuned best fixed
threshold using `clean_three_task_macro_accuracy` (GSM8K, AQuA, and SVAMP).
MAWPS is excluded from that primary macro because the locked decontamination
audit finds normalized prompt overlap with Math-10K training data. If the
canonical paper-default-controller anchor is not selected, it must nevertheless
be run across all five fresh final seeds so that it remains distinguishable from
the selected target-conditioned extension. That five-seed anchor, fixed
`C=C_0`, canonical fixed `C=1`, schedule-transfer, matched-noise, individual-task,
MAWPS, and four-task macro results are exploratory controls or secondary
descriptions. Do not label the canonical controller arm as a verbatim
original-paper task reproduction. Historical test-set access further limits any
confirmatory claim; the lock supports an auditable comparison, not a claim that
this is the first untouched evaluation of the hypothesis.

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
5. run two update steps with the real Gemma checkpoint for `baseline`,
   `slaclip`, and any planned replay arm, including a short deterministic
   public-holdout generation smoke;
6. confirm completed statuses, a fixed baseline `C`, finite adaptive trajectories, correct proxy targets/bounds, and expected telemetry fields;
7. only then submit the complete campaign as one two-GPU allocation, using one
   Python process and one Slurm task per GPU. The two independent one-GPU lanes
   may run paired arms concurrently inside that allocation; do not split the
   campaign into arrays or separate queued jobs.

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

bash scripts/run_math10k_q99_analysis_pair.sh \
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
- treat interrupted resume checkpoints as controlled training state because they contain sampler and RNG state; never publish them as DP outputs.

The raw files may be used for internal mechanistic analysis. In this exploratory
campaign they also determine the `C_transition`/target grid, while public
validation—not raw telemetry—ranks utility. Neither use changes the raw files'
privacy classification, and their influence on the grid prevents an end-to-end
DP claim without composition or independent confirmation. See
[telemetry_schema.md](telemetry_schema.md) for the field-level boundary.

## Run identity and resume protocol

Every run records a canonical configuration fingerprint plus a readable run ID. The fingerprint includes the Git implementation commit (and a content hash for dirty changes), dataset content hash, requested model revision, method, schedule, LoRA/PRISM settings, privacy parameters, clipping/controller parameters, and telemetry mode. Do not manually reuse an output directory for a different fingerprint, and do not use a dirty worktree for a formal run.

The dataset hash/fingerprint/status bundle is publishable only when the dataset identity and hash are declared public auxiliary information, as for the tracked benchmarks. A hash computed from a genuinely private dataset is a deterministic data-dependent side output and must stay out of a claimed DP-safe release.

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
