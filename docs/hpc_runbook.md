# HPC runbook

This runbook moves as much work as possible out of the expensive GPU queue. The
supported production shape is one Python process on one CUDA GPU. The current
trainer is not DDP-aware; never use `torchrun`, multiple Slurm tasks, or multiple
nodes for one run.

## 1. Clone an immutable code revision

Clone the repository on persistent project storage and record the exact commit:

```bash
git clone https://github.com/ZsyRock/PRISM-DP-LoRA.git
cd PRISM-DP-LoRA
git switch agent/prism-slaclip-integration
git rev-parse HEAD
git status --short
```

Formal experiments should start from a clean worktree and a recorded commit.
Do not put model caches, checkpoints, or raw telemetry in the Git repository.

## 2. Create the environment

The reference definition uses Python 3.11 and bounded dependency families:

```bash
conda env create -f environment.yml
conda activate prism-dp-lora
python -m pip check
```

PyTorch must be compatible with both the cluster driver and the CUDA capability
of the allocated GPU. If the wheel selected by `environment.yml` is unsuitable,
create the environment with Python and pip first, install the site-recommended
PyTorch build, and then install this repository's requirements. For example:

```bash
conda create -n prism-dp-lora python=3.11 pip packaging -y
conda activate prism-dp-lora
# Install the cluster-supported PyTorch wheel/module here.
python -m pip install -r requirements-dev.txt
python -m pip check
```

Do not blindly copy a CUDA wheel command from another cluster. The NVIDIA driver
is the compatibility boundary; `nvidia-smi` and the site's software guide decide
which build to use.

The dependency layers are:

- `requirements-core.txt`: Transformers, PEFT, datasets, evaluation, and I/O;
- `requirements.txt`: the core layer plus bounded PyTorch and Opacus versions;
- `requirements-dev.txt`: runtime requirements plus CPU tests;
- `environment.yml`: the reference Python 3.11 Conda environment.

## 3. Put caches and outputs on persistent scratch

Hugging Face caches and model checkpoints are too large for a typical home
quota. Configure paths before downloading anything:

```bash
export PRISM_REPO_ROOT="$PWD"
export PRISM_RUN_ROOT="$SCRATCH/prism-dp-lora"
export HF_HOME="$PRISM_RUN_ROOT/cache/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TMPDIR="$PRISM_RUN_ROOT/tmp/login"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$TMPDIR" "$PRISM_RUN_ROOT"
```

`PRISM_RUN_ROOT` must survive node preemption. Node-local temporary storage is
appropriate for `TMPDIR`, but not for `_resume_checkpoint.pt`, adapters, or raw
telemetry.

## 4. Accept and stage the Gemma model

`google/gemma-3-4b-pt` is gated. On the Hugging Face website, accept Google's
Gemma terms with the same personal account used on the cluster. Then authenticate
without placing a token in shell history or Git:

```bash
hf auth login
hf auth whoami
```

For a cluster with no compute-node internet, download on a login or transfer
node. Pin a reviewed commit rather than a moving branch:

```bash
MODEL_REVISION="REPLACE_WITH_REVIEWED_HF_COMMIT_SHA"
export PRISM_MODEL_REVISION="$MODEL_REVISION"
export PRISM_BASE_MODEL="${PRISM_BASE_MODEL:-google/gemma-3-4b-pt}"
hf download google/gemma-3-4b-pt \
  --revision "$MODEL_REVISION" \
  --cache-dir "$HF_HOME/hub"
```

The most robust offline input is the resolved snapshot directory. Set
`PRISM_BASE_MODEL` to that absolute directory before submitting jobs, and set
`HF_HUB_OFFLINE=1` only after all needed files are cached. Never add `HF_TOKEN`
to an sbatch file, command log, config snapshot, or repository file.

## 5. Run preflight checks

On a CPU login node, validate packages and immutable data assets without network
access:

```bash
python scripts/preflight_hpc.py \
  --dataset math10k \
  --output-root "$PRISM_RUN_ROOT"
```

The repository currently expects these tracked training assets:

- Math-10K: 9,919 rows, SHA-256
  `0342d0d860ad8592b579329337c90e42eefd3d9f2898043140cbd120630418b8`;
- GLUE8: 10,000 rows, SHA-256
  `281bad3305e8be7a7660b64b22704a9462f983b90df5d6f0c1c4abb649d6d091`.

Inside an interactive one-GPU allocation, additionally require CUDA and verify
the gated model endpoint:

```bash
python scripts/preflight_hpc.py \
  --dataset math10k \
  --output-root "$PRISM_RUN_ROOT" \
  --require-cuda \
  --check-hf-access \
  --model-id google/gemma-3-4b-pt \
  --model-revision "$MODEL_REVISION"
```

The preflight reports only whether a Hugging Face credential exists. It never
prints the credential value. A failed preflight must be fixed before a long job.

## 6. Run the synthetic DP smoke test

This fast check exercises Opacus per-sample gradients, PRISM tangent clipping,
Gaussian noise, privacy accounting, raw telemetry, the fixed baseline threshold,
and the adaptive SlaClip threshold:

```bash
python scripts/smoke_dp_path.py --device cpu --steps 2
python scripts/smoke_dp_path.py --device cuda --steps 2
```

It uses synthetic data and a tiny LoRA-shaped network. Passing it is necessary
but does not prove that Gemma fits in memory or that every Gemma module is
compatible with Opacus.

## 7. Run a two-step Gemma smoke test

Use unique disposable directories and disable the expensive generation
evaluation. Run both methods with identical shared parameters:

```bash
SMOKE_ROOT="$PRISM_RUN_ROOT/smoke/$(date -u +%Y%m%dT%H%M%SZ)"

python train_eval.py \
  --dataset math10k \
  --method baseline \
  --privacy dp \
  --base_model "$PRISM_BASE_MODEL" \
  --model_revision "$MODEL_REVISION" \
  --require_cuda \
  --steps 2 \
  --batch_size 4 \
  --micro_batch_size 1 \
  --initial_clip_threshold 1.0 \
  --telemetry_mode research_raw \
  --allow_non_private_telemetry \
  --output_dir "$SMOKE_ROOT/baseline/model" \
  --result_dir "$SMOKE_ROOT/baseline/result" \
  --run_eval false

python train_eval.py \
  --dataset math10k \
  --method slaclip \
  --privacy dp \
  --base_model "$PRISM_BASE_MODEL" \
  --model_revision "$MODEL_REVISION" \
  --require_cuda \
  --steps 2 \
  --batch_size 4 \
  --micro_batch_size 1 \
  --initial_clip_threshold 1.0 \
  --slaclip_target_non_small_clip_fraction 0.5 \
  --telemetry_mode research_raw \
  --allow_non_private_telemetry \
  --output_dir "$SMOKE_ROOT/slaclip/model" \
  --result_dir "$SMOKE_ROOT/slaclip/result" \
  --run_eval false
```

Check that both adapters complete, `run_status.json` records the expected method
and privacy accounting, baseline `dp_clip_threshold` remains fixed, SlaClip has
a finite threshold trajectory, and both raw JSONL files contain norm quantiles,
histograms, clipping fraction, SNR, loss, tokens, and batch size.

If this step runs out of memory, reduce `micro_batch_size` first. Changing the
expected logical `batch_size`, update count, epsilon, or delta changes the noise
calibration and is an experimental change, not merely a memory workaround.

## 8. Submit paired arrays or one dynamics campaign

The templates map array tasks as follows:

| Task | Method | Seed |
|---:|---|---:|
| 0 | baseline | 42 |
| 1 | slaclip | 42 |
| 2 | baseline | 43 |
| 3 | slaclip | 43 |
| 4 | baseline | 44 |
| 5 | slaclip | 44 |

Pass cluster-specific account and partition settings at submission time. Keep a
stable `PRISM_RUN_ID` when intentionally resuming the same array:

```bash
export PRISM_PYTHON="$(command -v python)"
export PRISM_RUN_ID="math10k-default-v1"
export PRISM_TELEMETRY_MODE="research_raw"
export PRISM_CHECK_HF_ACCESS=0

sbatch --account=YOUR_ACCOUNT --partition=YOUR_GPU_PARTITION \
  --export=ALL slurm/math10k_train_array.sbatch
```

For GLUE8 training:

```bash
export PRISM_RUN_ID="glue8-default-v1"
sbatch --account=YOUR_ACCOUNT --partition=YOUR_GPU_PARTITION \
  --export=ALL slurm/glue8_train_array.sbatch
```

Both templates train only (`--run_eval false`) so generation evaluation cannot
consume the training walltime. They use one GPU and one task, run preflight in
the allocation, write to `PRISM_RUN_ROOT`, and give every method/seed a distinct
directory. Slurm stdout is appended on requeue.

The Math-10K 4B fixed-scan/SlaClip dynamics protocol should not use those
arrays. Its portable wrapper stages one clean Git revision and submits the
complete protocol as one two-H200 allocation:

```bash
bash scripts/submit_math10k_4b_dynamics_campaign.sh --test-only
bash scripts/submit_math10k_4b_dynamics_campaign.sh --submit
```

Run `--test-only` first; it calls `sbatch --test-only` to validate the exact
request without queuing a job. Then invoke `--submit` exactly once. Inside the allocation, the worker
uses two independent one-task/one-GPU lanes for the eight-point fixed scan,
locks the raw-telemetry-derived five-point target mapping, runs the two-by-five
SlaClip screen, confirms the public-validation short list, and executes the
fresh-seed final controls. This is one queued job, not an array and not one job
per arm. The trainer remains single-GPU and is not converted to DDP merely
because two independent arms share the allocation.

This campaign retains exact `research_raw` artifacts under scratch and uses
them to construct its exploratory target grid. The complete campaign is
therefore `NON_PRIVATE`; see [the experiment protocol](experiment_protocol.md)
before making any privacy or confirmatory claim.

### One-A100 dense fixed-C follow-up

After the locked `math10k-4b-c1-c2-refinement` campaign, use the dedicated
follow-up wrapper to test whether an apparent SlaClip gain is only the result
of moving between fixed C=1 and C=2:

```bash
bash scripts/submit_math10k_4b_dense_fixed_followup.sh --test-only
bash scripts/submit_math10k_4b_dense_fixed_followup.sh --submit
```

The wrapper submits exactly one dependent Slurm job requesting one A100, one
task, eight CPUs, 128G host memory, and the cluster's 60-hour A100 maximum.
Inside that allocation, all arms run sequentially. The worker first resumes
any incomplete locked fresh-seed arms from the source campaign using its
original immutable training commit. It then evaluates fixed
`C={1.25,1.5,1.75}` on seeds 42--46, combines those public-validation results
with the already locked C=1 and C=2 runs, and locks one dense-best fixed
control. Fresh full evaluation is enabled only if the selected SlaClip beats
that control by at least 0.5 percentage points on average, wins at least three
of five paired validation seeds, and passes the trajectory-integrity audit.
Final-test measurements never select C or change this gate.

The default dependency and source paths can be replaced for another account
or resumed source campaign with `PRISM_REFERENCE_JOB_ID` and
`PRISM_REFERENCE_CAMPAIGN_ROOT`. User, home, scratch, environment, cache, run,
and repository paths are discovered or supplied through the portable
`PRISM_*` overrides documented by `--help`; no username is embedded in the
worker. The new receipt records both the follow-up orchestration revision and
the frozen experiment-mechanism revision. Use `--resume-submit` only after a
failed or timed-out follow-up allocation; completed arms are validated and
skipped, while partial arms use their fingerprinted checkpoints.

### One-A100 low-target paired confirmation

Use the low-target wrapper to test conditional Full-SlaClip targets
`rho={0.8,0.9}` against the predeclared strongest established fixed `C=2`
baseline:

```bash
bash scripts/submit_math10k_4b_low_target_campaign.sh --test-only
bash scripts/submit_math10k_4b_low_target_campaign.sh --submit
```

This is one non-array allocation with one A100, one task, eight CPUs, 128G
host memory, and the 60-hour partition maximum. The worker runs every arm
sequentially on that allocation. A four-arm, 150-step public-validation pilot
chooses between `C_0={2,3}` separately for each conditional target. The locked
final phase then trains both selected Full-SlaClip configurations and fixed
`C=2` on the same five previously unused seeds for 300 steps
and evaluates all three arms with the same decoding settings. Neither test
accuracy nor test assets participate in candidate selection.

The primary locked comparison is `rho=0.9` versus fixed `C=2`; `rho=0.8` is
secondary. `C=2` was predeclared because the complete earlier
five-seed sweep made it the strongest established fixed threshold before this
campaign was designed; this is not a claim that every possible fixed threshold
has been exhaustively optimized. The analyzer reports paired five-seed accuracy
differences and a two-sided 95% paired-t interval. Endpoint accuracy standard
deviation and trajectory metrics are descriptive stability evidence, not a
formal variance-superiority test. Here `rho` is conditional on the noisy
non-small proxy mass: the realized whole-batch clipped target is
`Proj_[0,1](rho*(1-z_t))`, where `z_t=s_hat[t,K]/C_t`. Report `rho`, the
dynamic target, and the realized clipping rate separately.

The benchmark's test results were inspected by earlier campaigns. This run is
therefore a locked internal paired confirmation, not an untouched external
replication. A positive result should be replicated on an independent dataset
or otherwise untouched evaluation before being presented as definitive
journal evidence.

The wrapper discovers the current user, home, and scratch paths and supports
the portable `PRISM_*` overrides printed by `--help`. It stages a clean,
detached Git revision, pins the model revision, places caches/logs/checkpoints
under scratch, validates the exact Slurm request with `--test-only`, and
supports `--resume-submit` only after a terminally unsuccessful allocation.
The exact per-step research telemetry is `NON_PRIVATE` even though model
training follows the configured DP mechanism.

Useful environment overrides include:

```text
PRISM_EPSILON=6
PRISM_DELTA=1e-5
PRISM_CLIP_NORM=1.0
PRISM_BATCH_SIZE=64
PRISM_MICRO_BATCH_SIZE=4
PRISM_STEPS=300                 # Math-10K; GLUE8 template defaults to 500
PRISM_CHECKPOINT_EVERY=10
PRISM_TELEMETRY_MODE=research_raw
PRISM_SLACLIP_TARGET_NON_SMALL_CLIP_FRACTION=0.5
PRISM_SLACLIP_ETA=0.5
PRISM_SLACLIP_C_MIN=0.1
PRISM_SLACLIP_C_MAX=50
PRISM_SLACLIP_NUM_SLOTS=0
```

For the baseline, `PRISM_CLIP_NORM` is the fixed clipping threshold. For
SlaClip, it is the initial threshold `C_0`; SlaClip then adapts it.
`PRISM_SLACLIP_TARGET_NON_SMALL_CLIP_FRACTION` is full SlaClip's `rho`, the
target clipped fraction within the noisy residual/non-small mass. Its
paper-default value is 0.5, and `PRISM_SLACLIP_BETA` is only the legacy name.
The dynamic global target is
`Proj_[0,1](1-rho*(1-s_hat_K/C_t))`, so `rho` is neither a fixed global
clipping rate nor an exact achieved rate. `PRISM_SLACLIP_ETA` is the update
gain. Predeclare custom values before looking at test-set results.

The templates default to `research_raw` because they are intended for internal
training-dynamics analysis. Set `PRISM_TELEMETRY_MODE=dp_safe` for release
candidates.

## 9. Resume and evaluate

Periodic checkpoints are stored inside each model output directory. A Slurm
requeue retains the array job/task identifiers and therefore selects the same
directory. For a fresh submission intended to resume an older run, reuse the
same `PRISM_RUN_ID` and the same task mapping and hyperparameters.

Never resume a directory with changed clipping, SlaClip, privacy, model, data,
or optimizer settings. Inspect the recorded status/config before trusting a
resumed result. Preserve the Slurm log together with `run_status.json`.

After training is complete, run evaluation in separate GPU jobs using the exact
model and result directories and `--run_train false --run_eval true`. GLUE8
evaluation downloads Hugging Face datasets and metrics unless they were staged
in the cache; Math-10K evaluation assets are tracked in this repository.

## 10. Privacy and artifact handling

With `research_raw`, model updates still use the configured DP mechanism, but
the diagnostic log is not a DP release. It is written below:

```text
<result_dir>/research_raw/NON_PRIVATE_train_log.jsonl
```

Keep that directory access-controlled. Do not upload it to GitHub, attach it to
a public release, or claim that the model-plus-raw-log bundle is DP. Release
only artifacts whose status says `telemetry_mode: dp_safe` and
`non_private_telemetry: false`.

For every formal run, retain:

- Git commit and clean/dirty state;
- Slurm job and array task IDs;
- package versions, Python, CUDA, driver, and GPU model;
- model ID or local snapshot plus resolved Hugging Face revision;
- dataset SHA-256;
- complete arguments and output paths;
- final epsilon/delta and update count;
- adapter rank and any spectral-rebase metadata.
