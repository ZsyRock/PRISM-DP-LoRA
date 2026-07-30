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

## 8. Submit paired arrays

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
