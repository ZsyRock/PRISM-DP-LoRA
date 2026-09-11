# Default-baseline clipping-rate interval screen

## Definition and distinction from the completed screen

This campaign implements the user's min/max **clipping-rate interval** recipe,
not a gradient-norm CDF quantile and not an empirical percentile of the
step-by-step clipping-rate samples:

```
r_min = min(raw_clip_fraction_t for t in complete default baseline)
r_max = max(raw_clip_fraction_t for t in complete default baseline)
rho_q = r_min + q * (r_max - r_min), q in {0.25, 0.50, 0.75}
```

The full 500-step, seed42, C=1 GLUE trajectories are used with no burn-in
exclusion. In statistical terms these are quartile positions of the uniform
min/max interval; they do not account for how often a clipping rate occurred.
For a 70%--100% interval they yield 77.5%, 85%, 92.5%, even if most steps
were near 70%. Therefore they are sensitive to extreme transient steps.

The completed job1529190 used empirical sample quartiles. Its results and
`glue-quantile-target-screen` profile are preserved unchanged. This new profile
is `glue-range-target-screen`; neither its targets nor its results should be
relabeled as the older empirical experiment.

| Setting | Baseline Min (%) | Baseline Max (%) | Range P25 rho (%) | Range P50 rho (%) | Range P75 rho (%) | Tuned Fixed C |
|---|---:|---:|---:|---:|---:|---:|
| GLUE8 / 4B / epsilon6 / rank8 | 32.352941 | 100 | 49.264706 | 66.176471 | 83.088235 | 40 |
| GLUE8 / 4B / epsilon6 / rank16 | 27.586207 | 100 | 45.689655 | 63.793103 | 81.896552 | 30 |
| GLUE8 / 4B / epsilon6 / rank32 | 31.343284 | 100 | 48.507463 | 65.671642 | 82.835821 | 15 |
| GLUE8 / 4B / epsilon3 / rank16 | 32.758621 | 100 | 49.568966 | 66.379310 | 83.189655 | 30 |

Full-precision targets are computed from pinned source extrema and rechecked
against hashed baseline artifacts at submission and compute-node startup.
Source provenance is retained in `selection/range-sources.lock.json`.

## Matched arms and privacy scope

Each of the four settings runs five fresh seed50 arms: fixed C=1, the existing
selected tuned-fixed C, and Full SlaClip with the three interval-derived rho
values. **Every adaptive arm starts at C0=1**, matching the default fixed
baseline. This differs from the earlier empirical screen, whose adaptive arms
started at the tuned-fixed value; historical differences are therefore not a
controlled ablation of target construction alone.

All twenty arms use Gemma-3-4B-pt at the pinned revision, 150 updates,
batch64/microbatch4, LR2e-4, length384, delta1e-5, LoRA alpha16/dropout0.05,
the same 9200 training records and 800 public validation records (split seed
1729). Each run calibrates noise for its actual 150-step budget. Full SlaClip
uses eta0.05, K15, C bounds[0.1,100]. No SlaClip-Q arm is run.

The optimizer's existing joint Gaussian mechanism is unchanged. The controller
uses the interval-derived rho directly, without division by 1-z:

```
z_t = noisy_last_indicator / (C_t + 1e-6)
gamma_t = clip(1 - rho_q * (1 - z_t), 0, 1)
C_next = clip(C_t * exp(eta * (gamma_t - noisy_first_indicator)), 0.1, 100)
```

The whole-batch target proxy is rho_q*(1-z_t), subject to projection, and is
not guaranteed to equal rho_q or the exact realized clipping count. First-bin
CDF feedback is a bin-averaged surrogate, not an exact count.

Raw baseline telemetry and parameter selection from it are NON_PRIVATE
research diagnostics. This screen does not claim end-to-end DP for publishing
raw telemetry or repeated hyperparameter search; the training accountant's
per-run epsilon is not a privacy budget for those operations.

## Selection and journal interpretation

Record validation loss at steps0/50/100/150. For each setting, select the best
of the three adaptive candidates by **step150 validation loss**, breaking ties
by lower rho then candidate ID. Preserve all three candidates, both fixed
controls, loss-curve areas, C trajectories, clipping ratios, K-bin CDFs,
clipping bias, noise, gradient norms and bound-hit diagnostics.

`artifacts/range_target_comparison.json` reports each candidate against both
fixed controls. `selection/best-range-targets.lock.json` freezes the four
selected configurations, full rankings and source/result hashes. A winner is
selected even if all three candidates lose to fixed; it must not be presented
as a positive result in that case. No test metric is used for selection.

These are short, single-seed exploratory results, **not official accuracy or
journal confirmation**. Best-of-three validation performance has selection
bias. Independent full-length, fresh-seed task evaluation is required before
claiming reproducible utility improvements. An improvement over C1 does not
establish superiority over the tuned-fixed benchmark or an unknown global
fixed-C optimum. Higher interval targets are not assumed to outperform the
previous lower empirical targets.

## Explicit Math scope

The first-stage audit has ten baseline configurations. This 24-hour allocation
prioritizes the four GLUE settings; it does not complete SlaClip coverage of all
ten. The other six remain listed, not silently omitted:

- Five Math configurations have min=max=1, so all interval targets equal1;
  they do not provide three distinct targets. Do not substitute0.99.
- Math / 4B / epsilon6 / rank32 has min=0.9594594594594594, max=1,
  yielding rho=[0.9695945945945945,0.9797297297297297,0.9898648648648649].
  This source is **not degenerate**:13 of300 steps are below1. It is deferred
  because the interval is close to saturation and adding another matched
  dataset screen leaves little runtime margin in the requested24h allocation.
  It remains a legitimate future test of the literal interval recipe.

## Single-allocation execution

```bash
PRISM_COVERAGE_PROFILE=glue-range-target-screen \
  bash scripts/submit_paper_coverage_campaign.sh --test-only
PRISM_COVERAGE_PROFILE=glue-range-target-screen \
  bash scripts/submit_paper_coverage_campaign.sh --submit
```

The portable wrapper derives account paths, requires a clean committed source,
creates a SHA-locked snapshot, validates all inputs before queueing, and asks
for one A100,8CPUs,80G host RAM,24h. All20arms run sequentially in one Slurm
allocation, no array or separate child submissions. The previous20-arm shape
took18h10; allow roughly18--22h after allocation, not a guaranteed deadline.

The same allocation begins with CUDA/synthetic checks and two-step real-model
fixed/Full smokes (C0=1,rho=.65 for Full). Passing login-node tests and
`sbatch --test-only` does not establish GPU compatibility before these run.
Checkpoint every25steps; existing unsuccessful-campaign resume is available.
Another conversation's DP-LoRA jobs are independent and must not be changed.

Logs, checkpoints and plot-ready artifacts live under the dynamically
resolved scratch campaign root. This tracked protocol and source/target audit
are the persistent backup; selected final outputs should also be backed up
after completion because scratch is not a backup service.

## Login-node verification on 2026-09-11

- Full repository regression:473tests passed (14existing Opacus/backward-hook
  warnings), including range derivation, malformed inputs, source integrity,
  matched-arm validation, best-of-three selection and unchanged old profiles.
- CPU synthetic baseline/Full pair:3updates, C0=1,rho=.65,eta=.05,K15,
  C bounds[.1,100]; fixedC stayed1, FullC changed, and the accountants agreed.
  This toy budget is not the epsilon of any formal experiment.
- Existing environment `prism-dp-lora-71bce55` reused without dependency changes;
  `pip check`, Python compilation, shell syntax and `git diff --check` passed.
- Login-node preflight:0failures. Model-hub authentication is absent; this
  campaign uses already staged pinned offline weights, and the submission
  wrapper requires their completion marker. No new credentials are needed.
  Runtime versions:Python3.11.15,torch2.10.0+cu128,transformers5.1.0,
  peft0.18.1,opacus1.6.0. The visible login-environment L4 devices are not
  evidence that the requested A100 compute-node smoke has run.
- Live quota snapshot:home35/130GB,scratch1330/2000GB. Live Slurm association
  permits account/QoS normal; a100 partition max60h covers the requested24h.
- Historical raw-source/fixed-selection hashes were reverified independently;
  range-target content SHA256 is
  `65dd9e7267f04b5ad0855d007992682e9614e5bc4532cf1a5ecdad2e7cdfdc6a`.
- GPU and real-model smoke checks are guarded inside the one requested
  allocation, not claimed as prevalidated on this login node.
