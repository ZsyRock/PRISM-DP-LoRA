# Default-baseline quantile target screen

The completed default-baseline inventory is in
[baseline_audit_2026-09-07.md](baseline_audit_2026-09-07.md). This next screen
implements the requested empirical 25th, 50th and 75th percentile targets,
not the preceding first/last-window weighted recipe.

## Locked experiment

| GLUE8 / Gemma-3-4B-pt setting | Default fixed C | Tuned fixed C / SlaClip C0 | Q25 rho | Q50 rho | Q75 rho |
|---|---:|---:|---:|---:|---:|
| epsilon 6, rank 32 | 1 | 15 | 0.4909090909090909 | 0.5410800385728062 | 0.6068788171006108 |
| epsilon 6, rank 8 | 1 | 40 | 0.4444444444444444 | 0.5 | 0.5488402678144428 |
| epsilon 6, rank 16 | 1 | 30 | 0.4633017163504969 | 0.5096189419163892 | 0.5689655172413793 |
| epsilon 3, rank 16 | 1 | 30 | 0.4741902834008097 | 0.5250069463739928 | 0.5797101449275363 |

Each setting has five fresh arms: two fixed controls and three Full SlaClip
targets. All twenty arms use seed49, 150 updates, batch64/microbatch4, LR2e-4,
length384, delta1e-5, LoRA alpha16/dropout0.05 and the same public 800-record
selection holdout (9200 training records). The accountant calibrates each
150-update run independently to its epsilon budget. The Full SlaClip arms use
eta0.05, K15 and C bounds [0.1,100]. No SlaClip-Q arm is included.

Sources are the complete, default C1, seed42, 500-step trajectories. Quantiles
are linear empirical quantiles with one observation per logical update and no
burn-in exclusion. They are not quartile positions in the min/max interval,
nor quantiles of the private noisy CDF estimate. Derived rho is passed directly
to the Full controller: z=s_hat_K/(C+1e-6), gamma=clip(1-rho*(1-z),0,1),
C_next=clip(C*exp(eta*(gamma-s_hat_1)),0.1,100). The code does not divide rho by
1-z. Raw statistics are research-only, NON_PRIVATE diagnostic artifacts.

The stronger fixed C values are from the independently locked 200-step
calibration. They are comparators, NOT the source of this screen's targets.
These fixed values are not claimed to maximize 150-step utility. Same-seed
150-step fixed controls are rerun so neither a historical seed nor a different
training budget can create an apparent improvement.

The six Math default configurations all have Q25=Q50=Q75=1. They are recorded as
deferred, degenerate target sources, not silently replaced by 99% and not
scheduled three times. Their baseline coverage remains complete. Exploring
Math away from saturation would require a separately declared fixed-C
calibration, not reinterpreting an uninformative percentile as a distinct target.

## Interpretation and next gate

All source raw logs, model/data revisions, earlier fixed-selection manifest and
artifact hashes are rechecked before submission and at compute startup. The
derived source lock, twenty-arm manifest and sequential plan are immutable.

The primary comparison is against tuned fixed C. The secondary comparison is
against C1; because SlaClip starts at the tuned C, that secondary comparison is
not an isolated test of adaptivity. A win against C1 alone is insufficient to
claim an improvement over a well-tuned fixed threshold. Previous weighted
screens found lower tuned-derived targets more promising; this profile tests
the user's default-derived quartile hypothesis without promising a win.

Validation loss is measured at updates0/50/100/150 on the same public holdout.
Full and late loss-curve areas, clipping trajectories, all K CDF coordinates,
reference-CDF bias, C bounds, clipping bias/noise and epsilon are retained. The
screen does not evaluate official task utility and is not journal confirmation.
After screening, lock a candidate and evaluate full500-update runs, with noise
recalibrated from the start, multiple fresh seeds and official GLUE metrics.
Do not use test metrics for iterative target selection. Report null outcomes.

## Execution and outputs

```bash
PRISM_COVERAGE_PROFILE=glue-quantile-target-screen \
  bash scripts/submit_paper_coverage_campaign.sh --test-only
PRISM_COVERAGE_PROFILE=glue-quantile-target-screen \
  bash scripts/submit_paper_coverage_campaign.sh --submit
```

Use the canonical wrapper only from a clean, committed checkout. It creates a
SHA-named immutable source snapshot and a receipt containing the real job ID.
One allocation runs all twenty arms sequentially: one A100, eight CPUs, 80G
host RAM, 24h. No arrays or child sbatch calls. Estimated runtime is20–22h
after queueing, not a guarantee; checkpointing remains enabled. CUDA/synthetic
and two-step real fixed/Full smokes execute at the front of that allocation.

Paths are resolved by the existing portable PRISM_* variables. The default
campaign root is `$SCRATCH/runs/prism-dp-lora/campaigns/paper-coverage-<SHA>-glue-quantile-target-screen-v2`.
Look for `selection/quantile-sources.lock.json`, `plans/manifest.json`,
`artifacts/quantile_target_comparison.json`,
`artifacts/baseline_telemetry_steps.csv`, `artifacts/public_validation_curve.csv`
and per-arm `results/research_raw/` and `results/validation/` outputs.
