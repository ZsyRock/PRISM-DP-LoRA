# Default baseline audit: 2026-09-07

The initial coverage stage is complete for the ten unique **PRISM DP utility
configurations** in paper Tables 2–4. It is not a reproduction of every other
optimizer, non-private row, or appendix diagnostic. Each included run completed
its planned updates and full task evaluation. Four interrupted historical
attempts are superseded by the complete runs listed in the companion CSV.

## Utility comparison

Paper references are [Tables 2–4](https://arxiv.org/html/2606.00944v1#S4).
The two epsilon-3 suite averages are calculated from the rounded task values in
Table 2; the other paper values are reported suite averages. Differences below
are percentage points on the suite score, not necessarily accuracy points.

| Dataset | Model | Epsilon | Rank | Paper Average | Local Average | Difference (pp) |
|---|---|---:|---:|---:|---:|---:|
| GLUE8 | Gemma-3-4B-pt | 6 | 8 | 0.74400 | 0.76624 | +2.224 |
| GLUE8 | Gemma-3-4B-pt | 6 | 16 | 0.74000 | 0.76888 | +2.888 |
| GLUE8 | Gemma-3-4B-pt | 6 | 32 | 0.74000 | 0.75262 | +1.262 |
| GLUE8 | Gemma-3-4B-pt | 3 | 16 | 0.73450 | 0.76200 | +2.750 |
| Math-10K | Gemma-3-4B-pt | 6 | 8 | 0.56500 | 0.55905 | -0.595 |
| Math-10K | Gemma-3-4B-pt | 6 | 16 | 0.59000 | 0.57172 | -1.828 |
| Math-10K | Gemma-3-4B-pt | 6 | 32 | 0.56600 | 0.55894 | -0.706 |
| Math-10K | Gemma-3-4B-pt | 3 | 16 | 0.57075 | 0.57362 | +0.287 |
| Math-10K | Gemma-2-9B | 6 | 16 | 0.70190 | 0.69717 | -0.473 |
| Math-10K | Gemma-3-12B-pt | 6 | 16 | 0.69660 | 0.71043 | +1.383 |

These are broadly similar suite-level utilities, not numerical equivalence or a
multi-seed reproduction certificate. Task-level discrepancies can be larger:
GLUE epsilon-6/rank-16 STS-B is 0.83168 locally versus 0.718 in Table 2, while its
Math counterpart has AQuA 0.40551 versus 0.445. GLUE averages mix different task
metrics. Similar averages do not establish identical clipping dynamics.

## Complete clipping records and quantile definition

All ten raw telemetry files were read independently. Four GLUE runs each contain
exactly steps 1–500; six Math runs each contain exactly steps 1–300: **3,800
contiguous logical-update records**, no missing or duplicated step, and C=1 at
every step. Raw telemetry also contains loss, gradient-norm histograms and
quantiles, clipping coefficients, clipping-bias and noise norms, signal-to-noise
ratios, and epsilon spent. These support training-trajectory plots. Official
task metrics are endpoint evaluations, not per-step accuracy curves.

The quantiles below use **all steps, no burn-in exclusion**, with equal weight
per logical update. For sorted clipping fractions x and h=(N-1)p, interpolate
linearly between floor(h) and ceil(h). The input is `raw_clip_fraction` (actual
hard clipping count divided by the realized batch count), not the noisy first
CDF bin, not a time-position percentile, and not an evenly spaced point between
the observed minimum and maximum.

| Setting | Steps | Clipping Min–Max (%) | Q25 (%) | Q50 (%) | Q75 (%) |
|---|---:|---:|---:|---:|---:|
| GLUE8 / 4B / epsilon 6 / rank 8 | 500 | 32.35–100.00 | 44.4444 | 50.0000 | 54.8840 |
| GLUE8 / 4B / epsilon 6 / rank 16 | 500 | 27.59–100.00 | 46.3302 | 50.9619 | 56.8966 |
| GLUE8 / 4B / epsilon 6 / rank 32 | 500 | 31.34–100.00 | 49.0909 | 54.1080 | 60.6879 |
| GLUE8 / 4B / epsilon 3 / rank 16 | 500 | 32.76–100.00 | 47.4190 | 52.5007 | 57.9710 |
| Math-10K / 4B / epsilon 6 / rank 8 | 300 | 100.00–100.00 | 100.0000 | 100.0000 | 100.0000 |
| Math-10K / 4B / epsilon 6 / rank 16 | 300 | 100.00–100.00 | 100.0000 | 100.0000 | 100.0000 |
| Math-10K / 4B / epsilon 6 / rank 32 | 300 | 95.95–100.00 | 100.0000 | 100.0000 | 100.0000 |
| Math-10K / 4B / epsilon 3 / rank 16 | 300 | 100.00–100.00 | 100.0000 | 100.0000 | 100.0000 |
| Math-10K / 9B / epsilon 6 / rank 16 | 300 | 100.00–100.00 | 100.0000 | 100.0000 | 100.0000 |
| Math-10K / 12B / epsilon 6 / rank 16 | 300 | 100.00–100.00 | 100.0000 | 100.0000 | 100.0000 |

Default-derived quantile experiments are therefore meaningful for the four GLUE
settings. All six Math settings yield the same degenerate triple (1,1,1): do not
schedule three duplicate rho=1 arms or silently replace them by 0.99. Their
saturation is an observed outcome of this implementation; it does not prove a
universal SlaClip failure threshold. A distinct non-saturated Math fixed-C
calibration would be needed to supply informative target candidates there.

The target is used directly as conditional rho, with no inverse small-mass
adjustment: global target = rho*(1-z_t), gamma = 1-rho*(1-z_t). Baseline clipping
statistics are a heuristic source of rho, not a guarantee that the resulting
global clipping fraction will match the same numerical percentile.

## Tuned-fixed sources: separate, not silently substituted

The previously locked, loss-selected fixed baselines each have 200 contiguous
steps. Their full-trajectory quantiles differ substantially from the defaults:

| Setting | Selected Fixed C | Q25 (%) | Q50 (%) | Q75 (%) |
|---|---:|---:|---:|---:|
| GLUE8 / epsilon 6 / rank 8 | 40 | 24.1578 | 29.4495 | 34.7538 |
| GLUE8 / epsilon 6 / rank 16 | 30 | 7.7825 | 11.7544 | 17.9011 |
| GLUE8 / epsilon 6 / rank 32 | 15 | 12.5000 | 19.4237 | 26.0234 |
| GLUE8 / epsilon 3 / rank 16 | 30 | 6.6393 | 11.2588 | 17.2580 |

These selected fixed thresholds are useful stronger comparators. They are not
the source of the requested default-derived quantiles and are not established
global utility optima. Stage-one selection used 200-step public validation loss;
new 150-step results must be compared only against fresh matched 150-step arms.

## Fidelity and privacy qualifications

The local defaults share the paper's core dataset/model, rank, privacy budget,
batch, learning-rate, sequence-length and update-count settings. Their completed
configs show PRV accounting, seed 42, LoRA alpha 16/dropout 0.05, batch 64 and
microbatch 4. GLUE uses all 10,000 training rows and official validation tasks;
Math uses 9,919 training rows and complete component tests. No fast-dev evaluation
subset is used.

The documented optimizer is the **hardened public-code variant**, with scalar
floor 0.5 and no second-moment debias. The manuscript also describes an
additive geometry-dependent floor/debias formulation. Local causal-LM losses
are per-record normalized and microbatch invariant, unlike the released
batch-token reduction. Do not describe this as bitwise upstream reproduction or
silently change this protocol within a fixed/SlaClip comparison. Model/data
revisions are pinned locally; exact correspondence to unreleased paper run
artifacts is not established.

Appendix C.1 describes Math-10K gauge diagnostics with 300 updates and gauges
0.25/0.5/1/2/4; its Figure 10 interpretation reports low late PRISM clipping.
However, it does not supply a complete diagnostic run configuration establishing
the loss reduction, numerical optimizer options and numeric intrinsic threshold
used for that figure. The general Appendix B defaults are not independent
verification of those missing diagnostic details. Thus that trajectory is **not
verified as replicated here**, but this audit does not establish a like-for-like
contradiction or identify it as a fault in the completed default runs. See
[Appendix C.1](https://arxiv.org/html/2606.00944v1#A3.SS1).

`NON_PRIVATE_train_log.jsonl` records research-only raw statistics. These files
and target selection from them are not automatically covered by the training
run's epsilon. A joint-noise CDF controller's per-run accounting must not be
presented as end-to-end privacy for raw telemetry release and repeated searches.

## Sources and reproducibility

The compact, plot-ready backup
[`data/baseline_trajectories_2026-09-07.csv`](data/baseline_trajectories_2026-09-07.csv)
contains all 3,800 logical-update rows and is versioned with this report.
It includes scalar clipping/C/loss/norm-quantile/bias/noise statistics and original
raw-file hashes. Full histogram arrays remain in the original scratch artifacts;
older unavailable scalar fields are blank rather than fabricated. This CSV is
explicitly marked NON_PRIVATE research telemetry, not a sanitized DP release.
Rebuild it with `scripts/export_baseline_trajectories.py`, an explicitly configured
campaign root, `--source-landscape` to verify raw hashes, and
`--acknowledge-public-research-telemetry`. The exporter rejects incomplete or
inconsistent sources and refuses to overwrite differing exports.

The companion `baseline_audit_2026-09-07.csv` gives full-precision utilities,
clipping quantiles, source campaign and run-relative path for all ten defaults.
Resolve those paths under the configured scratch campaign root, rather than
hardcoding a username. Each raw file is at
`<campaign>/<relative_run_root>/results/research_raw/NON_PRIVATE_train_log.jsonl`;
plot-ready scalar CSV is `results/research_raw/telemetry_steps.csv`; task scores
are `results/summary.csv`; execution configuration is `adapter/run_status.json`.

The pre-existing source inventory is
`analysis/baseline-target-readiness-20260905-weighted-v1/baseline_landscape.json`.
Its post-burn-in quantiles are intentionally different from this audit's
all-step quantiles. Tuned-source identifiers and hashes are in
`campaigns/paper-coverage-7199f4eb4002-glue-weighted-target-screen-v2/selection/weighted-targets.lock.json`.
