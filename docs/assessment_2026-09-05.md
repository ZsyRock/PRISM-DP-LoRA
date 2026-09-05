# PRISM baseline coverage and next experiment, 2026-09-05

Job 1492016 completed all ten calibration arms successfully on 2026-09-02 at
18:57 BST (13h08m). The paper's ten unique PRISM DP configurations already
have complete C=1, seed42 runs. Four incomplete historical attempts were
superseded; they do not represent missing configurations.

Paper source: https://arxiv.org/html/2606.00944v1 (Tables 2–4 and Appendix B.1).
GLUE8 is in the paper: eight tasks, not eight independently fine-tuned models.
Math-10K has four evaluation tasks: GSM8K, AQuA, MAWPS, SVAMP. The paper does
not report GLUE8 with 9B/12B. Other optimizer methods and Non-DP paper rows are
outside this PRISM fixed-C trajectory census.

All runs match the documented default schedule: batch64/microbatch4, delta1e-5,
LoRA alpha16/dropout0.05, C1. GLUE uses 500 updates, LR2e-4, length384,
response-only loss; Math uses 300 updates, LR3e-4, length256, input supervision.
The implementation uses the documented hardened public-code protocol: scalar
floor0.5/no second-moment debias and record-normalized loss. These differ from
the manuscript additive-floor geometry and upstream batch-token-mean loss;
coverage is complete but exact original numerical reproduction is not claimed.

## Complete default baseline census

Clipping quantiles use steps51..T. Utility is the task-average score, not a
single homogeneous accuracy metric across GLUE's tasks.

| Dataset | Model | Epsilon | Rank | Task-average utility | Clipping P10–P90 | Median clipping |
|---|---|---:|---:|---:|---:|---:|
| GLUE8 | Gemma-3-4B-pt | 6 | 8 | 0.76624 | 40.35–57.38% | 49.09% |
| GLUE8 | Gemma-3-4B-pt | 6 | 16 | 0.76888 | 41.88–58.47% | 50.00% |
| GLUE8 | Gemma-3-4B-pt | 6 | 32 | 0.75262 | 43.86–63.80% | 52.90% |
| GLUE8 | Gemma-3-4B-pt | 3 | 16 | 0.76200 | 43.33–60.03% | 51.67% |
| Math-10K | Gemma-3-4B-pt | 6 | 8 | 0.55905 | 100–100% | 100% |
| Math-10K | Gemma-3-4B-pt | 6 | 16 | 0.57172 | 100–100% | 100% |
| Math-10K | Gemma-3-4B-pt | 6 | 32 | 0.55894 | 100–100% | 100% |
| Math-10K | Gemma-3-4B-pt | 3 | 16 | 0.57362 | 100–100% | 100% |
| Math-10K | Gemma-2-9B | 6 | 16 | 0.69717 | 100–100% | 100% |
| Math-10K | Gemma-3-12B-pt | 6 | 16 | 0.71043 | 100–100% | 100% |

Math/rank32 has rare nonsaturated steps (minimum95.95%); other Math trajectories
are100% at every step. These observations cannot identify a universal SlaClip
failure threshold or imply that small models have no useful long-tail gradients.

## Direct weighted targets

Use rho=.3*mean(first25 hard clipping fractions)+.7*mean(last25), directly as
conditional rho. These values are hypotheses, not exact global rate promises.

| GLUE setting | Early mean | Late mean | Direct weighted rho |
|---|---:|---:|---:|
| Epsilon6/rank8 | 90.72% | 44.22% | 58.17% |
| Epsilon6/rank16 | 94.83% | 47.76% | 61.88% |
| Epsilon6/rank32 | 97.89% | 48.22% | 63.12% |
| Epsilon3/rank16 | 97.68% | 49.31% | 63.82% |

The controller already implements u=indicator[0], z=indicator[-1]/(C+1e-6),
gamma=clip(1-rho*(1-z),0,1), C_next=clip(C*exp(eta*(gamma-u)),bounds).
An early100%/late70% trajectory gives rho.79; at z.2 the global clipped proxy
target is.632. The first coordinate must not be divided by C again. The new
recipe intentionally does not use the old inverse p/(1-z) conversion.

Fixed-C raw-gradient telemetry does not modify the baseline gradient query.
Full SlaClip fills remaining per-record norm space with slack coordinates;
the gradient marginal retains the same clipping and Gaussian distribution at
the same current C. C affects subsequent noise scale as intended. The single-run
accountant and the complete raw-telemetry-driven hyperparameter search have
different scopes; only the former has the per-run epsilon guarantee.

The analysis SNR bug was corrected: z needs sigma_z=sigma_indicator/(C+1e-6).
The previous denominator understated signal/noise by about C at large C.
Historical artifacts remain unchanged; corrected evidence is stored separately.
The old SNR>=2 eligibility filter is a screening heuristic, not a theorem
about controller success. A small last-coordinate signal can coexist with a
well-resolved first-coordinate CDF. This new exploratory profile therefore
records the diagnostics without treating that heuristic as a mandatory gate.

## New staged screen

| Stage | Settings | Candidates | Seed | Steps | Arms |
|---|---|---|---:|---:|---:|
| Fixed boundary calibration | Epsilon6/rank8 | C15,40,80 | 47 | 200 | 3 |
| Fixed boundary calibration | Epsilon6/rank16 | C15,30,60 | 47 | 200 | 3 |
| Matched Full-SlaClip screen | All four GLUE settings | Selected fixed; default-derived rho; tuned-derived rho | 48 | 150 | 12 |

Epsilon6/rank32 reuses C15 and epsilon3/rank16 C30 from job1492016's calibration.
Their endpoint validation losses were0.261466 and0.252440, respectively.
The two other fixed C values are selected automatically from Stage1 using
public validation loss. The second rho per setting follows the same weighted
recipe on that selected fixed trajectory. Both adaptive arms start at the
same C as their matched fixed control; eta.05, K15, bounds[.1,100].

The old rank8/rank16 adaptive runs often hit C_max15, and the fixed winners
were themselves at15. Expanding this boundary is necessary before attributing
their negative results solely to slack quality. The new larger C bound is
recorded explicitly. If the new fixed winner is again at an edge, the result
remains a boundary-limited screen requiring further fixed calibration.

Compare only Stage2's same-seed, same-150-step, same-epsilon fixed and adaptive
arms. Do not compare them to Stage1's200-step values. A200-step-selected C is
not necessarily the best150-step C. This job explores candidates and produces
loss/CDF/clipping/bias/noise trajectories; it cannot establish final task
accuracy superiority. Next confirmation needs500 steps and multiple fresh seeds.

All18 arms run sequentially within one A100 allocation,8 CPUs,80G host RAM,
24h. Existing timings suggest roughly19–22h plus queueing; the estimate is not
a guarantee. Model snapshots and datasets are already cached. Real GPU smokes
run before the long arms inside this allocation.

Login validation: 346 repository tests passed, dependency check passed, all
three submission/lane scripts passed shell syntax checks, and the offline
GLUE data/model preflight reported zero failures. The two-step real-Gemma
fixed/adaptive smoke remains an in-job guard, not a pre-submission GPU claim.

## Broader journal evidence

Prioritize full-model DP-SGD/DP-Adam on CIFAR-10 (Opacus), full Transformer DP
fine-tuning on SST-2/QNLI (private-transformers), and the Andrew et al. private
quantile-clipping comparator with matched total privacy cost. These add a
vision task family, full-model gradient spaces, and a direct adaptive-clipping
competitor. DP-AdamBC and DP-BiTFiT are additional options. Do not equate more
methods with publication readiness; report matched tuned-fixed controls,
uncertainty across seeds, negative results, and the privacy scope of selection.

Primary links:

- https://github.com/meta-pytorch/opacus
- https://arxiv.org/abs/2110.05679
- https://github.com/lxuechen/private-transformers
- https://proceedings.neurips.cc/paper/2021/hash/91cff01af640a24e7f9f7a5ab407889f-Abstract.html
- https://github.com/tensorflow/privacy/blob/master/tensorflow_privacy/privacy/dp_query/quantile_adaptive_clip_sum_query.py
- https://arxiv.org/abs/2312.14334
- https://github.com/ubc-systopia/DP-AdamBC
- https://proceedings.mlr.press/v235/bu24c.html
- https://github.com/awslabs/fast-differential-privacy

The separately managed local DP-LoRA-paper-repro README identifies a federated
aggregate-gradient reconstruction (arXiv2312.17493), marks epsilon=null /
NOT_CERTIFIED, and does not claim a complete downstream-benchmark reproduction.
It should not be counted as a second certified sample-level DP result without
closing that separate accounting/reproduction gap. This task did not change
that repository or download any new baseline repository.
