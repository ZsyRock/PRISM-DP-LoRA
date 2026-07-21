# Telemetry schema and privacy classification

This document describes the per-update JSONL artifacts emitted by the current trainer. Field names are the implementation contract; downstream analysis should also check `telemetry_schema_version`, `run_id`, and `config_fingerprint` rather than joining files only by line number.

## Artifact layout

| Artifact | Present in | Privacy classification | Intended use |
|---|---|---|---|
| `<output_dir>/train_log.jsonl` | all runs | DP-safe for DP runs, subject to the assumptions below | progress, accounting, threshold trajectory, and post-processing diagnostics |
| `<result_dir>/research_raw/NON_PRIVATE_train_log.jsonl` | `research_raw` only | **non-DP** | trusted internal training-dynamics analysis |
| `<output_dir>/run_status.json` | all runs | configuration/status metadata | identity, completion, accounting, and reproducibility |
| `<result_dir>/run_status.json` | completed runs | mirrored configuration/status metadata | binds evaluation output to the adapter |

`research_raw` requires both `--telemetry_mode research_raw` and `--allow_non_private_telemetry`. Its directory also contains `README_NON_PRIVATE.txt`. The raw record copies the DP-safe fields before adding exact diagnostics, so it can be analyzed without a positional join to `train_log.jsonl`.

## Common identity and context

The main records contain fields such as:

| Field | Meaning |
|---|---|
| `telemetry_schema_version` | schema version for parser compatibility |
| `run_id` | readable label, clipping threshold, and short configuration hash |
| `config_fingerprint` | canonical full experiment fingerprint |
| `method` | `baseline` or full `slaclip` |
| `privacy` / `telemetry_mode` | training and observation modes |
| `step` | completed logical update count |
| `base_model` | requested model ID or local snapshot |
| `model_revision` / `resolved_model_revision` | requested and best-effort resolved model revisions |
| `dataset`, `lora_r`, `lr` | core experiment context |
| `loss_definition` | record-level loss-reduction contract |
| `spectral_*`, `lift_gauge_fix`, `prism_*` | PRISM numerical/update configuration |

The fingerprint, not a hand-written run name, determines artifact compatibility.

## DP-safe per-update fields

These fields are emitted to `train_log.jsonl` during DP training. They are public configuration/accounting values, jointly noised quantities, or post-processing of the DP release/model update.

| Field or group | Meaning |
|---|---|
| `dp_expected_batch_size` | fixed normalization denominator used by the mechanism |
| `dp_noise_multiplier` | calibrated Gaussian noise multiplier |
| `dp_clip_threshold` | threshold `C_t` used for this update |
| `dp_next_clip_threshold` | threshold selected for the next update; equal to `C_t` for baseline |
| `dp_std_per_factor` | base per-factor scale `noise_multiplier * C_t / expected_batch_size` before geometry-aware factor mapping |
| `slack_indicator` | jointly noised Slack Indicator vector; full SlaClip only |
| `slack_unclipped_proxy`, `slack_clipped_proxy` | summaries derived from the noised indicator, not exact clipping fractions; full SlaClip only |
| `slaclip_gamma_t`, `slaclip_eta`, `slaclip_beta` | full-controller update values; present when SlaClip performs an adaptive update |
| `slaclip_num_slots` | Slack Indicator dimension `K`; full SlaClip only |
| `dp_noisy_tangent_gradient_norm` | norm of the noised tangent-gradient release |
| `dp_factor_product_update_norm` | exact norm of the resulting low-rank product update, a function of the released model transition |
| `dp_floor*`, `dp_precond_*`, `dp_trust_ratio_*`, `dp_update_clip_coef_min` | numerical optimizer diagnostics computed while post-processing the noised release |
| `eps_spent` | accountant epsilon after this completed update at configured delta |

`dp_safe` intentionally does **not** emit the exact DP-training loss, target-token count, exact per-record norm distribution, exact clipping fraction, unclipped signal, clipping bias, or realized noise decomposition. In particular, it does not log an actual noise norm: revealing output and its exact random-noise decomposition together can reveal the pre-noise signal.

Non-DP runs may put `loss_mean`, `tokens`, and `batch_n` in the main log; that does not make those fields safe for a private dataset. Release classification must always consider `privacy` as well as the filename.

## Exact `research_raw` fields

Every raw record is marked `NON_PRIVATE_TELEMETRY: true` and adds:

| Field | Meaning |
|---|---|
| `loss_mean` | mean of per-record losses for the realized logical batch |
| `tokens` | total valid shifted target-token count |
| `batch_n` | realized record count used for the loss summary |
| `raw_realized_batch_size` | realized size of the Poisson sample (kept out of the DP-safe release) |
| `raw_clip_fraction` | exact fraction of realized records clipped |
| `raw_clip_coefficient_mean` / `raw_clip_coefficient_min` | exact clipping-coefficient summaries |
| `raw_clipped_signal_norm` | norm of the clipped, pre-noise tangent signal after expected-batch normalization |
| `raw_unclipped_signal_norm` | norm of the unclipped tangent signal after the same normalization |
| `raw_clipping_bias_norm` | norm of the difference between unclipped and clipped signals |
| `raw_realized_noise_norm` | norm of the realized tangent noise |
| `raw_signal_to_noise_ratio` | clipped signal norm divided by realized noise norm |
| `raw_global_norm_mean/std/min/max` | exact per-record tangent-gradient norm summaries |
| `raw_global_norm_quantiles` | exact quantiles at 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, and 0.99 |
| `raw_global_norm_hist_counts` / `raw_global_norm_hist_edges` | fixed-edge norm histogram |
| `raw_global_norm_hist_overflow` | records above the histogram's upper edge |

By default the histogram upper edge is `4 * C_0`, fixed for the run, and `raw_hist_bins` defaults to 32. A fixed edge makes distributions comparable across adaptive steps. `raw_hist_max` can set another predeclared fixed edge. Histogram resolution is independent of SlaClip's `K`.

## Loss interpretation

`loss_mean` is not a global mean over all target tokens in a microbatch. For each record, the trainer averages shifted cross-entropy over that record's non-ignored target tokens; it then averages the resulting record losses. This preserves record-level gradient semantics when physical microbatch boundaries or target lengths differ.

The same `loss_definition` string is stored in the main log, raw log, status, and checkpoint. Parsers should reject or separate records with a different definition.

## Safe analysis practice

For release-oriented plots, use only `train_log.jsonl` from a run whose status says `privacy: dp`, `telemetry_mode: dp_safe`, and `non_private_telemetry: false`.

For internal mechanistic analysis, raw fields can be plotted against `step`, `dp_clip_threshold`, or `dp_next_clip_threshold` to study gradient norms, clipping bias, and SNR. Store those analyses with the same access controls as the raw JSONL. Aggregating, plotting, or paraphrasing an exact raw statistic does not automatically make it DP.

Do not use the raw trajectory to choose a releasable checkpoint, threshold, seed, or model unless that data-dependent selection is explicitly included in the privacy analysis. The configured DP mechanism protects the adapter update; it does not retroactively protect a separate observer or a selection rule driven by that observer.

Before release, follow the checklist in [experiment_protocol.md](experiment_protocol.md).
