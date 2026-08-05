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
| `method` | `baseline`, dynamic full `slaclip`, fixed-global-target `slaclip_q`, or deterministic `replay` |
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
| `dp_next_clip_threshold` | threshold selected for the next update; equal to `C_t` for baseline and read from the locked schedule for replay |
| `replay_schedule_index` / `replay_clip_schedule_sha256` | zero-based schedule entry used for this update and identity of the immutable replay input; replay only |
| `dp_std_per_factor` | base per-factor scale `noise_multiplier * C_t / expected_batch_size` before geometry-aware factor mapping |
| `dp_tangent_noise_sampler` / `dp_tangent_query_chart` / `dp_factor_rank_rcond` / `dp_factor_relative_gram_eigenvalue_min` | exact full-rank QR tangent sampler and matching isometric query-chart identifiers, fail-closed numerical-rank tolerance, and the minimum checked relative Gram eigenvalue across all LoRA factors |
| `slack_indicator` | jointly noised Slack Indicator vector; either SlaClip controller only |
| `slack_unclipped_proxy`, `slack_clipped_proxy` | summaries derived from the noised first coordinate, not exact clipping fractions |
| `slack_indicator_noise_std` | public normalized per-coordinate noise s.d. `sigma*sqrt(K)/expected_batch_size` |
| `slaclip_controller` | `slaclip` (dynamic full controller) or `slaclip_q` (fixed-target ablation) |
| `slaclip_target_non_small_clip_fraction` | full-SlaClip `rho`: requested clipped fraction within the residual/non-small proxy mass; 0.5 is the paper default |
| `slaclip_beta` | legacy telemetry alias for the same full-SlaClip `rho`, not a separate gain |
| `slaclip_small_gradient_proxy_noisy` | full-SlaClip paper proxy `z_t=s_hat_K/C_t` before target projection; DP-safe post-processing of the jointly noised Slack Indicator, so it can lie outside `[0,1]` |
| `slaclip_remaining_mass_proxy_noisy` | full-SlaClip residual proxy `1-z_t` before target projection; also DP-safe post-processing of the jointly noised release and not a non-private raw measurement |
| `slaclip_target_unclipped_proxy_preprojection` | full-SlaClip value `1-rho*(1-z_t)` before `Proj_[0,1]` |
| `slaclip_gamma_t`, `slaclip_target_unclipped_proxy` | projected dynamic target-unclipped proxy `gamma_t` for full SlaClip; fixed target for SlaClip-Q |
| `slaclip_target_clipped_proxy` | complement of the projected target-unclipped proxy; for full SlaClip it is dynamic, not equal in general to `rho` |
| `slaclip_observed_unclipped_proxy` | controller observation `s_hat_1`; equivalent to `slack_unclipped_proxy` but named explicitly for the update equation |
| `slaclip_controller_error` | `target_unclipped_proxy - observed_unclipped_proxy`, whose sign determines the threshold-update direction |
| `slaclip_eta` | exponential threshold-update gain; this, not `rho`, controls the response step size |
| `slaclip_target_clip_fraction` | requested fixed global clipped-rate label for SlaClip-Q; 0.99 maps internally to an unclipped-proxy target of 0.01 |
| `slaclip_c_next_unbounded`, `slaclip_c_min/max`, `slaclip_c_hit_min/max` | pre-clamp candidate, declared bounds, and post-processing bound diagnostics |
| `slaclip_num_slots` | Slack Indicator dimension `K`; either SlaClip controller only |
| `dp_noisy_tangent_gradient_norm` | norm of the noised tangent-gradient release |
| `dp_factor_product_update_norm` | exact norm of the resulting low-rank product update, a function of the released model transition |
| `dp_floor*`, `dp_precond_*`, `dp_trust_ratio_*`, `dp_update_clip_coef_min` | numerical optimizer diagnostics computed while post-processing the noised release |
| `eps_spent` | accountant epsilon after this completed update at configured delta |

For full SlaClip, the safe telemetry fields above encode

```text
z_t = s_hat_K / C_t
gamma_t = Proj_[0,1](1 - rho * (1 - z_t))
C_next = clip(C_t * exp(eta * (gamma_t - s_hat_1)), c_min, c_max)
```

The paper-default `rho=0.5` yields its literal `1/2`. A different `rho` remains
full SlaClip because the global target still depends on `s_hat_K`. By contrast,
SlaClip-Q omits `s_hat_K` and supplies a fixed target for `s_hat_1`. Neither the
full conditional `rho` nor the SlaClip-Q requested label is an exact achieved
clipping fraction; exact achievement is available only as non-private
`raw_clip_fraction`. A positive `slaclip_controller_error` increases `C_t` and
tends to reduce clipping; a negative value decreases `C_t` and tends to
increase clipping. The `_noisy` proxy fields belong to `train_log.jsonl` and
must not be confused with `_raw` fields in the explicitly non-private
`research_raw` artifact.

`dp_safe` intentionally does **not** emit the exact DP-training loss, target-token count, exact per-record norm distribution, exact clipping fraction, unclipped signal, clipping bias, or realized noise decomposition. In particular, it does not log an actual noise norm: revealing output and its exact random-noise decomposition together can reveal the pre-noise signal.

The non-private telemetry summarizer reports `slaclip_c_hit_min/max` under `boolean_metrics`, including `true_count` and `true_rate`, so boundary saturation can be audited without treating booleans as ordinary numeric measurements.

The DP interpretation assumes Poisson subsampling with add/remove record adjacency. It also treats benchmark identity, size, and content hash as public auxiliary information. For a genuinely private dataset, a deterministic content hash, the fingerprint/run ID derived from it, and a config/status snapshot containing it are not DP-safe outputs. Resume checkpoints contain sampler/RNG state and are controlled training state, never release artifacts.

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
| `raw_slack_indicator` | exact, unnoised Slack Indicator using the same `lambda=C/sqrt(K)` and expected-batch denominator as the released indicator; adaptive SlaClip arms only |
| `raw_slack_indicator_noise_residual` | per-slot released indicator minus exact indicator; its L2 norm, RMSE, and first-coordinate residual are also recorded |
| `raw_unclipped_clipped_cosine` | cosine between the exact unclipped and clipped tangent signals |
| `raw_clipped_noisy_cosine` | cosine between the clipped signal and its realized noised release |
| `raw_clipping_bias_to_noise_ratio` | clipping-bias norm divided by realized-noise norm |
| `raw_bias_noise_squared_error_proxy` | sum of squared clipping-bias and realized-noise norms; a diagnostic proxy, not an exact squared total error because it omits the cross term |
| `raw_global_norm_mean/std/min/max` | exact per-record tangent-gradient norm summaries |
| `raw_global_norm_quantiles` | exact quantiles at 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, and 0.99 |
| `raw_global_norm_hist_counts` / `raw_global_norm_hist_edges` | fixed-edge norm histogram |
| `raw_global_norm_hist_overflow` | records above the histogram's upper edge |
| `raw_reference_slaclip_num_slots` | telemetry-only `K` used to evaluate the full-SlaClip small-gradient proxy from exact norms; present when a positive reference `K` is configured, including fixed arms in the exploratory scan |
| `raw_reference_expected_batch_size_normalization` | fixed expected-batch denominator used by the corresponding SlaClip DP release (not the randomly realized Poisson batch size) |
| `raw_reference_slack_indicator_last` | exact unnoised final coordinate of the telemetry-only reference Slack Indicator |
| `raw_reference_small_gradient_proxy` | exact reference `z_t=raw_reference_slack_indicator_last/(C_t+1e-6)` matching the implemented full-SlaClip normalization |
| `raw_reference_remaining_mass_proxy` | exact reference residual proxy `1-z_t` |
| `raw_reference_conditional_clip_fraction` | telemetry-only ratio `raw_clip_fraction/(1-z_t)` when the residual proxy is finite and positive; not itself a controller target and not clipped to `[0,1]` |
| `raw_reference_conditional_clip_fraction_valid` | whether that conditional ratio has a finite, strictly positive denominator |

By default the histogram upper edge is `4 * C_0`, fixed for the run, and `raw_hist_bins` defaults to 32. A fixed edge makes distributions comparable across adaptive steps. `raw_hist_max` can set another predeclared fixed edge. Histogram resolution is independent of SlaClip's `K`.

The `raw_reference_*` fields are observer-only counterfactuals. On a fixed arm
they are computed from exact norms after the mechanism's clipping decision;
they do not append Slack coordinates to the DP query, affect the clipped
gradient or Gaussian noise, update `C`, or change accounting. They support the
exploratory mapping from a global target `p` to full SlaClip's conditional
target: since `p*_t=rho*(1-z_t)`, a fixed arm's `raw_clip_fraction` cannot be
used directly as `rho`.

`replay` loads exactly one positive finite threshold per update from
`--clip_schedule_path`. The file-byte SHA256 and provenance metadata are stored
in status, and the SHA participates in the experiment fingerprint and resume
checkpoint identity. Replay uses the baseline gradient mechanism with `C_t`
supplied before each update; it never constructs Slack coordinates for the DP
query. Consequently replay has no controller Slack/CDF fields. A baseline raw
log can contain the explicitly named telemetry-only `raw_reference_*`
counterfactuals when a positive reference `K` is configured; these are not a
Slack release or controller state.

The replay run's accountant is conditional on that locked schedule. If the
schedule was derived from private-data-dependent releases (for example, one or
more earlier SlaClip trajectories), an end-to-end privacy statement must compose
the privacy cost of those source releases with the replay run. Run status records
this scope and propagates the schedule's declared privacy class; it must not be
read as a fresh standalone epsilon claim.

## Loss interpretation

`loss_mean` is not a global mean over all target tokens in a microbatch. For each record, the trainer averages shifted cross-entropy over that record's non-ignored target tokens; it then averages the resulting record losses. This preserves record-level gradient semantics when physical microbatch boundaries or target lengths differ.

The same `loss_definition` string is stored in the main log, raw log, status, and checkpoint. Parsers should reject or separate records with a different definition.

## Safe analysis practice

For release-oriented plots, use only `train_log.jsonl` from a run whose status says `privacy: dp`, `telemetry_mode: dp_safe`, and `non_private_telemetry: false`.

For internal mechanistic analysis, raw fields can be plotted against `step`, `dp_clip_threshold`, or `dp_next_clip_threshold` to study gradient norms, clipping bias, and SNR. Store those analyses with the same access controls as the raw JSONL. Aggregating, plotting, or paraphrasing an exact raw statistic does not automatically make it DP.

Do not use the raw trajectory to choose a releasable checkpoint, threshold,
seed, or model unless that data-dependent selection is explicitly included in
the privacy analysis. The exploratory 4B campaign intentionally uses fixed-arm
raw trajectories to lock `C_transition` and a five-point `p`/`rho` grid, so the
resulting campaign bundle is `NON_PRIVATE` and cannot support a standalone
end-to-end DP claim. A formal claim needs an independently predeclared or
independently confirmed grid, or privacy composition covering the calibration.
The configured DP mechanism protects each adapter update; it does not
retroactively protect a separate observer or a selection rule driven by that
observer.

Before release, follow the checklist in [experiment_protocol.md](experiment_protocol.md).
