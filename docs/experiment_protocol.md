# Experiment protocol

## Supported comparison

The primary comparison has exactly two method labels:

1. `baseline`: fixed-threshold PRISM;
2. `slaclip`: full SlaClip controlling the threshold of that same PRISM mechanism.

Do not introduce SlaClip-Q results into the primary tables. For a paired run, keep every field identical except `method` and the fields that are intrinsically SlaClip-only (`slaclip_*`). In particular, match model revision, dataset split, seed, update count, learning rate, LoRA configuration, expected batch size, microbatch size, accountant, epsilon, delta, initial clipping threshold, and evaluation settings.

Dynamic clipping necessarily changes the absolute per-step noise standard deviation because it is `noise_multiplier * C_t / expected_batch_size`. The noise multiplier and privacy accountant remain matched.

## Run levels

### Reproduction/default comparison

- use the paper/default baseline threshold;
- use the same value as SlaClip's initial threshold;
- run at least seeds 42, 43, and 44;
- report mean and standard deviation.

### Validation-budget-matched comparison

- tune the baseline threshold over a predeclared grid, such as `{0.25, 0.5, 1, 2, 4}`;
- give SlaClip the same number of validation-driven choices;
- do not choose a method's configuration using the test set.

## Diagnostic modes

Use `dp_safe` for release candidates. Use `research_raw` only on an access-controlled system and only when exact training-dynamics analysis is required.

Before copying or publishing a run, check:

- `run_status.json` says `telemetry_mode: dp_safe` and `non_private_telemetry: false`;
- no `research_raw/` directory or `NON_PRIVATE_*.jsonl` file is included;
- no exact training losses, norm distributions, or clipping fractions were copied from terminal output;
- the reported epsilon/delta correspond to the released model's update count and accountant;
- training rank and saved adapter rank are reported separately.

For internal raw analysis, retain the explicit warning marker and record who can access the output. The DP model and non-DP diagnostic log should be treated as separate artifacts with separate release policies.
