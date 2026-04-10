# Stage 11 Summary - Post-Bootstrap Residual-Control De-Saturation Audit

## Stage Goal
Add diagnostics-only instrumentation that explains the new residual failure mode after Stage 10:

- `anchor_beta` vs `effective_beta` is now visible and honest,
- Task 2+ no longer collapses to `effective_beta=1.0`,
- but the residual branch can still drive `delta_raw` into the tanh-flat regime and pin `delta_logit` at the residual cap.

Stage 11 does not change thresholds, bootstrap structural policy, CHU, router behavior, classifier behavior, inference-profile semantics, or soft-rank.

## Stage 10 Closeout
Stage 10 is treated as a partial success:

- hybrid gating is now policy-anchored instead of absolute,
- the anchor prior and learned residual are separated explicitly in logs,
- and post-bootstrap tasks can be distinguished from Task 1 bootstrap saturation.

The remaining unknown is narrower: why the residual branch rapidly becomes cap-limited and why parameter gradients disappear again even when gate/logit gradients remain present.

## What Stage 11 Adds
- `PlannerResidualRaw`
  - `delta_raw` mean/min/max, `p50/p90/p99`
  - fractions `|delta_raw| > 2`, `> 4`, `> 6`
  - tanh-derivative summaries
  - epoch drift and drift from task start
- `PlannerControlHead`
  - control-head weight norm / bias norm
  - control-head weight/bias drift from init
  - contribution split between representation term and bias term
  - bias share of the raw delta
  - correlation between representation norm and `|delta_raw|`
- `PlannerResidualGradients`
  - gradients on `delta_raw`
  - gradients on `delta_logit`
  - bridge-loss fraction where `delta_logit` has signal but `delta_raw` does not
  - loss-output-zero fraction for the effective logit
- `PlannerResidualCap`
  - fraction near positive / negative / any cap
  - mean absolute gap to the cap
  - simple cap-state label: `bounded_active`, `high_but_movable`, or `hard_pinned`
- `PlannerResidualSummary`
  - one-line interpretive summary per layer with:
    - anchor beta
    - effective beta
    - residual cap state
    - slot structural availability
    - shared-vs-slot ratio
    - Layer 6 focus marker

## Files Changed
- `src/models/planner.py`
- `src/engine/trainer.py`
- `tests/test_paper_alignment_units.py`
- `README.md`
- `docs/paper_alignment.md`
- `outputs/dev/stage_11_residual_control_desaturation_summary.md`

## Validation Commands
- `python -m compileall src tests`
- `python -m unittest tests.test_config_summary tests.test_paper_alignment_units tests.test_synthetic_continual_smoke`
- `git diff --check`

## Validation Results
- `python -m compileall src tests` passed.
- `python -m unittest tests.test_config_summary tests.test_paper_alignment_units tests.test_synthetic_continual_smoke` passed.
- `git diff --check` passed with line-ending warnings only.

## Rerun Evidence Required
Minimum rerun target remains Task 3 with these slices:

- Task 1 as bootstrap context only
- Task 2 Epoch 1, 2, 3
- Task 3 Epoch 1

Primary questions for the rerun:

1. Does `delta_raw` enter the tanh-flat regime immediately from Task 2 onward?
2. Is the residual cap becoming `hard_pinned`, or only `high_but_movable`?
3. Are gradients disappearing at the tanh bridge or earlier/later?
4. Is the main driver output-head scale/bias, representation amplitude, or the bounded transform itself?
5. Why can Layer 6 still show useful slot contribution when the residual path is already cap-limited?
6. Does the rerun keep showing that Task 1 bootstrap anchor is context, not the main explanation for Task 2+ collapse?

## Narrowest Next Recommendation
Do not patch thresholds, soft-rank, router, CHU, or classifier next.

Use the Stage 11 rerun to classify one dominant bucket only:

- control-head runaway,
- representation amplitude runaway,
- bounded-transform choke point,
- or mixed case with one primary driver.

Only after that classification should the next fix stage touch residual-update dynamics.
