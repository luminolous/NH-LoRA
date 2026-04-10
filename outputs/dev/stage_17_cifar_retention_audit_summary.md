# Stage 17 -- CIFAR-100 Full-Block Retention Bottleneck Audit

## Summary

Stage 17 keeps NH-LoRA method semantics unchanged and extends the existing
retention/evaluation debug path so CIFAR-100 full-block forgetting can be
decomposed without patching planner behavior, residual gating, CHU, router
behavior, classifier behavior, or loss weights.

## What Changed

- Added task-boundary retention audit helpers in
  `src/engine/trainer.py` for:
  - weighted loss-balance summaries
  - forgetting decomposition against best-prior and latest-prior accuracy
  - old-vs-new classifier calibration on seen-task evaluation batches
  - eval-time route/profile retention summaries
  - Pre-CHU vs Post-CHU forgetting deltas
  - per-layer retention attribution over later tasks
- Extended `_evaluate_up_to(...)` so task-boundary evaluation can optionally
  collect retention diagnostics while preserving the existing accuracy outputs.
- Extended task metrics with:
  - `weighted_loss_balance`
  - `forgetting_decomposition`
  - `current_train_accuracy`

## Logging Added

When `training.retention_debug_logging=true` and `task_number > 1`, the trainer
now emits:

- `RetentionAudit`
- `RetentionOldTask`
- `RetentionCalibration`
- `RetentionRoute`
- `RetentionCHUDiff`
- `RetentionCHUSummary`
- `RetentionLayerAttribution`

These logs are observational only.

## Also Corrected

- `configs/base.yaml` now actually defaults `model.selected_blocks` to
  `[0..11]`, matching the Stage 16 full-block-default intent and the current
  repository docs.
- Config smoke expectations were updated to match the current repo config
  surface.
- The Stage 15 benchmark-transition note was corrected so it no longer points
  to a non-existent `configs/cifar100_hybrid_all_layers.yaml` file.

## Validation

Executed locally:

- `python -m compileall src tests`
- `python -m unittest tests.test_config_summary tests.test_paper_alignment_units tests.test_synthetic_continual_smoke`

Result:

- `83 tests` passed

## Intended Use

Stage 17 is meant to answer whether CIFAR-100 full-block forgetting is driven
primarily by:

- feature drift
- classifier/logit calibration
- eval-time route/profile under-service
- CHU-side consolidation effects
- or a mixed case with one dominant driver

without turning the pass into a tuning stage.
