# Stage 2 Summary - Head-Dominance Ablation

## Stage Hypothesis
Current-task fitting may be dominated by classifier-head updates after the Stage 4 adapter wiring fix. Stage 2 adds default-off classifier ablation knobs to test that hypothesis without changing the NH-LoRA objective or planner.

## Stage 4 Carryover
- Stage 4 main bug is considered fixed: the adapter path is no longer inert after preserving the attended `out_proj` base output.
- Residual Stage 4 note remains open for monitoring: in the Stage 4 CIFAR-100 rerun, Task 3 showed slot grad/delta and routing candidates collapse to zero while shared path stayed active.
- Stage 2 must monitor `AdapterDelta`, `FinalFeatureDiff`, adapter grad norms, and routing candidate counts during reruns.

## Files Changed
- `configs/base.yaml`
- `src/engine/trainer.py`
- `tests/test_config_summary.py`
- `tests/test_paper_alignment_units.py`
- `README.md`
- `docs/paper_alignment.md`
- `outputs/dev/stage_2_summary.md`

## Diff Summary
- Added `training.classifier_lr_scale`, `training.freeze_new_classifier_epochs`, and `training.freeze_all_classifier_epochs` with baseline-safe defaults.
- Added optional classifier LR scaling through a separate classifier optimizer group only when `classifier_lr_scale != 1.0`.
- Added classifier gradient masking for early all-classifier freeze and new-classifier-row freeze, independent of optimizer grouping.
- Preserved existing old-classifier gradient masking.
- Added seed config logging for the Stage 2 knobs.
- Added tests for config defaults, LR grouping, and classifier gradient masking behavior.
- Documented that the knobs are diagnostic/default-off and do not alter the paper objective at defaults.

## Validation Commands
- `python -m unittest tests.test_config_summary tests.test_paper_alignment_units.PaperAlignmentUnitTests.test_classifier_lr_scale_default_keeps_classifier_in_base_group tests.test_paper_alignment_units.PaperAlignmentUnitTests.test_classifier_lr_scale_nondefault_uses_separate_param_group tests.test_paper_alignment_units.PaperAlignmentUnitTests.test_freeze_all_classifier_epochs_masks_all_rows tests.test_paper_alignment_units.PaperAlignmentUnitTests.test_freeze_new_classifier_epochs_masks_only_new_rows tests.test_paper_alignment_units.PaperAlignmentUnitTests.test_freeze_new_classifier_epochs_expires_after_configured_epochs tests.test_paper_alignment_units.PaperAlignmentUnitTests.test_old_classifier_row_gradient_masking tests.test_paper_alignment_units.PaperAlignmentUnitTests.test_old_classifier_gradient_masking_can_be_disabled tests.test_paper_alignment_units.PaperAlignmentUnitTests.test_seed_config_logging_tolerates_missing_optional_fields`
- `python -m compileall src tests`
- `python -m unittest tests.test_config_summary tests.test_paper_alignment_units tests.test_synthetic_continual_smoke`

## Validation Result
- Targeted Stage 2 tests passed.
- Compileall passed.
- Unit/smoke validation passed: 42 tests.

## CIFAR-100 Rerun Log
- Not generated in this local workspace during this patch.
- Required apples-to-apples run target: compare against the Stage 4 patched rerun using the same seed, `epochs_per_task`, `batch_size`, Stage 1 debug flags, and runtime settings where possible.
- Required log path format: `outputs/logs/<stage2-cifar100-ablation-run>.log`.

## Acceptance Criteria For Rerun
- Stage 2 supports head dominance only if old-task retention improves versus the Stage 4 patched baseline, current-task accuracy does not drop sharply, and adapter path remains active.
- If retention appears better only because current-task learning collapses, Stage 2 does not support the hypothesis.
- If `AdapterDelta`, `FinalFeatureDiff`, or adapter grad norms collapse under classifier ablation, do not interpret retention naively.

## Next-Stage Decision
- Await the Stage 2 CIFAR-100 rerun.
- If retention improves without sharp current-task degradation and adapters stay active, keep the Stage 2 patch candidate and consider Stage 3 only in a later planning step.
- If Stage 2 does not help and Task 3 still shows `candidate_count=0` plus slot grad/delta collapse, proceed next to Stage 5 routing/inference audit.
- If the adapter path becomes inert again, reopen Stage 4 residual investigation.
