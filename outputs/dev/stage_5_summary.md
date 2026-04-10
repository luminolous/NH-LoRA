# Stage 5 Summary - Routing And Inference Profile Audit

## Stage Hypothesis
Stage 2 LR-scale ablations did not support classifier head dominance as a sufficient fix. The remaining consistent signal is slot/router candidate inactivity around Task 3, so Stage 5 adds routing and inference-profile diagnostics before any behavior patch.

## Files Changed
- `src/engine/trainer.py`
- `tests/test_paper_alignment_units.py`
- `README.md`
- `docs/paper_alignment.md`
- `outputs/dev/stage_5_summary.md`

## Diff Summary
- Added default-off routing audit logs under existing `training.routing_debug_logging`.
- Added raw planner, materialized plan, applied plan, slot lifecycle, inference profile, and same-input train-vs-eval route comparison diagnostics.
- Added slot id visibility for live, retained, candidate, selected, frozen, and pruned slots.
- Preserved RNG state around train-vs-eval route probes so diagnostics do not perturb later stochastic operations.
- Added tests for slot lifecycle visibility, retained-profile semantics, mismatch flags, same-input train-vs-eval probing, and no-mutation diagnostics.

## Validation Commands
- `python -m unittest tests.test_paper_alignment_units.PaperAlignmentUnitTests.test_stage5_slot_lifecycle_summary_exposes_slot_ids_without_mutating_inputs tests.test_paper_alignment_units.PaperAlignmentUnitTests.test_stage5_profile_summary_uses_retained_live_slots_for_shared_only tests.test_paper_alignment_units.PaperAlignmentUnitTests.test_stage5_route_comparison_flags_empty_applied_nonempty_profile tests.test_paper_alignment_units.PaperAlignmentUnitTests.test_stage5_plan_debug_does_not_mutate_plan_or_profile_state tests.test_paper_alignment_units.PaperAlignmentUnitTests.test_stage5_train_eval_comparison_uses_same_input_and_restores_rng`
- `python -m compileall src tests`
- `python -m unittest tests.test_config_summary tests.test_paper_alignment_units tests.test_synthetic_continual_smoke`

## Validation Result
- Targeted Stage 5 diagnostics tests passed locally.
- Compileall passed.
- Unit/smoke validation passed: 47 tests.

## Rerun Log
- Not generated locally during this patch.
- Required run target: `outputs/logs/<stage5-cifar100-routing-audit>.log`.
- Use baseline classifier settings: `classifier_lr_scale=1.0`, `freeze_new_classifier_epochs=0`, and `freeze_all_classifier_epochs=0`.

## Main Diagnostic Findings
- Stage 5 implementation is diagnostics-only. No planner, CHU, objective, rehearsal, dataset, or inference policy behavior was changed.
- The audit logs are designed to classify whether candidate collapse comes from raw planner shared-only decisions, materialization, apply-structure state, CHU/retention state, inference profile construction, or logging hook mismatch.

## Next-Stage Decision
- Await a short CIFAR-100 rerun through at least Task 3, preferably Task 4, with `routing_debug_logging=true` and Stage 1 diagnostics enabled.
- If raw planner chooses shared-only for all Task 3 layers and no local mismatch appears, classify it as policy weakness / Stage 6 synthesis input, not an implementation bug.
- If retained slots exist but inference profile or applied route candidates disagree, patch that proven local mismatch in a follow-up Stage 5 patch.

## Risks Or Open Questions
- Same-input train-vs-eval probing is extra compute under debug logging, but the flag is default-off.
- The rerun must inspect slot lifecycle IDs, not only candidate counts, to distinguish removed slots from retained slots that never enter the candidate set.
