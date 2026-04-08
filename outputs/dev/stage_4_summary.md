# Stage 4 Summary - Adapter Activation Audit And Patch

## Stage Hypothesis
Active NH-LoRA adapters were not affecting the forward output because a local forward-semantics bug dropped the attended attention path around `out_proj`. A secondary risk was that Stage 1 delta diagnostics might have been reading the wrong path; the wiring tests below distinguish these cases.

## Git Checkpoint
- Starting branch: `codex-stage-1-instrumentation`
- Starting commit: `d3bbd967923e096d1a0a5f727e7d26734fd3ca79`
- Stage branch: `codex-stage-4-adapter-activation`

## Files Changed
- `src/backbones/vision_transformer.py`
- `tests/test_paper_alignment_units.py`
- `outputs/dev/stage_4_summary.md`

## Diff Summary
- Preserved the attended tensor as the no-op/base value when applying the `out_proj` adapter in `_forward_block_with_adapter`.
- Added Stage 4 regression tests for zero-LoRA block parity with `out_proj` selected and not selected.
- Added wiring tests proving manually nonzero active adapters change block output and are visible to the Stage 1 delta accumulator.
- Added an integration gradient test proving CE loss reaches active `slot_b` and active `shared_b`.

## Wiring Correctness Findings
- Before the patch, the Stage 4 tests failed:
  - zero-LoRA parity failed for `["q_proj", "v_proj"]`
  - zero-LoRA parity failed for `["q_proj", "v_proj", "out_proj"]`
  - CE gradient did not reach active `slot_b`
- After the patch, all Stage 4 wiring tests passed.
- The active-adapter test sets nonzero shared and slot deltas manually and confirms the block output changes and the delta logger records nonzero shared/slot values.
- The integration CE test confirms nonzero grad on active `slot_b` and active `shared_b` for the current deterministic test configuration.

## Baseline Parity Findings
- Zero-LoRA/no-op initialization now preserves baseline block output.
- Parity is covered for both `out_proj` not selected and `out_proj` selected.
- This keeps baseline semantics unchanged when LoRA deltas are zero.

## Validation Commands
- `python -m unittest tests.test_paper_alignment_units.PaperAlignmentUnitTests.test_stage4_zero_lora_block_parity_preserves_out_proj_semantics tests.test_paper_alignment_units.PaperAlignmentUnitTests.test_stage4_active_adapter_changes_output_and_delta_logger tests.test_paper_alignment_units.PaperAlignmentUnitTests.test_stage4_ce_grad_flows_to_active_slot_and_shared_b`
- `python -m compileall src tests`
- `python -m unittest tests.test_paper_alignment_units tests.test_synthetic_continual_smoke`

## Validation Result
- Stage 4 targeted tests: passed after patch.
- Compileall: passed.
- Unit/smoke validation: passed, 36 tests.

## CIFAR-100 Rerun Log
- Not generated in this local workspace during this patch.
- Repo docs state real benchmark training is run on the SSH training machine.
- Required next run path format: `outputs/logs/<stage4-cifar100-debug-run>.log`.
- Required flags: keep Stage 1 diagnostics enabled and rerun CIFAR-100 at least through task 3.

## Main Diagnostic Expectations For Rerun
- `AdapterDelta` should no longer remain zero for every active shared/slot path.
- `FinalFeatureDiff` should no longer remain exactly zero for every debug epoch if the adapter path changes features.
- `GradNorm` should continue to report nonzero slot grads and should report nonzero shared grads when shared path is active.
- If slot becomes active but shared remains zero, keep Stage 4 open as a sub-issue unless the active config explains shared inactivity.

## Next-Stage Decision
- Do not proceed to Stage 3 or Stage 5 from local tests alone.
- Run the short CIFAR-100 Stage 1 debug rerun on the training environment.
- If adapter deltas become active but forgetting remains high, proceed to Stage 2 head-dominance ablation.
- If wiring tests pass but CIFAR `AdapterDelta` remains zero, audit the Stage 1 diagnostic hook/logger path before changing algorithms.
- If routing/inference mismatch becomes the dominant post-patch signal, document it for Stage 5 but do not patch routing in Stage 4.

## Risks And Open Questions
- CIFAR-100 post-patch evidence is still pending.
- Local tests confirm shared and slot paths can affect the forward output and receive CE gradients, but benchmark behavior still needs verification.
- Stage 4 should not be considered fully closed until the post-patch CIFAR debug log is reviewed.
