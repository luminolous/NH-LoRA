# Stage 5 Follow-Up Summary - Inference Profile Semantics Alignment

## Stage Hypothesis
Stage 5 diagnostics proved a local mismatch: for `reuse_shared` and `freeze_old_strong_retention`, the applied training route was shared-only with empty candidates, but `build_inference_profile()` still reactivated retained slots during evaluation. This follow-up patch aligns inference-profile semantics with the declared repo policy.

## Files Changed
- `src/models/nh_lora.py`
- `tests/test_paper_alignment_units.py`
- `outputs/dev/stage_5_followup_summary.md`

## Diff Summary
- Aligned `build_inference_profile()` so `reuse_shared` and `freeze_old_strong_retention` now emit shared-only inference with empty `active_slot_candidates`.
- Preserved retained/live slot metadata, `rank_cfg`, `shared_gate`, `consolidate_flag`, and `strong_retention`.
- Kept non-shared-only actions unchanged.
- Updated unit tests to validate the declared shared-only inference policy and added a Stage 5 regression that checks the applied-vs-profile mismatch flag clears for shared-only actions.

## Validation Commands
- `python -m compileall src tests`
- `python -m unittest tests.test_config_summary tests.test_paper_alignment_units tests.test_synthetic_continual_smoke`
- `git diff --check`

## Validation Result
- `python -m compileall src tests` passed.
- `python -m unittest tests.test_config_summary tests.test_paper_alignment_units tests.test_synthetic_continual_smoke` passed (`48 tests`).
- `git diff --check` passed with line-ending warnings only (`LF` -> `CRLF` on next Git touch).

## Rerun Log Path
- Pre-patch reference rerun: `outputs/logs/cifar100_seed1_20260408_170302.log`
- Post-patch rerun: pending

## Before/After Invariant Summary
The table below records the proven pre-patch mismatch from the baseline Stage 5 rerun and the expected post-patch invariant for the same affected layers. Post-patch values are derived from the new shared-only inference-profile contract and covered by unit tests; full CIFAR-100 confirmation still requires a rerun.

| Task | Layer | Raw action | Applied candidates before patch | Profile candidates before patch | Profile candidates after patch | Mismatch before | Mismatch after |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 3 | 6 | `freeze_old_strong_retention` | `[]` | `[0, 1]` | `[]` | `True` | `False` |
| 3 | 7 | `freeze_old_strong_retention` | `[]` | `[0]` | `[]` | `True` | `False` |
| 3 | 8 | `freeze_old_strong_retention` | `[]` | `[0]` | `[]` | `True` | `False` |
| 3 | 9 | `reuse_shared` | `[]` | `[0]` | `[]` | `True` | `False` |
| 3 | 10 | `freeze_old_strong_retention` | `[]` | `[0]` | `[]` | `True` | `False` |
| 3 | 11 | `freeze_old_strong_retention` | `[]` | `[0]` | `[]` | `True` | `False` |
| 4 | 6 | `freeze_old_strong_retention` | `[]` | `[0, 1]` | `[]` | `True` | `False` |
| 4 | 7 | `freeze_old_strong_retention` | `[]` | `[0]` | `[]` | `True` | `False` |
| 4 | 8 | `freeze_old_strong_retention` | `[]` | `[0]` | `[]` | `True` | `False` |
| 4 | 9 | `reuse_shared` | `[]` | `[0]` | `[]` | `True` | `False` |
| 4 | 10 | `freeze_old_strong_retention` | `[]` | `[0]` | `[]` | `True` | `False` |
| 4 | 11 | `freeze_old_strong_retention` | `[]` | `[0]` | `[]` | `True` | `False` |

## Main Findings
- The mismatch was caused by `build_inference_profile()` ignoring the shared-only meaning of `last_structural_action` for `reuse_shared` and `freeze_old_strong_retention`.
- Retained slots are still preserved in layer state and continue to appear in slot lifecycle diagnostics; only their activation in inference routing changed for the affected actions.
- For actions other than `reuse_shared` and `freeze_old_strong_retention`, inference-profile candidate behavior is unchanged.

## Next-Stage Decision
- Rerun the same baseline CIFAR-100 configuration with Stage 1 and Stage 5 diagnostics still enabled.
- If `mismatch_applied_empty_profile_nonempty=True` disappears for the affected Task 3/4 layers and forgetting remains high, close Stage 5 as an implementation-level mismatch fix and move to Stage 6 synthesis.
- If the mismatch persists, re-audit `_evaluate_up_to` and profile plumbing before expanding scope.
