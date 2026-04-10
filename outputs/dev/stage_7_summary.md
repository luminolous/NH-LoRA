# Stage 7 Summary - Planner Shared-Only Bias Audit

## Stage Hypothesis
The remaining long-horizon forgetting after the Stage 4 and Stage 5 fixes is most likely caused by a planner-driven shared-only regime. Stage 7 adds diagnostics to classify whether that regime comes from a planner training-path disconnect, an optimizer-connected but effectively static planner, near-threshold conservative behavior, or upstream task-state/history signal collapse.

## Files Changed
- `configs/base.yaml`
- `src/engine/trainer.py`
- `tests/test_config_summary.py`
- `tests/test_paper_alignment_units.py`
- `README.md`
- `docs/paper_alignment.md`
- `outputs/dev/stage_7_summary.md`

## Validation Commands
- `python -m compileall src tests`
- `python -m unittest tests.test_config_summary tests.test_paper_alignment_units`
- `python -m unittest tests.test_config_summary tests.test_paper_alignment_units tests.test_synthetic_continual_smoke`
- `git diff --check`

## Validation Result
- `python -m compileall src tests` passed.
- `python -m unittest tests.test_config_summary tests.test_paper_alignment_units` passed (`52 tests`).
- `python -m unittest tests.test_config_summary tests.test_paper_alignment_units tests.test_synthetic_continual_smoke` passed (`53 tests`).
- `git diff --check` passed with line-ending warnings only (`LF` -> `CRLF` on next Git touch).

## Rerun Log Path
- Planned Stage 7 audit rerun artifact: `outputs/logs/stage7_planner_audit_task10.log`
- Current status: rerun pending on the main training machine.

## Main Diagnostic Coverage Added
- Per-task, per-layer planner logs for `novelty`, `conflict`, `shared_gate`, `consolidate`, `rank_budget`, `tau_novelty`, `tau_conflict`, and their margins.
- Planner action trajectories across tasks, plus a focused comparison log for the special layer that occasionally crosses into `open_new_slot` behavior.
- Planner training-path audit:
  - planner optimizer membership
  - `requires_grad=True` coverage
  - pre-step planner grad statistics before clipping
  - planner optimizer step counts
- Planner parameter drift summaries relative to initialization and the post-Task-1 checkpoint.
- Task-state / planner-input summaries covering embedding norm, similarity, entropy, gradient-sketch norm, planner-input norm, planner-representation norm, and history-attention concentration.

## Dominant Bucket Classification
- Pending rerun with `training.planner_audit_logging=true`.
- Current code inspection makes a training-path disconnect a strong hypothesis because planner decisions are materialized before task training and later reused as discrete plans, but Stage 7 does not claim that classification until rerun evidence is collected.

## Layer 6 Special-Case Note
- Stage 7 logs now include a dedicated comparison summary so reruns can explain why Layer 6 sometimes reaches `open_new_slot` while neighboring layers remain in shared-only actions.
- Final explanation pending rerun.

## Next-Stage Recommendation
- Run the Stage 7 planner audit on the same best baseline used for the successful Stage 5 updated rerun.
- After the rerun, classify the dominant bucket:
  - training-path disconnect
  - optimizer-connected but effectively static planner
  - near-threshold conservative planner behavior
  - upstream task-state/history collapse
- Do not patch planner thresholds or planner behavior before that rerun is reviewed.
