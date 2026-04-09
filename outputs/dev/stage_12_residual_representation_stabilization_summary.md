# Stage 12 Summary - Hybrid Residual Representation-Scale Stabilization

## Stage Goal
Implement the narrowest hybrid-only fix justified by Stage 11:

- control representation amplitude was dominating `delta_raw`,
- gradients were then dying at the bounded residual transform,
- Task 1 bootstrap anchor remained contextual but was not the main Task 2+ failure.

Stage 12 therefore changes only the post-bootstrap hybrid residual path.

## Why This Fix Was Chosen
Stage 11 showed two linked problems:

1. `delta_from_representation_mean` dominated the residual head while bias contribution stayed tiny.
2. `delta_logit` still had gradient signal, but `delta_raw` lost it at the bounded transform.

To target both without reopening planner policy, thresholds, CHU, router, classifier, or Stage 5 inference semantics, Stage 12:

- RMS-normalizes the control representation immediately before the residual gate head,
- replaces the residual `tanh` bound with `softsign`,
- keeps `planner_control_delta_logit_scale=2.0` unchanged to isolate the fix.

## Exact Hybrid Gate Formula
- `anchor_beta = applied_plan["shared_gate"]`
- `anchor_logit = logit(clamp(anchor_beta, 1e-4, 1 - 1e-4))`
- `h_control_norm = h_control / clamp(sqrt(mean(h_control^2)), min=1e-6)`
- `delta_raw = control_head(h_control_norm)`
- `delta_logit = planner_control_delta_logit_scale * softsign(delta_raw)`
- `effective_logit = anchor_logit + delta_logit`
- `effective_beta = sigmoid(effective_logit)`

The control output head remains zero-initialized, so hybrid startup still begins at the policy anchor.

## Files Changed
- `src/models/planner.py`
- `src/engine/trainer.py`
- `tests/test_paper_alignment_units.py`
- `README.md`
- `docs/paper_alignment.md`
- `outputs/dev/stage_12_residual_representation_stabilization_summary.md`

## What Remained Unchanged
- pre-task structural planning and `decide_action()`
- planner thresholds
- `materialize_action()` / `apply_structure_change()`
- CHU behavior
- routing semantics
- classifier behavior
- Stage 5 inference-profile semantics
- soft-rank
- `planner_control_delta_logit_scale=2.0`

## Validation Commands
- `python -m compileall src tests`
- `python -m unittest tests.test_config_summary tests.test_paper_alignment_units tests.test_synthetic_continual_smoke`

## Residual Failure Recheck
- Task 1 remains bootstrap context only; Stage 12 does not change bootstrap structural policy.
- Task 2 is the authoritative onset window: the residual path should no longer become universally hard-pinned immediately if Stage 12 is helping.
- Task 3 checks persistence: at least one hybrid-active layer should remain movable with surviving `delta_raw` gradient if the fix is working.

## Short-Window Acceptance Gate
Because Stage 12 combines representation-scale stabilization and bounded-transform replacement in one step, **short-window evidence is the authoritative decision gate**.

If early Task 2 still shows all hybrid-active layers hard-pinned, or `bound_derivative_mean` and `delta_raw_grad_nonzero_batches` remain effectively zero across all active layers, Stage 12 must be treated as failed without proceeding to long-horizon reruns.

## Full Rerun Command
```bash
python -m src.engine.train --config configs/cifar100_hybrid.yaml --seed 1 --benchmark cifar100 --output-root outputs
```

## Next Recommendation If Hard-Pinning Persists
Do not tune thresholds or enable soft-rank next.

If Stage 12 fails, the next stage should stay residual-only and decide whether the remaining issue is:

- still representation-scale dominated despite RMS normalization,
- now mostly a bounded-transform choice issue,
- or a narrower control-head/output geometry problem.
