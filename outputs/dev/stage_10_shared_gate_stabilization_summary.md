# Stage 10 Summary - Hybrid Shared-Gate Stabilization

## Why Stage 9 Justified A Fix Stage
Stage 9 closed the audit loop around Stage 8 hybrid control:

- Stage 8 Checkpoint A successfully restored a real task-loss training path for `planner_control`.
- The new failure was no longer a training-path disconnect.
- The dominant remaining mechanism was early-onset shared-gate logit runaway followed by sigmoid saturation and loss of gradient to control parameters.

That made Stage 10 appropriate as a narrow fix stage focused only on the hybrid learned shared-gate parameterization.

## Why Anchored Residual Gating Was Chosen
The Stage 9 evidence showed that the absolute hybrid gate could relearn an unconstrained always-shared solution almost immediately. Stage 10 therefore keeps the paper-facing structural policy untouched and constrains learned control to modulate the policy prior rather than replace it.

The fix is hybrid-only and preserves:

- pre-task structural planning,
- discrete planner action quadrants,
- CHU semantics,
- router semantics,
- classifier semantics,
- Stage 5 inference-profile semantics,
- deferred soft-rank.

## Exact Formula
Stage 10 replaces the hybrid absolute gate with an anchored residual gate:

- `anchor_beta = applied_plan["shared_gate"]`
- `anchor_logit = logit(clamp(anchor_beta, 1e-4, 1 - 1e-4))`
- `delta_raw = planner_control(block_id, task_context, history_summary)`
- `delta_logit = planner_control_delta_logit_scale * tanh(delta_raw)`
- `effective_logit = anchor_logit + delta_logit`
- `effective_beta = sigmoid(effective_logit)`

`effective_beta` is the only gate consumed by the NH-LoRA forward path in hybrid mode.

The control output head is zero-initialized so the initial hybrid forward pass satisfies:

- `delta_raw = 0`
- `delta_logit = 0`
- `effective_beta = anchor_beta`

## Files Changed
- `configs/base.yaml`
- `configs/cifar100_hybrid.yaml`
- `src/models/planner.py`
- `src/engine/trainer.py`
- `tests/test_config_summary.py`
- `tests/test_paper_alignment_units.py`
- `tests/test_synthetic_continual_smoke.py`
- `README.md`
- `docs/paper_alignment.md`
- `outputs/dev/stage_10_shared_gate_stabilization_summary.md`

## What Remained Unchanged
- planner thresholds and `decide_action()`
- `materialize_action()`
- `apply_structure_change()`
- CHU logic
- router candidate selection and sparse routing semantics
- classifier expansion and classifier losses
- Stage 5 inference-profile behavior
- soft-rank implementation
- CIFAR-100 benchmark protocol

## Validation Commands
- `python -m compileall src tests`
- `python -m unittest tests.test_config_summary tests.test_paper_alignment_units tests.test_synthetic_continual_smoke`

## Validation Results
- `python -m compileall src tests` passed.
- `python -m unittest tests.test_config_summary tests.test_paper_alignment_units tests.test_synthetic_continual_smoke` passed (`63 tests`).

## Short-Window Rerun Evidence
Local implementation validation only was completed in this pass. The required short-window CIFAR-100 rerun for Task 1-3 has **not** been executed from this workspace in this stage summary.

Because of that:

- Stage 10 code is implemented and test-validated.
- Stage 10 early-window behavioral validation is still pending.
- No claim is made yet that Stage 10 solved early-onset saturation on benchmark data.

## Full Rerun Command
Use the existing hybrid baseline and keep Stage 9 diagnostics on:

```bash
python -m src.engine.train --config configs/cifar100_hybrid.yaml --seed 1 --benchmark cifar100 --output-root outputs
```

The intended first pass keeps:

- `training.planner_mode=hybrid`
- `training.planner_audit_logging=true`
- `training.planner_control_delta_logit_scale=2.0`

Short-window review must inspect:

- Task 1 Epoch 1
- Task 1 Epoch 2
- Task 2 Epoch 1
- Task 2 early epochs
- Task 3 early epochs

Do not trust a full Task-10 rerun unless that short window passes first.

## Anchor-vs-Effective Gate Analysis
Stage 10 adds explicit audit signals so reruns can separate anchor prior problems from residual-control problems:

- `PlannerControlAnchor`
  - `anchor_beta_mean/min/max`
  - `frac_anchor_gt_099`
  - `frac_anchor_gt_0999`
  - `anchor_logit_mean/p90/p99`
- `PlannerControlDelta`
  - `delta_raw_mean/min/max`
  - `delta_logit_mean/min/max`
  - `beta_anchor_gap_mean_abs`
  - `logit_anchor_gap_mean_abs`
- `PlannerControlValues`
  - effective beta summaries
- `PlannerControlLogits`
  - effective logit summaries

Interpretation rule for reruns:

- if `effective_beta` saturates while `anchor_beta` stays moderate, the residual control dynamics remain the primary problem;
- if `anchor_beta` itself is already near `1.0`, then the anchor prior is part of the remaining problem;
- if both happen, the summary should state both explicitly instead of collapsing them into one story.

Important implementation note:

- bootstrap Task 1 still comes from the existing structural bootstrap plan;
- that plan can legitimately provide a very high or saturated `anchor_beta`;
- Stage 10 does **not** change bootstrap structural policy;
- Stage 10 only makes the anchor-vs-effective distinction explicit and auditable.

## Residual Scale Validation
`training.planner_control_delta_logit_scale=2.0` is treated as a **default initial hypothesis only**.

This pass does **not** treat `2.0` as a final tuned or trusted value. The value survives only if the required short-window rerun shows:

- no near-universal effective-gate saturation by Task 1 Epoch 2,
- nonzero `PlannerControlTrainPath` gradient activity in Task 2,
- at least one hybrid-active layer with nonzero effective-logit gradient in Task 2,
- and a visible separation between anchor prior and learned residual effect in at least one active layer.

If the short window fails, the correct Stage 10 report is:

- the anchored residual design is implemented,
- but the default residual scale is not yet validated.

This stage does **not** authorize broad scale tuning.

## Whether Stage 10 Solved Early-Onset Saturation
Not yet determined from benchmark evidence in this pass.

What is established:

- the hybrid control path now uses a policy-anchored bounded residual gate,
- the control head starts at the anchor instead of at an unconstrained absolute gate,
- anchor-vs-effective behavior is now first-class audit output,
- and the default scale is explicitly treated as a short-window validation hypothesis.

What remains to be checked by rerun:

- whether early-onset saturation is reduced,
- whether Task 2 still loses control gradients,
- whether Layer 6 keeps meaningful control modulation when structurally active,
- whether the anchor prior itself is still too saturated in bootstrap or later tasks.

## Next Recommendation If Saturation Persists
If the short-window rerun still fails, do **not** jump to threshold tuning, regularizers, router changes, or soft-rank.

The next narrow planning step should classify the remaining issue as one of:

1. residual dynamics still too permissive even with anchored gating,
2. anchor prior itself already too saturated,
3. mixed case where both anchor and residual contribute.

That next step should stay focused on shared-gate stabilization rather than reopening Stage 3, CHU, or routing policy.
