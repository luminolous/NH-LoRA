# Stage 9 Summary - Early-Onset Planner Control Saturation Audit

## Stage Goal
Add diagnostics-only instrumentation to explain the new residual failure mode after Stage 8 Checkpoint A:

- `planner_control` is no longer training-path disconnected,
- but the learned shared gate now shows **early-onset saturation** that persists into the late task horizon.

Stage 9 does not tune thresholds, clip beta, add regularizers, redesign routing, change CHU, change classifier behavior, change inference-profile semantics, or implement soft-rank.

## Why Stage 8 Checkpoint A Is Considered Successful
- Hybrid policy/control semantics are now explicit and truthful.
- `planner_policy` is no longer left in a fake task-loss-trainable state.
- `planner_control` stays in the optimizer and receives real task-loss gradients.
- The Stage 8 hybrid CIFAR-100 rerun improved Task 10 from the pre-Stage-8 baseline:
  - prior baseline: `avg_acc=0.2885`, `forgetting=0.7518`
  - Stage 8 hybrid rerun: `avg_acc=0.4555`, `forgetting=0.5258`

## Earliest Observed Onset
Primary evidence source: `outputs/logs/cifar100_seed1_20260409_022247.log` from the user-provided Stage 8 hybrid Task-10 rerun.

Earliest onset already appears in Task 1:
- Task 1 Epoch 1: control gradients are present and nonzero, while beta is not yet fully saturated.
  - `PlannerControlTrainPath`: `grad_nonzero_batches=138/139`, `pre_step_grad_l2_mean=1.1471e-03`
  - `PlannerControlValues`: beta means around `0.9583-0.9626`, with mins around `0.4665-0.5091`
- Task 1 Epoch 2: beta is already almost fully saturated.
  - beta means jump to roughly `0.9992-0.9999`
  - control pre-step grad mean shrinks to `1.3982e-04`
- Task 2 Epoch 1: the collapse is fully persistent.
  - `PlannerControlTrainPath`: `grad_nonzero_batches=0/139`, `pre_step_grad_l2_mean=0.0000e+00`
  - `PlannerControlValues`: beta is `1.0000` on all audited layers

This is why Stage 9 uses the framing **early-onset shared-gate saturation with late-horizon persistence**, not "late-only collapse."

## What Stage 9 Adds
- Pre-sigmoid beta-logit summaries per layer/epoch:
  - mean / min / max
  - `p50`, `p90`, `p99`
  - fractions above `0`, `2`, `logit(0.99)`, and `logit(0.999)`
  - epoch drift and drift from task-start control state
- Post-sigmoid beta summaries per layer/epoch:
  - mean / min / max
  - `p50`, `p90`, `p99`
  - fractions `beta > 0.99`, `beta > 0.999`, `beta < 0.01`
  - epoch drift and drift from task-start control state
- Gradient-to-gate summaries per layer/epoch:
  - beta-output grad magnitude
  - beta-logit grad magnitude
  - present/nonzero batch counts
  - effectively-zero fractions
- Shared-vs-slot contribution summaries per layer/epoch:
  - shared norm before beta
  - shared norm after beta
  - slot norm
  - shared-to-slot ratio
  - structural slot availability fraction
  - nontrivial slot-contribution fraction
  - `beta > 0.99` / `beta > 0.999` while slots are structurally available
- Control-input summaries:
  - task-context norm/variance
  - per-layer control-input norm/variance
  - per-layer control-representation norm/variance
  - control-input pairwise cosine across layers
  - policy-input pairwise cosine from the pre-task planner side when available
- Branch-specific planner grad cleanup:
  - `planner_policy` and `planner_control` now log separately in grad-norm debug
  - aggregate planner grad remains available but is no longer the only signal

## Files Changed
- `src/models/planner.py`
- `src/models/lora.py`
- `src/engine/trainer.py`
- `tests/test_paper_alignment_units.py`
- `README.md`
- `docs/paper_alignment.md`
- `outputs/dev/stage_9_planner_control_saturation_summary.md`

## Validation Commands
- `python -m compileall src tests`
- `python -m unittest tests.test_config_summary tests.test_paper_alignment_units tests.test_synthetic_continual_smoke`
- `git diff --check`

## Validation Results
- `python -m compileall src tests` passed.
- `python -m unittest tests.test_config_summary tests.test_paper_alignment_units tests.test_synthetic_continual_smoke` passed (`62 tests`).
- `git diff --check` passed with line-ending warnings only.

## Why Layer 6 Still Matters
Stage 8 evidence already showed that some later tasks can still keep structural slot activity concentrated in Layer 6 while the learned control path is saturated. Stage 9 therefore instruments:
- whether Layer 6 still has structurally available slots,
- whether those slots contribute nontrivially,
- and whether `beta=1` merely co-occurs with shared dominance or actively suppresses useful slot modulation.

The audit is not complete until that Layer 6 split is explained explicitly.

## Why Soft-Rank Remains Deferred
Soft-rank is still deferred because the current bottleneck is not "missing differentiable rank control" yet. The immediate unknown is narrower:
- why the restored learned shared-gate path saturates so early,
- why its gradients disappear so quickly,
- and whether that is driven primarily by logit saturation, shared-path dominance, or control-input collapse.

Adding soft-rank before that audit would widen scope and blur causal attribution.

## Narrowest Next Patch Recommendation
Run the Stage 8 hybrid CIFAR-100 baseline again with the new Stage 9 diagnostics enabled, then classify the dominant mechanism into one of:
- sigmoid / parameterization saturation,
- shared-path incentive dominance,
- control-input collapse,
- or a mixed case with one primary driver.

Only after that classification should a narrow mitigation patch be planned.

## What Stage 9 Explicitly Did Not Change
- planner thresholds
- beta clipping or beta temperature as a fix
- regularizers or load-balancing penalties
- router behavior
- CHU behavior
- classifier behavior
- inference-profile semantics
- soft-rank implementation
- CIFAR-100 benchmark protocol
