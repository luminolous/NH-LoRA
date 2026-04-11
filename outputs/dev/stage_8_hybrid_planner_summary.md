# Stage 8 Summary - Hybrid Planner Refactor And Training-Path Fix

## Stage Goal
Restore a truthful task-loss training path for the planner without changing NH-LoRA's pre-task structural planning, CHU, routing policy, classifier, or loss stack.

## Why Hybrid Was Chosen
Stage 7 showed a planner training-path disconnect: planner parameters were present in the optimizer and had `requires_grad=True`, but gradients never appeared and planner drift stayed zero. The hybrid split keeps structural policy pre-task and non-differentiable, while making forward-time control explicitly trainable.

## Files Changed
- `configs/base.yaml`
- `configs/cifar100_hybrid.yaml`
- `src/models/planner.py`
- `src/models/lora.py`
- `src/engine/trainer.py`
- `tests/test_config_summary.py`
- `tests/test_paper_alignment_units.py`
- `tests/test_synthetic_continual_smoke.py`
- `README.md`
- `docs/paper_alignment.md`
- `outputs/dev/stage_8_hybrid_planner_summary.md`

## Semantic Split
- `planner_policy`
  - implemented as `HorizonPlanner.policy_branch`
  - still owns novelty, conflict, consolidate, rank score, and discrete structural action inputs
  - remains pre-task and not task-loss-trained in Stage 8 hybrid mode
- `planner_control`
  - implemented as `HorizonPlanner.control_branch`
  - currently owns learned `shared_gate` / `beta`
  - is recomputed during training forward passes and remains inside the computation graph

## Training Path Before vs After
- Before:
  - raw planner outputs were computed before task training
  - structural plans were materialized once
  - shared-gate values were effectively cached scalars
  - planner parameters stayed in the optimizer even though task loss never updated them
- After:
  - hybrid mode freezes/excludes the policy branch from task-loss training
  - control-branch parameters stay in the optimizer
  - control outputs are recomputed per batch from fixed task context plus trainable control parameters
  - `shared_gate` is no longer forced through `float(...)` before NH-LoRA shared deltas are computed

## Gradient Flow Restoration
- `ProjectionBank.shared_delta()` now accepts tensor-valued `shared_gate` and preserves autograd.
- `NHLoRALayer._shared_delta()` passes planner `shared_gate` through without detaching to `float`.
- `NHLoRATrainer._plans_for_training_forward()` recomputes `planner.forward_control(...)` per batch in hybrid mode and injects tensor `shared_gate` values into the runtime planner config.
- Hybrid logs now report:
  - `HybridPlannerConfig`
  - `PlannerPolicyTrainPath`
  - `PlannerControlTrainPath`
  - `PlannerControlValues`
  - `PlannerControlParamDrift`

## What Remains Non-Differentiable / Heuristic
- discrete structural planning quadrants remain pre-task
- `materialize_action()` remains hard and explicit
- slot open / expand / freeze decisions remain structural
- CHU remains heuristic
- inference profile semantics remain hard-action-based

## Soft-Rank Status
Soft-rank training was **deferred** in this pass. Stage 8 implements Checkpoint A only. The code and summary remain explicit that learned shared-gate control is restored, while differentiable rank control is still pending and intentionally not claimed.

## Validation Commands
- `python -m compileall src tests`
- `python -m unittest tests.test_config_summary tests.test_paper_alignment_units`
- `python -m unittest tests.test_config_summary tests.test_paper_alignment_units tests.test_synthetic_continual_smoke`
- `git diff --check`

## Validation Results
- `python -m compileall src tests` passed.
- `python -m unittest tests.test_config_summary tests.test_paper_alignment_units` passed (`57 tests`).
- `python -m unittest tests.test_config_summary tests.test_paper_alignment_units tests.test_synthetic_continual_smoke` passed (`59 tests`).
- `git diff --check` passed with line-ending warnings only.

## Rerun Command
```bash
python -m src.engine.train --config configs/cifar100_hybrid.yaml --seed 1 --benchmark cifar100 --output-root outputs
```

For apples-to-apples comparison with the latest planner audit baseline, reuse the same task-10 debug config values (batch size, epochs per task, and diagnostic flags) and only switch `training.planner_mode` to `hybrid`.

## Known Limitations
- Stage 8 currently supports `training.planner_control_recompute=per_batch` only.
- The hybrid control path restores task-loss gradient flow for training-time shared gating, but it does **not** yet implement differentiable rank control.
- Policy outputs remain pre-task structural signals and are not claimed to be the paper's final intended training rule beyond the operational truth established in this repo.
- Evaluation and inference profile semantics remain structurally driven; this pass does not redesign inference to use soft control-time rank modulation.

## Next Recommended Stage
Run a CIFAR-100 hybrid rerun with the same best audit baseline settings, then inspect:
- whether `PlannerControlTrainPath` now reports nonzero grad batches and nonzero drift
- whether learned `beta` values move meaningfully across tasks/layers
- whether hybrid control reduces the long shared-only regime without touching planner thresholds

If the control branch now learns but long-horizon forgetting remains high, the next planning step should focus on whether paper-faithful planner control needs richer continuous signals beyond `shared_gate`, not on reopening Stage 4 or Stage 5.
