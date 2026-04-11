# NH-LoRA

Paper-aligned implementation of **NH-LoRA: Future-Aware Structural Expansion of Low-Rank Adapters for Rehearsal-Free Class-Incremental Learning**.

This repository is organized to serve as the implementation-facing reference for NH-LoRA. The latest NH-LoRA design paper is the primary source of truth. Any remaining practical gaps or ambiguities are documented explicitly in [docs/paper_alignment.md](/C:/Users/Syauqi%20Nabil/research/self/NH-LoRA/docs/paper_alignment.md).

## What Is Implemented

- Frozen ViT backbone with configurable selected blocks and paper-consistent insertion points
- Shared Core LoRA plus expandable task slot bank
- Fixed-capacity prefix rank mask and additive rank expansion
- Task-State Encoder from feature mean, feature variance, gradient sketch, similarity, and entropy
- Layer-wise Horizon Planner with history-aware aggregation and explicit planner input `[z_t; h_t_tilde; |z_t-h_t_tilde|; z_t ⊙ h_t_tilde; e_l]`
- Pure `materialize_action`, separate `apply_structure_changes`, and paper action semantics
- Sparse cosine Instance Router over candidate slots
- Incremental cosine classifier with prototype-based imprinting
- Heuristic but explicit CHU decisions: merge, prune, keep, freeze
- `AdamW` default plus optional standard `SGD` backend for optimizer ablations and recipe-parity experiments
- Bootstrap mode for task 1
- Summary-only history bank
- Rehearsal-free class-incremental training loop
- Default-off classifier ablation knobs for diagnosing head dominance
- Multi-seed metrics and summaries
- One lightweight final model artifact per seed-run

## Repository Layout

```text
configs/
scripts/
src/
  datasets/
  backbones/
  models/
  engine/
  utils/
tests/
docs/
outputs/
  logs/
  metrics/
  summaries/
  checkpoints/
```

## Core NH-LoRA Flow

1. Warm-up sensing builds a temporary prototype-imprinted auxiliary head.
2. TSE encodes the current task state from warm-up statistics.
3. Horizon Planner produces raw per-layer signals.
4. `materialize_action` converts raw signals into pure structural plans.
5. `apply_structure_changes` mutates shared memory and slot bank explicitly.
6. Task training uses the paper loss branch for task 1 or task > 1.
7. CHU consolidates slot/shared memory after the task.
8. The task summary is appended to the history bank.
9. Post-consolidation inference profile is rebuilt for evaluation and future tasks.

Optional Stage 2 diagnostic knobs live under `training`: `classifier_lr_scale`,
`freeze_new_classifier_epochs`, and `freeze_all_classifier_epochs`. They default to
baseline behavior and are intended for head-dominance ablation, not as a change to
the NH-LoRA paper objective.

Optional Stage 5 routing diagnostics are enabled by `training.routing_debug_logging`.
They log planner/materialized/applied routing state, slot ids, retained inference
profile slots, and same-input train-vs-eval route comparisons without changing
planner, CHU, loss, or inference policy.

Optional Stage 7 planner diagnostics are enabled by `training.planner_audit_logging`.
They log planner threshold margins, action trajectories, optimizer membership,
pre-step planner grad statistics, parameter drift from initialization and post-Task-1,
and task-state/planner-input summaries without changing planner behavior.

Stage 8 adds an opt-in hybrid planner mode via `training.planner_mode=hybrid`.
In hybrid mode, structural policy remains pre-task and non-differentiable, while a
separate planner control branch recomputes learned shared-gate values during training
forward passes so task loss can update the control path honestly. Stage 8 Checkpoint A
does not change planner thresholds, CHU, or discrete structural actions, and it defers
soft-rank training.

Stage 9 extends `training.planner_audit_logging` with diagnostics for early-onset
shared-gate saturation in hybrid mode. These logs add pre-sigmoid beta-logit
summaries, post-sigmoid saturation fractions, branch-specific planner-policy vs
planner-control gradient reporting, shared-vs-slot contribution summaries, and
control-input similarity summaries. They are observational only and do not change
planner, router, CHU, classifier, or loss behavior.

Stage 10 keeps hybrid mode opt-in and stabilizes the learned shared gate with an
anchored residual parameterization. In `training.planner_mode=hybrid`, the
training-time control gate now starts from the policy-side `shared_gate` in the
applied plan, converts that anchor into logit space, adds a bounded residual
term, and maps back through sigmoid. The control head is zero-initialized so hybrid training begins at the
policy anchor instead of relearning an unconstrained absolute `beta`. The default
`training.planner_control_delta_logit_scale=2.0` is an initial validation
hypothesis for short-window reruns, not a claimed final tuned value. Stage 10
still does not tune thresholds, add regularizers, redesign CHU/router/classifier,
or enable soft-rank.

Stage 11 extends the same `training.planner_audit_logging` path with a
post-bootstrap residual-control audit. These logs keep behavior unchanged while
reporting `delta_raw` percentile and threshold summaries, bounded-transform derivative
collapse, control-head weight/bias norm and drift, bias-vs-activation
contributions to `delta_raw`, gradient flow through `delta_raw -> delta_logit ->
effective_logit`, cap-usage summaries around `planner_control_delta_logit_scale`,
and a per-layer residual interpretation summary that compares anchor gate,
effective gate, residual cap state, slot availability, and shared-vs-slot ratio.

Stage 12 keeps the Stage 10 anchored residual semantics but narrows the residual
path further in hybrid mode. The control representation is RMS-normalized just
before the residual gate head, and the bounded residual transform is now
`planner_control_delta_logit_scale * softsign(delta_raw)` instead of `tanh`.
This keeps the residual bounded in logit space while preserving more gradient
signal when `delta_raw` grows. Stage 12 does not tune thresholds, add
regularizers, redesign CHU/router/classifier behavior, change Stage 5
inference-profile semantics, or enable soft-rank.

Stage 13 keeps Stage 12 behavior intact and extends `training.planner_audit_logging`
with structural shared-only concentration diagnostics. These logs compare
policy-side growth rankings, cross-layer growth concentration, per-layer
structure lifecycle across tasks, route/usage concentration, and Pre-CHU vs
Post-CHU profile contraction so long-horizon shared-only collapse can be
classified without changing planner decisions, CHU, router behavior, classifier
behavior, or hybrid residual-control semantics.

Stage 14 keeps the same behavior and extends `training.planner_audit_logging`
with policy-side deconcentration diagnostics plus a requested-to-realized growth
trace. These logs add raw policy-logit decomposition, threshold-proximity and
history-conditioning summaries, cross-task rank stability, and explicit
`requested -> materialized -> applied -> fallback -> post_outcome` traces so we
can separate pure policy-side growth concentration from realization failure.
They remain observational and do not change thresholds, CHU, router behavior,
classifier behavior, residual-gate behavior, or soft-rank status.

Stage 16 still keeps NH-LoRA method semantics unchanged, but it makes
full-block injection `[0..11]` the global default in the base config and
strengthens the `imagenet_a` dataset contract. `ImageNet-A` now fails fast if
`train/` or `test/` is missing, if the split class-folder sets differ, or if
the layout does not contain exactly 200 classes. Benchmark sanity summaries
also log class counts, task counts, per-task class counts, and train/test
sample counts at seed start so dataset-layout issues are visible before they
are mistaken for method behavior.

Stage 17 still keeps NH-LoRA method semantics unchanged and extends the
existing retention/evaluation debug surface with task-boundary retention
diagnostics for CIFAR-100 full-block runs. These logs decompose forgetting into
per-old-task accuracy drops, teacher-vs-student old-logit drift, feature-drift
summaries, old-vs-new classifier calibration, eval-time route/profile
availability, and Pre-CHU vs Post-CHU forgetting deltas so retention failures
can be classified without changing planner decisions, residual gating, CHU,
router behavior, classifier behavior, or loss weights.

## Bootstrap Task 1

Task 1 is intentionally special:

- no history-aware similarity
- no teacher
- no KD
- no feature retention
- no growth penalty
- bootstrap shared-dominant structure plus one bootstrap slot per selected block
- light post-task consolidation followed by history-bank append

## Evaluation Policy

Evaluation does **not** use the old shortcut `all live slots + shared_gate=1.0`.

The default inference profile is built post-consolidation per layer from:

- latest consolidated shared gate
- surviving non-pruned slots with `retained_for_inference=True`
- usage ordering via `usage_ema` then `cumulative_usage`
- router `top-k` restriction

If the last structural action for a layer is `reuse_shared` or `freeze_old_strong_retention`, that layer defaults to shared-only inference.

## Benchmarks

Current final evaluation target set:

- CIFAR-100
- ImageNet-A
- ImageNet-R

Legacy optional adapter retained in the repo:

- OmniBenchmark

Dataset adapters are unified at the engine boundary but remain benchmark-specific internally where the benchmark requires it.

`ImageNet-A` uses a strict ImageFolder contract:

- `train/` and `test/` must both exist under `benchmark.data_root`
- the class-folder names in `train/` and `test/` must match exactly
- the benchmark must expose exactly 200 classes

## Local Validation

Only lightweight validation is intended in this environment:

```bash
python -m compileall src tests
python -m unittest tests.test_config_summary tests.test_paper_alignment_units tests.test_synthetic_continual_smoke tests.test_checkpoint_resume
```

## Run On SSH

Set dataset paths in the benchmark YAML files, then run:

```bash
bash scripts/run_cifar100.sh
bash scripts/run_imagenet_a.sh
bash scripts/run_imagenet_r.sh
```

Optional legacy benchmark:

```bash
bash scripts/run_omnibenchmark.sh
```

Direct CLI usage:

```bash
python -m src.engine.train --config configs/cifar100.yaml --seed 1 --benchmark cifar100 --output-root outputs
```

Outputs:

- logs: `outputs/logs/`
- raw metrics: `outputs/metrics/`
- summaries: `outputs/summaries/`
- final model artifact: `outputs/checkpoints/<benchmark>/<benchmark>_seed<seed>_final.pt`

## Notes

- Local smoke tests use the internal toy ViT path.
- Full ViT-B/16-IN21K validation still belongs on the SSH machine with `timm`.
- Benchmark runner scripts stream logs live to notebook output and save the same stream to `outputs/logs/`.
- Development status is tracked in [outputs/logs/dev_status.md](/C:/Users/Syauqi%20Nabil/research/self/NH-LoRA/outputs/logs/dev_status.md).
- `training.optimizer` supports `adamw` and `sgd`. SGD support is provided for optimizer ablations and recipe-parity experiments; it does not change NH-LoRA method semantics.
- AdamW and SGD results should not be treated as interchangeable. Even with the same parameter groups and cosine scheduler, optimization dynamics can differ materially.
