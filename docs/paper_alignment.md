# NH-LoRA Paper Alignment

## Scope

This document records how the repository maps the latest NH-LoRA design paper into code, which implementation choices were made under ambiguity, and which gaps still remain.

The design paper is the primary source of truth. When paper detail was ambiguous, the implementation chose the most conservative interpretation that stayed closest to the equations, module descriptions, and planning flow in the paper.

## Paper Section -> Code Mapping

### Frozen Vision Backbone

- Paper intent: fully frozen ViT backbone with configurable selected blocks and paper-consistent insertion points.
- Code:
  - [src/backbones/vision_transformer.py](/C:/Users/Syauqi%20Nabil/research/self/NH-LoRA/src/backbones/vision_transformer.py)
  - [src/models/nh_lora.py](/C:/Users/Syauqi%20Nabil/research/self/NH-LoRA/src/models/nh_lora.py)

### Shared Core LoRA And Expandable Task Slot Bank

- Paper intent: shared reusable low-rank memory plus task slots with fixed-capacity rank masks.
- Code:
  - [src/models/lora.py](/C:/Users/Syauqi%20Nabil/research/self/NH-LoRA/src/models/lora.py)
  - `ProjectionBank`
  - `NHLoRALayer.apply_structure_change`
  - `NHLoRALayer.active_rank_mask`

### Task-State Encoder (TSE)

- Paper intent: task state from feature mean, feature variance or covariance diagonal, gradient sketch, similarity, and entropy.
- Code:
  - [src/models/task_state.py](/C:/Users/Syauqi%20Nabil/research/self/NH-LoRA/src/models/task_state.py)
  - `TaskStateEncoder.build_raw_task_vector`
  - `TaskStateEncoder.forward`
  - [src/engine/trainer.py](/C:/Users/Syauqi%20Nabil/research/self/NH-LoRA/src/engine/trainer.py)
  - `NHLoRATrainer._run_warmup_sensing`

### Horizon Planner (HP)

- Paper intent: layer-wise planner with history-aware aggregation.
- Code:
  - [src/models/planner.py](/C:/Users/Syauqi%20Nabil/research/self/NH-LoRA/src/models/planner.py)
  - `HorizonPlanner.aggregate_history`
  - `HorizonPlanner.compose_layer_input`
  - `HorizonPlanner.forward`

### Materialize Action And Structural Mutation

- Paper intent: planner outputs raw signals first, then `materialize_action`, then explicit structural mutation.
- Code:
  - [src/models/planner.py](/C:/Users/Syauqi%20Nabil/research/self/NH-LoRA/src/models/planner.py)
  - `materialize_action`
  - [src/models/lora.py](/C:/Users/Syauqi%20Nabil/research/self/NH-LoRA/src/models/lora.py)
  - `NHLoRALayer.apply_structure_change`
  - [src/models/nh_lora.py](/C:/Users/Syauqi%20Nabil/research/self/NH-LoRA/src/models/nh_lora.py)
  - `NHLoRAModel.apply_structure_changes`

### Instance Router (IR)

- Paper intent: sparse top-k instance-level cosine routing over candidate slots.
- Code:
  - [src/models/lora.py](/C:/Users/Syauqi%20Nabil/research/self/NH-LoRA/src/models/lora.py)
  - `NHLoRALayer.route`

### Incremental Cosine Head (ICH)

- Paper intent: incremental cosine classifier with class expansion and optional imprinting.
- Code:
  - [src/models/classifier.py](/C:/Users/Syauqi%20Nabil/research/self/NH-LoRA/src/models/classifier.py)
  - [src/engine/trainer.py](/C:/Users/Syauqi%20Nabil/research/self/NH-LoRA/src/engine/trainer.py)
  - `NHLoRATrainer._expand_classifier_for_task`
- Diagnostic note: `training.classifier_lr_scale`, `training.freeze_new_classifier_epochs`,
  and `training.freeze_all_classifier_epochs` are default-off Stage 2 ablation knobs for
  testing head dominance. They do not add rehearsal memory or alter the paper objective
  when left at defaults.

### CHU

- Paper intent: consolidation and homeostasis based on usage, stability, redundancy, and planner consolidation flag.
- Code:
  - [src/models/chu.py](/C:/Users/Syauqi%20Nabil/research/self/NH-LoRA/src/models/chu.py)
  - `ConsolidationHomeostasisUnit.decide_slot`
  - `ConsolidationHomeostasisUnit.consolidate_layer`

### Bootstrap Task 1

- Paper intent: task 1 is special and history-free.
- Code:
  - [src/models/nh_lora.py](/C:/Users/Syauqi%20Nabil/research/self/NH-LoRA/src/models/nh_lora.py)
  - `NHLoRAModel.build_bootstrap_plan`
  - [src/engine/trainer.py](/C:/Users/Syauqi%20Nabil/research/self/NH-LoRA/src/engine/trainer.py)
  - `NHLoRATrainer._prepare_task_context`
  - `NHLoRATrainer._compute_loss`

### Training Objective

- Paper intent:
  - task 1: `L_cls + L_orth + L_rank + L_route`
  - task > 1: `L_cls + L_kd + L_feat + L_orth + L_rank + L_grow + L_route`
- Code:
  - [src/models/losses.py](/C:/Users/Syauqi%20Nabil/research/self/NH-LoRA/src/models/losses.py)
  - [src/engine/trainer.py](/C:/Users/Syauqi%20Nabil/research/self/NH-LoRA/src/engine/trainer.py)
  - `NHLoRATrainer._compute_loss`

### Evaluation And Inference Planning

- Paper intent: paper is not fully explicit; inference should stay aligned with structural decisions.
- Code:
  - [src/models/nh_lora.py](/C:/Users/Syauqi%20Nabil/research/self/NH-LoRA/src/models/nh_lora.py)
  - `NHLoRAModel.build_inference_profile`
  - [src/engine/trainer.py](/C:/Users/Syauqi%20Nabil/research/self/NH-LoRA/src/engine/trainer.py)
  - `NHLoRATrainer._evaluate_up_to`

### Final Model Artifact

- Code:
  - [src/utils/checkpoint.py](/C:/Users/Syauqi%20Nabil/research/self/NH-LoRA/src/utils/checkpoint.py)
  - [src/engine/trainer.py](/C:/Users/Syauqi%20Nabil/research/self/NH-LoRA/src/engine/trainer.py)
  - `NHLoRATrainer._save_final_model_artifact`
  - [src/engine/train.py](/C:/Users/Syauqi%20Nabil/research/self/NH-LoRA/src/engine/train.py)

## Final Slot Compatibility Definition

Official compatibility is:

- cosine similarity between normalized task embedding `z_t` and normalized slot key

This definition is used consistently for:

- `expand_rank_existing_slot`
- `open_new_slot -> expand_rank_existing_slot` fallback when slot capacity is full
- `compatibility_scores` stored in each materialized layer plan

Implementation note:

- slot keys are stored in the task-embedding space, not the backbone feature space
- the router query projection maps backbone tokens into the same slot-key space

## Operational Inference Policy Chosen Under Paper Ambiguity

The paper is not fully explicit about inference-time planning, so the repository fixes the following default policy:

1. Evaluation does not activate all live slots by default.
2. Each layer uses the latest consolidated shared gate.
3. Candidate slots come only from surviving non-pruned slots with `retained_for_inference=True`.
4. Candidate slots are sorted by `usage_ema`, then `cumulative_usage`, then clipped to `router_topk`.
5. If the latest structural action for a layer is `reuse_shared` or `freeze_old_strong_retention`, that layer defaults to shared-only inference.
6. Instance routing remains sparse and instance-level within that candidate set.

This policy is implemented in [src/models/nh_lora.py](/C:/Users/Syauqi%20Nabil/research/self/NH-LoRA/src/models/nh_lora.py) and validated in [tests/test_paper_alignment_units.py](/C:/Users/Syauqi%20Nabil/research/self/NH-LoRA/tests/test_paper_alignment_units.py) and [tests/test_synthetic_continual_smoke.py](/C:/Users/Syauqi%20Nabil/research/self/NH-LoRA/tests/test_synthetic_continual_smoke.py).

## Ambiguity Decisions

### Planner Materialization

- Chosen policy: planner produces raw signals first, then hard structural plans are materialized, then structure is mutated explicitly.
- Rationale: closest to the paper flow and keeps planning traceable.

### freeze_old_strong_retention

- Chosen policy:
  - freeze existing slots at the affected layer
  - force shared-only candidate set for that layer
  - include the layer in feature-retention layers during task training
- Rationale: conservative interpretation of strong retention without inventing a new auxiliary objective.

### CHU

- Chosen policy: heuristic CHU remains acceptable, but merge/prune/keep/freeze decisions are explicit and use usage, stability, redundancy, and `consolidate_flag`.
- Rationale: the paper explicitly allows heuristic CHU.

### L_rank

- Chosen policy: `L_rank` is computed from realized active-rank masks, not planner scores.
- Status: this is the most paper-faithful choice available in the current discrete structure design.
- Important note: because active rank changes through planning and structural mutation, `L_rank` can become near-constant during a task and may act more like a structural cost than a strong optimizer-driving signal.

## Final Artifact Policy

The runtime no longer writes periodic training checkpoints or exposes resume support.

Instead, when `experiment.save_checkpoints=true`, the repository writes exactly one lightweight final model artifact per seed-run:

- `outputs/checkpoints/<benchmark>/<benchmark>_seed<seed>_final.pt`

The final artifact contains:

- `model_state`
- `model_structure_state`
- `inference_profile`
- `benchmark`
- `seed`
- `final_metrics`

It intentionally does not contain optimizer state, scheduler state, current-task context, or large periodic snapshot history.

## Remaining Gaps

- The local smoke path still uses the internal toy ViT backend; full ViT-B/16-IN21K runtime still needs SSH validation.
- Benchmark-scale continual runs have not been executed in this environment by design.
- CHU remains heuristic, though now explicit and closer to the paper.
- `L_rank` is paper-faithful as a realized structural cost, but may provide weak gradient signal once structure is fixed inside a task.

## Validation Used In This Pass

- `python -m compileall src tests`
- `python -m unittest tests.test_config_summary tests.test_paper_alignment_units tests.test_synthetic_continual_smoke tests.test_checkpoint_resume`
