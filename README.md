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
- Bootstrap mode for task 1
- Summary-only history bank
- Rehearsal-free class-incremental training loop
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

- CIFAR-100
- CUB-200-2011
- ImageNet-R
- OmniBenchmark

Dataset adapters are unified at the engine boundary but remain benchmark-specific internally where the benchmark requires it.

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
bash scripts/run_cub200.sh
bash scripts/run_imagenet_r.sh
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
