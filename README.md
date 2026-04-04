# NH-LoRA

Paper-oriented implementation of **NH-LoRA: Future-Aware Structural Expansion of Low-Rank Adapters for Rehearsal-Free Class-Incremental Learning**.

The implementation follows the latest NH-LoRA design paper first, then the synced repo documents:
- `SPEC.md`
- `PLANS.md`
- `IMPLEMENT.md`
- `RUNS.md`
- `PROMPT_CODEX_ID.md`

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
outputs/
  logs/
  metrics/
  summaries/
  checkpoints/
```

## Implemented NH-LoRA Components

- Frozen Vision Transformer backbone wrapper
- Shared Core LoRA
- Expandable Task Slot Bank
- Fixed-capacity dynamic rank mask
- Task-State Encoder
- Horizon Planner
- Materialize Action
- Instance Router
- Incremental cosine classifier head
- Consolidation and Homeostasis Unit
- Bootstrap mode for task 1
- History bank based on summary statistics
- Rehearsal-free class-incremental training loop
- Multi-seed metric summarization

## Benchmark Support

- CIFAR-100 via a natural CIFAR-100 pickle adapter
- CUB-200-2011 via metadata-aware benchmark-specific parsing
- ImageNet-R via benchmark-specific folder parsing
- OmniBenchmark via realm-wise benchmark-specific task building

The engine contract is unified, but each benchmark adapter stays benchmark-specific where needed.

## Local Validation

Only lightweight validation is intended in this environment:

```bash
python -m unittest tests.test_config_summary
python -m unittest tests.test_synthetic_continual_smoke
```

The synthetic continual smoke test checks:
- task 1 bootstrap mode without teacher or history-aware similarity
- task-state creation
- bootstrap slot materialization
- task 2 planning/materialization/routing/classifier expansion
- CHU invocation
- history bank growth

## Running On The SSH Server

Set the dataset paths in each benchmark config first, then run:

```bash
bash scripts/run_cifar100.sh
bash scripts/run_cub200.sh
bash scripts/run_imagenet_r.sh
bash scripts/run_omnibenchmark.sh
```

Or run all:

```bash
bash scripts/run_all.sh
```

Use `nohup`, `tmux`, or `screen` on the SSH machine for long experiments. Logs are written to `outputs/logs/`, raw metrics to `outputs/metrics/`, summaries to `outputs/summaries/`, and lightweight checkpoints to `outputs/checkpoints/`.

## Backbone Runtime Note

Default benchmark configs target `timm` for ViT-B/16-IN21K on the SSH machine. Local smoke tests should use the internal toy ViT path instead of requiring `timm`.

## Status Tracking

Development progress and assumptions are tracked in:

```text
outputs/logs/dev_status.md
```
