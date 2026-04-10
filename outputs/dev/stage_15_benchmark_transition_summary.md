# Stage 15 — Benchmark Transition And Portability Gate

## Summary

Stage 15 keeps NH-LoRA method semantics unchanged and focuses on two practical goals:

1. transition the repository benchmark surface toward the current target set
2. separate portability checking from config-performance exploration

Checkpoint A is implemented in this pass:
- native `imagenet_a` benchmark support was added
- `cub200` was removed from the public registry/config/script surface
- Stage 15 probe configs were added without changing planner, CHU, router, classifier, or residual-gate behavior

Checkpoint B is prepared but not executed in this local workspace because benchmark-scale runs remain an SSH-side responsibility.

## Why The Target Set Changed

The current intended final evaluation set is:

- `cifar100`
- `imagenet_r`
- `imagenet_a`

CIFAR-100 remains the lightweight debug benchmark. `imagenet_r` is the first non-CIFAR portability gate because it already existed in the repo. `imagenet_a` is added now so the final benchmark set is wired before freeze decisions.

## What Changed

### Added

- `src/datasets/imagenet_a.py`
- `src/datasets/imagefolder_benchmark.py`
- `configs/imagenet_a.yaml`
- `configs/imagenet_a_hybrid.yaml`
- `configs/imagenet_r_hybrid.yaml`
- `scripts/run_imagenet_a.sh`

### Removed

- `src/datasets/cub200.py`
- `configs/cub200.yaml`
- `scripts/run_cub200.sh`
- `cub200` dataset-registry entry

### Updated

- `scripts/run_all.sh` now targets the current main benchmark trio:
  - `cifar100`
  - `imagenet_a`
  - `imagenet_r`
- docs/config references were updated to remove CUB and document the new benchmark transition

## ImageNet-A Implementation Notes

`imagenet_a` follows the same continual ImageFolder contract already used by `imagenet_r`:

- expected layout:
  - `<data_root>/train/<class_name>/*`
  - `<data_root>/test/<class_name>/*`
- transforms:
  - same ImageNet-style train/test transforms used by `imagenet_r`
- task construction:
  - class-incremental splits via `num_tasks` and `classes_per_task`

The external `data.py` / `data_manager.py` references were used only to confirm:
- train/test directory layout
- 224-sized ImageNet transform family
- path-based ImageFolder loading style

No external framework-specific data-manager design was copied into NH-LoRA.

## Selected-Block Strategy

Stage 15 keeps `selected_blocks = [6, 7, 8, 9, 10, 11]` as the first portability baseline so benchmark-switch results are not confounded by a layer-scope change.

The all-layer hypothesis is intentionally treated as a config/performance
question, not as a bug fix. In the later Stage 16 repo state, full-block
`[0..11]` is promoted directly through `configs/base.yaml` instead of being
maintained as a separate CIFAR-only probe config.

## Validation In This Pass

Implemented local validation target:

- `python -m compileall src tests`
- `python -m unittest tests.test_config_summary tests.test_paper_alignment_units tests.test_synthetic_continual_smoke`

New lightweight checks added in this pass cover:

- `imagenet_a` registry/config resolution
- `cub200` registry removal
- `imagenet_a` ImageFolder task construction
- the benchmark-transition registry/config surface without method changes

## Checkpoint B Commands

### B1 — ImageNet-R portability baseline (`selected_blocks = [6..11]`)

```bash
python -m src.engine.train --config configs/imagenet_r_hybrid.yaml --seed 1 --benchmark imagenet_r --output-root outputs
```

### B2 — CIFAR-100 all-layer config probe (`selected_blocks = [0..11]`)

Run only after B1 is clean:

```bash
python -m src.engine.train --config configs/cifar100_hybrid.yaml --seed 1 --benchmark cifar100 --output-root outputs
```

### ImageNet-A early sanity probe

Run only after `imagenet_r` portability looks healthy:

```bash
python -m src.engine.train --config configs/imagenet_a_hybrid.yaml --seed 1 --benchmark imagenet_a --output-root outputs
```

## Freeze Gate Status

No freeze decision is claimed yet in this implementation pass.

Current status:
- Checkpoint A: complete
- Checkpoint B1/B2: prepared, not yet run here

So the honest decision at the end of this pass is:

**the codebase is not yet frozen; portability still needs to be checked on `imagenet_r`, and the all-layer CIFAR setting remains a separate config hypothesis rather than a mainline default.**
