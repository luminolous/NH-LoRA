# NH-LoRA Development Status

## Latest updates
- Synced audit documents from `nh_lora_codex_pack.zip` into the repo root.
- Created the target repo skeleton under `configs/`, `scripts/`, `src/`, `tests/`, and `outputs/`.
- Locked the revised dataset plan:
  - engine-level interface stays uniform,
  - benchmark adapters remain benchmark-specific where needed,
  - OmniBenchmark stays realm-wise.
- Locked the additional end-to-end synthetic continual smoke test requirement.
- Implemented benchmark-specific dataset adapters in `src/datasets/`.
- Implemented the internal ViT fallback backbone and the NH-LoRA core modules in `src/backbones/` and `src/models/`.
- Implemented the rehearsal-free incremental training engine, summarizer CLI, and benchmark runners contract in `src/engine/`.
- Added lightweight validation tests, including the two-task synthetic continual integration smoke test.
- Removed the legacy top-level dataset utilities that were replaced by the new `src/datasets/` pipeline.

## Files touched
- `SPEC.md`
- `PLANS.md`
- `IMPLEMENT.md`
- `RUNS.md`
- `PROMPT_CODEX_ID.md`
- `configs/`
- `scripts/`
- `src/`
- `tests/`
- `README.md`
- `utils/data.py` (removed)
- `utils/data_manager.py` (removed)
- `outputs/logs/dev_status.md`

## Completed
- Document sync from the zip archive.
- Initial repo scaffolding.
- Implementation assumptions recorded before coding core modules.
- Unified engine-facing dataset contract with benchmark-specific adapters for CIFAR-100, CUB-200-2011, ImageNet-R, and OmniBenchmark.
- Frozen ViT wrapper with internal toy-ViT fallback for smoke tests and `timm` path for SSH runtime.
- Shared Core LoRA, expandable slot banks, rank masks, task-state encoder, horizon planner, materialize-action logic, instance routing, cosine classifier, CHU, and history bank.
- Rehearsal-free task loop with bootstrap task-1 branch, teacher-based task>1 branch, evaluation, metric writing, and multi-seed summarization entrypoints.
- Synthetic continual integration smoke test covering task 1 bootstrap and task 2 history-aware flow.
- Local sanity checks passed:
  - `python -m compileall src tests`
  - `python -m unittest tests.test_config_summary tests.test_synthetic_continual_smoke`

## Pending
- Full benchmark runs on real datasets.
- Validation of the `timm` ViT-B/16-IN21K path on the SSH machine.
- Validation of the exact dataset layouts on the SSH machine for CUB-200-2011, ImageNet-R, and OmniBenchmark.
- Stress-testing checkpoint size, runtime, and long continual sequences on real workloads.

## Assumptions
- Local development must avoid heavy compute and package installation.
- Benchmark execution on the SSH machine may use `timm` for ViT-B/16-IN21K, but local smoke tests must work without it.
- Benchmark parsers are allowed to differ internally as long as the engine receives a consistent continual-learning contract.
- OmniBenchmark is implemented as a realm-wise stress-test adapter with explicit assumptions documented in code and docs.
- CIFAR-100 is expected in extracted `cifar-100-python` format.
- CUB-200-2011 is expected in official metadata-file layout with `images.txt`, `image_class_labels.txt`, and `train_test_split.txt`.
- ImageNet-R is expected in `train/` and `test/` folder layout with class subdirectories.
- OmniBenchmark is expected as realm directories containing `train/` and `test/` class folders.

## Risks to verify later on SSH
- Full `timm`-backed ViT runtime path.
- Real dataset directory layouts for CUB-200-2011, ImageNet-R, and OmniBenchmark.
- Full training/evaluation throughput and long-run checkpoint behavior.
- Whether the current heuristic CHU thresholds need per-benchmark tuning after real runs.
- Whether planner thresholds and growth penalties need paper-driven retuning after benchmark-level validation.
