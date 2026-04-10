## Stage 16 Summary

Stage 16 is a repo-health and correctness pass, not a new NH-LoRA method patch.

### What changed

- `configs/base.yaml` now defaults `model.selected_blocks` to all 12 ViT blocks
  (`[0..11]`) as the global paper-faithful default.
- `ImageNet-A` now uses strict fail-fast validation through the shared
  ImageFolder benchmark builder:
  - `train/` must exist
  - `test/` must exist
  - train/test class-folder sets must match exactly
  - the benchmark must contain exactly 200 classes
- Benchmark sanity summaries are now logged at seed start:
  - benchmark class count
  - benchmark task count
  - per-task class count
  - per-task train/test sample count
  - ImageFolder split class-count and class-name preview

### What stayed unchanged

- planner thresholds
- planner policy/control semantics
- residual gate
- CHU
- router
- classifier
- soft-rank

### Why this stage exists

Stage 15 showed that poor `ImageNet-A` results could still be confounded by:

- benchmark adapter/layout mistakes
- silent class-folder mismatches
- sample starvation or unexpected task composition

Stage 16 removes that ambiguity before any benchmark-specific calibration work.

### Validation

Run for this stage:

- `python -m compileall src tests`
- `python -m unittest tests.test_config_summary tests.test_paper_alignment_units tests.test_synthetic_continual_smoke`

### Interpretation rule after Stage 16

If `ImageNet-A` remains poor after this pass, the result should be interpreted
first as a benchmark/config calibration issue or a later planner-calibration
issue, not as a silent dataset-adapter correctness bug.
