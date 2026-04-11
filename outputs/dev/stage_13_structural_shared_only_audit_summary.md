# Stage 13 Structural Shared-Only Audit

## Stage Goal

Stage 13 extends the existing `training.planner_audit_logging` path with
diagnostics for the long-horizon structural bottleneck that remains after Stage
12 stabilized the hybrid residual-control path.

The stage is intentionally diagnostics-only:

- no threshold tuning
- no residual-gate changes
- no CHU behavior changes
- no router or classifier redesign
- no inference-profile redesign
- no soft-rank

## Why This Stage Exists

Stage 12 removed the earlier residual-control collapse and kept
`planner_control` trainable through long horizons, but CIFAR-100 full-task
metrics still showed that structure growth remained concentrated, with Layer 6
preserving multi-slot diversity while later layers ended up `shared_only=True`.

Stage 13 audits that shift directly:

- policy-side growth rankings and cross-layer winner concentration
- per-layer structure lifecycle across tasks
- route and usage concentration after structure exists
- Pre-CHU vs Post-CHU contraction
- explicit Layer 6 focus logging

## Files Changed

- `src/engine/trainer.py`
- `tests/test_paper_alignment_units.py`
- `README.md`
- `docs/paper_alignment.md`

## Implementation Notes

- Added pure helper summaries for:
  - open/expand candidate ranking
  - per-task growth summary
  - cross-task winner concentration
  - route/usage concentration
  - Pre-CHU vs Post-CHU profile diff
  - layer lifecycle aggregation across tasks
- Extended planner-audit logging with:
  - `PlannerGrowthTask`
  - `PlannerGrowthConcentration`
  - `PlannerStructureTask`
  - `PlannerCHUDiff`
  - `PlannerStructureTrajectory`
  - `PlannerLayer6Audit`
  - `PlannerStructuralConcentration`
- Reused existing lifecycle logging under `planner_audit_logging=true` so
  `PreCHUProfile` and `PostCHUProfile` remain observable even when
  `routing_debug_logging=false`.

## Validation

Commands run:

```bash
python -m compileall src tests
python -m unittest tests.test_config_summary tests.test_paper_alignment_units tests.test_synthetic_continual_smoke
```

Result status:

- `python -m compileall src tests` passed
- `python -m unittest tests.test_config_summary tests.test_paper_alignment_units tests.test_synthetic_continual_smoke` passed (`69 tests`)
- `git diff --check` passed with line-ending warnings only

## Rerun Command

```bash
python -m src.engine.train --config configs/cifar100_hybrid.yaml --seed 1 --benchmark cifar100 --output-root outputs
```

## Dominant Bucket Classification

Pending rerun. Stage 13 is complete only after Task-10 evidence classifies one
dominant structural driver:

1. policy-side conservatism
2. cross-layer concentration
3. lifecycle / usage bottleneck
4. CHU / mixed case with one primary driver

## Layer 6 Focus

Stage 13 adds explicit Layer 6 summaries so the rerun can answer:

- why Layer 6 keeps winning or surviving growth
- whether it stays non-`shared_only` because of policy, lifecycle, routing, or
  usage concentration
- how its shared-vs-slot contribution differs from layers that collapse to
  `shared_only`

## What Stage 13 Does Not Change

- planner thresholds and quadrants
- hybrid residual gate math from Stage 12
- CHU merge/prune/keep/freeze rules
- router sparsity behavior
- classifier behavior
- inference-profile semantics
- soft-rank

## Next Recommendation

Do not patch structure yet. Use the Stage 13 rerun to classify one dominant
structural bottleneck first, then plan the narrowest fix stage against that
classified cause.
