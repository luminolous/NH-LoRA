# Stage 14 Policy Deconcentration Audit

## Stage Goal

Stage 14 extends the existing `training.planner_audit_logging` path with a
diagnostics-only audit for policy-side growth opportunity concentration after
Stage 12 stabilized the hybrid residual-control path and Stage 13 showed the
remaining long-horizon bottleneck is structural.

This stage is intentionally observational:

- no threshold tuning
- no residual-gate changes
- no CHU behavior changes
- no router or classifier redesign
- no inference-profile redesign
- no soft-rank

## Why This Stage Exists

Stage 13 showed that the remaining bottleneck is not a separate global top-1
allocator. Each layer independently produces planner signals, requests a regime
through `decide_action()`, and only then goes through materialization and
application.

That means the next diagnosis must distinguish between:

- pure policy-side concentration, where non-L6 layers rarely request growth
- realization failure, where non-L6 layers sometimes request growth but fail in
  `materialized` / `applied` / fallback paths

## Files Changed

- `src/models/planner.py`
- `src/engine/trainer.py`
- `tests/test_paper_alignment_units.py`
- `README.md`
- `docs/paper_alignment.md`

## Implementation Notes

- `PlannerSignals` now carries raw policy logits plus activation-vs-bias
  decomposition metadata from the frozen policy head.
- Stage 14 extends planner-audit logging with:
  - `PlannerPolicyLogits`
  - `PlannerPolicyDecomposition`
  - `PlannerThresholdProximity`
  - `PlannerPolicyRankStability`
  - `PlannerRealizationTrace`
  - `PlannerPolicyDeconcentration`
  - `PlannerRequestedAppliedTrace`
- Added pure helper summaries for:
  - threshold proximity
  - cross-task rank stability
  - history-conditioning concentration
  - requested/materialized/applied trace aggregation
  - grouped fallback-reason and final-outcome summaries

## Validation

Commands run:

```bash
python -m compileall src tests
python -m unittest tests.test_config_summary tests.test_paper_alignment_units tests.test_synthetic_continual_smoke
```

Result status:

- `python -m compileall src tests` passed
- `python -m unittest tests.test_config_summary tests.test_paper_alignment_units tests.test_synthetic_continual_smoke` passed (`71 tests`)
- `git diff --check` passed with line-ending warnings only

## Rerun Command

```bash
python -m src.engine.train --config configs/cifar100_hybrid.yaml --seed 1 --benchmark cifar100 --output-root outputs
```

## Requested -> Materialized -> Applied Trace

Stage 14 adds explicit trace logging so the rerun can answer, for Layer 6,
Layer 9, and grouped layers 7/8/10/11:

- frequency of requested growth
- frequency of materialized growth
- frequency of applied growth
- dominant fallback reasons
- final shared-only vs non-shared-only outcome

## What Stage 14 Does Not Change

- planner thresholds and quadrants
- hybrid residual gate math from Stage 12
- CHU merge/prune/keep/freeze rules
- router sparsity behavior
- classifier behavior
- inference-profile semantics
- soft-rank

## Next Recommendation

Use the Stage 14 rerun to classify whether the structural bottleneck is driven
primarily by:

1. static policy-layer bias
2. near-threshold conservatism
3. history-conditioned concentration
4. realization failure after growth is requested
5. a mixed case with one dominant driver
