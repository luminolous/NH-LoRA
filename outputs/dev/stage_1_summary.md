# Stage 1 Summary - Instrumentation Only

## Stage Hypothesis
NH-LoRA forgetting may be caused by inert adapters, classifier dominance/classifier-space collapse, or routing/inference mismatch. Stage 1 only adds diagnostics to separate those causes without changing training objective, loss weights, dataset protocol, planner policy, or inference policy.

## Files Changed
- `PLAN3.md`
- `configs/base.yaml`
- `src/engine/trainer.py`
- `src/models/lora.py`
- `tests/test_paper_alignment_units.py`
- `outputs/dev/stage_1_summary.md`

## Git Checkpoint
- Branch: `codex-stage-1-instrumentation`
- Starting commit: `7f66216e3cc19d574f492ef7525eb64a944cda44`
- Pre-stage status: clean tracked files with ignored/generated output directories excluded.
- Post-validation checkpoint commit: this Stage 1 checkpoint commit.

## Diff Summary
- Added default-off Stage 1 debug flags to `configs/base.yaml`.
- Added adapter delta collection in `NHLoRALayer` when a private debug accumulator is supplied.
- Added trainer diagnostics for adapter deltas, final feature diff, old classifier drift, gradient norms by group, and old/new logit margin.
- Added unit coverage for debug gating and emitted Stage 1 diagnostic log records.
- Added the revised staged roadmap in `PLAN3.md` and this stage artifact.

## Validation Commands
- Passed: `python -m compileall src tests`
- Passed: `python -m unittest tests.test_paper_alignment_units tests.test_synthetic_continual_smoke`
- Passed because config defaults changed: `python -m unittest tests.test_config_summary`
- Passed: `git diff --check`

## Validation Result
Passed. Unit/smoke validation reported `Ran 33 tests ... OK`; config summary validation reported `Ran 1 test ... OK`. `git diff --check` reported only line-ending warnings for files already touched by the patch and no whitespace errors.

## Main Diagnostic/Log Findings
- Existing CIFAR-100 log reviewed from `C:\Users\Syauqi Nabil\Downloads\CIFAR100_LOG_TRAINING.txt`.
- Prior run symptom: task 2 collapses task 1 accuracy to `0.0000` while task 2 remains around `0.599-0.603`.
- Prior run symptom: retention feature diff frequently logs as `0.0000e+00`.
- Prior run symptom: `kd_raw` is near numerical zero, roughly `1e-8` to `1e-7`.
- Prior run symptom: routing debug sometimes has empty candidate pools and sometimes flat distributions around `0.5/0.5`.
- New instrumentation targets: adapter delta stats, final feature diff, classifier drift, grad norms by group, old/new logit margin, and existing routing usage summary.

## Next-Stage Decision
Stage 1 implementation is complete locally. Do not jump directly to Stage 3/4/5 yet: first run a short CIFAR-100 task-3 diagnostic pass with the new Stage 1 flags enabled. If that run shows adapter delta or adapter grad near zero while slots are active, go to Stage 4. If adapters are active but classifier drift/logit margins dominate while final feature drift is small, run Stage 2 and then likely Stage 3. If routing/inference mismatch is obvious, go to Stage 5. If none of those conditions is decisive, run Stage 2.

## Risks Or Open Questions
- CIFAR-100 short rerun to task 3 has not been run in this patch.
- Stage 1 diagnostics are default-off; forgetting behavior must be rechecked with the new flags enabled before deciding Stage 2/3/4/5.
