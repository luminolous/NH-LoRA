# PLAN3.md - NH-LoRA Forgetting Diagnosis Plan

## Summary
This plan diagnoses NH-LoRA catastrophic forgetting with a conservative, audit-friendly workflow: one stage, one hypothesis, one patch, one validation cycle. The first code patch after this plan must be **Stage 1: instrumentation-only**. Later classifier, adapter, and routing changes are conditional on evidence from prior stages.

Design constraints are fixed: no rehearsal memory, frozen backbone remains frozen, no speculative planner redesign, no large objective change before evidence exists, and baseline behavior must remain identical when new flags are disabled.

## Mandatory Execution Rules
- One Codex thread may work on only one stage at a time.
- One patch may answer only one primary hypothesis.
- Do not combine Stage 1 with Stage 2, 3, 4, or 5 in the same patch.
- Do not expand scope if the current stage acceptance criteria are not met.
- If validation fails, the stage is incomplete until fixed or explicitly documented as blocked.
- Do not change loss/objective, inference policy, dataset split, or benchmark protocol unless the active stage explicitly permits it.

## Git Checkpoint Protocol
- Before each stage: record `git status --short`, create or identify a checkpoint branch/commit, and note the starting commit hash.
- After implementation and passing validation: create a second checkpoint commit for that stage.
- Each stage must record: changed files, diff summary, validation commands/results, rerun log paths, and the next-stage decision.
- If a stage is abandoned or blocked, record the reason and do not proceed to the next stage as completed.

## Required Stage Artifact
Each stage must write `outputs/dev/stage_X_summary.md` with:
- stage hypothesis
- files changed
- validation commands
- validation result
- main diagnostic/log findings
- next-stage decision
- risks or open questions

## Stage 1 - Instrumentation Only
Objective: determine whether forgetting is driven by inert adapters, classifier dominance, classifier-space collapse, or routing/inference mismatch without changing training behavior.

Required outputs: config-gated diagnostics for `delta_shared` and `delta_slots` norm/mean_abs per selected block/module, final normalized feature diff student-vs-teacher, old classifier weight drift, grad norm by group (`shared`, `slot`, `router`, `planner`, `classifier`), old/new logit margin, and routing usage summary.

Acceptance criteria: baseline behavior is unchanged when debug flags are off; debug logs appear clearly when enabled; smoke/unit tests pass; `outputs/dev/stage_1_summary.md` contains concrete readings from at least one local smoke run or CIFAR-100 log review; no objective/loss/inference policy change is introduced.

Next-stage decision rules: if adapter delta or adapter grad is approximately zero while slots are active, go to Stage 4; if adapter is active but final feature drift is small and classifier drift/logit margin dominates, go to Stage 3 after Stage 2 ablation; if routing train/eval or inference profile mismatch is obvious, go to Stage 5; otherwise run Stage 2 to test head dominance.

## Stage 2 - Head-Dominance Ablation
Objective: test whether current-task fitting is dominated by classifier updates rather than NH-LoRA memory path.

Required outputs: default-off config controls for classifier LR scaling and optional early classifier update limiting; logs comparing current-task accuracy, per_task_acc, forgetting, adapter deltas, final feature drift, classifier drift, and grad group norms.

Acceptance criteria: baseline config remains identical to current behavior; each ablation is independently toggled; validation passes; a short CIFAR-100 run to at least task 3 is run if feasible, or explicitly marked unavailable; `outputs/dev/stage_2_summary.md` compares baseline vs ablation evidence.

Next-stage decision rules: if lowering/limiting classifier updates increases adapter activity or improves old-task retention, prioritize Stage 3; if adapters remain inert across ablations, go to Stage 4; if routing/inference differences dominate the evidence, go to Stage 5; if validation fails, do not proceed.

## Stage 3 - Small Classifier Alignment Patch
Objective: apply only small, optional classifier-space alignment if evidence shows adapter path is active but old/new class competition collapses.

Required outputs: one or two default-off classifier alignment options such as post-task normalization/re-centering or old-vs-new logit calibration; before/after logs for per_task_acc, forgetting, old/new margin, old classifier drift, and classifier weight norms by old/new group.

Acceptance criteria: baseline unchanged when flags are off; no rehearsal data is introduced; no large alignment loss is added; validation passes; `outputs/dev/stage_3_summary.md` states whether classifier-space collapse was reduced.

Next-stage decision rules: if old/new margin improves without harming current-task accuracy severely, keep the patch candidate and proceed to Stage 5 audit; if classifier alignment does not change the collapse and adapters are weak, go to Stage 4; if routing/inference mismatch remains suspicious, go to Stage 5; if validation fails, stage is incomplete.

## Stage 4 - Adapter Activation Audit And Patch
Objective: verify that planner output, rank masks, shared gate, candidate slots, active slots, and routing weights genuinely affect adapter output and gradients.

Required outputs: tests or assertions proving active shared/slot paths change forward outputs; diagnostics for active adapter gradient flow; a minimal bug fix only if wiring/scaling/gating/materialization/rank-mask behavior is demonstrably wrong.

Acceptance criteria: active adapter paths measurably affect output in a focused test; shared/slot grads are nonzero when expected; no planner redesign or objective redesign is introduced; validation passes; `outputs/dev/stage_4_summary.md` names the exact bug or states no bug found.

Next-stage decision rules: if adapter inertness is fixed, rerun short CIFAR-100 and then continue to Stage 5; if no adapter bug is found but classifier collapse remains, return to Stage 3; if routing profile explains inactivity, go to Stage 5; if validation fails, do not proceed.

## Stage 5 - Routing And Inference Profile Audit
Objective: determine whether post-consolidation inference profile or train-vs-eval routing mismatch worsens forgetting.

Required outputs: diagnostics for old-task vs new-task routing usage, routing entropy per layer, slot overlap by task, retained slots that are never dominant, and comparison between train-time candidate pools and eval-time inference profile.

Acceptance criteria: current inference policy is documented; diagnostics are config-gated; any patch is local to a proven mismatch; no full router redesign is introduced; validation passes; `outputs/dev/stage_5_summary.md` states whether inference/routing is a primary cause.

Next-stage decision rules: if a concrete inference-profile bug exists, patch locally and rerun short CIFAR-100; if routing is ambiguous but not clearly buggy, document as weakness and proceed to Stage 6; if earlier classifier or adapter evidence remains stronger, recommend returning to Stage 3 or 4; if validation fails, stage is incomplete.

## Stage 6 - Final Synthesis
Objective: convert staged evidence into a ranked, paper-faithful next-action recommendation.

Required outputs: a synthesis separating implementation bugs, current implementation weaknesses, and unconfirmed hypotheses; ranked next patches with expected effect, code risk, and distance from NH-LoRA paper design.

Acceptance criteria: all prior completed stage summaries are read; CIFAR-100 rerun logs are referenced by path; conclusions distinguish evidence from speculation; `outputs/dev/stage_6_summary.md` includes the recommended next patch and experiments.

Next-stage decision rules: if a bug is confirmed, fix it before adding new algorithmic behavior; if only weakness is confirmed, propose the smallest optional config-gated improvement; if evidence is inconclusive, add missing diagnostics rather than changing methodology.

## Validation Defaults
- Local validation after code changes: `python -m compileall src tests`.
- Unit/smoke validation after continual-path changes: `python -m unittest tests.test_paper_alignment_units tests.test_synthetic_continual_smoke`.
- Add `tests.test_config_summary` when config schema/defaults change.
- Add `tests.test_checkpoint_resume` only if artifact/checkpoint behavior is touched.
- CIFAR-100 short rerun target: at least task 3, with log path recorded in the stage summary.

## What Changed
- Added explicit objective, required outputs, acceptance criteria, and next-stage decision rules for every stage.
- Added hard execution rules: one thread, one patch, one stage, one primary hypothesis.
- Added Git checkpoint protocol and mandatory per-stage summary artifact.
- Made Stage 1 instrumentation-only the required first patch after planning.
- Preserved NH-LoRA research constraints and default-safe config behavior.

## Why This Revision Is Better
- It prevents scope creep by making every stage complete or incomplete based on validation and artifacts.
- It gives coding agents operational branch rules instead of vague investigate-later guidance.
- It keeps algorithmic changes conditional on evidence, which protects paper faithfulness.
- It makes every patch auditable through Git checkpoints, log paths, summaries, and validation commands.
