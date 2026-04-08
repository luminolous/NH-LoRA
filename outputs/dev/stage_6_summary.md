# Stage 6 Summary - Evidence Synthesis And Next-Step Recommendation

## Executive Decision
Stage 5 follow-up is closed as a success. The NH-LoRA forgetting diagnosis now has two confirmed implementation-level fixes, one tested hypothesis that was not supported, and one residual open weakness that should be treated as a planner/policy audit target rather than as another routing or adapter bug.

No further code behavior change is authorized in Stage 6; this stage is documentation and recommendation only.

The next recommended stage is a planner shared-only bias audit. Stage 3 classifier alignment remains deferred. Stage 4 and Stage 5 should not be reopened unless new evidence contradicts the conclusions recorded here.

## Evidence Inputs
- Stage 1 instrumentation baseline:
  - `outputs/dev/stage_1_summary.md`
  - prior CIFAR-100 log reviewed during Stage 1 from `C:\Users\Syauqi Nabil\Downloads\CIFAR100_LOG_TRAINING.txt`
- Stage 4 adapter fix evidence:
  - `outputs/dev/stage_4_summary.md`
  - rerun after `base_output=attended`: `C:\Users\Syauqi Nabil\Downloads\reun-cifar100-update-stage4.txt`
  - key signal: `AdapterDelta` and `FinalFeatureDiff` became nonzero; shared and slot grads became active on Task 2
- Stage 2 rejection evidence:
  - `outputs/dev/stage_2_summary.md`
  - LR-scale reruns:
    - `C:\Users\Syauqi Nabil\Downloads\stage2-rerun-cifar100-lrscale075.txt`
    - `C:\Users\Syauqi Nabil\Downloads\stage2-rerun-cifar100-lrscale05.txt`
    - `C:\Users\Syauqi Nabil\Downloads\stage2-rerun-cifar100-lrscale025.txt`
  - key signal: lower classifier LR degraded both retention and current-task accuracy instead of helping
- Stage 5 mismatch discovery evidence:
  - `outputs/dev/stage_5_summary.md`
  - baseline routing audit rerun: `C:\Users\Syauqi Nabil\Downloads\log-rerun-cifar100-stage5.txt`
  - key signal: same-checkpoint same-input `RouteModeCompare` showed `mismatch_applied_empty_profile_nonempty=True` while raw planner still chose shared-only actions
- Stage 5 mismatch fix evidence:
  - `outputs/dev/stage_5_followup_summary.md`
  - updated rerun after `Align inference profile shared-only semantics`: `C:\Users\Syauqi Nabil\Downloads\log-rerun-cifar100-stage5-updated.txt`
  - key signal: `shared_only_train=True`, `shared_only_profile=True`, and `mismatch_applied_empty_profile_nonempty=False` for the affected Task 3/4 layers

## Confirmed Implementation Bugs Fixed
1. Stage 4 adapter forward inert bug
   - Problem:
     active NH-LoRA adapters were not reliably affecting forward output because the attended path around `out_proj` could be dropped.
   - Evidence before fix:
     Stage 1 diagnostics reported zero adapter deltas and zero final feature drift despite nontrivial forgetting symptoms.
   - Fix:
     preserve correct forward semantics around the `out_proj` path so zero-LoRA parity is maintained while active LoRA deltas can modify the block output.
   - Evidence after fix:
     in `reun-cifar100-update-stage4.txt`, Task 2 reported nonzero `AdapterDelta`, nonzero `FinalFeatureDiff`, and nonzero shared/slot grads. Task summaries improved from total collapse to partial retention.

2. Stage 5 applied-train vs inference-profile mismatch
   - Problem:
     for `reuse_shared` and `freeze_old_strong_retention`, training-time applied plans were shared-only with empty candidates, but `build_inference_profile()` still reactivated retained slots during evaluation-related paths.
   - Evidence before fix:
     in `log-rerun-cifar100-stage5.txt`, Task 3/4 showed `mismatch_applied_empty_profile_nonempty=True` even when raw planner and applied route were clearly shared-only.
   - Fix:
     align `build_inference_profile()` with declared repo policy so those actions produce shared-only inference while preserving retained/live slot state metadata.
   - Evidence after fix:
     in `log-rerun-cifar100-stage5-updated.txt`, Task 3/4 now show `shared_only_train=True`, `shared_only_profile=True`, `mismatch_applied_empty_profile_nonempty=False`, `mismatch_train_eval_applied=False`, and `mismatch_eval_profile=False`.

## Tested Hypotheses Not Supported
1. Simple classifier head-dominance fix via lower classifier LR
   - This was tested in Stage 2 with `classifier_lr_scale=0.75`, `0.5`, and `0.25`.
   - Result:
     the ablation consistently worsened both retention and current-task accuracy.
   - Evidence:
     - baseline patched run before Stage 2 rejection: Task 3 `per_task_acc=[0.1340, 0.3440, 0.8780]`, forgetting `0.7135`
     - `0.75`: Task 3 `per_task_acc=[0.1170, 0.3140, 0.8640]`, forgetting `0.7350`
     - `0.50`: Task 3 `per_task_acc=[0.1000, 0.2910, 0.8380]`, forgetting `0.7520`
     - `0.25`: Task 3 `per_task_acc=[0.0650, 0.2540, 0.8030]`, forgetting `0.7785`
   - Conclusion:
     simple classifier LR reduction is not supported as the next patch candidate.

2. Logger or hook mismatch as the main explanation of Task 3+ collapse
   - This was plausible before Stage 5 because routing and inference behavior disagreed.
   - After the Stage 5 follow-up patch, the same-checkpoint same-input comparison now agrees across applied route and inference profile for shared-only actions.
   - Conclusion:
     logging/hook mismatch is no longer a credible main explanation for the remaining forgetting behavior.

## Residual Open Weaknesses
1. Persistent shared-only planner regime from Task 3 onward
   - In the updated Stage 5 rerun, raw planner repeatedly chose `freeze_old_strong_retention` on layers 6, 7, 8, 10, and 11, and `reuse_shared` on layer 9 for Task 3, Task 4, and Task 5.
   - The relevant decision boundary in `planner.decide_action()` is still:
     - `reuse_shared` when novelty `< tau_novelty` and conflict `< tau_conflict`
     - `freeze_old_strong_retention` when novelty `< tau_novelty` and conflict `>= tau_conflict`
   - This means the remaining collapse is now best framed as a planner/shared-only bias question, not a routing-profile inconsistency.

2. Shared path stays active while slot/router/planner participation collapses
   - In the updated rerun:
     - Task 3 shared grad stayed nonzero while slot/router/planner grads were zero through the debug epochs.
     - Task 4 showed the same pattern.
   - Final feature drift remained nonzero, so the model is not globally inert; the issue is that adaptation is happening through shared-only behavior, not through task-slot routing.

3. Old/new competition remains asymmetric
   - In the updated rerun, old/new logit margin still strongly favored new classes on Task 3 and Task 4 even while old-classifier drift stayed small.
   - This indicates the remaining weakness is not simply old-classifier row corruption.

4. Residual classification
   - The remaining Task 3+ behavior is now classified as a policy/planner weakness or paper-interpretation issue.
   - It is not currently supported as a Stage 4 adapter bug, a Stage 5 inference-profile bug, or a simple classifier LR imbalance.

## Ranked Next Actions
1. Planner shared-only bias audit
   - Inspect `planner.decide_action()` thresholds against the logged `novelty` and `conflict` values from Task 3+.
   - Trace why Task 3 onward sits on the shared-only side of the thresholds across most selected layers.
   - Compare that threshold behavior against NH-LoRA paper intent before proposing any code change.

2. Keep Stage 3 classifier alignment deferred
   - Classifier alignment is not the right next step while the dominant remaining evidence still points to a planner/shared-only regime.
   - Revisit Stage 3 only if a later planner audit shows routing/planner behavior is healthy but old/new class competition still remains the dominant bottleneck.

3. Do not reopen Stage 4 or Stage 5 without contradictory evidence
   - Stage 4 should remain closed unless adapter deltas or CE gradients become inert again in a future run.
   - Stage 5 should remain closed unless a new same-checkpoint mismatch appears between applied route and inference profile.

## Risks And What Not To Do Next
- Do not interpret the residual shared-only collapse as proof that the overall NH-LoRA design is wrong.
- Do not jump directly to classifier alignment, loss redesign, or planner patching based on the current evidence alone.
- Before any planner change is authorized, audit thresholding and the interpretation of `novelty` / `conflict` against the intent of the NH-LoRA paper.
- The Stage 5 follow-up patch affected every path that uses `build_inference_profile()`, including warm-up sensing, teacher profile construction, and evaluation. Any CIFAR-100 improvement after that patch should therefore be interpreted as a broader mechanistic correction, not as an eval-only artifact.

## Recommended Next Stage
The next stage to plan is a **planner shared-only bias audit**.

That audit should be framed narrowly:
- inspect threshold-triggered action selection,
- compare Task 2 vs Task 3+ planner signals layer by layer,
- determine whether the current novelty/conflict interpretation is faithful to paper intent or overly conservative,
- avoid proposing any planner patch until the audit separates policy weakness from intended behavior.
