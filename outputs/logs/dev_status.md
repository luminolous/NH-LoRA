# NH-LoRA Development Status

## Latest updates

- Completed the official paper-alignment implementation pass focused on correctness first, then robustness and documentation.
- Rebuilt the training engine around the explicit paper flow:
  - warm-up sensing
  - raw planner
  - pure `materialize_action`
  - explicit `apply_structure_changes`
  - task training
  - CHU consolidation
  - history append
  - post-consolidation inference profile
- Added save/load checkpoint support with resume state, RNG state, sampler state, and current task context.
- Updated tests so critical NH-LoRA behaviors are validated directly.

## Files changed in this pass

- `README.md`
- `docs/paper_alignment.md`
- `src/engine/train.py`
- `src/engine/trainer.py`
- `src/models/chu.py`
- `src/models/lora.py`
- `src/models/losses.py`
- `src/models/nh_lora.py`
- `tests/test_paper_alignment_units.py`
- `tests/test_synthetic_continual_smoke.py`
- `tests/test_checkpoint_resume.py`
- `outputs/logs/dev_status.md`

## Gap analysis

### Already close before this pass

- Frozen backbone structure
- shared-versus-slot NH-LoRA decomposition
- basic slot bank and rank-mask idea
- incremental cosine head
- summary-only history bank direction
- bootstrap branch skeleton

### Corrected in this pass

- Planner input now explicitly uses `[z_t; h_t_tilde; |z_t-h_t_tilde|; z_t ⊙ h_t_tilde; e_l]`.
- History-aware aggregation is explicit and separated from structural materialization.
- `materialize_action` is pure; structure mutation happens only in `apply_structure_changes`.
- `reuse_shared` is now truly shared-only by default.
- `open_new_slot` falls back to `expand_rank_existing_slot` with the official compatibility rule when slot capacity is full.
- `freeze_old_strong_retention` now has real effect:
  - old slots are frozen
  - inference candidates are emptied
  - the affected layer is treated as shared-dominant
- Warm-up sensing now uses a temporary prototype-imprinted auxiliary head and produces task-state statistics used by TSE.
- Slot compatibility is now defined consistently as cosine similarity between normalized task embedding and normalized slot key.
- Slot keys now live in task-embedding space, matching the official compatibility definition.
- `L_orth` is now based on active low-rank factors.
- `L_rank` is now based on realized active-rank masks, not planner scores.
- `L_grow` is now based on actual slot opening events.
- `L_route` now uses `KL(mean routing distribution || uniform)`.
- Eval/inference no longer uses the old shortcut `all live slots + shared_gate=1.0`.
- Checkpoint/resume now restores model/planner/TSE/history/inference profile plus trainer and RNG state.

### Still approximation

- CHU remains heuristic by design, although it now uses explicit merge/prune/keep/freeze decisions.
- `L_rank` is paper-faithful as realized structural cost, but can become near-constant during a task once structure is fixed.
- Full ViT-B/16-IN21K validation still depends on the SSH runtime with `timm`.

### Not fully validated here

- Real dataset layouts for CUB-200-2011, ImageNet-R, and OmniBenchmark
- Full benchmark throughput and long-run training
- Real multi-worker or CUDA-heavy resume determinism

## Completed

- PASS 1:
  - planner input correctness
  - history-aware aggregation
  - pure materialization
  - explicit structure mutation
  - action semantics
  - bootstrap task-1 behavior
  - paper-target losses
  - inference policy replacement
- PASS 2:
  - checkpoint/resume wiring
  - scheduler activation
  - efficiency metrics in trainer output
  - README update
  - paper alignment documentation
  - checkpoint/resume test coverage

## Pending

- SSH validation with the real `timm` backbone path
- real benchmark execution
- benchmark-scale calibration only if benchmark evidence later shows it is needed

## Assumptions

- The design paper remains the primary source of truth.
- Heuristic CHU is acceptable because the paper allows it.
- The chosen inference policy is the most conservative operationalization under paper ambiguity.
- `L_rank` remains realized-mask based even when its optimizer signal is weak, because that is more paper-faithful than reverting to planner-score surrogates.

## Risks to verify later on SSH

- ViT-B/16-IN21K runtime path with `timm`
- dataset path/layout details on the real server
- long continual runs and checkpoint stress behavior
- exact determinism limits under real CUDA and multi-worker dataloading

## Validation run

- `python -m compileall src tests`
- `python -m unittest tests.test_config_summary tests.test_paper_alignment_units tests.test_synthetic_continual_smoke tests.test_checkpoint_resume`
