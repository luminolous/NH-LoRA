# NH-LoRA Development Status

## Latest updates
- Completed a paper-faithful alignment pass against the latest `NH-LoRA Design Paper [ID].pdf` using local PDF text extraction without installing new packages.
- Tightened the implementation around the paper's information flow, not just the module list:
  - persistent prototype-imprinted warm-up head,
  - pooled raw-vector TSE,
  - richer history-bank summaries,
  - history-aware planner q/k/v aggregation,
  - explicit separation between raw planner signals and `materialize_action`,
  - additive rank expansion with max-slot fallback,
  - normalized CHU usage and exponential stability,
  - shared-core lower learning-rate path.
- Extended the paper-alignment tests so task 2 now proves that history attention, materialized candidates, classifier imprinting, CHU, and history growth are all wired together.
- Re-ran lightweight validation after each patch; all current alignment checks are green.

## Files touched
- `configs/base.yaml`
- `src/engine/trainer.py`
- `src/models/task_state.py`
- `src/models/planner.py`
- `src/models/lora.py`
- `src/models/losses.py`
- `src/models/chu.py`
- `src/models/classifier.py`
- `src/models/nh_lora.py`
- `tests/test_paper_alignment_units.py`
- `tests/test_synthetic_continual_smoke.py`
- `outputs/logs/dev_status.md`

## Gap analysis
### Already paper-faithful
- Frozen ViT backbone remains fully frozen, with configurable selected blocks and default `q_proj`/`v_proj` insertion plus optional MLP path support.
- Shared Core LoRA, expandable task slots, fixed-capacity rank-mask behavior, and sparse instance routing remain in the repo and match the intended NH-LoRA decomposition.
- The training loop still follows the paper phase order: warm-up sensing, planning, structure expansion, task training, consolidation, and task-summary save.
- Bootstrap task-1 behavior remains paper-consistent: no teacher, no history-aware similarity, no KD, no feature-retention loss, and deterministic single-slot routing.
- The incremental cosine classifier head still expands per task and stays aligned with the paper's frozen-backbone setting.

### Corrected in this pass
- Replaced the old ad-hoc warm-up behavior with a persistent auxiliary warm-up head initialized by prototype-based imprinting and used only to form warm-up statistics.
- Reworked TSE to encode a pooled raw task vector `[pool(mean), pool(var), grad, similarity, entropy]` through a normalized MLP, which is closer to the paper formula than the previous separated-projection shortcut.
- Expanded history entries to store pooled feature mean, pooled feature variance, gradient sketch, usage summary, active-rank summary, entropy summary, similarity anchor, task embedding, and a history summary vector.
- Changed similarity-to-history from mean-style matching to max-style matching over summary vectors and anchors, which is closer to the paper's similarity intent.
- Replaced mean history aggregation in the planner with history-aware q/k/v attention and formed planner inputs as `[z_t, h_t, z_t * h_t, e_l]` per layer.
- Preserved raw planner state as a first-class object: novelty, conflict, rank score, rank budget, consolidation score, shared gate, history attention, history context, and planner representation now exist separately from materialized structural actions.
- Made `materialize_action` closer to the paper by using additive rank expansion, explicit candidate-slot sets, and `open_new_slot -> expand_rank_existing_slot` fallback when slot capacity is full.
- Wired `shared_lr_scale` into optimizer param groups so shared memory now actually behaves as the slower-plasticity path described in the paper.
- Normalized CHU usage by task sample count, switched slot stability to an exponential drift form, and based redundancy on adapter update signatures instead of slot-key similarity.
- Tightened `L_rank` and `L_grow` so they use planner raw signals and active-rank surrogates that are closer to the paper's intent than the earlier metadata-only constants.
- Tightened the two-task synthetic continual smoke test to confirm task-2 history attention, planner/action separation, classifier imprinting, CHU invocation, and history-bank growth.

### Still approximation
- The local fallback path still uses the internal toy ViT for smoke checks; exact ViT-B/16-IN21K behavior still depends on the `timm` path on the SSH machine.
- Planner action selection still uses explicit thresholds over learned novelty/conflict/consolidation signals. This is paper-consistent as a heuristic materialization rule, but not benchmark-tuned yet.
- Slot orthogonality is enforced in adapter update-space signatures, not the exact factor-only form `A_{l;s}^T A_{l;u}`.
- CHU is still heuristic by design, although its inputs now follow the paper much more closely.
- Feature retention is layer-aware over configured block outputs, but not every possible internal sub-activation is retained.

### Not yet safe for real benchmark runs
- The `timm` ViT-B/16-IN21K path has not been executed locally in this environment.
- Real CIFAR-100, CUB-200-2011, ImageNet-R, and OmniBenchmark layouts have not been re-validated in this paper-alignment pass.
- No full benchmark run, long continual sequence, or real checkpoint/runtime stress test was executed here by design.

## Completed
- Paper-vs-code gap analysis completed and reflected in the implementation.
- Warm-up sensing, TSE, history bank, planner, materialization, losses, and CHU were patched toward paper-faithful behavior.
- Lightweight paper-alignment validation passed:
  - `python -m compileall src tests`
  - `python -m unittest tests.test_config_summary tests.test_paper_alignment_units tests.test_synthetic_continual_smoke`

## Pending
- Real benchmark execution on the SSH machine.
- Validation of the true `timm`-backed ViT-B/16-IN21K runtime path.
- Validation of real dataset layouts for CUB-200-2011, ImageNet-R, and OmniBenchmark.
- Real-task retuning of planner and CHU thresholds only if benchmark evidence later shows they are necessary.

## Assumptions
- Local development must avoid heavy compute, package installation, and full benchmark runs.
- Fixed `r_max` rank masking is always on in this repo because NH-LoRA is the only target method here; `use_rank_mask` remains informational rather than a toggle.
- History bank stores summary statistics only; no old-task raw data is retained.
- Growth regularization uses planner open-slot pressure (`novelty * conflict`) as the soft surrogate for the paper's new-slot indicator.
- Shared-usage history summary is represented by per-block planner shared-gate averages.
- TSE pooling dimension follows `planner.history_pool_dim` so the current task-state and history-bank summaries stay shape-consistent.

## Risks to verify later on SSH
- Full `timm`-backed ViT runtime path.
- Real dataset directory layouts for CUB-200-2011, ImageNet-R, and OmniBenchmark.
- Full training/evaluation throughput and long-run checkpoint behavior.
- Whether the current heuristic thresholds need mild retuning once real benchmark evidence is available.
