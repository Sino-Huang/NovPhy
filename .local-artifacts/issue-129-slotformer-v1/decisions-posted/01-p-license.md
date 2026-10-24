## Decision P-license (2026-10-24): vendor SlotFormer's slot Transformer, loss and LR schedule (MIT)

**Trigger.** Phase 0 vendoring for B1 (#127 rule P-license).

**License check.**
- Upstream [pairlab/SlotFormer](https://github.com/pairlab/SlotFormer) at commit `10a6d9b7bc05ef2397ce2586da3fc0212f8c47aa` (2023-09-20, the merge of PR #8). Its `LICENSE` is MIT, "Copyright (c) 2022 Ziyi Wu", sha256 `1ddc76632bc1d6daecf1240e457e69d424f6251e8667b3d8fb69a7b98fdc2317`.
- SlotFormer's model base class and LR scheduler come from the author's toolbox [Wuziyi616/nerv](https://github.com/Wuziyi616/nerv) at tag `v0.1.0` (commit `e83ac66c6ce30e1ca3d0a287df9d3699ed9ec499`), which SlotFormer's `docs/install.md` pins. It is also MIT, "Copyright (c) 2022 Ziyi Wu", with a byte-identical `LICENSE`.
- Released weights (`pretrained.zip`, Google Drive folder linked from `docs/benchmark.md`) are used only for the Tier-E fidelity check. They are never redistributed or committed, and they stay under `data/external/slotformer/` (gitignored).

**What is vendored** into `world_model/baselines/slotformer/vendor/`: verbatim copies, sha256-pinned in `VENDOR.json` (schema `issue_127_vendor_v1`).
- `slotformer/video_prediction/models/slotformer.py`: `SlotRollouter`, the sine temporal P.E. and `SlotFormer`, including the official loss `calc_train_loss` (per-slot MSE, the `vid_len` truncation and the RPIN temporal weighting).
- `slotformer/video_prediction/models/single_step_slotformer.py`: `SingleStepSlotRollouter` / `SingleStepSlotFormer`, the PHYRE burn-in-1 model with iterative overlapping and a `cond_len` window.
- nerv `nerv/training/model.py` (`BaseModel`) and `nerv/training/lr.py` (`CosineAnnealingWarmupRestarts`, the official scheduler).
- Both `LICENSE` files.

**Edits.**
- The `nerv` import in `slotformer.py` becomes package-relative.
- `slotformer.py`'s `from slotformer.base_slots.models import StoSAVi` is replaced by `StoSAVi = None`. StoSAVi is used only by `SlotFormer._build_decoder` / `decode`, i.e. the frozen SAVi pixel decoder for the optional image loss. The NovPhy adapter has no image decoder: E99 slots are the state. The adapter overrides both methods, so vendoring SAVi plus its nerv module tree would add unused code.
- Every edit is listed in `VENDOR.json` with its reason.

**Not vendored.**
- Optimizer construction lives in `SlotBaseMethod._configure_optimizers` / nerv `BaseMethod`, which pull in the wandb trainer. The adapter re-states it (Adam on the trainable parameters, weight decay 0, no clipping; `CosineAnnealingWarmupRestarts(total, max_lr=lr, min_lr=lr/100, warmup_steps=0.05 total)`, stepped per step). G-port holds it to the official methods, which run in the reference env.
- SAVi / evaluation code: the Tier-E check runs the official `test_vp.py` from the pinned clone in its own env (`b127-slotformer`). Nothing from it is imported at NovPhy run time.

**Options considered.**
- (a) Vendor SAVi and the nerv model tree verbatim: more unused code, no gain in fidelity.
- (b) Re-implement the slot Transformer from the paper: rejected. The license permits vendoring, and an official copy is the faithful choice.
- **Choice: (c)**, the minimal verbatim vendor above.
