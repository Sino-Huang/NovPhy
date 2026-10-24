## Decision P-fid (2026-10-24): frozen Tier-E band for B1, CLEVRER mIoU 49.42 ± 0.5

**Trigger.** The B1 Tier-E G-fid run, one evaluation run of the released weights, is about to be enqueued on the night-window scheduler as job `129-gfid-tier-e` (GPU 1). The band is frozen before that run.

**Target.** CLEVRER video-prediction mIoU **49.42**: README of pairlab/SlotFormer @ 10a6d9b, update 2023-09-20.
- The README says "The updated mIoU of SlotFormer on CLEVRER is **49.42** (using the provided pre-trained weight)".
- That number follows the mIoU fix in PR #7 (commit fb1c39b: union = |A| + |B| − |A ∩ B| in `vp_utils.hungarian_miou`) and was posted in PR #8.
- The paper's pre-fix Table 2 value is not the target.

**Metric.**
- The official `slotformer/video_prediction/test_vp.py` prints `miou: {v.mean():.4f}`; ours is that value × 100.
- `v` holds the 42 per-step means (6 burn-in + 42 rollout frames, `frame_offset` 2, 64 × 64) of the per-clip Hungarian-matched IoU between the ground-truth foreground masks (CLEVRER `derender_proposals`) and SlotFormer's post-processed slot masks.
- The clips are the CLEVRER validation videos under the official `filter_enter` rule, with burn-in slots from the released `clevrer_slots.pkl`.
- Weights: `slotformer_clevrer_params/model_80.pth` and the frozen SAVi decoder `stosavi_clevrer_params/model_12.pth`.

**Band:** [48.92, 49.92] (#127 / #129: 49.42 ± 0.5), from one evaluation run.
- Pass iff ours lies inside the band.
- A fail follows P-fid: at most two logged repair attempts, each fixing a diagnosed port defect, never the band; then the arm is dropped with `fidelity_not_reproduced`.
- The optional Tier T (PHYRE planning AUCCESS 82.0 ± 1.1) is not run in Phase 0.

`fidelity/band.json` sha256: **`02ff5b00e06530e63927b3ff26f5a13c6c3213695c3eba40f33f1f4a437868de`**. `scripts/b127_slotformer_fidelity.py` refuses to run unless this hash is in `decisions.md`.

**Job.** `129-gfid-tier-e`, GPU 1, non-resumable, `est_minutes` 60.
- The CPU dry run measured about 1.1 s per clip of single-process data loading. There are 2 314 clips; the loader uses 8 workers.
- Loading the dataset and its annotations took 106 s, and the slot pickle is 13.6 GB.
- On SIGTERM the run exits 75 and records no number.
- Command: `/srv/home/shua0171/miniconda3/envs/b127-slotformer/bin/python -u scripts/b127_slotformer_fidelity.py`, which writes `fidelity/gfid.json`.
