## Decision P-fid (2026-10-28): G-fid not reproduced after one logged repair → T7 DeepPHY-VLM is dropped, typed `fidelity_not_reproduced`; listed as not run

**Trigger.** Repair attempt 1 (issuecomment-6450386391) also misses the frozen band [7.47, 11.77] % (`band.json` sha256 `ad5bfb65…3c30`, issuecomment-6437344622).

| attempt | env | success within 10 attempts | within / cross template | attempt-1 rate | invalid inputs | empty / malformed responses |
| --- | --- | --- | --- | --- | --- | --- |
| 0 (`139-gfid`) | `b127-deepphy` | **6.0 %** | 6.8 / 5.2 % | 0.5 % | 3 190 / 9 723 | 0 / 0 |
| 1 (`139-gfid-r1`) | `b127-deepphy-gfid` (DeepPHY's pinned matplotlib 3.9.4 / pillow 11.3.0) | **6.1 %** | 6.4 / 5.8 % | 0.4 % | 3 159 / 9 733 | 0 / 0 |

For reference, the paper (Table 6, Qwen-7B VLA) reports 0.60 % at attempt 1 and **9.62 %** within 10 attempts; its random MOCK agent reaches 7.64 %.

Records: `fidelity/gfid-attempt0.json` (sha256 `acdf3fa2…db72`) and `fidelity/gfid.json` (repair 1, sha256 `c2aea4d1…91b4`).

**Diagnosis status.**
- The two attempts agree closely (6.0 % and 6.1 %), so the matplotlib glyph rendering repaired in attempt 1 does not explain the gap.
- The attempt-1 rate matches the paper. The gap builds over the later attempts, where our near-greedy agent mostly sweeps neighbouring cells at radius 3.
- Everything checked in the repair decision still holds:
  - PHYRE outcomes: all 9 723 actions are status-identical under DeepPHY's pinned phyre 0.2.2 and numpy 1.26.4.
  - Decoding: DeepPHY's pinned pipeline is effectively greedy, for them as for us.
  - Images: pixel-identical to the pinned stack.
  - No OOM, call failures or format errors.
- Remaining differences we cannot remove: the GPU (RTX 5090) and torch's CUDA build (cu128).
- **No further port or environment defect is diagnosed.** P-fid allows a second repair only for a diagnosed defect, so none is run.

**Choice.** Under P-fid, T7 is dropped with typed reason **`fidelity_not_reproduced`**. The orchestrator writes `not-run.json` `{"typed_reason": "fidelity_not_reproduced", "decision": <this comment>}`. Then `scripts/run_issue_139_deepphy.py --publish` writes the not-run child summary: `extra.pending_arms` {T7: `fidelity_not_reproduced`}, G-port pass, G-fid fail with both attempts' records, and the projection. The program summary lists T7 as not run.
- No NovPhy policy or evaluation data was scored, no selection ran, and the evaluation split was never read.
- The queued `139-selection`, `139-selro` and `139-smoke` stay blocked and should be removed. Phase B is not enqueued.
- GPU used: 3.23 GPU-h in total (G-fid 3.20 GPU-h over both attempts, including the 1.41 h attempt-1 run killed at 07:00, plus the 0.03 GPU-h timing probe), against the funded 8.8 GPU-h. The remaining ≈ 5.6 GPU-h return to the program remainder.

**Reason.** The frozen rule permits at most two repairs and only for diagnosed defects; the band is never changed. One defect was diagnosed and fixed, the miss reproduced, and no other defect was found. Disclosed: the band is binomial and does not model run-to-run variation from hardware numerics under near-greedy decoding. The paper's own 3-run spread is not published.
