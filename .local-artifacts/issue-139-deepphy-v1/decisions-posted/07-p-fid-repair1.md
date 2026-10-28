## Decision P-fid (2026-10-28): G-fid attempt 0 misses the band (6.0 % vs [7.47, 11.77] %); one diagnosed environment deviation in the VLM's input (matplotlib glyph rendering) → logged repair attempt 1 in DeepPHY's pinned rendering stack; band unchanged

**Trigger.** Attempt 0 (`139-gfid`, the official runner, 1.61 GPU-h) scored **6.0 %** success within 10 attempts on the 1 000 PHYRE tasks. That is outside the band [7.47, 11.77] % frozen in issuecomment-6437344622 (`band.json` sha256 `ad5bfb65…3c30`), so the gate failed. The record is kept as `fidelity/gfid-attempt0.json`.
- Our curve by attempt: 0.5 / 1.0 / 2.0 / 2.2 / 3.3 / 3.8 / 4.6 / 4.9 / 5.4 / 6.0 %.
- Paper Table 6: 0.60 / 2.10 / 4.10 / 4.30 / 5.50 / 6.40 / 7.16 / 8.16 / 8.82 / 9.62 %.
- No empty responses (no OOM or call errors) and no invalid-format answers. 3 190 of 9 723 attempts were `INVALID_INPUT`.

**Diagnosis (CPU; official code unedited).**
1. **Environment outcomes: no defect.** Every one of the 9 723 parsed actions was replayed through PHYRE in two environments:
   - DeepPHY's pinned environment (py3.9, the released `phyre==0.2.2` wheel, numpy 1.26.4; env `b127-phyre022`);
   - our build (`b127-deepphy`).
   
   Both give exactly the recorded status for every action (NOT_SOLVED 6 473, INVALID_INPUT 3 190, SOLVED 60), with the same task order. `INVALID_INPUT` is therefore a legitimate agent outcome, the official `simulate_action` verdict for a ball that overlaps an object or leaves the scene. The official runner counts it as a failed attempt, so it belongs to the paper's protocol.
   
   Consequences:
   - Neither the PHYRE build nor the `np.float` shim changes any outcome. The shim is needed in DeepPHY's own pins too: `phyre/vis.py` of 0.2.2 uses `np.float`, which numpy 1.26.4 lacks, so the official runner cannot run unshimmed there.
   - The MOCK check (6.20 % against the paper's 7.64 %) is therefore not an environment offset. It is sampling noise (≈ 1.5 SE).
2. **Decoding: no defect.** In transformers 4.49, 4.50, 4.51, 4.52.4 (DeepPHY's pin), 4.53 and 4.55, the image-text-to-text pipeline does not forward `do_sample` / `temperature` to `generate`. DeepPHY's runs therefore also decoded with the checkpoint's generation config, which is effectively greedy, as ours did (addendum issuecomment-6437519414).
3. **VLM input images: one deviation found.** We rendered the official initial scenes and 8 × 8 gridded images of 20 tasks (every 50th of each setup) twice: in DeepPHY's pinned stack (matplotlib 3.9.4, pillow 11.3.0, numpy 1.26.4) and in ours (matplotlib 3.11.2, pillow 12.3.0).
   - Initial scenes, and therefore the simulation keyframes, which are rendered by the same `plt.imsave`: **pixel-identical**.
   - Gridded images: same size, same lines and scene, but **the cell-number glyphs are rasterised differently** (up to 7 972 px per image, max |Δ| 255; confined to the 64 labels).
   
   Under effectively greedy decoding, these inputs determine the agent's whole 10-attempt trajectory. This is a deviation of our environment from DeepPHY's pins in what the VLM sees.
4. Remaining unavoidable differences: the GPU (RTX 5090) and torch's CUDA build (cu128 instead of cu126, which has no sm_120 kernels).

**Options.** (a) Drop now with `fidelity_not_reproduced`. (b) One logged repair fixing the diagnosed input deviation. (c) Change the band, decoding or protocol: forbidden.

**Choice.** (b), **repair attempt 1** of at most 2.
- Run the same official runner, arguments and seeding (`scripts/b127_deepphy_fidelity.py --run --repair 1`, output `fidelity/runs-r1/`) in env **`b127-deepphy-gfid`**. This is `b127-deepphy` with DeepPHY's pinned **matplotlib 3.9.4 and pillow 11.3.0**. Verified on CPU: its initial and gridded renders of the 20 tasks are **pixel-identical** to the pinned-stack renders (40 / 40 images).
- The T7 scoring env `b127-deepphy` is unchanged.
- `--finish --repair 1` writes `fidelity/gfid.json`, with `repair_attempt` 1 and the sha256 of `gfid-attempt0.json`. The band is unchanged.
- A miss again, with no further diagnosed defect, means T7 is dropped with `fidelity_not_reproduced`.
- Package lists: `envs/b127-deepphy-gfid.txt`, `envs/b127-phyre022.txt`. Cost about 1.6 GPU-h, charged to #139.

**Reason.** The repair removes the only deviation found between our G-fid environment and DeepPHY's in the VLM's inputs, and changes nothing else. Outcomes and decoding were shown to be equivalent. Disclosed: whether the glyph difference caused the miss is not established. Under near-greedy decoding, any input perturbation reshuffles trajectories, so attempt 1 also redraws that variation, which the binomial band does not model.
