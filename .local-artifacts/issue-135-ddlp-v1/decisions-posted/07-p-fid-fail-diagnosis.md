## Decision P-fid (2026-10-25): T2 G-fid Tier E is outside the frozen band — two pre-registered diagnostics before any repair

**Result (attempt 1, frozen band issuecomment-6396727991, `band.json` sha256 `06cbd43b…4581`).** The official `eval/eval_gen_metrics.py -d phyre -c 10 --horizon 100 -b 2 --cpu` on the released `ddlp-phyre/phyre_ddlp.pth` and the released PHYRE test split (400 episodes × 90 predicted frames), torch 2.13 CPU, exit 0, 5 261 s CPU, 0 GPU-s:

| metric | ours | paper (Table 3, DDLP) | band | paper G-SWM row |
| --- | --- | --- | --- | --- |
| **LPIPS (gating)** | **0.0745** (SD 0.053) | 0.055 (SD 0.04) | [0.0505, 0.0595] | 0.078 |
| PSNR (descriptive) | 24.55 (SD 5.13) | 26.98 (SD 5.3) | [26.445, 27.515] | 24.64 |
| SSIM (descriptive) | 0.935 (SD 0.045) | 0.95 (SD 0.04) | [0.941, 0.959] | 0.93 |

**G-fid = FAIL** (`fidelity/gfid.json` sha256 `785fa3ef…ed46`, log sha256 `e1cb633a…b4c9`). The band is not changed.

**Rule.** P-fid allows at most two logged repair attempts, each fixing a *diagnosed* defect of our harness, never the band; otherwise T2 is dropped (`fidelity_not_reproduced`). Our harness differs from the authors' in exactly two respects that could matter: (1) the software stack (P-route: torch 2.13 / torchvision 0.28 / numpy 2.4 vs the repository's pins torch 2.0.1 / torchvision 0.15.2 / numpy 1.24.3; piqa 1.3.1 on both), (2) the batch size (2 vs the script default 10). The data, checkpoint, hparams (except the data root), script and command are the authors'.

**Decision: two diagnostics (CPU, no GPU), defined before they run; nothing else is varied.**
- **D1 (stack).** A new env `b127-ddlp-orig` with the repository's `requirements.txt` pins for every package the evaluation imports (torch 2.0.1 CPU, torchvision 0.15.2, numpy 1.24.3, piqa 1.3.1, einops 0.6.1, matplotlib 3.7.1, imageio 2.28.1, scikit-image 0.19.2; opencv headless where the pinned 3.4 wheel is unavailable); the unchanged official script on the **first 50 test episodes** (the official dataset wrapped in `Subset(range(50))` by the harness; batch 2), versus the same 50 episodes in `b127-ddlp`.
- **D2 (batch).** The same 50 episodes in `b127-ddlp` at batch 10 (the script default) versus batch 2.
- **Defect criterion:** a diagnostic **differs** iff its LPIPS differs from the `b127-ddlp` batch-2 value by more than 1e-4 relative. A differing diagnostic is a diagnosed harness defect → repair attempt 1 = the full frozen protocol (400 episodes) without that difference, judged against the unchanged band (logged as attempt 2 in `gfid.json`). If neither differs, no harness defect is diagnosed: G-fid fails and **T2 is dropped with typed reason `fidelity_not_reproduced`** before any GPU work (listed as not run; the G-port pass, the G-fid record and these diagnostics are published).
- **Not done** (outcome-driven protocol changes): a shorter horizon (the paper's appendix figures show 40 predicted PHYRE frames, but §5.2 states 90 for the metrics, and the band froze 90), `--sample`, another checkpoint, a different test split. **Disclosed context [INFERENCE]:** the released `hparams.json` has β_KL 0.15 (`run_prefix` "_bkl_0.15_std_1.0") where Table 4 prints 0.1 for PHYRE, so the released checkpoint may not be the run behind Table 3; our numbers sit at the paper's G-SWM row. A Tier-T retrain at the paper's budget (several days on 4 GPUs) is outside P-fid's repair rule and the remaining budget.
