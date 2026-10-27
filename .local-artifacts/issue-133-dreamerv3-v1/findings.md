# #133 T1 DreamerV3 (coverage arm): findings

T1 was **dropped** (fidelity_not_reproduced, rule P-fid, https://github.com/Sino-Huang/NovPhy/issues/133#issuecomment-6444334625) before any NovPhy selection, training or scoring; no evaluation data was read. It never enters the program claim.

- G-port: pass True (exact against NM512/dreamerv3-torch @ 6ef8646).
- G-fid (DINO-WM Table 1 DreamerV3 PushT, paper 0.3; frozen band [0.18, 0.42]): 0.02 (1/50), run .local-artifacts/issue-133-dreamerv3-v1/fidelity/runs/tier-t-1.
- Diagnosis (no port defect): open-loop prediction follows the actions; the paper's latent objective ranks the dataset's ground-truth actions weakly. arXiv v1 of DINO-WM reports 0.04 for the same row.

  - probs: median rank of the ground-truth actions among 301 = 38.0, top-30 share 0.48
  - stoch_mode: median rank of the ground-truth actions among 301 = 27.5, top-30 share 0.52
  - feat: median rank of the ground-truth actions among 301 = 57.5, top-30 share 0.38
  - image: median rank of the ground-truth actions among 301 = 0.0, top-30 share 0.76
  - proprio_dec: median rank of the ground-truth actions among 301 = 0.0, top-30 share 0.84

Repair attempts used: 0 of 2; the band was never changed.
