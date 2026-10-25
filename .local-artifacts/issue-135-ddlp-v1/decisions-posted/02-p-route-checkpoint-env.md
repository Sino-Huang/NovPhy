## Decision P-route (2026-10-25): the released DDLP checkpoints are reachable → G-fid stays Tier E (PHYRE-128); official code on torch 2.13 in env `b127-ddlp`

**Trigger.** #135 makes checkpoint reachability the first decisive Phase-0 check: the README's model zoo hosts DDLP checkpoints on MEGA.nz; if none is reachable, the Tier-E G-fid becomes Tier T (retrain at the paper's budget). Separately, taldatech/ddlp pins `torch==2.0.1` / `torchvision==0.15.2` (no RTX 5090 kernels) while our stack is torch 2.13 + CUDA 13 (rule P-route).

**Facts (2026-10-25, this host).** No MEGA client is installed (no megatools / mega-get). MEGA's public-link API answers from this host (`POST https://g.api.mega.co.nz/cs` `[{"a":"g","g":1,"p":<id>}]` → size, encrypted attributes, temporary download URL). A 150-line fetcher (`scripts/b127_ddlp_mega.py`: AES-128-CTR decryption with the link key and MEGA's chunked CBC-MAC checked against the key's meta-MAC; download rejected on mismatch) downloaded, all with **meta-MAC verified**:

| file (README link) | bytes | sha256 |
| --- | --- | --- |
| `ddlp-phyre.zip` (DDLP PHYRE 128×128, `UBcl2LgQ`) | 86 888 423 | `efa5878b6c6317896886c6e77a7a0ef83672390e67ecd194031657d51aceb06a` |
| `ddlp-clevrer.zip` (DDLP CLEVRER 128×128, `oUUjXY4B`) | 34 712 687 | `427388ea49fcb67cb0b1b39a6f6a0a52af905f98351a88fd6d0ea827825da450` |
| `phyre_dataset.zip` (the authors' generated PHYRE data, `lIkDnZZD`) | 453 909 158 | `22d346893213b5b619139ba1958ee035e00e6de4de9ce226b57d06355d0a20ec` |

`ddlp-phyre/phyre_ddlp.pth` loads into the official `ObjectDynamicsDLP` built from its `hparams.json` with all keys matched (23 386 410 parameters); its hparams equal the paper's Table 4 PHYRE column (K = 25 particles, L = 64 prior proposals, 15 training frames, MSE loss, feature dim 4). The PHYRE data holds train / valid / test = 2 574 / 312 / 400 episodes, exactly the paper's appendix G split. Stored in `data/external/ddlp/{checkpoints,data}/` with `MANIFEST.json` (url, bytes, sha256, downloaded_at, meta_mac_ok); gitignored, never committed.

**Options.** (a) Tier E on PHYRE-128 (released checkpoint + released test data, official `eval/eval_gen_metrics.py`); (b) Tier E on CLEVRER-128 (needs the CLEVRER validation videos from MIT and the authors' unspecified per-video trimming); (c) Tier T retrain.

**Decision.** (a). Tier T is not needed. PHYRE is also the closest published setting to NovPhy (2-D physics puzzles) and the configuration the NovPhy adapter starts from. The band is frozen and posted in a separate Decision P-fid before any fidelity run. The evaluation is cheap enough for the CPU (measured: `model.sample` of 10 episodes × 100 frames in 110 s at 4 threads → ≈ 75 min for the 400 test episodes plus LPIPS), so G-fid needs no GPU funding.

**Environment (P-route).** `b127-ddlp` = a clone of `novphy` (Python 3.11, torch 2.13.0+cu130) plus `torchvision 0.28.0+cu130` (installed `--no-deps`, so torch is unchanged), `piqa 1.3.1` (the repo's pin, used by the official metrics), `opencv-python-headless`, `scikit-image`, `imageio`, `accelerate` (imports only) and `pycryptodome` (the fetcher). The official code imports and runs on torch 2.13 with no edit. The G-port reference (the unedited clone) and the adapter both run in this env on the same torch build (the #129 lesson: no cross-version Adam round-off). The NovPhy runner also runs in this env (the vendored code imports torchvision / cv2, which `novphy` lacks); `novphy` itself is not modified. Package list: `.local-artifacts/issue-135-ddlp-v1/envs/b127-ddlp.txt`.

**Upstream defect handled without editing the clone.** `datasets/get_dataset.py` imports `datasets.langtable_ds`, a file that does not exist at any commit of the repo, so `eval/eval_gen_metrics.py` cannot be imported as released. The fidelity harness registers an empty stand-in module under that name before running the unchanged script (the PHYRE path never touches it); disclosed in `fidelity/gfid.json`.
