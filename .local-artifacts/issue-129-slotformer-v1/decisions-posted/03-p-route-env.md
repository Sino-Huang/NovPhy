## Decision P-route (2026-10-24): run the official SlotFormer code unmodified on torch 2.7.1 + CUDA 12.8 (env `b127-slotformer`)

**Trigger.** SlotFormer's `docs/install.md` pins Python 3.8.8, PyTorch 1.10.1 + CUDA 11.3, nerv v0.1.0, `phyre==0.2.2` and `einops==0.3.2`. CUDA 11.3 builds have no kernels for the RTX 5090 (sm_120). The `phyre` 0.2.2 wheels exist only for Python 3.6, yet SlotFormer's dataset package imports `phyre` unconditionally.

**Options considered.**
- (a) Port the code to the novphy env (torch 2.13). Rejected: it would mix the reference stack with ours.
- (b) Build PHYRE from source in a Python 3.6 / torch 1.10 env. Rejected: it cannot use the GPU, and PHYRE is unused by the CLEVRER evaluation and by G-port.
- (c) **Chosen:** a separate env, `b127-slotformer`, with no edit to any official file.

**The env** (Python 3.10; package list in the artefact `envs/b127-slotformer.txt`, sha256 `ea85388def7e982cbff9d51f2ff13229a065f86d9651464d49a220600135682a`):
- torch 2.7.1 + cu128 and torchvision 0.22.1 (the oldest torch line with sm_120 kernels);
- nerv v0.1.0 (commit e83ac66), installed `--no-deps` with only the packages its imported modules need. open3d and pytorch-lightning are listed in its setup.py but never imported;
- numpy 1.26.4, scikit-image 0.25.2, lpips 0.1.4, pycocotools, einops 0.3.2 and pandas.

Both the evaluation and the G-port reference run the official files from the pinned clone `SlotFormer@10a6d9b`.

**Run-time measures.** None of these touches official code.
1. A `phyre` shim module is placed on `PYTHONPATH`. Any attribute access raises, so a PHYRE call would fail loudly. Only the import is satisfied.
2. The evaluation runs from a work directory (`fidelity/run`) with the relative paths the official config expects: `pretrained/` and `data/CLEVRER/`. Both are symlinks to `data/external/slotformer/`.
3. The released checkpoints load under torch ≥ 2.6's `weights_only=True` default (checked on CPU), so no load flag is changed.

**Data assembled for the official CLEVRER evaluation.** The official CLEVRER layout is built under `data/external/slotformer/CLEVRER/` (MANIFEST.json there; CLEVRER is CC0):
- the #106 validation videos (symlinked);
- the validation annotations (unzipped from the #106 copy);
- the `derender_proposals` object masks (downloaded, 396 807 825 B, sha256 `68ea4785…b7f3`);
- the released `clevrer_slots.pkl` (13 581 152 270 B, sha256 `05bbab80…ef27`).

**Check.** A CPU dry run of the official pipeline ran on 8 validation clips with the released weights: dataset build (2 314 filter_enter clips), slots, masks, forward and metrics, with no error. This is a pipeline check only; the Tier-E statistic comes from the full GPU run.
