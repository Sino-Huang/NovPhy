# NovPhy server setup — 4x RTX 5090 (2026-10-03)

Self-check of the NovPhy environment after the move from the RTX 3090 workstation.
Scope: steps 1–6 of the server-migration checklist. Only this file is committed.

## Environment

| | value |
|---|---|
| host | `MU00097158L`, Ubuntu 24.04.4 LTS, user `sukai` |
| CPU | AMD Ryzen Threadripper 9960X 24-core (48 threads) |
| GPU | 4x NVIDIA GeForce RTX 5090, 32607 MiB, compute capability 12.0 |
| driver | 595.91.07; `nvidia-smi` reports max CUDA 13.2 (≥ 13.0 required) |
| repo path | `/home/sukai/Project/NovPhy` (`~/Project` → `/mnt/array/sukaih/Project`) |
| conda env | `novphy` at `/home/sukai/miniconda3/envs/novphy`, Python 3.11.16 |
| torch | `2.13.0+cu130`, CUDA 13.0, cuDNN 92000 (9.20.0), triton 3.7.1 |
| torch arch list | `sm_75 sm_80 sm_86 sm_90 sm_100 sm_120`; device capability `(12, 0)` |
| capture stack | TigerVNC `Xvnc` 1.13.1, ffmpeg 6.1.1-3ubuntu5, libgl1/libegl1/libglx 1.7.0, Mesa 25.2.8, libvulkan1 1.3.275, `libnvidia-gl-595` |
| GitHub / git | `gh` logged in as `Sino-Huang` (done by the owner); global git identity `Sukai Huang <hsk6808065@163.com>`; SSH remote `origin` reachable |
| Java | OpenJDK 21.0.12.1 (`openjdk-21-jre-headless`, installed by the owner) |

## Pass/fail by step

| step | result | detail |
|---|---|---|
| 1. Paths | **FAIL (needs root)** → replaced by a user-directed record rewrite | `/p` doesn't exist and creating it needs root (`/` is `root:root 755`, no passwordless sudo). The owner said the canonical path on this server is `~/Project/NovPhy` and had every `/p/Project/NovPhy` in the records rewritten (see Changes). All required files exist. |
| 2. Conda / torch / GPU | PASS | env created from `novphy-env.yml` with no relaxation; torch cu130 with `sm_120`; `deterministic_scoring()` sets the policy and restores flags afterwards; `torch.compile` on CUDA works |
| 3. Unit tests | PASS | `47 passed in 5.93s` (all five files) |
| 4. Cross-hardware replication | **FAIL** | before the rewrite: 3/5 blocked by missing `/p` paths, 2/5 numeric disagreements (deltas below). After the rewrite: 5/5 fail on frozen-input hash bindings. **Cross-hardware replication did not pass.** |
| 5. Training sanity | PASS | `--version 3 --dry-run` exit 0; compiled 4-member ensemble: 0.24–0.28 s per Δ = 1 full-horizon update (numbers below) |
| 6. Capture stack | PASS (with rendering caveat) | Xvnc, ffmpeg, GL/Vulkan and Java 21 present; player renders its title screen correctly in **software (llvmpipe)**, no crash; Unity editor licensed (re-activated by the owner), batch-mode check exit 0 |

## Changes made

1. **Conda environment** (allowed environment change):
   `conda env create -f novphy-env.yml`. It solved and installed as specified; I relaxed no package. `novphy-env.yml` is unchanged.
2. **User-directed rewrite of frozen records.** This breaks hard rule 1. The owner chose it in this session after being told it would break the hash bindings, and then chose to keep the rewritten state.
   - Command (throwaway script, deleted): for each file listed by
     `find .local-artifacts data -type f -print0 | xargs -0 -P 32 -n 2000 grep -lF '/p/Project/NovPhy'`
     (38,076 files), replace the bytes `/p/Project/NovPhy` with `/home/sukai/Project/NovPhy`. I wrote the expanded path instead of `~/Project/NovPhy` because `open()` does not expand `~`.
   - **35,581 text files rewritten**: `.json`, `.log`, `.xml`, `.csv`, `.jsonl`, `.html`, `.md`, `plan.json.superseded-*`. That includes **96 `plan.json` files**, among them `issue-77-n1-dynamics-v1`, `issue-87-closed-loop-oracle-v1`, `issue-93-second-parameterization-v1`, `issue-94-launch-power-v1` and `issue-96-tau-ad-within-checkpoint-v1`.
   - **2,495 binary files not touched** (2,335 `.pt`, 160 `.dll`). They still contain `/p/Project/NovPhy`; a byte rewrite would corrupt their length-prefixed encodings.
   - Backup of every original: `/mnt/array/sukaih/novphy-p-path-backup-20261003/originals.tar` (3.5 GB), plus `manifest.tsv` (path, sha256 before, sha256 after, occurrences) and `skipped_binary.txt`. To restore: `tar -xf originals.tar -C ~/Project/NovPhy`.
   - Git: 1,735 tracked files under `.local-artifacts/`/`data/` now show as modified (5 before; those are earlier, unrelated changes to `issue-92-cross-pool-audit-v1`). None are committed.
   - Not affected: the `#99`, `#100` and `#112` output directories that step 4 compares against. They hold no `/p` text, and their 8,198 files match a sha256 manifest taken before any validate and before the rewrite.
3. **Git identity** (owner asked for it), copied from the author of the existing commits:
   - `git config --global user.name "Sukai Huang"`
   - `git config --global user.email "hsk6808065@163.com"`
   - `gh auth setup-git` makes `gh` the credential helper for `https://github.com`, so HTTPS pushes use the gh login. `origin` itself is SSH and already works (`git ls-remote origin` succeeds).
4. **Done by the owner, not by me:** `gh auth login`; installed `openjdk-21-jre-headless` (21.0.12.1+1-1-24.04.4-Ubuntu); re-activated the Unity license through Unity Hub, which wrote `~/.local/share/unity3d/Unity/Unity_lic.ulf`.
5. **Unity Hub link handler.** The extracted Hub had no handler for `unityhub://`, so the browser login could not return to Hub. Fix:
   - Created `~/.local/share/applications/unityhub.desktop` with `Exec=…/hub-3.20.0/root/usr/bin/unityhub --no-sandbox --disable-gpu %U` and `MimeType=x-scheme-handler/unityhub;application/x-unityhub;`.
   - Ran `update-desktop-database ~/.local/share/applications` and `xdg-mime default unityhub.desktop x-scheme-handler/unityhub`.
   - Why the flags: Hub needs `--no-sandbox` because Ubuntu 24.04 restricts unprivileged user namespaces (`kernel.apparmor_restrict_unprivileged_userns=1`); without it Hub exits with `No usable sandbox!`. `--disable-gpu` avoids `GPU process isn't usable` on displays without GPU GL.
6. **Nothing else.** No other system packages, drivers, environment variables or symlinks were changed. I edited no runner or module.

## Step 1 — paths

- `.local-artifacts/`, `data/`, `novphy-env.yml`, `.local-artifacts/issue-77-n1-v1/player/` and `.local-artifacts/issue-99-slot-encoder-fix-v1/encoder/encoder.pt` all exist.
- `whoami` = `sukai`. Scripts with hard-coded home paths (not edited; not in the current work path):
  - `/home/sukaih/...` (no such home on this server):
    - `scripts/issue_76_native_player.py` (lines 10–11)
    - `scripts/issue_76_shared_history_player.py` (10–11)
    - `scripts/run_issue_76_canonical_player.py` (13)
    - `scripts/run_issue_76_canonical_smoke.py` (18)
  - `/home/sukai/...`: `scripts/capture_issue_50_evidence.py` (36). This one is an issue-50 script, not issue-76, and it resolves here.
- No runner, module or test contains `/p/Project`. Runners derive `ROOT` from `Path(__file__).resolve()`. The `/p` dependency comes only from absolute paths stored in records. The step-4 failure came from `.local-artifacts/issue-93-second-parameterization-v1/anchor_retention.json` (14 `/p` paths; sha256 `178c1797…` is bound by the #93, #94, #96 and #99 plans), read by `run_launch_power_probe.frozen_anchor`.
- I tried a no-root alias with bubblewrap. It fails with `setting up uid map: Permission denied` because `kernel.apparmor_restrict_unprivileged_userns=1`.

## Step 2 — conda env export and diff

`conda env export -n novphy --no-builds`:

```yaml
name: novphy
channels:
  - defaults
dependencies:
  - _libgcc_mutex=0.1
  - _openmp_mutex=5.1
  - bzip2=1.0.8
  - ca-certificates=2026.8.13
  - ld_impl_linux-64=2.44
  - libexpat=2.8.2
  - libffi=3.4.8
  - libgcc=15.2.0
  - libgcc-ng=15.2.0
  - libnsl=2.0.0
  - libstdcxx=15.2.0
  - libuuid=1.41.5
  - libxcb=1.17.0
  - libzlib=1.3.2
  - ncurses=6.6
  - openssl=3.5.7
  - packaging=26.3
  - pip=26.1.2
  - pthread-stubs=0.3
  - python=3.11.16
  - readline=8.3
  - setuptools=83.0.0
  - sqlite=3.53.2
  - tk=8.6.15
  - tzdata=2026c
  - wheel=0.47.0
  - xorg-libx11=1.8.13
  - xorg-libxau=1.0.12
  - xorg-libxdmcp=1.1.5
  - xorg-xorgproto=2025.1
  - xz=5.8.2
  - zlib=1.3.2
  - pip:
      - contourpy==1.3.3
      - cuda-bindings==13.3.1
      - cuda-pathfinder==1.8.0
      - cuda-toolkit==13.0.3.0
      - cycler==0.12.1
      - et-xmlfile==2.0.0
      - filelock==3.32.4
      - fonttools==4.63.0
      - fsspec==2026.7.0
      - h5py==3.16.0
      - iniconfig==2.3.0
      - jinja2==3.1.6
      - joblib==1.5.3
      - kiwisolver==1.5.1
      - markupsafe==3.0.3
      - matplotlib==3.11.1
      - mpmath==1.3.0
      - narwhals==2.25.0
      - networkx==3.6.1
      - numpy==2.4.6
      - nvidia-cublas==13.1.1.3
      - nvidia-cuda-cupti==13.0.85
      - nvidia-cuda-nvrtc==13.0.88
      - nvidia-cuda-runtime==13.0.96
      - nvidia-cudnn-cu13==9.20.0.48
      - nvidia-cufft==12.0.0.61
      - nvidia-cufile==1.15.1.6
      - nvidia-curand==10.4.0.35
      - nvidia-cusolver==12.0.4.66
      - nvidia-cusparse==12.6.3.3
      - nvidia-cusparselt-cu13==0.8.1
      - nvidia-nccl-cu13==2.29.7
      - nvidia-nvjitlink==13.3.33
      - nvidia-nvshmem-cu13==3.4.5
      - nvidia-nvtx==13.0.85
      - openpyxl==3.1.5
      - pandas==3.0.5
      - pillow==12.3.0
      - plotly==7.0.0
      - pluggy==1.6.0
      - pygments==2.21.0
      - pyparsing==3.3.2
      - pytest==9.1.1
      - python-dateutil==2.9.0.post0
      - scikit-learn==1.9.0
      - scipy==1.17.1
      - seaborn==0.13.2
      - six==1.17.0
      - sympy==1.14.0
      - threadpoolctl==3.6.0
      - torch==2.13.0+cu130
      - triton==3.7.1
      - typing-extensions==4.16.0
prefix: /home/sukai/miniconda3/envs/novphy
```

`diff novphy-env.yml <export>`: only cosmetic differences, with identical package versions.

```diff
38d37
<       - --extra-index-url https://download.pytorch.org/whl/cu130
89c88
<       - torch==2.13.0
---
>       - torch==2.13.0+cu130
91a91
> prefix: /home/sukai/miniconda3/envs/novphy
```

Checks:

```
$ python -c "import torch;print(torch.__version__, ...)"
2.13.0+cu130 13.0 True 4 NVIDIA GeForce RTX 5090 (12, 0) ['sm_75', 'sm_80', 'sm_86', 'sm_90', 'sm_100', 'sm_120']
$ python -c "...with deterministic_scoring() as p: print(p, torch.backends.cudnn.deterministic)"
{'cudnn_benchmark': False, 'cudnn_deterministic': True, 'cudnn_allow_tf32': False, 'matmul_allow_tf32': False} True
(after exit: deterministic False, cudnn.allow_tf32 True, matmul.allow_tf32 False — prior flags restored)
$ python -c "...torch.compile(lambda x:x*2+1)..."
tensor([3., 3., 3., 3.], device='cuda:0')
```

## Step 3 — unit tests

`pytest -q tests/test_scoring_harness.py tests/test_issue_112_decision_dynamics.py tests/test_issue_112_stable_dynamics.py tests/test_issue_100_readout_fidelity.py tests/test_issue_99_decision_chain.py`
→ `47 passed in 5.93s` (`CUDA_VISIBLE_DEVICES=0`). This includes the in-process reproducibility test in `tests/test_scoring_harness.py`, so scoring is self-consistent on this GPU.

## Step 4 — deterministic-scoring replication

Every command ran with `CUDA_VISIBLE_DEVICES=0` under the shared lock at `/tmp/novphy-addexp-gpu.lock`.

### Run A: before the record rewrite (records as frozen on the RTX 3090)

| command | exit | failure | class |
|---|---|---|---|
| `run_decision_relevant_dynamics --validate` | 1 | `[Errno 2] No such file or directory: '/p/Project/NovPhy/.local-artifacts/issue-87-closed-loop-oracle-v1/attempts/oracle--issue-77-n1-001-a00--a03/decision-frame.png'` | (a) environment: missing `/p` |
| `run_decision_dynamics_controls --validate` | 1 | same missing file | (a) |
| `run_decision_dynamics_versions --version 3 --validate` | 1 | `compute.json differs from the fresh recomputation` | **(b) numeric** |
| `run_decision_chain_guard_revision --validate` | 1 | same missing file (via `run_decision_chain_attribution.validate` → `run_launch_power_probe.frozen_anchor`) | (a) |
| `run_readout_fidelity_guard_revision --validate` | 1 | `compute.json differs from the fresh recomputation` (from the nested `run_readout_fidelity_jepa.validate` of issue-100 v1) | **(b) numeric** |

Numeric disagreements (b). I recomputed them read-only with a throwaway script that repeats each runner's validate logic and diffs the JSON leaves.

1. **#112 v3** (`issue-112-decision-dynamics-v3/compute.json`). The `selection.json` and `gate_b_handoff.json` re-derivations passed. Exactly 3 leaves differ, all in the G3 spot replication of one re-scored record (225 steps at Δ = 1):

   | leaf | RTX 3090 (retained) | RTX 5090 (fresh) |
   |---|---|---|
   | `guards.G3.observed.spot_replication.max_abs_delta.pig_presence` | 0.0 | 7.152557373046875e-07 |
   | `guards.G3.observed.spot_replication.max_abs_delta.pig_displacement` | 0.0 | 3.957967866230483e-07 |
   | `guards.G3.observed.spot_replication.max_abs_delta.tie_free` | 0.0 | 7.036926030767887e-07 |

   The G3 pass flag, every token and every selection are unchanged; all deltas are about 1e-6, far inside the 1e-3 tolerance. The byte-compare of `compute.json` fails because the 3090 recorded bit-exact 0.0 and the 5090 does not reproduce it bit-exactly.

2. **#100 v1, nested inside the #100 v2 validate** (`issue-100-readout-fidelity-jepa-v1/compute.json`). Exactly 2 leaves differ, and a guard flips:

   | leaf | RTX 3090 (retained) | RTX 5090 (fresh) |
   |---|---|---|
   | `guards.G6.observed.max_abs_delta` | 2.384185791015625e-05 | 0.019077062606811523 |
   | `guards.G6.pass` (tolerance `CARRIER_BUILD_TOLERANCE` = 1e-4) | true | **false** |

   G6 compares the differentiable JEPA carrier with the retained adapter carriers on 5 frames (positions 0, 1, 80, 200, 330) of the probe capture. It runs under default flags, not `deterministic_scoring()` (v1 predates the #111 harness). Breakdown per seed, read-only:

   | seed | flags | raw max | where | non-motion max | harness `carrier_replication` (scaled motion) |
   |---|---|---|---|---|---|
   | 20260908 | default (cudnn TF32 on) | 0.019077 | pos 1, col 114 (motion) | 0.001661 | **fail** (non-motion 1.66e-3 > 1e-3) |
   | 20260909 | default | 0.016725 | pos 80, col 179 (motion) | 0.000310 | pass |
   | 20260910 | default | 0.005850 | pos 80, col 62 (motion) | 0.000316 | pass |
   | 20260908 | `deterministic_scoring()` | 0.001972 | pos 1, col 114 (motion) | 0.000184 | pass |
   | 20260909 | `deterministic_scoring()` | 0.004572 | pos 80, col 179 (motion) | 0.000062 | pass |
   | 20260910 | `deterministic_scoring()` | 0.003752 | pos 80, col 62 (motion) | 0.000091 | pass |

   The largest deltas are in motion columns (center difference / elapsed, elapsed 0.02–0.087 s). Even with TF32 off, every seed exceeds G6's 1e-4.

   With the nested v1 check skipped, the #100 v2 recomputation itself matches its `compute.json` exactly (0 differing leaves).

The three path-blocked commands never reached scoring, so they have no numeric result.

### Run B: after the record rewrite (current state)

| command | exit | failure |
|---|---|---|
| `run_decision_relevant_dynamics --validate` | 1 | `frozen input issue_77_dynamics_plan changed after the freeze` |
| `run_decision_dynamics_controls --validate` | 1 | `frozen input issue_77_dynamics_plan changed after the freeze` |
| `run_decision_dynamics_versions --version 3 --validate` | 1 | `frozen input issue_77_dynamics_plan changed after the freeze` |
| `run_decision_chain_guard_revision --validate` | 1 | `frozen input issue_96_plan changed after the freeze` |
| `run_readout_fidelity_guard_revision --validate` | 1 | `frozen input issue_77_dynamics_plan changed after the freeze` |

The rewrite changed the sha256 of the bound plans, so every validate now stops at the binding check, before any scoring. Passing it would require re-freezing hashes in frozen plans; hard rules 1 and 2 forbid that, and it was not done.

**Verdict: cross-hardware replication did not pass.**

## Step 5 — training sanity

- `python -u -m scripts.run_decision_dynamics_versions --version 3 --dry-run` exit 0, both before and after the record rewrite. No `--smoke` or `--train`.
- Throwaway `/tmp` script, now deleted:
  - `decision_dynamics.Ensemble` with 4 × `CNNHybridPredictor`, `compile=True`.
  - Each update: `local_loss` (B = 64, 61-frame windows, hybrid symbol labels) + `truncated_long_and_rank_loss` (truncation 45, horizon 225, endpoint 225; 64 long rows, 2 cells × 20 candidates) + `update`.
  - 18 updates cycling the 9 `cnn_hybrid.PAIRS`, random carriers, GPU 0.
- Compile warm-up (first 9 updates): 114.5 s in total.
- After warm-up, seconds per update:

  | pair Δ | continuous | micro | macro |
  |---|---|---|---|
  | 1 (225 transitions, full horizon) | 0.240 | 0.279 | 0.274 |
  | 5 | 0.066 | 0.077 | 0.072 |
  | 15 | 0.031 | 0.039 | 0.037 |

  Mean 0.124 s/update. Peak GPU memory 3.52 GiB allocated, 3.87 GiB reserved. No retirements.
- Reference: about 0.25 s per full-horizon update for a **13**-member group on the RTX 3090. This run used 4 members, as the checklist specifies, so the two figures are not directly comparable.

## Step 6 — capture stack

- Present: `/usr/bin/Xvnc` (TigerVNC 1.13.1), `/usr/bin/ffmpeg` 6.1.1, libGL/libEGL/libGLX (glvnd 1.7.0), Mesa 25.2.8 (GL and Vulkan), libvulkan1, NVIDIA GL/Vulkan ICDs (`libnvidia-gl-595`), gcc/g++.
- **Java:** `/usr/bin/java` is OpenJDK 21.0.12.1 (`openjdk-21-jre-headless`). `scripts/smoke_physics_capture.py` launches `java -jar ./game_playing_interface.jar` (built with JDK 13.0.2, `Main-Class: server.ABServer`).
  - Compatibility: the jar's newest class files are major version 56 (Java 12); Java 21 reads up to 65.
  - `java --dry-run -jar .local-artifacts/issue-77-n1-v1/player/game_playing_interface.jar` exits 0. That loads `server.ABServer` without running `main`, so no engine was started.
- Player smoke (two short launches of a `/tmp` copy; no level, no agent, no gameplay):
  - Ran a `/tmp` copy of `.local-artifacts/issue-77-n1-v1/player/`, because the `9001.x86_64` wrapper renames files in its own directory.
  - Launched `./9001.x86_64 -logFile …` (wrapper adds `-force-glcore -screen-width 840 -screen-height 480`) for 20 s on a private `Xvnc :197`, with `LD_LIBRARY_PATH` stripped.
  - Result: alive after 20 s, no crash or signal.
  - `Renderer: llvmpipe (LLVM 20.1.2, 256 bits)`, `Vendor: Mesa`, `Version: 4.5 (Core Profile) Mesa 25.2.8-0ubuntu0.24.04.2`: **software rendering, no NVIDIA GPU used.** `nvidia-smi` showed no player process.
  - One `DirectoryNotFoundException` for `9001_Data/StreamingAssets/Levels`. Expected for the bare player: the capture fills `Levels/` per attempt.
  - Second launch, for a rendering check: ffmpeg `x11grab` screenshots of the private display at 5, 12 and 20 s. All three show the correct Science Birds title screen (logo, birds, PLAY button, ground) filling the 840×480 player area, with 8,920 distinct colours. The renderer was again llvmpipe on Mesa 25.2.8.
  - The RTX 3090 workstation also rendered in software. A retained #87 attempt log shows `llvmpipe (LLVM 22.1.6)`, `Mesa 26.1.2-arch1.1`.
  - **Risk:** the Mesa/LLVM version differs (25.2.8/20.1 here vs 26.1.2/22.1 there), so frames captured here may not be byte-identical to the retained ones. The title-screen check shows that rendering works; it cannot show that in-level frames match the 3090, because that needs a capture run.
- Unity editor `~/.local/share/novphy-unity/2019.4.41f2-6b23d448b533/editor/Editor/Unity` (`UNITY_2019_4_41F2` is unset):
  - Before re-activation: `-batchmode -quit -nographics -logFile -` exited 1 with `Failed to activate/update license Missing or bad username or password`; no `Unity_lic.ulf` existed anywhere.
  - After the owner re-activated through Hub, the same command **exits 0**. Log: `Successfully connected to LicensingClient`, `Serial number assigned to: "F4-HCSV-G8FX-6VYN-NB2J-XXXX"`, `Pro License: NO` (Personal), `Current license is already valid and activated`, `Exiting batchmode successfully now!`. The license file `UpdateDate` is 2026-10-04T03:27:31.
  - **Caveat: the license is found only without `env.sh`'s `XDG_DATA_HOME`.** With `XDG_DATA_HOME=$PWD/.cache/xdg` the editor exits 1 with the same license error, because it looks in `.cache/xdg/unity3d/Unity/`. This is the existing repo convention, not a migration break: `run_issue_76_canonical_player.py` drops `XDG_DATA_HOME` before calling the editor ("Do not make the editor activate a second repository-local seat"). `scripts/build_physics_player.sh` does not drop it, so run it as `env -u XDG_DATA_HOME scripts/build_physics_player.sh …` or from a shell that has not sourced `env.sh`.
  - Note: Hub created `~/.local/share/unity3d/Unity/` with mode 777 and `Unity_lic.ulf` with mode 777, so the other accounts on this shared server can read the license. Left unchanged; the owner can tighten it with `chmod 700 ~/.local/share/unity3d/Unity && chmod 600 ~/.local/share/unity3d/Unity/Unity_lic.ulf`.
