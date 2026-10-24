# #127 automatic decision log

56 decisions on #127, #128, #129, #130, #131, #132 (fetched 2026-10-24T19:22:23Z; order: created_at, url).

## #127 (program): 1

| created (UTC) | rule | title |
| --- | --- | --- |
| 2026-10-24T19:14:39Z | P-budget | [how the 400 GPU-h program budget is counted](https://github.com/Sino-Huang/NovPhy/issues/127#issuecomment-6388410336) |

## #128 (A (A1, A2, A3, A4)): 22

| created (UTC) | rule | title |
| --- | --- | --- |
| 2026-10-24T08:58:46Z | P-license | [A4 DynDQN — vendor facebookresearch/DynamicsAware @ eb65576 (Apache-2.0)](https://github.com/Sino-Huang/NovPhy/issues/128#issuecomment-6382269414) |
| 2026-10-24T08:59:39Z | P-license | [A3 PhyQ-RelDQN — vendor phy-q/benchmark @ 514491c (MIT) model and training code](https://github.com/Sino-Huang/NovPhy/issues/128#issuecomment-6382277009) |
| 2026-10-24T09:03:17Z | P-license | [A1/A2 DQN — vendor facebookresearch/phyre @ d0765dd (Apache-2.0) and torchvision v0.4.0 ResNet (BSD-3-Clause)](https://github.com/Sino-Huang/NovPhy/issues/128#issuecomment-6382309351) |
| 2026-10-24T09:11:56Z | P-route | [A4 DynDQN — port the official DynamicsAware stack off Python 3.6 / CUDA 10.2 to a modern env, held to G-port](https://github.com/Sino-Huang/NovPhy/issues/128#issuecomment-6382387919) |
| 2026-10-24T09:11:57Z | P-fid | [A4 DynDQN G-fid Tier T band frozen before any run — \[82.8, 87.2\]](https://github.com/Sino-Huang/NovPhy/issues/128#issuecomment-6382388098) |
| 2026-10-24T09:13:44Z | P-fid | [A1 Tier-E G-fid band frozen before the run — PHYRE-B within-template AUCCESS in \[75.4, 79.8\]](https://github.com/Sino-Huang/NovPhy/issues/128#issuecomment-6382404486) |
| 2026-10-24T09:15:55Z | P-other | [A4 DynDQN — mapping the official hand-crafted trajectory distance onto NovPhy fit-shot engine trajectories](https://github.com/Sino-Huang/NovPhy/issues/128#issuecomment-6382423720) |
| 2026-10-24T09:15:57Z | P-cap | [A4 DynDQN Tier-T fidelity projects above the #127 estimate — fund from unallocated program budget, no cut](https://github.com/Sino-Huang/NovPhy/issues/128#issuecomment-6382423910) |
| 2026-10-24T09:17:43Z | P-other | [A1/A2 adapter choices (PHYRE DQN recipe on the NovPhy fit shots), fixed before any training](https://github.com/Sino-Huang/NovPhy/issues/128#issuecomment-6382444832) |
| 2026-10-24T09:42:03Z | P-other | [#128 Phase-1 selection design (≤ 3 configurations per arm), native budgets, child budget projection — fixed before any selection run](https://github.com/Sino-Huang/NovPhy/issues/128#issuecomment-6382674719) |
| 2026-10-24T10:01:21Z | P-cap | [A4 Tier-T fidelity — measured projection 40.3 GPU-h (folds 0–2), funded from unallocated program budget](https://github.com/Sino-Huang/NovPhy/issues/128#issuecomment-6382844952) |
| 2026-10-24T10:13:46Z | P-other | [A3 PhyQ-RelDQN — state source: Phy-Q's symbolic state rebuilt from our engine's ground-truth JSON (A3-sym); the 320 × 240 frame through the official image transform (A3-img)](https://github.com/Sino-Huang/NovPhy/issues/128#issuecomment-6382972582) |
| 2026-10-24T10:13:48Z | P-other | [A3 PhyQ-RelDQN — action mapping: each launch → the nearest native action of Phy-Q's 180-way angle head](https://github.com/Sino-Huang/NovPhy/issues/128#issuecomment-6382972743) |
| 2026-10-24T10:13:49Z | P-other | [A3 PhyQ-RelDQN — upstream training defects: the is_done long-index bug is not on the official path; the image agents' uint8 memmap cast is fixed (disclosed)](https://github.com/Sino-Huang/NovPhy/issues/128#issuecomment-6382972952) |
| 2026-10-24T10:13:50Z | P-other | [A3 PhyQ-RelDQN — native recipe on the fit shots, 3 pre-declared configurations, projected GPU time](https://github.com/Sino-Huang/NovPhy/issues/128#issuecomment-6382973132) |
| 2026-10-24T10:38:37Z | P-other | [A4 Tier-T runtime — torch CPU threads capped at 8 (fold 0 paused and resumed once)](https://github.com/Sino-Huang/NovPhy/issues/128#issuecomment-6383201139) |
| 2026-10-24T16:40:18Z | P-cap | [A4 DynDQN's native budget is measured above its 3 GPU-h/seed cap → draw from unallocated program budget; no cut](https://github.com/Sino-Huang/NovPhy/issues/128#issuecomment-6386804015) |
| 2026-10-24T16:41:11Z | P-crash | [A4 Tier-T folds killed by host-RAM exhaustion (operator-caused); relaunched once, at most two folds at a time](https://github.com/Sino-Huang/NovPhy/issues/128#issuecomment-6386813372) |
| 2026-10-24T16:42:48Z | P-novelty | [the descriptive #110 novelty scoring pass (owner decision D6)](https://github.com/Sino-Huang/NovPhy/issues/128#issuecomment-6386830623) |
| 2026-10-24T17:23:53Z | P-other | [capture-noise tolerance for the frame-0 check; fit/policy layout redundancy measured; NN-px / NN-car no-model references added (descriptive)](https://github.com/Sino-Huang/NovPhy/issues/128#issuecomment-6387263066) |
| 2026-10-24T18:35:01Z | P-other | [addendum: NN-px / NN-car are defined on grid and angle only; the fit split never executed offset or power launches, so those inventory columns are empty by construction.](https://github.com/Sino-Huang/NovPhy/issues/128#issuecomment-6388000077) |
| 2026-10-24T19:12:47Z | P-engine | [A3 native-task fidelity design (target, band, cost) and its funding under the shared engine allowance](https://github.com/Sino-Huang/NovPhy/issues/128#issuecomment-6388391126) |

## #129 (B1 SlotFormer-E99): 9

| created (UTC) | rule | title |
| --- | --- | --- |
| 2026-10-24T08:59:20Z | P-license | [vendor SlotFormer's slot Transformer, loss and LR schedule (MIT)](https://github.com/Sino-Huang/NovPhy/issues/129#issuecomment-6382274253) |
| 2026-10-24T09:08:55Z | P-other | [B1 adapter — SlotFormer's PHYRE model on E99 slots](https://github.com/Sino-Huang/NovPhy/issues/129#issuecomment-6382359850) |
| 2026-10-24T09:13:30Z | P-route | [run the official SlotFormer code unmodified on torch 2.7.1 + CUDA 12.8 (env `b127-slotformer`)](https://github.com/Sino-Huang/NovPhy/issues/129#issuecomment-6382402424) |
| 2026-10-24T09:13:32Z | P-fid | [frozen Tier-E band for B1, CLEVRER mIoU 49.42 ± 0.5](https://github.com/Sino-Huang/NovPhy/issues/129#issuecomment-6382402612) |
| 2026-10-24T09:32:26Z | P-other | [B1 selection configurations, native schedule, native head, budget projection](https://github.com/Sino-Huang/NovPhy/issues/129#issuecomment-6382590719) |
| 2026-10-24T09:51:11Z | P-crash | [selection runs relaunched once after a co-scheduling OOM](https://github.com/Sino-Huang/NovPhy/issues/129#issuecomment-6382755036) |
| 2026-10-24T10:38:06Z | P-fid | [correction — the B1 G-port parameter criterion is a declared variant of #123's rule](https://github.com/Sino-Huang/NovPhy/issues/129#issuecomment-6383196400) |
| 2026-10-24T10:47:36Z | P-fid | [follow-up to the G-port correction — the frozen G-port reference runs on torch 2.13](https://github.com/Sino-Huang/NovPhy/issues/129#issuecomment-6383281200) |
| 2026-10-24T16:42:49Z | P-novelty | [the descriptive #110 novelty scoring pass (owner decision D6)](https://github.com/Sino-Huang/NovPhy/issues/129#issuecomment-6386830857) |

## #130 (C1 DINO-WM): 10

| created (UTC) | rule | title |
| --- | --- | --- |
| 2026-10-24T09:12:36Z | P-fid | [Tier-E G-fid band and protocol frozen before the run](https://github.com/Sino-Huang/NovPhy/issues/130#issuecomment-6382394238) |
| 2026-10-24T09:15:50Z | P-license | [vendor DINO-WM (MIT) and pin DINOv2 (Apache-2.0)](https://github.com/Sino-Huang/NovPhy/issues/130#issuecomment-6382422982) |
| 2026-10-24T09:29:22Z | P-other | [Tier-E runs in a full night window; tonight's GPU 2 goes to the feature cache](https://github.com/Sino-Huang/NovPhy/issues/130#issuecomment-6382562140) |
| 2026-10-24T09:30:55Z | P-other | [C1 adapter design (preprocessing, launch encoding, schedule, C1-car tokens, R1, feature cache, readiness smoke)](https://github.com/Sino-Huang/NovPhy/issues/130#issuecomment-6382576605) |
| 2026-10-24T10:40:19Z | P-other | [C1-dino stops at the readiness smoke (typed `readiness_ceiling_below_threshold`); C1-car proceeds](https://github.com/Sino-Huang/NovPhy/issues/130#issuecomment-6383216316) |
| 2026-10-24T10:56:25Z | P-other | [Tier-E fidelity run made resumable at MPC-iteration boundaries (protocol unchanged)](https://github.com/Sino-Huang/NovPhy/issues/130#issuecomment-6383358026) |
| 2026-10-24T16:32:55Z | P-other | [C1-car runner, native schedule, selection configurations, guards, budget](https://github.com/Sino-Huang/NovPhy/issues/130#issuecomment-6386706483) |
| 2026-10-24T16:38:42Z | P-crash | [Tier-E run 2 (`130-gfid-tier-e-r#1`) crashed on CUDA OOM caused by a foreign GPU process; relaunched once](https://github.com/Sino-Huang/NovPhy/issues/130#issuecomment-6386786676) |
| 2026-10-24T16:41:36Z | P-crash | [addendum to issuecomment-6386786676 — cause identified, relaunch placement](https://github.com/Sino-Huang/NovPhy/issues/130#issuecomment-6386817726) |
| 2026-10-24T16:42:51Z | P-novelty | [the descriptive #110 novelty scoring pass (owner decision D6)](https://github.com/Sino-Huang/NovPhy/issues/130#issuecomment-6386831073) |

## #131 (D1 THICK): 11

| created (UTC) | rule | title |
| --- | --- | --- |
| 2026-10-24T08:46:44Z | P-route | [THICK route (b), our PyTorch port under G-port](https://github.com/Sino-Huang/NovPhy/issues/131#issuecomment-6382162858) |
| 2026-10-24T09:03:07Z | P-license | [vendor THICK @ 934980a (MIT), verbatim, as the G-port / G-fid reference](https://github.com/Sino-Huang/NovPhy/issues/131#issuecomment-6382307800) |
| 2026-10-24T09:20:26Z | P-fid | [frozen G-fid band for THICK on VisualPinPad (before any fidelity run)](https://github.com/Sino-Huang/NovPhy/issues/131#issuecomment-6382475412) |
| 2026-10-24T09:23:27Z | P-other | [D1 THICK NovPhy adapter (declared before any training)](https://github.com/Sino-Huang/NovPhy/issues/131#issuecomment-6382505350) |
| 2026-10-24T09:51:21Z | P-route | [G-fid on our PyTorch port (route ii), not the official TF2 stack](https://github.com/Sino-Huang/NovPhy/issues/131#issuecomment-6382756572) |
| 2026-10-24T09:51:23Z | P-cap | [D1 projection within the child share; fidelity above the #127 sub-estimate](https://github.com/Sino-Huang/NovPhy/issues/131#issuecomment-6382756903) |
| 2026-10-24T10:37:44Z | P-other | [D1 runner — guards (G-collapse latent, G-action), matched D1-m, exploratory outputs, readiness and scoring gates](https://github.com/Sino-Huang/NovPhy/issues/131#issuecomment-6383193006) |
| 2026-10-24T10:51:09Z | P-other | [G-fid runner implementation (route ii), declared before the G-fid runs](https://github.com/Sino-Huang/NovPhy/issues/131#issuecomment-6383312669) |
| 2026-10-24T10:58:59Z | P-other | [correction to the D1 runner decision, item 8 — training precision is float32 with TF32 on (per Decision P-route)](https://github.com/Sino-Huang/NovPhy/issues/131#issuecomment-6383381590) |
| 2026-10-24T16:40:34Z | P-crash | [G-fid timing run 2 contaminated by host-RAM exhaustion; clean rerun](https://github.com/Sino-Huang/NovPhy/issues/131#issuecomment-6386806922) |
| 2026-10-24T16:42:52Z | P-novelty | [the descriptive #110 novelty scoring pass (owner decision D6)](https://github.com/Sino-Huang/NovPhy/issues/131#issuecomment-6386831318) |

## #132 (E1 AIBIRDS agents): 3

| created (UTC) | rule | title |
| --- | --- | --- |
| 2026-10-24T08:59:00Z | P-license | [per-artifact licenses of the E1 agents and their transport, verified before any execution](https://github.com/Sino-Huang/NovPhy/issues/132#issuecomment-6382271495) |
| 2026-10-24T09:46:39Z | P-engine | [E1 engine path (choice in the stock engine, verdict in the #102 channel) and the 20-level smoke design](https://github.com/Sino-Huang/NovPhy/issues/132#issuecomment-6382714673) |
| 2026-10-24T19:12:46Z | P-engine | [E1 smoke result, attempts per agent and the N1 projection against the 190 worker-hour allowance](https://github.com/Sino-Huang/NovPhy/issues/132#issuecomment-6388390880) |

