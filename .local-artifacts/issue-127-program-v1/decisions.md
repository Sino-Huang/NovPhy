# #127 program-level decisions

## Decision P-budget (2026-10-25): how the 400 GPU-h program budget is counted

**Trigger.** Rule P-budget allocates "the 400 GPU-h"; the first night ran several #127 jobs side by side on the same GPU (light jobs of 1–4 GB beside each other), as the scheduler's `shares_gpu` allows. The #123 ledger convention counts each process's wall time on its GPU, so two jobs sharing one GPU for an hour count as two GPU-hours. Fixed now, before any budget-dependent decision (no Tier-C or coverage arm has been funded or considered).

**Decision.**
- **Program budget = device time.** For each GPU, the union of the time intervals in which at least one #127 scheduler run (job ids `128-*` … `132-*` and later #127 children) was running on it, summed over the four GPUs; computed from the scheduler's run records (`launched_at` / `ended_at` in `.local-artifacts/gpu-window/queue.json`). This is the resource the owner's budget controls. P-budget, P-cap and P-coverage funding checks use it.
- **Child ledgers are unchanged:** each child keeps the #123 per-process GPU-seconds (each arm-seed's training counted once from its `report.json`), and its frozen per-child cap is checked against that per-process number, which can only be larger than the child's device time. Where an arm-seed shared its GPU, its recorded training GPU-h overstates exclusive compute; the program summary marks shared runs and also reports each arm's exclusive-throughput estimate (updates × the smoke-measured seconds per update alone) beside the recorded number, for the AUC-vs-compute frontier.
- Both numbers are published in the program summary; the budget statement uses device time.

**Not changed:** child caps, the 48 h-on-4-GPUs projection rule, every arm's native budget and seeds.

## Decision P-other (2026-10-25): G-action in carrier space — the frozen guard stands for every child; the calibration is reported beside every world-model arm

**Trigger.** #129 (B1) is published: B1 and B1-m fail the frozen G-action (endpoint action sensitivity ≥ 0.05) on every seed. A descriptive fit-split calibration made afterwards (`.local-artifacts/issue-129-slotformer-v1/analysis/g-action-calibration.json`, the same `eb.action_sensitivity` call on the 480 fit anchors × 16 grid launches, 236-d carriers at frame min(225, end)) gives:

| source | carrier-space action sensitivity |
| --- | --- |
| engine truth (true carriers of the executed grid shots) | 0.0395 |
| U (Gate-B unit, predicted) | 0.0229 |
| C\* (predicted) | 0.0334 |
| B1 (predicted) | 0.0130 |

The 0.05 threshold was set on #123's latent states. In carrier space it is not attainable even by a perfect model, so as applied to the carrier-space world models (B1, C1-car, D1, and the coverage arms on carriers) the guard turns every token into `readiness_or_precision_insufficient` regardless of the model.

**Options.** (a) Keep the frozen guard for every child. (b) Relax or rescale it for the children not yet scored. (c) Add a calibrated guard as a secondary token set.

**Decision: (a).** The guard is part of the #127 shared protocol, frozen at the first child freeze and applied to B1 already; it is not changed, relaxed or supplemented after an evaluation score has been seen (integrity rule). Changing it now would treat B1 and the later arms under different rules, and a relaxed guard would license dispositions in U's favour (B1 − U = −0.197) that the pre-registered rule does not. Every world-model child and the program summary instead report, beside each arm, its carrier-space action sensitivity as a share of the engine truth's (descriptive), with this calibration and a plain statement that the frozen guard is unattainable in carrier space. Descriptive estimates are published for every contrast regardless of tokens. Nothing published changes. The direct predictors (#128), whose guards do not include G-action, are unaffected.

## Decision P-budget (2026-10-25): Tier-C trigger — DDLP

**Trigger (frozen in #127's ladder):** "DDLP: if B1 fails G-readout or its position error at frame 225 exceeds U's, to separate slot dynamics from slot representation." #129 publishes (descriptive, `summary.extra.positions`): position error at frame 225, B1 **0.0508** [0.0462, 0.0555] vs U **0.0474** [0.0440, 0.0508]. B1 has no readout, so the G-readout clause does not apply; the frame-225 clause compares the published means and B1's exceeds U's. The trigger fires (the rule names no interval condition; the overlapping intervals are disclosed).

**Decision.** DDLP enters the program as a triggered Tier-C arm in its own child ticket, with the full protocol (vendoring, G-port, G-fid in a band frozen before the run, the NovPhy adapter, ≤ 3 configurations, 3 seeds, one scoring pass). Its GPU funding follows P-budget order: after (1) the program set's fidelity, selection and training, (2) required controls and (3) secondary arms are allocated, triggered Tier-C arms come next (4), before the coverage arms (5: DreamerV3 #133, γ-models #134). CPU implementation starts now; no GPU job is enqueued before a funding decision posted on its ticket. If the remainder cannot fund it, it is listed as not run (typed `cap`).

## Decision P-coverage (2026-10-26): coverage arm 1 (DreamerV3, #133) funded; Diffusion Forcing's projection measured next; γ-models decided after it

**Inputs (as of 2026-10-26 20:31 AEDT).**
- Program device time used: **93.27 GPU-h** of 400 (P-budget, device time; `run_issue_127_program --budget`). Remaining **306.73 GPU-h**.
- P-budget items (1)–(4) still to run, projected (process GPU-h, an upper bound on device time): #128 ≈ 47 (A4 Tier-T folds 1–2 ≈ 22.7, A4 selection and two seeds 21.5 from the #128 smoke, readouts and scoring ≈ 3); #130 ≈ 12 (Tier-E ≈ 10, scoring 2); #131 ≈ 26 (D1 seeds ≈ 20 left, D1-m ≈ 1, readouts 3, scoring 2). Sum **≈ 85**.
- Reserve for item (2) U-native, which fires only if an arm's X − U is `supported` at a native budget above U's: three seeds at the largest program-set native budget (D1, 12.5 GPU-h) plus scoring, **40 GPU-h**.
- Item (4): DDLP (the only Tier-C trigger fired) was dropped, `fidelity_not_reproduced`, 0 GPU-h. The Diffusion Forcing / γ-models trigger (D1 − U or D1 − J_dec `supported`) and the DreamerV3 / DIAMOND / V-JEPA 2-AC trigger (a world-model arm at disposition 1) are not evaluable yet; if one fires, the triggered arm moves into item (4) and is funded ahead of the remaining coverage arms.
- **Remainder for P-coverage: 306.73 − 85 − 40 ≈ 181.7 GPU-h.**

**Clean timing probes** (exclusive GPU 3 tonight; #133 issuecomment-6402603214, #134 issuecomment-6402603433; contended records kept):

| arm | s/update (contended → clean) | projection (measured, clean) | child cap | within caps |
| --- | --- | --- | --- | --- |
| #133 T1 DreamerV3 | PushT 0.725 → 0.130; T1 native 0.128 | **59.75 GPU-h** (G-fid 8.75; selection 11.41 + 11.41 + 2.85; two seeds 22.82; readouts 1.5; scoring 1.0); serial-ladder wall 33.0 h on 4 GPUs | 76 | yes |
| #134 T3 γ-models | 0.061 (clean) | **4.74 GPU-h** (G-fid on CPU, done; selection 3 × 0.84; two seeds 1.69; scoring 0.5) | 12 | yes |

**Decision (P-coverage order: DreamerV3, Diffusion Forcing, γ-models, DDLP, DIAMOND, V-JEPA 2-AC, DeepPHY).**
1. **DreamerV3 (#133) is funded**: 59.75 ≤ 181.7. `funded.json` is written now; its queue (cache build, G-fid training and planning against the frozen band [0.18, 0.42], then selection) is enqueued, never on GPU 2 while #130's Tier-E run needs it exclusively. Remainder after it ≈ 122 GPU-h.
2. **Diffusion Forcing** is next in order. It has no implementation or projection yet; the scout's estimate (88–184 GPU-h [INFERENCE]) straddles the remainder, so neither funding nor skipping is decidable. Its Phase 0 (vendoring, port, G-port, fidelity band, a ≤ 1 GPU-h timing probe) starts now on CPU in its own child ticket; it is funded only if its measured projection fits the remainder then.
3. **γ-models (#134)** comes after Diffusion Forcing in the order, so its funding is decided right after Diffusion Forcing's (it is small, 4.74 GPU-h; waiting costs no wall time because the GPUs are busy with the program set and #133 for the next nights).
4. DIAMOND, V-JEPA 2-AC and DeepPHY are decided in order after that, each by a measured projection against the remainder; any arm that does not fit is listed in the summary as not run, typed `cap`.

## Decision P-coverage (2026-10-27): Diffusion Forcing (#136) and γ-models (#134) funded; DIAMOND's projection measured next

**Inputs (as of 2026-10-27 07:08 AEDT).**
- Program device time used **128.33 GPU-h**; remaining **271.67 GPU-h** (`run_issue_127_program --budget`).
- #130 and #131 are published and validated (no further GPU time). #129 published earlier.
- Still to run under P-budget (1)–(4): #128 ≈ 36 GPU-h (A4 Tier-T fold 2 ≈ 11.8 h left; A4 selection and two seeds 21.5; readouts and scoring ≈ 3).
- U-native reserve (item 2), re-derived from the arms still unscored: it can fire only for A1, A2 or A4 (B1, C1-car and D1 have X − U well below zero and stop tokens). Three seeds at A4's native budget (6.9 GPU-h) plus scoring: **25 GPU-h** (was 40 while D1 was unscored).
- Tier-C triggers, now evaluable for the world models: **Diffusion Forcing / γ-models** (D1 − U or D1 − J_dec `supported`): does not fire (D1 − U −0.2456, D1 − J_dec −0.2471, stop tokens). **DreamerV3 / DIAMOND / V-JEPA 2-AC** (a world-model arm at disposition 1): does not fire (B1, C1-car, D1 at disposition 4; C1-dino dropped). So all three stay coverage arms.
- Already funded: #133 DreamerV3, 59.75 GPU-h projected, 10.41 used; its G-fid training is running ~15 % slower than the probe (real batches), so ≈ 52 GPU-h is committed.
- **Remainder for further coverage arms: 271.67 − 36 − 25 − 52 ≈ 158.7 GPU-h.**

**Measured projections.**

| arm | probe | projection (measured) | within caps |
| --- | --- | --- | --- |
| #136 T4 Diffusion Forcing | exclusive GPU 2 (issuecomment-6413975265): ts 0.0674 s/update, planning 0.1024, 1.22 s/decision | **31.7 GPU-h** (G-fid 9.5; selection 4.8 + 1.9 + 3.4; two seeds 9.7; readouts 0.6; scoring 1.8); wall 21.2 h | yes (cap 184, 48 h) |
| #134 T3 γ-models | exclusive GPU 3 (issuecomment-6402603433) | **4.74 GPU-h** | yes |

**Decision (P-coverage order).**
1. **Diffusion Forcing (#136) is funded** (31.7 ≤ 158.7); remainder ≈ 127.
2. **γ-models (#134) is funded** (4.74 ≤ 127); remainder ≈ 122.
3. **DDLP** (next in order) was already run as the triggered Tier-C arm and dropped (`fidelity_not_reproduced`); it is not re-run as a coverage arm.
4. **DIAMOND** is next: no implementation or projection yet; the scout estimate (64–136 GPU-h [INFERENCE]) straddles the remainder, so its Phase 0 (vendoring, port, G-port, fidelity band, timing probe) starts now on CPU in a new child ticket; funded only if its measured projection fits then. V-JEPA 2-AC and DeepPHY follow in order the same way.

## Decision P-coverage (2026-10-27, evening): Diffusion Forcing dropped (fidelity); DIAMOND (#137) funded; V-JEPA 2-AC's projection measured next

**Inputs (2026-10-27 18:18 AEDT).**
- Program device time used **129.35 GPU-h**; remaining **270.65 GPU-h**.
- Committed: #128 ≈ 36 (A4 fold 2 running; A4 selection and seeds; scoring); U-native reserve 25; #133 DreamerV3 ≈ 52; #134 γ-models 4.74 (selection running tonight).
- **#136 Diffusion Forcing dropped**, typed `fidelity_not_reproduced` (issuecomment-6436060220): the official pipeline (paper branch, unedited) gave Exchange CRPS-sum 0.0489 against the frozen band [0.0005, 0.0055); no port defect was diagnosed, so P-fid allows no repair. It used 0.19 GPU-h; ≈ 31.5 GPU-h return to the remainder.
- **Remainder for further coverage arms: 270.65 − 36 − 25 − 52 − 4.74 ≈ 153 GPU-h.**

**DIAMOND (#137) measured projection** (exclusive GPU 1 tonight, issuecomment-6428331215): native 0.058 s/update, ar4 0.139, 6.38 s/decision → **80.3 GPU-h** (selection 6.6 + 15.9 + 13.2; two seeds at the most expensive configuration 31.7; selection readouts 0.5; readouts 3.1; scoring incl. novelty 9.4; G-fid 0, done on CPU); serial-ladder wall 42.6 h on 4 GPUs; within the child cap (136), the seed cap (24) and the 48 h wall cap. Phase 0 had passed readiness (0.6783 ≥ 0.65), G-port (exact) and G-fid (Tier E: Pong 20.89, Freeway 32.22, Boxing 90.74, inside the frozen bands).

**Decision (P-coverage order).**
1. **DIAMOND (#137) is funded** (80.3 ≤ 153); remainder ≈ 73 GPU-h. Its selection runs are placed on GPUs 0, 1 and 3 (GPU 2 carries #128's A4 selection and seeds, the program set's critical path).
2. **V-JEPA 2-AC** is next: no implementation or projection yet; the scout estimate (44–90 GPU-h [INFERENCE]) straddles the remainder, so its Phase 0 starts now on CPU in a new child ticket, funded only if its measured projection fits then. **DeepPHY** (scout 4–18 GPU-h, open-weight VLMs run locally) follows in order the same way.

## Decision P-coverage (2026-10-28): V-JEPA 2-AC not run (`cap`); DeepPHY (#139) funded — the P-coverage list is exhausted

**V-JEPA 2-AC (#138)** was not run, typed `cap` (issuecomment-6436855469): its FLOP-count projection from the official native budget (94 500 updates × 256 clips on the ViT-g AC predictor) is 2 109–5 271 GPU-h; one native seed alone (≥ 285 GPU-h) is ≈ 4× the remainder (≈ 73 GPU-h) and ≈ 12× the 24 GPU-h seed cap. Cutting the native budget would weaken the baseline, which #127 forbids. Phase 0 (licences, route, adapter, G-port exact, G-fid band) is committed and reported.

**DeepPHY (#139)**, the last arm in P-coverage, measured on an exclusive GPU (issuecomment-6437800465): Qwen2.5-VL-7B 3.78 s per decision, 3B 3.11 s → **8.8 GPU-h** in total (G-fid on PHYRE 4.5; selection 0.2; readouts 0.5; scoring 1.05; novelty 0.5; validation 1.05), within the 18 GPU-h child cap and the remainder. **Funded**; remainder after it ≈ 64 GPU-h.

**The P-coverage list is exhausted** (DreamerV3, γ-models, DIAMOND and DeepPHY funded; Diffusion Forcing and DDLP dropped at fidelity; V-JEPA 2-AC `cap`). The leftover (≈ 64 GPU-h, minus any overrun of the funded children) stays as headroom for the funded children's own overruns (#133's G-fid training is running ≈ 1.8× its probe projection under GPU sharing) and for P-budget item (6), a P-follow-up ticket, which fires only if an arm reaches disposition 1. No further arm is added.

## Decision P-budget / P-follow-up (2026-10-29): #128 published — U-native fires for A4; P-follow-up fires (A1, A2, A2-native, A4)

**Trigger.** #128's primary group is published and validated (exit 0; 33 records; spot check A2 exact): A1 0.9946, A2 0.9757, A2-native 0.9740, A4 0.9945 pooled grid AUC against U 0.7444; X − U +0.230 to +0.250, every one `supported` with guards passing → **disposition 1** for A1, A2, A2-native and A4. Context published beside it (descriptive): the no-model nearest-fit reference NN-px reaches 0.9039 on the same grid (U − NN-px −0.1595), and on the #110 novelty pairs the A-family falls to 0.52–0.58 pooled on novel levels (U 0.47).

**Mechanical evaluation of the frozen triggers** (`run_issue_127_program --summary --allow-partial`):
- **Budget safeguard / U-native: fires for A4 only.** A4's training compute is 2 435 GPU-s per seed, 1.88 × U's B1 training compute (1 294 GPU-s). A1 (0.32 ×), A2 (0.04 ×) and A2-native (0.05 ×) reach disposition 1 below U's budget, so their wording is not budget-qualified. A4's wording stays budget-qualified ("A4 at its native budget (1.88 × U's training GPU-h) ranks better than U at B1") until **U-native** — U trained to A4's training GPU-h — has run (P-budget item 2, a required control). It runs as a Tier-C child with the full protocol.
- **P-follow-up: fires** (A1, A2, A2-native, A4 at disposition 1). A follow-up ticket is opened now with a pre-registered design, frozen before training, that adds the winning arms' distinguishing component to U ("if the baseline is better, borrow its ideas"); it is funded under P-budget item 6 from the remainder if its measured projection fits.
- DreamerV3 / DIAMOND / V-JEPA 2-AC (a world-model arm at disposition 1): does not fire. Diffusion Forcing / γ-models (D1 − U or D1 − J_dec supported): does not fire.
- The program-level intersection–union claim ("U ranks launches better than every baseline in this set") is **not licensed**: vetoed by A1, A2 and A4 (disposition 1) and by B1, C1-dino and D1 (guards / drop).

**Budget.** Program device time 204.17 GPU-h used; remaining 195.83 (P-budget, device time). Still committed: #137 DIAMOND scoring (≈ 10), #128's A3 group (≈ 2). U-native (item 2) is funded first; the P-follow-up (item 6) after it.
