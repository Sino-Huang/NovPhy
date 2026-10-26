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
