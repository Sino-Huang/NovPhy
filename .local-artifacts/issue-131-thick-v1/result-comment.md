## Result (2026-10-27): #131 D1 THICK — disposition 4 (per-contrast tokens only); D1 vetoes the program claim

Published and validated (`--validate` exit 0, scheduler run `131-validate#1`, 2026-10-27 05:20 AEDT; records manifest `sha256:170dde49…`; publication commits `b4b0df9a`, `f3be7485`). Full write-up: `docs/issue-131-thick.md`.

**Gates.**
- **G-port pass.** World model (carrier and official VisualPinPad configurations): 40/40 module forwards; worst gradient rel. L2 1.6e-5. Full agent (THICK Dreamer, DreamerV2, Plan2Explore): worst gradient rel. L2 1.4e-5, actions equal.
- **G-fid pass** (VisualPinPadFour, band frozen before the run; 3 seeds per method, the paper used 7):
  - THICK Dreamer 370.0, inside [359.7, 388.8]; DreamerV2 376.7, inside [361.3, 384.4].
  - THICK reaches 359 at 120k env steps vs DreamerV2's 210k (paper: 140k vs 200k).
- **Global guards pass.**
- **D1 fails G-action on every seed.** Endpoint action sensitivity is 0.00064 / 0.00461 / 0.00166 against the frozen 0.05; that is 5.8 % of the engine truth's 0.0395 in carrier space (descriptive).
- **D1-m fails G-action and G-collapse on every seed.** Effective rank is 53–57 of 1 082.

**Sealed grid (311 mixed cells × 3 seeds; descriptive; every token `readiness_or_precision_insufficient` because of the guard failures).**

| | AUC / estimate [CI] |
| --- | --- |
| D1 | 0.4989 [0.4810, 0.5165] (U 0.7444 [0.7269, 0.7620]) |
| D1-m | 0.4820 [0.4659, 0.4974] |
| D1 − U | −0.2456 [−0.2689, −0.2226] |
| D1 − J_dec | −0.2471 [−0.2710, −0.2236] |
| D1-m − U | −0.2624 [−0.2903, −0.2351] |
| D1-coarse − D1 (exploratory) | −0.0088 [−0.0295, +0.0117] |
| D1-hl − D1 (exploratory) | −0.0018 [−0.0239, +0.0200] |
| wrong anchor, right − wrong: D1 / D1-m | −0.0176 [−0.0353, −0.0002] / +0.0206 [+0.0081, +0.0326] |

**Disposition 4.**
- No scoped ranking claim in either direction: disposition 1 and 2 need D1's guards to pass, and disposition 3 needs the guards to pass too.
- D1 is a program-set member that fails a guard, so it vetoes the #127 program claim.
- The budget safeguard does not fire. D1's training compute is 30.5 × U's.

**Selection and commit points (exploratory).**
- Selected `ctxt_sparsity` 1, at policy AUC 0.5312. The alternatives were 0.4802 (10) and 0.4598 (25), so all three sit near chance.
- D1's context gates open at 42.2 of the 45 rollout steps on average, and D1-m's at every step. On NovPhy the learned adaptive Δ therefore collapses to every step.

**Tier-C trigger (owned here): not met.** D1 − U and D1 − J_dec are not `supported`. Diffusion Forcing and γ-models are therefore not triggered as Tier C; they run as P-coverage arms.

**P-novelty:** neither arm is scored, because both fail guards. The U / C* / P references are computed (`findings.md`).

**Compute:** about 108.0 process GPU-h against the child cap of 146 (G-fid 48.5, selection and training 58.8 (the selection sparsity-1 run is D1 seed 20260908, counted once), readouts and scoring 0.1, unrecorded 0.55); engine 0 s.

**Disclosures** (doc, "Deviations and disclosures"):
- Two P-crash relaunches in the G-fid pipeline: the contaminated timing run, and the exploration-cache race. The cache race was fixed with a build-once, verified cache: 0 mismatches over 1 000 000 steps.
- Correction to the addendum (comment 6398164689): the discarded run-1 seconds of DreamerV2 s20260909 and s20260910 (≈ 0.42 h each) are included in their ledger entries; I wrote there that they were not counted.
- The `report.json` wall seconds of the G-fid runs exclude each run's final segment. Compute uses the ledger instead.
- One listed vendored edit.
- TF32 was on for training, and scoring was deterministic.
- The readout uses zero gate noise.
- THICK has no native success head on NovPhy.
- The equal tuning allowance applies (D1 tuned on the policy split; U was not).
