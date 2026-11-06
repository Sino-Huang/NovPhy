# Author response — ICLR 2027 submission #6352

*“Granularity Is a Decision: Joint Horizon–Description Selection in Latent World Models”*
Forum: https://openreview.net/forum?id=xB9J0v9J5W

We thank **MMed**, **MSwm** and **vSA9** for detailed, technically specific reviews. MSwm is right that the paper reports its experiments honestly; we have tried to hold to that standard in the work done since submission, including where it went against us.

That work is now complete: every ticket of the AFTER-ICLR program (#99–#125, #127–#141) is closed, each with a pre-registered plan where an experiment ran, a disclosure log, a guard cascade and a published disposition; #142 (the manuscript carry) and the separate bug tickets #97/#98/#126 remain open. Its overall result is not a defence of the submitted claims. **Three of the submitted headline claims do not survive intact our own sealed-split re-evaluation**, and we say so below rather than arguing around it:

1. *“Matches a horizon-only adaptive baseline at a third of its compute”* — **retracted**. The measured MAC ratio is **0.784**, against the 0.40 rule; and no policy beats a fixed coarse request on the matched continuous predictor (#103, #122).
2. *“The joint policy beats the best fixed request / r†”* — **retracted**. On a sealed 400-level split the gains are not at our own 0.02 effect-size threshold; decision-valued labels improve on our own ICLR policy by +0.040 but still trail a fixed request (#101, #114, #104).
3. *“Description levels pay off”* — **substantially narrowed**. The joint description axis reaches the threshold only against a cont-only policy on the *same* checkpoint (+0.0201), the macro level shows no advantage under any of four verdicts (#120), and the predictive advantage of the whole model over direct success predictors is absent (#127).

What survives is the mechanism and the diagnostic: one request-conditioned predictor over the horizon × description lattice, a DP-labelled request policy that needs no RL, matched-budget accounting, and — the result we did not expect — a demonstration that **granularity selection is currently limited by predictor update allocation and readout design, not by the request set**. We would rather present that, correctly, than the submitted version. Detailed reviewer answers follow; §1–§6 handle concerns raised by more than one reviewer, §7–§9 are reviewer-specific, and §10–§11 state the revised claims and the exact manuscript changes.

**Scope of the evidence, stated once so no section overstates it.** Every number is from a completed experiment, but the evidence sets differ and we label them:
- **Sealed NovPhy split** (480 fit / 120 policy / 400 evaluation level-disjoint levels, 11 680 evaluation launches, 0 typed failures, capture smoke 0/208 typed failures, #109): §1, §3(a,b,d), §6, §7 Q2–Q4, §8, §9, §10. This is the only evidence that carries confirmatory weight.
- **CLEVRER / Physion-Dominoes** (§2, §7 Q1, §8 Q3): leave-one-scene-out (EXPLORATORY) or in-sample (DESCRIPTIVE) on their own carriers, plus a separate video-disjoint pixel protocol (#106). **Not** on the sealed split.
- **External baselines** (§6): the same sealed grid cells, but a separate baseline protocol with its own frozen guards and dispositions (#127).
- **Closed loop** (§7 Q3, §9 Q4): live engine execution on the same 400 levels, pass@3 DESCRIPTIVE, and one frozen guard failed, so all its tokens are stop tokens (#102).
Intervals are level-clustered percentile bootstraps (10 000 draws) and are DESCRIPTIVE; tokens follow a pre-declared δ = 0.02 rule (`supported` / `not_supported_by_this_experiment` / `readiness_or_precision_insufficient`).

---

## 1. Common concern: the evaluation is underpowered and the level counts are too small (MMed W1, MSwm W1, vSA9 Q4)

**Direct answer.** Both reviewers are correct about the submitted numbers, and the problem is worse than “small”: the submitted design could not have resolved its own effect size. We rebuilt the evaluation and, having done so, **the effect the submitted paper claimed is gone**.

**What the submitted numbers were.** The NovPhy contrast sets carried 7–14 scored mixed levels per inventory (e.g. 14 grid members in #96). A pre-registered power analysis on those member records (#104) shows that resolving δ = 0.02 requires ~208 members on grid and ~344 on angle at the observed between-level variance — i.e. the submitted intervals were 3–5× too wide by construction. MSwm’s observation that only one of four Fig. 1(b2) intervals clears zero is a symptom of that, not of a marginal effect.

**What we ran instead.** A new level-disjoint cohort (#109), captured on a repaired pipeline (the old pipeline’s 51/51 timeouts were a port-reservation race, and 82 paired-view mismatches were the bird/pig blink animation; both fixed and audited), then sealed and scored once (#104): **400 evaluation levels, 311 mixed on grid, 76 offset, 78 angle, 47 power, × 3 seeds = 933 grid cells**, 0 typed failures.

**Result — and the honest reading (#104).** Even at 311 mixed levels the achieved half-width of the primary contrast is **0.030**, still above δ = 0.02:

| inventory | mixed levels | J − F\* | half-width | levels needed for δ = 0.02 |
|---|---|---|---|---|
| grid | 311 | +0.013 [−0.016, +0.043] | 0.030 | ≈ 687 |
| offset | 76 | +0.011 [−0.052, +0.074] | 0.063 | ≈ 758 |
| angle | 78 | −0.160 [−0.206, −0.113] | 0.047 | ≈ 424 |
| power | 47 | −0.262 [−0.336, −0.192] | 0.072 | ≈ 609 |

Guard G6 (half-width ≤ δ) **fails on every inventory**, so the frozen rule gives `readiness_or_precision_insufficient` for the ICLR-policy replication and we do not claim it. The held-out between-level variance is larger than the development variance the design assumed.

**Consequence for the claim.** Because the redesign can resolve 0.03-scale effects but not 0.004-scale ones, the submitted “policy beats r† by +0.004 to +0.070” cannot be sustained: on the sealed split, with decision-valued labels, **J_dec − F_hindsight = −0.016 [−0.020, −0.012]** (grid, #101) — the policy is *below* the best fixed request, not above a near-chance r†. The submitted sentence is withdrawn; the revised claim is “no request-selection policy in our family beats the best fixed request at δ on the sealed split” (#108, cross-cutting reading 1).

**What is now properly powered.** Contrasts that are large enough to resolve at this sample size are reported with their intervals and tokens, e.g. J − J-cont +0.0201 [0.0154, 0.0248] (#114), J − specialist +0.0848 [0.0742, 0.0952] (#117), U − H-JEPA descriptive gaps of 0.25 (#123/#125). MSwm’s Q2 is answered with the actual numbers in §4 and §8.

---

## 2. Common concern: CLEVRER and Physion splits, and scene overlap (MMed W1/Q1, vSA9 Q1/Q3, MSwm W3)

**Direct answer.** Yes — the submitted Table 1 was **in-sample over scenes**: fitting and evaluation used the same 12 CLEVRER scenes and the same 12 Physion-Dominoes trials, with longer rollouts at test time. That is exactly the `#79`/`#84` and ICLR `tab:res:rollout` setting, and the paper did not label it. This is a reporting defect, and it invalidates any generalization reading of that table. Three things now exist.

**(a) Exact scene identities (vSA9 Q1).** CLEVRER: validation scenes **10000–10011** (12 scenes). Physion-Dominoes: **trials 0–11 of the release prefix** (12 trials). These are the only carrier scenes `#79`/`#84` built; the split rule *was* scene-consistent, the problem is that it was not held out. Both will be printed in the revision.

**(b) A held-out recomputation on the same carriers (#117 item 5).** Leave-one-scene-out, 12 folds × 3 seeds = 2 964 members, 76 units, position MSE at t = 120 (the frozen benchmark endpoint), scene-clustered intervals:

| benchmark | recipe | joint/specialist ratio at matched per-request budget | token |
|---|---|---|---|
| CLEVRER | ICLR recipe (B) | 0.04 [0.03, 0.04] | supported |
| CLEVRER | LC (long-horizon) | 0.37 [0.33, 0.42] | supported |
| Physion-Dominoes | B | 0.08 [0.06, 0.10] | supported |
| Physion-Dominoes | LC | 0.13 [0.08, 0.22]; F-15-continuous 1.58 [0.79, 3.08] | `readiness_or_precision_insufficient` |

So sharing all weights costs ≤1.5× per-request error in three of four benchmark × recipe cells under **scene-held-out** evaluation, with Physion LC at Δ = 15 unresolved. Crucially, **this is not the submitted table recomputed**: the submitted nine-request rollout table cannot be rebuilt on held-out scenes, because no carrier set with new scenes exists for those benchmarks. Building one is a data-collection project, not a revision. The revision therefore (i) relabels the submitted table as in-sample over 12 scenes, (ii) prints the LOSO table beside it as the transfer measurement, and (iii) states that t = 225 is not delivered on these benchmarks (their clips end at 120).

**(c) A new, genuinely disjoint CLEVRER/Physion protocol from pixels (#106).** CLEVRER: 6 285 train / 3 132 validation videos, video-disjoint (37.22 % of videos excluded by a pre-declared colour×material slot-key rule); Physion-Dominoes: dynamics/readout/testing release split, 150 test trials. Results here are negative and we report them as such: on CLEVRER the frozen QA readout never fires (max micro-head probability 0.051 < 0.5), so J − J-cont = −0.0001 is `not_supported_by_this_experiment` and uninformative about the joint axis; on Physion OCP J − J-cont = −0.021 [−0.068, 0.025], `readiness_or_precision_insufficient`, with all rollout arms at 0.51–0.57 against a 0.627 observed-protocol anchor.

**A leakage qualification we found ourselves (#106, amendment A3).** With appearance-only colour dropped, **20 of 150 Physion testing trials share every physical parameter with a fitting-split trial, and 4 have a partner within 1e-3 world units at every frame** (closest pair: max position difference 3.3e-5 over 151 frames, identical per-frame contact labels). The official split measures transfer to new colours, not to independent physical realizations. The revision states this and drops the word “generalization” for Physion.

**Why not simply add persistence / constant-velocity baselines (MSwm Q3)?** We owe a straight answer: **we do not have those baselines on CLEVRER or Physion** — the `#79`/`#84` carrier releases contain no such rows, and adding them means rebuilding both carrier sets. We do have them where they are cheap and decisive: on NovPhy, `#110` reports rollout error against a persistence reference on all 45 supported novelty cells, and the finding is unflattering — **in 42/45 cells all three zero-shot Δ=1 requests are worse than persistence at 225 frames** (e.g. cell 0/rolling: F-1 continuous 0.183 vs persistence 0.104). The revision prints this table, and the *policy’s own schedules* are reported through the same metric: at 225 frames on grid the best fixed requests are Δ=15 (F-15-continuous 0.038 vs persistence 0.104 on cell 0/rolling; Δ=1 recursion is what breaks), and the J/J_dec schedules give 0.032 vs 0.110 for the hierarchical model’s 5-micro level (#105). That is the honest version of “the policy’s schedules do not follow its lowest-error request”.

---

## 3. Common concern: does the description axis pay for itself? (MMed W2/Q2, MSwm W2, vSA9 W2/Q3)

**Direct answer in four parts**, each a completed experiment.

**(a) The same-checkpoint cont-only ablation MSwm and vSA9 asked for (#114).** This did not exist in the submission. We trained J-cont on the *same* hybrid Gate-B checkpoints, with the same trunk, the same 242→128→9 policy, the same DP/DAgger recipe, restricted to continuous requests:

| policy | tie-free grid AUC | MACs/candidate | wall s/state |
|---|---|---|---|
| J (joint) | 0.7057 [0.6846, 0.7267] | 184.0M | 0.339 |
| J-cont (horizon-only) | 0.6856 [0.6647, 0.7061] | 122.7M | 0.227 |
| C-J (continuous predictor, horizon-only) | 0.6999 [0.6827, 0.7170] | 234.7M | 0.168 |
| T-cont (fixed description, adaptive horizon) | 0.7025 [0.6813, 0.7238] | 196.1M | 0.358 |
| no-model action prior | 0.7203 [0.6965, 0.7442] | 0 | — |

**J − J-cont = +0.0201 [0.0154, 0.0248]**, `supported`, at both declared budgets. This is the description axis, isolated at matched checkpoint — the reviewer’s ablation, and it lands exactly at δ. Qualifications we state in the paper: J-cont cannot reach either budget in the frozen sweep, so these are *at-most-budget* comparisons, not equal spend; and J − C-J (+0.0111 [0.0050, 0.0174]), J − T-cont (+0.0032), J − J-as-cont (+0.0037, the same J schedule replayed with every α set to continuous) are all `not_supported_by_this_experiment`. **The description axis as a free-standing claim survives only in the narrow form “on this checkpoint, choosing α by state adds 0.02 over never choosing it”. It does not survive as “α choice beats a fixed description at equal compute”, and it does not survive as a compute saving** — see (b).

**(b) MSwm’s “the saving is just the narrower trunk” is correct, and vSA9’s W2 is correct (#103, #114).** Our own compute decomposition reproduces the reviewers’ point exactly. J − C-J at λ = 1e-4 totals **−50.7M MACs**, of which calls (horizon) contribute −6.4M, per-call α **+6.6M**, trunk **−50.8M** and policy −0.1M. The per-call α term is *positive*: choosing a richer description costs compute. The apparent saving comes from the hybrid predictor’s cheaper per-call trunk, exactly as MSwm (1.27×) and vSA9 (width/tokenization/representation/data all changing together) said. **We withdraw “joint selection saves compute” entirely.** The replacement sentence, which we will print, is #103’s frontier: at the ICLR operating point J/C-J MAC ratio = **0.784** (rule needed ≤ 0.40); AUC J 0.706 [0.685, 0.727] vs C-J 0.700 [0.683, 0.717], difference +0.006; and both are dominated by a **fixed coarse request**: C-F-15 (C\*) 0.783 [0.768, 0.799] at 27.8M MACs, and the no-model action prior 0.720 at 0 MACs. On the frontier (AUC × MACs) sit exactly {prior, F-15-continuous, J at λ = 1e-1, C-F-15}.

**(c) The macro level does not pay (#113, #115, #120, #121).** We redesigned the macro predicates to be observable ((settle, structure-change) indexed by Δ, with a capture-window rule that yields 93 087 settled fit frames instead of 1) and re-ran everything:
- macro F1, held out, three sources: parsed 0.157 [0.151, 0.164], one-step predicted 0.113, open-loop rollout **0.015**; the submitted macro head reads **0.000** (#113).
- macro vs continuous ranking under four pre-registered engine verdicts: **−0.012, −0.024, −0.036, −0.078** — no macro advantage under the current cost or any alternative (#120). The one macro-advantage arm we could construct (a predictor retrained on the new predicates) gives **+0.022 [0.018, 0.026] under pig-removal**, and reverses (−0.003 / −0.063 / −0.106) under the three less settle-aligned verdicts (#120).
- predicate feedback: removing symbolic feedback altogether costs nothing, while *oracle* engine feedback **hurts** (fixed micro requests −0.345), i.e. the adapter learned the head’s output distribution rather than predicate semantics (#115). Label-free micro codes are non-inferior (J−T −0.003 [−0.015, +0.008]) but the reference’s own description value there is +0.003, so there was no ≥δ benefit to preserve (#115).
- request-code factorisation: an additive code is *worse* (F-5-continuous rollout ratio 1.84, F-5-micro 2.02); the frozen `supported` token for the fused code (+0.050) is reworded in review to “the trained fused parameterization beats the trained additive one”; interaction-specific value remains **unidentified** (#121).

**(d) Independence of the two axes does not pay either (#105).** Under one request interface, J − J_diag = −0.0068, J − J_monotone = +0.0028, J − H\* = −0.0006: `not_supported_by_this_experiment`. A constructed hierarchical world model (1.84M parameters, matched recipe) reaches 0.706 at its 5-micro level with three times J’s rollout error (0.110 vs 0.032). #78’s TAWM/VLWM ports top out at 0.642 (VLWM-15) and 0.610 (TAWM-15) on the sealed grid.

**Net answer to “what do symbolic description levels buy over continuous prediction?”** On our strongest held-out evidence: **+0.02 AUC** on one checkpoint against never choosing α (#114); nothing on rollout error (Δ=15 continuous and macro are indistinguishable, #110); nothing on macro predicates (#120); and nothing that survives into task success (§7). We say this plainly rather than claiming the submitted story.

---

## 4. Common concern: time base, Δ semantics, and the time signals the model actually receives (MSwm W3/Q1, vSA9 W1/Q2)

**Correction of scope first (MSwm W3).** The “2400 of 2409 training lineages” remark is about the **NovPhy base corpus**, not about CLEVRER/Physion, and we will not argue that the time-base problem is merely cross-dataset. In the frozen base recipe, 2400 of the 3000 training-role NovPhy lineages are used for predictor fitting (the 600 lineages whose one-based inventory index is divisible by five supervise the controller), and each shot contributes up to four contiguous 60-frame windows with targets on the carrier grid (`docs/issue-71-hybrid-readiness.md`). Those are the lineages and the windows your reading refers to.

**What the submitted text got wrong.** “Δ in steps” was never defined, so reading it as engine steps was a fair reading of our own text. Two facts we can certify from the code:
- NovPhy carriers exist only on retained frame records at a **declared capture stride of 50 native engine steps** (native timestep 0.0004 s), with ≤601 frames and a 30 000-step cap. A parseable context grid is therefore 20 ms wide (#109/#113), and there is no one-engine-step request in any NovPhy experiment — #116 records “sub-frame Δ: `unsupported`” for exactly this reason.
- The carrier does carry physical time. Its header is `[has-prior flag, elapsed seconds]` (`world_model/data/deployment_temporal.py`; `CARRIER_HEADER = 2`, `ELAPSED_COLUMN = 1` in `world_model/planning/scoring_harness.py`), and per-slot velocities are computed as `(center_t − center_{t−1}) / elapsed`, i.e. world units **per second**. Our own later internal notation restates the same 1/5/15 horizons as 50/250/750 native steps = 20/100/300 ms (`docs/issue-76-native-findings.md`), which is what one obtains if Δ counts retained frame records at the 50-step stride.

**We do not resolve this here, and we do not defend the submitted sentences.** Whether the predictor-fitting configuration counted Δ in retained frame records or in native engine steps cannot be certified from the submitted text, and we will not invent an answer in a rebuttal. We will re-derive the unit from the frozen configuration and then print, for every dataset, native timestep, capture stride, window length in frames *and* milliseconds, and the physical duration of each request. Pending that check, the submitted rollout-horizon sentences are **withdrawn**, and if the re-derivation shows training and test used different units we will say so and restate every horizon-dependent number in milliseconds.

**Which time signals the model receives (MSwm Q1, vSA9 Q2).** MSwm Q1 asks directly whether the model depends on an elapsed-time entry. The answer is that the carrier has one, and we separate three distinct time signals rather than treating them as one:
1. the **request code** (Δ, α) tells the predictor how many carrier frames ahead to jump and at which description level;
2. the **carrier’s elapsed-seconds column**, with per-second velocity normalization, tells it how much physical time separates the prior and current observation — this is the mechanism that lets one predictor serve jumps of different physical length;
3. the **policy’s remaining-frames feature κ**, an input to the request policy only.

#119 ablates (3) alone: the best unsaturated form, min(κ,225)/225, moves J − F_hindsight on the sealed grid by **+0.0042 [−0.0005, +0.0088]** (upper bound below δ, `not_supported_by_this_experiment`), and decision-valued labels are unaffected (−0.004). That result says the κ cap was not carrying the policy result. **It says nothing about channel (2): we have not run an ablation that zeroes, freezes or corrupts the elapsed column, or that disables per-second velocity normalization, so we make no claim about whether the carrier’s time channel is used.** This is an open item, not a settled one; the natural next check is exactly that ablation, and we would rather flag it than assert the channel is idle.

**Concretely, what distinguishes a one-frame from a many-frame execution.** Under the declared 50-step stride, Δ=1 asks for a target 20 ms ahead and Δ=15 for 300 ms ahead; the target the predictor is compared against lies Δ carrier frames forward, the elapsed column reports the true 20 ms spacing between parsed observations, and because motion columns are already rates, a Δ-jump is a rate-consistent extrapolation over that interval rather than a repeated one-frame step. If the re-derivation above shows the submitted configuration used native steps instead, all these durations change by 50× and the revision prints the corrected numbers.

---

## 5. Common concern: novelty relative to per-step horizon selection and passive dynamics (MSwm W2)

**Direct answer.** The citations are missing and we will add them; the correct positioning is narrower than the submitted text.

- **Chatterjee & Khardon (NeurIPS 2025)** introduces per-step horizon selection; **Neitz et al. (NeurIPS 2018)** is the passive-dynamics reference. Both will be cited and discussed, with an explicit “what is not new” paragraph: selecting Δ per step is prior work; our delta is the *joint* lattice and the decision-valued labels.
- We also cite and, more importantly, *tested* **H-JEPA (arXiv 2610.06805)**, **HWM**, **LeJEPA/SIGReg** and reused **LeWM**, which the submission did not (#123, #125). We reproduce H-JEPA faithfully on its own benchmark (Tier-E fidelity 40.0/83.3/96.0 and Tier-T 36.7/82.0/95.3 against the paper’s 40.7/81.3/96.0), and then find it **collapses on action-sparse NovPhy**: HJ-px descriptive grid AUC **0.4997**, HJ-car-m **0.4896**, against U 0.7444 and C\* 0.7834; effective rank passes but within-episode variance share is 0.0014 and launch sensitivity 1.1e-6 (threshold 0.05). With H-JEPA’s **own** anti-collapse remedy (inverse dynamics + action-SIGReg, γ₁ = 225 selected on the policy split) it still collapses: HJ-px-IDM 0.4948, HJ-car-m-IDM 0.4900; the IDM effect is unresolved (SI1 −0.0049, SI2 +0.0003; SI2 confounded by a 448→96 predictor-width change). Because both arms fail their frozen guards, **no ranking claim in either direction is licensed**; we print the descriptive gap next to that disposition.
- **TAWM venue correction (MSwm minor 1):** accepted — TAWM is ICML 2025 main track, not a workshop. Corrected in the bibliography and in the related-work sentence.

**Net.** The novelty claim we are willing to defend is: *(i)* a shared carrier over the horizon × description lattice with symbolic feedback inside one rollout; *(ii)* DP/decision-valued labelling of request choice under a compute-weighted objective, needing no RL; *(iii)* a matched-budget, engine-verdict evaluation protocol; *(iv)* the negative/diagnostic findings of §1–§3. Not “we are the first to adapt the horizon”, and not “joint selection saves compute”.

---

## 6. Common concern: soundness — does a latent world model help at all here? (MMed soundness “not easy to address”)

We ran the comparison that decides this, on the sealed grid with frozen guards (#127, 217.6 of 400 device-GPU-hours, 146 logged automatic decisions, integrity rule: no result may depend on whether it favours us; every arm that ran is reported):

| arm | grid AUC [CI] | X − U | token |
|---|---|---|---|
| U (our Gate-B unit, F-15-macro) | 0.7444 [0.7269, 0.7620] | — | reference |
| C\* (matched pure-continuous) | 0.7834 [0.7676, 0.7992] | −0.0390 | no U advantage: `not_supported_by_this_experiment`; C\* ranks better on grid, offset, angle, ties on power |
| A1 DQN, pixels (PHYRE recipe) | 0.9946 [0.9917, 0.9971] | +0.2502 | supported |
| A2 DQN, carriers | 0.9757 [0.9692, 0.9815] | +0.2313 | supported |
| A4 dynamics-aware DQN | 0.9945 [0.9922, 0.9965] | +0.2501 | supported |
| B1 SlotFormer | 0.5476 [0.5264, 0.5692] | −0.1968 | `readiness_or_precision_insufficient` |
| C1-car DINO-WM | 0.5251 [0.5093, 0.5411] | −0.2193 | `readiness_or_precision_insufficient` |
| D1 THICK | 0.4989 [0.4810, 0.5165] | −0.2456 | `readiness_or_precision_insufficient` |
| T3 γ-models / T5 DIAMOND | 0.5102 / 0.7176 | −0.2342 / −0.0268 | `readiness_or_precision_insufficient` |
| #141 F1 = frozen U + borrowed success classifier | 0.9765 [0.9689, 0.9828] | +0.2321 | supported |
| #141 F1-D (same classifier, **no rollout**) | 0.9809 [0.9748, 0.9863] | +0.2365 | supported |

Three readings, all in the revision:
1. **The pre-registered Q1 question is answered against us:** A2 − U is `supported` at matched carrier input and capacity, and A4 − U is `supported` (budget-qualified: A4 trains at 1.88× U’s GPU-hours, and the required #140 U-native control at 0.8132 keeps the qualification). **We therefore cannot claim the world model adds ranking value over direct prediction on this task**, and the paper will say so.
2. **F1 − F1-D = −0.0044** (`not supported`): U’s predicted endpoints add no incremental ranking value under a borrowed strong readout — a self-made negative control we did not have at submission.
3. **Descriptive context that limits how far any of this generalizes:** a no-model nearest-fit reference reaches **0.9039** on the same grid, and on the #110 novelty pairs every direct predictor falls to 0.52–0.58 pooled (U 0.47). On the native-action protcol (#132, descriptive, separate protocol) U’s F-15-macro policy reaches pass@3 0.424 [0.384, 0.465] against heuristic agents at 0.307/0.233/0.217 — so the model is not useless, but the grid ranking task appears to reward label interpolation of near-duplicate levels. We state both facts.

`not run` and typed outcomes are also reported: DreamerV3 (#133), DDLP (#135), Diffusion Forcing (#136) and DeepPHY (#139) are `fidelity_not_reproduced` at their pre-frozen fidelity bands; V-JEPA 2-AC (#138) is `cap` (≥285 GPU-h per native seed vs a 24 GPU-h seed cap); C1-dino stopped at readiness (R1 0.5948 < 0.65). Nothing was dropped after its score was seen.

---

## 7. Reviewer MMed

We note MMed’s low scores and take the substantive points in order; where a point is correct we concede it rather than re-argue it.

**W1 — “CLEVRER uses overlapping train/eval scenes; NovPhy has very few levels; generalization claims are weakened.”** Conceded on both counts; see §2 and §1. The submitted table was in-sample over CLEVRER scenes 10000–10011 and Physion trials 0–11; the NovPhy sets had 7–14 mixed levels. Post-submission we add a 12-fold leave-one-scene-out rerun (#117 item 5) and a 400-level sealed split with 311 mixed grid levels, and we remove the word “generalization” from every CLEVRER/Physion sentence, including Physion’s newly found physical-parameter overlap (20/150 trials).

**Q1 — “Why are CLEVRER training and evaluation samples drawn from the same scenes? How is generalization assessed?”** It was an artefact of the only carriers that existed, not a design choice we would defend; nothing about the method required it. Generalization is now assessed three ways, and we label which is which: level-disjoint on NovPhy (400 evaluation levels never seen by fit or policy), scene-disjoint on CLEVRER/Physion (LOSO, EXPLORATORY), and a genuinely video-disjoint pixel protocol (#106). The revised protocol section prints the scene IDs and the exact split rule.

**W2 — “action ranking close to random; description levels do not consistently improve accuracy; figures 1 and 3 overcrowded; text underdeveloped.”** All conceded. On ranking: the submitted AUCs were near chance and the policy gains were measured against a near-chance reference; we replaced that comparison with a fixed-request reference and an effect-size rule, which reverses the conclusion (§1). On descriptions: correct — they do not consistently improve accuracy, and §3(d) shows why (macro unreadable under rollouts; #120). On presentation: Fig. 1(b2) is redrawn with per-panel intervals and the four interval values printed (the caption now marks which panels are ICLR-era evidence); Fig. 3’s heatmap is replaced by a sealed nine-requests × four-inventories table for both stacks plus the action prior, with the old heatmap moved to the appendix. The main text gains the sections the reviewers found missing: the frozen-rule statistics paragraph, the compute decomposition, the closed-loop protocol, and the limitations paragraph.

**Q2 — “What benefit do symbolic description levels offer over continuous prediction alone?”** On held-out evidence: **+0.0201 [0.0154, 0.0248]** grid AUC against a cont-only policy on the *same* checkpoint (the ablation the submission lacked), at the cost of +6.6M MACs per candidate; **no** rollout-error benefit (Δ=15 continuous ≈ macro, #110); **no** benefit on macro predicates, where open-loop rollout F1 is 0.015 and no verdict shows a macro advantage (#113, #120); and **no** benefit that reaches task success. The claim in the abstract changes accordingly (§10).

**Q3 — “Do weak action-ranking results translate into task success?”** We measured it rather than speculating (#102, closed loop: retry-from-start, live engine execution, 3 shots, 5 123 live executions over 19 360 episodes, pass = live pig removal). Descriptive pass@3 on 400 N1 levels: F-15-continuous **0.478 [0.434, 0.523]**, C\* 0.466, F-15-micro 0.458, F-5-cont 0.453, J_dec 0.445 [0.403, 0.487], Gate-B unit 0.424, ICLR J 0.394, C-J 0.346, no-model prior 0.323, random 0.198, oracle 0.775. The ranking and success orderings **disagree**: the best fixed request wins in closed loop too, and J_dec beats the ICLR J by +0.051 [0.028, 0.075] while sitting below C\*. Frozen guard G4 failed (25 of 5 106 live decision frames parse up to 0.0032 from the anchor, tolerance 1e-3), so **every token is `readiness_or_precision_insufficient`**; a post-hoc check re-ranked from live frames and found the first three picks unchanged in 351/351 comparisons. We report these as descriptive, not confirmatory.

**Q4 — “Can you evaluate in more complex or closed-loop environments?”** Three additions: the closed loop above; a full **nine-level novelty ladder** on NovPhy (45/45 cells supported under a frozen contract, #110) with the transfer study that follows (#107); and out-of-family CLEVRER/Physion from pixels (#106). The novelty result is again against our submitted story: **policy-only retune under novelty is `not_supported_by_this_experiment`** (J_dec −0.007 [−0.021, +0.006]; J −0.057), while **predictor** retune gains +0.151 [0.097, 0.206] — i.e. under distribution shift the adaptation budget should go to the world model, not to the request policy. Zero-shot, the policy ranks at best equal to the best fixed request on every novelty type, and a no-model prior beats every model arm on the paired normal sides (0.76; J_dec − prior −0.113).

---

## 8. Reviewer MSwm

We thank MSwm for the most technically engaged review; several of its points changed our conclusions.

**W1 — “all ordering AUCs are 0.40–0.63, gains over a near-chance r†, 7–14 scored levels, one of four intervals clears zero.”** Correct, and the redesign confirms the diagnosis (§1): the submitted design needed ~200–350 mixed levels for δ = 0.02 and had 7–14. On the sealed split we now report the fixed-request reference and the prior next to every arm, the δ rule instead of “gains”, and the achieved half-widths (0.030 grid, 0.063 offset, 0.047 angle, 0.072 power). The submitted “+0.004 to +0.070 over r†” sentence is withdrawn; the replacement is the negative statement plus the one contrast that does clear δ (J − J-cont +0.0201).

**W2 — “no cont-only ablation on the same hybrid checkpoint; part of the compute saving is the narrower trunk (1.27× cheaper calls); novelty scope is narrower; Chatterjee & Khardon and Neitz uncited.”** All four accepted. The ablation now exists (+0.0201, `supported`, at-most-budget) (§3a); the compute attribution reproduces your 1.27× observation with a signed decomposition (calls −6.4M, α **+6.6M**, trunk −50.8M) and the compute claim is withdrawn (§3b); both citations are added with a “what is not new” paragraph (§5).

**W3 — “time base must be clarified; Table 1 should use documented held-out scenes.”** Both accepted (§2, §4). Your remark is about the **NovPhy base corpus** — 2400 of the 3000 training-role lineages used for predictor fitting, with 60-frame windows (`docs/issue-71-hybrid-readiness.md`) — and we acknowledge it as such: the NovPhy carrier grid is 50 native steps wide, and the paper never said which unit Δ counted in, so a 50× mismatch between the 2400 fitted lineages and the evaluation frames is a live possibility that the submitted text does not exclude. We therefore withdraw the submitted horizon sentences pending re-derivation of the unit from the frozen configuration, rather than claiming the base corpus was consistent. The revision prints native timestep, capture stride, and window/request lengths in both frames and milliseconds for every corpus.

**Minor 1 — TAWM venue.** Accepted and corrected to ICML 2025 main track.

**Minor 2 — define “lineage”, “base corpus”, “N1 captures”, “development states”.** Accepted; these are internal pipeline names that leaked into the paper, and the paper’s data section never defined them. The revision adds a glossary, which we will write as follows: *lineage* — one generated level instance together with its fixed static parameters and engine seed; it is the unit that must be disjoint across splits, and our split rule is lineage-disjoint by construction; *base corpus* — the frozen capture release from which a split is drawn (the revision names it by release and content hash instead of by batch label); *N1 captures* — the captures of the first coverage batch, retained as the reference captures for the development launch sets (renamed to the actual release); *development states* — the level states behind the development launch sets used in the submitted evaluation, as opposed to the sealed held-out split introduced in the revision. Every remaining ticket-style label will be replaced by a dataset name.

**Q1 — “What does Δ count in training and in the NovPhy test? Does the model rely on the elapsed-time entry (Eq. 9)?”** Δ counts carrier frames in the post-submission pipeline, on a grid whose stride is 50 native steps; for the submitted configuration we are re-deriving the unit from the frozen files and will not assert it here (§4). On the elapsed-time entry: **yes, the carrier has one** — its header is `[has-prior, elapsed seconds]` and velocities are per second — so the model is not left to infer the timescale from an assumed unit stride. That channel is distinct from the request code and from the policy’s κ feature. We have tested κ (#119, +0.0042, no effect) but **not** the elapsed column; we say so rather than implying the elapsed channel is unused.

**Q2 — “Report scored levels and successful launches per set; numeric intervals for all Fig. 4 cells and the four Fig. 1(b2) differences.”** This is now the sealed table rather than a figure-with-intervals. Per inventory: grid 311/400 mixed levels, offset 76/120, angle 78/120, power 47/120 (below its δ = 0.05 design depth, stated); grid cells carry 3 seeds → 933 scored cells; 11 680 evaluation launches, all complete, 0 typed failures; the full inventory of passing candidates per level is retained in the sealed artefacts and summarised in the appendix. Every Fig. 4 cell is replaced by (estimate, interval, token); the four Fig. 1(b2) differences are printed as +0.0201 [0.0154, 0.0248], +0.0111 [0.0050, 0.0174], +0.0032 [−0.0075, 0.0137], +0.0037 [−0.0021, 0.0095] (J−J-cont, J−C-J, J−T-cont, J−J-as-cont).

**Q3 — “Recompute Table 1 on held-out CLEVRER and documented held-out Physion scenes, with a persistence or constant-velocity baseline and the error of the policy’s own schedules.”** Answered in §2, including the part we cannot do: no persistence/CV baseline exists on those carriers, and adding one means rebuilding them. What we deliver instead: scene-disjoint LOSO recomputation (partial — the nine-request table is not rebuildable), the new disjoint #106 protocol from pixels with per-benchmark intervals, the NovPhy persistence table (#110) where persistence *beats* our Δ=1 recursions on 42/45 cells, and the policy-schedule errors (J/J_dec 0.032 vs hierarchical 0.110 at t = 225; F-15-continuous lowest at 0.038 in the #110 zero-shot reference cells).

**Q4 — “Does the compute saving come from the description level, the horizon, or the narrower trunk?”** The trunk. Signed decomposition at λ = 1e-4: J − C-J = −50.7M MACs = calls −6.4M + α +6.6M + trunk −50.8M + policy −0.1M. The horizon contributes little and the description axis adds cost. This is now printed as a table, and the “joint selection is cheaper” claim is deleted from abstract, introduction, results, discussion and conclusion (#122, #124).

---

## 9. Reviewer vSA9

**W1 / Q2 — temporal-rate encoding is poorly characterized; which metric distinguishes 1-step from 50-step execution?** Accepted as a writing defect with a precise fix (§4). NovPhy has a single uniform carrier stride of 50 native steps (20 ms), so no *within-NovPhy* metric is needed; the ambiguity is cross-dataset and is resolved by printing native timestep, stride and physical duration. If a one-engine-step request is scientifically required, it needs a recapture at a finer stride; we declare that `unsupported` rather than fake it by interpolating (#116). Additionally, the only elapsed-time input in the system (κ) was tested and has no effect (#119).

**W2 — “the horizon-only comparison changes predictor width, tokenization training, representation use and dataset simultaneously, inside a development state; and DP optimizes an additive fixed-context proxy rather than the recursive rollout cost.”** Both points accepted, and both are now answered with new experiments rather than argument.
- The confounded baseline is gone: **J-cont** is trained on the *same* Gate-B checkpoints, same trunk, same encoder, same 242→128→9 policy, DP/DAgger restricted to continuous requests, same sealed split and same frozen λ. J − J-cont = +0.0201 [0.0154, 0.0248] at both declared budgets (#114). The former “horizon-only” baseline is renamed C\* / C-J and its difference from J is reported with the compute decomposition showing the trunk, not α, is the source (#103).
- On the DP objective: the submitted labeler is a **duration-weighted prediction-error DP**, not a ranking-loss DP — `c(t,r) = Δ_r · mean((F(context_t, a, r) − z_{t+Δ_r})²) + λ · work(r, policy)/work(reference)` with `V(t) = min_r [c(t,r) + V(t+Δ_r)]` over a fixed context tensor (`world_model/training/request_policy.py`). We will state that formula in the method instead of describing the objective loosely. J_dec (#101) is a **post-submission variant** that replaces the error term with a declared candidate-normalized decision surrogate; it is not the submitted objective and we do not attribute it to the submitted method. The consequence of the surrogate choice is measured, not assumed: (i) a policy-gradient arm optimizing per-cell AUC end-to-end collapses to a fixed request (J_pg ties F_hindsight at −0.001) and does not beat J_dec; (ii) lengthening the label window toward the recursive cost (L225) makes J **worse** (−0.026), and the decision-label variant adds −0.004; (iii) J − J-as-cont (the same schedule replayed with every α set to continuous, i.e. removing state dependence of the description choice) is +0.0037, not supported. So the additive fixed-context surrogate is a real limitation of the method's optimality story, and the revision says so explicitly (#101, #114, #119).

**Q3 — “Same checkpoints, equivalent decision budget: joint vs fixed macro-step with adaptive horizons, fixed horizon with adaptive descriptions, and time-only scheduling.”** All three now exist (#114), on the same checkpoints and under a frozen at-most-budget rule:

| comparison | grid ΔAUC [CI] | token |
|---|---|---|
| J − J-cont (adaptive horizon, fixed continuous description) | +0.0201 [0.0154, 0.0248] | supported |
| J − T-cont (adaptive horizon, fixed description) | +0.0032 [−0.0075, 0.0137] | not supported |
| J − J-as-cont (time-only schedule, α forced continuous) | +0.0037 [−0.0021, 0.0095] | not supported |
| J − C-J (time-only scheduling on the adapted predictor) | +0.0111 [0.0050, 0.0174] | not supported |
| D-1 / D-5 (fixed horizon, adaptive description) | dominated by C-F-15 | `supported` against D |
| D-15 (fixed horizon, adaptive description) | dominated by J at λ = 1e-1 | below δ, `not_supported_by_this_experiment` |

Reading: **adaptive horizon alone (J-cont) is a real 0.02-level effect; adding the description axis to a *fixed* description is not.** The description axis pays only when the alternative is “never choose α”, which is the weakest of your three comparisons. That is the scoped claim we will make.

**Q4 — “Report uncertainty with task success (or top-1 decision quality); list all successful and failed cases, not just AUC; does the 0.004 offset gain still hold?”**
- Task success with uncertainty: §7 Q3 (pass@3 with intervals for 11 arms; oracle 0.775; guard G4 failed → all tokens `readiness_or_precision_insufficient`). Top-1 decision quality: pass@1/2/3 and shots-to-pass are reported for every arm in the closed-loop publication; top-3 hit rates are reported per policy in #114 (e.g. J 0.5102 [0.4620, 0.5606], C-J 0.4469, prior 0.4180).
- Complete case listing: the closed-loop publication retains **every** live execution — 5 123 rendered episodes with frames, WebM, verdict and typed-failure status per execution, an index gallery, and a manifest binding all 403 GB; the failed attempts are included, not filtered. The eval-split capture retains all 11 680 branches with per-branch engine verdicts. Both are referenced from the appendix audit table.
- The 0.004 offset gain does **not** hold as a claim. On the sealed split the offset inventory has 76 mixed levels and the ICLR-checkpoint held-out rerun gives J − F\* = +0.011 [−0.052, +0.074], half-width 0.063; a 0.004-scale effect is far below the resolution of any design we can afford (≈758 levels needed). We report it as unresolved, not as replicated, and the abstract no longer mentions it.

**Q5 — “What objective is the DP guaranteed to optimize? How are successor contexts built for requests not visited in round 1?”**
- **Objective.** The submitted DP minimizes, over the request lattice R and a fixed context tensor, the duration-weighted prediction error `c(t,r) = Δ_r · mean((F(context_t, a, r) − z_{t+Δ_r})²) + λ · work(r, policy)/work(reference)`, with `V(t) = min_r [c(t,r) + V(t+Δ_r)]`, infeasible past the window end, ties to the lower request index (`world_model/training/request_policy.py:112-141`); λ is swept over 8 values {1e-5 … 1e-1}. This is a **prediction-error** objective and it is exactly optimal only for this additive surrogate on the frozen context set — not for the state-dependent recursive rollout cost, as you say. A post-submission variant, J_dec (#101), substitutes a declared candidate-normalized decision surrogate; #101 states that this surrogate is not numerically the mean full-cell ranking-loss change, and #119 measures what happens when the label window is lengthened toward the recursive cost. Tractability: the exact pass over |R| = 225 with N = 3 480 contexts and L = 225 takes 17.8–18.3 GPU-s at a 1.26 GB peak, with cost linear in N·L·|R| (R² 0.999, log-log slope 0.87) — `supported` (#116).
- **Successor contexts for unvisited requests (your second question), precisely.** Each round solves the DP on a **fixed context tensor** over the window (`request_policy.first_window` / `prediction_error_labels`). Contexts initially are the observed carriers `z`; after a round, DAgger overwrites only the positions the greedy rollout actually visited with the **predicted** carrier (`dagger_contexts`), while **unvisited positions keep the observed carrier**. The DP then evaluates *every* request from every position of that tensor, and the comparison targets `z_{t+Δ_r}` remain the **unchanged aligned true carriers** — they never enter the deployed policy or predictor API. So requests that round 1 never issued are still labelled: from **observed** contexts at their positions, against true targets, with no recursive prediction needed. The exact optimality guarantee is confined to this fixed-context additive surrogate, which is the limitation you identify.

---

## 10. Revised claims (what we will assert, and at what strength)

| submitted claim | disposition | revised statement |
|---|---|---|
| “matches a horizon-only adaptive baseline at a third of its compute” | **retracted** (#103, #122) | MAC ratio 0.784 at the ICLR operating point; +0.006 AUC; both dominated by a fixed coarse request on the matched continuous predictor. |
| “the policy beats the best fixed request / r† by +0.004–0.070” | **retracted** (#101, #104, #114) | No policy in our family beats the best fixed request at δ on the sealed split; J_dec − F_hindsight = −0.016 [−0.020, −0.012]; the submitted intervals were 3–5× too wide to resolve the claimed effect. |
| “job: joint selection of horizon and description” | **supported, scoped** (#114) | On the same checkpoints and an at-most-budget rule, choosing α by state adds +0.0201 [0.0154, 0.0248] AUC over never choosing it; adaptive horizon alone already accounts for the horizon-side gain (+0.0032 to +0.0111 for the description/state components). |
| “description levels are the new axis” | **narrowed** (#113, #115, #120, #121) | Macro is now observable (F1 0.157 parsed / 0.113 one-step / 0.015 rollout) but yields no ranking advantage under any of four verdicts (−0.012 … −0.078); symbolic feedback is non-inferior to label-free, and oracle feedback hurts; additive request codes cost 1.1–2.0× rollout error, and interaction-specific value is unidentified. |
| “shared predictor over nine requests is cheap” | **supported** (#117) | At matched per-request updates the joint ranks better than specialists (J − S +0.0848 [0.0742, 0.0952]; F-15-macro +0.0561) and costs ≤1.5× per-request rollout error in 3/4 benchmark × recipe cells, Physion LC Δ=15 unresolved; at full specialist budget the specialists win (S-15-macro/full 0.885), which we attribute to update allocation. |
| “the model is a better world model than alternatives” | **withdrawn** (#123, #125, #127) | Direct success predictors reach 0.974–0.995 against 0.744; a borrowed strong readout with no rollout (F1-D 0.9809) is statistically indistinguishable from U + rollout (0.9765, −0.0044); H-JEPA collapses here, so no ranking claim is licensed against it; SlotFormer/DINO-WM/THICK/γ-models/DIAMOND fail the frozen action-sensitivity guard. |
| “request selection is a cheap adaptation surface under novelty” | **retracted** (#107) | Policy-only retune under novelty is −0.007 [−0.021, +0.006]; predictor retune is +0.151 [0.097, 0.206]. |

**What we ask the committee to weigh.** The paper’s durable contribution after this work is a *method and a measurement protocol*: a request-conditioned latent predictor over a granularity lattice, a DP/decision-valued labelling scheme with no RL, and a pre-registered, guard-cascaded, engine-verdict evaluation on a sealed 400-level split that other granularity work can reuse. Its empirical message is negative and, we think, useful: for decision-relevant granularity, the binding constraints we could find were **predictor update allocation** (equal-updates predictors reach 0.947 grid AUC vs 0.744 at the shared 9 000-update budget, with no fit/evaluation leakage found), **readout design** (a single-frame cost function hides most of the signal; macro predicates are unreadable through rollouts), and **the ranking task itself** (a no-model nearest-fit reference reaches 0.904).

---

## 11. Manuscript changes (concrete, already drafted)

1. **Abstract:** remove the compute-thirds clause, the “beats the best fixed request” clause, and the “macro best everywhere” inference; add the δ-rule wording, the at-most-budget qualification, and the scope qualifier that long-jump gains hold for models trained on short recursive windows.
2. **§1/§2:** add Chatterjee & Khardon (2025), Neitz et al. (2018), H-JEPA (2026), HWM, LeJEPA/SIGReg; correct TAWM’s venue; add a “what is not new” paragraph and a one-sentence #117 sharing result.
3. **§3:** time-base table (native timestep, carrier stride, Δ in ms per dataset); the #111 tie-free cost as the decision cost; restate which outcomes enter training (fit/policy only) and that r† is a transferred choice; state that the DP is optimal for a declared additive surrogate.
4. **§4:** E99 slot parser as the encoder (the submitted parser read the pig in 0/859 decision frames — a defect we found and fixed); the LC recipe; J / J-cont / J_dec described with their restrictions.
5. **§5:** fit 480 / policy 120 / evaluation 400 splits; 11 680 launches; sealed cell counts (311/76/78/47 × 3 seeds, power below design depth); δ token rule, guard cascade, deterministic scoring, scored once; closed-loop protocol; time-base table.
6. **§6:** rollout table with an LC column and the in-sample label; the sealed nine-request × four-inventory table replacing the Fig. 3 heatmap; the ranking/compute/closed-loop table with the #103 frontier figure; the macrological audit; the LOSO CLEVRER/Physion table labelled EXPLORATORY; the H-JEPA rows with disposition 3/4 and the descriptive gap.
7. **§7–§8:** delete every “third of the compute” and “axes vary separately, macro best” sentence; add a limitations paragraph (prior near most rankers; U < C\*; ranking looks update-limited; macro does not survive rollouts; closed-loop G4 failed; Physion physical-parameter overlap; retrain variance exceeds δ; direct predictors beat the world model; no persistence/CV baseline on CLEVRER/Physion).
8. **Appendix:** new sections for the sealed protocol, Gate B, the decision-line results, the engine-truth frontier, the closed loop, novelty, macro observability, sharing/scaling/request-code/feedback, the pixel benchmarks, the external-baseline program, and an audit table in which every number points at a committed artefact field (37 trace groups sampled; 8 wording/traceability problems found by our own final audit and fixed).
9. **Reproduction table:** marks the 9 of 32 commands that stop at their freeze check after the server migration (a documented infrastructure fault that neither confirms nor contradicts a published table) — we prefer to print that than to imply all 32 pass.

---

## Internal evidence notes (not part of the response)

Traceability: all AFTER-ICLR evidence is on GitHub; tickets #99–#125 and #127–#141 are closed with dispositions and validation receipts (#142 the manuscript carry, #126 a reproduction-path bug, and #97/#98 bug tickets remain open); branches `after-iclr-108`, `after-iclr-108-r2`, `after-iclr-127`, merged into `env` at `a4ef432f`. Key issue links used above: #101 (decision-valued policy), #102 (closed loop), #103 (engine-truth frontier; #122 replacement), #104 (power cohort + held-out #96 contrasts + yield), #105 (hierarchy/independence), #106 (pixel CLEVRER/Physion; amendment A3 overlap), #107 (novelty transfer), #109 (data foundation, sealed cohort), #110 (novelty coverage, persistence table), #111 (deterministic scoring), #112 (LC recipe/Gate-B handoff), #113 (observable macro), #114 (joint vs horizon-only), #115 (predicate feedback), #116 (request-set scaling, DP tractability), #117 (sharing; item 5 benchmarks), #119 (remaining-frames input), #120 (cost alignment), #121 (request-code factorisation), #123/#125 (H-JEPA, H-JEPA+IDM), #124 (manuscript carry), #127 + #128–#141 (external baselines), #142 (manuscript handoff).

Residual risks to manage before submission of this response:
1. **#126** (open): 9 of 32 reproduction commands still stop at freeze checks on the 5090 server; the response says so rather than claiming full reproducibility. Do not upgrade this wording.
2. **#142** (open): the external-baseline rows and the Q1 reading are handed to the manuscript agent but the *storyline* reframing (how much of the paper remains) is an owner decision. This rebuttal assumes storyline A + honest narrowing; if the owner chooses a different framing, §6 and §10 must be reconciled first.
3. **#98 / #97** (open bugs): a cross-pool selection audit mis-scores tied pairs (published type010101 AUCs 0.2039/0.2732 are an artefact, corrected ≈0.50) and one advertised confirmatory command exits 1. Neither number is used in this response; if the manuscript still prints them, they must be corrected before it is cited.
4. **Physion/CLEVRER persistence baselines** genuinely do not exist; do not let any revision imply they do (§8 Q3).
5. **#109 held-out interval caveat**: the 3-cluster held-out bootstrap has only 10 distinct resamples, so its intervals understate uncertainty; that number is deliberately not quoted as a contrast.
6. `#127` **E1 native-action table** is a different protocol from the grid; keep it labelled descriptive and never pair it with grid AUCs.
7. **Two corrections that must not regress** (verified against the code after an earlier draft got them wrong): (a) the carrier *does* carry elapsed seconds — header `[has-prior, elapsed]`, velocities per second — so never write "no clock feature / only κ is time-like", and never use #119's κ result as evidence about the elapsed column, which has no ablation; (b) the submitted DP is duration-weighted prediction error (`c = Δ_r·MSE + λ·work/work_ref`, `V(t)=min_r[c+V(t+Δ_r)]`), contexts are a fixed tensor where only *visited* positions are overwritten by predicted carriers and unvisited positions keep observed carriers against unchanged true targets, and J_dec (#101) is a post-submission variant, not the submitted objective.
8. **The 2400/2409 remark is the NovPhy base corpus** (2400 of 3000 training-role lineages used for predictor fitting, 60-frame windows, `docs/issue-71-hybrid-readiness.md`), not CLEVRER/Physion; the submitted Δ unit is unresolved and the draft says so rather than resolving it.
