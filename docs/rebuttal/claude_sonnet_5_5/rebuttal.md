# Author Response — ICLR 2027 Submission #6352
## "Granularity Is a Decision: Joint Horizon–Description Selection in Latent World Models"

Reviewers: **MMed** (1, clear rejection), **MSwm** (2, weak rejection), **vSA9** (1, clear rejection).

We thank all three reviewers. Between submission and this response we ran a pre-registered programme of follow-up experiments (GitHub tickets #99–#141 of `Sino-Huang/NovPhy`, referred to as `#N` below). Several results are **negative for claims in the submitted paper**. We say so plainly, narrow those claims, and state what the evidence still supports.

**Evidence labels used throughout**

- **[Completed]**: a finished, frozen-protocol experiment or audit run after submission. These were not in the submitted PDF. Each number comes with its interval and ticket.
- **[Descriptive]**: reported but underpowered or exploratory. It carries no confirmatory weight.
- **[Planned]**: a manuscript change we commit to. It is *not* yet in the submission.
- **[Unresolved]**: a limitation we cannot close with current evidence.

Intervals are level- or member-clustered bootstrap intervals, treated as descriptive. "δ-rule" is our pre-registered labelling rule (δ = 0.02 AUC): `supported` iff mean ≥ δ and lower bound > 0. The tokens `not_supported_by_this_experiment` and `readiness_or_precision_insufficient` are quoted as published. Ticket URLs: `https://github.com/Sino-Huang/NovPhy/issues/<N>`.

---

# Part 0. Shared response

## 0.1 What the follow-up evidence says, in one page

The reviewers' central worries were correct, and the follow-up work makes several of them stronger:

1. **The submitted NovPhy ranking evidence was weak, and its parser was part of the reason.** The submitted perception stack read the pig as present in **0 of 859** decision frames (the engine has it in 859/859) (#99). Near-chance action ranking was therefore expected. Perception was repaired post-submission (pig accuracy 0.000 → 1.000, #99). Dynamics became the binding stage.
2. **On a larger, level-disjoint sealed split, no request-selection policy beat the best fixed request or the matched continuous baseline** (#101, #105, #107, #114, #116, #119, #121). This covers decision-valued labels, policy gradient, joint versus tied axes, state dependence, and an unsaturated remaining-frames input. The submitted policy's apparent gains over a fixed request did not replicate: ICLR-checkpoint J − F\* on the sealed grid is **+0.013 [−0.016, +0.043]** (#104).
3. **The submitted compute claim ("a third of the compute of a horizon-only adaptive baseline") is not supported.** J costs 0.784× the MACs of the horizon-only policy, against a pre-declared bar of ≤ 0.40×. The saving that exists comes from the narrower trunk, not from description selection (#103, #114). We withdraw the sentence.
4. **The macro description level shows no demonstrated benefit.** The macro head read nothing on the redefined predicates (F1 0.000). A temporal head reaches F1 0.157 on parsed carriers but 0.015 in open-loop rollout. Macro has no ranking advantage under any cost we tried (#113, #115, #120).
5. **Against direct predictors, the world model does not add ranking value on this benchmark.** Direct success predictors score 0.97–0.99 grid AUC against 0.744 for our Gate-B model. A no-model nearest-fit lookup scores 0.904. The sealed split rewards interpolation among near-duplicate levels. Under novelty, every ranker falls to roughly chance (#127–#141).
6. **What survives:**
   - A single shared request-conditioned predictor does not cost per-request ranking. At matched per-request updates it ranks better than single-request specialists (J − S **+0.085 [+0.074, +0.095]**, #117).
   - Adding description options to the policy's request set improves over horizon-only selection on the same hybrid checkpoint: J − J-cont **+0.020 [+0.015, +0.025]** (#114). The qualifications are stated in §0.4.
   - The finding that the best granularity depends on the downstream use is reproduced: the best fixed request changes between inventories on both the submission checkpoint and the Gate-B stack.
   - The DP labeller is tractable at |R| = 225 (#116).
   - A reproducible, sealed, engine-verdicted evaluation protocol with guards, controls, and negative results.

**Our recommended scope for the revised paper.** "Granularity is a decision" stands as a *problem formulation and a diagnostic framework*. It does not stand as a *method that wins* on prediction accuracy, decision quality, or compute. We will say that in the abstract and introduction.

## 0.2 Common concern A — held-out evaluation and splits
*(MMed W1/Q1; MSwm W3/Q3; vSA9 Q1)*

**Direct answer.** The reviewers are right about the submitted Table 1 and the submitted NovPhy decision sets.

- **CLEVRER and Physion-Dominoes in Table 1.** The annotation-carrier cohorts hold **12 scenes per benchmark (480 windows of 61 frames, null action)**. The reported rows are the in-sample, fold-"all" setting. Fit and evaluation scenes are the same scenes (#117, item 5). These rows measure in-sample rollout fit and cannot support a generalization claim. The Physion-Dominoes data and split were not documented in the submission.
- **NovPhy.** The submitted launch sets were built from N1 captures. Of the members used, 8 were predictor-training lineages and 3 were controller lineages. Only 4 (007, 008, 014, 016) were held out (#109). Each launch set had 7–14 mixed-verdict members.

**Completed evidence after submission.**

1. **Sealed, level-disjoint NovPhy split** [Completed] (#109, #104, #108).
   - Fit / policy / evaluation = 480 / 120 / 400 levels over two level families (rolling, sliding), level-disjoint, with fresh seeds checked against 83 prior-exposure sources. Of 49,560 scheduled branches in the whole campaign, 49,508 were captured; the evaluation stream has 11,680 branches, all complete, 0 typed failures.
   - Mixed-verdict evaluation levels: **grid 311, offset 76, angle 78, power 47**. Power is below its pre-declared design depth (53).
   - The 4-lineage held-out re-score of the submitted contrasts showed no exposure optimism on grid: held-out minus exposed J − F\* = +0.0005 (#109). Intervals there come from 3 clusters and understate uncertainty.
   - **Scope caveat.** "Held out" means level-disjoint, **not template-disjoint**. The cohort has two generator templates, and 100 of 120 policy levels sit closer to a fit level than to another policy level (#128). The parameter-free nearest-fit lookup reaches 0.904 on the evaluation grid. Generalization claims are therefore scoped to new instances of the same two families, and we say so [Planned].
2. **Leave-one-scene-out re-run of the CLEVRER / Physion rollout experiment** [Descriptive] (#117 item 5).
   - 12 folds × 3 seeds, hybrid joint model under the submission's training recipe, position MSE at t = 120, scene-clustered intervals.
   - Held-out versus in-sample (published hybrid): CLEVRER Δ = 15 continuous **0.091 [0.060, 0.128] vs 0.017**; Δ = 5 0.143 [0.103, 0.186] vs 0.044; Δ = 1 16.3 [9.1, 26.4] vs 2.98.
   - Physion-Dominoes: Δ = 15 0.003 [0.002, 0.006] vs 0.001; Δ = 5 0.038 [0.017, 0.076] vs 0.016; Δ = 1 360 [292, 436] vs 167.
   - So held-out error is roughly 2–5× the in-sample error, and **the ordering "Δ = 15 < Δ = 5 < Δ = 1" is preserved** under the submission recipe. Under the long-horizon (LC) recipe the three horizons become close: CLEVRER 0.081 / 0.072 / 0.067 for Δ = 1 / 5 / 15, with overlapping intervals.
   - **Limit.** This is a 12-scene pool, exploratory, and it recomputes the *hybrid joint model's* error. It does not recompute Table 1's hybrid-versus-continuous-only contrast.
3. **CLEVRER and Physion from pixels on the official splits** [Completed] (#106).
   - A pixel-trained slot carrier (no annotation access at test time) with an official-split design.
   - CLEVRER: 6,285 train / 3,132 validation videos after a pre-declared 37.22 % exclusion. The scene sets are disjoint (guard G1).
   - Physion-Dominoes: the official `dynamics` / `readout` / `testing` split, with 150 `redyellow` test trials. No full static parameter set is shared across splits. **However**, excluding appearance-only colour, **20 of 150 test trials share every physical parameter with a fitting trial, and 4 are near-identical**. The official split therefore measures transfer to new colours, **not** to independent physical realisations.
   - Rollout error is lowest at Δ = 15: CLEVRER (t = 120) 0.0974–0.0996 vs 0.163 at Δ = 1; Physion (t = 90) 0.0143–0.0146 vs 0.0430.

**What remains unresolved.** [Unresolved] A Table-1-format recomputation (hybrid versus continuous-only, annotation carriers) on a documented, disjoint scene set has not been run. Building such a set needs new carriers beyond the 12-scene cohorts. A persistence or constant-velocity baseline on CLEVRER and Physion has not been run either. On NovPhy, persistence (anchor held fixed) is reported in #110 (§MSwm Q3).

**Revision** [Planned]. Table 1 will be relabelled "in-sample rollout fit" and moved beside the leave-one-scene-out and pixel-benchmark results. We will document the Physion-Dominoes data and split, and the near-duplicate qualification. We will not claim generalization from Table 1.

## 0.3 Common concern B — weak ranking and the link to task success
*(MMed W2/Q3/Q4; MSwm W1; vSA9 Q4)*

**Direct answer.** Yes, the submitted ranking numbers were near chance, and the reference they were compared against (the best fixed request, r†) was also near chance. The reviewers' reading of Fig. 4 and Fig. 1(b2) is accurate.

**Numbers behind the submitted figures** [Completed re-extraction of the retained #96 artefact]. Fig. 1(b2) is the policy-minus-fixed-request difference J − F\* on the four launch sets (member-clustered, descriptive):

| launch set | scored (mixed) levels / cells | successful launches in those levels | J − F\* (95 % interval) |
|---|---|---|---|
| grid | 14 / 42 | 17 (11 reached within 225 frames) | +0.0700 [−0.0057, +0.1437] |
| offset | 9 / 27 | 9 (3) | +0.0037 [−0.0796, +0.0926] |
| angle | 7 / 21 | 7 (5) | +0.0595 [+0.0079, +0.1131] |
| power | 8 / 24 | 8 (4) | +0.0273 [−0.0691, +0.1281] |

Only angle clears zero, as MSwm noted. G4 (ties) and G6 (precision) failed on grid, so the primary token was `readiness_or_precision_insufficient`. Per-cell Fig. 4 values with intervals are in the Appendix at the end of this response.

**Completed re-test on the sealed split** [Completed] (#104; faithful rerun of all 18 submitted arms on the ICLR checkpoint, tie-free cost, 311 / 76 / 78 / 47 levels × 3 seeds):

| launch set | J − F\* (95 % interval) | half-width | pre-declared precision target |
|---|---|---|---|
| grid | **+0.013 [−0.016, +0.043]** | 0.030 | ≤ 0.02, **missed** |
| offset | +0.011 [−0.052, +0.074] | 0.063 | missed |
| angle | **−0.160 [−0.206, −0.113]** | 0.047 | missed |
| power | −0.262 [−0.336, −0.192] | 0.072 | missed |

The +0.070 grid gain and the +0.060 angle gain from the submission **did not replicate** (replication on offset / angle / power: 0 of 3). On sealed grid levels the ICLR J ranks at AUC 0.555 [0.539, 0.571], F\* at 0.542, and the hindsight-best fixed request (F-15-continuous) at 0.639. All guards except G6 pass, so every contrast token is `readiness_or_precision_insufficient`. Precision is the problem: the held-out between-level variance exceeds the variance the power analysis assumed. The point estimates do not support a policy advantage.

**The offset-set 0.004 question (vSA9 Q4).** The offset estimate on sealed levels is +0.011 [−0.052, +0.074]. It has the same sign and is equally indistinguishable from zero. It does not "still hold" as an effect.

**Model-quality evidence on the repaired stack** [Completed] (Gate B, #108; sealed grid, 311 cells × 3 seeds):

- The Gate-B unit (hybrid, E99 encoder, long-horizon + decision-contrastive recipe) scores **0.7444 [0.7269, 0.7620]**. Per seed: 0.7141, 0.8011, 0.7181.
- The matched pure-continuous model C\* scores **0.7834 [0.7676, 0.7992]**. Unit − C\* is **−0.0390 [−0.0442, −0.0338]**, so the hybrid is *worse* than the matched continuous model on grid, offset and angle (power ties).
- The no-model action prior (success frequency of the identical action in other fit levels) scores **0.7203 [0.6965, 0.7442]**. Unit − prior is +0.0241 [−0.0008, +0.0487]. The anchor does carry information: right-anchor minus wrong-anchor is **+0.1322 [+0.1216, +0.1430]**.
- Gate B certifies the training *recipe's* ranking, not a hybrid advantage.

**Does ranking translate to task success?** [Descriptive] (#102). Closed loop: single-bird levels reset to start for each attempt; a model ranks the frozen 16-launch grid and the best remaining candidate is executed live in the engine. A level is passed by pig removal, with up to 3 shots (400 sealed levels × 3 seeds):

| arm | pass@3 (95 % interval) |
|---|---|
| F-15-continuous | 0.478 [0.434, 0.523] |
| C\* | 0.466 [0.427, 0.505] |
| J_dec (decision-valued policy) | 0.445 [0.403, 0.487] |
| Gate-B unit | 0.424 [0.384, 0.465] |
| submitted-style J | 0.394 [0.352, 0.438] |
| C-J (horizon-only) | 0.346 [0.315, 0.378] |
| no-model prior | 0.323 [0.278, 0.370] |
| random | 0.198 [0.172, 0.223] |
| oracle over recorded candidates | 0.775 |

pass@1 for the Gate-B unit is 0.151 [0.131, 0.172], against prior 0.140 and random 0.068 (#127 summary of #102). Models clearly beat random and sit above the prior at pass@3. The best-performing arm is a *fixed* request, and the adaptive policies do not beat it. **Every contrast token is `readiness_or_precision_insufficient`**: frozen guard G4 failed because 25 of 5,106 live decision frames drifted up to 0.0032 against a 1e-3 tolerance. A post-hoc re-rank from the live frames left the first three picks unchanged in 351/351 comparisons. We therefore report these as descriptive and make **no task-success claim for adaptive granularity**. The ICLR-style J − C-J difference (+0.048 [+0.028, +0.069]) is descriptive only. Because 5,123 live executions were shared across 19,360 episodes, the evaluation covers the grid inventory under a retry-from-start protocol on single-bird levels. It is not sequential, multi-bird gameplay.

**Revision** [Planned]. The abstract and results will state that ranking is near chance for the submission checkpoint, that the matched continuous model ranks at least as well as the hybrid on the sealed split, and that closed-loop evidence is descriptive.

## 0.4 Common concern C — what the description axis buys, and where the compute saving comes from
*(MMed Q2; MSwm W2/Q4; vSA9 W2/Q3)*

All numbers below are on the same Gate-B hybrid checkpoints, the same policy pool, and the sealed evaluation split [Completed] unless noted.

**Direct answers.**

1. **A continuous-only policy on the same hybrid checkpoint now exists** (the ablation MSwm asked for) (#114). J-cont uses the same 242→128→9 policy and recipe as J, with the six non-continuous logits masked and DP/DAgger restricted to continuous requests.

   | contrast (grid AUC, 311 levels × 3 seeds) | estimate [95 %] | frozen token |
   |---|---|---|
   | J − J-cont, at J's MAC budget | +0.0201 [+0.0154, +0.0248] | supported |
   | J − J-cont, at C-J's MAC budget | +0.0201 [+0.0154, +0.0248] | supported |
   | J − J-as-cont (J's schedule replayed with α forced to continuous) | +0.0037 [−0.0021, +0.0095] | not_supported |
   | J − T-continuous (adaptive Δ, α fixed continuous) | +0.0032 [−0.0075, +0.0137] | not_supported |
   | J − C-J (horizon-only on the pure-continuous predictor) | +0.0111 [+0.0050, +0.0174] | not_supported (< δ) |

   The first row is `supported` but **qualified**: J-cont cannot reach either budget in the frozen sweep and is clamped at its own 122.7M-MAC endpoint, so this is an *at-most-budget* comparison, not equal expenditure. Both budgets reuse the same endpoint pair and are not independent replications. J also spent **61.2M more MACs per candidate** than J-cont (184.0M vs 122.7M). The decomposition is +53.5M from more calls, +6.6M from per-call α surcharge, and +1.2M from policy calls. When J's own schedule is replayed with every α set to continuous, the AUC difference is only +0.0037. So the +0.020 reflects the horizon schedule that the hybrid policy chooses when description options exist. It is **not** evidence that *selecting* α per step adds decision value, and we will not word it that way.
2. **Where the compute saving comes from (MSwm Q4).** Over C-J (λ = 1e-4), J uses **−50.7M** MACs per candidate. The terms are: trunk **−50.8M**, number of calls −6.4M, per-call α **+6.6M**, policy −0.1M (#114). **The saving is the narrower trunk. Description selection adds cost.**
3. **The submitted compute sentence is withdrawn** (#103). "Matches a horizon-only adaptive baseline at a third of its compute" was tested at the submitted operating point on 311 sealed levels × 3 seeds:

   | | J | C-J | ratio |
   |---|---|---|---|
   | linear MACs per candidate-state | 184.0M | 234.7M | **0.784** (bar: ≤ 0.40) |
   | wall seconds per state (confirmatory, frozen seed) | — | — | **16.74** |
   | tie-free AUC | 0.706 [0.685, 0.727] | 0.700 [0.683, 0.717] | +0.006 [0.000, 0.012] |

   The token is `not_supported_by_this_experiment`. A three-seed re-timing (post hoc, descriptive) gives a wall ratio of 1.63. J is slower in wall time either way. The earlier count-cost proxy behind the submitted sentence was construct-invalid: `continuous_adaptive` tied all candidates in 196 / 73 / 61 of 200 states, and one seed's hybrid policy used (1, macro) for 203,688 of 203,690 steps (#103 background).
4. **Neither adaptive policy is on the engine-truth compute–quality frontier** (#103). On the grid, the Pareto set under AUC × MACs is: the no-model prior (0 MACs, 0.720), F-15-continuous (21.5M, 0.762), J at λ = 0.1 (22.1M, 0.770), and C-F-15 (27.8M, 0.783). C-F-15 dominates J, C-J, J-cont, and every Δ = 1 / Δ = 5 request. Differences among the three frontier model points are small. J(λ = 0.1) − F-15-continuous is +0.008 [0.005, 0.010], below δ.
5. **Is the description axis valuable?** The submitted claim "additional description levels improve accuracy" was never consistent (MMed W2). Post-submission:
   - *Micro* is readable (edge F1 0.989 for the repaired encoder, #100). A label-free micro bottleneck is non-inferior to engine-labelled predicates, (J−T)\_free − (J−T)\_ref = −0.003 [−0.015, +0.008] (#115). But the reference's own description-axis value is only +0.003 [−0.007, +0.014]: nothing ≥ δ was there to preserve. Fixed micro requests fall 0.145 on offset and 0.106 on angle without labels (descriptive). Removing predicate feedback costs nothing; oracle engine labels hurt (fixed micro requests −0.345).
   - *Macro* has no demonstrated value. The Gate-B macro head predicts at most one positive frame per predicate and Δ (F1 0.000). A redesigned temporal head reaches held-out F1 **0.157 [0.151, 0.164]** on parsed carriers, **0.113** one-step predicted, and **0.015** in open-loop rollout, far below the 0.7 readout bar (#113). Under the pre-declared cost-alignment test, macro − continuous is **−0.012 [−0.019, −0.004]** on the Gate-B predictor and negative under every alternative cost (−0.024 to −0.078). The one positive reading (+0.022 on a predictor retrained on the new predicates under the original cost) vanishes or reverses under the least-aligned verdict (#120).
   - *Independent axes versus tied axes*: J − J-diag −0.0068 [−0.0164, +0.0026]; J − J-monotone +0.0028 [+0.0014, +0.0043]; J − best hierarchical level −0.0006 [−0.0188, +0.0170]; all `not_supported_by_this_experiment` (#105).
   - *Fused versus additive request code*: the additive code is not within 1.5× per-request rollout error (F-5-continuous 1.84 [1.79, 1.90]); fused − additive is +0.050 grid AUC, but the contrast changes the trained parameterisation as well as interaction capacity, so interaction-specific value is unidentified (#121).
6. **Larger or continuous request sets** do not improve ranking *per se*. Gains appear only at equal updates per request, where the same predictors' fixed requests also rise (+0.100 / +0.237), so the gain tracks training updates, not the request set (#116). The DP labeller remains tractable at |R| = 225 (exact pass 17.8–18.3 GPU-s, 1.26 GB peak, cost linear in N·L·|R|, R² 0.999).
7. **Control matrix requested by vSA9 Q3**, same checkpoint and policy recipe, λ = 1e-4, DESCRIPTIVE, unmatched MACs (#114; J 0.7057 [0.6846, 0.7267] on grid at 184.0M MACs):

   | scheduler | what is fixed / adaptive | grid AUC | J − scheduler |
   |---|---|---|---|
   | T-continuous | fixed continuous description, adaptive Δ (time-only scheduling) | 0.7025 [0.6813, 0.7238] | +0.0032 [−0.0075, +0.0137] |
   | T-micro | fixed micro, adaptive Δ | 0.7044 | +0.0014 [−0.0086, +0.0112] |
   | T-macro | fixed macro, adaptive Δ | 0.6776 [0.6555, 0.7002] | +0.0281 [+0.0227, +0.0337] |
   | D-1 | fixed Δ = 1, adaptive description | 0.6851 | +0.0207 [+0.0173, +0.0241] |
   | D-5 | fixed Δ = 5, adaptive description | 0.7378 | −0.0320 [−0.0416, −0.0230] |
   | D-15 | fixed Δ = 15, adaptive description | 0.7619 [0.7433, 0.7802] | −0.0562 [−0.0685, −0.0441] |
   | F-15-continuous | one fixed request | 0.762 | −0.057 (J − F_hindsight) |

   Reading: adaptive-Δ with any fixed description (except macro) matches J within δ. A fixed Δ = 15 request with adaptive description beats J by 0.056. **Joint selection is not beneficial relative to these simpler schedules.** vSA9's "fixed macro-step size with adaptive horizons" is read here as T-macro (fixed macro description). If vSA9 meant a fixed *Δ = 15* with adaptive descriptions, the corresponding row is D-15.

**Novelty and related work (MSwm W2).**

- We agree that choosing the horizon per step is prior art. We have verified Chatterjee & Khardon (NeurIPS 2025, *Improving planning and MBRL with temporally-extended actions*, where the planner treats action duration as an optimisation variable) and Neitz et al. (NeurIPS 2018, *Adaptive Skip Intervals*, where the model chooses its own sampling rate). Both belong in related work, and TAWM is an ICML 2025 main-track paper (minor point 1); we will correct it. [Planned]
- The residual novelty is the **description axis, the joint choice, and a single predictor serving all requests**. The post-submission evidence supports only the *shared-predictor* part (J − S +0.085 [+0.074, +0.095], #117) and a qualified at-most-budget description-axis result (above). It does **not** support a benefit from jointness.
- Adaptive-horizon baselines were also ported post-submission (#78 ports, scored in #105): TAWM-15 0.610, VLWM-15 0.642, TAWM-adaptive 0.450 on the sealed grid. TAWM and VLWM are continuous-only ports. These are descriptive, and we make no superiority claim from them.
- End-to-end hierarchical JEPA (H-JEPA, arXiv 2610.06805) was reproduced on its own benchmark (success 40.0 / 83.3 / 96.0 vs the paper's 40.7 / 81.3 / 96.0 for LeWM / H-JEPA-2 / H-JEPA-3). On NovPhy, H-JEPA, LeWM and HWM collapse to launch-insensitive latents and rank at chance (0.4997, 0.4896; with the authors' own inverse-dynamics remedy 0.4948 and 0.4900) (#123, #125). The pre-registered rule licenses **no ranking claim in either direction**. We do not claim to beat H-JEPA.

## 0.5 Common concern D — the time base
*(MSwm W3/Q1; vSA9 W1/Q2)*

**Direct answer.** The reviewers are right that this was poorly characterised. We give a scoped acknowledgment and a status report. We do not claim a resolution.

- **Two NovPhy data generations feed the submitted predictor.** The base corpus is a frozen set of 3,000 training-role lineages from an earlier aligned player. 2,400 fit the predictor and 600 supervised the controller. Windows are 60 *fixed-step* windows with stride-one frames (#71 recipe). The N1 captures use the native player: native timestep 0.0004 s, RGB every **50 native steps**, up to 30,000 steps per shot. One N1 carrier frame is 50 native steps (#113; `docs/issue-76-native-time-capture-amendment.md`). The repository also records an older "1/5/15-step" versus newer "50/250/750-native-step" horizon convention (`docs/issue-76-native-findings.md`). The submission's wording ("engine steps", "frames") did not separate these, and the paper did not say how the two generations relate. We will state the unit and seconds-per-frame for each corpus. [Planned]
- **Three distinct time-like signals exist and should not be conflated.**
  1. *The request code.* The request (Δ, α) enters the predictor through a conditioner on ϕ(Δ) (16 sinusoids) and a description embedding. Δ counts **carrier frames**; the code carries no physical unit.
  2. *The carrier's elapsed-time entry (Eq. 9).* The 236-value carrier begins with `[has-prior, elapsed seconds between prior and current frame]`, and per-slot motion is the centre difference divided by that elapsed time (`world_model/data/deployment_temporal.py`, `world_model/planning/scoring_harness.py`). So the carrier does contain an elapsed-seconds channel and per-second velocities.
  3. *The policy's remaining-frames input κ.* It is clamped at 60 in the submission (separate from the carrier channel).
- **Does the model rely on the elapsed-time entry?** [Unresolved] We have **not** run an ablation of that carrier channel (for example zeroing or rescaling it at inference). The post-submission κ study tests only the *policy* input: an unsaturated κ changes J − F_hindsight by +0.0042 [−0.0005, +0.0088], `not_supported_by_this_experiment` (#119). It says nothing about the carrier's elapsed channel.
- **What the post-submission evidence does establish.**
  - The Gate-B stack (E99 encoder + the long-horizon, decision-contrastive recipe) is trained on the #109 fit split and evaluated on the #109 evaluation split. Both were captured by the same pipeline with the same 50-native-step frames (#104, #109, #112). **For the primary stack there is no train/test time-base shift.**
  - The ICLR checkpoint, trained on the base corpus and N1 lineages and scored at the 50-step N1 frame rate on 400 sealed levels, ranks at AUC 0.555 (§0.3). That is consistent with *no* usable transfer across the time base, and also with the parser defect (pig read in 0/859 frames). **The data cannot separate these explanations**, so we make no claim about rate transferability.
- **Revision** [Planned]. Add a table of corpus × native-steps-per-frame × seconds-per-frame; state which signals carry time; remove any rate-transfer implication; and add the elapsed-channel ablation as an inference-time perturbation study. The ablation is not run yet.

## 0.6 Common concern E — what the DP optimises
*(vSA9 W2/Q5)*

The submitted controller labels come from a backward DP over a window (`world_model/training/request_policy.py`, "ICLR eq. dp"):

- cost of request r at position t: c(t, r) = Δ_r · mean( F(context_t, a, r) − z_{t+Δ_r} )² + λ · work(r, policy) / work(reference), with λ = 1e-4;
- value: V(t) = min_r [ c(t, r) + V(t + Δ_r) ], with ties going to the lower request index and requests infeasible past the window end;
- window: the first 60-frame window of each controller-training shot, with terminal absorption.

**What it is guaranteed to optimise.** Backward induction gives the exact minimiser of this *additive* surrogate over request sequences, *given the fixed table of per-position costs*. It is a duration-weighted one-step prediction error from the context stored at each position, plus a MAC penalty. It is **not** the realised state-dependent recursive rollout cost, and it is not task outcome or ranking loss. We agree with vSA9 that no optimality statement extends beyond the surrogate.

**Round-2 successor contexts.** In the DAgger round, the policy is rolled out greedily from position 0. Every *visited* position's context is overwritten by the predicted carrier. **Unvisited positions keep their observed (teacher-forced) carriers.** The DP then evaluates *every* request from this one fixed context tensor. For a request not visited in round 1, no successor context is constructed recursively: its cost at each position uses that position's stored context (predicted if the greedy path visited it, observed otherwise).

**Post-submission tests of state-dependent labels** [Completed] (#101). We built a *new* decision-valued variant, J_dec. For each mixed policy cell and candidate, it rolls out a continuation policy to step 225 and, at each visited state and feasible request, deviates to that request and continues. The cost is a candidate-normalised pairwise ranking loss plus the same MAC term, and the label is the argmin. This is an approximate one-step lookahead over a base policy (heuristic, not an exact DP). A REINFORCE policy-gradient variant (J_pg) optimised imagined per-cell AUC. Results on sealed grid:

- J_dec − J (submitted recipe) **+0.040 [+0.029, +0.052], supported**;
- J_dec − F_hindsight **−0.016 [−0.020, −0.012]**, upper bound below δ;
- J_dec − M_dec +0.001 [−0.001, +0.002] and J_dec − OL_dec −0.003 [−0.006, +0.000]: no demonstrated value from state dependence or jointness;
- J_pg collapses to a single fixed request at λ ≥ 1e-2 (J_pg − F_hindsight −0.001 [−0.003, +0.001]).

J_dec is a post-submission method and should not be attributed to the submitted DP.

**Revision** [Planned]. State the surrogate, its fixed-context construction, and the absence of any recursive-cost guarantee. Report J_dec as an exploratory alternative.

## 0.7 Common concern F — external comparisons and the claim we can make

We ran a pre-registered external-baseline programme (#127–#141; 146 logged automatic decisions; 217.6 of 400 device GPU-h; 159.96 of 190 engine worker-hours). Its sealed-grid results (311 cells × 3 seeds) are:

| arm | grid AUC [95 %] | X − U [95 %] | frozen reading |
|---|---|---|---|
| Gate-B unit U | 0.7444 [0.7269, 0.7620] | — | — |
| A1 PHYRE-DQN (pixels) | 0.9946 [0.9917, 0.9971] | +0.2502 [+0.2327, +0.2675] | disposition 1 |
| A2 DQN on E99 carriers (matched input and capacity) | 0.9757 [0.9692, 0.9815] | +0.2313 [+0.2131, +0.2493] | disposition 1 |
| A4 Dynamics-aware DQN | 0.9945 [0.9922, 0.9965] | +0.2501 [+0.2327, +0.2674] | disposition 1, budget-qualified (1.88× U's training compute) |
| no-model nearest-fit lookup (NN-px) | 0.9039 [0.8838, 0.9231] | −0.1595 for U − NN-px | reference |
| B1 SlotFormer-E99 | 0.5476 [0.5264, 0.5692] | −0.1968 | fails G-action guard; stop token |
| C1-car DINO-WM | 0.5251 [0.5093, 0.5411] | −0.2193 | fails G-action guard; stop token |
| D1 THICK | 0.4989 [0.4810, 0.5165] | −0.2456 | fails G-action guard; stop token |
| T5 DIAMOND (coverage arm) | 0.7176 [0.6897, 0.7453] | −0.0268 [−0.0589, +0.0046] | fails G-action guard; stop token |
| U-native (U trained to A4's compute) | 0.8132 [0.7977, 0.8287] | +0.0688 vs U | fails G-action on 1 seed |

Observations:

- **The pre-registered Q1 reading applies: on this task the paper cannot claim that the world model adds ranking value over direct prediction.** A2 reads the same carrier, has matched capacity, and trains on the same fit shots.
- A follow-up that attaches A2's classifier to U's predicted endpoint reaches 0.9765, and the same classifier with **no rollout** reaches 0.9809 (F1 − F1-D −0.0044 [−0.0081, −0.0014], not supported either way). U's predicted endpoints add no incremental ranking value under that scorer (#141).
- The sealed grid rewards near-duplicate interpolation: A1 scores 0.99 in the near, mid and far distance strata, and NN-px scores 0.904.
- **Under novelty every ranker is near chance.** Pooled novelty levels 1–7 (novel / paired-normal): A1 0.578 / 0.750, A2 0.517 / 0.740, A4 0.569 / 0.720, U 0.474 / 0.656 (#128). The post-submission decision-valued policy J_dec scores 0.54 [0.49, 0.59] zero-shot on the same novelty levels (#107). Retuning the predictor, not the policy, recovers value (+0.151 [+0.097, +0.206]; policy-only −0.007 [−0.021, +0.006]).
- The world-model baselines B1, C1, D1 and T3 *fail the frozen G-action guard in carrier space*. We calibrated that guard post hoc: the engine truth itself scores 0.0395, U 0.0229, and the threshold is 0.05. We kept the frozen guard, so every one of those tokens is the stop token. This means those arms cannot establish a ranking claim in either direction.
- Not run (typed): DreamerV3, DDLP, Diffusion Forcing, DeepPHY (`fidelity_not_reproduced`); V-JEPA 2-AC (`cap`); C1-dino (`readiness_ceiling_below_threshold`). Phy-Q relational DQN failed native-task fidelity and is tokens-only. AIBIRDS agents, acting with free shot choice (a different protocol, descriptive): Datalab pass@1 / pass@3 0.273 / 0.307, Eagle's Wing 0.110 / 0.233, PigShooter 0.120 / 0.217.

**Consequence.** We will not claim superiority of the world model, adaptive granularity, or the policy over any external method. The program-level claim ("U ranks launches better than every baseline in the set") is **vetoed** by A1, A2, A4 and by the failed or dropped arms. **Status.** The external-baseline results are not yet carried into the local next version (ticket #142 is the handoff). The storyline decision is pending.

---

# Part 1. Reviewer MMed

*Rating 1. Reviewer's stated strength: the shared predictor serving different requests is interesting.*

We thank MMed for the concise assessment. We agree with the main thrust and have tried to turn the objections into measurements. The shared-predictor point is the part of the work that the follow-up evidence supports most strongly (J − S +0.085 [+0.074, +0.095], §0.1).

### W1. "CLEVRER experiments use overlapping scenes; NovPhy results rely on very few evaluation levels; generalization claims weakened."

**Answer.** Both observations are correct for the submission. CLEVRER and Physion-Dominoes Table 1 rows are in-sample on 12 scenes per benchmark. NovPhy decision sets had 7–14 mixed-verdict levels each. Detailed response: **§0.2** (splits) and **§0.3** (sealed re-test).

- **More levels** [Completed]: 400 sealed evaluation levels, 11,680 launches; mixed-verdict levels grid 311, offset 76, angle 78, power 47 (#104, #109).
- **Held-out scenes for the rollout result** [Descriptive]: leave-one-scene-out on the 12-scene cohorts keeps the ordering Δ = 15 < Δ = 5 < Δ = 1 under the submission recipe, with 2–5× higher error than in-sample (§0.2 item 2). Pixel-based CLEVRER on the official train/validation split and Physion-Dominoes on the official split show the same direction (§0.2 item 3).
- **Scope** we will adopt: no generalization claim from Table 1; the sealed split is level-disjoint but not template-disjoint; the Physion official split measures transfer to new colours (20/150 test trials share all physical parameters with fitting trials).
- **Novelty.** Generalization under novel physics was not a claim of the paper. When tested post-submission, rankers do not transfer (§0.7).

### W2. "Several action-ranking results are near random; additional description levels do not consistently improve accuracy; Figures 1 and 3 are overcrowded; the main text feels underdeveloped."

**Answer.** Agreed on all of the substance.

- **Near-random ranking.** Submitted-checkpoint AUCs lay between 0.40 and 0.64 (Appendix). On the sealed split the ICLR J scores 0.555 and F\* 0.542 (§0.3). The submitted parser read the pig in 0/859 decision frames (#99). After repairing perception, the Gate-B unit reaches 0.744 [0.727, 0.762], but the matched continuous C\* reaches 0.783 and the no-model prior 0.720 (§0.3). We state this as a limitation.
- **Description levels do not consistently help.** Confirmed by #113, #115, #120, #105, #121 (§0.4 item 5). The justified claim is limited to J − J-cont = +0.020 at most-budget, which is mostly the effect of J's horizon schedule (J − J-as-cont +0.0037).
- **Figures and presentation** [Planned; drafted in a local next version, #124]:
  - the Fig. 1 b2 panel has a minimal redraw with a caption that marks all panels as submission-checkpoint, short-window, development-set evidence and points to the sealed-split section;
  - the Fig. 4 heatmap is replaced in the main text by a table of nine fixed requests × four launch sets for both the Gate-B stack and the submission checkpoint, with the prior as a reference, and the heatmap moves to the appendix;
  - Fig. 3 will be split or moved to the appendix.
  - The reviewer's remark that the text was underdeveloped is fair. The revised results sections add the sealed-split tables, a frontier table, a closed-loop paragraph and a limitations paragraph. **Risk [Unresolved]:** the current local draft's main text ends on page 11 (#124 build record), so it needs compression or relocation before it fits a conference page limit.

### Q1. "Why are CLEVRER training and evaluation samples drawn from the same scenes? How is generalization assessed?"

The submitted cohorts were 12 annotation-carrier scenes per benchmark (480 windows of 61 frames, null action) scored in-sample (#117 item 5). We do not have a good reason to present that as a test of generalization, and the paper will not. As of now, generalization is assessed by (i) leave-one-scene-out on those 12 scenes [Descriptive], (ii) official-split pixel-based CLEVRER (disjoint scenes) and Physion-Dominoes (colour-disjoint, with the near-duplicate qualification) [Completed], (iii) the level-disjoint sealed NovPhy split [Completed]. See §0.2 for numbers. A Table-1-format hybrid-versus-continuous contrast on a documented disjoint annotation set remains [Unresolved].

### Q2. "What benefit do symbolic description levels offer over continuous prediction alone?"

- On the sealed split the matched continuous model ranks at least as well as the hybrid: unit − C\* = **−0.0390 [−0.0442, −0.0338]** (#108).
- The description-axis effect on decisions is small: J − J-cont +0.0201 [+0.0154, +0.0248] (at-most-budget), J − J-as-cont +0.0037 [−0.0021, +0.0095], J − T-continuous +0.0032 [−0.0075, +0.0137] (§0.4).
- Micro predicates are readable and need no engine labels for ranking (non-inferiority only). Macro predicates are not readable in rollout (F1 0.015) and show no ranking advantage (§0.4 item 5).
- The remaining value we can state is **rollout accuracy under some training protocols and interface generality**, not decision quality. The benefit of description levels in the submission is therefore **not demonstrated**.

### Q3. "Do the weak action-ranking results translate into meaningful task-success improvements?"

No improvement has been demonstrated. In descriptive closed-loop pass@3 (§0.3), models beat random (0.198) and sit above the prior (0.323) at 0.39–0.48, but the best arm is a fixed continuous request (0.478), and adaptive policies do not beat it. Frozen guard G4 failed, so all tokens are `readiness_or_precision_insufficient`. We make no task-success claim for adaptive granularity.

### Q4. "Can the authors evaluate in more complex or closed-loop environments?"

- **Closed loop** [Completed, descriptive] (#102): live engine execution of chosen launches, 5,123 executions over 19,360 episodes. The setting is single-bird retry-from-start with up to 3 shots (§0.3).
- **More complex levels** [Completed] (#110, #107): 45 novelty cells (forces, agents, appearance) with engine-verdicted evaluation. Rankers do not transfer well (§0.7). The policy-only retune does not recover value.
- **Other simulators** [Not done]: we did not run PHYRE or other closed-loop suites for the method. PHYRE and others appear only as fidelity references for baselines. Multi-bird sequential play is also not done.

---

# Part 2. Reviewer MSwm

*Rating 2. Reviewer's strengths: simple, well-motivated idea; honest reporting; the finding that the best granularity depends on how the prediction is used.*

We thank MSwm for the careful reading and for numerical specificity. The third strength (granularity depends on use) is the finding we consider most robust: on the sealed split the best fixed request still changes between inventories on both stacks (#124 §6.2; #104 held-out table).

### W1. "Fig. 4 AUCs are 0.40–0.63; policy gains over r† (+0.004 to +0.070) are measured against a reference near chance; 7–14 scored levels; only one of four Fig. 1(b2) intervals clears zero."

**Answer.** Agreed. The numeric intervals are in §0.3 (Fig. 1(b2) differences) and in the Appendix (Fig. 4 cells). The sealed rerun with 311 / 76 / 78 / 47 levels did **not** replicate the gains: grid +0.013 [−0.016, +0.043], offset +0.011 [−0.052, +0.074], angle −0.160 [−0.206, −0.113], power −0.262 [−0.336, −0.192]. All four precision targets were missed (half-widths 0.030 / 0.063 / 0.047 / 0.072) [Completed] (#104). The submission's claim that the selection policy beats fixed settings is **withdrawn**. The wording becomes: no AUC advantage of 0.02 or more over a fixed request tuned on other sets.

### W2. "Chatterjee & Khardon and Neitz et al. not cited; the novelty is the description axis and joint choice; no cont-only ablation on the same hybrid checkpoint; part of the compute saving is the narrower trunk."

**Answer.** All four points are correct and addressed in §0.4:

- Both works are cited in the revision [Planned]. The novelty is narrowed to the description axis, the joint interface and the single shared predictor.
- **The cont-only ablation on the same hybrid checkpoint exists** [Completed] (#114): J − J-cont = +0.0201 [+0.0154, +0.0248], supported but at-most-budget; J − J-as-cont +0.0037, not supported.
- **Compute.** The saving relative to C-J is trunk-driven (−50.8M of −50.7M net MACs), and description selection adds +6.6M per-call surcharge. The submitted "a third of the compute" sentence is withdrawn: ratio 0.784 versus ≤ 0.40 (#103).

### W3. "2,400 of 2,409 training lineages count Δ in engine steps; NovPhy test seems to use 50-step frames; CLEVRER test contexts come from the 12 training scenes; no Physion-Dominoes details."

**Answer.**

- **Time base.** The distinction you draw between the base-corpus convention and the 50-step N1 frames is real; the submission did not characterise it. The 2,400 lineages are NovPhy base-corpus lineages (not CLEVRER or Physion). Our scoped acknowledgment and status are in §0.5. The central unresolved item is an ablation of the carrier's elapsed-time channel.
- **CLEVRER and Physion.** Correct. Table 1 is in-sample on 12 scenes. Physion-Dominoes data are now documented: 12 scenes, 480 windows, null action for the annotation cohort; the official split for the pixel experiment, with near-duplicate qualification (§0.2).
- **Request.** (1) Clarify the time base: [Planned]. (2) Table 1 on documented held-out scenes: the leave-one-scene-out and pixel results above are as close as we currently have [Descriptive/Completed]; the direct Table-1-format recompute is [Unresolved].

### Minor points

1. **TAWM is ICML 2025 main track.** Confirmed (PMLR volume 267; OpenReview "ICML 2025 Conference"). We will correct the citation.
2. **Terminology** (as used in the repository; we will define them in the revision [Planned]):
   - *Scenario lineage*: a scenario specification together with every rollout, seed, intervention and observation configuration derived from it. A lineage belongs to exactly one dataset partition.
   - *Base corpus*: the frozen set of 3,000 training-role NovPhy lineages from the earlier aligned player, of which 2,400 fit the predictor and 600 supervised the controller (#71).
   - *N1 captures*: the native-player campaign on two level families (rolling, sliding) × 8 lineages × 13 fixed actions = 208 branches (#77). Each branch is up to 30,000 native steps with an RGB frame every 50 steps.
   - *Development states*: the exposed N1 lineages used for development-time scoring. The sealed #109 evaluation split replaces them for confirmatory reading.

### Q1. "What does Δ count in training and the NovPhy test? Does the model rely on the elapsed-time entry of the carrier (Eq. 9)?"

Δ counts **carrier frames** in the request code. What a frame spans depends on the corpus (§0.5). The carrier does contain an elapsed-seconds entry and per-second velocities. Whether the model *relies* on that entry has **not** been measured. We have not ablated it, and the κ study (#119) does not test it. We commit to adding that ablation and the unit table [Planned].

### Q2. "Number of scored levels and successful launches per set; intervals for all Fig. 4 cells and the four Fig. 1(b2) differences."

- **Counts (submission launch sets).** Scored (mixed-verdict) levels: grid 14, offset 9, angle 7, power 8 (cells = levels × 3 seeds = 42 / 27 / 21 / 24). Successful launches in those levels: 17 / 9 / 7 / 8, of which the engine success event occurs within the 225-frame endpoint: 11 / 3 / 5 / 4.
- **Sealed split.** Mixed levels: 311 / 76 / 78 / 47 (§0.2). Mixed share by family on the evaluation split is 0.72 / 0.73 / 0.82 / 0.42 for rolling (grid / offset / angle / power) and 0.84 / 0.53 / 0.48 / 0.37 for sliding.
- **Fig. 1(b2) differences and the Fig. 4 cells:** §0.3 and the Appendix.

### Q3. "Can Table 1 be recomputed on held-out CLEVRER scenes and documented Physion-Dominoes scenes, with a persistence or constant-velocity baseline and the policy's own schedule error?"

- **Held-out recomputation:** see §0.2 (leave-one-scene-out [Descriptive]; official-split pixel pipeline [Completed]).
- **Policy's own schedule error.** On the pixel benchmarks, J's own-schedule position MSE is CLEVRER 0.157 [0.147, 0.166] at t = 120 and Physion 0.0235 [0.0188, 0.0285] at t = 90 (#106). J mostly selects Δ = 1, and lies close to F-1 (CLEVRER 0.163; Physion 0.0430). The fixed Δ = 15 request has lower error (0.0996, 0.0143). J / J-cont log ratio: −0.035 [−0.065, −0.006] (CLEVRER), −0.10 [−0.18, −0.03] (Physion), descriptive. So the learned schedule is **not** the best rollout schedule there.
- **Persistence / constant-velocity baseline** on CLEVRER and Physion: [Unresolved], not run. On NovPhy novelty cells, persistence (anchor held fixed) is reported (#110, masked carrier MSE, level-0 zero-shot): persistence 0.105, F-15-continuous 0.042, F-1-continuous 0.169. Zero-shot Δ = 1 rollouts are worse than persistence in 42 of 45 cells.

### Q4. "Does the compute saving come from the description level, the horizon, or the narrower trunk?"

The trunk. Over C-J at λ = 1e-4: trunk −50.84M MACs, calls −6.35M, per-call description surcharge **+6.58M**, policy −0.11M; net −50.71M (#114). Description selection adds cost, and horizon calls contribute little. Against a fixed request on the continuous predictor (C-F-15, 27.8M MACs, AUC 0.783), J and C-J are dominated (#103). The compute claim is withdrawn.

---

# Part 3. Reviewer vSA9

*Rating 1, confidence 4. Reviewer's strengths: a shared carrier allows switching between temporal and symbolic feedback in one rollout; matched training budgets, three seeds, explicit compute accounting, standardised inventories.*

We thank vSA9 for the most technical review, and for recognising the budget and accounting discipline. Your two weaknesses are well-founded, and we answer each question with completed evidence where it exists.

### W1. "Carrier frames span one engine step and 50 engine steps, with request horizons in the same frames; temporal-rate encoding is poorly characterised."

Agreed. See **§0.5** for the scoped acknowledgment, the three separate time signals, and what is and is not established. For the primary post-submission stack, training and evaluation share the capture pipeline and the 50-native-step frame (#104, #109, #112), so no corpus mismatch applies. For the submission checkpoint, transferability across the time base is **not** established and cannot be separated from the parser defect.

### W2. "The horizon-only cost comparison changes predictor width, tokenisation training, representation use and training data at once; it does not support 'jointly selecting the representation yields compute savings'. DP optimises an additive proxy on a fixed-context carrier, not the state-dependent recursive cost."

Agreed on both.

- **Confounded comparison** (#103, #114). The submitted comparison was J (hybrid trunk) versus a horizon-only policy on a different predictor. The follow-up evidence separates the factors:
  - same hybrid checkpoint, horizon-only policy: J-cont. J uses +61.2M more MACs (§0.4 item 1);
  - same predictor, replayed schedule: J-as-cont AUC difference +0.0037;
  - J versus C-J: the saving is the trunk (−50.8M), and description selection adds +6.6M per call.

  The compute claim is withdrawn (ratio 0.784 vs ≤ 0.40; wall 16.74 on the frozen single-seed timing, 1.63 on the post-hoc three-seed timing).
- **DP objective.** Agreed. See **§0.6**. The submitted objective is an additive, fixed-context surrogate. The post-submission J_dec (recursive continuation, ranking loss) and J_pg (imagined-rollout policy gradient) were tested, and neither beats the best fixed request.

### Q1. "Do the training windows and evaluation contexts on page 15 come from the same CLEVRER scenes? Provide scene IDs and recalculate Table 1."

Yes. For the submitted Table 1 the fit and evaluation scenes are the same 12 scenes per benchmark (in-sample, fold "all"). Disjoint recomputation:

- leave-one-scene-out on the 12 scenes, held-out rollout error [Descriptive] (§0.2);
- pixel pipeline with disjoint official train/validation videos (CLEVRER 6,285 / 3,132 after exclusion) [Completed] (#106). Video lists are in the committed data manifest (`data/clevrer/manifest.json`) and the leakage audit.

A Table-1-format recompute with the continuous-only baseline on a disjoint annotation set is [Unresolved] (§0.2).

### Q2. "What predictor-input metric (rate or elapsed time) distinguishes a one-engine-step from a 50-engine-step execution?"

The carrier contains an **elapsed-seconds entry** and per-second slot motion. The request code ϕ(Δ) counts carrier frames and has no unit. The policy has a separate remaining-frames input. Whether the trained model uses the elapsed entry to separate the two time bases is **not measured** (§0.5) [Unresolved]. We will tabulate seconds-per-frame per corpus and run the ablation [Planned].

### Q3. "Same checkpoints and equal decision budget: fixed macro step with adaptive horizons, fixed horizon with adaptive description, and a time-only scheduler?"

The table in **§0.4 item 7** answers this: T-continuous (time-only), T-macro (fixed macro, adaptive Δ), D-1 / D-5 / D-15 (fixed Δ, adaptive description), same Gate-B hybrid checkpoints, same policy pool and recipe. J − T-continuous +0.0032 [−0.0075, +0.0137]; J − T-macro +0.0281 [+0.0227, +0.0337]; J − D-15 **−0.0562 [−0.0685, −0.0441]**. The comparison is at the same λ, **not** MAC-matched (J 184.0M, T-continuous 196.1M). The matched-budget J − J-cont and J − C-J contrasts are in §0.4 item 1.

### Q4. "Report task-success-paired uncertainty, a full list of successes and failures, and whether the 0.004 offset gain still holds."

- **Task success with uncertainty:** closed-loop pass@3 and pass@1 with intervals (§0.3) [Descriptive, G4 failed]. Top-3 hit rate on the sealed grid (#114): J 0.5102 [0.4620, 0.5606]; J-cont 0.4662; C-J 0.4469; T-continuous 0.5348; no-model prior 0.4180 (descriptive).
- **Complete results rather than AUC alone.** Capture accounting for the sealed campaign: 49,508 / 49,560 branches complete, 52 typed failures (0.105 %), none in the evaluation stream. The evaluation stream's mixed-verdict yield is grid 311/400, offset 76/120, angle 78/120, power 47/120. The ICLR-era campaigns carried typed failures that we disclose (N1 30 / 208 branches; n2n normal side 70 / 104; closed-loop oracle pass 55 / 289 slots) (#109). Per-level verdicts and per-arm per-level records are retained in committed artefact manifests. We will publish the per-level verdict table and the closed-loop episode table (44 arm × seed rankings × 440 levels) as supplementary material [Planned].
- **Does +0.004 (offset) hold?** On sealed levels the offset difference is +0.011 [−0.052, +0.074]. It is not distinguishable from zero. On angle the sign reverses (§0.3).

### Q5. "What objective is the DP guaranteed to optimise? How are round-2 successor contexts built for requests not visited in round 1?"

See **§0.6**. In short: exact backward-induction minimisation of the additive, fixed-context, duration-weighted prediction-error + MAC surrogate; no recursive-cost guarantee. Unvisited positions retain their observed carriers.

---

# Part 4. Manuscript change ledger

| # | Change | Concern | Status |
|---|---|---|---|
| 1 | Abstract, intro, discussion: remove or scope "selection beats fixed requests", "best on two sets / worst on a third", "a third of compute", "unsolvable by any fixed setting, and cheap to adapt". Replace with δ-rule wording and sealed-split numbers. | MSwm W1/W2, vSA9 W2 | Drafted in local next version (#124); abstract final-sentence wording flagged for owner confirmation |
| 2 | Compute paragraph replaced by the #103 engine-truth frontier and the 0.784 / 16.74 figures; "a third" sentences removed everywhere. | MSwm W2/Q4, vSA9 W2 | Drafted (#122/#124) |
| 3 | Sealed-split section: fit/policy/evaluation splits, 311/76/78/47 mixed levels, level-disjoint versus template-disjoint caveat, guards, scored once. Fig. 4 heatmap replaced by a table; heatmap to appendix. | MMed W1/W2, MSwm W1, vSA9 Q4 | Drafted (#124) |
| 4 | Table 1 relabelled in-sample; add leave-one-scene-out and pixel-benchmark results; document Physion-Dominoes data and split; near-duplicate qualification. | MMed W1/Q1, MSwm W3/Q3, vSA9 Q1 | The local draft (#124) lists the #106 pixel results and the #117 sharing results in its appendix; the Table 1 relabelling, the leave-one-scene-out rollout table and the Physion data documentation are not confirmed in the #124 record. Treat as [Planned] |
| 5 | Add Chatterjee & Khardon (NeurIPS 2025), Neitz et al. (NeurIPS 2018); correct TAWM to ICML 2025 main track; narrow the novelty paragraph. | MSwm W2, minor 1 | [Planned] (#124 added H-JEPA/HWM; these three citations are not in its log) |
| 6 | Define lineage, base corpus, N1 captures, development states. | MSwm minor 2 | [Planned] |
| 7 | Time-base table (native steps per frame, seconds per frame, per corpus); distinguish request code, carrier elapsed entry, policy κ; elapsed-channel ablation. | MSwm W3/Q1, vSA9 W1/Q2 | [Planned]; the ablation has not been run |
| 8 | Method section: state the additive fixed-context DP surrogate and the DAgger context rule; report J_dec as exploratory. | vSA9 W2/Q5 | [Planned] |
| 9 | Macro description: report as unreadable in rollout (F1 0.015); remove "macro best everywhere". | MMed W2/Q2 | Drafted (#124) |
| 10 | Limitations: prior sits within δ of most rankers; matched continuous C\* ≥ hybrid; ranking is update-limited (interpretation); macro does not survive rollouts; closed-loop G4 failed; Physion near-duplicates; retrain variance exceeds δ. | all | Drafted (#124, §7) |
| 11 | External-baseline programme (direct predictors outrank U; world-model baselines fail guards; NN-px; novelty collapse; U-native; E1) and the claim veto. | MMed Q2, vSA9, significance | Pending (#142); storyline decision not yet posted |
| 12 | Figures 1 and 3 simplified. | MMed W2 | Fig. 1 b2 redrawn (#124); Fig. 3 [Planned] |
| 13 | Supplementary per-level verdict and per-episode closed-loop tables. | vSA9 Q4 | [Planned] |

---

# Part 5. Unresolved limitations (stated plainly)

1. **No Table-1-format recompute** (hybrid versus continuous-only) on a documented, disjoint CLEVRER / Physion annotation set; no persistence or constant-velocity baseline on those benchmarks.
2. **Carrier elapsed-time channel not ablated**; time-base transfer from the base corpus to N1 not tested; physical frame spacing per corpus not yet tabulated in the paper.
3. **Precision**: held-out contrasts missed their pre-declared half-width targets on all inventories (grid 0.030 versus 0.02). Power is below its design depth.
4. **Split structure**: two generator templates; near-duplicate levels; NN-px 0.904 on the evaluation grid. The sealed split has been scored by many tickets, so it is not a fresh holdout (disclosed in #127).
5. **Closed loop** is descriptive (G4 failed), single-bird retry-from-start, grid inventory only.
6. **Ranking quality looks update-limited** (equal-per-request-update predictors reach up to 0.947 grid AUC (#116); full-budget specialists rank much higher (#117); U-native with 1.88× updates 0.8132). This is an interpretation, not an isolated causal result.
7. **Carrier-space G-action guard** is unattainable even by the engine truth, so external world-model comparisons yield stop tokens.
8. **Reproducibility.** The first cross-hardware replication check (RTX 3090 records re-scored on the RTX 5090 server) did not pass bit-exactly (#108 server self-check); later runs re-validate on the 5090 server. A rerun of the 32 reproduction commands gave 23 exit-0 and 9 stopping at freeze checks (path rewrites and edited capture sources); these neither confirm nor contradict published tables (#126, still open). Several tickets disclose operational deviations (for example, #114 duplicate scoring workers broke literal scored-once execution for fixed-arm prefixes).
9. **Scope of post-submission evidence.** Not every result is on the sealed NovPhy split: #106 (CLEVRER, Physion), #117 item 5, and the ICLR-era #96 numbers use other data.

---

# Appendix A. Fig. 4 cells with numeric intervals (submission checkpoint, development launch sets)

Source: retained artefact of the submitted #96 contrast (`issue-96-tau-ad-within-checkpoint-v1/summary.json`); member-clustered descriptive percentile bootstrap, 10,000 draws. These are the same cells whose range (0.401 to 0.635) matches the 0.40–0.63 the reviewer quotes. Mapping to the figure's panels is by that match. Mean [lower, upper]; grid 14 members / 42 cells, offset 9 / 27, angle 7 / 21, power 8 / 24.

| request | grid | offset | angle | power |
|---|---|---|---|---|
| F-1-continuous | 0.510 [0.483, 0.536] | 0.517 [0.481, 0.556] | 0.520 [0.500, 0.544] | 0.509 [0.488, 0.536] |
| F-1-micro | 0.493 [0.464, 0.520] | 0.500 [0.489, 0.511] | 0.502 [0.446, 0.560] | 0.520 [0.504, 0.542] |
| F-1-macro | 0.513 [0.467, 0.563] | 0.504 [0.444, 0.559] | 0.506 [0.423, 0.583] | 0.527 [0.474, 0.592] |
| F-5-continuous | 0.415 [0.337, 0.480] | 0.574 [0.507, 0.648] | 0.581 [0.497, 0.663] | 0.458 [0.374, 0.549] |
| F-5-micro | 0.425 [0.368, 0.484] | 0.576 [0.491, 0.659] | 0.609 [0.554, 0.671] | 0.469 [0.379, 0.546] |
| F-5-macro | 0.401 [0.348, 0.461] | 0.594 [0.517, 0.674] | 0.635 [0.587, 0.679] | 0.501 [0.414, 0.606] |
| F-15-continuous | 0.492 [0.411, 0.584] | 0.441 [0.348, 0.504] | 0.450 [0.323, 0.561] | 0.462 [0.403, 0.528] |
| F-15-micro | 0.479 [0.395, 0.578] | 0.487 [0.415, 0.544] | 0.450 [0.359, 0.524] | 0.452 [0.399, 0.520] |
| F-15-macro | 0.491 [0.455, 0.526] | 0.522 [0.489, 0.559] | 0.534 [0.500, 0.569] | 0.499 [0.438, 0.559] |
| J (submitted policy) | 0.471 [0.416, 0.526] | 0.507 [0.456, 0.559] | 0.565 [0.520, 0.625] | 0.529 [0.469, 0.591] |

---

# Appendix B. Evidence index (traceability; the argument above does not depend on following links)

Base URL `https://github.com/Sino-Huang/NovPhy/issues/<N>`. Ticket status at the last scan: the AFTER-ICLR evidence tickets cited here (#99–#125 and #127–#141) were closed. #142 (manuscript handoff for the #127 results) and #126 (reproduction bug) were open.

| item | tickets | key verified result |
|---|---|---|
| Roadmap, Gate B | #108 | pooled grid AUC 0.7444 [0.7269, 0.7620]; unit − C\* −0.0390 [−0.0442, −0.0338]; prior 0.7203; right − wrong anchor +0.1322 |
| Perception, readout | #99, #100 | submission parser pig 0/859; repaired parser pig 1.000; micro edge F1 0.989; macro unreadable |
| Sealed split, held-out rerun | #109, #104 | 400 evaluation levels; 311/76/78/47 mixed; ICLR J − F\* +0.013 [−0.016, +0.043] (grid), half-widths 0.030 / 0.063 / 0.047 / 0.072 |
| Decision-valued labels | #101 | J_dec − J +0.040 [+0.029, +0.052]; J_dec − F_hindsight −0.016 [−0.020, −0.012] |
| Closed loop | #102 | pass@3 table; G4 failed |
| Compute frontier | #103 | MAC ratio 0.784 vs ≤ 0.40; wall 16.74 (frozen) / 1.63 (post hoc) |
| Horizon-only versus joint | #114 | J − J-cont +0.0201 [+0.0154, +0.0248] (at-most-budget); J − J-as-cont +0.0037; trunk −50.84M MACs |
| Tied axes, hierarchy | #105 | J − J_diag −0.0068; J − H\* −0.0006; TAWM/VLWM ≤ 0.642 |
| Pixels | #106 | CLEVRER QA readout never fired; Physion OCP −0.021 [−0.068, +0.025]; 20/150 near-duplicate test trials |
| Novelty | #110, #107 | 45/45 cells supported; policy-only retune −0.007 [−0.021, +0.006]; predictor +0.151 [+0.097, +0.206] |
| Macro | #113, #115, #120 | held-out macro F1 0.157 / 0.113 / 0.015; macro − continuous −0.012 [−0.019, −0.004] |
| Sharing | #117 | J − S +0.0848 [+0.0742, +0.0952]; item 5 LOSO ratios |
| Request set, κ, code | #116, #119, #121 | gains track updates; κ +0.0042 [−0.0005, +0.0088]; fused − additive +0.050 (interpretation limited) |
| H-JEPA | #123, #125 | G-fid pass; collapse to ~0.49–0.50; no ranking claim either way |
| External programme | #127–#141, #142 | direct predictors 0.97–0.99; NN-px 0.904; world-model baselines fail guards; U-native 0.8132; F1 − F1-D −0.0044 |
| Manuscript carry-over | #124, #142 | next version drafted (storyline A; Gate-B stack primary); #127 results pending |

---

# Internal notes (not part of the response text)

**Evidence risks for the author team before posting.**

1. The abstract's final sentence and the "worst on a third" clause were rewritten in the local draft beyond the two clauses the owner named; the owner's confirmation is open (#124).
2. The #127 storyline decision (reframing after the Q1 reading) is not posted. This draft *recommends* narrowing rather than reporting a decision.
3. The local draft's main text ends on page 11 (#124 receipt, after H-JEPA material); compression is needed.
4. The Appendix A mapping of the #96 cells onto the submitted Fig. 4 is by range match (0.401–0.635), not by reading the figure.
5. The time-base subsection deliberately makes no equivalence claim between corpora. The capture notes name a 0.02 s timestep for the earlier aligned player and 50 × 0.0004 s = 0.02 s for N1 frames. We did not verify the base-corpus frame spacing against the shards and left it out of the argument.
6. "T-macro" for vSA9's "fixed macro-step size with adaptive horizons" is an interpretation, and is flagged as such.
7. Several "supported" tokens are at-most-budget or non-inferiority readings (#114, #115, #117, #121); the draft qualifies each.
8. The J − C-J pass@3 difference (+0.048) carries no token because G4 failed; it is quoted only as descriptive.
