# Integrated author response — ICLR 2027 submission #6352

**Paper:** *Granularity Is a Decision: Joint Horizon–Description Selection in Latent World Models*  
**Reviewers:** MMed, MSwm, vSA9

> **Author-only note:** Evidence tags such as **[E1]** link to the named repository in the traceability section. Remove those links or replace them with an anonymized supplement before posting. The reviewer-facing response is self-contained without them.

## Opening

We thank MMed, MSwm, and vSA9 for precise, constructive criticism. The reviews identify a distinction that the submission did not make sufficiently sharply: **supporting joint horizon–description requests is not the same as demonstrating that a learned selector improves decisions, symbolic descriptions help, or computation falls**. We have now run follow-up experiments addressing the requested controls. We report both positive and negative outcomes and distinguish completed post-submission evidence from proposed manuscript changes.

The strongest supported contributions are narrower and, we think, clearer:

1. **One shared predictor serves all nine requests.** At matched updates per request on the sealed NovPhy split, sharing outperforms single-request specialists by **+0.0848 [0.0742, 0.0952] request-mean AUC**. This is a sharing result, not evidence that joint adaptation is necessary; full-budget specialists can perform better.[E9]
2. **Prediction granularity materially changes rollout error.** On new scene-disjoint, pixel-carrier experiments, fixed $\Delta=15$ requests have lower last-endpoint error than $\Delta=1$ on CLEVRER and Physion-Dominoes. The learned policy's own schedules do not inherit that advantage.[E7]
3. **The missing same-checkpoint control now exists.** Joint $J$ minus a separately distilled continuous-only $J_{\mathrm{cont}}$ on the same hybrid checkpoint is **+0.0201 [0.0154, 0.0248] AUC**. This is a qualified, at-most-budget result, not exact-equal-spend or superiority over the best fixed request.[E6]
4. **The evaluation is now substantially stronger:** 480 fit / 120 policy / 400 evaluation NovPhy levels, 11,680 completed evaluation launches, fixed-request and no-model controls, wrong-anchor tests, live engine execution, and explicit uncertainty and guard reporting.[E1–E4]

The same experiments also require three corrections to the submission: the original scene-overlap results are not generalization evidence; the original policy's small gains do not replicate robustly; and the “one-third compute” statement is not supported. We will revise the paper's claims, figures, and protocol accordingly rather than defend those statements.

**Reading the numbers.** Unless stated otherwise, intervals are descriptive 95% level/member-clustered bootstrap intervals; three model seeds are not three independent sets of levels. The frozen practical AUC margin is $\delta=0.02$. A positive point estimate or an interval excluding zero is not by itself a supported practical improvement. All experiments discussed below are post-submission unless explicitly called submitted results.

---

## Shared responses to the decisive concerns

### A. Evaluation isolation, held-out scenes, and statistical power

**Direct answer.** MMed, MSwm, and vSA9 are correct about the submitted evaluation. The original CLEVRER fitting windows and longer evaluation rollouts came from the same 12 scenes, **10000–10011**. The original Physion-Dominoes rollout diagnostic likewise reused a small within-release cohort. Those tables measured longer rollouts on fitting scenes, not scene generalization. The original NovPhy launch inventories contained only **14/9/7/8 mixed levels** for grid/offset/angle/power, with exposed development members. We will relabel these results as development/in-sample diagnostics and remove their generalization interpretation.[E1, E7, E9]

**Completed NovPhy repair.** We froze a new level-disjoint cohort before capture, with fresh seeds checked against 83 prior-exposure sources and zero collisions:[E1]

| Quantity | Completed evaluation |
|---|---:|
| Predictor-fit / policy-fit / evaluation levels | 480 / 120 / 400 |
| Scheduled evaluation launches completed | 11,680 / 11,680 |
| Mixed-outcome grid / offset / angle / power levels | 311 / 76 / 78 / 47 |
| Model seeds per scored level | 3 |
| Evaluation-stream typed capture failures | 0 |

The non-grid inventories use 120 of the 400 levels; inventories are never pooled. Power remains below its planned depth of 53 mixed levels. Identity separation also is not physical-novelty separation: the cohort covers two familiar generator families, and a nearest-fit lookup later reached 0.904 grid AUC, exposing substantial within-family similarity. We therefore claim **identity-disjoint, within-distribution evaluation**, not broad physical generalization.[E11]

**Completed CLEVRER/Physion repair.** A new pixel-carrier study uses the official train/validation or release splits:[E7]

- **CLEVRER:** 6,285 fitting videos from IDs 0–9999 and 3,132 evaluation videos from IDs 10000–14999, with zero ID overlap. Exact included/excluded IDs are frozen in the manifest. A predeclared unique colour–material slot rule excludes 37.22% of videos, so this is a parser-compatible subpopulation, not full-release CLEVRER.
- **Physion-Dominoes:** the documented dynamics/readout/testing releases, with 150 red–yellow test trials. No full static-parameter set is shared across splits. However, when appearance-only colour is ignored, **20/150 test trials share all physical parameters with a fitting trial**, and four are near-identical. The official split therefore measures transfer to new colours, not independent physical realizations.

The new last-endpoint, parser-relative position-MSE results are:

| Schedule | CLEVRER, $t=120$ | Physion-Dominoes, $t=90$ |
|---|---:|---:|
| Fixed (1, continuous) | 0.163 [0.152, 0.176] | 0.0430 [0.0327, 0.0535] |
| Fixed (5, continuous) | 0.109 [0.104, 0.115] | 0.0227 [0.0184, 0.0272] |
| Fixed (15, continuous) | 0.0996 [0.0958, 0.104] | 0.0143 [0.0108, 0.0182] |
| Fixed (15, micro) | 0.0993 [0.0956, 0.103] | 0.0145 [0.0109, 0.0187] |
| Fixed (15, macro) | 0.0974 [0.0944, 0.101] | 0.0146 [0.0110, 0.0187] |
| $J$, its own schedule | 0.157 [0.147, 0.166] | 0.0235 [0.0188, 0.0285] |
| $J_{\mathrm{cont}}$, its own schedule | 0.162 [0.152, 0.173] | 0.0252 [0.0206, 0.0303] |

Thus coarse horizons reduce recursive error in this held-out pixel study, while the learned schedules remain close to $\Delta=1$. This is a **new pixel-carrier measurement**, not a like-for-like recomputation of the original annotation-carrier table; the error reference is the parsed carrier, not simulator-ground-truth physical state. On the original 12-scene carriers, an exploratory leave-one-scene-out analysis also preserves the coarse-horizon ordering under the submitted recipe, but it is not a complete replacement for Table 1.[E9]

Two requested additions are not completed results: **persistence and constant-velocity references on this exact pixel protocol**, and a physically group-disjoint Physion evaluation. We will add the simple references as scoring-only rows on stored observations; we make no superiority claim over them here. The Physion physical-overlap limitation will remain attached to the table.

### B. Near-chance ranking, statistical significance, and task success

**Direct answer.** The reviewers' reading of the submitted ranking results is correct. A small gain over a near-chance cross-fit reference is not evidence of useful action selection, and AUC is not top-1 task success. The original $J$ had zero top-1 hits on all four inventories. The complete original intervals and counts appear in Numerical Supplement S1.[E2]

**Submitted-checkpoint rerun.** On the new level-disjoint split, using the original checkpoint and controllers, the original adaptivity conclusion does not survive:[E3]

| Inventory | Mixed levels | $J-F^*$ | Half-width |
|---|---:|---:|---:|
| Grid | 311 | +0.0134 [−0.0163, +0.0431] | 0.0297 |
| Offset | 76 | +0.0111 [−0.0520, +0.0743] | 0.0632 |
| Angle | 78 | −0.1596 [−0.2061, −0.1130] | 0.0466 |
| Power | 47 | −0.2617 [−0.3360, −0.1921] | 0.0720 |

The precision guard fails on every inventory. On sealed grid levels, submitted $J$ scores **0.555 [0.539, 0.571]**, $F^*$ scores 0.542, and the hindsight-best fixed request scores 0.639. We will withdraw the claim that the submitted selector beats fixed requests; the original offset gain of approximately +0.004 is not a replicated effect.[E3]

**Improved, but not hybrid-superior, predictor.** Repairing perception and adding long-horizon plus decision-contrastive supervision produces a fixed-request unit $U=F(15,\mathrm{macro})$ with sealed grid AUC **0.7444 [0.7269, 0.7620]**. A wrong-anchor control falls to 0.6122, giving right-minus-wrong **+0.1322 [0.1216, 0.1430]**. This verifies state/anchor dependence. It does not show a hybrid or adaptive advantage: matched continuous $C^*$ scores **0.7834 [0.7676, 0.7992]**, $U-C^*=-0.0390$ [-0.0442, −0.0338], and the no-model action prior scores 0.7203, with $U-\mathrm{prior}=+0.0241$ [-0.0008, +0.0487].[E4]

**Live task outcomes.** We subsequently executed chosen launches live on all 400 evaluation levels, using retry-from-start with at most three shots. The best arm is a fixed coarse request, not a learned selector. Illustrative pass@3 values are fixed (15, continuous) **0.478 [0.434, 0.523]**, $C^*$ 0.466, decision-valued $J_{\mathrm{dec}}$ 0.445, $U$ 0.424, submitted-recipe $J$ 0.394, horizon-only $C-J$ 0.346, no-model prior 0.323, random 0.198, and inventory oracle 0.775. The complete arm table is in Numerical Supplement S2.[E5]

These results are **descriptive only**: 25 of 5,106 comparable live decision frames on five levels exceeded the frozen anchor tolerance, so every formal contrast token is `readiness_or_precision_insufficient`. A post-hoc reranking left the first three picks unchanged in 351/351 comparisons, but does not repair the guard. Moreover, this is single-bird retry-from-start, not a changing-state multi-action controller. It answers whether chosen launches can produce live successes, not general closed-loop superiority.[E5]

**Strong direct-prediction control.** A direct success predictor using the same carrier, matched capacity, fit shots, and update budget reaches **0.9757 [0.9692, 0.9815]**, versus 0.7444 for $U$. Attaching a learned success readout to $U$'s predicted endpoint reaches 0.9765, while the corresponding no-rollout control reaches 0.9809; their difference is −0.0044 [−0.0081, −0.0014], with no supported advantage in either direction. We therefore do **not** claim that world-model rollout adds ranking value over direct success prediction for this task. On novel-level pairs, direct predictors fall to 0.52–0.58 and $U$ to 0.47, further limiting any in-distribution interpretation.[E11]

### C. What description levels contribute, and what they cost

**Direct answer.** The submission did not establish a general symbolic-description benefit, and the new evidence does not establish one either. It does establish a narrow same-checkpoint policy contrast and a stronger parameter-sharing result.

Let $J$ select horizon and description; $J_{\mathrm{cont}}$ be separately distilled with only continuous requests on the **same hybrid checkpoint**; $T_\alpha$ fix description and adapt horizon using restricted $J$ logits; $D_\Delta$ fix horizon and adapt description; and $C-J$ be horizon-only selection on the separately trained continuous predictor. These controls must not be conflated.[E6]

| Control | Grid AUC | Top-3 hit | MACs/candidate |
|---|---:|---:|---:|
| $J$ | 0.7057 [0.6846, 0.7267] | 0.5102 [0.4620, 0.5606] | 184.0M |
| $J_{\mathrm{cont}}$, same checkpoint | 0.6856 [0.6647, 0.7061] | 0.4662 [0.4223, 0.5123] | 122.7M |
| $T_{\mathrm{cont}}$ | 0.7025 [0.6813, 0.7238] | 0.5348 [0.4909, 0.5809] | 196.1M |
| $T_{\mathrm{macro}}$ | 0.6776 [0.6555, 0.7002] | — | — |
| $D_1$ | 0.6851 [0.6631, 0.7069] | — | — |
| $D_5$ | 0.7378 [0.7193, 0.7565] | — | — |
| $D_{15}$ | 0.7619 [0.7433, 0.7802] | — | — |
| $C-J$, continuous predictor | 0.6999 [0.6827, 0.7170] | 0.4469 [0.4148, 0.4802] | 234.7M |
| No-model action prior | 0.7203 [0.6965, 0.7442] | 0.4180 [0.3633, 0.4727] | 0 model MACs |

The frozen primary contrast is **$J-J_{\mathrm{cont}}=+0.0201$ [0.0154, 0.0248]**. Three qualifications are essential: $J_{\mathrm{cont}}$ is clamped at its largest observed expenditure at both declared budgets; both readings reuse the same endpoint pair; and an accidental duplicate-launch incident violated literal scored-once execution for some fixed-arm prefixes. The retained deterministic records were verified, but this is not a pristine independent replication.[E6]

The narrower mechanistic controls do not reach the margin: $J-T_{\mathrm{cont}}=+0.0032$ [-0.0075, +0.0137], and replaying $J$'s recorded horizons with continuous descriptions gives $J-J_{\mathrm{as-cont}}=+0.0037$ [-0.0021, +0.0095]. $D_{15}$ outperforms $J$ by 0.0562 descriptively. **The defensible claim is therefore a small, qualified trained-policy contrast—not that symbolic feedback or joint selection is generally necessary.**[E6]

Macro semantics remain a limitation. A redesigned temporal macro head reaches held-out F1 **0.157** on parsed carriers, **0.113** on one-step predictions, and only **0.015** on open-loop rollouts. Macro-minus-continuous AUC averaged over horizons is −0.012 under pig removal and negative under all three alternative verdicts. Oracle predicate feedback can hurt, and label-free micro is a non-inferiority result only; neither demonstrates an intrinsic semantic benefit.[E8]

### D. Compute attribution: the reviewers' trunk concern is correct

**Direct answer.** The apparent saving in the submitted cross-system comparison comes primarily from the narrower trunk, not from description selection. At the submitted operating recipe, the exact $J-C-J$ MAC decomposition is:[E6]

| Component | Difference, million MACs |
|---|---:|
| Calls/horizon | −6.353 |
| Description-dependent work | **+6.583** |
| Predictor trunk | **−50.836** |
| Policy work | −0.106 |
| **Total** | **−50.713** |

Description modes add cost. On the same hybrid checkpoint, $J$ costs **61.241M more** than $J_{\mathrm{cont}}$, mainly from additional calls.

Accordingly, we withdraw “matches a horizon-only adaptive baseline at a third of its compute.” The corrected engine-truth comparison gives $J=184.0$M and $C-J=234.7$M MACs, a ratio of **0.784**, with $J-C-J$ AUC only +0.006 [0.000, +0.012]. Fixed coarse continuous $C-F-15$ reaches 0.783 AUC at 27.8M MACs and dominates both; the zero-model prior also lies on the frontier. Wall time does not rescue the claim: the frozen single-seed ratio is 16.74, and a post-hoc three-seed descriptive retime gives 1.63; neither is a speedup for $J$.[E10]

### E. Time base and elapsed-time encoding

**Direct answer.** MSwm and vSA9 identified a real mixed-unit problem in the submitted NovPhy setup, not merely ambiguous terminology. The original base corpus and N1 captures did not use the same physical duration per request index.[E12]

- In the original base-corpus recipe, $\Delta\in\{1,5,15\}$ indexes native fixed-step frames in 60-step windows. The 2,400 predictor-fitting lineages are NovPhy base-corpus lineages, not CLEVRER/Physion lineages.
- In N1, one carrier frame is **50 native engine steps**. Thus N1 $\Delta=1/5/15$ mean 50/250/750 native steps; the same request code denotes a 50-times longer native-step interval than in the base corpus.
- The request code encodes the integer $\Delta$ but carries no physical unit. Separately, the carrier begins with **[has-prior, elapsed seconds]**, and object motion is centre displacement divided by elapsed seconds. The policy also has a distinct remaining-frames input $\kappa$. These three signals are not interchangeable.

The elapsed entry makes observation cadence available, but **we did not ablate that carrier channel and cannot claim that the submitted model used it to reconcile the two rates**. The post-submission $\kappa$ study tests only the policy's remaining-frame input, not the carrier's elapsed channel, and is not evidence of rate invariance. We will state this limitation explicitly.[E12]

The new primary model avoids the particular train/test mismatch: its 480-level fitting split and evaluation split use the same capture pipeline and 50-native-step stride. This does not retroactively repair the submitted mixed-unit model. In the revision we will tabulate corpus, native steps per carrier frame, physical duration, and split role; move submitted-checkpoint results to a clearly labelled developmental appendix; and make no cross-time-base transfer claim. Sub-frame $\Delta$ is unsupported by the existing 50-step captures.[E12]

### F. Exact DP guarantee and second-round contexts

**Direct answer.** vSA9's characterization is correct. The submitted DP exactly optimizes a finite-window **additive fixed-context prediction-error surrogate**, not the true state-dependent recursive rollout cost or task value.

For context $\tilde z_t$, observed target $z_{t+\Delta_r}$, action $a$, and request $r=(\Delta_r,\alpha_r)$, the submitted prediction-error teacher uses

$$
c(t,r)=\Delta_r\,\operatorname{mean}\!\left[\left(F(\tilde z_t,a,r)-z_{t+\Delta_r}\right)^2\right]
+\lambda\,\frac{\operatorname{work}(r,\pi)}{\operatorname{work}(r_{\mathrm{ref}})},
$$

with continuous $\Delta=15$ as the reference, terminal value zero, infeasible overruns masked, and exact ties resolved by frozen request order:

$$
V(t)=\min_{r:\,t+\Delta_r\le L}\left[c(t,r)+V(t+\Delta_r)\right].
$$

Backward induction is exact for this surrogate **given the fixed context tensor**. It does not recompute the continuation graph from every counterfactual predicted successor, and distillation does not guarantee an optimal deployed policy.[E13]

For the aggregation round, the current policy is rolled out greedily from the observed window start. **Visited positions are overwritten by predicted carriers; unvisited positions retain observed carriers.** The next DP evaluates every feasible request from this fixed mixed context against unchanged aligned observed targets. A request that leads to an unvisited position therefore uses that stored observed context in its continuation value; the method does not recursively expand every alternative branch.[E13]

The later decision-valued $J_{\mathrm{dec}}$ uses a different, post-submission candidate-ranking surrogate and must not be attributed to the submitted DP. It improves over the submitted policy recipe by +0.040 [+0.029, +0.052] on grid but remains 0.016 below the hindsight-best fixed request, so it does not overturn the decision-level conclusion.[E14]

---

## Reviewer-specific responses

### Reviewer MMed

**W1/Q1 — overlapping scenes and too few levels.** You are correct. The original CLEVRER scenes 10000–10011 appear in both fitting and evaluation, so the submitted table is not a scene-generalization test. Shared Response A gives the exact old scope, the disjoint new CLEVRER/Physion protocols, and the 400-level NovPhy replacement, including the remaining parser-population, physical-overlap, and near-duplication qualifications. We will remove the original generalization wording and print exact split membership in the supplement.

**W2/Q2 — weak ranking and inconsistent description gains.** We agree. Shared Responses B–C give the original failure, corrected held-out results, and the narrow same-checkpoint contrast. The benefit we can defend is a shared interface and matched-update sharing, not universal macro utility or superior adaptive decisions.

**Q3 — task success.** The original near-chance AUCs did not establish task success; original $J$ had zero top-1 hits on each launch set. The new live-engine measurements and all-arm outcomes are in Shared Response B and Supplement S2. They show useful live behavior for some fixed requests but no demonstrated adaptive-policy advantage, and remain descriptive because of the failed guard.

**Q4 — more complex or closed-loop environments.** We added live retry-from-start execution, physical/appearance-novelty evaluation, and pixel-carrier CLEVRER/Physion studies. None is presented as a general changing-state control result. Under novelty, policy-only retuning gives −0.007 [−0.021, +0.006], whereas retuning the world-model stack gives +0.151 [+0.097, +0.206]; cheap policy-only adaptation is therefore not supported.[E15]

**Presentation.** Figures 1 and 3 will be separated into readable interface and result panels, with exact denominators, comparator definitions, and intervals. The dense development heatmap moves to an appendix; the main text will prioritize the sealed protocol, fixed-request and no-model controls, same-checkpoint comparison, compute frontier, live outcomes, and limitations. We will use the available space for these definitions and results rather than additional introductory prose.

### Reviewer MSwm

**W1 — weak gains over a weak reference.** Agreed. Supplement S1 reports the original $J-F^*$ intervals and all retained per-arm AUC intervals; Shared Response B reports the larger original-checkpoint rerun. Only the original angle interval clears zero, and the new evidence does not establish the submitted adaptivity claim.

**W2 — novelty and missing citations.** We will add Chatterjee and Khardon, *Improving Planning and MBRL with Temporally-Extended Actions* (NeurIPS 2025), and Neitz et al., *Adaptive Skip Intervals: Temporal Abstraction for Recurrent Dynamical Models* (NeurIPS 2018). Per-step temporal adaptation is prior work. Our narrower distinction is a joint $(\Delta,\alpha)$ request interface on one shared carrier, including description feedback and DP-distilled selection. We will also correct TAWM to an **ICML 2025 main-track** paper.[E16]

The empirical novelty claim is narrowed as well: tied-axis restrictions do not show a $\delta$-sized benefit for the submitted-style policy ($J-J_{\mathrm{diag}}=-0.0068$, $J-J_{\mathrm{monotone}}=+0.0028$, $J-H^*=-0.0006$), while the qualified $J-J_{\mathrm{cont}}$ result is reported separately.[E17]

**W3/Q1 — time base and elapsed time.** You are right about the base-corpus/N1 mismatch. Shared Response E gives the exact units and separates request $\Delta$, carrier elapsed seconds, and policy $\kappa$. Whether the submitted model relies on the elapsed channel remains unmeasured; the new primary stack uses one consistent 50-step stride.

**Q2 — counts and numeric intervals.** Supplement S1 gives the original scored levels, successful candidate slots, endpoint coverage, top-1 outcomes, the four $J-F^*$ differences, and every retained per-arm AUC interval. Shared Response A gives the expanded 311/76/78/47 mixed-level counts. Per-level outcome records will accompany the revision rather than being represented only by a colour scale.

**Q3 — held-out Table 1, simple references, and policy schedules.** Shared Response A supplies the new disjoint-scene pixel table, including $J$ and $J_{\mathrm{cont}}$'s actual schedules, and clearly distinguishes it from the original annotation-carrier experiment. Persistence/constant-velocity rows are not completed results; we will add them as scoring-only references and make no claim over them here. Physion's physical-overlap qualification is stated with the table.

**Q4 — compute source.** Primarily the narrower trunk. Shared Response D gives the signed decomposition and withdraws the one-third-compute statement.

**Minor 2 — terminology.** We will define terms at first use: a **lineage** is a generated scene identity and its derived launches/windows kept together in provenance and splitting; the **base corpus** is the original 3,000-training-role-lineage NovPhy collection, with 2,400 predictor-fitting and 600 controller-supervision lineages; **N1 captures** are the later normal-condition, 50-native-step observation campaigns; **development states** are exposed states used for model/selector development rather than sealed evaluation. Main-text claims will use explicit fit/policy/evaluation roles instead of these shorthand labels wherever possible.

### Reviewer vSA9

**W1/Q2 — temporal-rate encoding.** Agreed. Shared Response E identifies elapsed seconds and elapsed-normalized motion as the carrier's rate-bearing signals, distinct from request $\Delta$ and policy $\kappa$. No elapsed-channel ablation establishes rate invariance. The new primary stack is trained and evaluated at a uniform 50-native-step stride; the submitted mixed-rate evidence will be qualified rather than generalized.

**W2 — confounded compute and fixed-context DP.** Both points are correct. Shared Responses C–D provide the same-checkpoint controls and exact compute decomposition; Shared Response F provides the exact surrogate and aggregation-context rule. We will remove any implication of exact recursive-rollout optimality.

**Q1 — scene IDs and recalculation.** The submitted CLEVRER fitting/evaluation scenes were the same 10000–10011. The new experiment uses disjoint retained subsets of 0–9999 and 10000–14999, with exact membership in the frozen manifest. Shared Response A supplies the recalculated errors and explains why this is a new pixel-carrier measurement, not a like-for-like rescue of the original table.

**Q3 — fixed-coordinate and time-only controls.** Shared Response C reports $T_\alpha$, $D_\Delta$, $J_{\mathrm{cont}}$, continuous schedule replay, and $C-J$ on the same improved checkpoint and evaluation inventory. It covers both possible readings of “fixed macro-step”: fixed macro description with adaptive horizon ($T_{\mathrm{macro}}$) and fixed $\Delta=15$ with adaptive description ($D_{15}$). These are equal-coverage comparisons, not equal realized MACs; the only frozen at-most-budget $J-J_{\mathrm{cont}}$ contrast is qualified accordingly.

**Q4 — task success, complete outcomes, and offset +0.004.** The offset effect is not established: the original estimate is +0.0037 [−0.0796, +0.0926], and the original-checkpoint new-level estimate is +0.0111 [−0.0520, +0.0743] with a failed precision guard. On the improved predictor, the corresponding $J-F^*$ offset contrast is −0.090 [−0.126, −0.055]. Supplement S2 reports every live-evaluation arm with pass@1/pass@3, uncertainty, and denominators. Complete level-by-candidate and episode-level success/failure records, including typed failures and no-candidate oracle cases, will be included as supplementary tables; no successful-only subset is used.[E3, E5]

**Q5 — DP objective and unvisited successors.** Shared Response F gives the exact recurrence and the visited-overwrite/unvisited-observed rule. The guarantee applies only to the fixed-context additive teacher, not recursive control. For scalability context, the exact labeller also completes an $N=3,480$, $L=225$, $|R|=225$ pass in 17.8–18.3 GPU seconds in the completed request-set study; this does not improve the policy's decision result.[E14]

---

## Numerical supplement S1 — original development launch inventories

These are the retained submitted-era #96 records, not the new sealed-split results. Intervals are descriptive member-clustered 95% intervals. Successful slots are distinct successful candidates among the informative levels, not successful choices by $J$ and not multiplied by seed. Endpoint coverage denotes successes resolved by frame 225.[E2]

| Inventory | Eligible levels | Mixed levels / cells | Successful slots | Resolved by 225 | $J-F^*$ | Top-1, $J/F^*$ |
|---|---:|---:|---:|---:|---:|---:|
| Grid | 15 | 14 / 42 | 17 | 11/17 | +0.0700 [−0.0057, +0.1437] | 0/42 / 0/42 |
| Offset | 15 | 9 / 27 | 9 | 3/9 | +0.0037 [−0.0796, +0.0926] | 0/27 / 1/27 |
| Angle | 14 | 7 / 21 | 7 | 5/7 | +0.0595 [+0.0079, +0.1131] | 0/21 / 0/21 |
| Power | 15 | 8 / 24 | 8 | 4/8 | +0.0273 [−0.0691, +0.1281] | 0/24 / 0/24 |

Only angle clears zero. On grid, 17/42 cells were fully tied and the primary half-width was 0.0747, so both tie and precision guards failed. $F^*$ is a cross-inventory choice, not a hindsight-best or independently selected oracle.

**All retained request-ranking cells.** Each entry is mean AUC [95% interval]. $T$/$D$ are fixed-coordinate restrictions, $M$ the factorized selector, $OL$ the open-loop schedule, and $C$-$*$ the continuous-family controls; these are not the later same-checkpoint $J_{\mathrm{cont}}$ control.

| Arm | Grid | Offset | Angle | Power |
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
| $J$ | 0.471 [0.416, 0.526] | 0.507 [0.456, 0.559] | 0.565 [0.520, 0.625] | 0.529 [0.469, 0.591] |
| $T$ | 0.498 [0.463, 0.530] | 0.541 [0.507, 0.582] | 0.553 [0.525, 0.579] | 0.523 [0.475, 0.572] |
| $D$ | 0.509 [0.459, 0.568] | 0.530 [0.452, 0.607] | 0.486 [0.389, 0.569] | 0.509 [0.466, 0.556] |
| $M$ | 0.492 [0.435, 0.546] | 0.524 [0.480, 0.574] | 0.571 [0.521, 0.631] | 0.550 [0.507, 0.591] |
| $OL$ | 0.523 [0.474, 0.574] | 0.509 [0.474, 0.544] | 0.543 [0.508, 0.579] | 0.537 [0.483, 0.588] |
| C-F-1 | 0.468 [0.447, 0.487] | 0.520 [0.489, 0.552] | 0.477 [0.433, 0.526] | 0.509 [0.474, 0.550] |
| C-F-5 | 0.486 [0.432, 0.533] | 0.624 [0.528, 0.709] | 0.576 [0.508, 0.650] | 0.605 [0.491, 0.728] |
| C-F-15 | 0.506 [0.430, 0.586] | 0.552 [0.459, 0.641] | 0.475 [0.405, 0.544] | 0.577 [0.496, 0.677] |
| C-$J$ | 0.522 [0.474, 0.566] | 0.546 [0.496, 0.593] | 0.546 [0.488, 0.612] | 0.558 [0.497, 0.606] |

## Numerical supplement S2 — all live task-evaluation arms

Protocol: all 400 N1 levels, retry-from-start, at most three shots, with pass defined as engine pig removal. Model rows use 1,200 level–seed episodes; the seed-free prior/oracle rows use 400; random uses 1,200. Unsuccessful and typed-failure episodes count in the denominator. Intervals are descriptive and level-clustered. The inventory oracle is not a deployable method. **Every formal contrast is guard-vetoed as described in Shared Response B.**[E5]

| Arm | Episodes | Pass@1 | Pass@3 |
|---|---:|---:|---:|
| F-1-continuous | 1,200 | 0.070 [0.045, 0.095] | 0.334 [0.295, 0.375] |
| F-1-micro | 1,200 | 0.070 [0.045, 0.095] | 0.326 [0.287, 0.366] |
| F-1-macro | 1,200 | 0.112 [0.089, 0.135] | 0.343 [0.297, 0.390] |
| F-5-continuous | 1,200 | 0.180 [0.151, 0.210] | 0.453 [0.408, 0.499] |
| F-5-micro | 1,200 | 0.167 [0.138, 0.197] | 0.438 [0.394, 0.481] |
| F-5-macro | 1,200 | 0.107 [0.083, 0.132] | 0.413 [0.371, 0.457] |
| F-15-continuous | 1,200 | 0.189 [0.161, 0.218] | 0.478 [0.434, 0.523] |
| F-15-micro | 1,200 | 0.221 [0.192, 0.251] | 0.458 [0.416, 0.502] |
| F-15-macro ($U$) | 1,200 | 0.151 [0.131, 0.172] | 0.424 [0.384, 0.465] |
| $C^*$ | 1,200 | 0.148 [0.126, 0.172] | 0.466 [0.427, 0.505] |
| $C-J$ | 1,200 | 0.161 [0.136, 0.187] | 0.346 [0.315, 0.378] |
| $J$ | 1,200 | 0.123 [0.098, 0.150] | 0.394 [0.352, 0.438] |
| $J_{\mathrm{dec}}$ | 1,200 | 0.139 [0.117, 0.162] | 0.445 [0.403, 0.487] |
| No-model prior | 400 | 0.140 [0.107, 0.175] | 0.323 [0.278, 0.370] |
| Uniform random | 1,200 | 0.068 [0.053, 0.082] | 0.198 [0.172, 0.223] |
| Inventory oracle | 400 | 0.775 [0.733, 0.815] | 0.775 [0.733, 0.815] |

For calibration, $J$ passes in 473/1,200 episodes and does not pass in 727/1,200, including retained typed failures. The supplement to the revision will contain the per-episode records for every arm rather than only these aggregates.

---

## Claim-level revision summary

| Submitted implication | Revision |
|---|---|
| Scene-overlap rollouts demonstrate generalization | Relabel as in-sample development evidence; use the new disjoint protocols with their stated population and overlap limits |
| The learned selector beats fixed requests | Withdraw; report the original-checkpoint rerun and fixed-request/prior controls |
| Symbolic/macro descriptions generally improve decisions | Narrow to the shared interface and qualified $J-J_{\mathrm{cont}}$ contrast; report macro/readout failures |
| Joint selection provides a threefold compute saving | Withdraw; publish the 0.784 ratio, signed decomposition, and engine-truth frontier |
| Weak AUC gains imply task improvement | Replace with all-arm live outcomes, marked descriptive and guard-vetoed |
| The time base is implicit and transferable | Define all units and channels; acknowledge unresolved original mixed-rate semantics; use uniform-stride training for the new stack |
| DP optimizes recursive rollout cost | State the exact fixed-context surrogate and the observed-context rule for unvisited positions |

The resulting paper makes a smaller but better-supported claim: **a shared request-conditioned world model provides a useful interface for studying prediction granularity, while the present evidence does not show that learned joint selection outperforms the best fixed coarse request or direct task-specific prediction.** We will make that distinction central to the abstract, results, and limitations.

---

## Author-only evidence traceability

| Tag | Primary evidence |
|---|---|
| E1 | [Data foundation #109](https://github.com/Sino-Huang/NovPhy/issues/109); [capture, cohort, and power #104](https://github.com/Sino-Huang/NovPhy/issues/104) |
| E2 | [Original within-checkpoint contrast #96](https://github.com/Sino-Huang/NovPhy/issues/96); retained `issue-96-tau-ad-within-checkpoint-v1` findings/compute artifacts |
| E3 | [Sealed original-checkpoint contrasts in #104](https://github.com/Sino-Huang/NovPhy/issues/104) |
| E4 | [Gate B and integrated status #108](https://github.com/Sino-Huang/NovPhy/issues/108); [decision-relevant dynamics #112](https://github.com/Sino-Huang/NovPhy/issues/112); [perception diagnosis #99](https://github.com/Sino-Huang/NovPhy/issues/99) |
| E5 | [Live engine evaluation #102](https://github.com/Sino-Huang/NovPhy/issues/102) |
| E6 | [Same-checkpoint joint/continuous and restricted controls #114](https://github.com/Sino-Huang/NovPhy/issues/114) |
| E7 | [Pixel benchmarks and split audit #106](https://github.com/Sino-Huang/NovPhy/issues/106) |
| E8 | [Observable macro #113](https://github.com/Sino-Huang/NovPhy/issues/113); [predicate feedback #115](https://github.com/Sino-Huang/NovPhy/issues/115); [cost alignment #120](https://github.com/Sino-Huang/NovPhy/issues/120) |
| E9 | [Shared-predictor and scene-holdout studies #117](https://github.com/Sino-Huang/NovPhy/issues/117) |
| E10 | [Engine-truth compute frontier #103](https://github.com/Sino-Huang/NovPhy/issues/103); [compute correction #122](https://github.com/Sino-Huang/NovPhy/issues/122) |
| E11 | [External-baseline synthesis #127](https://github.com/Sino-Huang/NovPhy/issues/127); [direct predictors #128](https://github.com/Sino-Huang/NovPhy/issues/128); [no-rollout follow-up #141](https://github.com/Sino-Huang/NovPhy/issues/141) |
| E12 | [Request-set/time-stride study #116](https://github.com/Sino-Huang/NovPhy/issues/116); `docs/issue-71-hybrid-readiness.md`; `world_model/data/deployment_temporal.py`; `world_model/planning/scoring_harness.py` |
| E13 | Submitted DP and DAgger implementation: `world_model/training/request_policy.py:112–155` |
| E14 | [Decision-valued policy #101](https://github.com/Sino-Huang/NovPhy/issues/101); [request-set scaling #116](https://github.com/Sino-Huang/NovPhy/issues/116) |
| E15 | [Novelty transfer #107](https://github.com/Sino-Huang/NovPhy/issues/107) |
| E16 | [Chatterjee–Khardon proceedings](https://proceedings.neurips.cc/paper_files/paper/2025/hash/cec445dfc292392af716e9a4fe8de99b-Abstract-Conference.html); [Neitz et al. proceedings](https://proceedings.neurips.cc/paper_files/paper/2018/hash/0f0ee3310223fe38a989b2c818709393-Abstract.html); [TAWM proceedings](https://proceedings.mlr.press/v267/nhu25a.html) |
| E17 | [Independent-axis and hierarchy controls #105](https://github.com/Sino-Huang/NovPhy/issues/105) |
