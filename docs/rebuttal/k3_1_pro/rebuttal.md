# Author response — ICLR 2027 submission #6352

**Paper:** *Granularity Is a Decision: Joint Horizon–Description Selection in Latent World Models*

We thank MMed, MSwm, and vSA9 for identifying concrete failures in the evaluation, comparison design, and presentation. The shared request-conditioned predictor remains the central systems contribution, but the submitted empirical claims were broader than the evidence warrants. In particular, the original CLEVRER rollout table used evaluation contexts from the 12 fitting scenes; the original NovPhy ranking inventories were small and development-exposed; and the submitted “a third of its compute” sentence was based on a confounded comparison. We will correct, rather than defend, those claims.

**Evidence status.** Except where a paragraph explicitly discusses the submitted results, the experiments below are **post-submission work**, not evidence that was already in the paper. Bracketed intervals are descriptive 95% cluster bootstraps, generally over levels with three model seeds; the pre-declared practical AUC margin is $\delta=0.02$. “Supported” below refers only to the frozen rule for the named contrast, not to a broader claim. We distinguish guard failures and unperformed controls explicitly.

## Shared response: what the new evidence changes

### 1. Evaluation validity and generalization (MMed W1/Q1/Q4; MSwm W3/Q3; vSA9 Q1)

**The reviewers are correct about the submitted evaluation.** The original CLEVRER training windows and longer evaluation rollouts came from the same 12 scenes, so that table measured longer-rollout behaviour on fitting scenes, not scene generalization. The original NovPhy ranking result used only 14/9/7/8 mixed development levels for grid/offset/angle/power, respectively, with exposed members and wide intervals. Neither supports the original generalization wording.

We subsequently froze a level-disjoint NovPhy cohort with **480 fit, 120 policy, and 400 sealed evaluation levels**, covering 11,680 evaluation launches. The scored mixed-verdict populations are **311 grid, 76 offset, 78 angle, and 47 power levels**, each with three seeds. All 11,680 scheduled evaluation branches completed. This repairs scene/level isolation and scale, but not every precision target: the original-checkpoint $J-F^*$ contrast is $+0.0134\,[-0.0163,+0.0431]$ on grid, and the required half-width guard fails on every inventory ([data foundation #109](https://github.com/Sino-Huang/NovPhy/issues/109), [capture and held-out contrasts #104](https://github.com/Sino-Huang/NovPhy/issues/104)).

A new decision-relevant predictor improves the sealed grid ranking to **0.7444 [0.7269, 0.7620]**, but the matched continuous predictor scores **0.7834 [0.7676, 0.7992]** and the no-model action prior scores 0.7203. The hybrid-minus-continuous contrast is **−0.0390 [−0.0442, −0.0338]**, so the experiment does not support a hybrid advantage. A wrong-anchor control falls to 0.6122, giving right-minus-wrong $+0.1322\,[0.1216,0.1430]$, which verifies anchor dependence but not superiority ([Gate B in #108](https://github.com/Sino-Huang/NovPhy/issues/108)).

Two further qualifications are material:

* On held-out novelty pairs, retuning only the request policy does **not** help: for the decision-valued policy, policy-only-minus-zero-shot is **−0.007 [−0.021,+0.006]**, whereas predictor-only adaptation improves AUC by **+0.151 [+0.097,+0.206]**. Thus “request selection is a cheap adaptation surface” is not supported ([novelty transfer #107](https://github.com/Sino-Huang/NovPhy/issues/107)).
* Strong direct success predictors outperform the world-model ranker on the sealed grid. In particular, the direct predictor using the **same carrier, matched parameters, and fit shots** reaches **0.9757 [0.9692,0.9815]**, versus 0.7444 for the world-model unit. A no-rollout classifier attached to the same carrier reaches **0.9809 [0.9748,0.9863]**, compared with 0.9765 when it reads the frozen model’s predicted endpoint; the rollout-minus-no-rollout difference is −0.0044, with no supported advantage in either direction. On novel levels, the direct predictors fall to 0.52–0.58 and the world model to 0.47. We therefore cannot claim that world-model prediction adds ranking value over direct prediction for this task ([direct predictors #128](https://github.com/Sino-Huang/NovPhy/issues/128), [no-rollout control #141](https://github.com/Sino-Huang/NovPhy/issues/141)).

**Revision:** we will restrict the contribution to the request interface, shared-predictor formulation, and the specifically supported contrasts; remove broad scene-generalization and world-model-superiority claims; and report the direct-predictor, action-prior, novelty, and wrong-anchor controls next to the main results.

### 2. Description-level benefit and the same-checkpoint ablation (MMed W2/Q2; MSwm W2; vSA9 Q3)

**There is no established general benefit of symbolic descriptions over continuous prediction.** The requested ablation has now been run on the **same hybrid checkpoint**, with the same policy architecture, initialization, data, and DP/distillation recipe, masking the six noncontinuous logits for `J-cont`.

At the submitted operating recipe, joint `J` scores 0.7057 [0.6846,0.7267] and same-checkpoint continuous-only `J-cont` scores 0.6856 [0.6647,0.7061]. The frozen at-most-budget contrast is **$J-J_{\rm cont}=+0.0201\,[0.0154,0.0248]$**, just above the 0.02 margin. This is qualified evidence for that particular comparison, not a general symbolic advantage:

* `J-cont` is clamped at its largest observed expenditure (122.7M MACs) at both declared budgets, so these are **at-most-budget**, not exactly equal-spend, comparisons; both budgets reuse the same endpoint contrast.
* The narrower checks $J-T_{\rm continuous}=+0.0032\,[-0.0075,+0.0137]$ and $J-J_{\rm as\text{-}cont}=+0.0037\,[-0.0021,+0.0095]$ do not meet the margin.
* `J` remains below the no-model prior and the best fixed continuous requests.
* Accidental duplicate deterministic scoring violated literal scored-once execution for some fixed-arm prefixes. The retained records reproduce exactly, but this is not a pristine one-look confirmatory run ([same-checkpoint ablation #114](https://github.com/Sino-Huang/NovPhy/issues/114)).

The macro level specifically does **not** show a sustained benefit. A redesigned observable macro head reaches held-out F1 0.157 on parsed carriers, 0.113 on one-step predicted carriers, and only **0.015 in open-loop rollout**. On the original Gate-B predictor, macro-minus-continuous AUC averaged over horizons is **−0.012 [−0.019,−0.004]** under pig removal and −0.036 under the least-aligned alternative verdict. Even a retrained macro-reference predictor’s $+0.022$ advantage under pig removal vanishes or reverses under alternative verdicts ([macro readability #113](https://github.com/Sino-Huang/NovPhy/issues/113), [cost alignment #120](https://github.com/Sino-Huang/NovPhy/issues/120)). Oracle predicate feedback can hurt rather than help; the label-free micro result is only non-inferiority, not evidence of a symbolic benefit ([feedback robustness #115](https://github.com/Sino-Huang/NovPhy/issues/115)).

**Revision:** replace “symbolic levels improve prediction/decision-making” with the narrower statement that one shared predictor can service all nine requests, and that the same-checkpoint joint-versus-continuous policy contrast reaches the pre-declared margin under the qualifications above. Macro utility remains an unresolved limitation.

### 3. Compute attribution and correction (MSwm W2/Q4; vSA9 W2)

**The submitted compute sentence should be withdrawn.** On the sealed grid at the submitted operating point, `J` costs **184.0M** linear MACs per candidate-state and `C-J` costs 234.7M: a ratio of **0.784**, not “a third.” Their AUC difference is only $+0.006\,[0.000,0.012]$. Fixed coarse continuous `C-F-15` dominates both, scoring 0.783 at 27.8M MACs. The no-model prior also dominates both at zero model MACs ([engine-truth frontier #103](https://github.com/Sino-Huang/NovPhy/issues/103)).

The decomposition at $\lambda=10^{-4}$ makes the source of the apparent saving explicit. For $J-C-J$, the total difference is −50.713M MACs: **−50.836M from the trunk**, −6.353M from calls/horizon, **+6.583M from description**, and −0.106M from policy work. Thus micro/macro calls add cost; they do not cause the saving. For $J-J_{\rm cont}$, the joint policy actually spends +61.241M more, mainly +53.460M from additional calls. These are linear-MAC accounts, not full FLOPs or universal wall-time savings. The frozen single-seed wall comparison and a post-hoc three-seed retime differ substantially, so we will not use wall time to rescue the claim.

**Revision:** remove every “third of its compute” statement and the unmatched proxy comparison; report the full AUC–MAC frontier and the fixed coarse and no-model references. The surviving compute result is descriptive: a joint policy at another operating point ($\lambda=0.1$) lies on the measured MAC frontier, but no adaptive policy is shown to dominate the best fixed coarse model.

## Reviewer MMed

### W1/Q1 — Overlapping CLEVRER scenes and limited NovPhy evaluation

**Direct answer:** the overlap is real, and the original table does not assess scene generalization. We agree that this is a validity defect, not merely missing wording. The new sealed NovPhy cohort and its limitations are given in Shared Response 1.

We also completed a fresh pixel-based CLEVRER/Physion-Dominoes study using the official train/validation or release splits. For CLEVRER, fitting scene IDs are the parser-compatible subset of **0–9999** and validation scene IDs are the parser-compatible subset of **10000–14999**; their intersection is empty. The exact 6,285 and 3,132 IDs are frozen in the experiment manifest. Repeated colour/material keys exclude 37.22% of videos, so this is a parser-compatible subpopulation, not full CLEVRER. Physion has no full-static-parameter overlap after content hashing, but a colour-excluded audit finds 20/150 test trials sharing physical parameters with fitting trials, including four near-identical trajectories; it therefore does not establish generalization to independent physical realizations ([pixel benchmarks #106](https://github.com/Sino-Huang/NovPhy/issues/106)).

The new results do not show a downstream joint-selection benefit: CLEVRER $J-J_{\rm cont}=-0.0001\,[-0.0002,+0.0001]$ is dominated by a readout floor, and Physion OCP gives $-0.021\,[-0.068,+0.025]$, with insufficient precision. We will present these as limitations and remove the original scene-overlap table from any generalization argument.

### W2/Q2 — Weak ranking, inconsistent description gains, and presentation

**Direct answer:** the original ranking evidence is weak, and the description levels do not consistently help. We will not interpret small gains over a near-chance reference as successful decision-making. The original per-request intervals and the post-submission same-checkpoint answer are provided under MSwm Q2 and Shared Response 2.

The strongest defensible shared-model result is not a symbolic advantage but a sharing result: at matched per-request updates on sealed NovPhy, joint-minus-specialist request-mean AUC is **+0.0848 [+0.0742,+0.0952]**, with non-inferiority at every request. Full-budget specialists can outperform the joint model, which indicates an update-allocation limitation rather than unrestricted superiority of sharing ([sharing ablation #117](https://github.com/Sino-Huang/NovPhy/issues/117)).

**Revision:** Figures 1 and 3 will be split into readable panels with one comparison per panel, exact denominators and intervals in captions, and no dense nine-request heatmap as the primary evidence. The main text will contain the sealed split, same-checkpoint, fixed-request, action-prior, and live-task tables; development-only heatmaps and qualitative rollout panels move to the appendix. We will use the available main-text space to define the protocol and state the negative results rather than expand the introduction.

### Q3 — Does weak action ranking translate into task success?

**Direct answer:** the submitted near-chance AUCs did not establish useful task success. We subsequently executed live engine trials on all 400 sealed levels using retry-from-start with at most three shots. Descriptively, the new predictor’s arms exceed uniform random on N1, but fixed coarse requests are best and the joint policy is not. For example, pass@3 is 0.478 [0.434,0.523] for fixed (15, continuous), 0.394 [0.352,0.438] for the submitted-style joint policy, 0.323 [0.278,0.370] for the no-model prior, and 0.198 [0.172,0.223] for random. The complete table and its guard caveat appear under vSA9 Q4. These results are **descriptive only**, because a frozen decision-frame guard failed; they do not convert the original marginal AUC gains into a confirmed success improvement ([closed-loop evaluation #102](https://github.com/Sino-Huang/NovPhy/issues/102)).

### Q4 — More complex or closed-loop environments

**Direct answer:** we have now evaluated live action execution, but only in a limited retry-from-start protocol. Every shot resets the same level and excludes previously failed grid candidates; the underlying state does not evolve between replans. It is therefore not evidence of general closed-loop control in a changing environment. N2 appearance trials also show no model-minus-prior interval clearing zero. The new CLEVRER and Physion experiments broaden the environments but fail to establish a downstream selection benefit, as detailed under MSwm Q3.

As an additional external check, H-JEPA was reproduced on its own FourRoom benchmark, then collapsed to launch-insensitive predictions on action-sparse NovPhy; its inverse-dynamics/action-SIGReg remedy did not prevent that collapse. Because those baselines fail their guards, **no superiority or state-of-the-art claim is licensed** ([H-JEPA #123](https://github.com/Sino-Huang/NovPhy/issues/123), [anti-collapse follow-up #125](https://github.com/Sino-Huang/NovPhy/issues/125)). We will not use a collapsed baseline as evidence for our method.

## Reviewer MSwm

We thank MSwm for the precise time-base observation and for requesting the same-checkpoint ablation; both exposed genuine defects in the submitted evidence.

### W1 — Near-chance AUC and a weak reference

**Direct answer:** agreed. In the submitted development evaluation, $J-F^*$ is +0.0700 [−0.0057,+0.1437] on grid, +0.0037 [−0.0796,+0.0926] on offset, +0.0595 [0.0079,+0.1131] on angle, and +0.0273 [−0.0691,+0.1281] on power. Only angle clears zero; the grid precision and tie guards fail. The reference itself is weak, and Top-1 hits are 0/42 for both $J$ and $F^*$ on grid. These data cannot support a practical ranking improvement ([original contrast #96](https://github.com/Sino-Huang/NovPhy/issues/96)).

The larger sealed rerun improves ranking quality but removes the adaptivity conclusion: on the original checkpoint, $J$ itself scores 0.555 [0.539,0.571] on sealed grid and $J-F^*$ is unresolved; on the new predictor, the best fixed continuous model outranks the hybrid and adaptive policies. We will frame the original figure as a development diagnostic, not as evidence of successful selection.

### W2 — Prior work and narrower novelty

**Direct answer:** the missing citations and the scope of novelty should be corrected. We will cite:

* Chatterjee and Khardon, *Improving planning and MBRL with temporally-extended actions*, NeurIPS 2025, which optimizes action duration and automatically selects duration ranges using a bandit formulation ([proceedings record](https://proceedings.neurips.cc/paper_files/paper/2025/hash/cec445dfc292392af716e9a4fe8de99b-Abstract-Conference.html));
* Neitz et al., *Adaptive Skip Intervals: Temporal Abstraction for Recurrent Dynamical Models*, NeurIPS 2018, which lets a recurrent dynamics model choose its prediction sampling rate ([proceedings record](https://proceedings.neurips.cc/paper_files/paper/2018/hash/0f0ee3310223fe38a989b2c818709393-Abstract.html));
* TAWM as an **ICML 2025 main-track** paper, not a workshop paper ([PMLR record](https://proceedings.mlr.press/v267/nhu25a.html)).

Our novelty statement will not claim per-step temporal adaptation as new. The remaining distinction is the **joint $(\Delta,\alpha)$ request interface on one shared carrier**, including description feedback and DP-distilled request selection. Even that is a formulation contribution, not a demonstrated performance advantage: tied-axis and hierarchical restrictions do not show a $\delta$-sized gain for the submitted-style policy, with $J-J_{\rm diag}=-0.0068$, $J-J_{\rm monotone}=+0.0028$, and $J-H^*=-0.0006$ on sealed grid ([axis restrictions #105](https://github.com/Sino-Huang/NovPhy/issues/105)). The matched-checkpoint result in Shared Response 2 is the narrow positive exception, with its stated qualifications.

### W3/Q1 — What does $\Delta$ count, and is elapsed time used?

**Direct answer:** the original NovPhy base corpus and N1 captures do **not** use one consistent physical time base. In the original base-corpus training, $\Delta$ counts native fixed steps in 60-step windows; the N1 deployment carrier renders one frame per 50 native steps. With the original 0.0004-second timestep, base $\Delta=1/5/15$ correspond to 0.4/2/6 ms, whereas N1 $\Delta=1/5/15$ correspond to 20/100/300 ms. Thus N1 $\Delta=15$ is indeed 50 times the native-step duration of base $\Delta=15$. We agree that the submitted wording obscured this and that the original results cannot be interpreted as clean time-base transfer. The original protocol records are explicit about the old 1/5/15-step models and the newer 50/250/750-native-step horizons ([base training record](https://github.com/Sino-Huang/NovPhy/blob/env/docs/issue-71-hybrid-readiness.md), [native time-base record](https://github.com/Sino-Huang/NovPhy/blob/env/docs/issue-76-native-findings.md)).

The carrier does contain a separate **elapsed-seconds entry** between the prior and current observations, together with a has-prior flag; object motion is centre displacement divided by that elapsed time. This is distinct from the request code $\Delta$ and from the policy’s remaining-frames feature $\kappa$. The elapsed channel makes rate information available to the predictor, but the submitted experiments do **not** isolate whether it resolves the mixed training-time semantics, so we will not claim that it does. The post-submission sealed NovPhy model is trained and evaluated on the same 50-native-step N1 carrier, avoiding that particular train/test mismatch for the new result only; it does not retroactively repair the original base-corpus experiment.

**Revision:** define native step, carrier frame, request horizon, elapsed seconds, and remaining frames separately; print physical durations beside $\Delta$; and remove any cross-time-base transfer claim.

### Q2 — Scored levels, successful launches, and numeric intervals

**Direct answer:** the original scored populations and endpoint coverage are:

| Launch set | Mixed levels | Cells (levels × 3 seeds) | Resolved / successful endpoint slots | Top-1 hits, $J/F^*$ |
|---|---:|---:|---:|---:|
| Grid | 14 | 42 | 11 / 17 | 0/42 / 0/42 |
| Offset | 9 | 27 | 3 / 9 | 0/27 / 1/27 |
| Angle | 7 | 21 | 5 / 7 | 0/21 / 0/21 |
| Power | 8 | 24 | 4 / 8 | 0/24 / 0/24 |

The sealed expansion has 311/76/78/47 mixed levels for grid/offset/angle/power. On all 400 sealed N1 levels, 311 have a passing grid launch and 89 do not; the live oracle’s descriptive pass@3 is 0.775 rather than 1 because of that inventory ceiling and one typed engine failure.

The four Fig. 1(b2) differences $J-F^*$ are:

| Grid | Offset | Angle | Power |
|---|---|---|---|
| +0.0700 [−0.0057,+0.1437] | +0.0037 [−0.0796,+0.0926] | +0.0595 [+0.0079,+0.1131] | +0.0273 [−0.0691,+0.1281] |

For completeness, the original Fig. 4 nine-request cells are below. Each entry is mean AUC [descriptive 95% interval].

| Request | Grid | Offset | Angle | Power |
|---|---|---|---|---|
| (1, continuous) | 0.510 [0.483,0.536] | 0.517 [0.481,0.556] | 0.520 [0.500,0.544] | 0.509 [0.488,0.536] |
| (1, micro) | 0.493 [0.464,0.520] | 0.500 [0.489,0.511] | 0.502 [0.446,0.560] | 0.520 [0.504,0.542] |
| (1, macro) | 0.513 [0.467,0.563] | 0.504 [0.444,0.559] | 0.506 [0.423,0.583] | 0.527 [0.474,0.592] |
| (5, continuous) | 0.415 [0.337,0.480] | 0.574 [0.507,0.648] | 0.581 [0.497,0.663] | 0.458 [0.374,0.549] |
| (5, micro) | 0.425 [0.368,0.484] | 0.576 [0.491,0.659] | 0.609 [0.554,0.671] | 0.469 [0.379,0.546] |
| (5, macro) | 0.401 [0.348,0.461] | 0.594 [0.517,0.674] | 0.635 [0.587,0.679] | 0.501 [0.414,0.606] |
| (15, continuous) | 0.492 [0.411,0.584] | 0.441 [0.348,0.504] | 0.450 [0.323,0.561] | 0.462 [0.403,0.528] |
| (15, micro) | 0.479 [0.395,0.578] | 0.487 [0.415,0.544] | 0.450 [0.359,0.524] | 0.452 [0.399,0.520] |
| (15, macro) | 0.491 [0.455,0.526] | 0.522 [0.489,0.559] | 0.534 [0.500,0.569] | 0.499 [0.438,0.559] |

These values make the weakness visible rather than hiding it behind point estimates. We will include the table in the appendix and replace the overcrowded main-text heatmap with the sealed results.

### Q3 — Held-out Table 1, simple baselines, and policy schedules

**Direct answer:** we recomputed long-rollout errors on scene-disjoint CLEVRER validation and the documented Physion release testing split, including the policies’ own schedules. The last-endpoint position-MSE results are:

| Arm | CLEVRER, $t=120$ | Physion-Dominoes, $t=90$ |
|---|---|---|
| (1, continuous) | 0.163 [0.152,0.176] | 0.0430 [0.0327,0.0535] |
| (1, micro) | 0.164 [0.153,0.176] | 0.0460 [0.0343,0.0591] |
| (1, macro) | 0.180 [0.163,0.197] | 0.0660 [0.0482,0.0894] |
| (5, continuous) | 0.109 [0.104,0.115] | 0.0227 [0.0184,0.0272] |
| (5, micro) | 0.109 [0.104,0.114] | 0.0231 [0.0186,0.0278] |
| (5, macro) | 0.108 [0.103,0.112] | 0.0256 [0.0199,0.0320] |
| (15, continuous) | 0.0996 [0.0958,0.104] | 0.0143 [0.0108,0.0182] |
| (15, micro) | 0.0993 [0.0956,0.103] | 0.0145 [0.0109,0.0187] |
| (15, macro) | 0.0974 [0.0944,0.101] | 0.0146 [0.0110,0.0187] |
| $J$, own schedule | 0.157 [0.147,0.166] | 0.0235 [0.0188,0.0285] |
| $J_{\rm cont}$, own schedule | 0.162 [0.152,0.173] | 0.0252 [0.0206,0.0303] |

Thus coarse horizons still reduce recursive error in this new pixel-carrier study, while the learned policies stay close to $\Delta=1$ and do not inherit that advantage. **The requested persistence/constant-velocity baselines were not included in this run**, so we will not claim that comparison. The CLEVRER result covers the parser-compatible subpopulation, and the Physion split has the near-duplicate qualification above. These are rollout errors against parsed carriers, not evidence of downstream task success ([pixel benchmark protocol and results #106](https://github.com/Sino-Huang/NovPhy/issues/106)).

### Q4 — Source of compute saving

**Direct answer:** primarily the narrower trunk in the submitted comparison, not the description level. The exact decomposition is in Shared Response 3: description adds +6.583M MACs, while the trunk term is −50.836M for $J-C-J$. Horizon/call count contributes −6.353M and policy work −0.106M. On the same hybrid checkpoint, joint selection costs more than continuous-only selection because it makes more calls. The original compute claim is therefore withdrawn.

### Minor points

1. **TAWM venue:** corrected to ICML 2025 main track, PMLR 267:46265–46287.
2. **Undefined terms:** we will define them at first use and add a split/provenance diagram. Specifically, a **lineage** is the generated initial scene and all shots/windows derived from that same scene identity; the **base corpus** is the frozen pre-N1 collection used for predictor fitting; **N1 captures** are the normal, single-bird, 50-native-step carrier captures used for the later deployment-aligned evaluation; and **development states** are exposed states available for fitting, selection, or diagnostics, as distinct from the sealed evaluation levels. These labels will not be used interchangeably with train/policy/evaluation split names.

## Reviewer vSA9

### W1/Q2 — Temporal-rate encoding and transferability

**Direct answer:** the submitted description is inadequate, and the original mixed base/N1 time base prevents a clean transferability interpretation. The exact physical durations and scoped correction are under MSwm W3/Q1.

There are three distinct signals:

1. the request code $\Delta$, which chooses the learned transition horizon;
2. the carrier’s has-prior and **elapsed-seconds** fields, with motion normalized by elapsed time;
3. the policy’s remaining-frames input $\kappa$.

The predictor can therefore receive elapsed-rate information, but no submitted ablation establishes that this information is sufficient to reconcile mixed-rate training. We will state this as an unresolved limitation rather than infer a resolution from the architecture.

### W2 — Confounded compute comparison and the DP surrogate

**Direct answer:** both criticisms are correct. The original `C-J` comparison changes more than the policy’s allowed request set, so it cannot attribute savings to joint representation selection. The same-checkpoint `J-cont` ablation now isolates the policy restriction, while the corrected `C-J` comparison shows that the submitted “third” sentence is false. Full numbers are in Shared Responses 2–3.

The DP guarantee is also narrower than the submitted discussion implies. It exactly optimizes an additive, fixed-context prediction-error surrogate by backward induction; it does **not** optimize the true state-dependent recursive rollout cost. The exact objective and successor construction are given under Q5 below.

### Q1 — Scene IDs and a recomputed table

**Direct answer:** the submitted 12-scene CLEVRER fitting/evaluation sets overlap, so we cannot supply a no-overlap interpretation of that table. In the new experiment, fitting uses parser-compatible scene IDs within 0–9999 and evaluation uses parser-compatible IDs within 10000–14999, with zero overlap; exact frozen membership is available in the #106 manifest. Physion uses 2,000 dynamics-training, 996 readout-training, and 150 release-testing trials, with the physical-overlap qualification stated above. The recomputed rollout table, including the policies’ own schedules, appears under MSwm Q3. It does not establish a downstream joint-selection benefit.

### Q3 — Same-checkpoint and equivalent-budget policy comparisons

**Direct answer:** the same-checkpoint continuous-only contrast is now available, and all restricted selectors use the same hybrid predictor and `J` logits unless retraining is stated. At $\lambda=10^{-4}$, the sealed-grid results are:

| Selector | Meaning | Grid AUC | Top-3 hit |
|---|---|---:|---:|
| $J$ | joint horizon + description | 0.7057 [0.6846,0.7267] | 0.5102 [0.4620,0.5606] |
| $J_{\rm cont}$ | adaptive horizon, continuous only; separately distilled with masked labels | 0.6856 [0.6647,0.7061] | 0.4662 [0.4223,0.5123] |
| $T_{\rm cont}$ | adaptive horizon at fixed continuous description | 0.7025 [0.6813,0.7238] | 0.5348 [0.4909,0.5809] |
| $T_{\rm micro}$ | adaptive horizon at fixed micro description | 0.7044 [0.6827,0.7258] | — |
| $T_{\rm macro}$ | adaptive horizon at fixed macro description | 0.6776 [0.6555,0.7002] | — |
| $D_1$ | adaptive description at fixed $\Delta=1$ | 0.6851 [0.6631,0.7069] | — |
| $D_5$ | adaptive description at fixed $\Delta=5$ | 0.7378 [0.7193,0.7565] | — |
| $D_{15}$ | adaptive description at fixed $\Delta=15$ | 0.7619 [0.7433,0.7802] | — |
| $C-J$ | adaptive horizon on the separate continuous predictor | 0.6999 [0.6827,0.7170] | 0.4469 [0.4148,0.4802] |

The primary frozen at-most-budget result is $J-J_{\rm cont}=+0.0201\,[0.0154,0.0248]$, but $J-C-J$ does not reach $\delta$, nor do $J-T_{\rm cont}$ or the continuous schedule replay. The restricted $T/D$ rows are same-recipe descriptive comparisons, **not exact equal-MAC comparisons**; fixed coarse $D_{15}$ is stronger than $J$ here. We will report the entire frontier rather than claim that all joint choices win at equal compute ([restricted selectors #114](https://github.com/Sino-Huang/NovPhy/issues/114)).

### Q4 — Task success with uncertainty, complete outcomes, and the offset gain

**Direct answer:** the original offset gain of approximately +0.004 does **not** hold as a supported result. On sealed offset levels, the original-checkpoint $J-F^*$ estimate is **+0.0111 [−0.0520,+0.0743]**, with a failed precision guard; for the new Gate-B predictor, $J-F^*$ is **−0.090 [−0.126,−0.055]**. The answer is therefore no: the original marginal offset gain is not robust ([held-out contrasts #104](https://github.com/Sino-Huang/NovPhy/issues/104)).

We also ran live engine execution on all 400 N1 levels and a separate 40-level N2-appearance set. The following is the complete arm-level N1 table, with N2 pass@3 alongside it. N1 cells are 400 levels × 3 seeds; N2 cells are 40 × 3. Typed engine failures count as not passed and were not retried. All intervals are descriptive.

| Arm | N1 pass@1 | N1 pass@2 | N1 pass@3 | N2 pass@3 |
|---|---:|---:|---:|---:|
| (1, continuous) | 0.070 [0.045,0.095] | 0.203 [0.165,0.242] | 0.334 [0.295,0.375] | 0.167 [0.067,0.283] |
| (1, micro) | 0.070 [0.045,0.095] | 0.197 [0.166,0.228] | 0.326 [0.287,0.366] | 0.158 [0.058,0.267] |
| (1, macro) | 0.112 [0.089,0.135] | 0.226 [0.191,0.262] | 0.343 [0.297,0.390] | 0.175 [0.075,0.300] |
| (5, continuous) | 0.180 [0.151,0.210] | 0.333 [0.293,0.374] | 0.453 [0.408,0.499] | 0.208 [0.100,0.333] |
| (5, micro) | 0.167 [0.138,0.197] | 0.343 [0.302,0.384] | 0.438 [0.394,0.481] | 0.200 [0.092,0.317] |
| (5, macro) | 0.107 [0.083,0.132] | 0.295 [0.258,0.333] | 0.413 [0.371,0.457] | 0.192 [0.083,0.308] |
| (15, continuous) | 0.189 [0.161,0.218] | 0.333 [0.295,0.371] | 0.478 [0.434,0.523] | 0.233 [0.125,0.358] |
| (15, micro) | 0.221 [0.192,0.251] | 0.350 [0.313,0.388] | 0.458 [0.416,0.502] | 0.217 [0.108,0.333] |
| (15, macro), Gate-B unit | 0.151 [0.131,0.172] | 0.301 [0.268,0.335] | 0.424 [0.384,0.465] | 0.217 [0.117,0.325] |
| $C^*$, continuous (15) | 0.148 [0.126,0.172] | 0.371 [0.332,0.412] | 0.466 [0.427,0.505] | 0.225 [0.125,0.333] |
| $C-J$ | 0.161 [0.136,0.187] | 0.296 [0.263,0.329] | 0.346 [0.315,0.378] | 0.158 [0.083,0.242] |
| $J$ | 0.123 [0.098,0.150] | 0.242 [0.205,0.282] | 0.394 [0.352,0.438] | 0.183 [0.075,0.308] |
| $J_{\rm dec}$ | 0.139 [0.117,0.162] | 0.298 [0.266,0.331] | 0.445 [0.403,0.487] | 0.217 [0.117,0.333] |
| No-model prior | 0.140 [0.107,0.175] | 0.268 [0.225,0.312] | 0.323 [0.278,0.370] | 0.175 [0.075,0.300] |
| Uniform random | 0.068 [0.053,0.082] | 0.140 [0.118,0.162] | 0.198 [0.172,0.223] | 0.100 [0.042,0.175] |
| Oracle inventory ceiling | 0.775 [0.733,0.815] | 0.775 [0.733,0.815] | 0.775 [0.733,0.815] | 0.500 [0.350,0.650] |

**Important caveat:** 25 of 5,106 live decision frames on five levels differed from their frozen anchors by up to 0.0032, exceeding the 0.001 guard tolerance. Consequently every formal contrast token is `readiness_or_precision_insufficient`, even where a descriptive interval excludes zero. A post-hoc rerank left the first three picks unchanged in 351/351 comparisons, but that does not repair the guard. The table answers task outcomes descriptively; it is not confirmatory evidence of an adaptive-policy advantage ([live evaluation #102](https://github.com/Sino-Huang/NovPhy/issues/102)).

For additional context, native heuristic agents acting with continuous actions reached descriptive pass@3 values of 0.307 (Datalab), 0.233 (Eagle’s Wing), and 0.217 (PigShooter). Their retry mechanisms differ from the grid-candidate protocol, so these are not paired contrasts ([native agents #132](https://github.com/Sino-Huang/NovPhy/issues/132)).

### Q5 — Exact DP objective and unvisited successor contexts

**Direct answer:** for the **submitted prediction-error policy**, at window position $t$, action $a$, and request $r=(\Delta_r,\alpha_r)$, define

$$
c_t(r)=\Delta_r\,\frac{1}{D}\left\|F_r(\tilde z_t,a)-z_{t+\Delta_r}\right\|_2^2
+\lambda\,\frac{w(r,\pi)}{w(r_{\rm ref})},
$$

where $F_r$ is the shared predictor, $\tilde z_t$ is the current fixed context carrier, $z_{t+\Delta_r}$ is the aligned observed target, $w(r,\pi)$ counts request and policy linear MACs, the denominator $w(r_{\rm ref})$ excludes policy work, and $r_{\rm ref}$ is continuous $\Delta=15$. With terminal value $V(L)=0$ and infeasible requests masked, backward induction computes

$$
V(t)=\min_{r:\,t+\Delta_r\le L}\left[c_t(r)+V(t+\Delta_r)\right],
\qquad
r^*(t)=\arg\min_{r:\,t+\Delta_r\le L}\left[c_t(r)+V(t+\Delta_r)\right].
$$

This recursion is exact for the stated finite-window additive surrogate on the fixed context tensor, with the frozen request order breaking exact ties. It is not a guarantee about recursive model rollout error or task value. The decision-valued $J_{\rm dec}$ discussed elsewhere is a **new post-submission variant** and is not the submitted objective ([policy evaluation #101](https://github.com/Sino-Huang/NovPhy/issues/101)).

For DAgger aggregation, the round-0 policy performs a greedy rollout. **Visited positions are overwritten by predicted carriers; unvisited positions retain their observed carriers.** The next DP then evaluates *every* feasible request from this fixed context tensor. A request not taken by the greedy policy therefore does not receive a freshly simulated successor; its value reads the stored context at $t+\Delta_r$, while its immediate prediction target remains the aligned observed carrier. Distillation trains on the pooled round-0 and aggregation-round labels. This construction exposes the policy to some predicted states, but it also explains why the DP is only a fixed-context surrogate, not exact recursive-rollout optimization.

## Revision summary

The revision will make five claim-level changes:

1. replace scene-overlap and small-development-set evidence with the sealed, level-disjoint evaluation, retaining all failed precision guards and split qualifications;
2. narrow the contribution to the shared $(\Delta,\alpha)$ request interface and same-checkpoint contrast, with no general symbolic or macro benefit claim;
3. withdraw the “one-third compute” claim and publish the engine-truth frontier, fixed coarse references, action prior, and compute decomposition;
4. present live task success as descriptive retry-from-start evidence, not proof of general closed-loop control or task-success superiority;
5. cite Chatterjee–Khardon, Neitz et al., and the correct TAWM venue, define the time-base and provenance terminology, and rebuild the main figures around readable tables with denominators and intervals.

The resulting paper is narrower than the submission: it demonstrates a flexible shared request-conditioned model and several useful diagnostics, but it does not establish that learned joint selection outperforms the best fixed coarse request, that symbolic descriptions improve decisions generally, or that the original ranking gains generalize to robust task success.
