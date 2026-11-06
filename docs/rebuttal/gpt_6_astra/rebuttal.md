# Rebuttal — ICLR 2027 submission #6352

**Granularity Is a Decision: Joint Horizon–Description Selection in Latent World Models**

We thank **MMed, MSwm, and vSA9** for identifying the distinction that the paper needs to make much more clearly: a model can expose horizon and description as selectable inputs without establishing that its learned selector improves decisions, that symbolic descriptions help, or that the resulting system is cheaper. We agree that the submitted experiments did not establish all of these stronger claims.

We have completed additional experiments since submission. They supply a substantially larger evaluation, the missing same-checkpoint continuous-only policy, more informative controls, and direct task measurements. They also reveal important negative results. Below we distinguish these **completed post-submission results** from the **manuscript revisions we propose** and the **limitations that remain unresolved**. We do not present the new experiments as part of the original submission.

## Shared response: what the new evidence does—and does not—establish

### 1. A larger, identity-disjoint evaluation replaces the tiny original sample, but does not establish broad generalization

The original NovPhy evaluation reused members exposed to predictor or policy development. Of its 15 members, eight had predictor-fitting exposure and three had policy-development exposure; only four were unexposed, and only three of those contributed mixed-outcome grid cells. The original generalization interpretation was therefore too strong.

The new protocol froze **480 predictor-fitting, 120 policy-development, and 400 evaluation levels**, equally divided between rolling and sliding families, before capture. Its identity audit found no reuse against 83 recorded exposure sources. All **11,680 evaluation candidates** completed with no typed capture failures. AUC is defined only where an inventory contains both successful and unsuccessful candidates: the resulting evaluation has **311 grid, 76 offset, 78 angle, and 47 power mixed-outcome levels**. Three training seeds do not turn these into three times as many independent levels. The power inventory also falls below its planned minimum of 53 informative levels. [Sealed membership and audit: #109](https://github.com/Sino-Huang/NovPhy/issues/109); [completed evaluation and qualifications: #104](https://github.com/Sino-Huang/NovPhy/issues/104).

Identity separation is not equivalent to physical diversity. A subsequent nearest-neighbor control achieved grid AUC **0.9039 [0.8838, 0.9231]** by using the nearest fitting example in pixel space. This exposes the narrowness and near-duplication of the two-family distribution, not evidence that held-out labels were used for fitting. We therefore describe the new evaluation as **identity-disjoint, within-distribution evaluation**, not generalization to independently diverse physical problems. [#128](https://github.com/Sino-Huang/NovPhy/issues/128#issuecomment-6492639730).

### 2. The strongest positive result is narrower than the submitted headline

For clarity, let **J** denote a joint horizon–description policy; **J-cont** a separately trained, continuous-only policy using the **same hybrid predictor checkpoint**; **C-J** a horizon policy using a separately trained pure-continuous predictor; and **C\*** the fixed-horizon pure-continuous reference. These are different controls and must not be conflated.

| Completed result | Estimate and 95% interval | What it supports |
|---|---|---|
| Same hybrid checkpoint: J − trained J-cont, at the declared 184.0M-MAC comparison point | **+0.0201 [0.0154, 0.0248] AUC** | A positive, checkpoint- and protocol-specific effect of allowing joint selection; not an exact equal-spend comparison |
| Shared predictor − request specialists, mean over requests, matched updates per request | **+0.0848 [0.0742, 0.0952] AUC** | Beneficial parameter sharing under this training-budget definition; the frozen test was noninferiority |
| Improved fixed hybrid unit U = F(15, macro) | **0.7444 [0.7269, 0.7620] AUC** | Above-chance ordering on the larger grid; U is a fixed request, not an adaptive policy |
| U − C\* | **−0.0390 [−0.0442, −0.0338] AUC** | No hybrid advantage over the strongest matched continuous reference |
| U − launch prior | **+0.0241 [−0.0008, 0.0487] AUC** | Insufficient evidence of improvement over this zero-model prior |

The J-cont frontier ends at **122.7M MACs**, whereas J uses **184.0M** at the first comparison point. The preregistered interpolation clamps to that endpoint. The second declared budget, 234.7M, reuses the same endpoint comparison; it is not an independent replication. We report the positive contrast, but **not** equal-spend superiority, Pareto dominance, or a general advantage over the best fixed request. At the same policy operating point, F(15, continuous) obtains **0.7619 [0.7433, 0.7802]**, versus J's **0.7057 [0.6846, 0.7267]**. [Same-checkpoint contrast and deviations: #114](https://github.com/Sino-Huang/NovPhy/issues/114#issuecomment-6169850625); [sharing: #117](https://github.com/Sino-Huang/NovPhy/issues/117); [Gate B: #108](https://github.com/Sino-Huang/NovPhy/issues/108).

The sharing result is also budget-specific. The joint model receives 9,000 updates, or 1,000 per request; each specialist receives 1,000 updates, and the nine-specialist bank receives 9,000 in aggregate. Giving **each** specialist 9,000 updates changes the comparison: the F(15, macro) specialist reaches AUC 0.885 versus 0.744 for the shared model. We do not claim that sharing dominates independently trained specialists regardless of training budget.

### 3. Direct success prediction is a strong counterexample to a broader world-model claim

The post-submission controls include a direct candidate-success predictor with the same carrier input, approximately matched parameter count (**1,838,781 versus 1,837,690**), and the same 9,000-update budget. It reaches **0.9757 [0.9692, 0.9815] AUC**, exceeding U by **0.2313 [0.2131, 0.2493]**. Thus, these experiments do **not** establish that a rollout world model is necessary or best for this candidate-ranking problem. A further learned readout over U's rollout endpoints reaches **0.9765 [0.9689, 0.9828]**, but a matched separately trained classifier without rollout endpoints reaches **0.9809 [0.9748, 0.9863]**; their difference is **−0.0044 [−0.0081, −0.0014]**. This does not demonstrate incremental value from the endpoints, nor establish equivalence between the models. [Direct baselines and diversity audit: #128](https://github.com/Sino-Huang/NovPhy/issues/128#issuecomment-6492639730); [readout isolation: #141](https://github.com/Sino-Huang/NovPhy/issues/141#issuecomment-6492488967).

Our proposed revision therefore centers the contribution on a **shared request-conditioned model, an explicit joint selection formulation, and controlled measurements separating parameter sharing, selection, representation, and downstream utility**. It removes claims of general task superiority, intrinsic macro-description benefit, and a threefold compute saving. The framework remains testable and useful, but the empirical conclusions must be scoped to the contrasts actually supported.

Unless otherwise stated, intervals below are level/member-clustered 95% bootstrap intervals; they are not multiplicity-adjusted. The frozen decision rule's `supported` label requires an estimate of at least 0.02 AUC, a positive lower bound, and passing guards. That label is **not** a confidence statement that the true effect is at least 0.02. We retain failed-guard dispositions even when a numerical interval appears favorable.

# Response to MMed

## MMed Q1 and evaluation weakness: overlapping CLEVRER scenes and limited generalization

**The original same-scene CLEVRER result is not a scene-generalization test.** Its 12 scenes were exactly **10000–10011**, used for both training and evaluation windows. Separating windows from these scenes does not remove shared scene structure. The appropriate interpretation is within-scene prediction, and we propose labeling the original table accordingly rather than defending it as independent generalization.

A completed pixel-based extension instead trains on **6,285 eligible scenes from CLEVRER's official training range 0–9999** and evaluates on **3,132 eligible scenes from the validation range 10000–14999**, with no shared scene IDs and five training seeds. Exact retained scene lists are in the frozen plan, not just the numeric ranges. Eligibility requires unique color–material object identities and excludes **37.22%** of the combined source scenes. Consequently, this is a filtered validation-set experiment, not evaluation on every CLEVRER scene or on the hidden test set. Its numbers are given in the response to MSwm Q3 below. [#106, including its corrective qualification](https://github.com/Sino-Huang/NovPhy/issues/106#issuecomment-6245842498).

The NovPhy identity-disjoint extension and its diversity limitation are described above. Neither experiment supports a broad claim that joint selection generalizes across physical settings. In particular, the new CLEVRER downstream comparison is essentially null, and the Physion result remains inconclusive.

## MMed Q2: what is the benefit of symbolic rather than continuous descriptions?

**The evidence does not establish an intrinsic task-accuracy benefit from macro descriptions.** “Description” is a conditioning choice within the shared predictor, not a replacement of the state by a guaranteed faithful symbolic world. All requests still predict the full continuous carrier. Micro and macro heads supply learned features to adapters and a shared conditioned trunk; their semantic fidelity must be evaluated separately from rollout error or ranking AUC.

Three completed checks constrain the interpretation:

1. **Selection versus a continuous-only policy.** The same-checkpoint J-versus-J-cont contrast above is positive, but allows a different schedule and different compute use. It does not isolate semantic macro reasoning. Restricting the existing J logits to continuous requests produces an even smaller contrast: **J − T-cont = +0.0032 [−0.0075, 0.0137]**. Replaying J's horizon sequence with continuous descriptions gives **+0.0037 [−0.0021, 0.0095]**. Neither establishes the 0.02-AUC benefit. [#114](https://github.com/Sino-Huang/NovPhy/issues/114).
2. **Semantic fidelity.** The current macro head had zero event F1 in the audited setup. A new temporally aligned label/head experiment improves parsed-state F1 to **0.157 [0.151, 0.164]**, but one-step predicted F1 is only **0.113 [0.107, 0.119]**, and rollout F1 falls to **0.015 [0.014, 0.015]**. The experiment changes both supervision and training; it does not isolate a purely temporal mechanism. These results are not evidence of a reliable symbolic rollout. [#113](https://github.com/Sino-Huang/NovPhy/issues/113#issuecomment-6220087939).
3. **Task dependence and label controls.** On the improved checkpoint, macro-minus-continuous AUC, averaged across horizons, is **−0.012 [−0.019, −0.004]** for pig removal, **−0.024 [−0.029, −0.018]** for block destruction, **−0.036 [−0.048, −0.025]** for displacement, and **−0.078 [−0.091, −0.065]** for early contact. There is no macro advantage across these objectives. Removing micro-label supervision passes the specified noninferiority tests; supplying oracle runtime labels does not rescue the presumed benefit. The label experiments therefore do not establish that symbolic semantic content is the source of the small selection effects. [#120](https://github.com/Sino-Huang/NovPhy/issues/120#issuecomment-6244323864); [#115](https://github.com/Sino-Huang/NovPhy/issues/115#issuecomment-6228202344).

We propose replacing the claim that symbolic descriptions improve decisions with the narrower statement that **description is an exposed, experimentally testable conditioning variable; its semantic quality and task benefit are separate empirical questions**. The benefit currently best supported is shared training across requests, not macro superiority.

## MMed Q3: does AUC improvement translate into task success?

**Not automatically, and not convincingly in the original experiment.** AUC measures whether a successful candidate is ranked ahead of a failed candidate, averaged over success–failure pairs within a level; it is not the probability of selecting a successful top-ranked action. The original J policy had **zero top-1 hits** in all four inventory evaluations: 0/42 grid, 0/27 offset, 0/21 angle, and 0/24 power level–seed cells. Those denominators contain only 14, 9, 7, and 8 distinct informative levels, respectively. They are descriptive, highly underpowered task results, not evidence of improvement.

We have now measured live retry outcomes on all 400 normal levels, including all-failure cases rather than restricting to mixed-outcome AUC cells. J achieves first-attempt pig-removal success **0.1233 [0.0983, 0.1500]** and success within three attempts **0.3942 [0.3517, 0.4383]**. The fixed continuous Δ=15 model reaches **0.4783 [0.4342, 0.5233]** within three attempts, and C\* reaches **0.4658 [0.4267, 0.5050]**. The complete arm table and the failed reproducibility guard are given under vSA9 Q4. These data do not support the claim that J's AUC improvement reliably becomes superior task success. [Original task counts: #96](https://github.com/Sino-Huang/NovPhy/issues/96#issuecomment-5831289359); [live task experiment: #102](https://github.com/Sino-Huang/NovPhy/issues/102#issuecomment-6203235032).

We also distinguish **pig-removal success** from **native level completion**. For example, J's native level-clear rate within three attempts is 0.2367, not 0.3942. The manuscript should not use the two outcomes interchangeably.

## MMed Q4: more complex settings and closed-loop decisions

The completed novelty-transfer experiment covers **29 informative novelty cells** across the available novelty families and levels 1–7, totaling 122 mixed-outcome level cells before seed replication. Sixteen other novelty cells are not informative for the decision comparison, including unsupported multi-force dynamics and cases without mixed outcomes. Representing their inputs is not the same as demonstrating successful transfer.

Policy-only adaptation does not improve ordering over zero-shot use: **−0.007 [−0.021, 0.006] AUC**. Retuning the world-model stack gives **+0.151 [0.097, 0.206]**, whereas policy-only adaptation trails world-model retuning by **−0.158 [−0.214, −0.104]**. The retuned stack includes the encoder and scene head as well as the transition predictor, so the gain cannot be attributed to predictor-only adaptation. Adding policy adaptation to world-model retuning gives **−0.006 [−0.020, 0.009]**. We therefore retract any implication that inexpensive policy-only changes are sufficient for the tested novelty. [#107](https://github.com/Sino-Huang/NovPhy/issues/107#issuecomment-6207410868).

The live task experiment is a **retry-from-start** test, not a genuine changing-state, multi-shot controller. These are single-bird levels: after failure, the system resets to the same starting anchor, excludes failed candidates, and reranks the remaining candidates. It does not demonstrate replanning after a world-changing action in a continuing episode. On the additional 40 novel levels, model pass-within-three rates range approximately from 0.158 to 0.233; the launch prior is 0.175, and every model-minus-prior interval includes zero. Genuine state-changing closed-loop control and robust complex-scene generalization remain unestablished. [#102](https://github.com/Sino-Huang/NovPhy/issues/102).

## Presentation: Figures 1 and 3 and use of main-text space

We agree that the crowded figures obscure the actual evidence. Our proposed revision separates the request interface from the evaluation figure, gives the horizon and description axes separate legends, and replaces miniature results panels with readable plots accompanied by numeric estimates, intervals, distinct-level counts, and comparator definitions. Figure 3 should show the shared full-carrier output and the conditioning heads explicitly, rather than suggesting three incomparable output spaces. We would move the temporal-unit definition, split table, DP guarantee, and the decisive same-checkpoint controls into the main text. These are concrete manuscript changes proposed in this response, not additional experiments claimed as completed.

# Response to MSwm

## Novelty and omitted related work

We agree that **learning or selecting temporal abstraction is not new**. The paper should cite and distinguish:

- Chatterjee and Khardon, **“Improving planning and MBRL with temporally-extended actions,” NeurIPS 2025**, which learns/plans action durations and studies their role in model-based control. [Paper](https://arxiv.org/abs/2505.15754).
- Neitz et al., **“Adaptive Skip Intervals: Temporal Abstraction for Recurrent Dynamical Models,” NeurIPS 2018**, which learns adaptive temporal skips for sequence prediction. [Proceedings](https://proceedings.neurips.cc/paper_files/paper/2018/hash/0f0ee3310223fe38a989b2c818709393-Abstract.html).

Our distinction is the **joint request `(horizon, description)` supported by a shared full-carrier predictor and a policy over the Cartesian product**, not the horizon axis alone. This is an interface/formulation distinction; it does not establish that joint adaptation is empirically necessary. The controlled contrasts below restrict the claim further: a joint-versus-factorized comparison and a state-conditioned-versus-open-loop comparison do not show the declared practical advantage.

The TAWM citation also needs correction: Nhu, Son, and Lin, **“Time-Aware World Model for Adaptive Prediction and Control,” is an ICML 2025 main-conference paper**, PMLR 267:46265–46287, not a workshop paper. TAWM's explicit time-aware formulation is relevant to our rate limitation; we should not imply that a historical elapsed-time carrier field establishes the same ability. [Official proceedings](https://proceedings.mlr.press/v267/nhu25a.html).

## MSwm Q1: elapsed time in Equation 9, and the base-corpus/N1 time mismatch

**Elapsed time is present, but three signals need to be distinguished.**

1. The observed temporal carrier starts with **`[has_prior, elapsed_seconds]`**. When a previous observation is available, elapsed time is the difference between its fixed-time timestamp and the current timestamp. Object motion features are displacement divided by that elapsed time, with availability indicators. When no prior is available, elapsed time and motion are zero and marked unavailable. Thus the input does contain a physical-time quantity.
2. The request **Δ ∈ {1,5,15}** is a skip in the trajectory's frame/index units. It is not itself an elapsed-seconds field.
3. The policy's **κ** input encodes remaining frames, separately from the carrier. It is a budget-to-go signal, not a substitute for the sampling interval.

The shared predictor consumes the carrier and the request code, but the implementation does not provide evidence that it has learned sampling-rate invariance or reliably converts every request to a rate-independent physical duration. The elapsed field is also part of the predicted carrier, not a separately guaranteed physical clock in a recursive rollout. We propose making these distinctions explicit next to Equation 9. [Carrier construction](https://github.com/Sino-Huang/NovPhy/blob/env/world_model/data/deployment_temporal.py); [request policy and remaining-frame input](https://github.com/Sino-Huang/NovPhy/blob/env/world_model/training/request_policy.py).

**Your 2400/2409 observation concerns the original NovPhy training corpus, and we agree that its time-base mismatch is substantive.** The predictor-fitting pool combined 2,400 base NovPhy lineages with nine N1 training members. The base corpus used native fixed-step windows, whereas N1 observations used a 50-native-step stride. Consequently, identical integer request codes did not automatically denote identical physical durations across those components. The documented fixed timestep is 0.0004 seconds: one native step is 0.0004 seconds, whereas an N1 frame separated by 50 steps is 0.02 seconds. Thus Δ=15 denotes 0.006 seconds on the former indexing and 0.30 seconds on the latter. This is not merely a difference between NovPhy and CLEVRER/Physion.

The elapsed carrier entry makes the distinction observable in principle; it does **not** retrospectively prove that the original mixed training learned to handle it. Later native-horizon experiments distinguish 1/5/15-step models from 50/250/750-native-step horizons, but they do not erase the original ambiguity. We propose documenting the indexing, sampling interval, and physical duration separately for every corpus and qualifying the original transfer interpretation. A controlled rate-transfer test remains missing. [Original readiness and training-window description](https://github.com/Sino-Huang/NovPhy/blob/env/docs/issue-71-hybrid-readiness.md); [native-step comparison](https://github.com/Sino-Huang/NovPhy/blob/env/docs/issue-76-native-findings.md).

## MSwm Q2: exact sample sizes, successful launches, and all plotted intervals

We agree that a plot alone is inadequate here. The following gives the original four inventory comparisons underlying Figure 1(b2). Counts of successful candidate launches are **distinct candidate slots among the informative levels**, not multiplied by the three model seeds; endpoint coverage shows how many successful slots had resolved by the model's 225-frame scoring horizon.

| Inventory | Eligible levels | Mixed-outcome levels / level–seed cells | Successful candidate slots | Successes resolved by frame 225 | J − cross-fitted F\*, AUC [95% CI] |
|---|---:|---:|---:|---:|---:|
| Grid | 15 | 14 / 42 | 17 | 11/17 | +0.0700 [−0.0057, 0.1437] |
| Offset | 15 | 9 / 27 | 9 | 3/9 | +0.0037 [−0.0796, 0.0926] |
| Angle | 14 | 7 / 21 | 7 | 5/7 | +0.0595 [0.0079, 0.1131] |
| Power | 15 | 8 / 24 | 8 | 4/8 | +0.0273 [−0.0691, 0.1281] |

The angle inventory excludes member `issue-77-n1-005` under the inherited typed-unmeasurable declaration. All-failure/all-success levels do not contribute AUC; they must not disappear from task-success denominators. The success-resolution column also exposes a real limitation: a predictor scored at frame 225 need not have seen the eventual outcome used to label a candidate.

Only the angle interval clears zero. On the primary grid, 17/42 level–seed cells had a fully tied J or F\* ranking, exceeding the 20% tie guard; the interval half-width was 0.0747, exceeding the 0.05 precision guard. The primary disposition is therefore **readiness/precision insufficient**, not a positive adaptation result. Moreover, the cross-inventory-selected F\* is F(5, macro), the worst fixed pair on this grid at AUC 0.4006. Against the grid's hindsight-best fixed pair, the difference reverses to **−0.0420 [−0.1295, 0.0475]**. Cross-inventory selection also does not create level-disjoint model selection; we should not call the resulting intervals optimism-free. [Full original experiment: #96](https://github.com/Sino-Huang/NovPhy/issues/96#issuecomment-5831289359).

Here are the numeric AUCs and intervals for **all nine fixed-request cells in each inventory**, replacing reliance on Figure 4's color scale:

| Request | Grid | Offset | Angle | Power |
|---|---|---|---|---|
| (1, continuous) | .5104 [.4827, .5355] | .5167 [.4815, .5556] | .5198 [.5000, .5437] | .5092 [.4883, .5362] |
| (1, micro) | .4932 [.4639, .5198] | .5000 [.4889, .5111] | .5020 [.4464, .5595] | .5198 [.5044, .5417] |
| (1, macro) | .5126 [.4675, .5626] | .5037 [.4444, .5593] | .5060 [.4226, .5833] | .5274 [.4737, .5921] |
| (5, continuous) | .4145 [.3367, .4796] | .5741 [.5074, .6481] | .5808 [.4975, .6627] | .4582 [.3737, .5493] |
| (5, micro) | .4253 [.3683, .4840] | .5759 [.4907, .6593] | .6093 [.5538, .6706] | .4688 [.3794, .5461] |
| (5, macro) | .4006 [.3475, .4612] | .5944 [.5167, .6741] | .6346 [.5873, .6786] | .5014 [.4135, .6056] |
| (15, continuous) | .4918 [.4108, .5836] | .4407 [.3481, .5037] | .4502 [.3232, .5614] | .4618 [.4026, .5276] |
| (15, micro) | .4791 [.3951, .5783] | .4870 [.4148, .5444] | .4502 [.3589, .5238] | .4516 [.3991, .5196] |
| (15, macro) | .4912 [.4545, .5261] | .5222 [.4889, .5593] | .5335 [.5000, .5693] | .4994 [.4380, .5592] |

These are descriptive clustered intervals on the original small sample, not corrected significance tests over the table. [Retained numerical summary](https://github.com/Sino-Huang/NovPhy/blob/env/.local-artifacts/issue-96-tau-ad-within-checkpoint-v1/summary.json).

The larger identity-disjoint **original-checkpoint rerun** does not rescue a universal selection claim: J − F\* is **+0.0134 [−0.0163, 0.0431]** on grid, **+0.011 [−0.052, 0.074]** on offset, **−0.160 [−0.206, −0.113]** on angle, and **−0.262 [−0.336, −0.192]** on power. All inventories fail their frozen precision requirement, so their dispositions remain insufficient rather than being upgraded from the signs of their intervals. These are distinct from the improved-checkpoint Gate B results. [#104](https://github.com/Sino-Huang/NovPhy/issues/104#issuecomment-6247382540).

## MSwm Q3: held-out CLEVRER and documented Physion, including policy-schedule errors

The completed extension addresses scene-ID separation on CLEVRER and supplies a documented Physion split. It is a **new pixel-based experimental configuration**, not an unchanged rerun of the original Table 1. Training uses annotations for supervision; held-out carrier extraction uses RGB. The table reports position MSE against the **parsed carrier**, not ground-truth physical-state error, for 120-frame CLEVRER and 90-frame Physion rollouts.

| Rollout schedule | CLEVRER, disjoint scene IDs | Physion, official split |
|---|---:|---:|
| F(1, continuous) | .163 [.152, .176] | .0430 [.0327, .0535] |
| F(1, micro) | .164 [.153, .176] | .0460 [.0343, .0591] |
| F(1, macro) | .180 [.163, .197] | .0660 [.0482, .0894] |
| F(5, continuous) | .109 [.104, .115] | .0227 [.0184, .0272] |
| F(5, micro) | .109 [.104, .114] | .0231 [.0186, .0278] |
| F(5, macro) | .108 [.103, .112] | .0256 [.0199, .0320] |
| F(15, continuous) | .0996 [.0958, .1040] | .0143 [.0108, .0182] |
| F(15, micro) | .0993 [.0956, .1030] | .0145 [.0109, .0187] |
| F(15, macro) | .0974 [.0944, .1010] | .0146 [.0110, .0187] |
| J | .157 [.147, .166] | .0235 [.0188, .0285] |
| J-cont | .162 [.152, .173] | .0252 [.0206, .0303] |

The CLEVRER split is specified above; its downstream J-minus-J-cont QA accuracy difference is **−0.0001 [−0.0002, 0.0001]**. This task readout is at a floor: the maximum collision probability is only 0.051 against a 0.5 threshold, and accuracy is approximately 0.525. Consequently, this does not establish useful reasoning accuracy even though the rollout MSE is measurable.

For Physion, the experiment uses **2,000 dynamics-training, 996 readout-training, and 150 testing videos** in the documented red/yellow subset. Filename lists and split membership are retained in the plan. A physical-content audit materially limits the interpretation: six layouts recur, and removing color from the parameter identities exposes **20/150 testing videos with matching physical parameters** in a training partition; four testing trajectories are within the specified 10⁻³ proximity tolerance. Official split membership is therefore not proof of independently held-out physical realizations. Downstream J AUC is **0.517 [0.445, 0.589]**, J-cont is **0.539 [0.467, 0.612]**, and the paired difference is **−0.021 [−0.068, 0.025]**. An observed-anchor readout reaches 0.627 [0.549, 0.704], emphasizing that successful extrapolative prediction has not been established. [Results and complete membership: #106](https://github.com/Sino-Huang/NovPhy/issues/106#issuecomment-6243043951); [physical-overlap correction](https://github.com/Sino-Huang/NovPhy/issues/106#issuecomment-6245842498).

**Two parts of your request remain unmet:** this extension does not provide the requested persistence and constant-velocity baselines, and it does not supply a newly trained, physical-parameter-group-disjoint Physion comparison. We should state that explicitly rather than presenting the table as a complete resolution. The schedules were trained under the same recipe, not matched to equal inference MACs. These limitations prevent the table from supporting a broad accuracy/efficiency superiority claim.

## MSwm Q4: compute attribution and the narrower predictor trunk

**The architecture confound is real.** Macro requests still predict the full carrier; a “symbolic” request is not intrinsically a cheaper narrow output. The pure-continuous and hybrid systems have different trunk costs, so their aggregate MAC difference cannot be attributed to jointly selecting descriptions.

At the reported operating point, J uses **184.0M MACs** versus **234.7M** for C-J: a ratio of **0.784**, or about 1.28× fewer MACs, not threefold savings. The corrected decomposition of J − C-J is:

| Component | Difference in millions of MACs |
|---|---:|
| Transition-call count | −6.353 |
| Description-dependent work | +6.583 |
| Predictor trunk | **−50.836** |
| Policy work | −0.106 |
| Total | **−50.713** |

The architecture term explains essentially all of the aggregate reduction. Holding the hybrid predictor fixed changes the story: J costs **61.241M more MACs** than J-cont, decomposed as +53.460M from transition-call count, +6.583M from description work, and +1.199M from policy work, with zero trunk difference. This is why we do not use the positive J − J-cont AUC contrast as evidence of compute saving. [#114](https://github.com/Sino-Huang/NovPhy/issues/114).

The wider efficiency claim is also contradicted by the frontier. C\* reaches approximately **0.783 AUC at 27.8M MACs**, numerically dominating the cited 184.0M/234.7M operating points. A lower-cost J point reaches approximately **0.770 at 22.1M**, but always uses Δ=15; its advantage over fixed F(15, continuous) is **0.008 [0.005, 0.010]**, below the declared 0.02 margin. This cannot be advertised as a practical benefit from adaptive horizons. The zero-model prior reaches about 0.720 AUC at zero model MACs and belongs in the comparison. [#103](https://github.com/Sino-Huang/NovPhy/issues/103).

MACs are not latency. The corrected timing record gives a frozen single-seed J/C-J wall-time ratio of **16.74** and a separate post-hoc three-seed descriptive ratio of **1.63** (0.138/0.085 seconds). Neither is a measured speedup for J. The #114 run also had duplicate workers on fixed prefixes, violating literal score-once execution; deterministic agreement does not turn these into independent replications, and contention-affected timing is not isolated latency. We propose replacing the original compute headline with the component accounting above and labeling timing by its actual protocol. [Timing correction: #103](https://github.com/Sino-Huang/NovPhy/issues/103#issuecomment-6208129522); [execution disclosure: #114](https://github.com/Sino-Huang/NovPhy/issues/114#issuecomment-6213932132).

## Minor terminology points

We propose defining the terms at first use and adding a compact provenance table:

- **Lineage:** a source trajectory/capture identity grouping related windows or descendants; windows from one lineage are not independent scene examples. Distinct lineage IDs alone do not guarantee distinct physical layouts.
- **Base corpus:** the original NovPhy native-fixed-step trajectory corpus, distinguished from subsequently captured N1 data and from CLEVRER/Physion.
- **N1 captures:** the normal/non-novel NovPhy capture set; the original training bundle contains 12 members, nine for predictor fitting and three for policy development. They are not all held-out evaluation levels.
- **Development states:** states used for policy fitting, request-label construction, or model/selector development; they must be distinguished from predictor-fitting samples and untouched evaluation states.

The corrected table should identify membership, sampling interval, physical horizon, fitting/development/evaluation role, and any overlap for each source. The TAWM venue correction is given above.

# Response to vSA9

## vSA9 Q1–Q2: exact scene split and temporal-rate encoding

The original CLEVRER scenes are **10000–10011**, with training/evaluation scene overlap; we do not claim otherwise. The completed scene-disjoint extension uses retained subsets of **0–9999 for fitting** and **10000–14999 for validation**, with full ID lists in the frozen plan. The counts, eligibility exclusions, all fixed-request errors, and policy errors appear in MSwm Q3. This answers the scene-ID issue for the new CLEVRER experiment, while the physically overlapping Physion split remains explicitly qualified. [#106](https://github.com/Sino-Huang/NovPhy/issues/106).

For time, the carrier's **historical elapsed seconds**, the model's **request Δ**, and the policy's **remaining-frame κ** are distinct. The elapsed channel and displacement-per-second features exist, but a request code is still trajectory-indexed. The original base/N1 training components used different physical durations per index. Accordingly, neither the architecture description nor the new benchmark results establish sampling-rate transfer. We propose correcting both the carrier equation and the dataset/horizon table, without inventing a rate-invariance result.

## vSA9 Q3: same-checkpoint fixed-coordinate controls and time-only schedules

The missing **trained** continuous-only policy has now been completed. We additionally distinguish it from merely restricting the logits of the trained joint policy. The following controls use the same improved hybrid checkpoint, anchors, candidate inventory, endpoint, and verdicts. They have equal decision coverage, but schedule-dependent transition counts and **not equal realized MACs**.

| Policy/control | Grid AUC [95% CI] | Interpretation |
|---|---:|---|
| J: joint horizon–description | .7057 [.6846, .7267] | Reference adaptive policy |
| J-cont: separately trained continuous-only | .6856 [.6647, .7061] | Same predictor; matched policy-training recipe |
| T-cont: fixed continuous description, adaptive horizon | .7025 [.6813, .7238] | Restricted J logits, not retrained |
| T-micro: fixed micro description, adaptive horizon | .7044 [.6827, .7258] | Restricted J logits, not retrained |
| T-macro: fixed macro description, adaptive horizon | .6776 [.6555, .7002] | Restricted J logits, not retrained |
| D1: fixed Δ=1, adaptive description | .6851 [.6631, .7069] | Restricted J logits |
| D5: fixed Δ=5, adaptive description | .7378 [.7193, .7565] | Restricted J logits |
| D15: fixed Δ=15, adaptive description | .7619 [.7433, .7802] | Restricted J logits |
| J-as-cont: replay J's horizon sequence with continuous requests | .7021 [.6805, .7231] | Holds the horizon trace fixed |

The paired differences are more informative than overlap between marginal intervals. J − T-cont is **+0.0032 [−0.0075, 0.0137]**; J − J-as-cont is **+0.0037 [−0.0021, 0.0095]**; J − D5 is **−0.0320 [−0.0416, −0.0230]**; and J − D15 is **−0.0562 [−0.0685, −0.0441]**. These controls do not demonstrate that adapting both coordinates is necessary. The positive trained J − J-cont contrast and its endpoint-clamping caveat are stated in the shared response rather than being generalized to all restrictions. [#114](https://github.com/Sino-Huang/NovPhy/issues/114#issuecomment-6169850625).

A separate post-submission **decision-aligned policy, J_dec**, improves over the original prediction-error policy recipe on the improved checkpoint by **+0.040 [0.029, 0.052] AUC**. However, it remains below the best fixed comparator by **−0.0163 [−0.0203, −0.0122]**, differs from the factorized selector by only **+0.001 [−0.001, 0.002]**, and differs from the open-loop, state-independent schedule by **−0.003 [−0.006, 0.000]**. Thus the new training objective does not establish a practical advantage from jointness or state dependence. J_dec is a new variant, **not** the objective used in the submitted DP. [#101, including the corrected interval](https://github.com/Sino-Huang/NovPhy/issues/101#issuecomment-6208129050).

We have supplied the same-checkpoint coordinate restrictions and open-loop control; we have **not** established superiority over every time-only schedule under exact equal inference spending. The revised claims should reflect that boundary.

## vSA9 Q4: task success, all outcomes, uncertainty, and the +0.004 offset effect

**The +0.004 offset estimate is not a stable demonstrated benefit.** The original value is **+0.0037 [−0.0796, 0.0926]**. On the new identity-disjoint original-checkpoint evaluation it is **+0.011 [−0.052, 0.074]**, with insufficient precision. The latter is a new sample, not an exact numerical replication, and neither interval establishes a positive effect.

For the original experiment, the complete J/F\* top-1 counts are grid **0/42 versus 0/42**, offset **0/27 versus 1/27**, angle **0/21 versus 0/21**, and power **0/24 versus 0/24**. A degenerate zero-hit bootstrap interval must not be read as certainty that population success is zero. The number of independent levels is small, and these data cannot resolve the claimed small ranking effects as task improvements. [#96](https://github.com/Sino-Huang/NovPhy/issues/96).

The completed live retry test provides a more direct denominator: **all 400 normal levels**, with each model evaluated across three seeds. The table includes every normal-level arm, not selected successful examples. “Pass” means the declared pig-removal outcome, not native level completion. A failure to pass—including a retained typed execution failure—counts in the denominator.

| Arm | Evaluated level–seed episodes | First-attempt pass [95% CI] | Pass within three attempts [95% CI] |
|---|---:|---:|---:|
| F(1, continuous) | 1,200 | .0700 [.0450, .0950] | .3342 [.2950, .3750] |
| F(1, micro) | 1,200 | .0700 [.0450, .0950] | .3258 [.2875, .3658] |
| F(1, macro) | 1,200 | .1117 [.0892, .1350] | .3425 [.2975, .3900] |
| F(5, continuous) | 1,200 | .1800 [.1508, .2100] | .4533 [.4083, .4992] |
| F(5, micro) | 1,200 | .1667 [.1383, .1967] | .4375 [.3942, .4808] |
| F(5, macro) | 1,200 | .1067 [.0825, .1317] | .4133 [.3708, .4567] |
| F(15, continuous) | 1,200 | .1892 [.1608, .2183] | .4783 [.4342, .5233] |
| F(15, micro) | 1,200 | .2208 [.1917, .2508] | .4583 [.4158, .5025] |
| F(15, macro), U | 1,200 | .1508 [.1308, .1717] | .4242 [.3842, .4650] |
| C\* | 1,200 | .1483 [.1258, .1717] | .4658 [.4267, .5050] |
| C-J | 1,200 | .1608 [.1358, .1867] | .3458 [.3150, .3775] |
| J | 1,200 | .1233 [.0983, .1500] | .3942 [.3517, .4383] |
| J_dec | 1,200 | .1392 [.1175, .1617] | .4450 [.4025, .4875] |
| Launch prior | 400 | .1400 [.1075, .1750] | .3225 [.2775, .3700] |
| Random | 1,200 | .0675 [.0533, .0817] | .1975 [.1717, .2225] |
| Retained-inventory oracle ceiling | 400 | .7750 [.7325, .8150] | .7750 [.7325, .8150] |

These intervals cluster by level; the repeated seeds are not independent new scenes. All unsuccessful outcomes are included in the displayed rates. For example, J passes within three attempts in **473/1,200** episodes and does not pass in **727/1,200**. The oracle is an inventory ceiling, not a deployable competing method.

Crucially, the experiment **failed its frozen carrier-replay guard**: 25/5,106 completed live frames across five levels exceeded the 0.001 carrier-agreement tolerance, reaching 0.0032. There were **17 retained typed execution failures** among 5,123 distinct live executions; these were not retried or discarded. The complete normal-plus-novel evaluation contains 19,360 arm–seed episodes, sharing executions where the selected candidate is identical. A post-hoc check found unchanged top-three decisions in 351/351 audited cases, but that does **not** retroactively pass the frozen guard. All confirmatory contrast dispositions therefore remain **readiness/precision insufficient**. We report the rates as descriptive evidence, not as a certified improvement in task success. [Full experiment, outcomes, guard failure, and retained records: #102](https://github.com/Sino-Huang/NovPhy/issues/102#issuecomment-6203235032).

Finally, as stated to MMed, this is retry-from-start, not state-changing multi-shot control. Full outcome accounting is an improvement over selective examples, but does not resolve that control limitation.

## vSA9 Q5: exactly what the DP guarantees, and unvisited round-one contexts

**The DP is exact for a fixed-context additive surrogate, not for the true recursive, state-dependent rollout cost.** Let `x_t` be the context fixed for time index `t` during label construction, `z_t` the observed carrier, and request `r=(Δ_r, α_r)`. The submitted implementation uses

\[
c(t,r)=\Delta_r\,\operatorname{mean}\!\left[(F_\theta(x_t,a,r)-z_{t+\Delta_r})^2\right]
+\lambda\,\frac{\operatorname{work}(r,\mathrm{policy})}{\operatorname{work}(15,\mathrm{continuous})},
\]

\[
V(H)=0,\qquad
V(t)=\min_{r:\,t+\Delta_r\le H}\{c(t,r)+V(t+\Delta_r)\}.
\]

Thus the first term is **duration-weighted endpoint carrier MSE**. It is not candidate-normalized ranking loss; the later J_dec experiment uses a different decision-oriented surrogate. With all `x_t` held fixed, the recurrence gives an exact minimum-cost path through the finite time-index graph. It does not account for the fact that selecting a different earlier request would generally produce a different future carrier and hence a different future error or cost. Consequently, it is not globally optimal control of the recursive world model.

The second-round context construction is also more limited than our wording suggested:

1. Initialize the context tensor from the observed trajectory, so every time position initially contains its observed carrier.
2. Roll out the first policy greedily from the initial state.
3. At **visited** positions, overwrite the stored context with the policy-generated carrier.
4. At **unvisited** positions, retain the observed carrier.
5. Evaluate every feasible request from each resulting stored context, form the fixed-context DP labels, and aggregate the observed-context and rollout-context training examples.

Every request is evaluated from the available context at a time index; every counterfactual successor context is **not** recursively expanded. In particular, a request that would lead to an unvisited time position uses that position's stored observed context in the continuation value, not a separately generated counterfactual state. This is a practical dataset-aggregation heuristic that exposes the policy to some self-generated states, not full branching search or a proof of recursive-rollout optimality. [Exact recurrence and context construction](https://github.com/Sino-Huang/NovPhy/blob/env/world_model/training/request_policy.py).

We propose stating this guarantee immediately with the recurrence, showing observed versus overwritten positions in the algorithm, and replacing any unqualified “optimal schedule” wording with **“optimal schedule for the fixed-context additive labeling surrogate.”** This addresses the mathematical claim directly rather than using the new task results to justify an invalid guarantee.

## Closing

The reviews prompted a necessary separation of four questions: **Can one model support multiple requests? Does shared training help? Does adaptive selection improve a decision? Does the resulting system outperform a simpler task predictor at lower cost?** The new evidence supports the first two within the studied regimes and a limited same-checkpoint selection contrast; it does not support the broadest versions of the latter two. We ask that the contribution be assessed on the corrected, narrower basis: a joint horizon–description interface and a controlled empirical study that makes these distinctions measurable, with negative controls and limitations reported alongside the positive results.

---

## Evidence and revision-status notes for the authors

These notes are separate from the reviewer-facing response above.

- **Status:** This draft reports completed experiments but proposes manuscript changes. It does not claim that the main manuscript, figures, bibliography, or tables have already been edited. The synthesis/carryover record includes unresolved manuscript work: [#124](https://github.com/Sino-Huang/NovPhy/issues/124), [#142](https://github.com/Sino-Huang/NovPhy/issues/142).
- **Unresolved requested evidence:** persistence/constant-velocity baselines; a physically group-disjoint Physion retraining/evaluation; a controlled sampling-rate-transfer test; exact equal-spend superiority over time-only schedules; and genuine changing-state closed-loop control. The rebuttal explicitly identifies these gaps rather than promising unperformed experiments.
- **Do not merge checkpoints or scopes:** original #96/#104 checkpoint results, improved Gate B fixed U, improved policy studies, the pixel benchmark extension, and novelty retuning answer different questions. U is F(15, macro), not J. The #106/#117 benchmark extensions are not all on the sealed NovPhy split.
- **Do not repair guard failures in prose:** the #104 precision failures, #102 replay failure, #114 endpoint clamping and duplicate execution, and original small-sample/tie failures retain their recorded interpretations. The original collection's missingness is not retrospectively made unbiased by the successful new capture.
- **External comparison boundary:** direct success prediction and readout controls are included because they directly test whether the rollout is useful for this task. The broader external-world-model program does not license a general superiority claim: its synthesis explicitly preserves fidelity/action-sensitivity and budget qualifications rather than converting invalid or collapsed comparators into wins. [#127](https://github.com/Sino-Huang/NovPhy/issues/127). This draft does not use unsupported detailed results from those individual arms.
- **Traceability:** Exact issue/comment links accompany substantive results. The original 36-cell AUC table is taken from the retained #96 numerical summary; the live task table is taken from the retained #102 result table. Code links explain mechanisms, not evidence of rate invariance or downstream success. Repository links use the source `env` branch where applicable; the linked issue records provide the experiment-specific commits and artifact locations.
