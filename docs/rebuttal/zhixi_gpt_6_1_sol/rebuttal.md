# Author response — ICLR 2027 submission #6352

We thank MMed, MSwm, and vSA9 for identifying the controls needed to distinguish a useful request interface from a demonstrated advantage of adapting that interface. We agree that the submission did not make this distinction sharply enough. Below, we separate **the submitted model and evidence**, **completed post-submission experiments**, and **specific manuscript revisions**. The new experiments do not retroactively strengthen the original experimental protocol.

Our revised claim is deliberately narrower: a shared carrier makes horizon and description jointly selectable within a rollout, and permits controlled study of when prediction granularity matters. We do **not** claim that joint adaptation generally improves task success, that macro descriptions consistently help, that the submitted policy saves two-thirds of computation, or that this method is superior to direct success prediction. The completed controls establish some scoped benefits, but also identify important negative results. We will place both in the main paper rather than leaving the qualifications in the appendix.

Notation below: **J** selects horizon and description; **J-cont** selects only the horizon on the *same hybrid predictor*; **C-J** selects horizons on the separately trained continuous-only predictor; **F-Δ-description** fixes both choices; **U** is the post-submission, fixed F-15-macro Gate-B unit, not the submitted adaptive policy. Unless otherwise specified, NovPhy intervals are descriptive 95% level-clustered percentile bootstrap intervals, with 10,000 resamples; three seeds are not treated as three independent sets of levels. Published decision tokens use their frozen practical margin, usually δ = 0.02, together with experiment-specific guards. A positive estimate or an interval excluding zero is not automatically a supported practical improvement.

## Shared responses to the main concerns

### 1. Evaluation isolation, sample size, and what the new data establish

**The scene-overlap criticism is correct.** The original CLEVRER experiment fitted and evaluated windows from the same 12 scenes: scene IDs **10000–10011**. Longer rollout contexts were not an unseen-scene test. Similarly, the original Physion-Dominoes experiment used 12 trials from a bounded prefix of the dynamics-training archive; it was a within-trial rollout diagnostic, not a documented independent-trial generalization test. We will relabel those original tables accordingly and replace the main generalization discussion with the following completed post-submission evaluation. [Sources: original manifests in `.local-artifacts/issue-79-clevrer-boundary-v1/plan.json` and `.local-artifacts/issue-84-third-family-v1/plan.json`; [new evaluation #106](https://github.com/Sino-Huang/NovPhy/issues/106).]

- **CLEVRER:** 6,285 fitting videos selected from official training IDs 0–9999, and 3,132 evaluation videos selected from official validation IDs 10000–14999. The ID ranges are disjoint, and the manifest enumerates every included and excluded ID. The predeclared colour/material-key and carrier-contract exclusions remove 37.22% of the release videos; this is not an unrestricted CLEVRER benchmark claim. Encoder, predictor, and policy use only the training membership.
- **Physion-Dominoes:** 2,000 dynamics-training trials, 996 readout-fitting trials, and 150 testing trials, with separate roles and a content-hashed membership manifest. A corrected content audit finds no full static-parameter-set overlap across splits. However, **20/150 test trials share all physical parameters except colour with a fitting trial, and four have near-identical trajectories**. Thus, this official-split evaluation measures transfer to new colours, not generalization to independent physical realizations. We will state this prominently rather than using “held-out” to imply stronger independence.
- **NovPhy:** the new #109/#104 campaign separates predictor fitting (**480 levels**), policy fitting (**120 levels**), and final evaluation (**400 levels**), rather than treating the small original development inventories as generalization evidence. All 11,680 scheduled evaluation branches completed, with no capture failures. Ranking uses mixed-success cells: **311 grid, 76 offset, 78 angle, and 47 power levels**, each scored for three seeds. The non-grid sets use 120 of the 400 levels; inventories are reported separately, not pooled. Power missed its design target of 53 mixed levels, and the faithful submitted-checkpoint rerun still fails its precision guard on every inventory. More data alone did not establish the submitted adaptation claim. [#109](https://github.com/Sino-Huang/NovPhy/issues/109), [#104](https://github.com/Sino-Huang/NovPhy/issues/104).

Even the isolated NovPhy grid has a further limitation: later nearest-fit and novelty controls show that it rewards interpolation among similar generated levels. We therefore distinguish **unseen level identities within two familiar families** from physical novelty and broad out-of-distribution generalization.

### 2. Does selecting a description add value beyond continuous prediction?

**There is a modest, qualified same-checkpoint result, not a general symbolic-description advantage.** The missing control requested by MSwm and vSA9 has now been completed: J-cont uses the same hybrid trunk, carrier, nine-output controller architecture, training pool, seeds, and update budget as J, but its labels and choices are restricted to the three continuous requests. On 311 sealed grid levels, at the transferred λ = 10⁻⁴ operating recipe:

| Controller | AUC [95% interval] | Top-3 hit rate [95% interval] | Linear MACs per candidate |
|---|---:|---:|---:|
| J | 0.7057 [0.6846, 0.7267] | 0.5102 [0.4620, 0.5606] | 184.0M |
| J-cont, same hybrid predictor | 0.6856 [0.6647, 0.7061] | 0.4662 [0.4223, 0.5123] | 122.7M |
| C-J, continuous-only predictor | 0.6999 [0.6827, 0.7170] | 0.4469 [0.4148, 0.4802] | 234.7M |
| J's recorded horizons replayed continuously, without policy calls | 0.7021 [0.6805, 0.7231] | 0.4920 [0.4480, 0.5370] | 173.5M |
| No-model action prior | 0.7203 [0.6965, 0.7442] | 0.4180 [0.3633, 0.4727] | 0 model MACs |

The frozen at-most-budget comparison gives **J − J-cont = +0.0201 [0.0154, 0.0248]**, with the published `supported` token. Two qualifications are essential: J-cont cannot reach either declared budget in the frozen sweep and is clamped at its largest-MAC endpoint; the two budget readings reuse the same endpoint pair and are not independent replications. This is **not exact equal expenditure, a best-quality-under-budget envelope, or Pareto dominance**. Duplicate scoring workers also violated literal scored-once execution for some fixed-arm prefixes; the ticket preserves the incident and discarded-compute accounting. No typed model failure was retried. We will disclose these limitations with the result. [#114](https://github.com/Sino-Huang/NovPhy/issues/114).

The mechanistic controls are less favourable: J minus continuous replay of its own horizons is **+0.0037 [−0.0021, 0.0095]**, and J minus a continuous-masked selector on the trained J logits is **+0.0032 [−0.0075, 0.0137]**. Neither establishes a ≥0.02 benefit attributable specifically to symbolic feedback. The supported J/J-cont result is a comparison of trained scheduling systems; it does not isolate symbolic semantics.

The broader results require further restraint:

- **Shared prediction is useful in one scoped contrast:** on the sealed NovPhy split, the shared model exceeds per-request specialists by **+0.0848 [0.0742, 0.0952]** averaged over requests; F-15-macro improves by +0.0561. This supports sharing under that training design, not joint adaptation or universally absent negative transfer. The annotation-carrier benchmark comparisons remain qualified, including an unresolved Physion LC Δ=15 error ratio of **1.58 [0.79, 3.08]**. [#117](https://github.com/Sino-Huang/NovPhy/issues/117).
- **Macro readability remains inadequate:** the new temporal macro head achieves held-out F1 **0.157 [0.151, 0.164]** on parsed carriers, **0.113 [0.107, 0.119]** on one-step predictions, and only **0.015 [0.014, 0.015]** on recursive rollouts. The original Gate-B macro head reads none of the redefined events. The temporal improvement does not establish a reliable macro abstraction during planning. [#113](https://github.com/Sino-Huang/NovPhy/issues/113).
- **Semantic fidelity is not established as the source of ranking gains:** the label-free micro bottleneck is non-inferior for the tested J−T contrast, **−0.003 [−0.015, 0.008]**, but the reference benefit being preserved is itself only +0.003. This is not preservation of an established description benefit. Removing feedback performs about as well as default feedback; oracle feedback can hurt. The macro robustness comparison remains unresolved. [#115](https://github.com/Sino-Huang/NovPhy/issues/115).
- **Changing the task cost does not rescue a general macro claim:** the Gate-B macro-minus-continuous differences are −0.012, −0.024, −0.036, and −0.078 across the four tested costs. A redesigned coordination arm gives a descriptive +0.022 under the first cost, but that advantage disappears or reverses under the others. [#120](https://github.com/Sino-Huang/NovPhy/issues/120).

Accordingly, we will describe continuous/micro/macro as **available request modes**, not a proven hierarchy of increasingly useful abstractions.

### 3. Ranking, actual success, and strong external controls

**The original near-chance rankings do not demonstrate meaningful task-success improvement.** In the original inventories, J's Top-1 result is zero successful choices across all four sets; the small positive AUC differences should not be presented as successful control. The complete original uncertainty table is included below.

A new dynamics recipe with long-horizon and decision-contrastive supervision passes the predeclared Gate-B threshold: U reaches **0.7444 [0.7269, 0.7620]** AUC on 311 sealed grid levels. However, the matched continuous fixed request C* scores **0.7834 [0.7676, 0.7992]**, and U−C* is **−0.0390 [−0.0442, −0.0338]**. The no-model prior scores 0.7203; U−prior is **+0.0241 [−0.0008, 0.0487]**, unresolved. A wrong-anchor control reduces U by **0.1322 [0.1216, 0.1430]**, showing context dependence, not superiority to C* or the prior. Gate B certifies usable ranking under a new recipe; it is not a joint-description win. [Gate-B evidence in #108](https://github.com/Sino-Huang/NovPhy/issues/108).

We also completed **live engine retry-from-start episodes** with at most three shots on all 400 evaluation levels. These are not multi-action, state-evolving closed-loop control: every failed attempt resets the same single-bird level, and replanning removes failed candidates. The table in the response supplement reports all arms, including failures. Illustrative pass@3 values are F-15-continuous **0.478 [0.434, 0.523]**, C* **0.466 [0.427, 0.505]**, decision-trained J_dec **0.445 [0.403, 0.487]**, U **0.424 [0.384, 0.465]**, original-objective J **0.394 [0.352, 0.438]**, C-J **0.346 [0.315, 0.378]**, prior **0.323 [0.278, 0.370]**, and random **0.198 [0.172, 0.223]**. Here success means engine pig removal, not necessarily native level clear.

**Every inferential contrast in that experiment is vetoed by the frozen decision-frame guard:** 25/5,106 frames differ from their anchors by up to 0.0032, above the 0.001 tolerance. A post-hoc re-ranking check preserved the first three choices in 351/351 comparisons, but does not repair that veto. The pass rates are completed, descriptive observations, not supported superiority claims. On the 40 appearance-novel levels, every model−prior interval includes zero. [#102](https://github.com/Sino-Huang/NovPhy/issues/102).

Finally, the new external baselines materially constrain the paper's practical claim. Direct action-conditioned success classifiers score **0.9946 [0.9917, 0.9971]** from pixels and **0.9757 [0.9692, 0.9815]** from the matched carrier; their differences from U are **+0.2502 [0.2327, 0.2675]** and **+0.2313 [0.2131, 0.2493]**, respectively, with supported scoped tokens. The dynamics-aware classifier scores 0.9945, but its comparison is budget-qualified: **1.88× U's training GPU-hours**. An increased-budget U control does not remove that qualification because it fails an action-sensitivity guard in one seed; its numerical AUC 0.8132 is descriptive. [#128](https://github.com/Sino-Huang/NovPhy/issues/128), [#140](https://github.com/Sino-Huang/NovPhy/issues/140).

Borrowing the discriminative readout improves U to **0.9765 [0.9689, 0.9828]**, but its independently trained **no-rollout control reaches 0.9809 [0.9748, 0.9863]**. The difference is −0.0044 [−0.0081, −0.0014], below the practical margin in both directions: **U's predicted endpoints have not shown incremental ranking value under that scorer**. The improvement over U is budget-qualified (1.03× its training GPU-hours). These are ranking results, not 98% live success rates. [#141](https://github.com/Sino-Huang/NovPhy/issues/141).

The external-baseline program therefore licenses neither “better than all baselines” nor a task-level advantage of world modelling over direct prediction. Its world-model arms that fail fidelity, readiness, or action-sensitivity checks cannot be counted as wins for us. We will include the full disposition table and interpolation/novelty qualifications, rather than selectively comparing with failed ports. [#127 synthesis](https://github.com/Sino-Huang/NovPhy/issues/127#issuecomment-6505377888).

## Response to Reviewer MMed

### MMed: overlapping scenes and too few NovPhy levels

Your concern changes the interpretation of the original results: they are rollout diagnostics, not generalization evidence. Shared response 1 gives the completed disjoint-ID CLEVRER evaluation and the enlarged NovPhy evaluation, including the remaining Physion physical-overlap and NovPhy interpolation limitations. We will publish the exact scene/trial membership beside the revised table and explicitly distinguish identity-disjoint evaluation from physical-novelty evaluation.

### MMed: benefit of symbolic levels and weak task results

The answer is not a consistent symbolic advantage. Shared responses 2–3 give the same-checkpoint continuous-only comparison, the negative macro/readout controls, the continuous fixed-request advantage, and the descriptive live-success measurements. We will retain the modest qualified J/J-cont result but remove claims of general macro benefit and successful adaptive control. The contribution that survives is a controllable shared prediction interface and a controlled diagnosis of granularity—not evidence that symbolic abstraction is always useful.

### MMed: more complex and closed-loop environments

Completed additions include pixel-carrier CLEVRER QA and Physion OCP, the live retry-from-start experiment, and physical/appearance-novel NovPhy evaluation. They do **not** demonstrate a broader closed-loop capability. CLEVRER's frozen collision readout never reaches its threshold (maximum predicted collision probability 0.051 versus threshold 0.5); option accuracy is at its 0.525 floor, so its near-zero J−J-cont result is not evidence that the two predictors are equivalent. Physion OCP gives J−J-cont **−0.021 [−0.068, 0.025]**, unresolved. Policy-only novelty retuning gives J_dec **−0.007 [−0.021, 0.006]** versus zero-shot, while predictor retuning improves by +0.151; cheap request-only adaptation is not supported. [#106](https://github.com/Sino-Huang/NovPhy/issues/106), [#107](https://github.com/Sino-Huang/NovPhy/issues/107).

We will explicitly scope the method to fixed-action prediction between decision points. State-evolving, multi-action replanning remains an unresolved limitation, not a completed experiment.

### MMed: overcrowded figures and underdeveloped main text

We will use the available main-text space to separate (i) the request interface and feedback path, (ii) the actual DP surrogate and data protocol, and (iii) task/compute evidence. Figure 1 will contain one readable interface diagram and one uncertainty plot; Figure 3 will be split into rollout accuracy and compute–quality panels with larger labels. The original dense ranking heatmap will move to a numerical appendix; the main text will show the fixed continuous, same-checkpoint J-cont, J, and no-model controls with intervals. Captions will identify the checkpoint, split, time unit, error reference, and whether a result is descriptive or guard-vetoed. These are proposed manuscript edits, not claims that the submitted figures already contain the new evidence.

## Response to Reviewer MSwm

### MSwm weakness 1 and Q2: small gains, counts, successful launches, and every interval

We agree that a gain relative to a weak cross-fit reference is insufficient. The original J−r† differences are:

| Inventory | Mixed scored levels × seeds | J−r† [95% interval] | Successful launch slots in original coverage audit | Reached by endpoint 225 |
|---|---:|---:|---:|---:|
| Grid | 14 × 3 | +0.0700 [−0.0057, 0.1437] | 17 | 11 |
| Offset | 9 × 3 | +0.0037 [−0.0796, 0.0926] | 9 | 3 |
| Angle | 7 × 3 | +0.0595 [0.0079, 0.1131] | 7 | 5 |
| Power | 8 × 3 | +0.0273 [−0.0691, 0.1281] | 8 | 4 |

Successful slots here are the source audit's available successful candidates, not successful selections by J, and are not multiplied by seed. J selects zero successful Top-1 candidates: **0/42, 0/27, 0/21, and 0/24**; the corresponding random-selection expectations are 0.0804, 0.1030, 0.0778, and 0.0521. The complete nine-request AUC uncertainty table appears below. The positive angle interval is a descriptive observation, not broad replication. The original grid has 17/42 fully tied cells and an inadequate primary half-width; its adaptation disposition is insufficient. [#96](https://github.com/Sino-Huang/NovPhy/issues/96).

On new evaluation levels, rerunning the *submitted checkpoints* with a tie-free cost gives J AUC **0.555 [0.539, 0.571]** and J−r† **+0.0134 [−0.0163, 0.0431]**. The primary half-width is 0.0297, above the 0.02 guard; every inventory fails that precision guard. The offset estimate is +0.011 [−0.052, 0.074], not confirmation of +0.004. The other contrasts are angle −0.160 [−0.206, −0.113] and power −0.262 [−0.336, −0.192], also guard-qualified. We will report these rather than imply that the enlarged cohort replicated the original gain. [#104](https://github.com/Sino-Huang/NovPhy/issues/104).

### MSwm weakness 2: prior work and scope of novelty

You are right that adaptive horizon selection is not new. We will add and discuss [Neitz et al., *Adaptive Skip Intervals*, NeurIPS 2018](https://neurips.cc/virtual/2018/poster/11932) and [Chatterjee and Khardon, *Improving planning and MBRL with temporally-extended actions*, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/cec445dfc292392af716e9a4fe8de99b-Abstract-Conference.html). ASI learns prediction skip intervals; Chatterjee and Khardon integrate adaptive temporally extended actions into planning/MBRL. We will not present choosing a horizon as our novelty. Our specific distinction is allowing the same carrier/predictor to accept a **joint horizon–description request**, including symbolic feedback, without conversion among independently learned latent spaces. The same-checkpoint control and its qualifications are in shared response 2.

We also completed a comparison with H-JEPA, including source-benchmark fidelity checks and its inverse-dynamics/action-regularization remedy. Its NovPhy ports fail the frozen collapse/action/readout criteria; both the original and remedy comparisons license **no scoped ranking superiority claim in either direction**. This is an unresolved transfer limitation of those comparisons, not evidence that our method surpasses hierarchical world models. [#123](https://github.com/Sino-Huang/NovPhy/issues/123), [#125](https://github.com/Sino-Huang/NovPhy/issues/125).

### MSwm weakness 3 and Q1: what Δ counts, and the elapsed-time channel

The concern is specifically about the **NovPhy training mixture**, not merely different datasets having different frame rates. Δ counts carrier-frame indices; it does not intrinsically mean a fixed number of native physics steps. In the original base corpus, much of the supervision is at native-step cadence, whereas N1 observation frames are separated by **50 native steps**. Accordingly, N1 requests Δ = 1, 5, 15 correspond to **50, 250, and 750 native steps**. The same request code therefore does not guarantee a common physical horizon across those sources. We acknowledge that the original presentation did not characterize this transfer adequately.

There are three distinct signals: **(1)** Δ specifies the requested jump in that stream's frames; **(2)** the carrier contains `has_prior` and **elapsed seconds between current and prior observations**, and computes motion as centre displacement divided by that elapsed time; **(3)** the policy's remaining-frames input κ specifies distance to the rollout/window endpoint. The elapsed entry is not the requested future horizon, and κ is not a temporal-rate code. The presence of elapsed seconds makes observation cadence available; it does **not** prove that the trained model uses it correctly or resolves the 50× supervision/evaluation mismatch.

The later N1-aligned training uses 60-observed-frame windows at 50-native-step spacing and matched training/evaluation conventions. We will give an explicit table mapping every corpus to native steps, observed frames, elapsed seconds, and request duration, and distinguish these later checkpoints from the original mixed-rate ones. We do **not** claim that the post-submission performance is an ablation proving rate invariance or dependence on the elapsed-time entry. That mechanistic question remains unresolved. [Implementation: `world_model/data/deployment_temporal.py:680–719`; recipe: `docs/issue-71-hybrid-readiness.md:48–80`; native-step disclosure: `docs/issue-76-native-findings.md`.]

### MSwm Q3: held-out Table 1, simple baselines, and policy schedules

The completed new pixel-carrier table covers all nine requests and the policies' actual schedules. Representative final-endpoint position MSE values **against parsed carriers** are:

| Request | CLEVRER, t=120 | Physion-Dominoes, t=90 |
|---|---:|---:|
| F-1-continuous | 0.1625 [0.1516, 0.1755] | 0.04295 [0.03274, 0.05347] |
| F-5-continuous | 0.1089 [0.1037, 0.1146] | 0.02266 [0.01842, 0.02724] |
| F-15-continuous | 0.09960 [0.09577, 0.1038] | 0.01426 [0.01076, 0.01821] |
| F-15-micro | 0.09929 [0.09558, 0.1033] | 0.01454 [0.01090, 0.01867] |
| F-15-macro | 0.09739 [0.09439, 0.1006] | 0.01456 [0.01103, 0.01865] |
| J, own schedules | 0.1569 [0.1472, 0.1663] | 0.02348 [0.01884, 0.02849] |
| J-cont, own schedules | 0.1616 [0.1516, 0.1729] | 0.02519 [0.02062, 0.03028] |

Intervals here are descriptive seed×clip intervals. The annotation-truth error is separately retained in the result artefact; parser-relative error is not interchangeable with simulator-truth error. Long fixed jumps have lower rollout error in this experiment, whereas the learned policy is not the best rollout predictor. Physion's independence qualification in shared response 1 still applies. [#106](https://github.com/Sino-Huang/NovPhy/issues/106).

**The requested persistence/constant-velocity comparison on this exact new pixel protocol is not established by the cited completed results.** We will not label this part of Q3 as completed or use the table to claim superiority to those simple predictors. The revised table must explicitly mark the absent comparison as a limitation; the positive claim is only the measured comparison among the evaluated requests and schedules.

### MSwm Q4: source of compute savings

The old “one-third compute at matching quality” statement is not supported under engine-truth evaluation and will be removed. At the transferred operating point, J uses **184.0M** versus C-J's **234.7M** linear MACs per candidate: ratio **0.784**, not one-third. Their AUC difference is +0.006 [0.000, 0.012], below δ. C-F-15 has higher AUC, 0.783, at only 27.8M MACs, and the zero-model prior also dominates both operating points on the AUC/MAC axes. [#103](https://github.com/Sino-Huang/NovPhy/issues/103).

The same-checkpoint decomposition is informative: relative to C-J, J saves 50.713M MACs overall, comprising **−50.836M from the trunk**, **−6.353M from calls/horizons**, **+6.583M from description computation**, and **−0.106M from controller work**. Description modes add cost in that decomposition; the cheaper trunk accounts for most of the saving. Relative to J-cont on the same trunk, J actually costs **61.241M more**. Thus the old cross-stack comparison cannot isolate representation-selection savings. [#114](https://github.com/Sino-Huang/NovPhy/issues/114).

We will show the complete quality–MAC frontier rather than a single favourable point. Wall time is separate: the frozen single-seed timing gives a J/C-J ratio of **16.74**; the later three-seed retiming gives **1.63**, but is post-hoc and descriptive. Neither supports a runtime saving, and the latter cannot replace the frozen result. [#103](https://github.com/Sino-Huang/NovPhy/issues/103).

### MSwm minor points

1. We will correct TAWM to the **ICML 2025 main conference**, not a workshop. [Official proceedings](https://proceedings.mlr.press/v267/nhu25a.html).
2. We will define terminology on first use: a **lineage** is the provenance/grouping unit whose related shot records remain together in splitting and uncertainty estimation; **base corpus** is the original training-role shard collection, not the later N1 captures; **N1 captures** are newly rendered normal-condition NovPhy shot trajectories at the stated 50-native-step observation cadence; **development states** are exposed states used for diagnostics/design rather than final evaluation. Each occurrence will identify its actual fitting, controller-training, or evaluation role. We will replace provenance shorthand in the main text with the corresponding dataset role wherever possible.

## Response to Reviewer vSA9

### vSA9 Q1: scene IDs and recalculation

Yes: the original fitting/evaluation CLEVRER scenes were the same IDs 10000–10011. We will remove the implied scene-generalization interpretation. The completed replacement uses selected training IDs within **0–9999** and selected validation IDs within **10000–14999**, with **zero scene-ID intersection**. Exact membership is enumerated in `.local-artifacts/issue-106-pixel-benchmarks-v1/plan.json`, field `membership.clevrer`, with split roles under `membership.roles`; we will carry that enumeration into the evaluation supplement rather than describe all IDs in the ranges as included. Shared response 1 gives counts and exclusions; MSwm Q3 gives recalculated errors and the outstanding simple-baseline gap. We cannot honestly claim independent physical generalization from the Physion official split.

### vSA9 Q2: temporal-rate encoding

The metric is **elapsed seconds between the preceding and current observations**, stored in carrier column 1 with a has-prior indicator in column 0; per-object motion is displacement divided by elapsed seconds. Request Δ and remaining-time κ are separate variables. The detailed answer to MSwm Q1 acknowledges the mixed-rate NovPhy supervision issue: exposing the rate in a carrier is not proof that transfer across rates has been learned. We will not conflate this with the new aligned training recipe or attribute its gains to an unperformed elapsed-channel ablation.

### vSA9 Q3: same-checkpoint restricted scheduling and decision budget

The new experiment includes **T-α** (fix the description and adapt the horizon, for each α), **D-Δ** (fix the horizon and adapt the description, for each Δ), the retrained **J-cont**, all nine fixed requests, the continuous-only model C-J, and continuous replay of J's own horizon sequence. All use the same endpoint/candidate inventory; the same-trunk restrictions hold the predictor fixed. Shared response 2 reports the key intervals and explains why the frozen budget contrast is only at-most-budget, not exact matching. At grid λ=10⁻⁴, the continuous-masked T-cont selector scores 0.7025 versus J's 0.7057, difference +0.0032 [−0.0075, 0.0137]; no ≥0.02 description-selection improvement is established. [#114](https://github.com/Sino-Huang/NovPhy/issues/114).

The faithful submitted-stack rerun also reports J−T **−0.001**, J−D **+0.058**, J−factorized-selector **+0.008 [−0.004, 0.020]**, and J−open-loop **+0.040 [0.025, 0.056]** on the new grid. **All tokens remain insufficient because the primary precision guard fails.** We will retain those numerical observations without converting the favourable open-loop contrast into a supported jointness/state-dependence claim. [#104](https://github.com/Sino-Huang/NovPhy/issues/104).

### vSA9 Q4: task quality, failed cases, and the +0.004 offset result

The answer to the offset question is **no established replication**: +0.0037 [−0.0796, 0.0926] originally, and +0.011 [−0.052, 0.074] in the faithful new-level rerun, with its precision guard failed. J is also below the hindsight-best original offset fixed request by **−0.0870 [−0.1500, −0.0259]**. We will not summarize this as a retained adaptive benefit.

The response supplement below reports original Top-1 quality and complete live pass@1/pass@3 results for every evaluated arm, with ordinary unsuccessful episodes, typed failures, and no-candidate episodes separated. It does not condition on eventual success or omit bad requests. The experiment's complete episode/execution manifests retain individual successful and failed attempts; the manuscript supplement will link them with the outcome convention, not show only selected successful trajectories. The guard failure and reset semantics remain attached to the success table. Ranking AUC, Top-3 inclusion, pig-removal pass rate, and native level clear will be labelled separately.

### vSA9 Q5: exact DP objective and second-round successor contexts

Your characterization of the optimality limit is correct. For a fixed tensor of contexts \(\tilde z_t\), observed targets \(z_t\), action \(a\), request \(r=(\Delta_r,\alpha_r)\), and window endpoint \(T\), the **submitted prediction-error teacher** uses

\[
c(t,r)=\Delta_r\,\operatorname{mean}\!\left[(F(\tilde z_t,a,r)-z_{t+\Delta_r})^2\right]
+\lambda\,\frac{\operatorname{work}(r,\pi)}{\operatorname{work}(r_{\rm ref})},
\qquad
V(t)=\min_{r:\,t+\Delta_r\le T}\{c(t,r)+V(t+\Delta_r)\}.
\]

The terminal value is zero, requests past the valid window are infeasible, and exact ties use the frozen request order. Backward DP is exact for this finite **additive fixed-context surrogate**. It does not optimize the cost of every recursively imagined branch: choosing a request does not cause the continuation table to be recomputed from that request's own predicted successor. Nor does exact teacher optimization imply that the distilled policy is optimal, or that error/MAC minimization optimizes task success.

Round 1 uses observed contexts. To form the next aggregation round, a greedy rollout of the current policy starts at the observed anchor; **only visited landing positions are overwritten by predicted carriers**. **Unvisited positions retain their observed carriers.** DP then evaluates every admissible request from this now-fixed mixture of observed and visited-predicted contexts, against unchanged aligned observed targets. An unvisited alternative successor is not generated by recursively expanding every alternative branch. At deployment the observed future targets are unavailable and are not inputs to the predictor or controller.

We will replace the ambiguous “DP optimal” language with precisely this guarantee and include pseudocode exposing the retained observed contexts. The later **J_dec** ranking-trained policy is a post-submission variant with a different label objective, not the submitted duration-weighted prediction-error DP. [Implementation: `world_model/training/request_policy.py:112–155`; decision-trained variant [#101](https://github.com/Sino-Huang/NovPhy/issues/101).]

## Response supplement: complete numerical results underlying the small-inventory and task-success answers

### A. Original request-ranking cells, with uncertainty

These are the original #96 checkpoint/inventory results, not the new sealed-grid scores. Each entry is AUC [95% descriptive level-clustered interval]. The four columns have 14, 9, 7, and 8 mixed levels respectively; each has three model seeds. All nine fixed requests are included, followed by adaptive and control arms. `T` fixes J's modal description while adapting horizons, `D` fixes its modal horizon while adapting descriptions, `M` factorizes its selection logits, and `OL` uses a state-independent schedule distribution. They are not interchangeable with the newly retrained J-cont control. [#96](https://github.com/Sino-Huang/NovPhy/issues/96).

| Arm | Grid | Offset | Angle | Power |
|---|---|---|---|---|
| F-1-continuous | 0.5104 [0.4827, 0.5355] | 0.5167 [0.4815, 0.5556] | 0.5198 [0.5000, 0.5437] | 0.5092 [0.4883, 0.5362] |
| F-1-micro | 0.4932 [0.4639, 0.5198] | 0.5000 [0.4889, 0.5111] | 0.5020 [0.4464, 0.5595] | 0.5198 [0.5044, 0.5417] |
| F-1-macro | 0.5126 [0.4675, 0.5626] | 0.5037 [0.4444, 0.5593] | 0.5060 [0.4226, 0.5833] | 0.5274 [0.4737, 0.5921] |
| F-5-continuous | 0.4145 [0.3367, 0.4796] | 0.5741 [0.5074, 0.6481] | 0.5808 [0.4975, 0.6627] | 0.4582 [0.3737, 0.5493] |
| F-5-micro | 0.4253 [0.3683, 0.4840] | 0.5759 [0.4907, 0.6593] | 0.6093 [0.5538, 0.6706] | 0.4688 [0.3794, 0.5461] |
| F-5-macro | 0.4006 [0.3475, 0.4612] | 0.5944 [0.5167, 0.6741] | 0.6346 [0.5873, 0.6786] | 0.5014 [0.4135, 0.6056] |
| F-15-continuous | 0.4918 [0.4108, 0.5836] | 0.4407 [0.3481, 0.5037] | 0.4502 [0.3232, 0.5614] | 0.4618 [0.4026, 0.5276] |
| F-15-micro | 0.4791 [0.3951, 0.5783] | 0.4870 [0.4148, 0.5444] | 0.4502 [0.3589, 0.5238] | 0.4516 [0.3991, 0.5196] |
| F-15-macro | 0.4912 [0.4545, 0.5261] | 0.5222 [0.4889, 0.5593] | 0.5335 [0.5000, 0.5693] | 0.4994 [0.4380, 0.5592] |
| J | 0.4706 [0.4158, 0.5256] | 0.5074 [0.4556, 0.5593] | 0.5655 [0.5198, 0.6250] | 0.5287 [0.4693, 0.5906] |
| T | 0.4984 [0.4627, 0.5296] | 0.5407 [0.5074, 0.5815] | 0.5534 [0.5253, 0.5794] | 0.5232 [0.4749, 0.5724] |
| D | 0.5093 [0.4590, 0.5676] | 0.5296 [0.4519, 0.6074] | 0.4859 [0.3889, 0.5693] | 0.5087 [0.4659, 0.5559] |
| M | 0.4921 [0.4348, 0.5459] | 0.5241 [0.4796, 0.5741] | 0.5712 [0.5213, 0.6310] | 0.5497 [0.5070, 0.5910] |
| OL | 0.5226 [0.4738, 0.5740] | 0.5093 [0.4741, 0.5444] | 0.5435 [0.5076, 0.5794] | 0.5371 [0.4834, 0.5877] |
| C-F-1 | 0.4678 [0.4473, 0.4866] | 0.5204 [0.4889, 0.5519] | 0.4771 [0.4334, 0.5258] | 0.5088 [0.4737, 0.5504] |
| C-F-5 | 0.4856 [0.4323, 0.5325] | 0.6241 [0.5278, 0.7093] | 0.5765 [0.5079, 0.6501] | 0.6051 [0.4912, 0.7279] |
| C-F-15 | 0.5057 [0.4304, 0.5864] | 0.5519 [0.4593, 0.6407] | 0.4747 [0.4048, 0.5437] | 0.5766 [0.4955, 0.6775] |
| C-J | 0.5220 [0.4741, 0.5663] | 0.5463 [0.4963, 0.5926] | 0.5460 [0.4881, 0.6115] | 0.5580 [0.4967, 0.6060] |

The four J−r† intervals in MSwm Q2 are paired differences, not differences of marginal interval endpoints. The small original evaluation and the count-cost ties make them descriptive; the original primary guard failures must accompany the table. In particular, the cross-inventory-selected r† should not be described as an independently evaluated oracle-best fixed request.

### B. All live task-success arms, including unsuccessful outcomes

All rows below are **N1, 400 levels × 3 seed slots = 1,200 episodes per arm**, under retry-from-start and a maximum of three shots. The prior and oracle are seed-free repeated references. Intervals are descriptive 95% level-clustered intervals. `Failed` means completed without pig-removal success within the budget; `Typed` means an execution failure counted as unsuccessful; `None` means no candidate was executed (notably the oracle when no recorded successful candidate exists). `Passed + Failed + Typed + None` accounts for all episodes; there are no incomplete episodes. These counts must not be interpreted as 1,200 independent physical levels. Every comparison token is insufficient because the decision-frame guard failed. [#102](https://github.com/Sino-Huang/NovPhy/issues/102).

| Arm | Pass@1 [CI] | Pass@3 [CI] | Passed | Failed | Typed | None |
|---|---|---|---:|---:|---:|---:|
| C-F-15 | 0.148 [0.126, 0.172] | 0.466 [0.427, 0.505] | 559 | 633 | 8 | 0 |
| CJ-1e-04 | 0.161 [0.136, 0.187] | 0.346 [0.315, 0.378] | 415 | 777 | 8 | 0 |
| F-1-continuous | 0.070 [0.045, 0.095] | 0.334 [0.295, 0.375] | 401 | 791 | 8 | 0 |
| F-1-macro | 0.112 [0.089, 0.135] | 0.343 [0.297, 0.390] | 411 | 780 | 9 | 0 |
| F-1-micro | 0.070 [0.045, 0.095] | 0.326 [0.287, 0.366] | 391 | 802 | 7 | 0 |
| F-15-continuous | 0.189 [0.161, 0.218] | 0.478 [0.434, 0.523] | 574 | 618 | 8 | 0 |
| F-15-macro | 0.151 [0.131, 0.172] | 0.424 [0.384, 0.465] | 509 | 683 | 8 | 0 |
| F-15-micro | 0.221 [0.192, 0.251] | 0.458 [0.416, 0.502] | 550 | 642 | 8 | 0 |
| F-5-continuous | 0.180 [0.151, 0.210] | 0.453 [0.408, 0.499] | 544 | 647 | 9 | 0 |
| F-5-macro | 0.107 [0.083, 0.132] | 0.413 [0.371, 0.457] | 496 | 695 | 9 | 0 |
| F-5-micro | 0.167 [0.138, 0.197] | 0.438 [0.394, 0.481] | 525 | 667 | 8 | 0 |
| J-1e-04 | 0.123 [0.098, 0.150] | 0.394 [0.352, 0.438] | 473 | 718 | 9 | 0 |
| J-dec-1e-01 | 0.139 [0.117, 0.162] | 0.445 [0.403, 0.487] | 534 | 658 | 8 | 0 |
| oracle | 0.775 [0.733, 0.815] | 0.775 [0.733, 0.815] | 310 | 0 | 1 | 89 |
| prior | 0.140 [0.107, 0.175] | 0.323 [0.278, 0.370] | 129 | 268 | 3 | 0 |
| random | 0.068 [0.053, 0.082] | 0.198 [0.172, 0.223] | 237 | 951 | 12 | 0 |

**Scope of the manuscript revision.** We will replace the original generalization and one-third-compute claims; expose the mixed-rate NovPhy issue and fixed-context DP guarantee; add the same-checkpoint restrictions and full error/decision/compute tables; and label every new experiment as post-submission. The work supports a shared request-conditioned interface and a bounded empirical investigation of granularity. It does not establish universal benefit from joint adaptation, useful macro semantics in recursive prediction, physical-novelty generalization, or superiority to direct classifiers.

---

## Internal evidence notes — not part of the author-facing response

- This draft deliberately distinguishes the original #77/#96 stack, its faithful #104 rerun, the new Gate-B #112 recipe, the #101 decision-trained variant, and the separate #106 pixel benchmark protocol. They cannot be merged into a single “our model” result.
- Material remaining gaps: persistence/constant-velocity baselines on the exact new #106 protocol are not established by the cited evidence; Physion retains physical near-duplicates; the elapsed-channel mechanism is not isolated; the at-most-budget #114 result is endpoint-clamped and has a scoring incident; #102 is reset/retry rather than state-evolving closed-loop control and fails its frozen frame guard; the faithful #104 rerun remains precision-limited.
- The #103 frozen single-seed wall result is 16.74×; the three-seed 1.63× retiming is post-hoc descriptive. Reporting only 1.63× as confirmatory would be inaccurate.
- H-JEPA #123/#125 and #127 external world-model failures do not license a superiority claim. #128 direct-classifier results and #141's no-rollout control materially undermine any task-level claim that a world model adds ranking value. They must remain in the rebuttal even though unfavourable.
- Exact CLEVRER and Physion memberships: `.local-artifacts/issue-106-pixel-benchmarks-v1/plan.json`, `membership`; content/physical-overlap correction: `leakage-audit.json`. The original within-scene IDs are 10000–10011. The new ID ranges are disjoint but have exclusions; do not state that all IDs in either range were used.
- Complete live outcome counts above are transcribed programmatically from `.local-artifacts/issue-102-closed-loop-v1/compute.json`, `tables.N1.arms`; original ranking intervals from `.local-artifacts/issue-96-tau-ad-within-checkpoint-v1/compute.json`, `inventories.*.arm_auc`. These are reporting operations, not fresh evaluation runs.
- Proposed figure/table/wording changes in this draft are manuscript recommendations, not claims of already completed manuscript edits. No build, test, formatter, or model evaluation was run for this drafting assignment.
