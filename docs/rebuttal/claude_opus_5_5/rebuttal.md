# Author Response — ICLR 2027 Submission #6352
**"Granularity Is a Decision: Joint Horizon–Description Selection in Latent World Models"**

> Posting note (internal, delete before posting): the evidence tags `[E#]` below map to the traceability table at the end of this file. Those links point to a named GitHub repository and would de-anonymise the submission. Do not paste them into OpenReview. Replace them with an anonymised mirror or move the numbers into the revised appendix.

---

## Shared response to all reviewers

We thank the three reviewers for careful and specific reviews. Several of the concerns are correct, and some go further than the reviewers could have known from the paper. Since submission we have run a pre-registered follow-up programme aimed at exactly these concerns: held-out evaluation, statistical power, matched ablations, closed-loop execution, engine-truth compute, out-of-domain data and external baselines. Every analysis below had its plan and decision rule frozen before scoring. We report positive and negative outcomes alike, and nothing below should be read as already contained in the submitted PDF.

**Bottom line.** The new evidence supports a narrower paper than the one we submitted. It confirms the reviewers' central doubts about the *decision-level* claims, and we withdraw or correct those claims (table below). Three things survive and are now better supported:

1. **The request interface.** A single latent predictor serves every (horizon Δ, description α) request. Sharing all weights across the nine requests does not cost per-request quality. At matched per-request updates it improves held-out ranking (+0.085 AUC) and reduces rollout error.
2. **Granularity matters.** The choice of request changes long-horizon rollout error by large factors and changes launch ranking by more than any learned selector recovers. This holds on held-out levels and on CLEVRER and Physion from pixels.
3. **A rigorous empirical account of when per-step selection does not pay**, on a new level-disjoint NovPhy evaluation split with 11,680 engine-verdicted launches. The account includes matched controls and faithfully reproduced external baselines. Several of these results are useful to the community in their own right, for example that hierarchical JEPA world models collapse on action-sparse single-launch physics.

**How to read the numbers.** Unless stated otherwise:
- AUC is within-level ranking AUC of engine-verdicted launches (pig removal) under a tie-free endpoint cost.
- Intervals are 95% level-clustered bootstrap intervals (10,000 draws) and are descriptive.
- A difference is called a gain only if its mean is ≥ δ = 0.02 and its lower bound is > 0. This rule was pre-registered.

### What changes in the paper

| Submitted claim | Status after new evidence | Revision |
|---|---|---|
| Choosing the request per prediction beats a fixed request tuned on other sets | **Not supported.** On 311 held-out mixed levels, the submitted policy J ranks at AUC 0.555 [0.539, 0.571]; J − F* = +0.013 [−0.016, +0.043]. No learned selector beats the best fixed request by ≥ 0.02 [E4, E6, E10]. | Withdrawn; replaced by the sealed-split results |
| "Matches a horizon-only adaptive baseline at a third of its compute" | **Not supported** on engine truth. The MAC ratio is 0.784, not ≤ 0.40. The saving comes from the narrower trunk. A fixed coarse request and a no-model prior dominate both policies [E8, E9]. | Sentence removed; engine-truth frontier added |
| Macro is the best description everywhere; axes vary independently | **Not supported.** Macro − continuous = −0.012 [−0.019, −0.004]. Tied-axis restrictions cost nothing [E12, E13]. | Withdrawn |
| CLEVRER/Physion rollouts show generality | **The submitted columns are in-sample** (12 training scenes). New pixel-carrier results on official held-out splits show the same coarse-Δ advantage. Physion has a near-duplicate qualification, and downstream decisions are at chance [E15]. | Table 1 replaced and qualified |
| Shared predictor across requests | **Supported and strengthened** [E14] | Promoted to a main result |

---

## Common concerns

### C1. Held-out evaluation and generalisation (MMed W1/Q1, MSwm W3/Q3, vSA9 Q1)

**The reviewers are right on both counts.**

**CLEVRER and Physion.** In the submitted Table 1, CLEVRER and Physion-Dominoes contexts came from the same 12 scenes used for fitting. We confirmed this when we re-ran the submitted setting, which reproduces the published numbers [E14b]. Those columns measure in-sample rollout, not generalisation.

**NovPhy.** The submitted decision evaluation was worse than "few levels". Of the 15 evaluated members, 8 were predictor-training lineages and 3 were policy-training lineages. Only 4 were held out [E1].

**What we did.**

1. **A level-disjoint NovPhy split, fixed before capture** [E1, E2].
   - 1,000 freshly seeded levels from the two evaluated families, split into fit 480 / policy 120 / evaluation 400.
   - Seeds were checked against 83 prior-exposure sources, with 0 collisions.
   - All 60 candidate launches of each evaluation level were captured and engine-verdicted: 11,680 launches, 0 typed failures, on a repaired capture pipeline with 0/208 smoke failures.
   - Levels whose candidates include both successes and failures ("mixed" levels) are scored: grid 311/400, offset 76/120, angle 78/120, power 47/120.
   - The evaluation split never enters training, policy fitting or selection.
2. **The submitted checkpoint re-scored on this split** [E4]. See C2. It ranks near chance.
3. **CLEVRER and Physion from pixels on official held-out splits** [E15].
   - CLEVRER uses official train videos for fitting and official validation videos for evaluation. Scene disjointness was verified: 6,285 train and 3,132 validation videos after a pre-declared exclusion of videos with ambiguous colour × material keys.
   - Physion-Dominoes uses the official dynamics, readout and testing sets (150 red–yellow test trials).
   - Rollout error at the last endpoint (position MSE against the parsed carrier):

   | request | CLEVRER t = 120 | Physion t = 90 |
   |---|---|---|
   | Δ = 1, continuous | 0.163 [0.152, 0.176] | 0.0430 [0.0327, 0.0535] |
   | Δ = 5, continuous | 0.109 [0.104, 0.115] | 0.0227 [0.0184, 0.0272] |
   | Δ = 15, continuous | 0.0996 [0.0958, 0.104] | 0.0143 [0.0108, 0.0182] |
   | Δ = 15, macro | 0.0974 [0.0944, 0.101] | 0.0146 [0.0110, 0.0187] |
   | policy J (own schedules) | 0.157 [0.147, 0.166] | 0.0235 [0.0188, 0.0285] |

   Coarse Δ still gives the lowest error on held-out scenes (1.6× and 3× below Δ = 1). The learned schedule J stays close to Δ = 1 and does **not** reach the best fixed request.

**Limitations we state explicitly.**
- **Physion near-duplicates.** No full parameter set is shared across Physion splits. However, when appearance-only colour is ignored, 20 of 150 test trials repeat the physics of a fitting trial, and 4 are near-identical. Official-split Physion results therefore measure transfer to new colours, not to new physical realisations [E15].
- **NovPhy near-duplicates.** The NovPhy split is level-disjoint, but its two generator templates produce near-duplicate layouts across splits. A parameter-free nearest-fit-level verdict copy reaches AUC 0.904 on the evaluation grid [E19]. We now report this reference beside every ranker (see C6).
- **Reuse.** The NovPhy evaluation split has been scored by many pre-registered analyses. No model was selected on it, but it is no longer a fresh holdout.
- **Persistence baseline.** MSwm asked for persistence or constant-velocity baselines. These are not yet computed on the held-out CLEVRER/Physion tables. They need no training, and we will add them. Until they are added, we make no claim that the CLEVRER/Physion rollouts beat a trivial baseline. On NovPhy (all novelty cells), persistence is reported [E17]: the best Δ = 15 request beats persistence (normal levels: 0.042 vs 0.105 masked carrier MSE at t = 225), while zero-shot Δ = 1 is *worse* than persistence in 42/45 cells.

### C2. Near-chance ranking and statistical power (MMed W2, MSwm W1/Q2, vSA9 Q4)

**Direct answer.** The reviewers' reading of the submitted Fig. 4 is correct. The submitted gains are not distinguishable from zero, and the ranking signal itself was near chance.

**Submitted evidence, with the numbers MSwm requested** [E4b; per-cell tables to appear in the appendix]:

| set | mixed members / cells | AUC J | AUC F* (cross-fit) | J − F* | J top-1 hits |
|---|---|---|---|---|---|
| grid | 14 / 42 | 0.471 [0.416, 0.526] | 0.401 [0.348, 0.461] | +0.070 [−0.006, +0.144] | 0/42 |
| offset | 9 / 27 | 0.507 [0.456, 0.559] | 0.504 [0.444, 0.559] | +0.004 [−0.080, +0.093] | 0/27 |
| angle | 7 / 21 | 0.566 [0.520, 0.625] | 0.506 [0.423, 0.583] | +0.060 [+0.008, +0.113] | 0/21 |
| power | 8 / 24 | 0.529 [0.469, 0.591] | 0.501 [0.414, 0.606] | +0.027 [−0.069, +0.128] | 0/24 |

These are the four Fig. 1(b2) differences. Only angle clears zero, as MSwm noted. 17/42 grid cells were fully tied under the submitted count cost.

**Root cause** [E3].
- The frozen object parser in the submission detected the pig in **0 of 859** decision frames; the engine has it in all 859. The count-based decision cost was therefore reading noise.
- A new spatial slot parser reads the pig at accuracy 1.0, and a tie-free cost removes ties.
- With perception fixed, the remaining bottleneck was dynamics. We added long-horizon supervision to the 225-frame decision endpoint and a candidate-contrastive loss, selected by leave-one-lineage-out validation on fit data only [E5].

**Result on the sealed split** (pre-registered target: AUC ≥ 0.65 with lower bound > 0.55) [E5]:

| ranker (grid, 311 levels × 3 seeds) | AUC |
|---|---|
| new primary model U (hybrid, request (15, macro)) | **0.744 [0.727, 0.762]**, target passed |
| matched continuous-only model C* (15, continuous) | 0.783 [0.768, 0.799] |
| no-model action prior (success frequency of the same launch on fit levels) | 0.720 [0.697, 0.744] |
| U from a wrong level's anchor | 0.612; right − wrong = +0.132 [+0.122, +0.143] |
| submitted J, re-scored | 0.555 [0.539, 0.571] |

**What this does and does not show.**
- The predictor now uses the level state: the wrong-anchor control is clearly worse.
- U − C* = −0.039 [−0.044, −0.034]. The hybrid stack does **not** beat its matched continuous-only twin.
- U − prior = +0.024 [−0.001, +0.049], below δ. The prior is within δ of most model rankers on grid.

**Does the submitted offset gain of +0.004 hold?** (vSA9 Q4) **No.**
- The submitted checkpoint re-scored on 76 held-out offset levels gives J − F* = +0.011 [−0.052, +0.074].
- On the new primary predictor, J − F* on offset is −0.090 [−0.126, −0.055].
- The grid primary on the submitted checkpoint is +0.013 [−0.016, +0.043], with half-width 0.030. That is still wider than δ, even with 311 levels [E4].

**Task success and top-k (vSA9 Q4, MMed Q3).** Beyond AUC:
- Submitted J had top-1 = 0 on all four sets [E3].
- On the sealed split, top-3 hit rates are: J 0.510 [0.462, 0.561], horizon-only J-cont 0.466, C-J 0.447, prior 0.418 [E7].
- Live engine execution results are in C5.
- Per-level success/failure lists will be released for every inventory. The level × candidate records already exist and are hash-manifested.

### C3. Joint versus horizon-only selection, and where compute goes (MSwm W2/Q4, vSA9 W2/Q3)

**Direct answer.**
- The submitted compute comparison changed several things at once: trunk width, training recipe, data and cost proxy. The reviewers are right that it does not support the "a third of the compute" sentence, and we withdraw it.
- Once confounds are removed, choosing α adds at most about δ beyond choosing Δ. The saving against the horizon-only model comes from the narrower trunk, not from α.

**The ablation MSwm and vSA9 asked for** [E7]. All arms below share the new primary hybrid checkpoint, the same policy-training pool, the same seeds and the same update budget:
- **J-cont** is the same policy network and recipe with DP labels restricted to {(1, c), (5, c), (15, c)} and the six other logits masked.
- **T-α / D-Δ** are restricted argmaxes of J's logits.
- **C-J** is the horizon-only policy on the matched continuous-only predictor.

| arm (grid, λ = 1e-4) | AUC | MACs / candidate |
|---|---|---|
| J (joint, nine requests) | 0.706 [0.685, 0.727] | 184.0M |
| J-cont (trained horizon-only, same trunk) | 0.686 [0.665, 0.706] | 122.7M |
| T-cont (time-only, J's logits) | 0.703 [0.681, 0.724] | 196.1M |
| T-macro (fixed macro, adaptive Δ) | 0.678 [0.656, 0.700] | — |
| D-1 / D-5 / D-15 (fixed Δ, adaptive α) | 0.685 / 0.738 / 0.762 | — |
| C-J (horizon-only, continuous model) | 0.700 [0.683, 0.717] | 234.7M |
| no-model prior | 0.720 [0.697, 0.744] | 0 |

Readings:
- **J − J-cont = +0.0201 [+0.0154, +0.0248].** This is the frozen primary token, `supported` at both declared budgets, with three qualifications:
  1. J-cont cannot spend as much compute and is clamped at its 122.7M endpoint, so this is an at-most-budget comparison, not equal spend.
  2. The estimate barely clears δ.
  3. A duplicate-launch incident affected some fixed-request scoring records. It is disclosed, and the deterministic records were re-verified.
- **J − T-cont = +0.003** and **J − (J's schedule replayed with α = continuous) = +0.004**. Neither reaches δ.
- **J − C-J = +0.006 [0.000, +0.012].** Not supported.
- **J − T-macro = +0.028** (supported). Fixing macro alone hurts.
- **J − D-15 = −0.056 [−0.069, −0.044].** Simply always using Δ = 15 and choosing only α ranks better than J.

**Where the saving comes from (MSwm Q4).** The J − C-J difference at λ = 1e-4 is −50.7M MACs per candidate. It decomposes exactly as:

| component | MACs |
|---|---|
| narrower hybrid trunk | **−50.8M** |
| fewer calls (horizon) | −6.4M |
| micro/macro heads and adapters (description) | **+6.6M** |
| policy | −0.1M |

MSwm's reading is therefore correct. The description level costs compute and does not save it.

**Engine-truth frontier** [E8, E9].
- J and C-J use 184.0M and 234.7M MACs (ratio 0.784; "a third" would need ≤ 0.40) with no ≥ δ AUC difference. J is also not faster in wall time.
- Both are dominated by the fixed request (15, continuous) on the continuous model: AUC 0.783 at 27.8M MACs, +0.078 over J.
- Both are also dominated by the no-model prior at 0 model MACs.
- The grid frontier consists of: the prior, F-15-continuous, J at λ = 0.1 (which degenerates to Δ = 15 at every step), and C-F-15.

**Other ablations of the selector.** We also tested whether richer selectors help. None beats the best fixed request by ≥ δ:
- **Decision-valued DP labels (J_dec).** A post-submission variant, not the submitted DP. It improves on J by +0.040 but trails the hindsight-best fixed request by −0.016 [−0.020, −0.012] [E6].
- **Policy gradient.** It collapses to a single Δ = 15 request [E6].
- **Unsaturated remaining-frames input.** +0.004 [E10].
- **Larger or continuous request sets.** Gains appear only when total updates grow, and the fixed requests gain equally [E11].
- **Policy-only adaptation under novelty.** −0.007 [−0.021, +0.006], while retuning the predictor gives +0.151 [E16].

### C4. Time base: what Δ counts, and what the predictor sees (MSwm W3/Q1, vSA9 W1/Q2)

**Direct answer.** The reviewers identified a real inconsistency in the submitted NovPhy setup, and the submitted paper did not characterise it.

**What Δ counted in the submitted recipe.**
- Predictor training pooled 2,400 fitting lineages of the original NovPhy base corpus with a small number of N1 lineages.
- In the base corpus, consecutive carrier frames are consecutive fixed engine steps. Training windows are 60 fixed steps long, so Δ ∈ {1, 5, 15} meant 1, 5 and 15 engine steps.
- In the N1 data, and in every N1 evaluation, carrier frames are observed frames captured every 50 engine steps. The same Δ therefore meant 50, 250 and 750 engine steps.
- So MSwm's arithmetic is right: at test time, a Δ-jump was about 50× longer in physical time than in most training windows.

**What the predictor could see (vSA9 Q2).**
- Δ enters only through the request code, as sinusoidal features of the integer Δ. It carries no unit.
- The carrier header (Eq. 9) does contain an elapsed-time entry: column 1 holds the elapsed seconds between the prior and current observed frame. The per-object motion columns are centre displacement divided by that elapsed time. These are the only inputs that distinguish one-engine-step spacing from 50-engine-step spacing.
- The request policy has a separate remaining-frames input κ (clamped at 60). It counts carrier frames, not seconds, and does not encode the rate.
- We did not ablate the elapsed-time entry or the motion normalisation for the submitted checkpoint. We therefore **cannot** say whether that checkpoint relied on them to tell the two rates apart. This remains an unresolved limitation for every submitted NovPhy number, and the revision states it.
- Our κ ablation (C3) concerns the policy's remaining-frames input only. It says nothing about the carrier's elapsed-time channel.

**How the revision avoids the issue.**
- The new primary model is trained only on the 480 fit levels of the new split.
- Those levels are captured at one uniform stride of 50 engine steps per carrier frame, the same stride as evaluation.
- Δ ∈ {1, 5, 15} therefore means 50/250/750 engine steps in both training and test.
- All sealed-split results in this response use that stack. Results for the submitted checkpoint move to an appendix that carries the time-base caveat.
- Sub-frame Δ (below 50 engine steps) cannot be supervised from these captures, and we label it unsupported [E11].

### C5. Closed-loop execution and task success (MMed Q3/Q4, vSA9 Q4)

**Direct answer.** The submitted paper evaluated decisions only. We have since executed chosen launches live in the engine.

**Protocol** [E18].
- Retry-from-start on all 400 held-out levels, 3 seeds, shot budget 3.
- Each level has a single bird, so a "replan" only excludes candidates that already failed.
- 19,360 episodes sharing 5,123 live executions. Live verdicts match the recorded verdicts in 5,106/5,106 cases.

| arm (pass@3) | rate |
|---|---|
| oracle | 0.775 |
| fixed (15, continuous) | 0.478 [0.434, 0.523] |
| C* | 0.466 |
| decision-valued J_dec | 0.445 |
| new primary U | 0.424 [0.384, 0.465] |
| submitted-recipe J | 0.394 [0.352, 0.438] |
| horizon-only C-J | 0.346 |
| no-model prior | 0.323 [0.278, 0.370] |
| uniform random | 0.198 [0.172, 0.223] |

**Qualifications.**
- A frozen guard failed. 25 of 5,106 live decision frames differed from their stored anchors by more than the 1e-3 tolerance (up to 0.0032), so every contrast token is "insufficient" and the table is **descriptive only**.
- The descriptive picture matches C2 and C3. Model-based rankers achieve more passes than the prior and random, but the best performer is a fixed coarse request, not the learned selector.
- On 40 appearance-novelty levels, every model − prior interval includes zero.
- Three AIBIRDS competition agents choosing their own continuous shots reach pass@1 of 0.273 (Datalab), 0.110 and 0.120, against 0.151 for U. This is a different action space, so the comparison is descriptive [E19].

**More complex environments (MMed Q4).**
- We extended NovPhy coverage from 4 to all 45 novelty cells, with representation coverage verified for every cell [E17], and evaluated request selection under novelty [E16].
- We ran CLEVRER and Physion from pixels [E15].
- Multi-object, multi-shot agentic settings remain outside this paper's scope, and we say so.

### C6. What does the world model add? (MMed Q2/Q3; context for all reviewers)

The reviewers did not ask this directly, but MMed's question about meaningful improvement led us to run it, and the AC should see the result. We compared U with reproducible external approaches under the same sealed protocol. Configurations were selected on the policy split with a tuning allowance that U did not receive. Each baseline had to reproduce its own published number before use [E19].

| approach | grid AUC | note |
|---|---|---|
| direct success classifier, pixels (PHYRE DQN recipe) | 0.995 [0.992, 0.997] | 0.32× U's training compute |
| direct classifier on U's carrier, matched parameters | 0.976 [0.969, 0.982] | 0.04× U's training compute |
| Dynamics-aware DQN | 0.995 | at 1.88× U's training compute |
| nearest fit level, copy verdict (no model) | 0.904 [0.884, 0.923] | split redundancy |
| U trained to that 1.88× budget | 0.813 [0.798, 0.829] | |
| U | 0.744 | |
| DIAMOND / SlotFormer / DINO-WM (carriers) / THICK / γ-models | 0.718 / 0.548 / 0.525 / 0.499 / 0.510 | all fail a frozen launch-sensitivity guard; no ranking claim either way |
| H-JEPA, pixels / carriers; with its own IDM remedy | 0.500 / 0.490; 0.495 / 0.490 | faithful on its own benchmark, collapses here [E20] |

Readings:
- **Direct prediction wins on this split.** Feeding U's predicted endpoint into the same classifier adds nothing over the classifier without rollout: −0.004 [−0.008, −0.001]. For this task and split, we therefore cannot claim that rolling the world model forward adds ranking value over direct prediction, and the revision says so.
- **The win comes largely from template redundancy.** The nearest-fit-level lookup alone reaches 0.904.
- **On the 40 novelty pairs every ranker collapses.** Direct classifiers fall to 0.52–0.58 on novel sides (0.72–0.76 on the paired normal sides); U falls to 0.47.
- **Contribution.** We regard the evaluation methodology, with its prior, wrong-anchor and nearest-neighbour references, as a contribution. It is exactly what exposes this redundancy.

---

## Reviewer MMed

We appreciate that you found the shared predictor interesting. The new evidence makes it the strongest part of the paper.

**W1 / Q1. CLEVRER scenes overlap; NovPhy has few levels.** Both points are correct (C1).
- The submitted CLEVRER and Physion columns are in-sample on 12 scenes.
- The revised table uses official scene-disjoint CLEVRER train/validation videos and the official Physion-Dominoes splits from pixels, with the Physion near-duplicate qualification.
- NovPhy evaluation moves from 7–14 partly exposed members to a 400-level split. The levels were fixed before capture and are disjoint from training. Results with 311 mixed grid levels are in C2.

**W2. Near-random ranking; description levels do not consistently help.** Both points are correct, and we narrow the paper accordingly.
- The submitted ranker was near chance because its parser never saw the pig (C2).
- With a working encoder and decision-relevant training, model rankers reach 0.70–0.78.
- Description levels do not consistently help. On the matched hybrid trunk, macro − continuous is −0.012 [E12]. Fixed micro/macro requests have rollout error equal to continuous within noise (C1 table). The only ≥ δ description effect is the qualified +0.020 at-most-budget contrast (C3).
- We now say this plainly in the abstract and the results.

**W2 (presentation). Figures 1 and 3 are overcrowded; the main text is underdeveloped.** Agreed.
- The per-set heatmap is replaced by a compact table: nine fixed requests × four sets, with the prior.
- The teaser's panels will be split, with secondary panels moved to the appendix and a caption marking submission-era evidence.
- The freed space goes to the sealed-split protocol, controls (prior, wrong anchor, nearest neighbour), the engine-truth frontier and the limitations.

**Q2. Benefit of symbolic description levels over continuous prediction alone?** Small at best, and not a ranking benefit on their own.
- **Engine-truth labels hurt.** Feeding engine micro labels as feedback lowers fixed-request AUC by 0.345 [E21].
- **Feedback can be removed.** Removing the micro feedback costs nothing [E21].
- **The macro level is not carried by rollouts.** Macro events are only weakly readable even from parsed frames (held-out F1 0.157), and almost not at all through rollouts (F1 0.015) [E22].
- **The macro advantage does not survive a less aligned verdict.** A predictor with redefined macro labels shows +0.022 macro − continuous under the pig-removal verdict, which reverses (−0.063) under the verdict least aligned with the macro predicates [E12].
- What remains is the design value of one interface: symbolic and continuous requests share a single latent carrier without separate latent spaces (vSA9 strength 1), and sharing does not cost quality (E14).

**Q3. Do ranking results translate into task success?** Only partially, and not to an advantage for learned selection. In live execution (C5), model rankers pass more often than the prior (0.32) and random (0.20). The best arm is a fixed coarse request (0.48), not the joint policy (0.39–0.45). These numbers are descriptive because a frozen guard failed.

**Q4. More complex or closed-loop environments?** We added closed-loop live execution (C5), all 45 NovPhy novelty cells (E16, E17), and CLEVRER/Physion from pixels (E15). Results under novelty are negative for policy-only adaptation (−0.007) and positive for predictor adaptation (+0.151). Richer multi-shot settings remain future work.

---

## Reviewer MSwm

Thank you for the precise reading. Several of your numerical observations (the near-chance reference, the single clearing interval, the 1.27× trunk) turned out to be the right threads to pull.

**W1. Weak gains against a near-chance reference.** Correct. The four differences you asked about appear in full in the C2 table: only angle clears zero, and no difference survives on held-out levels. The near-chance reference had a cause, the parser missing the pig, which we fixed. On the sealed split, with base rankers at 0.70–0.78, the learned selector still does not beat the best fixed request (C3).

**W2. Novelty relative to Chatterjee & Khardon (NeurIPS 2025) and Neitz et al. (NeurIPS 2018); no cont-only ablation; the narrower trunk.** Accepted on all three points.
- We will cite and discuss both works. Per-step horizon choice is prior work (adaptive skip intervals; per-step horizon selection). Our remaining delta is the description axis on a shared predictor and its joint selection.
- The cont-only policy on the same hybrid checkpoint now exists (J-cont, C3). J − J-cont = +0.020 is an at-most-budget, qualified result that barely meets δ.
- The compute decomposition confirms your reading: the saving relative to the horizon-only model comes from the narrower trunk (−50.8M MACs), while micro/macro add +6.6M MACs (C3).
- We accept that the novelty is narrower than presented, and we have rewritten the contribution statement accordingly.

**W3 / Q1. Time base.** See C4. You are right that Δ counted engine steps in the 2,400 base-corpus fitting lineages and 50-step observed frames in N1 evaluation.
- The carrier header carries an elapsed-time entry, and motion columns are per-second. Δ itself carries no unit, and we never tested whether the submitted model relied on the elapsed entry. That question stays open for the submitted checkpoint and is stated as a limitation.
- The new primary model removes the mismatch: one 50-step stride in training and evaluation.

**W3 / Q3. Held-out Table 1, Physion-Dominoes details, persistence baselines, the policy's own schedules.**
- **Held-out data.** See C1 for official scene-disjoint CLEVRER and official-split Physion-Dominoes from pixels. Data, carrier frame rate (Physion: one carrier frame per 3 native frames), exclusions and the near-duplicate audit go into the appendix.
- **The policy's own schedules** are in the table: J 0.157 on CLEVRER and 0.0235 on Physion, against 0.0996 and 0.0143 for the best fixed request. On NovPhy, J's own schedule has engine-referenced error 0.032 at t = 225 [E13].
- **Persistence and constant-velocity rows** on held-out CLEVRER/Physion are not yet computed (C1). We will add them and make no claim against trivial baselines until we do.
- **Within NovPhy**, persistence is beaten by coarse requests (0.042 vs 0.105) but not by zero-shot Δ = 1 [E17].
- **Training recipe changes the ordering.** Under long-horizon supervision on NovPhy, Δ = 1 is no longer catastrophic (0.065 vs 3,622 at t = 225, exploratory leave-one-lineage-out) [E5]. The "long jumps help" statement is now scoped to short-window training.

**Q2. Scored levels, successful launches and intervals.**
- The submitted per-set counts and the four Fig. 1(b2) intervals are in the C2 table.
- Per-set level and success counts, plus intervals for every Fig. 4 cell, will appear in the appendix table.
- On the sealed split the mixed-level counts are 311 / 76 / 78 / 47 (C1). Intervals for every arm are reported (E4–E7).

**Q4. Description, horizon or trunk?** Mostly the trunk; see the decomposition in C3.

**Minor 1. TAWM venue.** Thank you. We will cite TAWM as an ICML 2025 main-track paper.

**Minor 2. Undefined terms.** We will add a glossary to §4:
- **lineage**: one procedurally generated NovPhy level with all launches captured from it.
- **base corpus**: the 3,000-lineage normal-level corpus. 2,400 lineages fit the predictor and 600 supervise the policy.
- **N1 captures**: the first novelty-campaign captures of the rolling and sliding families, observed every 50 engine steps.
- **development states**: the N1 anchor states used during method development for the submitted decision evaluation, which were partly exposed to training (C1).

The revision replaces these development states with the sealed split.

---

## Reviewer vSA9

Thank you for the expert questions. Your two strengths, a shared carrier and matched budgets with explicit accounting, are what made the follow-up checks possible.

**W1 / Q2. Temporal-rate encoding is poorly characterised.** Agreed; see C4.
- The submitted predictor received Δ as a unitless request code. It received the 1-step versus 50-step distinction only through the carrier's elapsed-seconds entry and the per-second motion columns.
- We did not ablate those inputs, so whether the model used them is unknown. This is stated as a limitation of the submitted checkpoint.
- The new primary model trains and evaluates at one stride.

**W2. The compute comparison is confounded, and the DP optimises a fixed-context additive proxy.** Both points are correct.

*Compute.* The submitted horizon-only comparison differed in trunk width, training recipe and data. It was also measured on a count-cost proxy. It is withdrawn. Under the controls you asked for (C3), with the same checkpoint, same policy pool and same budget, the joint policy uses 0.78× the MACs of the horizon-only baseline with no ≥ δ AUC difference. The saving comes from the trunk, and both policies are dominated by a fixed coarse request [E8, E9].

*Q5. What the DP optimises, and how round-2 contexts are built.* Exactly as follows.

- **Objective.** For each training window (first 60 frames of a shot) with observed carriers z_t and the context tensor described below, the submitted DP solves

  c(t, r) = Δ_r · mean‖F(context_t, a, r) − z_{t+Δ_r}‖² + λ · work(r) / work(reference),

  V(t) = min_r [ c(t, r) + V(t + Δ_r) ].

  Requests that overrun the window are infeasible, and exact ties go to the lower request index. This is duration-weighted one-jump prediction error plus a compute charge. The backward recursion is the exact minimiser of this additive surrogate over request sequences within the window, **given the context tensor**. It is not the true state-dependent cost of a recursive rollout, and it is not a decision cost.
- **Round 2 (DAgger).** The round-1 policy is rolled out greedily from the window start.
  - Positions it **visits** have their context overwritten by the predicted carrier.
  - Positions it does **not** visit keep the observed carrier.
  - The DP then evaluates every request at every position from this mixed tensor, against the unchanged true targets.
  - So the successor context of an unvisited request is an observed carrier, not a recursive prediction. Labels from both rounds are pooled.
- **Revision.** We will state this definition and its limits in §4.

**Does fixing the objective help?** We tested this after submission, and the answer is no.
- Labels built on 225-frame windows, the full decision horizon, *lower* ranking by 0.026 [E10].
- A decision-valued variant (J_dec) runs the candidate rollouts to the endpoint and scores a candidate-normalised pairwise ranking surrogate. It improves on J by +0.040 but still trails the best fixed request by 0.016 [E6]. J_dec is a new post-submission variant, not the submitted DP.

**Q1. CLEVRER scene IDs and recomputed Table 1.**
- Yes: the submitted fitting windows and evaluation contexts came from the same 12 CLEVRER scenes (C1).
- The recomputed table uses official CLEVRER train videos for fitting and official validation videos for evaluation. Disjointness was verified by an automatic guard. The ID lists will be in the supplement.
- Numbers are in the C1 table. Coarse Δ remains best, and J does not reach it.
- This table uses pixel-derived carriers and the long-horizon recipe, so it is a re-measurement, not a like-for-like recomputation of the annotation-carrier numbers.

**Q3. Same checkpoint, matched decision budget: fixed-macro adaptive-Δ, fixed-Δ adaptive-α, time-only scheduling.** See the C3 table, all on one checkpoint and one J policy:

| arm | grid AUC | J − arm |
|---|---|---|
| T-macro (fixed macro, adaptive Δ) | 0.678 | +0.028 [+0.023, +0.034] |
| D-1 (fixed Δ = 1, adaptive α) | 0.685 | +0.021 [+0.017, +0.024] |
| D-5 (fixed Δ = 5, adaptive α) | 0.738 | −0.032 [−0.042, −0.023] |
| D-15 (fixed Δ = 15, adaptive α) | 0.762 | −0.056 [−0.069, −0.044] |
| time-only, restricted (T-cont) | 0.703 | +0.003 (not supported) |
| time-only, trained (J-cont) | 0.686 | +0.020 (at most budget) |
| time-only on the continuous model (C-J) | 0.700 | +0.006 (not supported) |

The restricted arms are not compute-matched: fixed-Δ arms make far fewer calls. At matched MACs, the engine-truth frontier (C3) gives the same picture: coarse fixed requests dominate.

**Q4. Task success, complete results, and the 0.004 gain.**
- The offset gain does not hold (C2).
- Submitted top-1 was 0 on every set.
- Sealed-split top-3 and live pass@k with intervals are in C2 and C5.
- Complete per-level success/failure lists for every inventory will be released with the appendix.

---

## Revised contribution and changes to the manuscript

**What the revised paper claims.**
1. **One predictor, nine requests, no sharing cost.** On held-out levels, the shared predictor ranks better than single-request specialists given the same per-request updates: +0.085 [+0.074, +0.095]. In leave-one-lineage-out validation (exploratory), its rollout error is 0.07–0.40× the specialists'. Sharing also stabilises Δ = 1 training. The 1.5× bound holds in 3 of 4 CLEVRER/Physion settings (leave-one-scene-out, exploratory) [E14].
2. **Prediction granularity matters**, both for rollout error and for decision quality. No single fixed request is best across launch sets, and on NovPhy coarse requests beat persistence [E8, E13, E15, E17].
3. **Learning to select granularity per step did not beat the best fixed coarse request on held-out data.** This held for every selector variant we tested (prediction-error DP, decision-valued DP, policy gradient, unsaturated κ input, larger request sets, tied-axis restrictions, policy-only novelty adaptation), each pre-registered with matched controls. The description axis adds at most about δ at matched trunk (C3).
4. **Evaluation methodology and external-baseline results.** These include:
   - a level-disjoint split with engine verdicts;
   - prior, wrong-anchor and nearest-neighbour references, which reveal split redundancy;
   - faithful reproductions showing collapse of hierarchical JEPA models on action-sparse physics;
   - the finding that direct classifiers win in-distribution but collapse under novelty [E19, E20].

We are open to retitling the paper so that the title does not imply that learned per-step selection pays off.

**Changes.**
- **Title.** We are open to retitling; we leave this to the AC's guidance.
- **Abstract and §1.** We remove the "beats a fixed request" and "a third of the compute" clauses and state results with δ-rule wording.
- **§2.** We add Chatterjee & Khardon (2025), Neitz et al. (2018), the TAWM venue, end-to-end H-JEPA, and the external baselines.
- **§3–§4.** We add the tie-free cost, the exact DP definition and its limits, the glossary, and the time-base statement.
- **§5–§6.** The sealed split becomes primary, and the new primary model is described. We add the J-cont / restricted-arm table, the engine-truth frontier, and closed-loop results with the guard caveat. Table 1 is replaced by held-out data, persistence rows are added, and the Physion qualification is stated.
- **Limitations.** We will state:
  - the prior sits within δ of most rankers;
  - the hybrid trails its continuous twin;
  - ranking appears limited by training updates (an interpretation, not an isolated cause) [E11];
  - macro is not carried by rollouts;
  - the closed-loop guard failed;
  - Physion and NovPhy splits contain near-duplicates;
  - direct prediction wins on this split;
  - the submitted checkpoint's time base is unresolved.
- **Appendix.** Submitted-checkpoint results are moved here and clearly labelled. Per-cell tables, the protocol and the guard cascade are added.

We hope this response shows that we take the reviews seriously. Where the reviewers doubted the decision-level claims, the evidence agrees with them. What remains is a smaller, carefully tested set of findings about granularity in latent world models.

---
---

# Internal notes (not for posting)

## Evidence traceability

All links are in `github.com/Sino-Huang/NovPhy` (de-anonymising). I read every ticket below in full: body and all comments, including corrections.

| Tag | Source | Exact values used |
|---|---|---|
| E1 | [#109](https://github.com/Sino-Huang/NovPhy/issues/109) | 8 predictor + 3 policy of 15 #96 members exposed; level-disjoint cohort 480/120/400 (240/60/200 per family); 0 collisions / 83 sources; 0/208 smoke failures; held-out rescore J − F* 0.070 [0.033, 0.122] on 3 clusters (understated) |
| E2 | [#104](https://github.com/Sino-Huang/NovPhy/issues/104) | 11,680 eval branches, 0 failures; mixed 311/76/78/47; power below design depth |
| E3 | [#99](https://github.com/Sino-Huang/NovPhy/issues/99) | parser pig 0/859; G4 17/42 ties; top-1 0/42, 0/27, 0/21, 0/24; E99 pig acc 1.0 |
| E4 | [#104 item 3 comment](https://github.com/Sino-Huang/NovPhy/issues/104) | ICLR J 0.555 [0.539, 0.571]; J − F* grid +0.0134 [−0.0163, +0.0431]; offset +0.011 [−0.052, +0.074]; Gate-B J − F* offset −0.090 [−0.126, −0.055] |
| E4b | `.local-artifacts/issue-96-tau-ad-within-checkpoint-v1/findings.md` (local, #96 pre-AFTER-ICLR record) | per-set J, F*, J − F* intervals in the C2 table; also cited in #104/#101 ticket backgrounds |
| E5 | [#108 Gate B](https://github.com/Sino-Huang/NovPhy/issues/108), [#112](https://github.com/Sino-Huang/NovPhy/issues/112) | U 0.7444 [0.7269, 0.7620]; C* 0.7834; U − C* −0.0390 [−0.0442, −0.0338]; prior 0.7203; U − prior +0.0241 [−0.0008, +0.0487]; wrong-anchor 0.6122, right − wrong +0.1322; #112 LOLO (1, cont) B 3622 vs LC 0.065; retrain on #104 fit split only |
| E6 | [#101](https://github.com/Sino-Huang/NovPhy/issues/101) | J_dec − F_hindsight −0.0163 [−0.0203, −0.0122]; J_dec − J +0.040 [+0.029, +0.052]; PG collapses to Δ = 15 |
| E7 | [#114](https://github.com/Sino-Huang/NovPhy/issues/114) + `.local-artifacts/issue-114-horizon-vs-joint-v1/findings.md` | AUC table; J − J-cont +0.0201 [0.0154, 0.0248] (clamped, duplicate-scoring incident); decomposition −50.713 = −6.353 + 6.583 − 50.836 − 0.106; T/D restricted contrasts; top-3 |
| E8 | [#103](https://github.com/Sino-Huang/NovPhy/issues/103) | MAC ratio 0.784; frontier; C-F-15 − J +0.078; wall: frozen single-seed 16.74, post-hoc 3-seed 1.63 (J slower either way) |
| E9 | [#122](https://github.com/Sino-Huang/NovPhy/issues/122) | compute sentence withdrawn in next version |
| E10 | [#119](https://github.com/Sino-Huang/NovPhy/issues/119) | κ unsaturated +0.0042 [−0.0005, +0.0088]; L225 labels −0.026 |
| E11 | [#116](https://github.com/Sino-Huang/NovPhy/issues/116) | update-confounded gains; DP 17.8–18.3 GPU s at \|R\| = 225; sub-frame Δ unsupported (50-step stride) |
| E12 | [#120](https://github.com/Sino-Huang/NovPhy/issues/120) | macro − cont (a) −0.012 [−0.019, −0.004]; ma-ref +0.022 (a), (c) −0.063 descriptive |
| E13 | [#105](https://github.com/Sino-Huang/NovPhy/issues/105) | J − J_diag −0.0068; J − J_monotone +0.0028; J − H* −0.0006; J rollout 0.032 at t = 225 |
| E14 | [#117](https://github.com/Sino-Huang/NovPhy/issues/117) | J − S +0.0848 [0.0742, 0.0952]; rollout ratios B 0.08, LC1 0.07, LC3 0.40; Δ = 1 stabilised; item 5: 1.5× bound supported in 3/4 (Physion LC insufficient), leave-one-scene-out exploratory |
| E14b | #117 item 5 | in-sample fold "all" reproduces #79/#84 (ICLR setting) on 12 scenes |
| E15 | [#106](https://github.com/Sino-Huang/NovPhy/issues/106) + `docs/issue-106-pixel-benchmarks.md` | rollout table; QA readout never fired; OCP 0.51–0.57 vs 0.627 anchor; Physion 20/150 physical repeats, 4 near-identical; exclusion 37.22% |
| E16 | [#107](https://github.com/Sino-Huang/NovPhy/issues/107) | policy-only −0.007 [−0.021, +0.006]; predictor +0.151 [+0.097, +0.206] |
| E17 | [#110](https://github.com/Sino-Huang/NovPhy/issues/110) | 45/45 cells; persistence 0.105 vs F-15-cont 0.042 (level 0 zero-shot); F-1 worse than persistence 42/45; Δ = 15 lowest in 81/85 |
| E18 | [#102](https://github.com/Sino-Huang/NovPhy/issues/102) | pass@3 table; G4 failed (25/5,106 frames, up to 0.0032); N2 model − prior intervals include 0 |
| E19 | [#127 program summary](https://github.com/Sino-Huang/NovPhy/issues/127), [#128](https://github.com/Sino-Huang/NovPhy/issues/128)–[#141](https://github.com/Sino-Huang/NovPhy/issues/141), [#142](https://github.com/Sino-Huang/NovPhy/issues/142) | program table; NN-px 0.9039; F1 − F1-D −0.0044 [−0.0081, −0.0014]; U-native 0.8132; novelty pairs; carrier-space G-action calibration (engine truth 0.0395 < 0.05); E1 agents; not-run arms (DreamerV3, DDLP, Diffusion Forcing, DeepPHY fidelity; V-JEPA 2-AC cap) |
| E20 | [#123](https://github.com/Sino-Huang/NovPhy/issues/123), [#125](https://github.com/Sino-Huang/NovPhy/issues/125) | G-fid pass; HJ-px 0.4997, HJ-car-m 0.4896; IDM 0.4948/0.4900; disposition 3 (+4 for #125) |
| E21 | [#115](https://github.com/Sino-Huang/NovPhy/issues/115) | oracle labels −0.345 fixed / −0.081 J; removal ≈ 0; micro label-free non-inferior only |
| E22 | [#113](https://github.com/Sino-Huang/NovPhy/issues/113) | macro F1 parsed 0.157, one-step 0.113, rollout 0.015 |
| code | `world_model/training/request_policy.py:112-155`; `world_model/data/deployment_temporal.py:680-719`; `world_model/planning/scoring_harness.py:31-35`; `scripts/run_issue_77_n1_train.py:9-19`; `docs/issue-71-hybrid-readiness.md:48-80` | DP objective, DAgger contexts, carrier header (has-prior, elapsed s), unit difference declared in the #77 recipe |

Synthesis tickets read: #108, #124, #127, #142. #142 (carrying the #127 results into the manuscript) is **open**; #126 (reproduction check) is open. Not every AFTER-ICLR result is on the sealed NovPhy split: #106, #117 item 5 and the #127 fidelity runs are not.

## Unresolved evidence risks (for the owner)

1. **Persistence promise.** Persistence and constant-velocity rows on held-out CLEVRER/Physion are promised but not run. They are cheap, but the owner must authorise them, or the promise should be softened.
2. **Submitted counts and intervals.** The C2 table comes from the #96 record. Confirm that it matches the submitted Fig. 4 / Fig. 1(b2) exactly; the reviewers' quoted ranges (+0.004 to +0.070; one of four clears zero) agree with it. The per-set count of *successful launches* was not extracted, so it is promised for the appendix.
3. **Glossary.** The glossary definitions (N1, development states) are inferred from ticket and code usage. Check them against the manuscript's own usage.
4. **Unverified citations.** The Chatterjee & Khardon (NeurIPS 2025) bibliographic details were not verified here, and the TAWM venue was taken from the reviewer. Verify both before citing.
5. **Revised manuscript status.**
   - The #108 carry-over (#124) is done in the local next version, but #142 (the #127 results) is still open.
   - The response promises #127 content and a possible retitling. The owner kept the title (#124 decision 1), so retitling is offered, not stated as done.
   - #124 reports the main text ending on p. 11 in the final build; check the ICLR page limit before uploading.
   - The teaser redraw contradicts the owner decision to keep the teaser (caption-only change). Confirm.
6. **Re-scored submission checkpoint.** The sealed rerun of the submitted checkpoint used the tie-free cost and #105 F77 anchors on new levels. Its time base is the 50-step N1 stride, the same issue raised in C4.
7. **Disclosed qualifications.** The #114 primary token is barely at δ and carries the duplicate-scoring incident. The #102 table is descriptive only (G4 failed). Both are disclosed above; do not upgrade them.
8. **Anonymity.** Do not post repository links (see the top note).
