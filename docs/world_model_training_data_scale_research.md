# Training-data scale for a NovPhy successor world model

**Research date:** 2026-08-30  
**Question:** How many independent training instances should issues
[#62](https://github.com/Sino-Huang/NovPhy/issues/62) and
[#64](https://github.com/Sino-Huang/NovPhy/issues/64) plan for, given the
published JEPA-world-model literature?

## Bottom line

No cited paper establishes a portable threshold at which an action-conditioned
world model has "enough" data. The closest primary evidence nevertheless makes
two conclusions clear:

1. NovPhy's six training lineages are far below the scale used in successful
   action-conditioned JEPA-world-model experiments. The closest published
   settings use hundreds to tens of thousands of complete trajectories.
2. "Enough" must be a held-out empirical stopping claim, not a trajectory or
   frame-count convention. The training-data curve must flatten for recursive
   prediction, action ranking, and gameplay planning, with uncertainty narrow
   enough to rule out a predeclared meaningful gain.

A source-informed planning ladder for #62 is therefore **legacy 6, then about
200, 1,000, 5,000, and 10,000 complete training lineages**. These are reference
rungs, not a claim that 10,000 is universally sufficient. A bounded runtime and
coverage pilot should freeze the feasible maximum and role allocations before
production. If the experiment stops while the held-out curve is still rising,
the correct conclusion is `resource_limited_non_saturated`, not `enough`.

Issue #64 addresses a different sample-size problem. Its final benchmark must
be powered from paired non-final pilot outcomes in units of **independent held-out
level instances**. Repeating several seeds on a handful of levels improves
within-level precision but does not replace more independent levels.

## What the closest papers actually used

### LeWorldModel

The likely intended reference is **LeWorldModel (LeWM): Stable End-to-End
Joint-Embedding Predictive Architecture from Pixels**, by Lucas Maes, Quentin Le
Lidec, Damien Scieur, Yann LeCun, and Randall Balestriero. The
[author paper (v3)](https://arxiv.org/html/2603.19312v3),
[official project page](https://le-wm.github.io/), and
[official code](https://github.com/lucas-maes/le-wm) identify it unambiguously.

LeWM trains a separate roughly 15-million-parameter, end-to-end pixel model for
each environment. Appendix E reports the following offline datasets; Appendix D
reports 224-by-224 frames, frame-skip 5, four-frame sub-trajectories, batch size
128, and ten training epochs.

| Environment | Complete episodes | Mean/declared raw steps | Approximate raw steps |
|---|---:|---:|---:|
| TwoRoom | 10,000 | mean 92 | 0.92 million |
| PushT | 20,000 | mean 196 | 3.92 million |
| OGBench-Cube | 10,000 | 200 | 2.00 million |
| Reacher | 10,000 | 200 | 2.00 million |

These are benchmark configurations, not minimum-data recommendations, and LeWM
does not report a trajectory-count scaling curve. The paper is particularly
useful as a warning against treating count as coverage: LeWM performs worse on
the simple TwoRoom task despite 10,000 episodes, and the authors hypothesize
that the dataset's low diversity and low intrinsic dimensionality conflict with
the high-dimensional Gaussian latent prior. Thus, ten thousand nearly
redundant trajectories need not be enough.

### DINO-WM: the most useful absolute scaling curve

[DINO-WM](https://arxiv.org/html/2411.04983) is an offline,
action-conditioned latent world model with a frozen DINOv2 encoder; its
[official repository](https://github.com/gaoyuezhou/dino_wm) publishes the code,
data, and checkpoints. Its PushT ablation is the most directly useful published
trajectory scaling curve:

| PushT training trajectories | Planning success rate | Decoded SSIM | Decoded LPIPS |
|---:|---:|---:|---:|
| 200 | 0.08 | 0.949 | 0.056 |
| 1,000 | 0.48 | 0.973 | 0.013 |
| 5,000 | 0.72 | 0.981 | 0.007 |
| 10,000 | 0.88 | 0.984 | 0.006 |
| 18,500 | 0.92 | 0.987 | 0.005 |

The paper's full datasets range from 1,000 trajectories for Rope and Granular,
through 1,920 Wall, 2,000 PointMaze, and 3,000 Reacher trajectories, to 10,240
randomized-Wall and 18,500--20,000 PushT-family trajectories. The PushT
trajectories contain 100--300 raw steps, use frame-skip 5, and the reported
models train for 100 epochs.

The curve supports logarithmic, nested data rungs and shows why a single
six-versus-full comparison is weak. It does **not** make 1,000 or 10,000 portable
thresholds for NovPhy: PushT uses replayed expert trajectories with noise,
reachable visual goals, a frozen internet-pretrained encoder, and dense
continuous actions. NovPhy instead has sparse destructive shots, heterogeneous
level instances, multi-shot post-intervention states, deployment-carrier error,
and binary level completion.

### V-JEPA 2 and V-JEPA 2-AC

[V-JEPA 2](https://arxiv.org/html/2506.09985) separates representation learning
from action-conditioned dynamics training:

- The action-free encoder is pretrained on VideoMix22M: 22 million video/image
  samples comprising more than one million hours of video, plus one million
  ImageNet images.
- The encoder is then frozen. V-JEPA 2-AC trains a new 300-million-parameter
  action-conditioned predictor on **23,000 DROID trajectories**, including
  successes and failures, using less than 62 hours of video.
- Training samples four-second clips at 4 fps (16 frames) and optimizes for
  94,500 iterations with batch size 256.

The paper also reports that increasing action-free pretraining from 2 million
to 22 million videos improves representation results, and that curated video
can outperform uncurated data. The relevant lesson for #62 is coverage and
conditioning alignment, not that 23,000 is a stand-alone target: V-JEPA 2-AC
inherits an enormous pretrained visual representation, unlike a from-scratch
NovPhy carrier/predictor comparison.

### Current cross-method JEPA-WM scaling study

The TMLR paper
[What Drives Success in Physical Planning with Joint-Embedding Predictive World Models?](https://arxiv.org/html/2512.24497v3)
and its [official code/data repository](https://github.com/facebookresearch/jepa-wms)
compare DINO-WM, V-JEPA-2-AC, and the authors' recommended JEPA-WM recipe. The
full datasets are:

| Dataset | Complete trajectories | Steps per trajectory |
|---|---:|---:|
| PointMaze | 2,000 | 100 |
| PushT | 18,500 | 100--300 |
| Wall | 1,920 | 50 |
| MetaWorld | 12,600 | 100 |
| DROID subset | 8,000 | 20--50 |

The authors train each method on nested 2%, 10%, 50%, and 100% subsets. They
report that planning performance clearly increases with data for every dataset
and method; DROID and Wall appear less saturated at the maximum, and they argue
that greater data diversity would likely help PushT and MetaWorld as well. This
is strong evidence against declaring sufficiency from one arbitrary full-set
size.

The same study finds that data quantity is not the only intervention that
matters. Its recommended models use multi-step rollout losses (two steps for
simulated navigation and six for real manipulation) and temporal contexts of
three and five respectively. That aligns with #62's requirement to collect
post-shot decision states and with #60's temporal carrier; adding adjacent
single-shot frames alone would not reproduce this training signal.

## Comparison with the current NovPhy evidence

The committed downstream-ingestion evidence records only **six training
rollouts and 2,156 correlated training frame records** in
[`cohort-v2-downstream-ingestion-evidence.json`](../data/runtime_evidence/issue-54/cohort-v2-downstream-ingestion-evidence.json).
The aligned release has six rollouts in each exposure role, as recorded in
[`aligned-observation-release-summary.json`](../data/runtime_evidence/issue-59/aligned-observation-release-summary.json).
Issue #57 subsequently produced zero successes in all 75 trials, while its
summary explicitly says that causal data insufficiency was not established;
see
[`cohort-v2-gameplay-success-summary-v2.json`](../data/runtime_evidence/issue-57/cohort-v2-gameplay-success-summary-v2.json).

Relative to the papers above, NovPhy differs in ways that can push the required
count either down or up:

- Its object/physics carrier and 3,023,202-parameter matched-capacity model
  (recorded in
  [`capacity-integrated-calibration-summary.json`](../data/runtime_evidence/issue-15/capacity-integrated-calibration-summary.json))
  are much smaller than LeWM's pixel model and the 19M--300M predictors in the
  cited work, which may reduce sample demand.
- Its frames within one physics rollout are highly correlated; six independent
  action/level lineages remain six independent lineages regardless of the
  number of adjacent frames.
- Angry Birds levels vary in layout, material, contacts, destruction, and
  multi-shot state. Generalization across generated levels can require more
  distinct level instances than fixed-layout PushT.
- The prior data are single-shot, while deployment is repeated observe-plan-act.
  This coverage mismatch cannot be repaired by resampling old frames.
- The #57 visual adapter supplied zero velocities, and all observations were
  classified structure-unstable. Data scale cannot by itself repair a missing
  carrier input or an objective that does not rank useful actions.

The defensible inference is therefore not "NovPhy needs exactly 10,000." It is:
**six is an inadequate basis for a deployed recursive planner, and the next
cohort must be large enough to measure its own saturation.**

## Recommended refinement for issue #62

The existing issue correctly makes the complete scenario lineage—not a frame—the
assignment unit. It should additionally freeze a nested geometric learning
curve rather than only a minimum-six subset and one unspecified larger set.

Recommended reference ladder:

| Rung | Purpose |
|---:|---|
| 6 | Reproduce the legacy data regime; not a plausible sufficiency candidate |
| about 200 | First source-grounded low-data diagnostic; DINO-WM PushT was still at 8% success |
| about 1,000 | First meaningful medium-scale checkpoint; not presumptively sufficient |
| about 5,000 | Test whether prediction and planning gains persist at substantial scale |
| about 10,000 | Test for saturation in the range used by successful task-specific JEPA-WMs |

The bounded pilot should determine collection cost and may freeze a smaller
maximum if 10,000 is infeasible. However, every planned rung must be a nested,
outcome-independent set of complete lineages with level-instance, generator,
behavior-policy, and action-stratum coverage preserved. Training and evaluation
should use multiple seeded fits per rung and the same held-out calibration and
model-selection lineages.

A rung is "enough" only if all predeclared conditions hold:

1. Held-out recursive rollout error and physical/event diagnostics no longer
   improve materially at the next rung.
2. Action-ranking agreement and counterfactual action sensitivity no longer
   improve materially at the next rung.
3. Non-final matched planning/gameplay performance no longer improves
   materially at the next rung, and the uncertainty interval rules out the
   frozen practical gain.
4. No required level-family, action, post-shot-state, terminal, or interaction
   stratum remains coverage-limited.
5. Training has not merely saturated its update budget; optimization exposure
   is comparable and validated across rungs.

Training loss, teacher-forced one-step error, total frame count, or one flat
adjacent pair are insufficient stopping criteria.

## Recommended clarification for issue #64

Training-data scale should not determine final-benchmark size. DINO-WM commonly
evaluates 50 initial/goal cases, and LeWM evaluates reachable pairs sampled from
its offline trajectories; neither study presents those counts as a confirmatory
power calculation.

The #64 pilot should estimate baseline success prevalence, paired discordance
between systems, and within-level correlation. It should then power the sealed
benchmark on independent level clusters for the frozen primary comparison and
effect margin. Trial seeds are repeated measurements within a level. A design
such as issue #57's five levels by three seeds has 15 trials per system but only
five independent held-out level instances for claims about new levels.

This clarification strengthens, rather than changes, #64's current rule: final
levels must be generated from prospectively frozen seeds and may never be
screened, replaced, or reordered after observing planner outcomes.
