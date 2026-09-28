# Issue #112 — decision-relevant dynamics on E99 carriers (the Gate-B model)

Parent: #108. Shared rules of #85 apply; #111 scoring contract adopted. Prior dispositions (#15–#100, #109,
#111) are read-only inputs; nothing here edits the ICLR 2026 submission.

| Version | Runner | Artifacts |
| --- | --- | --- |
| v1 (frozen 2026-09-27T13:08:12Z, before training and scoring; recipe frozen by selection 21:22:45Z, before any held-out record) | `scripts/run_decision_relevant_dynamics.py`, `world_model/training/decision_dynamics.py` | `.local-artifacts/issue-112-decision-relevant-dynamics-v1/` |
| v2 controls (frozen 2026-09-27T21:36:19Z, **after** every v1 outcome) | `scripts/run_decision_dynamics_controls.py` | `.local-artifacts/issue-112-controls-v2/` |
| v3 stable recipe (frozen 2026-09-28T01:18:40Z, after every v1/v2 outcome; LOLO only; **the Gate-B recipe**) | `scripts/run_decision_dynamics_versions.py --version 3`, `world_model/training/stable_decision_dynamics.py` | `.local-artifacts/issue-112-decision-dynamics-v3/` |

Validation (all exit 0, `conda activate novphy && source env.sh`):

```
python -u -m scripts.run_decision_relevant_dynamics --validate
python -u -m scripts.run_decision_dynamics_controls --validate   # also runs the v1 --validate
python -u -m scripts.run_decision_dynamics_versions --version 3 --validate
pytest tests/test_issue_112_decision_dynamics.py tests/test_issue_112_stable_dynamics.py
```

Zero engine seconds. Every scoring phase and `--validate` runs under `deterministic_scoring()` in a process that did not
train; every record carries `SCORING_POLICY`. Intervals are member-clustered percentile bootstraps (10 000 draws, PCG64
7201): leave-one-lineage-out (LOLO) results are EXPLORATORY; held-out results are DESCRIPTIVE.

## Design

- Encoder fixed: E99 (#99 spatial slot parser); its carriers of the 745 fit shots; anchors are E99 batch-1 parses of each
  member's sealed anchor.
- Four arms, both predictor families (hybrid: 9 fixed requests; continuous: C-F-1/5/15), same architectures
  (1 837 690 / 1 862 396 parameters), same 9000 updates of 64 #77 windows:
  - **B**: #77 recipe (`loss_for`, re-expressed in masked form, equal to the reference loss; smoke and unit test).
  - **L**: + long-horizon recursive loss: windows at starts 0–60 unrolled `horizon // Δ` transitions, MSE at every visited
    position; horizon curriculum 60 → 225 (+15 every 500 updates); full back-propagation (no truncation).
  - **C**: + pairwise ranking loss on the tie-free cost (`scoring_harness.tie_free` on tensors): 2 mixed fit-lineage cells
    per update, every verdict-bearing candidate rolled out from the member's anchor to 225 with the step's pair;
    softplus((c_success − c_failure)/0.1), weight 0.1, ramp over updates 1000–2000. Verdicts from fit-lineage
    inventory executions only.
  - **LC**: both.
- 13 folds (each of the 12 fit lineages left out, plus 'all') × 3 seeds = 39 members per (arm, family), trained in
  lock-step by a vmapped, compiled transition (`decision_dynamics.Ensemble`: per-member AdamW and gradient clipping;
  a member with a nonfinite loss or gradient norm is retired as a typed terminal failure).
- Selection by the frozen rule: S(arm, request) = mean of the grid and offset LOLO AUC under the tie-free cost; argmax.

## 1. Diagnostics (frozen #99 E predictors, fit lineages, zero training)

| request | predicted / engine SD of endpoint pig presence | Spearman(predicted, engine) of the tie-free cost | engine rollout error t=60 → 225 |
| --- | --- | --- | --- |
| F-1-* | 2.0–2.9 | 0.05–0.19 | 1.1–1.8 → 3649–3978 |
| F-5-* | 0.024–0.031 | 0.05–0.09 | 0.033–0.035 → 0.31–0.36 |
| F-15-* | 0.006–0.007 | −0.10 to −0.02 | 0.025 → 0.030–0.037 |
| C-F-1 / 5 / 15 | 2.51 / 0.047 / 0.011 | 0.01 / 0.02 / −0.02 | 1.26 → 2298 / 0.036 → 0.45 / 0.025 → 0.038 |
| E99 parsed endpoint | 0.0015 | 0.36 | |

Endpoint variance shares of the tie-free cost (grid, 10 members with every candidate executed):

| source | anchor | action | residual (anchor × action) |
| --- | --- | --- | --- |
| engine | 0.040 | 0.104 | 0.856 |
| #99 E predictors, 12 requests | 0.83–0.99 | 0.01–0.16 | ≤ 0.007 |

Reading: the engine endpoint is dominated by the anchor × action interaction, while the #99 E rollout endpoint is almost
entirely a function of the anchor. At Δ = 5 and 15 the predicted endpoint barely moves with the action (0.6–3 % of the
engine spread); at Δ = 1 it moves by noise (carriers diverge by 225). This is the dynamics failure the arms target. The
predictors do use the start carrier; they do not produce anchor-specific action effects.

## 2. Leave-one-lineage-out AUC (tie-free; EXPLORATORY)

Grid, request-mean over the 12 requests (11 lineages, 33 cells): B 0.503, L 0.485, C 0.669, LC 0.661.

| contrast (grid, request-mean) | estimate | token (v1 rule) |
| --- | --- | --- |
| Q1 long horizon L − B | −0.0240 [−0.0754, 0.0166] | readiness_or_precision_insufficient |
| with the contrastive loss, LC − C | −0.0078 [−0.0865, 0.0708] | |
| Q2 contrastive C − B | +0.1661 [0.0770, 0.2526] | readiness_or_precision_insufficient |
| with long horizon, LC − L | +0.2058 [0.1151, 0.3064] | |
| Q3 selected LC : C-F-15, grid AUC | 0.7717 [0.6867, 0.8538] | readiness_or_precision_insufficient |

All three tokens are forced by guard G2: 28 of 312 members were retired with nonfinite losses or gradient norms (L 5/78,
LC 23/78, B and C 0). G1, G3, G4, chronology and caps pass. Without G2 the frozen rules would read Q1
`not_supported_by_this_experiment` (upper 0.0166 < 0.02), Q2 `supported` and Q3 `supported`. This is a reading, not a
disposition. Q3's estimate is optimistic: the selection scans 48 units, and every one of the top 12 is an LC unit.

Selected recipe (frozen before any held-out record): **arm LC, request C-F-15, continuous family**
(S = 0.7559; offset 0.74). Full per-request tables for every inventory are in v1 `findings.md` / `comparisons.csv`.

## 3. Rollout error at t = 225 (#100 engine-referenced metric)

| request | B LOLO | L LOLO | C LOLO | LC LOLO | B held-out | L held-out | LC held-out | E99 held-out |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| F-1-continuous | 3622 | 0.088 | 5070 | 0.065 | 3359 | 0.060 | 0.043 | 3773 |
| F-5-continuous | 0.303 | 0.041 | 4.25 | 0.067 | 0.373 | 0.045 | 0.063 | 0.343 |
| F-15-continuous | 0.037 | 0.030 | 0.134 | 0.040 | 0.064 | 0.042 | 0.048 | 0.060 |
| C-F-1 | 2116 | 0.034 | 10635 | 0.061 | 1931 | 0.048 | 0.102 | 2289 |
| C-F-15 | 0.031 | 0.031 | 0.171 | 0.041 | 0.046 | 0.047 | 0.059 | 0.061 |

Long-horizon supervision removes the Δ = 1 divergence: the error falls from about 3600 to 0.09, and L − B at
(1, continuous) is −3622 [−4089, −3150]. The Δ = 5 error falls about sevenfold. Alone, it does not change ranking (Q1).
The contrastive loss alone improves ranking but degrades long-horizon accuracy (C at C-F-5: 44.9 versus B 0.27). LC keeps
both.

## 4. Held-out (007/008/014/016; scored once after the selection freeze; DESCRIPTIVE)

- Selected LC : C-F-15, grid: **0.6444 [0.4667, 0.8667]**. Below the Gate-B target; 3 member clusters.
- Grid requests meeting 0.65 with lower > 0.55: C 10/12 (0.75–0.78), LC 6/12 (hybrid F-5/F-15, 0.79–0.82),
  L 1, E99 reference 1 (C-F-1 0.7259, reproducing #99), B 0.
- Grid request-mean contrasts: LC − B +0.359 [0.283, 0.417], C − B +0.294 [0.057, 0.435], L − B −0.023.

## 5. v2 controls (post hoc; frozen after the v1 outcomes)

The C arm's AUC was identical across the nine hybrid requests on several inventories. That raised the possibility of an
action-only shortcut, which could pass a ranking gate without dynamics. v2 re-scored every v1 model from the wrong anchors
(every other fit member's anchor). Guard G5: the right-anchor rows reproduce every v1 row bit-for-bit.

| LOLO grid, request-mean | right − wrong anchor AUC | rank stability (Spearman right vs wrong) |
| --- | --- | --- |
| B | −0.0005 [−0.0120, 0.0111] | 0.869 |
| L | +0.0070 [−0.0056, 0.0190] | 0.688 |
| C | +0.1679 [0.0597, 0.2814] | 0.032 |
| LC | +0.0798 [0.0367, 0.1203] | 0.602 |
| **Q5 selected LC : C-F-15** | **+0.2108 [0.1413, 0.2804] — supported** (EXPLORATORY, post hoc) | 0.349 |

- The contrastive gains depend on the anchor state. With wrong anchors every arm falls to about 0.5, so they are not an
  action-only shortcut. B's ranking barely changes with the anchor and stays at chance.
- Endpoint variance shares move toward the engine's interaction structure. For LC : F-5-micro on the grid, anchor / action /
  residual = 0.47 / 0.23 / 0.30, against 0.98 / 0.02 / 0.001 for B.
- The no-model action prior (success frequency of the identical action over the other fit lineages) gives a grid AUC of 0.35
  under LOLO: each lineage's successful actions are rare elsewhere. On the held-out grid it gives **0.90**, and on the
  held-out power set 0.96. On these exposed lineages, a verdict-frequency prior beats every model.

## 6. v3: stable recipe (LOLO only; held-out not scored)

Diagnosis of the v1 failures:
- **Where:** all 28 retirements were Δ=1 updates at curriculum horizon ≥ 165.
- **Why:** an instrumented replay measured Δ=1 gradient norms of up to 2.2e3 (long-horizon term) and 1.1e4 (ranking term),
  against about 1 for the #77 local term. This is an exploding gradient through up to 225 recursive transitions.
- **What stayed finite:** every Δ=5 update (45 transitions) of all 312 members.

Change (v3 spec, frozen before training):
- **Truncated back-propagation every 45 transitions.** Δ=5 and Δ=15 are unchanged; Δ=1 gets 45-frame gradient segments.
- **Skip-on-nonfinite.** A nonfinite update is skipped per member, and a member is retired only after 10 consecutive
  skips.

The v1 runner and module stay hash-bound and untouched: v3 runs a private copy of the v1 implementation with the frozen
v3 spec. The version policy (v4–v6 need a named deficiency and stop when S gains < 0.02; the Gate-B recipe is the last
version whose G2 passed) is part of the v3 plan.

| question (LOLO, EXPLORATORY) | v1 | v3 |
| --- | --- | --- |
| training stability (G2) | 28/312 retired | **0/312 retired, 0 skipped updates** |
| Q1 long horizon L − B, grid | −0.0240 [−0.0754, 0.0166], insufficient (G2) | −0.0100 [−0.0569, 0.0370], readiness_or_precision_insufficient |
| Q2 contrastive C − B, grid | +0.1661 [0.0770, 0.2526], insufficient (G2) | +0.1200 [0.0337, 0.1951], **supported** |
| Q3 selected unit, grid AUC | LC:C-F-15 0.7717 [0.6867, 0.8538], insufficient (G2) | LC:F-15-macro **0.7153 [0.6564, 0.7731], supported** |
| Q5 anchor dependence of the selected unit, grid | +0.2108 [0.1413, 0.2804] (v2) | +0.1965 [0.1346, 0.2661], **supported** |
| selection score S (grid, offset mean) | 0.7559 | 0.6938 |

- **v1's S was optimistic:** it averaged over the surviving members of an unstable recipe, and the top unit scanned
  48 units. v3's S covers every member.
- **Grid, request-mean AUC:** B 0.508, L 0.498, C 0.628, LC 0.625. The selected unit's offset AUC is 0.672, but its offset
  anchor dependence is +0.091 [−0.029, 0.221], weaker than on the grid.
- **Contrasts beyond Q1/Q2:** LC − C is −0.003 on the grid and +0.184 [0.078, 0.328] on offset. Long-horizon supervision
  adds ranking value only together with the contrastive loss, and only on offset.
- **Rollout error at t = 225 (LOLO):** L and LC keep hybrid Δ=1 in the 0.05–0.06 range (LC C-F-1: 0.17), against 3600–3900 for B. C alone
  diverges further than in v1 (13 671 at F-1-continuous).
- **LC-continuous dynamics cost:** final local loss 0.018, against B's 0.00075. The contrastive term costs dynamics
  accuracy in the continuous family; the selected unit is hybrid, where LC's local loss equals B's.
- **Series stopped at v3.** Under the frozen policy another version was allowed, but no deficiency justified it for the
  Gate-B unit. Each version adds a look at the same 11 lineages, and a v4 would bind Gate B even if it scored lower.

## Gate-B handoff (v3 `gate_b_handoff.json`; supersedes the v1 handoff)

- Recipe: v3 arm LC (v3 plan `training` block + `version_spec`); request **F-15-macro**; hybrid family; encoder E99.
- Checkpoints: `models/LC/hybrid/fold-all/seed-{20260908,20260909,20260910}/predictor.pt`, all three complete. Load them
  with `implementation(3).load_member(OUTPUT, 'LC', 'hybrid', 'all', seed)`. 0 of 39 LC-hybrid members retired.
- **Scope (declared 2026-09-28, after the v3 publication, before any Gate-B scoring):** Gate-B candidates are
  hybrid-family units only. The paper's claim is joint horizon-description selection. The continuous family
  (C-F-1/5/15, C-J; the parameter-matched pure-continuous model of #74/#77, horizon only) is a matched baseline: it is
  trained with the identical recipe and is never a candidate. The hybrid family's own continuous-description requests
  (F-*-continuous) remain candidates. The v3 selection, a hybrid unit, is unchanged.
- **Pre-declared Gate-B contrast:** the selected hybrid unit minus C*, the continuous-family unit with the largest S under
  the same rule. On v3 LOLO: LC:F-15-macro − LC:C-F-15 = **+0.048 [−0.035, 0.149]** (grid, 33 cells, EXPLORATORY). The
  hybrid advantage over the matched baseline is not yet resolved.
- **Consequence for #99:** its Q3 "supported" rested on C-F-1, a continuous-family unit. Under this scope, that is
  baseline evidence, not method evidence.
- Gate B should report the no-model action prior and the wrong-anchor control next to the model (v2 and v3 controls). On
  the exposed lineages, the prior alone clears the target.
- The held-out lineages were not re-scored for v3: v1 used their single post-freeze look. The unbiased test of the v3
  recipe is Gate B on the #104 sealed split.

## Post-freeze changes (disclosed)

1. After the selection freeze, before any held-out record: the handoff gained per-checkpoint completeness, usable seeds
   and the retirement count. The selection is unchanged.
2. After the first v1 publication: rollout-error aggregation now averages each lineage's available seeds. Before, any
   typed failure dropped the whole lineage; the AUC tables already used the available seeds. This affects only the
   rollout tables; the LC-continuous held-out rollout had read n/a. The training table also shows n/a for loss terms an
   arm does not use.
3. v1 guard G3's held-out spot re-score was not exercised: the member it picks (LC/continuous/all/seed 20260908) is
   retired. v2 G5 re-scores every complete held-out and LOLO right-anchor block against the v1 records (max delta 0).
4. After the v2 freeze and the v2 records, before the v2 publication: the v2 runner reads its own records with the v2
   identity check. It had used the v1 reader, which rejects v2 records. No record or statistic changed.
5. v3 plan re-frozen before training (commit eb3cabbb): the policy text had lost a wrapped line; the spec was unchanged.
   v3 skip counts moved out of `training/`, because the v1 handoff code globs it. The v3 series-continuation field
   applies the stop rule only after a stable parent, as the frozen policy states. No record or statistic changed.
6. After the v3 publication: the Gate-B scope and the baseline contrast were added to the v3 handoff and compute
   tables, and `--validate` now also checks the handoff. The selection and every existing statistic are unchanged.

Compute: v1 29 964 GPU s (training 28 700 s; diagnostics 239 s; LOLO scoring 852 s; held-out 411 s), wall 29 995 s;
v2 about 1.6 h of deterministic scoring; v3 25 748 GPU s (training 24 822 s, LOLO 926 s) plus 864 s of controls and a
12 min failure replay; engine seconds 0. Checkpoints (2.2 GB per version) and training-data caches stay local under the
existing ignore policy; records, plans, selections, handoffs and renderings are tracked.
