# Writing Outline — Granularity Paper (v3.0, 2026-09-26)

Supersedes v2.x (ranking-failure thesis). The file name is kept because tex comments point here. Where `CONTEXT.md` ("Thesis = selection, not ordering") or older BRIEF blocks disagree with this file, **this file wins**; those notes get a docs sync after the body rewrite.

## 0. Thesis and reader

- **Reader.** A JEPA / latent world-model researcher (Dreamer, TD-MPC, V-JEPA 2-AC, DINO-WM). They have never seen NovPhy, and they know nothing about our ranker, proxy or statistics conventions.
- **Thesis (one sentence).** World models fix their *prediction granularity* (how far ahead one prediction reaches, and what level of description it predicts) as a design constant. In action-sparse persistent-effect environments, that choice plausibly changes as the cascade moves through flight, contact and settling. We make granularity a per-decision variable of one shared latent world model: formulation, architecture, controller training. We then evaluate it with a pre-registered protocol on NovPhy, where the results are negative or insufficient, and we say so plainly.
- **Pitch line.** "How far ahead to predict and at what level of detail should be decided state by state, not fixed at design time."
- **Title.** Frozen: *Granularity Is a Decision: Joint Horizon-Description Selection in World Models*. The title poses the question; the body supplies the formulation, the mechanism and an honest test. The test does not show a benefit.

## 1. Contribution–evidence map

| # | Contribution | What is new vs. closest work | Evidence / artifact | Scope boundary |
|---|---|---|---|---|
| K1 | **Problem formulation.** Action-sparse persistent-effect environment kept as an MDP at engine-step granularity. The per-decision *prediction request* r = (Δ, α), Δ ∈ {1,5,15}, α ∈ {continuous, micro, macro}, is a decision separate from the action. Three estimands τ_tr, τ_ex, τ_ad keep training, executed-readout and selection effects separable. | PHYRE frames single-action tasks as a contextual bandit (Bakhtin 2019). Options / action persistence / delayed MDPs are *control-side* commitments (Sutton 1999; Metelli 2020; Katsikopoulos 2003). None makes the prediction itself the per-step decision. | method.tex §3, §3.1; appendix G; code `cnn_hybrid.py:216-239` (adaptive_rollout: semi-MDP over carrier frames). | The environment class itself is not new. The claim is about the modeling of the request. |
| K2 | **Architecture.** One continuous 236-d latent carrier built by a frozen CNN object-slot parser (18 slots × 13 features + 2). A depth-3 residual predictor (hidden 384) is conditioned on (carrier, action) and a **joint** (Δ, α) code: sinusoidal Δ fused with an α embedding, AdaLN-Zero/FiLM on every block. Micro heads (contact/support over 18×18 slots) and macro heads (steady-state, structure-unstable) feed soft predicate readouts back into the transition through bias-free adapters. The rolled-forward state is always continuous. 1,837,690 params; capacity-matched continuous-only baseline 1,862,396. | H-JEPA (LeCun 2022) binds time scale to abstraction level by hierarchy level. SALT learns a joint abstraction once, as a fixed repertoire. Temporal-only and description-only adapters vary one axis. | `cnn_hybrid.py:33-87`, `predictor.py:214-278`, `matched_dynamics.py:15-69`. | **JEPA-style latent prediction over a frozen object-slot encoder.** No EMA target encoder or stop-grad target branch in evaluated systems. Say this plainly; never call it a full JEPA. |
| K3 | **Controller training.** 242→128→9 MLP picks one of 9 pairs per rollout step from (carrier, action, remaining time); horizon-infeasible pairs masked. Trained by cross-entropy on trajectory-optimal DP labels: cost = Δ × carrier MSE + 1e-4 × (linear MACs / continuous-h15 MACs), with continuation value. Round 2 is DAgger-style relabeling on controller-visited contexts. No RL, no task-outcome labels. Training schedule: 9,000 updates × batch 64 on 2,409 lineages; pairs cycled round-robin (PAIRS[step % 9]); loss = local MSE + recursive MSE + 0.01 carrier-bound + 0.1 symbolic BCE (hybrid). | Per-decision selection over a joint request. MuSix routes among a mixture of models; Wang 2023 selects resolution only. | `cnn_hybrid.py:105-213`, `run_issue_77_n1_train.py:140-570`. | The controller never ran in closed loop. |
| K4 | **Pre-registered evaluation on NovPhy with engine-truth outcomes**, reporting where granularity matters and where selecting it did not help. | NovPhy is unused as a world-model testbed (3 S2 / 2 OpenAlex citations; no world-model user). Frozen plans, typed failures, fixed disposition tokens. | See §3 evidence list. | Negative/insufficient results reported as such; no gameplay-benefit claim. |

## 2. Evidence the §5 rewrite draws on (granularity-first order)

**G. Granularity matters (the variable is not inert).** Source: `.local-artifacts/issue-96-tau-ad-within-checkpoint-v1/{comparisons.csv,findings.md}`. Engine verdicts; decision-only; member-clustered descriptive intervals. Every G1 statement is a post-hoc reading of pre-declared descriptive statistics about ORDERING (AUC), not selection.
- G1a (lead statistic): **the best fixed request changes with the candidate set.** Pooled fixed-request AUC:
  - offset sweep: F(5, macro) best at 0.5944 [0.5167, 0.6741];
  - original angle sweep: F(5, macro) best at 0.6346 [0.5873, 0.6786];
  - drag grid: F(5, macro) worst at 0.4006 [0.3475, 0.4612], while F(1, macro) is best at 0.5126 [0.4675, 0.5626];
  - launch power: F(1, macro) best at 0.5274 [0.4737, 0.5921].
  - Pooled per-request AUC ranges: grid 0.4006–0.5126, offset 0.4407–0.5944, angle 0.4502–0.6346, power 0.4516–0.5274.
  - Hence the cross-fitted F* (best on the other three sets) is the worst request on the grid.
- G1b (secondary, needs a caveat): within one decision cell, the nine requests' AUCs differ by **0.5098** on average (per-cell max − min, member mean; [0.4238, 0.5969]); offset 0.4444, angle 0.4969, power 0.4909. There is no null reference, and a per-cell AUC with one success is coarse, so this is an upper-bound-like descriptive. **CORRECTION 2026-09-26:** round-1 §1 ¶5 described 0.5098 as the span of the nine requests' AUC; that is wrong. The pooled span on the grid is 0.1120.
- G2: horizon-dependent carrier stability (C5; carrier MSE, not proxy-scored).
  - h=1 recursive carrier MSE exceeds 10² from t=150 in both arms, while h=15 stays bounded.
  - Lookahead-table carrier MSE: h1 2.32e3 / 3.25e4 (continuous), 1.61e3 / 4.09e4 (hybrid); h15 0.0262 / 0.179 and 0.0217 / 0.114.
  - CLEVRER S3 bounds generality.
- G3: the selector is exercised. #96 G5: non-modal step share 0.241; Kendall τ < 1 on 0.357 of cells.
- G4: cost differs by granularity (C12). Per-state walls: continuous h1 ≈ 1.79 s, h5 ≈ 0.37, h15 ≈ 0.13, hybrid adaptive ≈ 1.01, continuous adaptive ≈ 2.01.
- G5: the fixed-horizon rankers look 1–25 carrier frames ahead, while N1 outcomes resolve 97.8–400.9 frames later. Across 16 lookahead arms, longer horizon shows no consistent direction (C19). Motivates the variable; does not show a fix.

**S. Did choosing granularity help? (negative / insufficient.)**
- S1: sealed confirmatory (#15), teacher-forced endpoint carrier MSE. Paired gain −5.2e-4 [−7.1, −3.6]e-4 at the higher budget; zero at the lower. `not_supported_by_this_experiment`.
- S2: #96 within-checkpoint τ_ad. θ_grid = AUC(J) − AUC(F*) = 0.069983 [−0.005697, 0.143651]. Guards G4 (ties 17/42 = 0.4048 > 0.20) and G6 (half-width 0.0747 > 0.05) failed → `readiness_or_precision_insufficient`. The sign reverses against the hindsight pair (−0.0420).
- S3: deployed controller vs selected continuous_h5, −0.0200 [−0.045, +0.003], proxy-scored (appendix only).
- S4: τ_tr / τ_ex heuristic record (+0.0887 one-seed, demoted; macro h15 +0.0944 seed-mixed). Appendix E only.
- [OPTIONAL, needs owner OK and a manuscript source] #76 utility diagnosis: the compute charge exceeds symbolic local gains ≈52×, and DP teachers collapse to one pair. Candidate §6 explanation for why the controller learned little.

**E. Evaluation validity (compact; supports K4, not a headline).**
- E1: the replay-cost proxy is invalid (agreement 0.1299 vs 0.50 floor; engine channel 0.1064, +0.8958). We therefore score on engine removal events.
- E2: an engine-measured per-state ceiling turns a zero into a measurement: 7/14, 9/15, 14/15, 8/15.
  - Fixed-granularity rankers' pooled top-1 is at or below matched chance: 0/108, 7/81, 8/126, 0/72 (0/72 support-confounded).
  - Ordering AUC moves with the inventory (0.4180 → 0.7441), so AUC does not certify selection.
  - Held-out split: 13/612 zero-shot vs 0.0407; few-shot 29/612 reported separately.
  - One table, one paragraph.

## 3. Section plan and page budget (9 pages main text)

| § | File | Content | Pages |
|---|---|---|---|
| Fig. 1 + Abstract | main, abstract | Teaser (to be redesigned later around the request idea; figure task) + abstract rewritten after §1 | 0.9 |
| 1 Introduction | introduction | ¶1 world models fix granularity (Dreamer/TD-MPC fixed step; JEPA fixed target; V-JEPA 2-AC/DINO-WM fixed MPC horizon). ¶2 H-JEPA names multi-scale, multi-abstraction prediction but binds them by hierarchy level; single-axis relaxations exist. ¶3 action-sparse persistent-effect setting: one launch → cascade with phases whose useful granularity differs; PHYRE collapses it to a bandit; we keep the MDP and make the prediction request the decision; NovPhy as the testbed. ¶4 what we build (K1–K3, one-sentence mechanism). ¶5 what we find: granularity matters (G1, G2) but selecting it did not yet help (S1, S2); controller never closed-loop; the proxy failed, so we measure on engine outcomes (E1–E2, one sentence). Contribution list = K1–K4. | 1.3 |
| 2 Related Work | related_work (+ `related_work_table.tex`) | Gap defense (see §4). Compact positioning table `tab:rel:gap`. Physical-reasoning benchmarks and sparse-action formalisms. Evaluation precedents in one short paragraph. | 1.0 |
| 3 Problem Formulation | method | MDP at engine steps with sparse actions; prediction request r = (Δ, α); rollout as a semi-MDP over carrier frames; estimand taxonomy (τ_tr, τ_ex, τ_ad). Keep labels `sec:mth:setting`, `sec:mth:estimands`. | 0.8 |
| 4 Method | method | Architecture (K2) + figure (new architecture diagram; figure task), training objective, controller and DP distillation (K3), capacity-matched baseline. New label `sec:mth:model`. | 1.5 |
| 5 Experiments | experiments + synthesis | 5.1 protocol and outcome channel (pre-registration, engine verdicts, proxy invalidation E1; label `sec:mth:protocol` moves here, `sec:exp:validity` kept). 5.2 granularity matters (G1–G4; candidate 3×3 fixed-pair AUC heatmap, figure task). 5.3 does selection help (S1, S2; τ_ad). 5.4 engine-truth evaluation of fixed-granularity rankers (E2, one table; label `sec:syn:picture` kept). | 3.0 |
| 6 Discussion + 7 Conclusion | discussion | Limits (frozen parser, not full JEPA; controller never closed-loop; 15 members; saturated-speed engine). What would change the verdict. The measurement lesson in one sentence. | 0.6 |

Moves out of the body: `fig:syn:bands`, `fig:syn:qualitative`, `tab:syn:denominator`, the old §5 heuristic anatomy (keep only the E1 disclosure), most launch-power control detail. The appendix keeps everything; no evidence is deleted.

## 4. Related-work gap defense (§2 must carry this)

Organize by *what each line varies*. End every paragraph with the precise difference, stated as a capability or constraint, never "first" or "better". The claim is always "to our knowledge", and it must assert **joint AND per-decision AND one shared latent** together.

1. **Fixed prediction contract** (field default): Dreamer, TD-MPC, I-JEPA/V-JEPA, V-JEPA 2-AC, DINO-WM, LeWorldModel.
2. **Temporal axis only**: VLWM, SUNTA, TAWM, THICK, Adaptive Skip Intervals, Time-Agnostic Prediction, KeyIn, VTA, TD-VAE, Hafez 2020 rollout depth.
3. **Description axis only**: Wang et al. RSS 2023 (per-MPC-step resolution; horizon fixed), VisualPredicator, Scarafoni 2022.
4. **Both axes, fixed coupling**: LeCun H-JEPA (quote: predictions "at different time scales and different levels of abstraction"; JEPA-1 short-term low-level, JEPA-2 longer-term higher-level), Clockwork VAE, Shaj 2023, MSPred, Brüdigam 2021 (MPC), Director, Hieros.
5. **Both axes, learned jointly but fixed at use**: SALT (Huang & Freed 2026; joint temporal and state abstractions learned offline, used for skill planning; AntMaze).
6. **Per-state selection over something else**: MuSix (meta-router over a mixture of world models for adaptation), AdaJEPA (test-time weight adaptation).
7. **Setting**: PHYRE (contextual-bandit framing), I-PHYRE (intervention timing), Physion/CLEVRER (passive prediction), AIBIRDS (multi-shot Angry Birds competition), options/SMDP, action persistence, delayed MDPs. NovPhy (Pinto 2024) has no world-model users.

Closest three, to defend explicitly: SALT, MuSix, Wang 2023. Pre-empt H-JEPA by name.

## 5. Binding wording rules (all sections)

- Stage names, not ticket IDs, in main-text prose. Ticket IDs live only in comments, the registry and the appendix audit.
- "JEPA-style latent predictor over a frozen object-slot encoder". BG-NS-JEPA is named once, as the in-progress program.
- Never write that selecting granularity helps, that the two axes must be chosen together (untested coupling), or that fixed granularity fails and therefore adaptive works.
- Never pool zero-shot and few-shot. Intervals are descriptive; no significance language. The controller never ran in closed loop.
- G1 is a post-hoc reading of pre-declared descriptive statistics on ordering AUC; always say so. It measures ordering, not selection.
- "In point estimate" and the 0/72 support clause stay attached wherever those numbers appear.

## 6. Labels

Keep: `sec:introduction`, `sec:rel`, `sec:mth:setting`, `sec:mth:estimands`, `sec:mth:protocol`, `sec:exp:anatomy`, `sec:exp:validity`, `sec:syn:picture`, `sec:disc`, `sec:disc:conclusion`, `fig:teaser`, `fig:selection-ordering`, `tab:syn:oracle`, `tab:syn:crosspool` (the appendix references them).
New: `tab:rel:gap` (§2 positioning table), `sec:mth:model` (§4), `sec:exp:granularity` (§5.2), `sec:exp:selection` (§5.3), `fig:mth:arch`, `fig:exp:pairgrid` (planned).

## 7. Work order

1. §1 + §2 (paper-writer ×2) + `tab:rel:gap` (task agent) + new bib entries appended at the end of `iclr2026_conference.bib` under a separator comment for human validation. ← current
2. §3 + §4 method rewrite (paper-writer) + architecture figure (task agent).
3. §5 restructure (paper-writer) + table/figure moves and the pair-grid figure (task agent).
4. §6/§7, abstract, Fig. 1 redesign.
5. Docs sync (`CONTEXT.md`, `research_evidence.md`), appendix fixes (SALT footnote and matrix row; App. I runner list; registry range C1–C31 in body), page-limit check.
