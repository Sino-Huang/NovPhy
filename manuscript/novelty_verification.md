# Novelty Verification Note (2026-09-26)

Scope: verification of the three novelty claims in the ICLR 2026 manuscript
(`iclr2026/`): **(A)** the *action-sparse persistent-effect* problem modeling,
**(B)** NovPhy as an under-used environment, **(C)** prediction granularity
(horizon Δ ∧ description level α) as a jointly selected per-decision variable
over one shared latent carrier.

Method: Semantic Scholar API (bundled CLI), OpenAlex API, and primary pages
fetched directly (arXiv abstract/full-text pages, ar5iv full text, PMLR/ICML
virtual pages, publisher DOI landings, an Internet Archive OCR mirror of
LeCun 2022). Every quoted sentence below was read on the linked fetched page.
Existing context reused: `related_work_citation_map.md`, appendix §F of
`iclr2026_conference_appendix.tex` (lines ~1266–1537), and
`iclr2026_conference.bib` (42 entries). Per assignment, **no .tex or .bib
file was edited**; suggested bib additions are listed at the end.

---

## 0. Verdict summary

| Claim | Verdict | One-line rationale |
|---|---|---|
| A — problem modeling (action-sparse persistent-effect envs, MDP + per-decision prediction request) | **HOLDS WITH QUALIFIER** | The *environment class* is not new — PHYRE is single-action and its authors frame it as a contextual bandit; Angry Birds is multi-shot; options/SMDP, action persistence, and delayed MDPs formalize extended commitments. What is new and defensible: keeping the MDP at fixed engine-step granularity while making the **world model's prediction itself the per-step decision** (a per-decision (Δ, α) request against an engine-measured per-state ceiling), under novelty adaptation. No prior formalization of that framing was found. |
| B — NovPhy is under-used | **HOLDS** | AIJ version: 3 citations (S2), 2 (OpenAlex); arXiv version: 2 (S2). Among all citing works, one visual-world-model benchmark paper (Kim et al. 2023) cites NovPhy as *related work only*; the rest are open-world-AI surveys/position papers and the ANU group's own follow-up. No world-model/JEPA/MBRL paper found that uses NovPhy, Science Birds, or Angry-Birds-style physics as a learned-world-model testbed. |
| C — no prior world model treats granularity (Δ ∧ α) as a jointly selected per-decision variable over one shared latent | **HOLDS WITH QUALIFIER** | Verified across 25+ threats: prior work is temporal-only (VLWM, SUNTA, TAWM, THICK, ASI, TAP, KeyIn, VTA), description-only (Wang RSS 2023, VisualPredicator, Scarafoni), fixed-coupled (LeCun H-JEPA, CW-VAE, Director, Shaj, MSPred, Brüdigam, Hieros), adaptation-of-weights (AdaJEPA), or joint-but-fixed (SALT). The nearest per-decision selectors are single-axis (Wang) or over a model mixture for adaptation (MuSix). The joint-per-decision-request cell is empty in everything fetched — but the claim must stay scoped ("no prior work we could find", abstraction-as-trained-repertoire vs. request). |

---

## 1. Claim A — Problem modeling

### 1.1 How prior work actually frames one-action / long-cascade settings

| Setting | Paper (fetched source) | Their own framing (verbatim) |
|---|---|---|
| Single-action physics puzzle | PHYRE, Bakhtin et al., NeurIPS 2019 — [ar5iv full text](https://ar5iv.labs.arxiv.org/html/1908.05656) (abs: [arXiv:1908.05656](https://arxiv.org/abs/1908.05656)) | "Because the agent can only perform a single action to solve a PHYRE task, PHYRE is similar to a *contextual bandit* setting"; a tier is "a predefined set of all actions the agent is allowed perform and … a set of tasks that can be solved by at least one action from this action set." |
| Interactive physics, in-situ interventions | I-PHYRE, Li et al., ICLR 2024 — [arXiv:2312.03009](https://arxiv.org/abs/2312.03009) | "challenges agents to simultaneously exhibit intuitive physical reasoning, multi-step planning, and in-situ intervention … in-situ implies the necessity for timely object manipulation within a scene, where minor timing deviations can result in task failure." Decisions are *interventions*, not prediction requests; no learned-world-model readout contract. |
| Passive physical prediction | Physion, Bear et al., NeurIPS 2021 D&B — [arXiv:2106.08261](https://arxiv.org/abs/2106.08261) | Benchmark for "the ability to predict how physical scenarios will evolve over time" — readout prediction, **no agent action at all**. |
| Passive causal video QA | CLEVRER, Yi et al., ICLR 2020 — [arXiv:1910.01442](https://arxiv.org/abs/1910.01442) | Descriptive / explanatory / predictive / counterfactual questions about collision videos — no actions. |
| Multi-shot physics game, action ranking | AIBIRDS, Renz, AAAI 2015 — [DOI:10.1609/aaai.v29i1.9347](https://doi.org/10.1609/aaai.v29i1.9347); *Angry Birds as a Challenge for AI*, Renz et al., AAAI 2016 — [DOI:10.1609/aaai.v30i1.9875](https://doi.org/10.1609/aaai.v30i1.9875) | "Successful agents should be able to quickly analyze new levels and to predict physical consequences of possible actions in order to select actions" — several shots per level; simulation/heuristic agents; never formalized as an MDP whose per-step request is a model readout. |
| Temporally extended control | Options, Sutton, Precup & Singh, AIJ 1999 — [DOI:10.1016/S0004-3702(99)00052-1](https://doi.org/10.1016/S0004-3702(99)00052-1) (CrossRef landing fetched; no abstract on page) | Options formalize temporally extended *actions* over an SMDP; the cascade between decisions resembles an option's autonomous inner dynamics, but an option is a **control** construct with termination — not a prediction request with a description level. |
| Action persistence | Metelli et al., ICML 2020, PMLR 119 — [arXiv:2002.06836](https://arxiv.org/abs/2002.06836), [PMLR page](http://proceedings.mlr.press/v119/metelli20a.html) | "action persistence that consists in the repetition of an action for a fixed number of decision steps, having the effect of modifying the control frequency" — about the **agent's** control rate, not the environment's autonomous cascade. |
| MDPs with delays | Katsikopoulos & Engelbrecht, IEEE TAC 2003 — [DOI:10.1109/TAC.2003.809799](https://doi.org/10.1109/TAC.2003.809799) (S2 record fetched: 174 citations; TLDR: "reducing an MDP with delays to an MDP without delays, which differs only in the size of the state space") | Delayed *observation/action effect*; orthogonal to a cascade that is fully observable in principle and carries no pending action. |
| Delayed feedback | Walsh, Nouri, Li & Littman, JAAMAS 2009 — [DOI:10.1007/s10458-008-9056-7](https://doi.org/10.1007/s10458-008-9056-7) (CrossRef landing fetched; 99 citations per S2) | Same family: reward/feedback arrives late; not per-step prediction requests. |
| Jumpy / event-driven prediction | Time-Agnostic Prediction, Jayaraman et al., ICLR 2019 — [arXiv:1808.07784](https://arxiv.org/abs/1808.07784): "we decouple visual prediction from a rigid notion of time … predictable 'bottleneck' frames"; ASI, Neitz et al., NeurIPS 2018 — [arXiv:1808.04768](https://arxiv.org/abs/1808.04768): "a sampling rate which it can choose itself"; VTA, Kim et al., NeurIPS 2019 — [arXiv:1910.00775](https://arxiv.org/abs/1910.00775): "jumpy imagination"; TD-VAE, Gregor et al., 2018 — [arXiv:1806.03107](https://arxiv.org/abs/1806.03107) | Prediction-side temporal relaxation only; none ties the jump structure to a sparse *action* triggering an autonomous *physical* cascade, and none adds a description-level choice. |

### 1.2 What is genuinely new vs. the nearest formalisms

New (defensible):
1. The environment is an ordinary MDP at fixed engine-step granularity (continuous physics, novelty adaptation per NovPhy), but agent *decisions* are sparse; between them the physics runs autonomously. Prior benchmarks either collapse this to a contextual bandit (PHYRE), an intervention-timing game (I-PHYRE), or passive prediction (Physion/CLEVRER).
2. At every step of the cascade the world model faces an explicit **prediction request** — how far ahead (Δ) and at which description level (α) — decoupled from action selection. No fetched formalism makes the model's readout itself the per-step decision variable; the nearest are SMDP/options (control commitments), action persistence (control frequency), and delayed MDPs (information timing).
3. The request is scored against an engine-measured per-state ceiling over a candidate inventory (the paper's measurement contribution), which no surveyed benchmark or formalism provides.

Qualifier (must appear in any novelty sentence): the environment *class* — one action, long autonomous cascade — exists in PHYRE (single action, contextual-bandit framing) and, multi-shot, in Angry Birds; and the *control-side* formalisms for extended commitments are decades old. The claim must be about the **modeling of the prediction request per MDP step**, not about inventing sparse-action environments.

### 1.3 Exact defensible wording (Claim A)

> "In an *action-sparse persistent-effect environment*, a single intervention triggers a physical cascade that continues autonomously for many fixed simulation steps. PHYRE, whose tasks are solved by a single action, is framed by its authors as a contextual bandit (Bakhtin et al., 2019), and reinforcement learning formalizes extended commitments through options, action persistence, and delayed feedback (Sutton et al., 1999; Metelli et al., 2020; Katsikopoulos & Engelbrecht, 2003; Walsh et al., 2009). None of these makes the world model's prediction itself the per-step decision: we keep the MDP at fixed engine-step granularity and treat every step of the cascade as carrying a prediction request — how far ahead, and at what level of description, the model should predict."

Do **not** write: "environments where one action causes a long cascade have not been studied" (contradicted by PHYRE's own framing quote above).

---

## 2. Claim B — NovPhy is under-used

### 2.1 Citation counts (retrieved 2026-09-26)

| Source (fetched) | Record | Count |
|---|---|---|
| Semantic Scholar | [AIJ version, Pinto et al. 2024](https://www.semanticscholar.org/paper/5f308ebc707d92936cd429da14013568e1ffaf52) | **3** |
| Semantic Scholar | [arXiv:2303.01711 version](https://www.semanticscholar.org/paper/82a0fdc32b3b2b7c8a00900716e42debb4682969) | **2** |
| OpenAlex | [W4401241490](https://api.openalex.org/works/doi:10.1016/j.artint.2024.104198) (`cited_by_count`) | **2** |
| Semantic Scholar | [IJCAI 2025 abstract reprint](https://www.semanticscholar.org/paper/c7e4068975488adfff1df5fb2f80eb743e9a59a1) | 0 |
| Google Scholar | not retrievable programmatically (no API; search snippets did not surface a count) — do not quote a GS number without a manual check | — |

### 2.2 Who actually cites NovPhy (both versions, fetched via S2 `citations`)

- Ke et al. 2025, *Explain Before You Answer: A Survey on Compositional Visual Reasoning* ([arXiv:2508.17298](https://www.semanticscholar.org/paper/226aae3f8c246dc3df69eb856834889aad33cde8)) — survey.
- Holder et al. 2025, *Introduction to Open-World AI*, AIJ ([DOI:10.1016/j.artint.2025.104393](https://www.semanticscholar.org/paper/4c291f4babace9e871cf6fc4330c894b100ea47f)) — introduction/position.
- Cruz et al. 2025, *Open issues in open world learning*, AI Magazine ([DOI:10.1002/aaai.70001](https://www.semanticscholar.org/paper/100791293e491b3f0394befdc2f11a0a497b904c)) — position.
- Kim, Singh, Park, Gulcehre & Ahn 2023, *Imagine the Unseen World: A Benchmark for Systematic Generalization in Visual World Models* ([arXiv:2311.09064](https://arxiv.org/abs/2311.09064)) — cites the arXiv NovPhy as related OOD-benchmark work only; fetched full text: "Although OGRE and NovPhy provide a platform to study out-of-distribution objects, these lack well-defined primitives and systematic evaluation unlike ours." **Not used as a testbed.**
- Gamage et al., AIIDE 2023, *Physics-Based Task Generation through Causal Sequence of Physical Interactions* ([arXiv:2308.02835](https://www.semanticscholar.org/paper/098b6804dc7ded4f336420d274e0a8d9215131ca)) — the ANU group's own follow-up (task generation).

### 2.3 World-model / JEPA / MBRL usage of NovPhy, Science Birds, Angry-Birds physics

Searches (S2 keyword + web) found **no** world-model, JEPA, or model-based-RL paper that uses any of these as a learned-world-model testbed:
- Closest non-usages: JEPA-for-RL studies use standard control suites, not Angry-Birds physics ([arXiv:2504.16591](https://arxiv.org/html/2504.16591v1); [Value-guided action planning with JEPA world models, arXiv:2601.00844](https://arxiv.org/abs/2601.00844)); Science Birds appears in an MLLM stability-reasoning study (IEEE CoG 2025, "Using Science Birds, an Angry Birds-like physicsbased platform" — [S2 record](https://www.semanticscholar.org/paper/96428c28c123e54dcb1c02a17700eb1bef7b8503)) — an LLM evaluation, not a learned world-model testbed; the lineage *Science Birds Novelty* (Xue et al., AAAI-22 Spring Symposium, 7 citations) is the same group's predecessor testbed ([S2 record](https://www.semanticscholar.org/paper/ec4b9e1df64f97342a5832d4bc4af6296528fb4a)).

### 2.4 Verdict + exact wording (Claim B)

**HOLDS** (as of 2026-09-26). Defensible sentence:

> "NovPhy (Pinto et al., Artificial Intelligence 336:104198, 2024) remains marginal in the world-model literature: its journal version has three citing works on Semantic Scholar and two on OpenAlex, none of which uses it — or any Angry-Birds-style physics environment — as a learned world-model testbed; the only world-model paper that cites it (Kim et al., 2023) does so as related work."

Caveat: "under-used" should be time-indexed (published mid-2024) and count-indexed (small absolute numbers make the negative claim cheap to verify but also easy to flip).

---

## 3. Claim C — Granularity as a jointly selected per-decision variable

### 3.1 Threat table

Legend — *Varies*: T = temporal axis, D = description/representation axis, J = joint, F = fixed coupling of both, W = model weights (not granularity). *Per-dec.* = state-dependent selection at decision time. *Bib* = already in `iclr2026_conference.bib` (key given).

| # | Paper | Venue/Year | Varies | Per-dec.? | Bib | One-line defense |
|---|---|---|---|---|---|---|
| 1 | LeCun, *A Path Towards Autonomous Machine Intelligence* (H-JEPA) | OpenReview 2022 (v0.9.2) | F | No | no | Hierarchy level *binds* time span to description ("JEPA-1 performs short-term predictions using low-level representations. The second-level network JEPA-2 performs longer-term predictions using higher-level representations"); per-decision choice of both is left as an open issue, not a mechanism. |
| 2 | SALT (Huang & Freed) | ICML 2026 workshop | J (trained) | No | `huang2026salt` | "jointly learns temporal and state abstractions from offline trajectories" as a fixed trained repertoire used to plan over skills; no state-dependent (Δ, α) request and no physical-cascade evaluation (AntMaze-only prelim results). |
| 3 | THICK (Gumbsch et al.) | ICLR 2024 | T (+D at the top level) | No | `gumbsch2024thick` | Sparse latent changes form "invariant contexts"; the higher level "exclusively predicts situations involving context changes" — boundaries emerge from training under a fixed two-level architecture; no explicit request. |
| 4 | TAWM (Nhu et al.) | ICML 2025 workshop (PMLR) / arXiv | T | No | `nhu2025tawm` | Conditions on Δt and trains across Δt values; at inference the grain is "imposed by the environment's observation rate", not chosen per decision; single description. |
| 5 | VLWM (Du et al.) | arXiv 2026 (2606.21775) | T | No | `du2026vlwm` | Predicts "future latent states conditioned on action sequences of variable lengths" with a horizon-expanding curriculum — horizon follows the plan's action-sequence length; no description mode. |
| 6 | SUNTA (Iiyama et al.) | arXiv 2026 (2607.02087) | T | No | `iiyama2026sunta` | Surprise-driven chunk boundaries set by an internal metric inside imagined rollouts; chunking is emergent, and there is no description-axis choice. |
| 7 | AdaJEPA (Wang, Bounou, LeCun, Ren) | arXiv 2026 (2606.32026) | W | No | `wang2026adajepa` | Adapts the model's *weights* at test time inside MPC — the granularity contract itself is never selected. |
| 8 | MuSix (Jang et al.) | arXiv 2026 (2607.00457) | J over a mixture | **Yes** (scale by experiential distance) | `jang2026musix` | "a meta-router first maps this quantity to a weight over continuous scale space, then per-scale base routers select world models" — state-dependent *scale* selection over a mixture of specialized models for knowledge adaptation, not a joint (Δ, α) request to one shared carrier. Closest per-decision threat. |
| 9 | Clockwork VAE (Saxena, Ba, Hafner) | NeurIPS 2021 | F | No | no | "a hierarchy of latent sequences, where higher levels tick at slower intervals" — fixed architectural rates; no selection. |
| 10 | Adaptive Skip Intervals (Neitz et al.) | NeurIPS 2018 | T | Weakly (learned skip policy) | no | The model "make[s] predictions at a sampling rate which it can choose itself" — temporal axis only; no description level; pixel prediction, not a latent-carrier request. |
| 11 | Time-Agnostic Prediction (Jayaraman et al.) | ICLR 2019 | T | No | no | Decouples prediction "from a rigid notion of time" to hit predictable bottlenecks — temporal only. |
| 12 | KeyIn / *Keyframing the Future* (Pertsch et al.) | L4DC 2020 (PMLR 120:1–11); arXiv 2019 | T | No | no | Discovers keyframes then inpaints between them — event-time flexibility only. |
| 13 | Variational Temporal Abstraction (Kim, Ahn, Bengio) | NeurIPS 2019 | T | No | no | "infer the latent temporal structure … perform the stochastic state transition hierarchically" (enables "jumpy imagination") — inferred temporal hierarchy, no description choice. |
| 14 | Director (Hafner et al., *Deep Hierarchical Planning from Pixels*) | NeurIPS 2022 | F (control) | No | no | High-level policy sets subgoals at a coarser time scale inside the same latent model — a control hierarchy with fixed scales, not prediction-granularity selection. |
| 15 | Multi Time Scale World Models (Shaj et al.) | NeurIPS 2023 | F | No | `shaj2023multi` | Fixed multi-timescale SSM hierarchy; scales are architecture, not decisions. |
| 16 | MSPred (Villar-Corrales et al.) | BMVC 2022 | F (all scales at once) | No | `villarcorrales2022mspred` | "simultaneously forecast future possible outcomes of different levels of granularity at different spatio-temporal scales" — every prediction carries all granularities; nothing is selected. |
| 17 | Brüdigam et al. | ACC 2021 | F | No | `brudigam2021mpc` | "A detailed model is used for the short-term prediction horizon and a simplified model with an increased sampling time … for the long-term horizon" — the coupling is fixed by the control design (analytic MPC, not a learned world model). |
| 18 | Wang et al., dynamic-resolution model learning | RSS 2023 | D | **Yes** (per MPC step) | `wang2023dynamic` | "the agent can adaptively determine the optimal resolution at each model-predictive control (MPC) step" — state-dependent selection on the *description* axis only; horizon stays the standard fixed MPC rollout. |
| 19 | VisualPredicator (Liang et al.) | ICLR 2025 | D | Partial (online predicate invention) | `liang2025visualpredicator` | Invents predicates and plans over abstract states — what the model *expresses* adapts; prediction length is not a co-selected variable. (Full-text verification recorded in the repo's #91 citation pass; OpenReview currently behind a bot-wall.) |
| 20 | Scarafoni et al., *Finding Islands of Predictability* | arXiv 2022 (2210.07354) | D | Partial (per frame) | `scarafoni2022islands` | Per-frame abstraction of action forecasts; temporal extent fixed. |
| 21 | Hafez et al. | Robotics and Autonomous Systems 2020 | T (rollout depth) | Partial (reliability-driven) | `hafez2020improving` | Meta-control adapts rollout depth by reliability in a dual-system robot — depth only, no description mode. |
| 22 | V-JEPA 2 / V-JEPA 2-AC (Assran et al.) | arXiv 2025 (2506.09985) | — | No | `assran2025vjepa2` | Action-free pretraining; the action-conditioned variant plans with fixed MPC horizons toward image goals; no granularity variable exists in the interface. |
| 23 | DINO-WM (Zhou et al.) | arXiv 2024 (2411.04983) | — | No | `zhou2025dinowm` | One-step prediction of DINOv2 patch features; planners optimize action sequences at a fixed description and horizon. |
| 24 | DreamerV1/V3, TD-MPC | ICLR 2020 / ICML 2022 / Nature 2025 | — | No | `hafner2020dreamer`, `hafner2025dreamerv3`, `hansen2022tdmpc` | Fixed planning horizons and targets — the fixed-granularity default the manuscript's appendix §F already positions against. |
| 25 | TD-VAE (Gregor et al.) | 2018 | T | No | no | "go beyond simple step-by-step simulation, and exhibit temporal abstraction … rolled out directly without single-step transitions" — temporal jumpy rollouts only. |
| 26 | Schiewer et al., *Exploring the limits of hierarchical world models* | Scientific Reports 2024 | F ("static and environment agnostic temporal abstraction") | No | no | Hierarchy exists at several temporal abstractions but the abstraction is static by design. |
| 27 | Rao et al., Active Predictive Coding | Neural Computation 2024 | F | No | no | Hierarchical world models at "multiple abstraction levels" for planning; levels are architecture, not requests. |

Evidence URLs for rows 1–27 are in §3.4.

### 3.2 Top-3 closest threats, ranked, with defenses

1. **SALT (Huang & Freed, ICML 2026 workshop)** — the only title-level *joint* state+temporal abstraction world model. Its public abstract (now on the ICML virtual site — the appendix's "no public text" footnote is outdated and should be revised to cite this abstract) says it "jointly learns temporal and state abstractions from offline trajectories to support planning in a compact latent space." Defense: the two abstractions are *learned once, jointly, as a fixed repertoire* (skills + abstract latent states) and then used for planning; there is no state-dependent choice of a horizon-description pair at decision time, no single shared carrier serving different readouts, and the evaluation is offline AntMaze, not physical cascades. Suggested related-work sentence: "SALT learns temporal and state abstractions jointly, but as a fixed repertoire used to plan over skills, rather than as a request the model answers per state (Huang & Freed, 2026)."
2. **MuSix (Jang et al., arXiv 2026)** — the only fetched system that selects *per state* over an abstraction ("scale") space, via "a meta-router [that] maps this quantity to a weight over continuous scale space." Defense: MuSix routes among a *mixture of world models* to keep knowledge fresh as environments evolve; the selected variable is which model/scale to consult and adapt, not how far ahead and at which description level one shared latent should predict — and it is not evaluated on action-sparse physical cascades. Suggested sentence: "Mixture-based agents select a scale or expert per state (MuSix), but the variable being chosen is which specialized model to consult or update, not a joint prediction request to one shared carrier (Jang et al., 2026)."
3. **Wang et al., RSS 2023 (dynamic-resolution model learning)** — the nearest true per-decision granularity selector: "the agent can adaptively determine the optimal resolution at each model-predictive control (MPC) step." Defense: it varies only the *description* axis (particle resolution of a GNN dynamics model); the horizon remains the standard fixed MPC rollout, so the joint (Δ, α) request is still absent. Suggested sentence: "State-dependent resolution selection exists on the description axis alone (Wang et al., 2023), with the prediction schedule left fixed."

Honorable mentions to acknowledge preemptively: LeCun's H-JEPA (fixed coupling, per-decision choice named but not mechanized), Brüdigam et al. ACC 2021 (fixed coupling inside MPC), VLWM (temporal axis, plan-driven), SUNTA (surprise-set temporal chunks), THICK (emergent temporal hierarchy).

### 3.3 Suggested related-work defense paragraph (plain English, no overclaiming)

> Prior world models almost always fix their prediction contract to suit one objective: latent-dynamics methods predict one step at a fixed horizon (Dreamer, TD-MPC, DINO-WM, V-JEPA 2-AC), and joint-embedding methods fix both the target representation and the horizon at training time. Work that relaxes this does so on one axis at a time. On the temporal axis, VLWM varies the length of the predicted action sequence, SUNTA sets chunk boundaries from surprise, TAWM conditions on the environment's time step, and THICK learns when its latent state changes. On the description axis, Wang et al. (2023) select scene resolution per planning step and VisualPredicator invents predicates online. Fixed couplings of the two axes also exist, from LeCun's hierarchical JEPA — where higher levels predict longer spans in more abstract terms by construction — to Brüdigam's two-segment MPC horizon and MSPred's simultaneous multi-scale outputs. The nearest joint effort, SALT, learns temporal and state abstractions together, but as a fixed repertoire for planning over skills rather than as a decision made per state. To our knowledge, no prior world model poses prediction granularity — a horizon and a description level jointly — as a state-dependent request answered by one shared latent carrier, which is the variable this paper studies.

### 3.4 Verdict + exact novelty sentences (Claim C)

**HOLDS WITH QUALIFIER.** The joint-per-decision cell is empty in everything fetched, but (a) SALT is genuinely joint (at training time), (b) MuSix and Wang RSS 2023 are genuinely per-decision (on other variables), so "jointly" and "per-decision" must be claimed together and explicitly, and (c) the claim must be evidential ("no prior work we could find"), not categorical.

Recommended introduction sentences (consistent with the manuscript's claim-status vocabulary; "BG-NS-JEPA is in progress" boundaries preserved):

1. *"Existing world models vary prediction granularity on one axis at a time or fix the coupling between axes: hierarchical JEPAs bind description level to time span by construction (LeCun, 2022), temporal adapters change only how far or how coarsely the model predicts (Du et al., 2026; Iiyama et al., 2026; Nhu et al., 2025; Gumbsch et al., 2024), description adapters change only what the model expresses (Wang et al., 2023; Liang et al., 2025), and SALT learns temporal and state abstractions jointly but as a fixed trained repertoire for planning over skills (Huang & Freed, 2026). To our knowledge, no prior world model poses a joint horizon-and-description request (Δ, α) to one shared latent carrier as a per-decision variable."*
2. *"We therefore specify granularity as a per-decision prediction request: for each state, a request names a horizon Δ ∈ {1, 5, 15} and a description mode α ∈ {continuous, micro, macro} scored on one continuous latent carrier — a specification this paper measures the prerequisites of, not a controller benefit it demonstrates."* (matches the contribution item "A specification of granularity as a per-decision prediction request")
3. For the setting sentence (combining Claims A+B): *"NovPhy's slingshot scenarios make the setting concrete — one launch, then a collision cascade that resolves up to hundreds of engine steps later — and, to the best of our knowledge, no published world-model study has used NovPhy, or any Angry-Birds-style physics environment, as a learned-world-model testbed."*

---

## 4. Fetched primary sources (evidence index)

Claim A: PHYRE full text https://ar5iv.labs.arxiv.org/html/1908.05656 and https://arxiv.org/abs/1908.05656 · I-PHYRE https://arxiv.org/abs/2312.03009 · Physion https://arxiv.org/abs/2106.08261 · CLEVRER https://arxiv.org/abs/1910.01442 · AIBIRDS https://doi.org/10.1609/aaai.v29i1.9347 and https://doi.org/10.1609/aaai.v30i1.9875 (S2 records, abstracts fetched) · options https://doi.org/10.1016/S0004-3702(99)00052-1 (CrossRef landing) · Metelli https://arxiv.org/abs/2002.06836 + http://proceedings.mlr.press/v119/metelli20a.html (venue verified) · Katsikopoulos & Engelbrecht https://doi.org/10.1109/TAC.2003.809799 (S2 record + TLDR) · Walsh et al. https://doi.org/10.1007/s10458-008-9056-7 (CrossRef landing) · TAP https://arxiv.org/abs/1808.07784 · ASI https://arxiv.org/abs/1808.04768 · VTA https://arxiv.org/abs/1910.00775 · TD-VAE https://arxiv.org/abs/1806.03107.

Claim B: S2 AIJ record https://www.semanticscholar.org/paper/5f308ebc707d92936cd429da14013568e1ffaf52 (count 3) · S2 arXiv record https://www.semanticscholar.org/paper/82a0fdc32b3b2b7c8a00900716e42debb4682969 (count 2, citations fetched) · OpenAlex https://api.openalex.org/works/doi:10.1016/j.artint.2024.104198 (cited_by_count 2) · Kim et al. full text https://ar5iv.labs.arxiv.org/html/2311.09064 · Gamage et al. https://arxiv.org/abs/2308.02835 · Science Birds Novelty https://www.semanticscholar.org/paper/ec4b9e1df64f97342a5832d4bc4af6296528fb4a · MLLM/Science Birds CoG 2025 https://www.semanticscholar.org/paper/96428c28c123e54dcb1c02a17700eb1bef7b8503.

Claim C: LeCun 2022 — OpenReview forum https://openreview.net/forum?id=BZ5a1r-kVsf (bot-walled); text quoted from the Internet Archive OCR mirror https://archive.org/download/a-path-towards-autonomous-machine-intelligence/ (§4.4 "the world model must be able to make predictions at different time scales and different levels of abstraction"; §4.6 "JEPA-1 performs short-term predictions using low-level representations. The second-level network JEPA-2 performs longer-term predictions using higher-level representations") · SALT https://icml.cc/virtual/2026/83751 (full workshop abstract fetched) + OpenReview pdf ElO5M4Ojev (listed) · THICK https://cgumbsch.github.io/blog/thick (full abstract) · TAWM https://arxiv.org/abs/2506.08441 · VLWM https://arxiv.org/abs/2606.21775 · SUNTA https://arxiv.org/abs/2607.02087 · AdaJEPA https://arxiv.org/abs/2606.32026 · MuSix https://arxiv.org/abs/2607.00457 · CW-VAE https://arxiv.org/abs/2102.09532 · Neitz https://arxiv.org/abs/1808.04768 · Jayaraman https://arxiv.org/abs/1808.07784 · Pertsch https://arxiv.org/abs/1904.05869 + https://proceedings.mlr.press/v120/pertsch20a.html · Director https://arxiv.org/abs/2206.04114 · MSPred https://arxiv.org/abs/2203.09303 · Brüdigam https://arxiv.org/abs/2108.08014 (DOI 10.23919/ACC50511.2021.9482617) · Wang RSS 2023 https://arxiv.org/abs/2306.16700 · V-JEPA 2 https://arxiv.org/abs/2506.09985 · DINO-WM https://arxiv.org/abs/2411.04983 · Hieros https://arxiv.org/abs/2310.05167 · Shaj thesis https://arxiv.org/abs/2404.16078 · Schiewer https://arxiv.org/abs/2406.00483 · Rao https://arxiv.org/abs/2210.13461.

## 5. Cross-check notes and follow-ups for the author

- **SALT status change (action recommended):** the appendix §F sentence "No public text is available beyond the workshop's poster listing, so the SALT row of Table tab:app:matrix is marked unverified" and the corresponding footnote are now outdated — a full abstract is on https://icml.cc/virtual/2026/83751. Method-level characterization can be cited; the protocol columns of the matrix should still stay "?" (abstract-only).
- **I-PHYRE OpenReview id:** `related_work_citation_map.md` lists `openreview.net/forum?id=1bbPQShCT2`; that id could not be verified (OpenReview bot-wall). arXiv:2312.03009 and DBLP `conf/iclr/Li00024` independently confirm ICLR 2024, Li/Wu/Zhang/Zhu. Check the forum id before relying on it.
- **Venue confirmations:** Metelli et al. = ICML 2020, PMLR v119 (verified on PMLR + icml.cc); Pertsch et al. = L4DC 2020, PMLR v120:1–11 (not CoRL); Director = NeurIPS 2022 ("Deep Hierarchical Planning from Pixels", arXiv:2206.04114); Hieros = ICML 2024 (arXiv:2310.05167).
- **arXiv-only (cite as preprints):** VLWM, SUNTA, AdaJEPA, MuSix (ECCV 2026 acceptance self-reported per citation map), Scarafoni, DINO-WM, V-JEPA 2.
- **Namesake warning:** "H-JEPA" also names an unrelated wireless-control paper (arXiv:2602.07000) — when citing "H-JEPA", cite LeCun (2022) explicitly.
- **Bib additions to support the suggested sentences** (not added, per task constraints): LeCun 2022 (OpenReview, `lecun2022path`); Metelli et al. ICML 2020 (PMLR 119, `metelli2020control`); Katsikopoulos & Engelbrecht IEEE TAC 2003 (`katsikopoulos2003mdp`); Walsh et al. JAAMAS 2009 (`walsh2009learning`); optionally Renz AAAI 2015 (AIBIRDS) if Claim A's Angry-Birds contrast enters the main text.

## 6. Unresolved

- Google Scholar count for NovPhy: not obtainable via available tools (no API; snippets did not surface one). If a GS number is wanted, it needs a manual browser check; S2/OpenAlex numbers above are the verified ones.
- Full-text quotes for two Claim A theory anchors are metadata-level only: Sutton/Precup/Singh 1999 (CrossRef landing exposes no abstract) and Walsh et al. 2009 (Springer landing exposes no abstract); Katsikopoulos & Engelbrecht is characterized via the Semantic Scholar TLDR. The claims attached to them (options = temporal abstraction framework; delays = transformed MDP) are title/venue-level and safe.
- VisualPredicator and Scarafoni rows rest on the bib entries plus the in-repo #91 full-text verification pass (recorded in `related_work_citation_map.md`), because OpenReview/arXiv fetches for those two were not re-fetched in this pass.
- No `[TODO: cite]` placeholders or dangling keys were resolved in this task: the assignment forbids .tex/.bib edits, and no citation-resolution pass was requested.
