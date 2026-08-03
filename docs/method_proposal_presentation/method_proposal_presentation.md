---
marp: true
paginate: true
math: mathjax
title: "BG-NS-JEPA: Method Proposal"
description: "Supervisor presentation on the proposed regime-adaptive world-model method"
---

<style>
section {
  font-size: 18px;
}
</style>

<!-- _class: lead -->

# BG-NS-JEPA

## A regime-adaptive world model for persistent physical cascades

**Method proposal for supervisor discussion**  
NovPhy research project | 31 July 2026

> **One-sentence method:** the learned analogue of a hybrid kinetic–fluid solver — the world model estimates the local degree of scale separation and jointly changes how far ahead it predicts and at which level of description, as the physical regime changes.

---

# A century-old precedent: physics already solved this trade-off

Kinetic theory connects microscopic particle mechanics to macroscopic fluid behavior:

- **No single description level is valid everywhere.** Collision-dominated regions need a fine *kinetic* description; scale-separated regions admit a coarse *hydrodynamic* one.
- **The switching criterion is state-dependent.** The local Knudsen number (interaction scale vs. observation scale) decides which description applies — and how fine the time resolution must be.
- **Computation operationalized this.** Hybrid kinetic–fluid solvers run the expensive solver only where needed; equation-free / heterogeneous multiscale methods define lifting and restriction operators between levels; moment closures formalize the intermediate level.

**Our observation:** a learned world model rolling out a physical cascade faces *the same* regime heterogeneity — yet every existing world model uses one fixed clock and one fixed representation.

*Scope: we import the algorithmic lesson, not the mathematics — no rigorous derivation is claimed.*

---

# The micro–meso–macro hierarchy, realized as a world model

| Level | Physics analogue | BG-NS-JEPA realization | Valid when |
|---|---|---|---|
| **Micro** | Kinetic / particle description | Continuous JEPA latent $z_t$ | Always available; required at collision onset |
| **Meso** | Moments of the distribution | Micro-relational predicates: `contact`, `supports`, `velocity-bin` | When scale separation is high enough for predicates to be stable |
| **Macro** | Hydrodynamic fields; equilibria | Macro-event predicates: `cascade-active`, `collapsed`, `steady-state` | Remains useful even while individual contacts change |

Design consequences inherited from the hierarchy:

- The symbolic-reliability score $r_\psi$ is the **learned local Knudsen number**.
- Cross-level maps $A, R$ are **restriction / lifting operators** (equation-free modeling).
- $\mathcal{L}_{\text{cross}}$ is a **closure-consistency** loss: macro must be derivable from micro.
- Both controller axes are governed by *one* scale-separation quantity → **the controller must be joint**.

---

# What exactly are we proposing?

At each rollout decision, the model selects a **joint pair**:

$$
(\Delta_k,\alpha_k)\sim\pi_\kappa(\cdot\mid h_{t_k}),
\qquad
\alpha_k\in\{\text{continuous},\text{micro},\text{macro}\},
\qquad
t_{k+1}=t_k+\Delta_k .
$$

*One rollout is a sequence of adaptive decisions $k=0,\dots,K$: at decision $k$ (frame time $t_k$) the controller looks at the current state and jumps $\Delta_k$ frames ahead.*

- $\Delta_k$: **how far forward** the model advances this step.
- $\alpha_k$: **which description level** the predictor uses this step.
- $h_{t_k}$: **controller state** — current latent $z_t$ plus predictive uncertainty, event likelihood, and local event density: diagnostics of one underlying scale-separation quantity.
- $r_\psi(h_t)\in[0,1]$: **learned reliability** of fine object relations — the symbolic face of the scale-separation estimate (low → mask micro-relational constraints; continuous dynamics dominate).
- A horizon- and mode-conditioned JEPA predictor remains the continuous dynamics backbone — and the **sole rollout state carrier**.

---

# Proposed architecture

![BG-NS-JEPA architecture: diagnostics condition a joint controller, which selects a temporal horizon and representation for the predictor; relational heads provide gated training constraints.](image_assets/bg_ns_jepa_architecture.png)

The controller changes the *prediction problem itself*, not only the rollout step size — it replaces the hand-crafted domain-decomposition criterion of a hybrid solver with a learned one.

---

# One adaptive rollout step

![width:800px Joint controller decision flow: estimate predictive signals, gate unreliable micro relations, score admissible horizon-representation pairs, predict, and repeat.](image_assets/joint_controller_decision.png)

Low relational reliability removes **micro-relational supervision** from consideration; it does not imply that all symbols, especially macro events, are meaningless — hydrodynamic-level descriptions remain valid where moment-level detail is not.

---

# The controller's action space

The controller scores the cross-product $\mathcal{D}\times\mathcal{A}$ rather than making two independent decisions.

| Temporal choice | Continuous latent | Micro-relational state | Macro-event state |
|---|---|---|---|
| **Short horizon** | Collision onset; active contacts | Local support/contact update when reliable | Imminent event transition |
| **Medium horizon** | Settling dynamics | Evolving object graph | Collapse/clearance progress |
| **Long horizon** | Expensive and drift-prone | Risk of skipping relation changes | Stable endpoint or quiescent interval |

Why joint selection matters:

- A short step does not automatically imply an object-level symbolic representation.
- A long step does not automatically imply continuous prediction.
- The best pair depends on uncertainty, event density, and symbolic reliability together.
- **The principled reason:** one scale-separation quantity governs both axes; factorizing the policy assumes an independence the physics does not support.

**Key ablation:** joint controller vs. factorized $\pi_\Delta\pi_\alpha$ at matched compute.

**Training:** $\pi_\kappa$ is distilled from per-state best-pair labels (exhaustive scoring on oracle trajectories). The deployed controller is a one-pass forecast — no ground truth is needed at decision time; the oracle comparison survives only as teacher and as ceiling.

---

# How the selected pair changes during one cascade

![Regime-adaptive cascade timeline: the controller moves from long-horizon macro prediction in stable phases to short-horizon continuous prediction during collision and collapse, then through micro-relational settling back to macro equilibrium.](image_assets/regime_adaptive_cascade_timeline.png)

The central proposal is the **coupled switch**: both $\Delta$ and $\alpha$ change at physical regime boundaries rather than following one fixed clock or representation. In solver terms: kinetic-level resolution during collisions, hydrodynamic-level stepping during free streaming and equilibrium.

---

# Three state representations

Initial operational definition: a **shared latent backbone** is conditioned on $\alpha$, while the selected mode determines the active prediction target and head.

| Mode | Hierarchy level | Selected target/head | Best suited to | Constraint policy |
|---|---|---|---|---|
| **Continuous** | kinetic | Future latent $z_{t+\Delta}$ | Collision and active collapse | No fine relational correction |
| **Micro** | moment | Object/predicate head: `contact`, `supports`, `velocity-bin` | Stable or slowly changing local structure | Reliability-gated |
| **Macro** | hydrodynamic | Event head: `cascade-active`, `collapsed`, `pigs-cleared` | Event transitions and endpoints | Supervised when selected |

Cross-level maps are learned restriction/lifting operators:

$$
\underbrace{S_t^M=A(S_t^\mu)}_{\text{restriction}},
\qquad
\underbrace{\widetilde S_t^\mu=R(S_t^M,z_t)}_{\text{lifting}} .
$$

**The carrier is always $z$.** Whatever $\alpha$ is selected, $F$ always also outputs the continuous carrier $\hat{z}_{t+\Delta}$; the symbolic head is a supervised readout that enters only the loss (or the planner interface). No discrete step ever sits in the rollout path — so a one-frame symbolic mislabel cannot propagate, and the next decision re-selects $\alpha$ freely from the new $h$. Whether shared heads, separate experts, or another interface works best is an **architectural ablation**, not a result assumed in advance.

---

# How the world model itself is trained

The dual-output design keeps training simple:

$$
F_\theta^{\Delta,\alpha}(z_t, a, S^\mu_t) \;\longrightarrow\;
\underbrace{\hat z_{t+\Delta}}_{\text{carrier — always}}
\;+\;
\underbrace{\text{mode-head readout}}_{\text{loss / monitor only}}
$$

- **Teacher forcing is a choice, not a workaround.** Base training is single-step supervision from true encoded states over the $(\Delta,\alpha)$ grid — the standard JEPA regime. No gradient ever crosses a symbolic representation, because symbols are never the state.
- **Exposure bias** is mitigated structurally (multi-horizon targets at $\Delta{=}15$, endpoint-level macro jumps, terminal anchor); if pilot drift demands it, add short-window autoregressive fine-tuning **through the $z$ carrier only**.
- **Only one place** ever needs gradients through a discrete choice: the optional end-to-end controller (Gumbel-softmax) — which is why it stays optional.

| Phase | Regime |
|---|---|
| **A** (Stage 1) | Teacher-forced multi-task pretraining of $F$ over the full grid |
| **B** (Stage 3) | Same objective; $(\Delta,\alpha)$ sampling follows the controller |
| **C** (optional) | Short-window AR fine-tuning through the carrier |

---

# Symbolic reliability is learned, not assumed

The test-time gate is a learned estimator — the world-model analogue of a local Knudsen number:

$$
r_\psi(h_t)\in[0,1].
$$

Simulator state provides an oracle training target and upper bound:

$$
\phi^*(x_t)=
\mathbb{1}[\mathrm{KE}(x_t)<\epsilon_{\mathrm{KE}}]
\mathbb{1}[\mathrm{activeContacts}(x_t)<\epsilon_{\mathrm{contact}}].
$$

Interpretation:

- KE and contact activity estimate the ratio of **interaction timescale to observation timescale** — the physical content of a Knudsen number.
- High $r_\psi$: fine contact/support predicates are stable enough to regularize a rollout.
- Low $r_\psi$: mask fine relational loss and let continuous dynamics dominate.
- Macro-event predicates can remain useful even when individual contacts are unstable.

**Calibration check:** compare learned-gate decisions with oracle regimes and report switch precision/recall, not only task accuracy.

---

# Where do the symbols come from? A supervised predicate parser

The method consumes **grounded predicate truth values with calibrated confidence** — a scene graph is one serialization; the functional module is a predicate parser.

| Tier | Input | Symbol source | Role |
|---|---|---|---|
| **0 — Oracle** | Engine state | Ground-truth predicates | Defines predicate semantics; Stage-1 upper bound; all training labels |
| **1 — Feature parser** | Simulator state | MLP heads on engine features | Sanity tier: the vocabulary is learnable before vision is involved |
| **2 — Visual parser** | RGB | Frozen encoder (V-JEPA/DINO-style) + DETR-style object decoder + pairwise predicate heads + GINE | Deployed extractor (Stage 2) |

- **Frozen after supervised training.** Predicate semantics are anchored to engine labels; the global loss may not reshape them — otherwise `supports(A,B)` could drift into whatever reduces task loss, invalidating violation metrics and gate calibration. End-to-end fine-tuning is a Stage-4 variant gated by a **predicate-drift acceptance metric**.
- **Noise is represented, not ignored.** Heads emit calibrated probabilities; constraints apply only to **high-confidence positives** (a false-positive constraint costs far more than a missing one); and $r_\psi$ is trained on the real extractor's confidence, so it learns the parser's error profile. Extractor noise and regime instability are distinct sources — measured separately, absorbed by the same gate.

---

# SPSG: the relational mechanism

![SPSG mechanism: a directed physical scene graph is encoded with GINE and supervised through semantic predicate heads, a TPR role-filler head, and physics-validated negatives under a reliability-gated symbolic loss.](image_assets/spsg_mechanism_pipeline.png)

**Three complementary signals:**

- **Semantic heads** align selected predicates with learned embeddings.
- **TPR structure** preserves role binding: “A supports B” differs from “B supports A”.
- **Validated negatives** contrast the scene with an explicitly verified physical violation.

SPSG remains a **soft relational regularizer**; it does not imply an automatically exact hard manifold.
The constraint *content* comes from SPSG, but *when and how strongly* it applies is decided by the learned gate:
$\lambda_{\text{sym}}\,\omega_\psi(h_t)\,\mathcal{L}_{\text{sym}}$ — high $r_\psi$ enforces the geometry, low $r_\psi$ suspends it while continuous dynamics dominate.

---

# Unified training objective

For controller-selected steps $k$:

$$
\mathcal L_{\mathrm{total}}=
\mathbb E\!\left[
\sum_k\frac{\Delta_k}{T}\left(
\mathcal L_{\mathrm{pred}}
+\lambda_{\mathrm{sym}}\,\omega_\psi\mathcal L_{\mathrm{sym}}
+\lambda_{\mathrm{cross}}\mathcal L_{\mathrm{closure}}
+\lambda_{\mathrm{cost}}c(\Delta_k,\alpha_k)
\right)
+\lambda_{\mathrm{anchor}}\mathcal L_{\mathrm{anchor}}
\right].
$$

| Term | Function |
|---|---|
| $\mathcal L_{\mathrm{pred}}$ | Match the target latent at the selected horizon |
| $\omega_\psi\mathcal L_{\mathrm{sym}}$ | Gate semantic, TPR, and contrastive relational losses |
| $\mathcal L_{\mathrm{closure}}$ | Closure consistency: macro description must be derivable from micro |
| $c(\Delta,\alpha)$ | Prevent a degenerate always-fine policy |
| $\mathcal L_{\mathrm{anchor}}$ | Encourage an outcome-consistent terminal latent (equilibrium) |

The $\Delta_k/T$ weighting makes variable-horizon training and fixed-policy baselines comparable by represented duration.

---

# How $(\Delta,\alpha)$ trains the JEPA world model

Standard JEPA training fixes the prediction task by hand: context latent $z_t$ → predict a **stop-gradient target latent** at a **fixed temporal offset**, in a **fixed embedding space**.

BG-NS-JEPA turns both fixed choices into state-dependent decisions:

- $\Delta_k$ — **how far ahead the target is**: predict at $t_k+\Delta_k$
- $\alpha_k$ — **which space the target lives in**: continuous latent (the usual JEPA target encoder), micro scene-graph predicates, or macro event states

Each selected pair instantiates one JEPA sub-task; a single predictor conditioned on $(\Delta,\alpha)$ is trained **multi-task over the pair grid**, with the latent-prediction loss weighted by $\Delta_k/T$. $\pi_\kappa$ is thus a **learned task scheduler** — like making the masking/window strategy of masked representation learning adaptive per state, instead of a fixed hyperparameter. Multiscale reading: a learned coarse time-stepper that decides when fine (kinetic) integration is worth its cost.

**Chicken-and-egg:** scheduler labels depend on predictor performance, but the predictor's training distribution depends on scheduler choices. Staged resolution:

1. **Stage 1 (exhaustive scoring):** train candidate $(\Delta,\alpha)$ predictors on oracle trajectories; score every pair per state → per-state **best-pair labels**
2. **Stage 3 (amortized scheduler):** distill the per-state argmin into $\pi_\kappa$; ablate joint vs. factorized
3. **Stage 4 (optional end-to-end):** discrete relaxation (Gumbel-softmax), kept only if it improves the frontier

*Stage 2 (learned symbol extraction) runs orthogonally — it changes where scene graphs come from, not how the controller or predictor are trained.*

---


# What is fixed now, and what will experiments decide?

**Fixed scientific hypothesis**

> A state-dependent **joint** choice of $(\Delta,\alpha)$ can outperform fixed, one-axis, and independent choices at matched compute — because both axes are projections of one scale-separation quantity.

**Locked design decisions**

> $z$ is the sole rollout state carrier — symbolic heads are supervised readouts, never the state (dual-output $F$). $F$ trains teacher-forced, multi-task over the $(\Delta,\alpha)$ grid. The symbolic extractor is supervised, then frozen. The controller is distilled from exhaustive oracle scoring.

| Open method decision | Runnable starting point | How we resolve it |
|---|---|---|
| How is $\pi_\kappa$ trained? | Score every pair on oracle trajectories using duration-normalized prediction error + physical violations + compute; train one categorical controller on the argmin labels | Compare oracle-label supervision with an end-to-end discrete relaxation; retain the simpler method unless the latter improves the frontier |
| What does selecting $\alpha$ change? | Shared backbone with $(\Delta,\alpha)$-conditioned carrier computation + mode-specific readout heads; the carrier is always $z$ | Ablate shared heads vs. separate experts and test cross-mode rollout consistency; optionally a recurrent controller with $(\Delta_{k-1},\alpha_{k-1})$ in $h$ |
| Which horizons belong in $\mathcal D$? | Provisional $\mathcal D_0=\{1,5,15\}$ frames; rescale if pilot capture rate requires it | Select using dev-set prediction/compute trade-offs, then lock before test evaluation |
| Is Phase-C fine-tuning needed? | None initially | Short-window AR fine-tuning through the carrier, adopted only if measured pilot drift demands it |

These choices will be **resolved progressively by controlled experiments**. They are important research questions, but each needs an explicit initial implementation before it can be tested.

---

# What data does the method require?

| Payload | Purpose | Project status |
|---|---|---|
| RGB frames + shot action | Continuous JEPA training and temporal ablations | **Available in the read-only pipeline** |
| Object state and kinematics | Oracle graph nodes; stability supervision | **Requires physics instrumentation** |
| Contact/support relations | GINE edges; physical-violation metrics | **Requires physics instrumentation** |
| Macro event labels | Event head and closure consistency | **To derive from validated engine state** |
| Outcome labels | Terminal anchor and endpoint accuracy | **Collector/sidecar integration required** |

Current pipeline strengths already in place:

- immutable episode catalogs and provenance;
- shot-local temporal windows and deterministic sampling;
- resumable curricula;
- fixed-short, fixed-long, uniform, and curriculum temporal manifests;
- explicit sample-matched and compute-matched comparisons.

**Boundary:** reserved sidecar capability names are not evidence that symbolic/physics payloads are already readable.

---

# Staged implementation and de-risking

| Stage | What is learned? | Research question answered | Exit evidence |
|---|---|---|---|
| **1. Oracle symbols + fixed pairs** | Dual-output $F$ over the grid; mode heads, GINE, $G_\omega$, $A$, $R$; exhaustive pair scoring | Do perfect symbols help, and do preferred pairs vary by regime? | Oracle upper-bound gain + per-state best-pair labels |
| **2. Learned symbolic state** | Tier-2 predicate parser (frozen after training); $r_\psi$ on extractor confidence | Can the model recover useful relational structure? | Learned-vs-oracle gap, per-predicate F1, gate calibration, flip-rate coherence |
| **3. Controller + interface ablation** | $\pi_\kappa$ distilled from best-pair labels; alternative mode interfaces | Which training rule and abstraction interface work best? | Matched-compute Pareto frontier + selected design |
| **4. Full model** | End-to-end BG-NS-JEPA (+ optional variants, each with an acceptance criterion) | Does the complete method transfer? | NovPhy + Physhion/CLEVRER results |

The **minimum viable paper** requires Stage 1 and Stage 3: an oracle-symbol upper bound plus evidence that joint control beats fixed and independent choices. Learned perception remains a later extension if the upper bound is positive.

---

# The experiment that tests the central claim

Compare these policies at matched compute budget:

1. Fixed short + continuous.
2. Fixed long + macro.
3. Temporal-only adaptive with fixed representation.
4. Abstraction-only adaptive with fixed stride.
5. Factorized controller $\pi_\Delta\pi_\alpha$.
6. **Joint BG-NS-JEPA controller**.

Primary outcome is not a single accuracy number. It is the frontier over:

$$
\text{endpoint correctness}
\quad\times\quad
\text{physical plausibility}
\quad\times\quad
\text{compute}.
$$

The central claim is supported only if the joint controller dominates fixed and factorized choices at comparable compute and selects regime-aligned pairs.

---

# Measurements and falsification criteria

| Question | Measurements | Result that would weaken the proposal |
|---|---|---|
| Does prediction improve? | ADE@H, FDE@H, final-state accuracy, event F1 | No endpoint gain at matched local error |
| Is physics more plausible? | Penetration, unsupported floating, illegal contacts | Relational loss improves labels but not physical validity |
| Does adaptation matter? | Effective steps, controller cost, pair calibration | A fixed or factorized policy matches the joint policy |
| Is the gate necessary? | Oracle gate, learned gate, no gate | Always-on symbols perform equally well during active collapse |
| Does SPSG matter? | GINE/TPR vs. flat or no symbolic encoding | No gain over a simpler symbolic interface |
| Is the symbolic state coherent? | Predicate flip rate vs. engine flip rate | Learned symbols flicker far above engine rate in stable regimes |
| Is it robust to extractor noise? | Noise-injection degradation curve | Endpoint accuracy collapses at low injected noise |
| Does it help decisions? | Task success, shots-to-success, novelty adaptation | Better prediction does not improve planning |

This design separates failure of the **controller**, **symbol extraction**, and **relational inductive bias** rather than treating the method as one indivisible bundle.

---

# Optional mechanisms, not core claims

**Inference-time relational projection**

- When $r_\psi$ is high and a relational mode is selected, take 1–3 gradient steps to reduce the learned relational residual.
- Evaluate against a no-projection ablation.
- Describe it as approximate optimization, not exact projection onto a known manifold.

**Free-streaming + interaction decomposition**

- Optional predictor split $F_\theta = F_{\text{drift}} + \mathbf{1}[\text{event-active}]\,F_{\text{interact}}$, mirroring kinetic operator splitting.
- Architectural ablation only.

**PDDL serialization**

- Micro and macro states may be serialized for a downstream planner.
- PDDL is an interface, not the learned temporal abstraction mechanism.

**Terminal anchor**

- Pulls a reliable final latent toward an outcome-specific symbolic pole — an equilibrium target.
- Encourages endpoint consistency; it cannot guarantee a correct rollout.

Keeping these claims narrow makes the primary controller contribution easier to test and defend.

---

# Main risks and controlled responses

| Risk | Controlled response |
|---|---|
| Symbol extraction is unstable | Establish oracle-symbol upper bound first; Tier-1 feature parser as fallback |
| Symbolic noise corrupts rollouts | $z$ is the sole state carrier — symbols are readouts/constraints, never integrated forward; high-confidence-positive gating; degradation curve reported |
| Controller dismissed as a router / MoE | Forecast vs. measured error; joint prediction-task selection; sequential semi-MDP; modes compose — the oracle router survives only as label generator and ceiling |
| Reliability gate is miscalibrated | Report oracle gate, learned gate, and no-gate variants |
| Controller collapses to always-fine | Explicit cost term and matched-compute reporting |
| TPR is expensive | Efficient readout/projection approximation; ablate it |
| Vocabulary appears hand-engineered | Learn truth values/embeddings; test OOD transfer; keep vocabulary physically generic |
| Method appears to be a component bundle | The hierarchy *generates* the component list; the state-carrier principle fixes how they interact; make joint-vs-factorized control the primary experiment |
| Physics framing reads as name-dropping | Cite the algorithmic tradition only (hybrid solvers, equation-free, HMM, moment closure); every borrowed concept terminates in an ablatable design decision; no rigorous-derivation claims |
| Instrumentation delays the full model | Publishable minimum uses oracle state and locked temporal baselines |

---

<!-- _class: lead -->

# Proposed decision for tomorrow

Proceed with this hierarchy of claims:

1. **Primary:** joint temporal and representational control improves the prediction–plausibility–compute frontier, grounded in the micro–meso–macro description-level paradigm.
2. **Mechanism:** a learned scale-separation estimate (the Knudsen analogue) gates fine relational constraints without suppressing useful macro events.
3. **Secondary:** GINE + TPR provides structured relational regularization.

Immediate implementation gate:

> First complete physics instrumentation and the oracle-symbol experiment; continue to learned symbols only if structured supervision produces a measurable upper-bound gain.

Supervisor feedback requested on:

- Is the joint-controller claim sufficiently focused for the target venue?
- Does the multiscale/kinetic framing strengthen the novelty story without inviting "name-dropping" criticism?
- Is oracle-symbol Stage 1 an acceptable minimum publishable unit?
- Should planning results be required for the main paper or treated as downstream validation?

---

# Backup: expected questions

**Why not use only a stability gate?**  
A binary gate decides whether one loss is active. The proposed controller chooses among multiple horizons and description levels, including macro events when micro relations are unreliable.

**Isn't the physics framing name-dropping?**  
We borrow design decisions, not prestige: the scale-separation estimator replaces four ad-hoc diagnostics with one principled quantity; restriction/lifting and closure consistency come from equation-free modeling; the joint (non-factorized) controller follows from one quantity governing both axes. We cite the algorithmic tradition, not theorems, and claim no rigorous derivation.

**Why not predict every frame?**  
Dense rollouts preserve local detail but accumulate recursive error and compute. We test this rather than assuming it.

**Why not predict only the final state?**  
Endpoint prediction can skip collision events that determine the outcome and provides little local physical evidence.

**Is the symbolic geometry a hard manifold?**  
No. The implemented losses encourage proximity to relationally consistent level sets. Hard enforcement would require explicit constrained optimization.

**Why GINE rather than GIN?**  
The physical graph is edge-heavy: support direction, contact normal, distance, and relation type must participate in message passing.

**What would make us abandon the approach?**  
No oracle-symbol gain, no phase-specific failure in the continuous baseline, or no joint-controller advantage at matched compute.

---

# Backup: expected questions (from supervisor discussion)

**Isn't the controller just a router over prediction errors $d(\hat z, z^*)$ vs. $d(\hat g, g^*)$?**  
Error-based routing needs oracle access and can only exist as an upper bound. The deployed controller replaces measured error with a learned *forecast* of per-regime description validity, chooses jointly over horizon *and* description level (different prediction problems, not two answers to one), acts sequentially (each choice changes the next decision's state — a semi-MDP), and lets modes compose. The oracle router survives in exactly one place: the training-time label generator, and the ceiling in the oracle-gate ablation.

**Is the controller learnable, or does it need ground truth at test time?**  
It is learned. Ground truth is moved from test time to training time: every $(\Delta,\alpha)$ pair is scored per state offline, and $\pi_\kappa$ is distilled to predict the best pair from $h_t$ alone. The oracle is the teacher; the controller is the deployable student; the gap between them is a reported result.

**How is observation → symbol trained? Can the global loss distort predicate semantics?**  
Plain supervised learning against engine ground truth (frozen visual encoder + object/predicate heads), then frozen forever. End-to-end fine-tuning is a gated Stage-4 variant with a predicate-drift acceptance metric — any significant drift rejects it.

**If the symbolic generator is deterministic, how do you handle extraction noise and incoherence?**  
Deterministic mapping, probabilistic outputs: every predicate carries calibrated confidence, and nothing consumes symbols as certain. The gate is trained on the real extractor's outputs and learns its error profile; constraints apply only to high-confidence positives; and because $z$ is the sole state carrier, a one-frame mislabel cannot propagate. Robustness is measured (learned-vs-oracle gap, flip-rate coherence, noise-injection curve), not asserted.

---

# Backup: expected questions (continued)

**If $z$ and $g$ never interact, why not generate both in parallel — what is the point?**  
The perception front-end *is* parallel, by design. The interaction lives in the dynamics and the controller: $F_\theta(z_t, a_t, S^\mu_t)$ and $G_\omega(S^M, z, a)$ are cross-conditioned; symbolic losses shape the geometry of $z$ in training; symbolic confidence gates how $z$ is used. Parallel generation would still need a trust mechanism, would save no compute (the gains come from *not* rolling $z$ through quiescent stretches), and would lose symbolic anchoring of the rollout.

**How does $\alpha$ work across a rollout? Does it persist?**  
Within one decision, the chosen pair *defines* that step's prediction task — predict the state at $t_k{+}\Delta_k$ in representation $\alpha_k$. The result is mapped back into the shared carrier $z$, and the next decision re-selects $\alpha$ freely. No persistence assumption; consecutive steps can switch representation (the coupled switch). Feeding the previous pair into $h$ (recurrent controller) is an ablation, not a default.

**Does switching between latent and symbolic representations break gradients and force teacher forcing?**  
No discrete step sits in the rollout path: $z$ is always the carrier; symbolic heads are supervised readouts entering only the loss. Teacher forcing over the $(\Delta,\alpha)$ grid is the deliberate base regime (standard for JEPA), not a workaround; autoregressive rollout is differentiable throughout because the carrier is continuous. Exposure bias is handled by multi-horizon supervision, endpoint-level macro prediction, and — if needed — short-window fine-tuning through the carrier.

**At test time, what does $\alpha$ actually change, if symbols never re-enter the state?**  
$\alpha$ conditions how $F$ *computes* the next carrier $\hat z_{t+\Delta}$ — it selects the prediction task and parameterization, and which readout is emitted for the planner. $\Delta$ controls how far the state advances; $\alpha$ controls in which description space that advance is computed.

---

# Backup: notation

| Symbol | Meaning |
|---|---|
| $z_t$ | Continuous predictive latent (kinetic-level state); the sole rollout carrier |
| $h_t$ | Controller state: latent plus uncertainty/event/reliability features |
| $\Delta_k$ | Selected prediction horizon |
| $\alpha_k$ | Selected description level: continuous, micro, or macro |
| $r_\psi(h_t)$ | Learned reliability of fine relational constraints (Knudsen analogue) |
| $\phi^*(x_t)$ | Oracle reliability label from engine KE + contact activity |
| $G_t$, $z_{\text{sym}}$ | Scene graph and its GINE embedding (the symbolic representation) |
| $G_\omega$ | Macro-event predictor (not the scene graph $G_t$) |
| $S_t^\mu, S_t^M$ | Micro-relational (moment-level) and macro-event (hydrodynamic-level) states |
| $A,R$ | Restriction (abstraction) and lifting (refinement) maps |
| $\pi_\kappa$ | Joint controller policy over the $(\Delta,\alpha)$ grid |
| $c(\Delta,\alpha)$ | Compute/complexity cost charged to the controller |

*Note:* in discussion, "$g$" (the symbolic representation) corresponds to $G_t$ / $z_{\text{sym}}$ — not to the event predictor $G_\omega$; and $h_t$ reads *whether* symbols are reliable (via confidence into $r_\psi$), not *what they say*.

---

# Source documents scanned

Current project authority:

- `docs/research_proposal.md`
- `docs/training_mechanism_and_architecture_specs.md`
- `docs/world_model_data_pipeline.md`
- `docs/iratus_aves_levels.md`

Supervisor-discussion working notes (decisions incorporated above):

- `docs/controller_router_QA.md`
- `docs/symbolic_representation_QA_and_pipeline.md`
- `docs/symbolic_noise_QA.md`
- `docs/z_g_interaction_and_alpha_mechanics_QA.md`
- `docs/world_model_training_regime_QA.md`
- `docs/system_flow_and_training_overview.md`

Design history used for context, but superseded where it differs from the current proposal:

- `docs/deprecated/high_level_plan/research_project_plan.md`
- `docs/deprecated/implementation_plan/discussion_about_method_details.md`

Editable figure sources and rendered assets are in `docs/method_proposal_presentation/image_assets/`.
