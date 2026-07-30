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

> **One-sentence method:** let the world model jointly change how far ahead it predicts and what kind of state it predicts as the physical regime changes.

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
- $\alpha_k$: **what representation** the predictor uses this step.
- $h_{t_k}$: **controller state** — current latent $z_t$ plus predictive uncertainty, event likelihood, and local event density.
- $r_\psi(h_t)\in[0,1]$: **learned reliability** of fine object relations (low → mask micro-relational constraints; continuous dynamics dominate).
- A horizon-conditioned JEPA predictor remains the continuous dynamics backbone.

---

# Proposed architecture

![BG-NS-JEPA architecture: diagnostics condition a joint controller, which selects a temporal horizon and representation for the predictor; relational heads provide gated training constraints.](image_assets/bg_ns_jepa_architecture.png)

The controller changes the *prediction problem itself*, not only the rollout step size.

---

# One adaptive rollout step

![width:800px Joint controller decision flow: estimate predictive signals, gate unreliable micro relations, score admissible horizon-representation pairs, predict, and repeat.](image_assets/joint_controller_decision.png)

Low relational reliability removes **micro-relational supervision** from consideration; it does not imply that all symbols, especially macro events, are meaningless.

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

**Key ablation:** joint controller vs. factorized $\pi_\Delta\pi_\alpha$ at matched compute.

---

# How the selected pair changes during one cascade

![Regime-adaptive cascade timeline: the controller moves from long-horizon macro prediction in stable phases to short-horizon continuous prediction during collision and collapse, then through micro-relational settling back to macro equilibrium.](image_assets/regime_adaptive_cascade_timeline.png)

The central proposal is the **coupled switch**: both $\Delta$ and $\alpha$ change at physical regime boundaries rather than following one fixed clock or representation.

---

# Three state representations

Initial operational definition: a **shared latent backbone** is conditioned on $\alpha$, while the selected mode determines the active prediction target and head.

| Mode | Selected target/head | Best suited to | Constraint policy |
|---|---|---|---|
| **Continuous** | Future latent $z_{t+\Delta}$ | Collision and active collapse | No fine relational correction |
| **Micro** | Object/predicate head: `contact`, `supports`, `velocity-bin` | Stable or slowly changing local structure | Reliability-gated |
| **Macro** | Event head: `cascade-active`, `collapsed`, `pigs-cleared` | Event transitions and endpoints | Supervised when selected |

Cross-level maps connect the two symbolic resolutions:

$$
S_t^M=A(S_t^\mu),
\qquad
\widetilde S_t^\mu=R(S_t^M,z_t).
$$

Each head maps its prediction back to the shared rollout state $z$ before the next decision. Whether shared heads, separate experts, or another interface works best is an **architectural ablation**, not a result assumed in advance.

---

# Symbolic reliability is learned, not assumed

The test-time gate is a learned estimator:

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

- High $r_\psi$: fine contact/support predicates are stable enough to regularize a rollout.
- Low $r_\psi$: mask fine relational loss and let continuous dynamics dominate.
- Macro-event predicates can remain useful even when individual contacts are unstable.

**Calibration check:** compare learned-gate decisions with oracle regimes and report switch precision/recall, not only task accuracy.

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
+\lambda_{\mathrm{cross}}\mathcal L_{\mathrm{cross}}
+\lambda_{\mathrm{cost}}c(\Delta_k,\alpha_k)
\right)
+\lambda_{\mathrm{anchor}}\mathcal L_{\mathrm{anchor}}
\right].
$$

| Term | Function |
|---|---|
| $\mathcal L_{\mathrm{pred}}$ | Match the target latent at the selected horizon |
| $\omega_\psi\mathcal L_{\mathrm{sym}}$ | Gate semantic, TPR, and contrastive relational losses |
| $\mathcal L_{\mathrm{cross}}$ | Keep micro and macro descriptions consistent |
| $c(\Delta,\alpha)$ | Prevent a degenerate always-fine policy |
| $\mathcal L_{\mathrm{anchor}}$ | Encourage an outcome-consistent terminal latent |

The $\Delta_k/T$ weighting makes variable-horizon training and fixed-policy baselines comparable by represented duration.

---

# How $(\Delta,\alpha)$ trains the JEPA world model

Standard JEPA training fixes the prediction task by hand: context latent $z_t$ → predict a **stop-gradient target latent** at a **fixed temporal offset**, in a **fixed embedding space**.

BG-NS-JEPA turns both fixed choices into state-dependent decisions:

- $\Delta_k$ — **how far ahead the target is**: predict at $t_k+\Delta_k$
- $\alpha_k$ — **which space the target lives in**: continuous latent (the usual JEPA target encoder), micro scene-graph predicates, or macro event states

Each selected pair instantiates one JEPA sub-task; a single predictor conditioned on $(\Delta,\alpha)$ is trained **multi-task over the pair grid**, with the latent-prediction loss weighted by $\Delta_k/T$. $\pi_\kappa$ is thus a **learned task scheduler** — like making the masking/window strategy of masked representation learning adaptive per state, instead of a fixed hyperparameter.

**Chicken-and-egg:** scheduler labels depend on predictor performance, but the predictor's training distribution depends on scheduler choices. Staged resolution:

1. **Stage 1 (exhaustive scoring):** train candidate $(\Delta,\alpha)$ predictors on oracle trajectories; score every pair per state → per-state **best-pair labels**
2. **Stage 3 (amortized scheduler):** distill the per-state argmin into $\pi_\kappa$; ablate joint vs. factorized
3. **Stage 4 (optional end-to-end):** discrete relaxation (Gumbel-softmax), kept only if it improves the frontier

*Stage 2 (learned symbol extraction) runs orthogonally — it changes where scene graphs come from, not how the controller or predictor are trained.*

---


# What is fixed now, and what will experiments decide?

**Fixed scientific hypothesis**

> A state-dependent **joint** choice of $(\Delta,\alpha)$ can outperform fixed, one-axis, and independent choices at matched compute.

| Open method decision | Runnable starting point | How we resolve it |
|---|---|---|
| How is $\pi_\kappa$ trained? | Score every pair on oracle trajectories using duration-normalized prediction error + physical violations + compute; train one categorical controller on the argmin labels | Compare oracle-label supervision with an end-to-end discrete relaxation; retain the simpler method unless the latter improves the frontier |
| What does selecting $\alpha$ change? | Shared backbone + mode-specific continuous, micro, and macro target heads | Ablate shared heads vs. separate experts and test cross-mode rollout consistency |
| Which horizons belong in $\mathcal D$? | Provisional $\mathcal D_0=\{1,5,15\}$ frames; rescale if pilot capture rate requires it | Select using dev-set prediction/compute trade-offs, then lock before test evaluation |

These choices will be **resolved progressively by controlled experiments**. They are important research questions, but each needs an explicit initial implementation before it can be tested.

---

# What data does the method require?

| Payload | Purpose | Project status |
|---|---|---|
| RGB frames + shot action | Continuous JEPA training and temporal ablations | **Available in the read-only pipeline** |
| Object state and kinematics | Oracle graph nodes; stability supervision | **Requires physics instrumentation** |
| Contact/support relations | GINE edges; physical-violation metrics | **Requires physics instrumentation** |
| Macro event labels | Event head and cross-level consistency | **To derive from validated engine state** |
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
| **1. Oracle symbols + fixed pairs** | Candidate $(\Delta,\alpha)$ predictors; graphs and gate from simulator | Do perfect symbols help, and do preferred pairs vary by regime? | Oracle upper-bound gain + per-state best-pair labels |
| **2. Learned symbolic state** | GINE/predicate heads and learned $r_\psi$ | Can the model recover useful relational structure? | Learned-vs-oracle gap and gate calibration |
| **3. Controller + interface ablation** | Joint policy and alternative mode interfaces | Which training rule and abstraction interface work best? | Matched-compute Pareto frontier + selected design |
| **4. Full model** | End-to-end BG-NS-JEPA | Does the complete method transfer? | NovPhy + Physhion/CLEVRER results |

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
| Does it help decisions? | Task success, shots-to-success, novelty adaptation | Better prediction does not improve planning |

This design separates failure of the **controller**, **symbol extraction**, and **relational inductive bias** rather than treating the method as one indivisible bundle.

---

# Optional mechanisms, not core claims

**Inference-time relational projection**

- When $r_\psi$ is high and a relational mode is selected, take 1–3 gradient steps to reduce the learned relational residual.
- Evaluate against a no-projection ablation.
- Describe it as approximate optimization, not exact projection onto a known manifold.

**PDDL serialization**

- Micro and macro states may be serialized for a downstream planner.
- PDDL is an interface, not the learned temporal abstraction mechanism.

**Terminal anchor**

- Pulls a reliable final latent toward an outcome-specific symbolic pole.
- Encourages endpoint consistency; it cannot guarantee a correct rollout.

Keeping these claims narrow makes the primary controller contribution easier to test and defend.

---

# Main risks and controlled responses

| Risk | Controlled response |
|---|---|
| Symbol extraction is unstable | Establish oracle-symbol upper bound first |
| Reliability gate is miscalibrated | Report oracle gate, learned gate, and no-gate variants |
| Controller collapses to always-fine | Explicit cost term and matched-compute reporting |
| TPR is expensive | Efficient readout/projection approximation; ablate it |
| Vocabulary appears hand-engineered | Learn truth values/embeddings; test OOD transfer; keep vocabulary physically generic |
| Method appears to be a component bundle | Make joint-vs-factorized control the primary experiment |
| Instrumentation delays the full model | Publishable minimum uses oracle state and locked temporal baselines |

---

<!-- _class: lead -->

# Proposed decision for tomorrow

Proceed with this hierarchy of claims:

1. **Primary:** joint temporal and representational control improves the prediction–plausibility–compute frontier.
2. **Mechanism:** learned reliability gates fine relational constraints without suppressing useful macro events.
3. **Secondary:** GINE + TPR provides structured relational regularization.

Immediate implementation gate:

> First complete physics instrumentation and the oracle-symbol experiment; continue to learned symbols only if structured supervision produces a measurable upper-bound gain.

Supervisor feedback requested on:

- Is the joint-controller claim sufficiently focused for the target venue?
- Is oracle-symbol Stage 1 an acceptable minimum publishable unit?
- Should planning results be required for the main paper or treated as downstream validation?

---

# Backup: expected questions

**Why not use only a stability gate?**  
A binary gate decides whether one loss is active. The proposed controller chooses among multiple horizons and representation types, including macro events when micro relations are unreliable.

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

# Backup: notation

| Symbol | Meaning |
|---|---|
| $z_t$ | Continuous predictive latent |
| $h_t$ | Controller state: latent plus uncertainty/event/reliability features |
| $\Delta_k$ | Selected prediction horizon |
| $\alpha_k$ | Selected abstraction: continuous, micro, or macro |
| $r_\psi(h_t)$ | Learned reliability of fine relational constraints |
| $S_t^\mu, S_t^M$ | Micro-relational and macro-event states |
| $A,R$ | Abstraction and refinement maps |
| $c(\Delta,\alpha)$ | Compute/complexity cost charged to the controller |

---

# Source documents scanned

Current project authority:

- `docs/research_proposal.md`
- `docs/training_mechanism_and_architecture_specs.md`
- `docs/world_model_data_pipeline.md`
- `docs/iratus_aves_levels.md`

Design history used for context, but superseded where it differs from the current proposal:

- `docs/deprecated/high_level_plan/research_project_plan.md`
- `docs/deprecated/implementation_plan/discussion_about_method_details.md`

Editable figure sources and rendered assets are in `docs/method_proposal_presentation/image_assets/`.
