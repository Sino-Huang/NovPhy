# Regime-Adaptive World Models: Joint Temporal and Symbolic Abstraction for Persistent Physical Cascades

**Target Venue:** ICLR 2027  
**Keywords:** world models, JEPA, neuro-symbolic learning, adaptive abstraction, temporal abstraction, physical reasoning, scene graphs, multiscale modeling, kinetic theory, Angry Birds, NovPhy.

---

## 1. Abstract

A single action in Angry Birds can trigger a physical cascade lasting 50–150 frames while the agent does not act again. Existing JEPA world models normally predict on a fixed temporal clock with a fixed latent abstraction. In this setting, short strides compound rollout error while long strides can skip collision events that determine the outcome; a fixed object-level symbolic description can likewise be unreliable while contact topology changes rapidly.

Physics faced the same problem a century ago. Kinetic theory connects microscopic particle dynamics to macroscopic fluid behavior by recognizing that **no single description level is valid across all regimes**: collision-dominated intervals require a fine kinetic description, while scale-separated intervals admit a coarse hydrodynamic one, and the switching criterion is itself state-dependent (the local Knudsen number). We import this *algorithmic* lesson into learned world models.

We identify the world-model analogue as a **representation-control problem**. Persistent physical cascades are heterogeneous: interaction-active intervals require fine continuous prediction, while quiescent intervals and stable endpoints admit compact relational abstractions. Existing world models learn representations or improve training stability; neuro-symbolic world models learn symbolic states or operators. Neither treats prediction horizon and symbolic abstraction as **joint, state-dependent decisions**.

We propose **Bi-Granular Neuro-Symbolic JEPA (BG-NS-JEPA)**, a regime-adaptive world model that selects temporal resolution $\Delta t$ and representational abstraction $\alpha \in \{\text{continuous}, \text{micro-relational}, \text{macro-event}\}$ from the current predictive state. The three abstraction levels form a micro–meso–macro hierarchy analogous to kinetic, moment, and hydrodynamic descriptions: the continuous latent plays the role of the kinetic state, object-level predicates act as low-order *moments* of that state, and macro-event predicates act as hydrodynamic (slow, coarse) variables. A learned **scale-separation estimator** — the world-model analogue of a local Knudsen number — gates fine relational constraints, allowing continuous dynamics to dominate when contact/support predicates are unreliable while retaining event-level descriptions when they remain useful. The symbolic layer supplies compositional state and cross-level closure consistency as reliability-gated constraints and readouts on a continuous state carrier; it is not a universal rule engine, and it is never the rollout state.

Our secondary representational contribution, **Structured Physical Symbolic Geometry (SPSG)**, uses scene-graph and tensor-product structure to regularize relational states. We treat the resulting geometry as a learned soft constraint, with optional inference-time projection, rather than claiming an unconditional hard manifold or a dynamical-systems equivalence.

The one-sentence pitch is:

> **BG-NS-JEPA is the learned analogue of a hybrid kinetic–fluid solver: it estimates the local degree of scale separation and jointly selects how far ahead to predict and at which level of description — continuous, relational, or event-level.**

*Scope note:* we claim no rigorous derivation and no formal connection to the mathematical theory of the Boltzmann equation; we import the *algorithmic and conceptual* lessons of the multiscale tradition into learned latent dynamics.

---

## 2. Problem: The Granularity Mismatch in Action-Sparse Persistent-Effect Environments

### 2.1 Environment Definition

An episode is $\mathcal{E} = \{(o_t, a_t, x_t)\}_{t=0}^T$ with:

- **Action sparsity ratio**: $\rho_a = \frac{1}{T}\sum_{t=0}^{T-1} \mathbf{1}[a_t \neq \text{noop}] \ll 1$
- **Effect persistence**: $\tau_{\text{eff}} = \min\{k: \forall t > t_{\text{last}}+k, \|x_t - x_{t-1}\| < \epsilon\} \gg 1$

We term these **action-sparse persistent-effect environments**. They are not merely "long-horizon"; they are **action-sparse**, meaning the world evolves autonomously for most of the episode.

### 2.2 The Dual-Granularity Trade-off

**Temporal Axis (How often to predict):**

- **Fine $\Delta t$ (frame-by-frame)**: High local fidelity, but recursive errors can accumulate over $\tau_{\text{eff}}$.
- **Coarse $\Delta t$ (event-level)**: Low drift, but may miss critical collision events that determine final outcome.
- **Hypothesis**: No fixed stride is Pareto-optimal across all dynamical regimes; a regime-adaptive policy can improve the prediction–computation frontier.

**Physical Axis (What to predict):**

- **Continuous $\alpha = \text{continuous}$**: A neural latent for rapidly changing contact and motion — the *kinetic-level* description.
- **Micro-relational $\alpha = \text{micro}$**: Object-level graph predicates (contact, support, velocity-bin) — low-order *moments* of the full state, used only when their reliability is high.
- **Macro-event $\alpha = \text{macro}$**: Structure/event predicates (cascade-active, collapsed, pigs-cleared, steady-state) — *hydrodynamic-level* slow variables, which can remain useful even while individual contacts change. Terminal macro states play the role of equilibrium states that the cascade relaxes into.

### 2.3 Scale Separation and Symbolic Reliability

Kinetic theory switches description levels according to the **local Knudsen number** — the ratio of the interaction (mean-free) scale to the observation scale. Our world model needs a learned analogue. We define a **scale-separation estimator** $s_\psi(h_t)$, whose symbolic face is the **symbolic-reliability score** $r_\psi(h_t) \in [0, 1]$, where $h_t$ contains the latent state, predictive uncertainty, and event likelihood. An oracle label can be constructed from simulator features during training:

$$\phi^*(x_t) = \mathbb{1}_{\left[\text{KE}(x_t) < \epsilon_{\text{KE}}\right]} \cdot \mathbb{1}_{\left[\text{contacts}_{\text{active}}(x_t) < \epsilon_{\text{contact}}\right]}.$$

Kinetic energy and contact activity jointly estimate the ratio between the local interaction timescale and the observation timescale — the physical content of a Knudsen number. High reliability means that the selected fine relational ontology, for example `supports(A,B)`, is expected to be stable enough to constrain a rollout. Low reliability does not make all symbols meaningless: coarse event predicates (the hydrodynamic level) may still be available, exactly as fluid equations remain valid where moment-level detail is not. Kinetic energy, contact activity, and finite-time divergence are diagnostics for this estimator, not definitions equivalent to a largest Lyapunov exponent.

---

## 3. Research Gap: World Models Do Not Control Their Level of Description

### 3.1 The multiscale precedent

The idea that a physical system must be simulated at different description levels in different regimes — and that the switching criterion is part of the model — is a mature tradition in scientific computing:

- **Kinetic theory and moment closure.** The Boltzmann equation mediates between microscopic mechanics and fluid dynamics; hydrodynamic equations are obtained from it by Chapman–Enskog expansion, and moment-closure hierarchies (Grad; Levermore) formalize intermediate description levels. The rigorous derivation of this hierarchy remains an active frontier of mathematics, including the recent long-time derivation of the Boltzmann equation from hard-sphere dynamics by Deng, Hani, and Ma.
- **Hybrid kinetic–fluid solvers.** Adaptive mesh and algorithm refinement (Garcia, Bell, Crutchfield, and Alder) and kinetic–fluid domain-decomposition schemes run an expensive kinetic solver only where the local Knudsen number demands it, and a cheap fluid solver elsewhere — a hand-crafted, state-dependent choice of *both* resolution and description level.
- **Equation-free and heterogeneous multiscale methods.** Kevrekidis's equation-free framework and E and Engquist's HMM define *restriction* (fine → coarse) and *lifting* (coarse → fine) operators, enabling coarse time-steppers that call fine-scale dynamics only inside selected intervals. The Mori–Zwanzig formalism provides the projection-operator view: coarse variables evolve under a closed equation plus a memory term accounting for unresolved degrees of freedom.

Two lessons transfer to learned world models. First, **description level is a state-dependent decision**, not an architecture constant. Second, **temporal resolution and description level are not independent**: both are governed by the same underlying quantity — the local degree of scale separation. A controller that chooses them independently assumes a factorization that the physics does not support.

### 3.2 The gap in learned world models

The nearest research threads leave a specific gap:

1. **JEPA world models** improve representation learning, object structure, or optimization stability, but normally retain a fixed temporal sampling policy and prediction representation — a single, fixed description level.
2. **Temporal-abstraction methods** learn when an agent acts, rather than when a world model should advance time and change state abstraction.
3. **Symbolic world models** induce a fixed abstract transition system; they do not model a continuous cascade in which the reliability of object-level relations itself changes.

None of these lines exploits the multiscale lesson. Our response is **joint regime-adaptive representation control**, structured as a learned micro–meso–macro hierarchy:

| Level | Description | World-model realization | Physics analogue |
|---|---|---|---|
| Micro | Full fine-grained predictive state | Continuous JEPA latent $z_t$ | Kinetic / particle description |
| Meso | Low-order relational statistics | Micro-relational predicates (`contact`, `supports`) | Moments of the kinetic distribution |
| Macro | Coarse slow variables and equilibria | Macro-event predicates (`collapsed`, `steady-state`) | Hydrodynamic fields; equilibrium states |

**Structured Physical Symbolic Geometry (SPSG)** is the relational mechanism within that controller, not the paper's primary novelty. It uses learnable predicate projections to regularize a latent toward relationally consistent regions:

$$\mathcal{M}_{\text{valid}} = \{z \in \mathcal{Z} \mid \forall s: f_s(z) = c_s\}$$

where $f_s: \mathcal{Z} \to \mathbb{R}$ decodes whether symbolic predicate $s$ holds. The conjunction of symbols corresponds to the intersection of submanifolds: $\mathcal{M}_{s_1 \land s_2} = \mathcal{M}_{s_1} \cap \mathcal{M}_{s_2}$. This supplies an algebraic inductive bias compatible with conjunction when the predicate maps are sufficiently well behaved; the implementation enforces it with losses and optional projection rather than an unconditional hard constraint.

During interaction-active intervals, the controller may select a continuous latent and mask unreliable fine relational losses. At stable or slowly changing intervals, it can select micro-relational or macro-event states and enforce cross-level closure consistency. The novelty is the state-dependent, *joint* selection of this pair, grounded in a learned scale-separation estimate — not a claim that one representation is universally correct.

---

## 4. Method: Joint Regime-Adaptive Abstraction

At each decision point, the controller chooses a pair from the cross-product of temporal and representational choices:

$$t_{k+1}=\min(t_k+\Delta_k,T), \qquad (\Delta_k,\alpha_k) \sim \pi_\kappa(\cdot\mid h_{t_k}),$$

where $\Delta_k \in \mathcal{D}$ is a prediction horizon, $\alpha_k \in \{\text{continuous},\text{micro},\text{macro}\}$ is a description level, and $h_t$ includes the current latent, uncertainty, event likelihood, and the scale-separation estimate. The controller is joint rather than factorized: it must learn, for example, that a short continuous prediction is useful at collision onset while a long macro-event transition can be appropriate after settling. In multiscale terms, $\pi_\kappa$ replaces the hand-crafted domain-decomposition criterion of a hybrid kinetic–fluid solver with a learned one.

All components serve one duration-normalized objective:

$$\boxed{\mathcal{L}_{\text{total}} = \mathbb{E}\!\left[\sum_k \frac{\Delta_k}{T}\left(\mathcal{L}_{\text{pred}}^{\Delta_k,\alpha_k} + \lambda_{\text{sym}}\omega_\psi(h_{t_k},\alpha_k)\mathcal{L}_{\text{sym}}^{\alpha_k} + \lambda_{\text{cons}}\mathcal{L}_{\text{closure}} + \lambda_{\text{cost}}c(\Delta_k,\alpha_k)\right) + \lambda_{\text{anchor}}\mathcal{L}_{\text{anchor}}\right].}$$

$\mathcal{L}_{\text{sym}}$ collects semantic, structural, and contrastive terms. $\omega_\psi(h,\alpha)$ is $r_\psi(h)$ for micro-relational constraints, $1$ for selected macro-event supervision, and $0$ for a continuous step. $\mathcal{L}_{\text{closure}}$ enforces **closure consistency**: agreement between the meso (moment-level) and macro (hydrodynamic-level) descriptions, analogous to the requirement that a hydrodynamic equation be derivable from the kinetic level beneath it. $c$ prevents the controller from selecting a degenerate always-fine solution.

**Design principle — symbols are constraints and readouts, never the state carrier.** One principle generates several decisions below: **the continuous latent $z$ is the only rollout state carrier**. Whatever description level the controller selects, the predictor $F$ always produces the next continuous carrier $\hat{z}_{t+\Delta}$; symbolic heads are supervised readouts that enter only the loss (or a downstream planner/monitor). No discrete step — argmax, graph assembly, symbol decoding — ever sits in the rollout path. Three consequences follow. (i) The base training regime is teacher-forced and fully differentiable (§6.7). (ii) A one-frame symbolic misclassification cannot propagate into the rollout state: the symbolic incoherence that afflicts purely symbolic world models is structurally absent here, and residual extractor noise only adds down-weighted noise to a frame's loss. (iii) The only symbolic structure that persists across time is the macro-event chain, whose predicates are absorbing by vocabulary design (`collapsed`, once true, stays true).

### 4.1 Component A: Continuous Dynamics Backbone (the kinetic level)

Action-conditioned, horizon- and mode-indexed predictor with a **dual output**:

$$F_\theta^{\Delta_k,\alpha_k}(z_{t_k}, a_{[t_k,t_{k+1})}, S_{t_k}^\mu) \;\longrightarrow\; \underbrace{\hat{z}_{t_{k+1}}}_{\text{rollout carrier — always produced}} \;+\; \underbrace{\text{mode-head readout: } \hat{S}^\mu \;\text{or}\; (\hat{S}^M, \hat{\Delta}, \hat{e})}_{\text{enters only the loss / planner interface}}.$$

The selected pair $(\Delta_k, \alpha_k)$ *defines the prediction task*: it conditions how the carrier itself is computed, not merely which loss head is active. The rollout always continues from $\hat{z}_{t_{k+1}}$, and symbolic readouts never re-enter the state, so the carrier computation is an ordinary differentiable feed-forward path. Conditioning on the symbolic state $S_{t_k}^\mu$ is also a structural interaction between the two descriptions: continuous dynamics are symbol-aware even when symbolic constraints are gated off.

When the controller selects the continuous abstraction, it predicts rapidly changing physics without fine relational correction. It is not assumed that all event-level semantics disappear in this regime.

*Optional structural variant:* the predictor may be decomposed as a **free-streaming plus interaction** pair, $F_\theta = F_{\text{drift}} + \mathbf{1}[\text{event-active}] \cdot F_{\text{interact}}$, mirroring the operator splitting between ballistic transport and the collision operator in kinetic numerics. This is an architectural ablation, not a core claim.

### 4.2 Component B: Structured Physical Symbolic Geometry (SPSG)

**Scene Graph Encoding**:

- **Nodes**: Objects (bird, pig, wood_block, TNT) with features $[type, material, position, velocity, shape, health]$.
- **Edges**: Directed physical relations with features $[relation\_type, relative\_position, distance, contact\_normal, support\_strength]$.
- **Encoder**: GINEConv (Graph Isomorphism Network with Edge features), preserving permutation invariance, compositionality, and edge semantics.

The encoder maps a scene graph $G_t$ to a symbolic embedding:
$$z_{t,\text{sym}} = f_{\text{GINE}}(G_t)$$

**Physics-validated negative sampling**:
Instead of random corruption, use simulator-validated counterfactual configurations:

- Reversed gravity under an explicit simulator coordinate convention
- Massless materials: $\rho_{\text{wood}} = 0$
- Anti-support: remove support while keeping structure upright

Only configurations with an explicit semantic or physical violation are retained as negatives; a simulator failure is not itself evidence of an impossible world.

### 4.3 Component C: Dual-Resolution Relational State and Event Interface

**Micro-PDDL** (object-centric, fine physical granularity — the moment level):

```lisp
(contact block_1 block_2)
(supports block_1 pig_1)
(velocity-bin block_3 high)
(damaged wood_block_7)
```

**Macro-PDDL** (structure-centric, coarse physical granularity — the hydrodynamic level):

```lisp
(structure-unstable tower_1)
(cascade-active tower_1)
(collapsed tower_1)
(pigs-cleared left-region)
(steady-state level)
```

The cross-level maps are the learned analogues of **restriction and lifting operators** in equation-free multiscale computation:

- **Restriction (abstraction)**: $S_t^M = A(S_t^\mu)$
- **Lifting (refinement)**: $\tilde{S}_t^\mu = R(S_t^M, z_t)$

The macro state provides an event interface that can summarize a cascade:
$$\mathcal{C} = \{(\tau_i, S_{\tau_i}^M, e_i, \Delta_i, S_{\tau_{i+1}}^M)\}_{i=1}^K$$

Prediction becomes: $(S_{\tau_{i+1}}^M, \Delta_i, e_i) = G_\omega(S_{\tau_i}^M, z_{\tau_i}, a_{\tau_i})$.

The continuous JEPA backbone fills in micro-dynamics inside selected event intervals:
$$z_{t+1} = F_\theta(z_t, a_t, S_t^\mu) \quad \text{for } t \in [\tau_i, \tau_i + \delta_i]$$

This is the learned counterpart of a coarse time-stepper that invokes fine-scale dynamics only where the coarse closure is insufficient — in Mori–Zwanzig terms, the macro model is the closed equation and the continuous backbone supplies the memory term for unresolved degrees of freedom. It also illustrates that the modes **compose** rather than compete: macro-event prediction and continuous infilling can be active simultaneously inside one event interval.

PDDL is an optional serialization for a downstream planner; it is not itself claimed to be a learned temporal codec.

### 4.4 Component D: Joint Bi-Granular Controller

The controller jointly selects temporal resolution $\Delta t_k$ and description level $\alpha_k$:

$$(\Delta t_k, \alpha_k) = \pi_\kappa(u_t, r_\psi(h_t), p_\eta(E_t), \lambda_t).$$

where:

- $u_t$: JEPA predictor uncertainty (latent variance)
- $r_\psi(h_t)$: learned reliability of fine relational constraints (symbolic face of the scale-separation estimate)
- $p_\eta(E_t)$: Predicted event probability
- $\lambda_t$: Local event density

The four inputs are diagnostics of a single underlying quantity — the **local degree of scale separation**, the world-model analogue of a local Knudsen number. Just as the Knudsen number simultaneously dictates whether a kinetic description is needed *and* what temporal resolution a simulation requires, this single quantity governs both controller axes. This is the principled reason the controller must be joint: a factorized policy $\pi_\Delta \pi_\alpha$ assumes the independence of two projections of one physical quantity.

**Decision regimes**:

| Regime                 | $r_\psi$ | $u_t$  | $\lambda_t$ | $\Delta t$ | $\alpha$    |
| ---------------------- | ------ | ------ | ----------- | ---------- | ----------- |
| Pre-shot stable        | 1      | low    | low         | coarse     | macro       |
| Collision onset        | low    | high   | high        | fine       | continuous  |
| Interaction-active collapse | low | high | high        | fine       | continuous  |
| Post-collapse settling | 1→0→1  | medium | medium      | adaptive   | micro→macro |
| Equilibrium            | 1      | low    | low         | coarse     | macro       |

**Key design**: the controller selects the pair $(\Delta t,\alpha)$ jointly. Fine relational losses are masked when reliability is low, while a macro event representation may still be selected if it predicts the outcome more accurately or cheaply.

**How the controller is trained.** $\pi_\kappa$ is a learned policy network distilled from exhaustive offline scoring. In Stage 1, every $(\Delta,\alpha)$ pair is scored per state on oracle trajectories (duration-normalized prediction error + physical violations + compute), and the per-state argmin becomes the controller's supervision (§6.3). At test time a single forward pass of $\pi_\kappa$ produces the decision; ground truth is neither needed nor available. An optional end-to-end variant (Gumbel-softmax relaxation of the discrete pair choice) is kept only as a Stage-4 ablation. Feeding the previous pair $(\Delta_{k-1}, \alpha_{k-1})$ into $h$ — a recurrent controller — is likewise treated as an architectural ablation rather than a default.

**Why a controller, not a router.** The controller differs from error-based routing, or a mixture-of-experts gate over two predictors, in four ways. (i) Its decision signal is a **forecast** of which description will remain valid over the *upcoming* horizon; the measured errors $d(\hat{z}_t, z_t^*)$ that a router would compare do not exist at test time, especially dozens of open-loop steps in. The oracle comparison survives in exactly one place: as the training-time label generator, and as the ceiling in the oracle-gate ablation. (ii) The candidates are **different prediction problems** — different target spaces at different temporal offsets — not two answers to one problem; the controller selects *which question to ask*, jointly over horizon and representation. (iii) Decisions are **sequential**: choosing $\Delta{=}15$ skips 14 frames of information, so the next decision is made from a different state. This is a policy over prediction tasks in a semi-MDP, with decision-level credit assignment and a real cost term — not a per-step classifier. (iv) The modes **compose**: macro-event prediction and continuous infilling can be active simultaneously within one event interval (§4.3).

**What $\alpha$ selects at test time.** Because symbolic heads are readouts, $\alpha$'s test-time effect runs entirely through the $(\Delta,\alpha)$-conditioning of $F$: the selected pair parameterizes how the next carrier $\hat{z}$ is *computed*, and which readout is emitted for the planner or monitor. $\Delta$ controls how far the state advances; $\alpha$ controls in which description space that advance is computed. The controller therefore changes the prediction problem itself, rather than post-selecting among redundant outputs — and the compute gains come precisely from the decisions to *not* roll $z$ through quiescent stretches at fine resolution.

### 4.5 Component E: Reliability-Gated Constraints and Terminal Anchor

**Reliability-gated constraint**:
$$\ell_{\text{sym},k}=\sum_{(A,C)\in\mathcal{R}_{\text{valid}}}E(A,C)+\lambda\sum_{(A,C_{\text{neg}})\in\mathcal{R}_{\text{neg}}}\max(0,m-E(A,C_{\text{neg}})).$$

The total objective weights $\ell_{\text{sym},k}$ with $\omega_\psi(h_{t_k},\alpha_k)$, so only the constraint type selected by the controller receives supervision.

**Terminal anchor loss** (encourages an outcome-consistent endpoint):
$$\mathcal{L}_{\text{anchor}} = r_\psi(h_T) \cdot \sum_{c \in \mathcal{C}} \mathbb{1}_{[y=c]} \left\| g(z_c^{(K-1)}) - z_{\text{pole}}^{(c)} \right\|_2^2.$$

Terminal macro states act as **equilibria**: the endpoint a cascade relaxes into, which hydrodynamic-level prediction can reach directly even when the intermediate kinetic detail is chaotic. This applies outcome-level pressure at a reliable terminal state; it does not by itself guarantee a correct rollout.

### 4.6 Component F: Observation → Symbol Extraction (the supervised predicate parser)

What the method actually consumes is a set of **grounded predicate truth values with calibrated confidences**, plus — for SPSG — a structured embedding that preserves role binding. A scene graph is one convenient serialization of that requirement; the functional module is a *predicate parser*. Extraction is organized as a three-tier ladder:

| Tier | Input | Symbol source | Role |
|---|---|---|---|
| 0 — Oracle | Engine state | Ground-truth predicates | Defines predicate semantics; Stage-1 upper bound; generates all training labels |
| 1 — Feature parser | Simulator state vector | MLP heads on engine features | Sanity tier: verifies the predicate vocabulary is learnable before vision is involved |
| 2 — Visual parser | RGB frames | Frozen pretrained encoder (V-JEPA/DINO-style) + DETR-style object decoder + pairwise predicate heads + GINE | Deployed extractor (Stage 2) |

Semantics flow downward: Tier 0 *defines* what each predicate means; Tiers 1–2 are trained to imitate it and are evaluated by their agreement with it. In Tier 2, a query-based object decoder (DETR family, with Hungarian matching against engine objects) produces per-object slots with classification and attribute heads; an MLP over each ordered pair $(o_i, o_j)$ scores edge predicates (`contact`, `supports`, contact-normal and distance bins); unary and macro-event predicates are per-object or per-frame heads on the same slots; the graph is assembled only where GINE message passing and TPR role binding require relational composition. All outputs are **calibrated probabilities**, not hard labels — no downstream component ever consumes the symbolic state as certain.

**Frozen by design.** After supervised training, the extractor is frozen. Predicate semantics are defined by engine labels; if the global task loss could reshape the extractor, predicates such as `supports(A,B)` could drift into whatever reduces task loss — ontology reward-hacking — which would invalidate the physical-violation metrics, the switch precision/recall evaluation, and the calibration of $r_\psi$, whose oracle target is defined against exactly those engine semantics. End-to-end fine-tuning exists only as an optional Stage-4 variant, gated by an explicit **predicate-drift acceptance metric** (agreement with engine labels on held-out states, before vs. after); any significant drift rejects the variant.

**Noise handling.** Regime-level instability (fine predicates physically unstable in this state) and extractor-level error (the state is stable but the parser misclassifies) are conceptually distinct noise sources, and they are measured separately — but they are *statistically absorbed* by the same gate: $r_\psi$ is trained on the real extractor's confidence features, so it learns which states the parser tends to fail in (occlusion, particle-scale contacts, fast motion) and down-weights micro-relational constraints there. Constraints are applied only to **high-confidence positives**, exploiting the cost asymmetry: a false-positive constraint (enforcing a relation that does not hold) is far more damaging than a false negative (one missing constraint degrades gracefully to the continuous baseline). Because $z$ is the sole state carrier, residual symbolic errors cannot propagate into the rollout (§4, design principle).

---

## 5. Theoretical Contributions

### 5.1 Structured Relational Regularization

SPSG treats predicate projections as structured regularizers:

$$\mathcal{M}_s = \{z \in \mathcal{Z} : f_s(z) = c_s\}$$

If $c_s$ is a regular value of $f_s$, then $f_s^{-1}(c_s)$ is locally a submanifold. The implemented losses only encourage proximity to this set; a hard constraint requires an explicit projection or constrained optimizer. This distinction keeps the geometry claim precise.

### 5.2 Reliability-Aware Switching

The theory target is not a Lyapunov equivalence. It is a switching claim: when different regimes prefer different $(\Delta,\alpha)$ pairs, a sufficiently accurate state-dependent controller can outperform every fixed pair, subject to its prediction and compute costs. Simulator-derived stability measures provide supervision and diagnostics for the scale-separation estimator. The multiscale precedent supplies the prior expectation that regimes with different scale separation genuinely prefer different pairs — this is exactly the regime dependence that justifies hybrid kinetic–fluid solvers.

### 5.3 Joint Granularity Frontier

We evaluate a **Pareto frontier** over temporal and representational granularity:

| Controller choice | Endpoint correctness | Local fidelity | Compute |
| ----------------- | -------------------- | -------------- | ------- |
| Fixed coarse macro | May miss interactions | Low | Low |
| Fixed fine continuous | High locally, may drift | High | High |
| Joint regime-adaptive | **Hypothesized best trade-off** | Regime-dependent | Adaptive |

**Hypothesis**: A joint controller beats fixed and factorized choices at matched compute because physical regimes induce different preferred pairs, not merely different temporal strides — and because both axes are governed by one scale-separation quantity, which a factorized policy cannot represent.

---

## 6. Experimental Methodology

### 6.1 Stage 1: Oracle Symbol Upper Bound (Weeks 1-4)

Use ground-truth scene graphs from NovPhy/Box2D. Research question:

> If symbolic graphs are perfect, can structured symbolic geometry improve JEPA endpoint prediction and OOD robustness?

This de-risks the project before solving perception. Stage 1 also serves two mechanical roles: it trains the dual-output $F$ over the full $(\Delta,\alpha)$ grid (teacher-forced multi-task, §6.7), and it produces — by exhaustive per-state scoring of every pair — the **best-pair labels** that later supervise the controller.

### 6.2 Stage 2: Learned Symbolic State (Weeks 5-8)

Train the Tier-2 visual predicate parser (§4.6): a frozen pretrained visual encoder; a DETR-style object decoder with Hungarian matching against engine objects; attribute, pairwise-edge, unary, and macro predicate heads; the GINE embedding for SPSG; and a light temporal-consistency loss on predicate logits within oracle-stable regimes. Split by level/scenario, not by frame, so generalization is measured honestly. Then train $r_\psi$ on the extractor's confidence features against the oracle $\phi^*$. The extractor is frozen thereafter. Compare:

- Oracle symbolic state (Tier 0)
- Learned symbolic state from simulator features (Tier 1)
- Learned symbolic state from images (Tier 2)

**Stage-2 exit evidence**: the learned-vs-oracle gap on controller performance; per-predicate F1 against engine labels on held-out levels; gate calibration (switch precision/recall vs. oracle regimes); and symbolic coherence (predicate flip rate vs. engine flip rate).

### 6.3 Stage 3: Joint-Controller Ablation (Weeks 9-11)

Train $\pi_\kappa$ by distilling the Stage-1 best-pair labels (supervised classification over the pair grid). Compare fixed pairs, temporal-only adaptation, abstraction-only adaptation, a factorized controller $\pi_\Delta\pi_\alpha$, and the joint controller; optionally include a recurrent controller that receives $(\Delta_{k-1}, \alpha_{k-1})$ as an input. Measure:

- Terminal symbolic accuracy (zero-shot via pole distance)
- Horizon-normalized continuous prediction error at matched compute
- Physical violation rate (penetration, floating, illegal contact)
- OOD generalization (novel materials, gravity)

**Expected signature**: the joint controller improves the endpoint-correctness/physical-plausibility/compute frontier and chooses regime-aligned pairs more accurately than factorized alternatives.

### 6.4 Stage 4: Full BG-NS-JEPA (Weeks 12-14)

End-to-end training with bi-granular controller. Evaluate on:

- NovPhy standard and novelty scenarios
- Cross-domain: Physhion (8 scenarios), CLEVRER (collision reasoning)

Stage 4 also hosts the **optional variants, each gated by an acceptance criterion**: end-to-end controller relaxation (Gumbel-softmax through the discrete pair choice); Phase-C short-window autoregressive fine-tuning of $F$ through the $z$ carrier (adopted only if measured pilot drift demands it, §6.7); and extractor fine-tuning with an anchored symbol loss plus the predicate-drift acceptance metric (rejected on any significant drift).

### 6.5 Baselines

| Baseline                   | Purpose                              |
| -------------------------- | ------------------------------------ |
| LeWM                       | Pure continuous JEPA                 |
| Sub-JEPA                   | Latent regularization                |
| Uniform micro              | Always frame-by-frame + object-level |
| Uniform macro              | Always event-level + structure-level |
| ThinkJEPA                  | VLM external guidance                |
| Causal-JEPA                | Object masking                       |
| Temporal-only adaptive     | Learns $\Delta t$ with fixed abstraction |
| Abstraction-only adaptive  | Learns $\alpha$ with fixed stride   |
| Factorized controller      | Learns $\pi_\Delta\pi_\alpha$ rather than a joint policy |
| Recurrent controller       | Feeds $(\Delta_{k-1},\alpha_{k-1})$ into $h$ |
| BG-NS-JEPA (oracle reliability) | Upper bound for the learned gate |
| BG-NS-JEPA (no reliability gate) | Applies fine relational constraints uniformly |
| BG-NS-JEPA (no SPSG)       | Ablates structured symbolic geometry |
| BG-NS-JEPA (oracle graph)  | Upper bound for symbol extraction    |

### 6.6 Metrics

**Prediction metrics**: ADE@H, FDE@H, Final-state accuracy, Event prediction F1  
**Physical plausibility**: Object penetration rate, unsupported-floating rate, illegal-contact rate  
**Granularity metrics**: effective prediction steps, controller cost, switch precision/recall against oracle regimes, and joint-pair calibration  
**Symbolic robustness**: per-predicate F1 and confidence calibration; predicate flip-rate coherence vs. engine ground truth; noise-injection degradation curve (endpoint accuracy vs. increasing label noise); predicate-drift acceptance metric (Stage-4 extractor variant only)  
**Planning metrics**: Task success, shots-to-success, novelty adaptation speed

### 6.7 World-Model Training Regime

$F$'s base regime is **teacher-forced multi-task learning over the $(\Delta,\alpha)$ grid** — the standard JEPA regime of single-step supervision from true encoded states, made explicit here rather than left implicit. Teacher forcing is a deliberate choice, not a concession forced by non-differentiability: because $z$ is always the state carrier (§4), no gradient ever needs to cross a symbolic representation.

- **Phase A — $F$ pretraining (Stage 1).** Trajectories are encoded by the target encoder ($z^*$ sequence); engine state gives $S^{\mu*}, S^{M*}$. Sample $(t, \Delta, \alpha)$ over the grid; predict from true $z_t$; align the carrier to $\mathrm{sg}(z^*_{t+\Delta})$; supervise the selected mode head by cross-entropy; weight by $\Delta/T$. Fully teacher-forced, fully differentiable; no gradient-through-time.
- **Phase B — controller-coupled training (Stage 3).** Same objective; the $(\Delta,\alpha)$ sampling distribution follows the controller (or stays uniform, per ablation design); the reliability gate $\omega_\psi$ modulates the symbolic losses.
- **Phase C — optional anti-exposure-bias fine-tuning.** Teacher forcing's standard cost is the train/test mismatch (train on true states, test on the model's own predictions). Structural mitigations already exist: multi-horizon supervision (direct targets at $\Delta{=}15$, not only composed one-step predictions), endpoint-level macro jumps, and the terminal anchor. If pilot drift is still unacceptable, add short-window autoregressive fine-tuning / scheduled sampling **through the $z$ carrier only** — feasible precisely because the carrier path is differentiable end-to-end.
- **Stage 4 — optional end-to-end controller.** The only place gradients must cross a discrete choice is the end-to-end controller variant (Gumbel-softmax / REINFORCE-style) — one reason it is optional. By default only $\pi_\kappa$ is touched there, not $F$.

---

## 7. Risk and Mitigation

| Risk                                           | Mitigation                                                   |
| ---------------------------------------------- | ------------------------------------------------------------ |
| Learned symbolic extraction unstable           | Start with oracle; Tier-1 feature parser as fallback; frozen supervised extractor prevents semantic drift; GINE is more stable than slot attention |
| Symbolic extraction noise corrupts rollouts    | $z$ is the sole state carrier; symbols are readouts/constraints, never integrated forward; constraints apply only to high-confidence positives; noise-injection degradation curve is reported |
| Controller dismissed as error-based routing / MoE | Four structural differences (reliability *forecast* vs. measured error; joint prediction-task selection over $(\Delta,\alpha)$; sequential semi-MDP; mode composition); the oracle router survives only as the label generator and the oracle-gate ceiling |
| "Hand-engineered" vocabulary critique          | Vocabulary is universal physical substrate (like CNN locality); all truth values are learned |
| NovPhy is adaptation benchmark, not prediction | Add dedicated prediction track; use world model as adaptation engine |
| Reliability estimator misclassifies a regime   | Report oracle-gate upper bound; train with noisy labels and use calibrated confidence |
| Reviewer sees a component bundle               | Ground the design in the multiscale hierarchy (one scale-separation quantity generates all components); the state-carrier principle fixes how the components interact; make joint-controller and factorization ablations the primary evidence |
| Reviewer sees physics name-dropping            | Cite the *algorithmic* tradition (hybrid solvers, equation-free, HMM, moment closure), not theorems; every borrowed concept must correspond to a concrete design decision |
| Method too complex for 3-4 months              | Oracle-symbol version is minimum publishable unit            |

---

## 8. Contributions

1. **Problem**: Defines action-sparse persistent-effect environments as a world-model setting with nonuniform temporal and representational demands.
2. **Conceptual**: Identifies **joint regime-adaptive representation control** as the missing principle, and grounds it in the multiscale tradition: description level is a state-dependent decision, and both decision axes are projections of a single scale-separation quantity. BG-NS-JEPA is, to our knowledge, the first learned analogue of hybrid kinetic–fluid algorithm refinement for latent world models.
3. **Method**: BG-NS-JEPA, combining a joint controller distilled from exhaustive oracle scoring, a learned scale-separation/symbolic-reliability estimator, cross-level closure consistency via restriction/lifting maps, and a dual-output continuous JEPA backbone in which the latent is the sole rollout state carrier and symbolic heads are supervised readouts — keeping base training teacher-forced, fully differentiable, and structurally robust to symbolic extraction noise.
4. **Representation**: SPSG supplies compositional scene-graph and TPR regularization when fine relational predicates are reliable.
5. **Empirical**: Tests whether the joint controller improves the prediction–plausibility–compute frontier over fixed, single-axis, and factorized baselines on NovPhy and cross-domain physics tasks.

---

## 9. One-Sentence Pitch

> **BG-NS-JEPA is the learned analogue of a hybrid kinetic–fluid solver: it estimates the local degree of scale separation and jointly selects how far ahead to predict and at which level of description — continuous, relational, or event-level.**

---

## 10. References

**World models and representation learning**

- LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels, arXiv:2603.19312.
- V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning, arXiv:2506.09985.
- ThinkJEPA: Empowering Latent World Models with Large Vision-Language Reasoning Model, arXiv:2603.22281.
- Causal-JEPA: Learning World Models through Object-Level Latent Masking, arXiv:2602.11389.
- Sub-JEPA: Subspace Gaussian Regularization for Stable End-to-End World Models, arXiv:2605.09241.
- STRIPS-WM: Learning Grounded Propositional STRIPS-style World Models from Images, arXiv:2606.06832.

**Perception and symbol extraction**

- Carion et al., End-to-End Object Detection with Transformers (DETR), ECCV 2020.
- Xu et al., How Powerful are Graph Neural Networks?, ICLR 2019.

**Temporal abstraction and planning**

- TempoRL: Learning When to Act, arXiv:2106.05262.
- Sutton, Precup, and Singh, Between MDPs and Semi-MDPs: A Framework for Temporal Abstraction in Reinforcement Learning, AIJ 1999.
- Fox and Long, Modelling Mixed Discrete-Continuous Domains for Planning, JAIR 2006.
- Playing Angry Birds with a Domain-Independent PDDL+ Planner, arXiv:2107.04635.

**Benchmarks and tools**

- NovPhy: A physical reasoning benchmark for open-world AI systems.
- DeepPHY: Benchmarking Agentic VLMs on Physical Reasoning, arXiv:2508.05405.

**Multiscale and kinetic-theory lineage (conceptual grounding)**

- Chapman and Cowling, *The Mathematical Theory of Non-Uniform Gases*, Cambridge University Press.
- Grad, On the Kinetic Theory of Rarefied Gases, *Communications on Pure and Applied Mathematics*, 1949.
- Levermore, Moment Closure Hierarchies for Kinetic Theories, *Journal of Statistical Physics*, 1996.
- Garcia, Bell, Crutchfield, and Alder, Adaptive Mesh and Algorithm Refinement Using Direct Simulation Monte Carlo, *Journal of Computational Physics*, 1999.
- E and Engquist, The Heterogeneous Multiscale Methods, *Communications in Mathematical Sciences*, 2003.
- Kevrekidis et al., Equation-Free, Coarse-Grained Multiscale Computation, *Communications in Mathematical Sciences*, 2003.
- Zwanzig, Memory Effects in Irreversible Thermodynamics, *Physical Review*, 1961; Mori, Transport, Collective Motion, and Brownian Motion, *Progress of Theoretical Physics*, 1965.
- Deng, Hani, and Ma, Long-time derivation of the Boltzmann equation from hard-sphere dynamics, arXiv:2408.07818 (cited only as evidence that the micro–meso–macro hierarchy remains an active mathematical frontier; no technical connection is claimed).

---

Positioning note: the paper's novelty is the joint controller and its causal evidence, now grounded in a generative framework — the micro–meso–macro hierarchy generates the component list rather than merely licensing the combination. Present SPSG as an enabling relational inductive bias and evaluate it separately from the controller. The kinetic-theory connection is motivational and algorithmic: every borrowed concept (Knudsen-like scale separation, restriction/lifting, closure consistency, equilibria) corresponds to a concrete, ablatable design decision; no rigorous derivation is claimed. Two design commitments sharpen the defense: (i) the state-carrier principle — $z$ is the only rollout state, symbolic heads are readouts and constraints — which collapses questions about gradient flow, teacher forcing, and symbolic error propagation into one answer; and (ii) the training-time-oracle → test-time-amortized structure, which makes the oracle-gate ablation the ceiling of a deployable learned controller rather than an embarrassment to hide.
