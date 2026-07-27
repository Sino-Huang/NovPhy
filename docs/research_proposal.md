# Regime-Adaptive World Models: Joint Temporal and Symbolic Abstraction for Persistent Physical Cascades

**Target Venue:** ICLR 2027  
**Keywords:** world models, JEPA, neuro-symbolic learning, adaptive abstraction, temporal abstraction, physical reasoning, scene graphs, Angry Birds, NovPhy.

---

## 1. Abstract

A single action in Angry Birds can trigger a physical cascade lasting 50–150 frames while the agent does not act again. Existing JEPA world models normally predict on a fixed temporal clock with a fixed latent abstraction. In this setting, short strides compound rollout error while long strides can skip collision events that determine the outcome; a fixed object-level symbolic description can likewise be unreliable while contact topology changes rapidly.

We identify this as a **representation-control problem**, not only a long-horizon prediction problem. Persistent physical cascades are heterogeneous: interaction-active intervals require fine continuous prediction, while quiescent intervals and stable endpoints admit compact relational abstractions. Existing world models learn representations or improve training stability; neuro-symbolic world models learn symbolic states or operators. Neither treats prediction horizon and symbolic abstraction as **joint, state-dependent decisions**.

We propose **Bi-Granular Neuro-Symbolic JEPA (BG-NS-JEPA)**, a regime-adaptive world model that selects temporal resolution $\Delta t$ and representational abstraction $\alpha \in \{\text{continuous}, \text{micro-relational}, \text{macro-event}\}$ from the current predictive state. A learned symbolic-reliability estimator gates fine relational constraints, allowing continuous dynamics to dominate when contact/support predicates are unreliable while retaining event-level descriptions when they remain useful. The symbolic layer supplies compositional state and cross-level consistency; it is not a universal rule engine.

Our secondary representational contribution, **Structured Physical Symbolic Geometry (SPSG)**, uses scene-graph and tensor-product structure to regularize relational states. We treat the resulting geometry as a learned soft constraint, with optional inference-time projection, rather than claiming an unconditional hard manifold or a dynamical-systems equivalence.

The one-sentence pitch is:

> **BG-NS-JEPA lets a world model change both its prediction horizon and its symbolic resolution when the physical regime changes.**

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

- **Continuous $\alpha = \text{continuous}$**: A neural latent for rapidly changing contact and motion.
- **Micro-relational $\alpha = \text{micro}$**: Object-level graph predicates (contact, support, velocity-bin), used only when their reliability is high.
- **Macro-event $\alpha = \text{macro}$**: Structure/event predicates (cascade-active, collapsed, pigs-cleared, steady-state), which can remain useful even while individual contacts change.

### 2.3 Symbolic Reliability: When Is a Relational Constraint Useful?

We use a **symbolic-reliability estimator** $r_\psi(h_t) \in [0, 1]$, where $h_t$ contains the latent state, predictive uncertainty, and event likelihood. An oracle label can be constructed from simulator features during training:

$$\phi^*(x_t) = \mathbb{1}_{\left[\text{KE}(x_t) < \epsilon_{\text{KE}}\right]} \cdot \mathbb{1}_{\left[\text{contacts}_{\text{active}}(x_t) < \epsilon_{\text{contact}}\right]}.$$

High reliability means that the selected fine relational ontology, for example `supports(A,B)`, is expected to be stable enough to constrain a rollout. Low reliability does not make all symbols meaningless: coarse event predicates may still be available. Kinetic energy, contact activity, and finite-time divergence are diagnostics for this estimator, not definitions equivalent to a largest Lyapunov exponent.

---

## 3. Research Gap: World Models Do Not Control Their Representation

The nearest research threads leave a specific gap:

1. **JEPA world models** improve representation learning, object structure, or optimization stability, but normally retain a fixed temporal sampling policy and prediction representation.
2. **Temporal-abstraction methods** learn when an agent acts, rather than when a world model should advance time and change state abstraction.
3. **Symbolic world models** induce a fixed abstract transition system; they do not model a continuous cascade in which the reliability of object-level relations itself changes.

Our response is **joint regime-adaptive representation control**. **Structured Physical Symbolic Geometry (SPSG)** is the relational mechanism within that controller, not the paper's primary novelty. It uses learnable predicate projections to regularize a latent toward relationally consistent regions:

$$\mathcal{M}_{\text{valid}} = \{z \in \mathcal{Z} \mid \forall s: f_s(z) = c_s\}$$

where $f_s: \mathcal{Z} \to \mathbb{R}$ decodes whether symbolic predicate $s$ holds. The conjunction of symbols corresponds to the intersection of submanifolds: $\mathcal{M}_{s_1 \land s_2} = \mathcal{M}_{s_1} \cap \mathcal{M}_{s_2}$. This supplies an algebraic inductive bias compatible with conjunction when the predicate maps are sufficiently well behaved; the implementation enforces it with losses and optional projection rather than an unconditional hard constraint.

During interaction-active intervals, the controller may select a continuous latent and mask unreliable fine relational losses. At stable or slowly changing intervals, it can select micro-relational or macro-event states and enforce cross-level consistency. The novelty is the state-dependent selection of this pair, not a claim that one representation is universally correct.

---

## 4. Method: Joint Regime-Adaptive Abstraction

At each decision point, the controller chooses a pair from the cross-product of temporal and representational choices:

$$t_{k+1}=\min(t_k+\Delta_k,T), \qquad (\Delta_k,\alpha_k) \sim \pi_\kappa(\cdot\mid h_{t_k}),$$

where $\Delta_k \in \mathcal{D}$ is a prediction horizon, $\alpha_k \in \{\text{continuous},\text{micro},\text{macro}\}$ is an abstraction, and $h_t$ includes the current latent, uncertainty, event likelihood, and symbolic reliability. The controller is joint rather than factorized: it must learn, for example, that a short continuous prediction is useful at collision onset while a long macro-event transition can be appropriate after settling.

All components serve one duration-normalized objective:

$$\boxed{\mathcal{L}_{\text{total}} = \mathbb{E}\!\left[\sum_k \frac{\Delta_k}{T}\left(\mathcal{L}_{\text{pred}}^{\Delta_k,\alpha_k} + \lambda_{\text{sym}}\omega_\psi(h_{t_k},\alpha_k)\mathcal{L}_{\text{sym}}^{\alpha_k} + \lambda_{\text{cons}}\mathcal{L}_{\text{cross}} + \lambda_{\text{cost}}c(\Delta_k,\alpha_k)\right) + \lambda_{\text{anchor}}\mathcal{L}_{\text{anchor}}\right].}$$

$\mathcal{L}_{\text{sym}}$ collects semantic, structural, and contrastive terms. $\omega_\psi(h,\alpha)$ is $r_\psi(h)$ for micro-relational constraints, $1$ for selected macro-event supervision, and $0$ for a continuous step. $\mathcal{L}_{\text{cross}}$ enforces agreement between micro and macro descriptions; $c$ prevents the controller from selecting a degenerate always-fine solution.

### 4.1 Component A: Continuous Dynamics Backbone

Action-conditioned, horizon-indexed latent predictor:
$$\hat{z}_{t_{k+1}} = F_\theta^{\Delta_k,\alpha_k}(z_{t_k}, a_{[t_k,t_{k+1})}).$$

When the controller selects the continuous abstraction, it predicts rapidly changing physics without fine relational correction. It is not assumed that all event-level semantics disappear in this regime.

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

**Micro-PDDL** (object-centric, fine physical granularity):

```lisp
(contact block_1 block_2)
(supports block_1 pig_1)
(velocity-bin block_3 high)
(damaged wood_block_7)
```

**Macro-PDDL** (structure-centric, coarse physical granularity):

```lisp
(structure-unstable tower_1)
(cascade-active tower_1)
(collapsed tower_1)
(pigs-cleared left-region)
(steady-state level)
```

**Abstraction map**: $S_t^M = A(S_t^\mu)$
**Refinement map**: $\tilde{S}_t^\mu = R(S_t^M, z_t)$

The macro state provides an event interface that can summarize a cascade:
$$\mathcal{C} = \{(\tau_i, S_{\tau_i}^M, e_i, \Delta_i, S_{\tau_{i+1}}^M)\}_{i=1}^K$$

Prediction becomes: $(S_{\tau_{i+1}}^M, \Delta_i, e_i) = G_\omega(S_{\tau_i}^M, z_{\tau_i}, a_{\tau_i})$.

The continuous JEPA backbone fills in micro-dynamics inside selected event intervals:
$$z_{t+1} = F_\theta(z_t, a_t, S_t^\mu) \quad \text{for } t \in [\tau_i, \tau_i + \delta_i]$$

PDDL is an optional serialization for a downstream planner; it is not itself claimed to be a learned temporal codec.

### 4.4 Component D: Joint Bi-Granular Controller

The controller jointly selects temporal resolution $\Delta t_k$ and physical abstraction $\alpha_k$:

$$(\Delta t_k, \alpha_k) = \pi_\kappa(u_t, r_\psi(h_t), p_\eta(E_t), \lambda_t^{(w)}).$$

where:

- $u_t$: JEPA predictor uncertainty (latent variance)
- $r_\psi(h_t)$: learned reliability of fine relational constraints
- $p_\eta(E_t)$: Predicted event probability
- $\lambda_t^{(w)}$: Local event density

**Decision regimes**:

| Regime                 | $r_\psi$ | $u_t$  | $\lambda_t$ | $\Delta t$ | $\alpha$    |
| ---------------------- | ------ | ------ | ----------- | ---------- | ----------- |
| Pre-shot stable        | 1      | low    | low         | coarse     | macro       |
| Collision onset        | low    | high   | high        | fine       | continuous  |
| Interaction-active collapse | low | high | high        | fine       | continuous  |
| Post-collapse settling | 1→0→1  | medium | medium      | adaptive   | micro→macro |
| Equilibrium            | 1      | low    | low         | coarse     | macro       |

**Key design**: the controller selects the pair $(\Delta t,\alpha)$ jointly. Fine relational losses are masked when reliability is low, while a macro event representation may still be selected if it predicts the outcome more accurately or cheaply.

### 4.5 Component E: Reliability-Gated Constraints and Terminal Anchor

**Reliability-gated constraint**:
$$\ell_{\text{sym},k}=\sum_{(A,C)\in\mathcal{R}_{\text{valid}}}E(A,C)+\lambda\sum_{(A,C_{\text{neg}})\in\mathcal{R}_{\text{neg}}}\max(0,m-E(A,C_{\text{neg}})).$$

The total objective weights $\ell_{\text{sym},k}$ with $\omega_\psi(h_{t_k},\alpha_k)$, so only the constraint type selected by the controller receives supervision.

**Terminal anchor loss** (encourages an outcome-consistent endpoint):
$$\mathcal{L}_{\text{anchor}} = r_\psi(h_T) \cdot \sum_{c \in \mathcal{C}} \mathbb{1}_{[y=c]} \left\| g(z_c^{(K-1)}) - z_{\text{pole}}^{(c)} \right\|_2^2.$$

This applies outcome-level pressure at a reliable terminal state; it does not by itself guarantee a correct rollout.

---

## 5. Theoretical Contributions

### 5.1 Structured Relational Regularization

SPSG treats predicate projections as structured regularizers:

$$\mathcal{M}_s = \{z \in \mathcal{Z} : f_s(z) = c_s\}$$

If $c_s$ is a regular value of $f_s$, then $f_s^{-1}(c_s)$ is locally a submanifold. The implemented losses only encourage proximity to this set; a hard constraint requires an explicit projection or constrained optimizer. This distinction keeps the geometry claim precise.

### 5.2 Reliability-Aware Switching

The theory target is not a Lyapunov equivalence. It is a switching claim: when different regimes prefer different $(\Delta,\alpha)$ pairs, a sufficiently accurate state-dependent controller can outperform every fixed pair, subject to its prediction and compute costs. Simulator-derived stability measures provide supervision and diagnostics for the reliability estimator.

### 5.3 Joint Granularity Frontier

We evaluate a **Pareto frontier** over temporal and representational granularity:

| Controller choice | Endpoint correctness | Local fidelity | Compute |
| ----------------- | -------------------- | -------------- | ------- |
| Fixed coarse macro | May miss interactions | Low | Low |
| Fixed fine continuous | High locally, may drift | High | High |
| Joint regime-adaptive | **Hypothesized best trade-off** | Regime-dependent | Adaptive |

**Hypothesis**: A joint controller beats fixed and factorized choices at matched compute because physical regimes induce different preferred pairs, not merely different temporal strides.

---

## 6. Experimental Methodology

### 6.1 Stage 1: Oracle Symbol Upper Bound (Weeks 1-4)

Use ground-truth scene graphs from NovPhy/Box2D. Research question:

> If symbolic graphs are perfect, can structured symbolic geometry improve JEPA endpoint prediction and OOD robustness?

This de-risks the project before solving perception.

### 6.2 Stage 2: Learned Symbolic State (Weeks 5-8)

Train GINE encoder to decode latent states into scene graphs. Compare:

- Oracle symbolic state
- Learned symbolic state from simulator features
- Learned symbolic state from image latent only

### 6.3 Stage 3: Joint-Controller Ablation (Weeks 9-11)

Compare fixed pairs, temporal-only adaptation, abstraction-only adaptation, a factorized controller $\pi_\Delta\pi_\alpha$, and the joint controller. Measure:

- Terminal symbolic accuracy (zero-shot via pole distance)
- Horizon-normalized continuous prediction error at matched compute
- Physical violation rate (penetration, floating, illegal contact)
- OOD generalization (novel materials, gravity)

**Expected signature**: the joint controller improves the endpoint-correctness/physical-plausibility/compute frontier and chooses regime-aligned pairs more accurately than factorized alternatives.

### 6.4 Stage 4: Full BG-NS-JEPA (Weeks 12-14)

End-to-end training with bi-granular controller. Evaluate on:

- NovPhy standard and novelty scenarios
- Cross-domain: Physhion (8 scenarios), CLEVRER (collision reasoning)

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
| BG-NS-JEPA (oracle reliability) | Upper bound for the learned gate |
| BG-NS-JEPA (no reliability gate) | Applies fine relational constraints uniformly |
| BG-NS-JEPA (no SPSG)       | Ablates structured symbolic geometry |
| BG-NS-JEPA (oracle graph)  | Upper bound for symbol extraction    |

### 6.6 Metrics

**Prediction metrics**: ADE@H, FDE@H, Final-state accuracy, Event prediction F1
**Physical plausibility**: Object penetration rate, unsupported-floating rate, illegal-contact rate
**Granularity metrics**: effective prediction steps, controller cost, switch precision/recall against oracle regimes, and joint-pair calibration
**Planning metrics**: Task success, shots-to-success, novelty adaptation speed

---

## 7. Risk and Mitigation

| Risk                                           | Mitigation                                                   |
| ---------------------------------------------- | ------------------------------------------------------------ |
| Learned symbolic extraction unstable           | Start with oracle; GINE is more stable than slot attention   |
| "Hand-engineered" vocabulary critique          | Vocabulary is universal physical substrate (like CNN locality); all truth values are learned |
| NovPhy is adaptation benchmark, not prediction | Add dedicated prediction track; use world model as adaptation engine |
| Reliability estimator misclassifies a regime   | Report oracle-gate upper bound; train with noisy labels and use calibrated confidence |
| Reviewer sees a component bundle               | Make joint-controller and factorization ablations the primary evidence |
| Method too complex for 3-4 months              | Oracle-symbol version is minimum publishable unit            |

---

## 8. Contributions

1. **Problem**: Defines action-sparse persistent-effect environments as a world-model setting with nonuniform temporal and representational demands.
2. **Conceptual**: Identifies **joint regime-adaptive representation control** as the missing principle: a world model selects prediction horizon and abstraction together.
3. **Method**: BG-NS-JEPA, combining a joint controller, a learned symbolic-reliability estimator, cross-level relational consistency, and a continuous JEPA backbone.
4. **Representation**: SPSG supplies compositional scene-graph and TPR regularization when fine relational predicates are reliable.
5. **Empirical**: Tests whether the joint controller improves the prediction–plausibility–compute frontier over fixed, single-axis, and factorized baselines on NovPhy and cross-domain physics tasks.

---

## 9. One-Sentence Pitch

> **BG-NS-JEPA lets a world model change both its prediction horizon and its symbolic resolution when the physical regime changes.**

---

## 10. References

- LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels, arXiv:2603.19312.
- V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning, arXiv:2506.09985.
- ThinkJEPA: Empowering Latent World Models with Large Vision-Language Reasoning Model, arXiv:2603.22281.
- Causal-JEPA: Learning World Models through Object-Level Latent Masking, arXiv:2602.11389.
- Sub-JEPA: Subspace Gaussian Regularization for Stable End-to-End World Models, arXiv:2605.09241.
- STRIPS-WM: Learning Grounded Propositional STRIPS-style World Models from Images, arXiv:2606.06832.
- TempoRL: Learning When to Act, arXiv:2106.05262.
- Sutton, Precup, and Singh, Between MDPs and Semi-MDPs: A Framework for Temporal Abstraction in Reinforcement Learning, AIJ 1999.
- Fox and Long, Modelling Mixed Discrete-Continuous Domains for Planning, JAIR 2006.
- Playing Angry Birds with a Domain-Independent PDDL+ Planner, arXiv:2107.04635.
- NovPhy: A physical reasoning benchmark for open-world AI systems.
- DeepPHY: Benchmarking Agentic VLMs on Physical Reasoning, arXiv:2508.05405.
- Xu et al., How Powerful are Graph Neural Networks?, ICLR 2019.

---

Positioning note: the paper's novelty is the joint controller and its causal evidence, not a claim that symbolic constraints, TPRs, or temporal abstraction are individually new. Present SPSG as an enabling relational inductive bias and evaluate it separately from the controller.
