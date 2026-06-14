# Bi-Granular Neuro-Symbolic JEPA: Adaptive Temporal and Physical Abstraction for Persistent Physical Cascades

**Target Venue:** ICLR 2027  
**Keywords:** JEPA, world models, neuro-symbolic learning, adaptive granularity, physical reasoning, scene graph, energy-based constraints, Angry Birds, NovPhy.

---

## 1. Abstract

A single action in Angry Birds triggers a physical cascade lasting 50–150 frames. During this cascade, the agent acts only once; the world evolves autonomously. Standard JEPA world models predict $z_{t+1} = f(z_t, a_t)$. When $a_t = \text{noop}$ for hundreds of steps, recursive latent rollout drifts into physically impossible states—objects penetrate, unsupported blocks float, causal chains break.

We argue that this failure is not merely long-horizon error accumulation. It is a **structural granularity mismatch**: JEPA assumes uniform temporal resolution (frame-by-frame) and uniform physical abstraction (pixel-level) across the entire episode. But physical cascades are heterogeneous in time and structure. Collision phases demand fine-grained prediction; inertial glides permit coarse event jumps. Stable configurations support object-level symbolic reasoning; chaotic collapse defies discrete relational description.

We propose **Bi-Granular Neuro-Symbolic JEPA (BG-NS-JEPA)**, a world model that jointly adapts **temporal resolution** ($\Delta t$) and **physical abstraction** ($\alpha \in \{\text{micro}, \text{macro}, \text{none}\}$) based on the dynamical regime of the environment. The model uses pure neural JEPA during chaotic transients, but anchors to a **learned symbolic scene graph** at stable attractors. This converts PDDL from a planning formalism into a **learned temporal codec** for physical cascade compression.

Our core theoretical contribution is **Structured Physical Symbolic Geometry (SPSG)**: we replace crude text-rule encoders with scene-graph neural networks that treat symbols as **submanifolds in latent space**—hard geometric constraints rather than soft energy basins. We further ground the stability gate $\phi(x_t)$ in dynamical systems theory, interpreting it via the largest Lyapunov exponent.

The one-sentence pitch is:

> **BG-NS-JEPA treats temporal and physical granularity as jointly learnable control variables: it uses a microscope at collision moments, a telescope during inertial glides, and learns when to switch by monitoring the physical stability of the scene.**

---

## 2. Problem: The Granularity Mismatch in Action-Sparse Persistent-Effect Environments

### 2.1 Environment Definition

An episode is $\mathcal{E} = \{(o_t, a_t, x_t)\}_{t=0}^T$ with:

- **Action sparsity ratio**: $\rho_a = \frac{1}{T}\sum_{t=0}^{T-1} \mathbf{1}[a_t \neq \text{noop}] \ll 1$
- **Effect persistence**: $\tau_{\text{eff}} = \min\{k: \forall t > t_{\text{last}}+k, \|x_t - x_{t-1}\| < \epsilon\} \gg 1$

We term these **action-sparse persistent-effect environments**. They are not merely "long-horizon"; they are **action-sparse**, meaning the world evolves autonomously for most of the episode.

### 2.2 The Dual-Granularity Trade-off

**Temporal Axis (How often to predict):**

- **Fine $\Delta t$ (frame-by-frame)**: High fidelity, but error accumulates exponentially over $\tau_{\text{eff}}$.
- **Coarse $\Delta t$ (event-level)**: Low drift, but may miss critical collision events that determine final outcome.
- **Hypothesis**: There exists a **critical granularity** $\Delta t^*$ that maximizes both final-state symbolic accuracy and continuous prediction fidelity.

**Physical Axis (What to predict):**

- **Micro abstraction $\alpha = \text{micro}$**: Object-level scene graph (contact, support, velocity-bin). Valid when the system is quasi-static.
- **Macro abstraction $\alpha = \text{macro}$**: Structure-level predicates (collapsed, cascade-active, steady-state). Valid during smooth evolution or equilibrium.
- **Chaos regime $\alpha = \text{none}$**: Neither micro nor macro symbolic descriptions are valid. The system is in a high-entropy transient where splintering planks and flying debris defy discrete relational logic.

### 2.3 The Stability Gate: When Are Symbols Valid?

We define a **physical stability indicator** $\phi(x_t) \in \{0, 1\}$:

$$\phi(x_t) = \mathbb{1}_{\left[\text{KE}(x_t) < \epsilon_{\text{KE}}\right]} \cdot \mathbb{1}_{\left[\text{contacts}_{\text{active}}(x_t) < \epsilon_{\text{contact}}\right]}$$

When $\phi = 1$, the scene is in a **quasi-static attractor**—symbolic predicates like `supports(A,B)` or `all_pigs_dead` are well-defined. When $\phi = 0$, the system is in a chaotic transient where symbolic logic is **undefined**.

*Theoretical grounding*: $\phi(x_t)$ can be interpreted via the largest Lyapunov exponent $\lambda_{\max}$: $\phi(x_t) = \mathbb{1}_{[\lambda_{\max}(x_t) < 0]}$, connecting the engineering heuristic to dynamical systems theory. The chaotic collapse corresponds to $\lambda_{\max} > 0$ (expansion); the stable endpoints correspond to $\lambda_{\max} < 0$ (contraction).

---

## 3. Key Insight: Let the Neural Network Handle the Chaos, Anchor the Endpoints in Symbolic Logic

Existing neuro-symbolic JEPA approaches treat symbolic rules as flat textual objects encoded via generic text transformers. This introduces three limitations:

1. **Structural blindness**: Relational physical knowledge (support graphs, contact manifolds) is serialized into sequences, losing compositional semantics.
2. **Unstructured negativity**: Negative rules are constructed by random corruption, producing physically absurd yet semantically ambiguous samples that pollute the energy landscape.
3. **Point-wise embedding**: Symbols are treated as isolated attractor points in soft energy basins, rather than geometric constraints defining submanifolds.

Our solution is **Structured Physical Symbolic Geometry (SPSG)**. We treat symbols as **learnable projection heads** that factorize the latent space into explicit submanifolds:

$$\mathcal{M}_{\text{valid}} = \{z \in \mathcal{Z} \mid \forall s: f_s(z) = c_s\}$$

where $f_s: \mathcal{Z} \to \mathbb{R}$ decodes whether symbolic predicate $s$ holds. The conjunction of symbols corresponds to the intersection of submanifolds: $\mathcal{M}_{s_1 \land s_2} = \mathcal{M}_{s_1} \cap \mathcal{M}_{s_2}$. This gives the latent space an **algebraic structure** compatible with logical operations.

Critically, we accept that **symbols are only valid at stable attractors**. During the chaotic collapse ($\phi = 0$), the symbolic layer is **completely deactivated**. The neural network is free to predict continuous physics without symbolic "interference." This is not a limitation but a design choice: we do not force discrete logic onto phenomena that are inherently continuous and chaotic.

---

## 4. Method: Bi-Granular Neuro-Symbolic JEPA

All components serve a single objective:

$$\boxed{\mathcal{L}_{\text{total}}(\Delta t, \alpha) = \underbrace{\mathcal{L}_{\text{JEPA}}(\Delta t)}_{\text{chaotic phase}} + \underbrace{\phi(x_t) \cdot \mathcal{L}_{\text{EBC}}(\alpha)}_{\text{stable phase}} + \underbrace{\mathcal{L}_{\text{anchor}}}_{\text{terminal state}}}$$

### 4.1 Component A: JEPA Backbone (The Chaos Engine)

Standard action-conditioned latent predictor:
$$\hat{z}_{t+1} = F_\theta(z_t, a_t)$$

During chaotic phases ($\phi = 0$), this operates **unconstrained**. The neural network predicts continuous physics—splintering planks, flying debris, rolling pigs—without symbolic correction.

### 4.2 Component B: Structured Physical Symbolic Geometry (SPSG)

**Scene Graph Encoding**:

- **Nodes**: Objects (bird, pig, wood_block, TNT) with features $[type, material, position, velocity, shape, health]$.
- **Edges**: Directed physical relations with features $[relation\_type, relative\_position, distance, contact\_normal, support\_strength]$.
- **Encoder**: GINEConv (Graph Isomorphism Network with Edge features), preserving permutation invariance, compositionality, and edge semantics.

The encoder maps a scene graph $G_t$ to a symbolic embedding:
$$z_{t,\text{sym}} = f_{\text{GINE}}(G_t)$$

**Negative Sampling via Differentiable Physics**:
Instead of random corruption, we use the physics engine to generate **impossible worlds**:

- Anti-gravity: $g = +9.8$
- Massless materials: $\rho_{\text{wood}} = 0$
- Anti-support: remove support while keeping structure upright

These are validated through the engine; if the engine crashes or cannot simulate, the configuration is a **hard negative**. This ensures semantic coherence in the energy landscape.

### 4.3 Component C: Dual-Resolution PDDL as Temporal Codec

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

**Temporal Codec**: Instead of storing every frame, the model compresses a cascade into an event sequence:
$$\mathcal{C} = \{(\tau_i, S_{\tau_i}^M, e_i, \Delta_i, S_{\tau_{i+1}}^M)\}_{i=1}^K$$

Prediction becomes: $(S_{\tau_{i+1}}^M, \Delta_i, e_i) = G_\omega(S_{\tau_i}^M, z_{\tau_i}, a_{\tau_i})$

The continuous JEPA backbone only fills in micro-dynamics inside event intervals:
$$z_{t+1} = F_\theta(z_t, a_t, S_t^\mu) \quad \text{for } t \in [\tau_i, \tau_i + \delta_i]$$

### 4.4 Component D: Bi-Granular Controller

The controller jointly selects temporal resolution $\Delta t_k$ and physical abstraction $\alpha_k$:

$$(\Delta t_k, \alpha_k) = \pi_\kappa(u_t, \phi(x_t), p_\eta(E_t), \lambda_t^{(w)})$$

where:

- $u_t$: JEPA predictor uncertainty (latent variance)
- $\phi(x_t)$: Physical stability indicator (Lyapunov-based)
- $p_\eta(E_t)$: Predicted event probability
- $\lambda_t^{(w)}$: Local event density

**Decision regimes**:

| Regime                 | $\phi$ | $u_t$  | $\lambda_t$ | $\Delta t$ | $\alpha$    |
| ---------------------- | ------ | ------ | ----------- | ---------- | ----------- |
| Pre-shot stable        | 1      | low    | low         | coarse     | macro       |
| Collision onset        | 0      | high   | high        | fine       | **none**    |
| Chaotic collapse       | 0      | high   | high        | fine       | **none**    |
| Post-collapse settling | 1→0→1  | medium | medium      | adaptive   | micro→macro |
| Equilibrium            | 1      | low    | low         | coarse     | macro       |

**Key design**: During chaotic collapse, $\alpha = \text{none}$—the symbolic layer is **completely deactivated**. This is the core innovation: we do not attempt to symbolize the unsymbolizable.

### 4.5 Component E: Phase-Gated EBC and Terminal Anchor

**Phase-Gated EBC** (only active when $\phi = 1$):
$$\mathcal{L}_{\text{EBC}}(\Delta t) = \sum_{k \in \mathcal{K}_{\text{stable}}} \left[ \sum_{(A,C) \in \mathcal{R}_{\text{valid}}} E(A,C) + \lambda \sum_{(A,C_{\text{neg}}) \in \mathcal{R}_{\text{neg}}} \max(0, m - E(A, C_{\text{neg}})) \right]$$

where $\mathcal{K}_{\text{stable}} = \{k : \phi(x_{k\Delta t}) = 1\}$.

**Terminal Anchor Loss** (forces final state to symbolic pole):
$$\mathcal{L}_{\text{anchor}} = \mathbb{1}_{[\phi(x_T)=1]} \cdot \sum_{c \in \mathcal{C}} \mathbb{1}_{[y=c]} \left\| g(z_c^{(K-1)}) - z_{\text{pole}}^{(c)} \right\|_2^2$$

This ensures that even if the neural network hallucinated the trajectory of every splinter, the **final latent state** must collapse onto the symbolic pole of the true outcome.

---

## 5. Theoretical Contributions

### 5.1 From Energy Basins to Geometric Constraints

Prior neuro-symbolic work treats symbols as points in low-energy basins. We upgrade this to **submanifold constraints**:

$$\mathcal{M}_s = \{z \in \mathcal{Z} : f_s(z) = c_s\}$$

This converts "make the energy lower" into a **hard geometric constraint** on the latent manifold.

### 5.2 Lyapunov Interpretation of the Stability Gate

The stability indicator $\phi(x_t)$ can be grounded in dynamical systems theory:

- $\phi = 1$ corresponds to $\lambda_{\max} < 0$ (contracting/stable manifold)
- $\phi = 0$ corresponds to $\lambda_{\max} > 0$ (chaotic expansion)

This connects the engineering heuristic to rigorous mathematics.

### 5.3 Granularity Scaling Law

We hypothesize a **Pareto frontier** between temporal granularity and predictive performance:

| $\Delta t$            | Symbolic Accuracy $\mathcal{A}_{\text{sym}}$ | Frame MSE    | Error Accumulation |
| --------------------- | -------------------------------------------- | ------------ | ------------------ |
| Very coarse (T/2)     | High                                         | High         | Low                |
| Critical $\Delta t^*$ | **Maximal**                                  | **Balanced** | **Sub-linear**     |
| Very fine (1 frame)   | Low (drift)                                  | Low (early)  | **Exponential**    |

**Hypothesis**: There exists a critical granularity $\Delta t^*$ where symbolic accuracy and continuous fidelity are jointly optimized. BG-NS-JEPA learns to operate near this critical point by adapting $\Delta t$ dynamically.

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

### 6.3 Stage 3: Granularity Ablation (Weeks 9-11)

Vary $\Delta t \in \{2, 5, 10, 30, \text{full FPS}\}$ and measure:

- Terminal symbolic accuracy (zero-shot via pole distance)
- Continuous prediction MSE
- Physical violation rate (penetration, floating, illegal contact)
- OOD generalization (novel materials, gravity)

**Expected signature curve**: U-shaped symbolic accuracy vs. $\Delta t$, with maximum at $\Delta t^*$.

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
| BG-NS-JEPA (no phase gate) | Ablates stability gating             |
| BG-NS-JEPA (no SPSG)       | Ablates structured symbolic geometry |
| BG-NS-JEPA (oracle graph)  | Upper bound for symbol extraction    |

### 6.6 Metrics

**Prediction metrics**: ADE@H, FDE@H, Final-state accuracy, Event prediction F1
**Physical plausibility**: Object penetration rate, unsupported-floating rate, illegal-contact rate
**Granularity metrics**: Percentage in micro/macro/none mode, effective prediction steps, switch precision/recall
**Planning metrics**: Task success, shots-to-success, novelty adaptation speed

---

## 7. Risk and Mitigation

| Risk                                           | Mitigation                                                   |
| ---------------------------------------------- | ------------------------------------------------------------ |
| Learned symbolic extraction unstable           | Start with oracle; GINE is more stable than slot attention   |
| "Hand-engineered" vocabulary critique          | Vocabulary is universal physical substrate (like CNN locality); all truth values are learned |
| NovPhy is adaptation benchmark, not prediction | Add dedicated prediction track; use world model as adaptation engine |
| Chaos phase has no symbolic supervision        | By design. Chaos uses pure JEPA; symbols only anchor endpoints |
| Method too complex for 3-4 months              | Oracle-symbol version is minimum publishable unit            |

---

## 8. Contributions

1. **Problem**: Defines action-sparse persistent-effect environments as a stress test for JEPA, and identifies **granularity mismatch** as the root cause of failure.
2. **Conceptual**: Proposes **Bi-Granular** control—jointly adapting temporal resolution and physical abstraction based on dynamical regime.
3. **Method**: BG-NS-JEPA with Phase-Gated EBC, SPSG (GINE scene graphs), and PDDL temporal codec.
4. **Theoretical**: Symbols-as-submanifolds; Lyapunov grounding for stability gate; granularity scaling law.
5. **Empirical**: NovPhy/Angry Birds evaluation protocol with cross-domain validation (Physhion, CLEVRER).

---

## 9. One-Sentence Pitch

> **BG-NS-JEPA lets world models think like physicists: zoom in with neural prediction during chaotic collisions, zoom out with symbolic abstraction during stable glides, and learn the zoom schedule from the physics itself.**

---

## 10. References

- LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels, arXiv:2603.19312.
- V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning, arXiv:2506.09985.
- ThinkJEPA: Empowering Latent World Models with Large Vision-Language Reasoning Model, arXiv:2603.22281.
- Causal-JEPA: Learning World Models through Object-Level Latent Masking, arXiv:2602.11389.
- Sub-JEPA: Subspace Gaussian Regularization for Stable End-to-End World Models, arXiv:2605.09241.
- STRIPS-WM: Learning Grounded Propositional STRIPS-style World Models from Images, arXiv:2606.06832.
- Fox and Long, Modelling Mixed Discrete-Continuous Domains for Planning, JAIR 2006.
- Playing Angry Birds with a Domain-Independent PDDL+ Planner, arXiv:2107.04635.
- NovPhy: A physical reasoning benchmark for open-world AI systems.
- DeepPHY: Benchmarking Agentic VLMs on Physical Reasoning, arXiv:2508.05405.
- Xu et al., How Powerful are Graph Neural Networks?, ICLR 2019.

---

这份提案的核心改进在于：

1. **统一公式统领全文**：所有组件都是 $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{JEPA}} + \phi \cdot \mathcal{L}_{\text{EBC}} + \mathcal{L}_{\text{anchor}}$ 的实现细节，而非并列的五个模块。
2. **"混沌期关闭"作为设计哲学**：明确声明 $\alpha = \text{none}$ 不是缺陷而是核心设计，直接防御 Bitter Lesson 质疑。
3. **SPSG 理论升级**：将符号从"能量盆地中的点"提升为"子流形上的硬约束"，给出代数结构（交集=合取）。
4. **Lyapunov 解释**：给 $\phi(x_t)$ 动力系统理论基础，而非停留在工程启发式。
5. **Granularity Scaling Law**：提供实验钩子——U型曲线 + 临界粒度 $\Delta t^*$，这是 ICLR 评审容易记住的"杀手图"。

