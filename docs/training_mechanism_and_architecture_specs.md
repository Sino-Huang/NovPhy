# BG-NS-JEPA: Regime-Adaptive Training and Architectural Specification

**Joint temporal and symbolic abstraction with semantic tensor-product geometry**

*Technical Document — June 2026*

---

## 1. Overview

This document specifies the complete training mechanism for BG-NS-JEPA, integrating the architectural refinements discussed:
- **Semantic predicate embeddings** (CLIP-like similarity) over binary scalars
- **Tensor Product Representations (TPR)** for strict compositional binding
- **Hybrid GINE-TPR architecture** combining relational perception with algebraic structure
- **A joint controller** over prediction horizon and representational abstraction
- **Reliability-gated relational supervision** rather than universally active symbols
- **Optional inference-time relational projection** when fine relations are reliable

---

## 2. Problem Recap

### 2.1 Environment
An episode $\mathcal{E} = \{(o_t, a_t, x_t)\}_{t=0}^T$ where:
- **Action sparsity**: $\rho_a = \frac{1}{T}\sum_{t=0}^{T-1} \mathbf{1}[a_t \neq \text{noop}] \ll 1$
- **Effect persistence**: $\tau_{\text{eff}} = \min\{k: \forall t > t_{\text{last}}+k, \|x_t - x_{t-1}\| < \epsilon\} \gg 1$

### 2.2 Failure Mode
Standard JEPA $\hat{z}_{t+1} = F_\theta(z_t, a_t)$ with $a_t = \text{noop}$ for extended periods causes recursive latent rollout drift into physically impossible states.

### 2.3 Core Insight
Physical cascades are nonuniform in both time and representation. The model must jointly decide how far to predict and whether a continuous, micro-relational, or macro-event state is appropriate. Fine contact/support relations are reliability-gated; coarse event descriptions can remain useful during interaction-active intervals.

---

## 3. Architectural Components

### 3.1 Continuous Dynamics Backbone
Horizon- and abstraction-conditioned predictor:
$$\hat{z}_{t_{k+1}} = F_\theta^{\Delta_k,\alpha_k}(z_{t_k},a_{[t_k,t_{k+1})}).$$

**Active**: Always.
**Role**: Predicts continuous or abstract transitions selected by the controller.

---

### 3.2 GINE Scene Graph Encoder (Perception Backbone)
Handles variable-sized graphs with edge features:
- **Nodes**: Objects with features $[\text{type}, \text{material}, \text{position}, \text{velocity}, \text{shape}, \text{health}]$
- **Edges**: Directed physical relations with features $[\text{relation\_type}, \text{distance}, \text{contact\_normal}, \text{support\_strength}]$
- **Encoder**: GINEConv (Graph Isomorphism Network with Edge features)

**Output**: Primary latent state $z_t = f_{\text{GINE}}(G_t)$

**Active**: Always (perceives the current scene).

---

### 3.3 Tensor Product Representation (TPR) Head (Structural Regularizer)
A parallel readout head attached to GINE that computes explicit TPR targets:

**Role Vectors**: $\mathbf{r}_{o_i} \in \mathbb{R}^{d_r}$ for each object $o_i$
**Filler Vectors**: $\mathbf{v}_{s} \in \mathbb{R}^{d_v}$ for each predicate $s$

**TPR Scene State**:
$$z_{\text{TPR}} = \sum_{(o_i, o_j, s) \in \mathcal{G}} \mathbf{r}_{o_i} \otimes \mathbf{r}_{o_j} \otimes \mathbf{v}_{s}$$

Where $\otimes$ denotes the tensor product, producing a vector in $\mathbb{R}^{d_r \cdot d_r \cdot d_v}$.

**Semantic Predicate Embeddings**: Each predicate $s$ has a learned continuous embedding $\mathbf{v}_s$ (not a binary scalar). Similarity between predicates is computed as $\cos(\mathbf{v}_{s_1}, \mathbf{v}_{s_2})$.

**Active**: When the controller selects a relational abstraction and $r_\psi(h_t)$ is high enough to make fine relational constraints reliable.

---

### 3.4 Joint Bi-Granular Controller
Jointly selects temporal resolution $\Delta t_k$ and representational abstraction $\alpha_k$ from their cross-product:

$$(\Delta t_k, \alpha_k) \sim \pi_\kappa(\cdot \mid u_t, r_\psi(h_t), p_\eta(E_t), \lambda_t^{(w)}).$$

**Decision Regimes**:

| Regime                 | $r_\psi$ | $u_t$  | $\lambda_t$ | $\Delta t$ | $\alpha$    |
| ---------------------- | ------ | ------ | ----------- | ---------- | ----------- |
| Pre-shot stable        | 1      | low    | low         | coarse     | macro       |
| Collision onset        | low    | high   | high        | fine       | continuous  |
| Interaction-active collapse | low | high | high        | fine       | continuous  |
| Post-collapse settling | 1→0→1  | medium | medium      | adaptive   | micro→macro |
| Equilibrium            | 1      | low    | low         | coarse     | macro       |

**Active**: Always.

---

### 3.5 Symbolic-Reliability Estimator
$$r_\psi(h_t) \in [0,1], \qquad \phi^*(x_t)=\mathbb{1}_{[\mathrm{KE}(x_t)<\epsilon_{\mathrm{KE}}]}\mathbb{1}_{[\mathrm{contacts}_{\mathrm{active}}(x_t)<\epsilon_{\mathrm{contact}}]}.$$

$\phi^*$ is an oracle training label or diagnostic derived from simulator state. $r_\psi$ is the learned test-time estimator. It measures the expected reliability of fine relational constraints; it is not claimed to be equivalent to a Lyapunov exponent or to determine whether every possible symbol is meaningful.

---

## 4. Unified Training Loss

The complete duration-normalized objective is:

$$
\boxed{
\begin{aligned}
\mathcal{L}_{\text{total}} = \; &
\mathbb{E}\left[\sum_k\frac{\Delta_k}{T}\Big(
\underbrace{\mathcal{L}_{\text{pred}}^{\Delta_k,\alpha_k}}_{\text{(1) Prediction}} +
\lambda_{\text{sym}}\omega_\psi(h_{t_k},\alpha_k)\big[
\underbrace{\mathcal{L}_{\text{semantic}}}_{\text{(2) Semantic Alignment}} +
\underbrace{\mathcal{L}_{\text{struct}}}_{\text{(3) TPR Composition}} +
\underbrace{\mathcal{L}_{\text{contrastive}}}_{\text{(4) Physics-validated negatives}}
\big] \\
&\qquad + \lambda_{\text{cross}}\underbrace{\mathcal{L}_{\text{cross}}}_{\text{(5) Cross-level consistency}}
+ \lambda_{\text{cost}}\underbrace{c(\Delta_k,\alpha_k)}_{\text{(6) Controller cost}}
\Big) + \lambda_{\text{anchor}}\underbrace{\mathcal{L}_{\text{anchor}}}_{\text{(7) Terminal anchor}}\right].
\end{aligned}
}
$$

### 4.1 Component (1): Horizon-Conditioned Prediction Loss

$$\mathcal{L}_{\text{pred}}^{\Delta_k,\alpha_k} = \left\| \hat{z}_{t_{k+1}} - z_{t_{k+1}} \right\|_2^2, \qquad t_{k+1}=\min(t_k+\Delta_k,T).$$

Where $\hat{z}_{t_{k+1}}=F_\theta^{\Delta_k,\alpha_k}(z_{t_k},a_{[t_k,t_{k+1})})$.

The duration weighting makes fixed- and variable-horizon comparisons meaningful. Define $\omega_\psi(h,\alpha)$ as $r_\psi(h)$ for micro-relational constraints, $1$ for selected macro-event supervision, and $0$ for a continuous step. This masks unreliable fine relations without making all event-level supervision disappear.

---

### 4.2 Component (2): Reliability-Gated Semantic Alignment

For each ground-truth predicate $s \in \mathcal{S}_{\text{true}}$ active in the scene:

$$\mathcal{L}_{\text{semantic}} = \sum_{s \in \mathcal{S}_{\text{true}}} \| f_s(z_t) - \mathbf{v}_s \|_2^2$$

Where:
- $f_s: \mathcal{Z} \to \mathbb{R}^{d_v}$ is a learned projection head (MLP) decoding predicate $s$
- $\mathbf{v}_s \in \mathbb{R}^{d_v}$ is the learned semantic embedding for predicate $s$

**Why this matters**: This aligns the latent with a selected relational vocabulary when that vocabulary is reliable. Semantic similarity is an empirical property to test, not an automatic consequence of learning embedding vectors.

---

### 4.3 Component (3): TPR Compositional Regularizer

$$\mathcal{L}_{\text{struct}} = \left\| z_t - \sum_{(o_i, o_j, s) \in \mathcal{G}_t} \mathbf{r}_{o_i} \otimes \mathbf{r}_{o_j} \otimes \mathbf{v}_{s} \right\|_2^2$$

Where:
- $\mathcal{G}_t$ is the ground-truth scene graph at time $t$
- $\mathbf{r}_{o_i}$ are learned role embeddings for each object
- $\mathbf{v}_{s}$ are the same semantic predicate embeddings

**Why this matters**: The tensor product preserves role-filler binding, so "A supports B" and "B supports A" have different representations. The loss is a soft compositional regularizer, not a proof that arbitrary learned roles are orthogonal.

**Implementation Note**: In practice, we compute this loss using a **TPR Readout Head** attached to GINE, not by explicitly computing the full tensor product over all objects (which would be $O(N^3)$). The head predicts the role and filler vectors for each node/edge.

---

### 4.4 Component (4): Physics-Validated Contrastive Loss

Hard negatives are generated via differentiable physics:
- Reversed gravity under an explicit simulator coordinate convention
- Massless materials: $\rho_{\text{wood}} = 0$
- Anti-support: remove support while keeping structure upright

$$\mathcal{L}_{\text{contrastive}} = \sum_{\text{neg}} \max\left(0, m - \text{sim}(z_t, z_{\text{pos}}) + \text{sim}(z_t, z_{\text{neg}})\right)$$

Where:
- $\text{sim}(a, b) = \cos(a, b)$
- $z_{\text{pos}}$ is the latent of a physically valid configuration
- $z_{\text{neg}}$ is the latent of an impossible configuration
- $m$ is the margin hyperparameter

**Why this matters**: Retain a negative only when its intended physical or semantic violation is explicitly validated. A simulator crash is not evidence that a configuration is an invalid physical state.

---

### 4.5 Component (5): Cross-Level Consistency

For states where both descriptions are available, enforce that the macro state agrees with an abstraction of the micro state:

$$\mathcal{L}_{\text{cross}} = \left\|A(S_t^\mu)-S_t^M\right\|_2^2 + \left\|R(S_t^M,z_t)-S_t^\mu\right\|_2^2.$$

This is what makes temporal and representational granularity compose rather than merely coexist.

### 4.6 Component (7): Terminal Anchor Loss

$$\mathcal{L}_{\text{anchor}} = r_\psi(h_T) \cdot \sum_{c \in \mathcal{C}} \mathbb{1}_{[y=c]} \left\| g(z_c^{(K-1)}) - z_{\text{pole}}^{(c)} \right\|_2^2$$

Where:
- $z_T$ is the final latent state
- $z_{\text{pole}}^{(c)}$ is the learned symbolic pole for outcome $c$
- $g$ is a learned projection

**Why this matters**: This applies outcome-level pressure to a reliable terminal state. It improves endpoint consistency but does not guarantee a correct rollout.

---

## 5. Optional Inference-Time Relational Projection

When $r_\psi(h_t)$ is high and the controller selects a relational abstraction, an optional projection step can reduce violation of the learned relational regularizer before the next rollout step. It is an approximate optimization procedure, not an exact projection onto a known manifold.

### Algorithm

```
for each prediction step t:
    1. Select: (Delta_t, alpha_t) ~ pi_kappa(. | h_t)
    2. Predict: z_hat = F_theta^(Delta_t, alpha_t)(z_t, action_segment)
    
    3. If r_psi(h_t) is high and alpha_t is relational:
        a. Decode z_hat into a raw scene graph via GINE decoder
        b. Generate target tensor: z_target = TPR_Head(z_hat)
        c. Apply semantic projection (frozen backbone):
           z_hat ← z_hat - eta * grad_z_hat(1 - cosine_similarity(z_hat, z_target))
        d. Repeat steps b-c for 1-3 iterations
        
    4. Feed the selected prediction into the next controller step
```

**Why this works**: The step can reduce the chosen relational residual before another rollout. Its effect on physical plausibility and planning must be measured against a no-projection ablation.

---

## 6. Training Procedure (Staged Implementation)

### Stage 1: Oracle Symbols (Weeks 1-4)
- Ground-truth scene graphs available
- Train GINE + TPR Head with $\mathcal{L}_{\text{semantic}} + \mathcal{L}_{\text{struct}}$ using oracle predicates
- Verify that relational regularization improves endpoint prediction and cross-level consistency
- **Minimum publishable unit**

### Stage 2: Learned Symbols (Weeks 5-8)
- Remove oracle supervision
- Train GINE encoder to decode latent states into scene graphs
- Jointly train predicate embeddings $\mathbf{v}_s$ end-to-end
- Compare: oracle vs. learned symbolic state

### Stage 3: Joint-Controller Ablation (Weeks 9-11)
- Compare fixed $(\Delta,\alpha)$ pairs, temporal-only adaptation, abstraction-only adaptation, a factorized controller, and a joint controller
- Measure endpoint accuracy, horizon-normalized prediction error, physical violations, controller cost, and switch calibration
- Evaluate oracle-reliability and learned-reliability variants

### Stage 4: Full BG-NS-JEPA (Weeks 12-14)
- End-to-end training with a joint bi-granular controller
- Evaluate on NovPhy, Physhion, CLEVRER

---

## 7. Implementation Details

### 7.1 Network Architectures

**GINE Encoder**:
```python
class GINEEncoder(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, out_dim):
        self.conv1 = GINEConv(MLP(node_dim + edge_dim, hidden_dim))
        self.conv2 = GINEConv(MLP(hidden_dim + edge_dim, hidden_dim))
        self.readout = MLP(hidden_dim, out_dim)
    
    def forward(self, G):
        # G: graph with node features, edge features, adjacency
        h = self.conv1(G)
        h = self.conv2(h)
        z = global_mean_pool(h)  # or attention pool
        return z
```

**TPR Readout Head**:
```python
class TPRReadout(nn.Module):
    def __init__(self, hidden_dim, role_dim, filler_dim):
        self.role_predictor = MLP(hidden_dim, role_dim)  # per node
        self.filler_predictor = MLP(hidden_dim, filler_dim)  # per predicate
        self.embedding_layer = nn.Embedding(num_predicates, filler_dim)
    
    def forward(self, node_embeddings, edge_embeddings, predicate_indices):
        # Predict role vectors for each object
        roles = self.role_predictor(node_embeddings)  # [N, d_r]
        # Get filler vectors for each predicate
        fillers = self.embedding_layer(predicate_indices)  # [M, d_v]
        # Compute TPR sum (approximated via efficient implementation)
        z_tpr = tensor_product_sum(roles, fillers, edge_indices)
        return z_tpr
```

**Semantic Projection Head**:
```python
class SemanticProjection(nn.Module):
    def __init__(self, latent_dim, pred_dim):
        self.head = MLP(latent_dim, pred_dim)  # one per predicate type
    
    def forward(self, z, predicate_type):
        return self.heads[predicate_type](z)  # returns embedding vector
```

### 7.2 Hyperparameters

| Parameter                                | Value |
| ---------------------------------------- | ----- |
| Latent dimension $d_z$                   | 512   |
| Role dimension $d_r$                     | 64    |
| Filler dimension $d_v$                   | 128   |
| Predicate embedding dimension $d_{pred}$ | 128   |
| Margin $m$                               | 0.5   |
| $\epsilon_{\text{KE}}$                   | 0.01  |
| $\epsilon_{\text{contact}}$              | 2     |
| TPR projection iterations                | 3     |
| $\eta$ (semantic snap)                   | 0.01  |

### 7.3 Optimization

- **Optimizer**: AdamW ($\beta_1 = 0.9$, $\beta_2 = 0.999$)
- **Learning rate**: $3 \times 10^{-4}$ (warmup + cosine decay)
- **Gradient clipping**: 1.0
- **Batch size**: 64
- **Reliability estimator $r_\psi$**: learned from latent features; simulator-derived $\phi^*$ is used only as supervision and an oracle upper bound

---

## 8. The "Why This Works" Summary

| Requirement                      | How BG-NS-JEPA Satisfies It                                  |
| -------------------------------- | ------------------------------------------------------------ |
| **Semantic similarity**          | Predicate embeddings make semantic similarity measurable and testable |
| **Compositional reasoning**      | TPR enforces explicit Role-Filler binding; "A supports B" ≠ "B supports A" |
| **Handles arbitrary graph size** | GINE processes variable nodes/edges natively                 |
| **Interaction-active robustness** | Low reliability masks fine relational constraints; continuous dynamics remain active |
| **Reliable relational precision** | High reliability activates semantic, TPR, and contrastive regularizers |
| **Drift mitigation**             | Optional inference-time projection reduces the chosen relational residual |
| **Endpoint consistency**         | Terminal anchor encourages an outcome-consistent final latent |

---

## 9. Key Theoretical Insight

**Structured Relational Regularization with Semantic Smoothing**:

Prior work: Symbols as points in low-energy basins (soft, no structure)

**Ours**: Predicate projections define semantically grounded level sets in latent space:

$$\mathcal{M}_s = \{z \in \mathcal{Z} : f_s(z) = \mathbf{v}_s\}$$

Where $\mathbf{v}_s$ is a learned continuous embedding. Logical conjunction becomes:

$$\mathcal{M}_{s_1 \land s_2} = \mathcal{M}_{s_1} \cap \mathcal{M}_{s_2}$$

When $\mathbf{v}_s$ is a regular value of $f_s$, the level set is locally a submanifold. The losses encourage proximity to these sets; they do not enforce the proportional distance relation above without additional constraints.

**The hybrid GINE-TPR architecture implements this by**:
- GINE provides flexible perception of arbitrary graphs
- TPR provides the algebraic structure for composition
- The geometric loss encourages GINE's latent space to respect the TPR structure
- The learned reliability estimator determines when fine relational regularization is applied

---

## 10. Baselines for Comparison

| Baseline                    | Description                                                  |
| --------------------------- | ------------------------------------------------------------ |
| LeWM                        | Pure continuous JEPA                                         |
| Uniform micro               | Always frame-by-frame + object-level                         |
| Uniform macro               | Always event-level + structure-level                         |
| Temporal-only adaptive      | Learns $\Delta t$ with a fixed abstraction                   |
| Abstraction-only adaptive   | Learns $\alpha$ with a fixed temporal stride                 |
| Factorized controller       | Learns $\pi_\Delta\pi_\alpha$ rather than a joint policy    |
| BG-NS-JEPA (oracle reliability) | Uses simulator-derived reliability at test time          |
| BG-NS-JEPA (no reliability gate) | Applies fine relational regularization uniformly        |
| BG-NS-JEPA (no SPSG)        | Ablates structured symbolic geometry (uses flat text encoding) |
| BG-NS-JEPA (binary symbols) | Ablates semantic embeddings (uses $c_s \in \{0,1\}$)         |
| BG-NS-JEPA (oracle graph)   | Upper bound for symbol extraction                            |

---

## 11. Metrics

**Prediction**:
- ADE@H (Average Displacement Error at horizon H)
- FDE@H (Final Displacement Error)
- Final-state symbolic accuracy
- Event prediction F1

**Physical Plausibility**:
- Object penetration rate
- Unsupported-floating rate
- Illegal-contact rate

**Granularity**:
- Percentage of time in continuous/micro/macro mode
- Effective prediction steps
- Controller cost and matched-compute frontier
- Switch precision/recall and joint-pair calibration against oracle regimes

**Planning**:
- Task success rate
- Shots-to-success
- Novelty adaptation speed

---

## 12. Risk & Mitigation Summary

| Risk                                    | Mitigation                                                   |
| --------------------------------------- | ------------------------------------------------------------ |
| Learned symbolic extraction unstable    | Start with oracle; GINE more stable than slot attention      |
| "Hand-engineered" vocabulary critique   | Vocabulary is universal physical substrate; all truth values learned |
| TPR computation expensive               | Use efficient TPR approximations (e.g., random projections)  |
| Reliability estimator misclassification | Report oracle-reliability upper bound; train with noisy labels and calibrated confidence |
| Reviewer sees a component bundle        | Make joint-vs-factorized-controller ablations the primary evidence |
| Method too complex for timeline         | Oracle-symbol version is minimum publishable unit            |

---

## 13. Conclusion

BG-NS-JEPA is a regime-adaptive world-model framework that:

1. Treats temporal horizon and representational abstraction as joint, state-dependent choices
2. Uses reliability-gated relational regularization instead of assuming universal symbolic validity
3. Uses semantic embeddings and TPR as compositional inductive biases, not as the primary novelty claim
4. Evaluates optional projection and terminal anchoring as empirical drift-mitigation mechanisms
5. Is trainable end-to-end with explicit fixed, single-axis, factorized, and oracle-controller ablations

The hybrid GINE-TPR architecture supplies relational perception and compositional structure inside a controller that changes the world model's resolution when the physical regime changes.

---

**Document Version**: 1.1
**Last Updated**: July 2026
**Status**: Ready for supervisor presentation and implementation
