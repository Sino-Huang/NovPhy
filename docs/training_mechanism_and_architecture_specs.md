# BG-NS-JEPA: Training Mechanism and Architectural Specification

**Bi-Granular Neuro-Symbolic JEPA with Semantic Tensor-Product Geometry**

*Technical Document — June 2026*

---

## 1. Overview

This document specifies the complete training mechanism for BG-NS-JEPA, integrating the architectural refinements discussed:
- **Semantic predicate embeddings** (CLIP-like similarity) over binary scalars
- **Tensor Product Representations (TPR)** for strict compositional binding
- **Hybrid GINE-TPR architecture** combining relational perception with algebraic structure
- **Phase-gated training** that deactivates symbolic constraints during chaos ($\phi = 0$)
- **Inference-time semantic projection** to prevent latent drift

---

## 2. Problem Recap

### 2.1 Environment
An episode $\mathcal{E} = \{(o_t, a_t, x_t)\}_{t=0}^T$ where:
- **Action sparsity**: $\rho_a = \frac{1}{T}\sum_{t=0}^{T-1} \mathbf{1}[a_t \neq \text{noop}] \ll 1$
- **Effect persistence**: $\tau_{\text{eff}} = \min\{k: \forall t > t_{\text{last}}+k, \|x_t - x_{t-1}\| < \epsilon\} \gg 1$

### 2.2 Failure Mode
Standard JEPA $\hat{z}_{t+1} = F_\theta(z_t, a_t)$ with $a_t = \text{noop}$ for extended periods causes recursive latent rollout drift into physically impossible states.

### 2.3 Core Insight
Symbols are only valid at **stable attractors** ($\phi = 1$). During chaotic transients ($\phi = 0$), symbolic constraints must be **completely deactivated**.

---

## 3. Architectural Components

### 3.1 JEPA Backbone (The Chaos Engine)
Standard action-conditioned latent predictor:
$$\hat{z}_{t+1} = F_\theta(z_t, a_t)$$

**Active**: Always.
**Role**: Predicts continuous physics in latent space during all phases.

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

**Active**: Only during stable phases ($\phi = 1$).

---

### 3.4 Bi-Granular Controller
Jointly selects temporal resolution $\Delta t_k$ and physical abstraction $\alpha_k$:

$$(\Delta t_k, \alpha_k) = \pi_\kappa(u_t, \phi(x_t), p_\eta(E_t), \lambda_t^{(w)})$$

**Decision Regimes**:

| Regime                 | $\phi$ | $u_t$  | $\lambda_t$ | $\Delta t$ | $\alpha$    |
| ---------------------- | ------ | ------ | ----------- | ---------- | ----------- |
| Pre-shot stable        | 1      | low    | low         | coarse     | macro       |
| Collision onset        | 0      | high   | high        | fine       | **none**    |
| Chaotic collapse       | 0      | high   | high        | fine       | **none**    |
| Post-collapse settling | 1→0→1  | medium | medium      | adaptive   | micro→macro |
| Equilibrium            | 1      | low    | low         | coarse     | macro       |

**Active**: Always.

---

### 3.5 Stability Gate (Lyapunov Grounded)
$$\phi(x_t) = \mathbb{1}_{[\text{KE}(x_t) < \epsilon_{\text{KE}}]} \cdot \mathbb{1}_{[\text{contacts}_{\text{active}}(x_t) < \epsilon_{\text{contact}}]}$$

**Theoretical grounding**: $\phi = 1 \iff \lambda_{\max} < 0$ (contracting manifold), $\phi = 0 \iff \lambda_{\max} > 0$ (chaotic expansion).

---

## 4. Unified Training Loss

The complete training objective is:

$$
\boxed{
\begin{aligned}
\mathcal{L}_{\text{total}} = \; & 
\underbrace{\mathcal{L}_{\text{JEPA}}}_{\text{(1) Always Active}} \\
& + \phi(x_t) \cdot \Bigg[
\underbrace{\mathcal{L}_{\text{semantic}}}_{\text{(2) Semantic Alignment}} + 
\underbrace{\mathcal{L}_{\text{struct}}}_{\text{(3) TPR Composition}} + 
\underbrace{\mathcal{L}_{\text{contrastive}}}_{\text{(4) Hard Negatives}}
\Bigg] \\
& + \underbrace{\mathcal{L}_{\text{anchor}}}_{\text{(5) Terminal Pole}}
\end{aligned}
}
$$

### 4.1 Component (1): JEPA Prediction Loss (Always Active)

$$\mathcal{L}_{\text{JEPA}} = \mathbb{E}_{t} \left[ \| \hat{z}_{t+1} - z_{t+1} \|_2^2 \right]$$

Where $\hat{z}_{t+1} = F_\theta(z_t, a_t)$.

This ensures continuous physics prediction operates **unconstrained** during chaos.

---

### 4.2 Component (2): Semantic Alignment Loss (Active Only When $\phi = 1$)

For each ground-truth predicate $s \in \mathcal{S}_{\text{true}}$ active in the scene:

$$\mathcal{L}_{\text{semantic}} = \sum_{s \in \mathcal{S}_{\text{true}}} \| f_s(z_t) - \mathbf{v}_s \|_2^2$$

Where:
- $f_s: \mathcal{Z} \to \mathbb{R}^{d_v}$ is a learned projection head (MLP) decoding predicate $s$
- $\mathbf{v}_s \in \mathbb{R}^{d_v}$ is the learned semantic embedding for predicate $s$

**Why this matters**: This forces the latent space to arrange itself so that the geometric distance between $f_{\text{supports}}(z)$ and $f_{\text{contact}}(z)$ mirrors their physical/functional similarity. Predicate embeddings are learned jointly, enabling smooth generalization.

---

### 4.3 Component (3): TPR Compositional Loss (Active Only When $\phi = 1$)

$$\mathcal{L}_{\text{struct}} = \left\| z_t - \sum_{(o_i, o_j, s) \in \mathcal{G}_t} \mathbf{r}_{o_i} \otimes \mathbf{r}_{o_j} \otimes \mathbf{v}_{s} \right\|_2^2$$

Where:
- $\mathcal{G}_t$ is the ground-truth scene graph at time $t$
- $\mathbf{r}_{o_i}$ are learned role embeddings for each object
- $\mathbf{v}_{s}$ are the same semantic predicate embeddings

**Why this matters**: The tensor product preserves variable binding—"A supports B" and "B supports A" occupy orthogonal subspaces. The gradient flows backward through GINE, organizing its latent space according to the TPR algebra.

**Implementation Note**: In practice, we compute this loss using a **TPR Readout Head** attached to GINE, not by explicitly computing the full tensor product over all objects (which would be $O(N^3)$). The head predicts the role and filler vectors for each node/edge.

---

### 4.4 Component (4): Semantic Contrastive Loss (Active Only When $\phi = 1$)

Hard negatives are generated via differentiable physics:
- Anti-gravity: $g = -9.8$
- Massless materials: $\rho_{\text{wood}} = 0$
- Anti-support: remove support while keeping structure upright

$$\mathcal{L}_{\text{contrastive}} = \sum_{\text{neg}} \max\left(0, m - \text{sim}(z_t, z_{\text{pos}}) + \text{sim}(z_t, z_{\text{neg}})\right)$$

Where:
- $\text{sim}(a, b) = \cos(a, b)$
- $z_{\text{pos}}$ is the latent of a physically valid configuration
- $z_{\text{neg}}$ is the latent of an impossible configuration
- $m$ is the margin hyperparameter

**Why this matters**: Negative samples are semantically coherent (the engine validates them), ensuring the energy landscape has meaningful structure.

---

### 4.5 Component (5): Terminal Anchor Loss (Always Active, but Only Applies When $\phi(T) = 1$)

$$\mathcal{L}_{\text{anchor}} = \mathbb{1}_{[\phi(x_T)=1]} \cdot \sum_{c \in \mathcal{C}} \mathbb{1}_{[y=c]} \left\| g(z_c^{(K-1)}) - z_{\text{pole}}^{(c)} \right\|_2^2$$

Where:
- $z_T$ is the final latent state
- $z_{\text{pole}}^{(c)}$ is the learned symbolic pole for outcome $c$
- $g$ is a learned projection

**Why this matters**: Even if the neural network hallucinated the trajectory of every splinter, the final latent state must collapse onto the symbolic pole of the true outcome.

---

## 5. Inference-Time Semantic Projection (The "Reality Check")

During rollouts in stable phases ($\phi = 1$), we prevent drift by projecting the predicted latent back onto the valid semantic manifold **before** feeding it into the next JEPA step.

### Algorithm

```
for each prediction step t:
    1. Predict: \hat{z}_{t+1} = F_\theta(z_t, a_t)
    
    2. If \phi(x_t) = 1:
        a. Decode \hat{z}_{t+1} into a raw scene graph via GINE decoder
        b. Generate target tensor: z_target = TPR_Head(\hat{z}_{t+1})
        c. Apply semantic projection (frozen backbone):
           \hat{z}_{t+1} ← \hat{z}_{t+1} - η · ∇_{\hat{z}} (1 - cos-sim(\hat{z}_{t+1}, z_target))
        d. Repeat steps b-c for 1-3 iterations
        
    3. Feed corrected \hat{z}_{t+1} into the next JEPA step
```

**Why this works**: The projection "snaps" the latent to the nearest semantically plausible region of the manifold, preventing the accumulation of small errors that would otherwise lead to physically impossible states.

---

## 6. Training Procedure (Staged Implementation)

### Stage 1: Oracle Symbols (Weeks 1-4)
- Ground-truth scene graphs available
- Train GINE + TPR Head with $\mathcal{L}_{\text{semantic}} + \mathcal{L}_{\text{struct}}$ using oracle predicates
- Verify that SPSG improves JEPA endpoint prediction
- **Minimum publishable unit**

### Stage 2: Learned Symbols (Weeks 5-8)
- Remove oracle supervision
- Train GINE encoder to decode latent states into scene graphs
- Jointly train predicate embeddings $\mathbf{v}_s$ end-to-end
- Compare: oracle vs. learned symbolic state

### Stage 3: Granularity Ablation (Weeks 9-11)
- Fix controller, vary $\Delta t \in \{2, 5, 10, 30, \text{full FPS}\}$
- Measure: terminal symbolic accuracy, continuous MSE, physical violation rate
- Find $\Delta t^*$ (U-shaped curve)

### Stage 4: Full BG-NS-JEPA (Weeks 12-14)
- End-to-end training with bi-granular controller
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
- **Phase gate $\phi$**: Computed online from simulator state

---

## 8. The "Why This Works" Summary

| Requirement                      | How BG-NS-JEPA Satisfies It                                  |
| -------------------------------- | ------------------------------------------------------------ |
| **Semantic similarity**          | Predicates embedded as continuous vectors $\mathbf{v}_s$; $\cos(\mathbf{v}_{s_1}, \mathbf{v}_{s_2})$ enables smooth generalization |
| **Compositional reasoning**      | TPR enforces explicit Role-Filler binding; "A supports B" ≠ "B supports A" |
| **Handles arbitrary graph size** | GINE processes variable nodes/edges natively                 |
| **Chaos robustness**             | $\phi=0$ deactivates all symbolic constraints; pure JEPA handles splintering |
| **Stable phase precision**       | $\phi=1$ activates semantic + TPR + contrastive losses; sharp symbolic constraints |
| **Drift prevention**             | Inference-time semantic projection (Reality Check) snaps latents to valid manifold |
| **Final outcome guarantee**      | Terminal anchor loss forces final state onto symbolic pole   |

---

## 9. Key Theoretical Insight

**From Energy Basins to Geometric Constraints with Semantic Smoothing**:

Prior work: Symbols as points in low-energy basins (soft, no structure)

**Ours**: Symbols as **semantically grounded submanifolds** in latent space:

$$\mathcal{M}_s = \{z \in \mathcal{Z} : f_s(z) = \mathbf{v}_s\}$$

Where $\mathbf{v}_s$ is a learned continuous embedding. Logical conjunction becomes:

$$\mathcal{M}_{s_1 \land s_2} = \mathcal{M}_{s_1} \cap \mathcal{M}_{s_2}$$

And semantic similarity is preserved because $\text{dist}(\mathcal{M}_{s_1}, \mathcal{M}_{s_2}) \propto \|\mathbf{v}_{s_1} - \mathbf{v}_{s_2}\|$.

**The hybrid GINE-TPR architecture implements this by**:
- GINE provides flexible perception of arbitrary graphs
- TPR provides the algebraic structure for composition
- The geometric loss forces GINE's latent space to obey TPR geometry
- The phase gate $\phi$ determines when this structure is enforced

---

## 10. Baselines for Comparison

| Baseline                    | Description                                                  |
| --------------------------- | ------------------------------------------------------------ |
| LeWM                        | Pure continuous JEPA                                         |
| Uniform micro               | Always frame-by-frame + object-level                         |
| Uniform macro               | Always event-level + structure-level                         |
| BG-NS-JEPA (no phase gate)  | Ablates stability gating ($\phi$ always 1)                   |
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
- Percentage of time in micro/macro/none mode
- Effective prediction steps
- Switch precision/recall

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
| Phase gate misclassification            | Use hysteresis; train with noisy $\phi$ for robustness       |
| Chaos phase has no symbolic supervision | **By design**—chaos uses pure JEPA; symbols only anchor endpoints |
| Method too complex for timeline         | Oracle-symbol version is minimum publishable unit            |

---

## 13. Conclusion

BG-NS-JEPA is a **principled, implementable** framework that:

1. **Solves** the granularity mismatch in action-sparse persistent-effect environments
2. **Unifies** semantic similarity (via predicate embeddings) with compositional reasoning (via TPR)
3. **Respects** the physics by deactivating symbols during chaos ($\phi = 0$)
4. **Prevents drift** via inference-time semantic projection
5. **Is trainable end-to-end** with a clear staged de-risking plan

The hybrid GINE-TPR architecture gives us the best of both worlds: GINE's relational perception + TPR's algebraic compositionality, all grounded in a phase-gated training objective that matches the underlying dynamical systems.

---

**Document Version**: 1.0
**Last Updated**: June 2026
**Status**: Ready for supervisor presentation and implementation