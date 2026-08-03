# BG-NS-JEPA: End-to-End Flow from Observation, and the Full Training Picture

*Consolidated system description — synthesizes all decisions from the proposal and Q&A series 1–5. 31 July 2026*

---

## 1. The cast: components and their status

| Component | Symbol | What it is | Status |
|---|---|---|---|
| Visual encoder | — | Frozen pretrained ViT (V-JEPA/DINO-style) | **Frozen always** |
| JEPA context/target encoders | $o_t \mapsto z_t$ | Continuous latent state | Target encoder frozen (EMA / stop-grad); context encoder learned |
| Symbolic extractor (Tier 2) | $o_t \mapsto G_t$ | DETR-style object decoder + pairwise predicate heads; calibrated probability outputs | **Pretrained supervised, then frozen** |
| Symbolic embedding | $z_{\text{sym}} = f_{\text{GINE}}(G_t)$ | Relational embedding for SPSG | Learned (trained with SPSG losses) |
| Diagnostics | $u_t, r_\psi, p_\eta(E_t), \lambda_t$ | Uncertainty, learned reliability, event likelihood, event density | $r_\psi$ learned (supervised by oracle $\phi^*$); rest computed |
| Controller state | $h_t = \{z_t, u_t, r_\psi, p_\eta, \lambda_t\}$ | Summary state for decisions | — |
| Controller | $\pi_\kappa(\cdot \mid h_{t_k})$ | Policy over the $(\Delta, \alpha)$ grid | **Learned** (distilled from Stage-1 scoring) |
| World model | $F_\theta^{\Delta,\alpha}(z_t, a, S^\mu_t)$ | Dual-output predictor: carrier $\hat{z}_{t+\Delta}$ + mode-head readout | **Learned** (Phase A/B) |
| Event predictor | $G_\omega(S^M, z, a)$ | Macro-event transition + duration | Learned |
| Restriction / lifting | $A, R$ | Micro↔macro cross-level maps | Learned |
| Terminal anchor | $z_{\text{pole}}^{(c)}$ | Outcome-specific equilibrium targets | Learned |

---

## 2. Inference-time flow: one rollout from observation

```
                 ┌──────────────────────── PERCEPTION (parallel front-ends) ─────────┐
                 │                                                                     │
   o_t ──────────┼──► [frozen visual encoder] ─► [JEPA encoder] ──────────► z_t        │
                 │                                                                     │
                 └──► [frozen visual encoder] ─► [object decoder] ─► [predicate heads] │
                                                    (calibrated probs)      │          │
                                                                            ▼          │
                                                              G_t (soft symbolic state)│
                                                                            │          │
                                                              [GINE] ─► z_sym          │
                 └─────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
   DIAGNOSTICS:  z_t + conf(G_t) ─► u_t, r_ψ, p_η(E_t), λ_t ─► h_t
                                        │
                                        ▼
   DECISION:     (Δ_k, α_k) ~ π_κ( · | h_{t_k} )        ← one joint pair, not two choices
                                        │
                                        ▼
   PREDICTION:   F^{Δ_k,α_k}(z_{t_k}, a, S^μ) ─► ┌──────────────────────────┐
                                                  │ carrier: ẑ_{t_k+Δ_k}      │  (always)
                                                  │ readout: Ŝ^μ or (Ŝ^M, Δ̂, ê)│  (loss/monitor only)
                                                  └──────────────────────────┘
                                        │
                                        ▼
   UPDATE:       z ← ẑ_{t_k+Δ_k},  t ← t_k + Δ_k  ────► back to DIAGNOSTICS
```

Key properties of the loop:

- **$z$ is the only state carrier.** Symbolic outputs are readouts; nothing symbolic is integrated forward in time. No discrete step ever sits in the rollout path, so the whole loop is differentiable.
- **$\alpha$ is re-chosen at every decision.** Consecutive steps may use different $(\Delta, \alpha)$ pairs — the coupled switch (e.g., $(15,\text{macro})$ in quiescence → $(1,\text{continuous})$ at collision onset → back to macro at equilibrium).
- **Modes compose.** During a macro jump, $G_\omega$ predicts the event transition while the continuous backbone can fill micro-dynamics inside the interval ($z_{t+1} = F_\theta(z_t, a_t, S^\mu_t)$); macro symbols remain available even when micro predicates are gated off.
- **Re-anchoring.** In imagination (planning), the loop runs open-loop from the last observation. When new frames arrive, the perception front-ends simply re-run: $z$ and $G$ refresh, and the controller resumes from the new $h$.

---

## 3. Training flow: who is trained, when, on what, with which loss

### Stage 0 — Data and oracle layer (no learning)

Engine rollouts provide, per frame: RGB $o_t$, object states, contact/support relations, macro-event labels (derived), outcome labels. From these: ground-truth predicates $G^*$, target-encoder latents $z^*$, and the oracle gate label $\phi^*(x_t)$ (KE + contact-activity thresholds). **Tier-0 oracle defines the predicate semantics; everything downstream imitates it.**

### Stage 1 — World model + SPSG, with oracle symbols (Weeks 1–4)

- **Trained:** $F_\theta$ (dual-output, over the full $(\Delta,\alpha)$ grid), the mode heads, $f_{\text{GINE}}$, $G_\omega$, $A$, $R$, anchor poles.
- **Regime:** teacher-forced multi-task. Sample $(t, \Delta, \alpha)$; predict from true $z_t$; align to $\mathrm{sg}(z^*_{t+\Delta})$; symbolic heads supervised by oracle $S^{\mu*}, S^{M*}$.
- **Losses:** the unified objective — $\mathcal{L}_{\text{pred}} + \lambda_{\text{sym}}\omega_\psi \mathcal{L}_{\text{sym}} + \lambda_{\text{cross}}\mathcal{L}_{\text{closure}} + \lambda_{\text{cost}}c + \lambda_{\text{anchor}}\mathcal{L}_{\text{anchor}}$, all $\Delta/T$-weighted. Fully differentiable; no gradient-through-time.
- **Also produced:** exhaustive scoring of every $(\Delta,\alpha)$ pair per state → **best-pair labels** (the controller's future teacher) + the oracle-symbol upper bound.

### Stage 2 — Symbolic extractor + reliability gate (Weeks 5–8)

- **Trained:** object decoder, pairwise/unary predicate heads (supervised by engine $G^*$ — multi-task $\mathcal{L}_{\text{obj-cls}} + \lambda_1\mathcal{L}_{\text{attr}} + \lambda_2\mathcal{L}_{\text{edge}} + \lambda_3\mathcal{L}_{\text{unary}} + \lambda_4\mathcal{L}_{\text{temporal}}$); then $r_\psi$ on extractor confidence features, supervised by $\phi^*$.
- **Frozen:** visual encoder (throughout), and the extractor itself once trained — predicate semantics stay anchored to engine definitions; no global loss may reshape them.
- **Exit evidence:** learned-vs-oracle gap, per-predicate F1, gate calibration (switch precision/recall vs. oracle regimes), flip-rate coherence.

### Stage 3 — Controller + interface ablations (Weeks 9–11)

- **Trained:** $\pi_\kappa$ by distillation of Stage-1 best-pair labels (supervised classification over the pair grid).
- **Same unified objective** for $F$; the $(\Delta,\alpha)$ sampling distribution now follows the controller (or stays uniform, per ablation design).
- **Ablated here:** joint vs. factorized vs. single-axis vs. fixed controllers at matched compute; shared-backbone vs. separate-expert interfaces; (optionally) recurrent controller with $(\Delta_{k-1},\alpha_{k-1})$ in $h$.
- **This is the experiment the paper's central claim lives on.**

### Stage 4 — Full system (Weeks 12–14)

- End-to-end evaluation on NovPhy + Physhion/CLEVRER.
- **Optional variants, each gated by an acceptance criterion:**
  - end-to-end controller relaxation (Gumbel-softmax through the discrete pair choice — the *only* place gradients cross a discrete decision);
  - short-window autoregressive fine-tuning of $F$ through the $z$ carrier (anti-exposure-bias, adopted only if pilot drift demands it);
  - extractor fine-tuning with anchored symbol loss + predicate-drift acceptance metric (rejected on any significant drift).

### Gradient-flow summary

| Path | Differentiable? | Why |
|---|---|---|
| $z_t \to \hat{z}_{t+\Delta}$ (carrier) | Yes, always | Pure feed-forward; no discrete step in the state path |
| $z_t \to$ symbolic heads (readouts) | Yes | Predicate logits / GINE outputs; CE-style losses |
| Observation $\to G$ (extractor) | Not needed | Frozen after Stage 2; never in the training graph of $F$ |
| Through controller's pair choice | Only in optional Stage 4 | Gumbel/REINFORCE; default is supervised distillation |
| Through time (multi-step rollout) | Not in base training | Teacher-forced sub-tasks; optional Phase-C windows through $z$ only |

---

## 4. One-paragraph version (for the meeting)

> From an observation, two parallel frozen front-ends produce the continuous latent $z_t$ and a soft, calibrated symbolic state $G_t$. Their diagnostics — uncertainty, learned symbolic reliability, event likelihood, event density — form the controller state $h_t$, from which a learned policy $\pi_\kappa$ picks one joint pair: a horizon $\Delta_k$ and a description level $\alpha_k$. The world model $F$ then predicts the state $\Delta_k$ frames ahead in representation $\alpha_k$, always producing the continuous carrier $\hat{z}$ plus a symbolic readout that enters only the loss. The rollout continues from $\hat{z}$, re-choosing $(\Delta, \alpha)$ at every decision — fine continuous steps in collision-active regimes, long macro jumps in quiescent ones. Training is teacher-forced multi-task over the $(\Delta,\alpha)$ grid (the standard JEPA regime), with the symbolic extractor pretrained and frozen to keep predicate semantics anchored to engine ground truth, the controller distilled from an exhaustive per-state scoring of all pairs, and the reliability gate learned on the real extractor's confidence. Everything in the base training graph is differentiable; discrete-choice gradients appear only in one optional Stage-4 ablation.

---

## Series index

1. `controller_router_QA.md` — is the controller just a router over prediction errors? (No: amortized forecast, different prediction problems, sequential policy.)
2. `symbolic_representation_QA_and_pipeline.md` — how is the observation→symbolic model trained? (Supervised predicate parser, frozen; three-tier ladder.)
3. `symbolic_noise_QA.md` — deterministic generator and noise/incoherence. (Probabilistic outputs, gate absorbs, $z$ is the carrier, robustness measured.)
4. `z_g_interaction_and_alpha_mechanics_QA.md` — do $z$ and $g$ interact, and how does $\alpha$ work across steps? (Parallel perception, interacting dynamics/control; $\alpha$ re-chosen per step.)
5. `world_model_training_regime_QA.md` — teacher forcing vs. autoregressive, gradient breaks. (Dual-output carrier decision; teacher-forced multi-task by choice.)
6. **This document** — the consolidated flow.
