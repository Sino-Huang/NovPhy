# Is the BG-NS-JEPA Controller Just a Router? — Analysis and Responses

*Working notes for supervisor discussion, 31 July 2026*

---

## Q1. Hamid's simplified model: two predictors plus an error-based router

**Hamid's framing:**

> Suppose you have a Neural JEPA producing $\hat{z}_{t+\Delta t}$ close to the ground truth $z^*_{t+\Delta t}$, and a symbolic-graph predictor producing $\hat{g}_{t+\Delta t}$ close to $g^*_{t+\Delta t}$. Then your controller is essentially a router that, based on $d(\hat{z}_t, z^*_t)$ and $d(\hat{g}_t, g^*_t)$, selects which representation of the world to route to. Is that right?

### Short answer

Partially. The picture is correct at the highest level of abstraction — there *is* a state-dependent selection among predictive representations, and it is in fact a good description of **how the controller's training labels are generated**. But the deployed controller is not that router, for four reasons.

### Difference 1: the decision signal is a forecast of reliability, not a measured error

$d(\hat{z}_t, z^*_t)$ requires ground truth $z^*_t$, which does not exist at test time — especially after dozens of open-loop rollout steps. The controller cannot, and does not, compare current errors. Instead, $r_\psi(h_t)$, $u_t$, and $\lambda_t$ form a **feed-forward estimate of which description will remain valid over the upcoming horizon $\Delta$**. Hamid's router is *reactive* (switch after observing error); our controller is *predictive* (anticipate the regime, then choose). In multiscale terms: the Knudsen number in a hybrid kinetic–fluid solver is a local diagnostic computed from the state — not the result of running both solvers and comparing errors, which is precisely the computation the switch exists to avoid.

### Difference 2: the candidates are different prediction problems, not two answers to one problem

In Hamid's picture, $\hat{z}$ and $\hat{g}$ each regress toward their own agreed target, and the router picks one output — this is the MoE structure, where experts solve the same task with different weights. But $(1, \text{continuous})$ and $(15, \text{macro})$ predict **different quantities, in different spaces, at different temporal offsets**: one asks "what is the event-level state of the world 15 frames from now?", the other asks "what is the continuous latent 1 frame from now?". The controller is not choosing *which answer to trust*; it is choosing **which question to ask**. Note also that the framing above omits the $\Delta$ axis entirely — and jointly selecting $(\Delta, \alpha)$ is the core of our claim.

### Difference 3: decisions are sequential, and each decision changes the state of the next one

A routing decision does not alter the distribution of future inputs. Our controller, choosing $\Delta = 15$, **skips 14 frames of information**: the next decision is made at $t{+}15$, from a different state. This makes the controller a **policy over prediction tasks in a semi-MDP**, not a per-step classifier. Two consequences have no router analogue: decision-level credit assignment, and the cost term $c(\Delta, \alpha)$ (an always-fine policy carries a real cost; an always-fine "route" does not).

### Difference 4: the modes compose; they are not mutually exclusive

A router selects one path. In BG-NS-JEPA, macro-event prediction and continuous infilling can be active simultaneously ($G_\omega$ predicts an event interval while the JEPA backbone fills in the micro-dynamics inside it), and macro symbols remain available while micro relations are gated off. This is cooperation within a **hierarchy of descriptions**, not a two-way switch.

### Where Hamid's picture is exactly right

Stage 1 (exhaustive scoring) does almost literally what he described: at training time, every $(\Delta, \alpha)$ pair is scored per state against ground truth (duration-normalized prediction error + physical violations + compute), and the per-state argmin becomes the controller's supervision. **His "router" is our label generator.**

### Suggested response (ready to say)

> Partially. At the highest level, yes — there is a state-dependent selection among predictive representations, and your picture is actually a good description of how we *generate the controller's training labels*: at training time we do score every $(\Delta, \alpha)$ pair against ground truth per state. But the controller itself is not that router, for three reasons. First, $d(\hat{z}_t, z^*_t)$ is unavailable at test time — $r_\psi$ is a learned *forecast* of which description will remain valid over the *upcoming* horizon, not a measured current error. Second, the candidates are not two answers to one prediction problem; $(1, \text{continuous})$ and $(15, \text{macro})$ are different problems — different target spaces and different time offsets — so the controller selects *which question to ask*, and it selects horizon and representation jointly. Third, the decisions are sequential: choosing $\Delta{=}15$ skips 14 frames of information, so the next decision is made from a different state. That makes it a policy over prediction tasks, closer to a semi-MDP than to a router. And the modes compose: macro-event prediction and continuous infilling can be active simultaneously inside one event interval.

---

## Q2. Is the controller itself a learnable model?

**Question:**

> Can we confirm that the controller is also a learnable model — just implemented by something more elegant than comparing with ground truth?

### Short answer

Yes. There are in fact **two learned models**: the controller $\pi_\kappa$ and the reliability estimator $r_\psi$. But one caveat on phrasing: ground truth is not avoided — it is **moved from test time to training time**.

### What is learned, and how

| Component | Type | Supervision |
|---|---|---|
| Controller $\pi_\kappa(\cdot \mid h_{t_k})$ | Learned policy network over the $(\Delta, \alpha)$ pair grid | Best-pair labels distilled from Stage-1 exhaustive scoring |
| Reliability estimator $r_\psi(h_t)$ | Learned scalar estimator | Simulator oracle $\phi^*(x_t)$ built from kinetic energy and contact activity |

### The three-stage picture

| Phase | Role of ground truth |
|---|---|
| **Training — Stage 1** | $z^*, g^*$ are used to score *every* $(\Delta,\alpha)$ pair per state → per-state best-pair labels (this step *is* "comparing with ground truth", exhaustively) |
| **Training — Stage 3** | $\pi_\kappa$ is trained on these labels — learning "which $h_t$ implies which optimal pair" |
| **Test time** | One forward pass of $\pi_\kappa$ produces the decision. Ground truth is neither needed nor available. |

So the comparison mechanism Hamid described is not discarded; it is **amortized into a learned model**. The oracle exists only during training; deployment keeps only a neural-network forward pass. This is the standard logic of distillation — his router is the teacher; our controller is the student.

An optional **Stage 4** trains the controller end-to-end (Gumbel-softmax relaxation of the discrete pair choice, or equivalently an RL view with decision-level credit assignment, since decisions change the subsequent state distribution). It is kept only as an ablation, retained only if it improves the frontier; the supervised distillation default is preferred for stability.

### Suggested response (ready to say)

> Yes — the controller is a learned network, and so is the reliability estimator. Ground truth is not compared at decision time; it is used offline to score every $(\Delta,\alpha)$ pair per state, and the controller is trained to predict the best pair from $h_t$ alone. In other words, the oracle comparison you described is our teacher — the learnable controller is its distilled, deployable version. We also keep an end-to-end variant as an ablation, but the supervised version is the default for stability.

### Why this is a strength, not a concession

The training-time-oracle → test-time-amortized structure makes the experimental narrative self-consistent. Error-based routing can only ever exist as an **upper bound** (it needs oracle access); the **oracle-gate ablation** quantifies exactly how far the learned gate sits below that ceiling. The oracle defines the ceiling; the learned controller defines the deployable reality — and the gap between them is itself a reported result.

---

## Action item

Add a new entry to the anticipated-questions list in `method_innovation_in_one_paragraph.md`:

**Q: "Isn't the controller just a router over prediction errors $d(\hat{z}, z^*)$ vs. $d(\hat{g}, g^*)$?"**
Error-based routing requires oracle access and can only exist as an upper-bound mechanism. The deployable controller replaces measured error with a learned forecast of per-regime description validity, chooses jointly over horizon *and* description level, acts sequentially (each choice changes the state of the next decision), and allows modes to compose rather than compete. The oracle router survives in our pipeline in exactly one place: as the training-time label generator, and as the ceiling in the oracle-gate ablation.

# Observation → Symbolic Representation: Hamid's Question and a Concrete Pipeline

*Working notes for supervisor discussion, 31 July 2026*

---

## Part 1 — Hamid's question, answered

**Hamid's question (paraphrased):**

> I don't know what your symbolic representation is. Suppose you want to convert observations into a scene graph — nodes are objects with attributes, edges are relations. Then it is unknown to me how you train a model to convert observations into the scene graph $G$. Is this pretrained? Will it be fine-tuned during controller training with the global objective loss? Assume we have ground-truth $G$ from the engine: how do you train this model exactly? And if this model is a composition — one segmentation model to extract objects and attributes, one relation parser to extract relations — how can you ensure that your pretrain loss from ground-truth $G$ will not modify the semantics of those individual components?

### Short answer

This is deliberately orthogonal to the paper's claim — which is exactly why the plan is staged. The minimum publishable unit (oracle-symbol Stage 1 plus the controller ablation in Stage 3) never touches learned perception: the symbolic extractor is an upstream, replaceable module, and the controller claim is agnostic to where the symbolic state comes from — the same way a hybrid kinetic–fluid solver's switching criterion is agnostic to whether the kinetic state comes from a simulator or a sensor.

### The three sub-questions

**Q: How do you train observation → symbolic state? Is it pretrained?**

We are in the lucky regime: the engine provides ground-truth $G^*$, so extraction is **plain supervised learning — no external pretrained scene-graph model is needed**. The object/attribute component and the relation parser each have their own engine labels (detection/attribute supervision for the former; edge classification for the latter), so credit assignment between components is unambiguous. A frozen pretrained visual encoder (V-JEPA/DINO-style) may serve as the observation front-end, but all symbolic heads are trained from engine supervision.

**Q: Is it fine-tuned during controller training with the global loss?**

**No — it stays frozen by design.** The semantics of our predicates are *defined* by the engine labels. If the global task loss could reshape the extractor, predicates like `supports(A,B)` could drift into whatever reduces the task loss — ontology reward-hacking — which would invalidate (i) the physical-violation metrics, (ii) switch precision/recall evaluation, and (iii) the calibration of the reliability gate $r_\psi$, whose oracle target is defined against exactly those engine semantics. Freezing also keeps the learned-vs-oracle gap cleanly measurable, which is the Stage-2 exit evidence.

**Q: If end-to-end fine-tuning is ever attempted, how is semantic drift prevented?**

Stage 4 may try it, under safeguards: the supervised symbol loss remains as a fixed-weight **anchor term** alongside the global loss (or, more conservatively, only small adapter heads are trained on a frozen backbone); and **predicate drift** — agreement between the fine-tuned extractor and engine labels on held-out states — is an explicit acceptance metric. If drift is detected, the end-to-end variant is rejected. This is an ablation with an acceptance criterion, not a default path.

### Suggested response (ready to say)

> This is deliberately orthogonal to the paper's claim — which is why the plan is staged. The minimum publishable unit never touches learned perception: the scene graph is an upstream, replaceable module, and the controller claim is agnostic to where the graph comes from, the same way a hybrid solver's switching criterion is agnostic to whether the kinetic state comes from a simulator or a sensor.
>
> For Stage 2, we are in the lucky regime: the engine gives ground-truth $G^*$, so extraction is plain supervised learning, no external pretrained scene-graph model needed. The object/attribute component and the relation parser each have their own engine labels — detection/attribute supervision for the former, edge classification for the latter — so credit assignment is unambiguous. During controller training the extractor stays **frozen by design**: the semantics of our predicates are defined by the engine labels, and if the global loss could reshape them, predicates like `supports(A,B)` could drift into whatever reduces the task loss, which would invalidate the physical-violation metrics and the calibration of the reliability gate. If we later try end-to-end fine-tuning, it keeps the supervised symbol loss as an anchor term, and we measure predicate drift against engine labels on held-out states as an explicit acceptance criterion. The learned-vs-oracle gap is itself a reported Stage-2 result, so the perception contribution is bounded rather than assumed away.

---

## Part 2 — A concrete observation → symbolic representation pipeline

### 2.0 First, a clarification: you do not strictly need a "scene graph"

What the method actually consumes is **a set of grounded predicate truth values** (does `contact(A,B)` hold? what is the value of `velocity-bin(C)`?) plus, for SPSG, a structured embedding that preserves role binding. A "scene graph" is just one convenient serialization of that: objects = nodes, predicates = node/edge labels. The functional requirement is a **predicate parser**, and the graph is the data structure that organizes its outputs. This distinction matters practically: supervision, losses, and evaluation all live at the predicate level; the graph layer only needs to exist where relational *composition* (GINE message passing, TPR role binding) is computed.

### 2.1 The three-tier ladder (matches Stage 1 → Stage 2 of the proposal)

| Tier | Input | Symbol source | Purpose |
|---|---|---|---|
| **Tier 0 — Oracle** | Engine state | Direct readout of ground-truth predicates | Stage 1 upper bound; defines predicate semantics; generates all training labels |
| **Tier 1 — Feature parser** | Simulator state vector (positions, velocities, contacts) | MLP heads on engine features | Sanity tier: verifies the predicate vocabulary is learnable and well-posed before vision is involved |
| **Tier 2 — Visual parser** | RGB frames (+ optional engine states at train time) | Frozen pretrained encoder + supervised object and predicate heads | The real Stage-2 model; deployed symbolic extractor |

Semantics flow downward: Tier 0 *defines* what each predicate means; Tiers 1–2 are trained to *imitate* Tier 0 and are evaluated by their agreement with it.

### 2.2 Tier 2 architecture (recommended default)

```
RGB frame o_t
   │
   ▼
[Frozen visual encoder]  (V-JEPA 2 / DINO-style ViT; weights locked)
   │  patch/dense features
   ▼
[Object decoder]  ── DETR-style: N learned object queries, cross-attention
   │                 → per-object latent slots {o_1..o_N}
   │  supervised by engine object states:
   │    • existence/classification head (object type, material)  — cross-entropy
   │    • attribute heads (position, velocity, shape, health)    — smooth-L1
   │    • Hungarian matching against engine objects (standard DETR recipe)
   ▼
[Pairwise predicate parser]  ── for each ordered pair (o_i, o_j):
   │    MLP([o_i; o_j; o_i−o_j]) → logits for each edge predicate:
   │      contact(i,j), supports(i,j), contact_normal bins, distance bins
   │    per-predicate binary/multilabel cross-entropy against engine labels
   ▼
[Graph assembly]  nodes = decoded objects + attributes; edges = predicate logits
   │
   ▼
[GINE encoder]  →  z_sym  (consumed by SPSG heads, TPR role-filler head, r_ψ)
```

Design notes:

- **Small, known object counts** in Angry Birds/NovPhy make a query-based object decoder (DETR family) appropriate and far more stable than unsupervised slot discovery; every object has engine supervision, so nothing needs to be unsupervised.
- **Unary predicates** (`velocity-bin`, `damaged`, macro-event labels) are per-object or per-frame heads on the same slots — no graph needed for them.
- **The GINE layer is where the "graph" first matters.** Upstream of it, everything is per-object and per-pair heads; the graph is assembled only so that message passing and TPR role binding can compose relational structure for SPSG. If SPSG ablates poorly, the GINE can be removed and predicate embeddings concatenated directly — the predicate parser itself is unaffected.
- **Temporal consistency**: a lightweight loss encouraging predicate logits to vary smoothly within stable regimes (as indicated by high oracle $r_\psi$ frames) reduces flicker; the reliability gate already tolerates fast changes in active regimes, so no heavy temporal model is needed.

### 2.3 Training recipe

1. **Data**: engine rollouts paired with full engine state per frame (already required for Stage-1 oracle labels — no new collection).
2. **Front-end**: freeze the visual encoder throughout; only object decoder, predicate heads, and GINE are trained (a few M parameters).
3. **Losses**: multi-task sum —
   $\mathcal{L}_{\text{extract}} = \mathcal{L}_{\text{obj-cls}} + \lambda_1 \mathcal{L}_{\text{attr}} + \lambda_2 \mathcal{L}_{\text{edge-pred}} + \lambda_3 \mathcal{L}_{\text{unary-pred}} + \lambda_4 \mathcal{L}_{\text{temporal}}$.
4. **Split by level/scenario**, not by frame, so generalization (novel materials, novel configurations) is measured honestly.
5. **Calibration**: report per-predicate precision/recall and confidence calibration; $r_\psi$ training consumes the extractor's confidence as an input feature.

### 2.4 Interface contract with the rest of BG-NS-JEPA

| Downstream consumer | What the extractor provides | Guarantee required |
|---|---|---|
| Controller $\pi_\kappa$ | Predicate confidences → $r_\psi$ input features | Calibrated confidence, not just argmax labels |
| SPSG losses | $z_{\text{sym}}$ from GINE + predicate logits | Role binding preserved (ordered pairs) |
| Closure consistency $\mathcal{L}_{\text{cross}}$ | Micro predicates for restriction map $A$ | Macro labels derivable from micro labels |
| Physical-violation metrics | Decoded predicates at rollout states | Semantics identical to engine definitions |

### 2.5 Evaluation and acceptance criteria (Stage-2 exit evidence)

- **Learned-vs-oracle gap**: controller performance with Tier-2 symbols vs. Tier-0 oracle symbols — the headline Stage-2 number.
- **Predicate agreement**: per-predicate F1 against engine labels on held-out levels; minimum bar set per predicate criticality (`supports` and `contact` matter most for SPSG).
- **Gate calibration**: switch precision/recall of $r_\psi$ trained on Tier-2 features vs. oracle regimes.
- **Drift check (only if Stage-4 fine-tuning is attempted)**: predicate agreement before vs. after end-to-end training; any significant drop rejects the variant.

### 2.6 Risks and fallbacks

| Risk | Fallback |
|---|---|
| Visual extraction underperforms | Publish Stage 1+3 with oracle symbols; Tier-1 (feature parser) still demonstrates the learned-gate mechanism without vision |
| DETR-style decoder unstable on small data | Use engine bounding boxes at train time to pre-train queries; or predict a fixed object canvas (max-N slots) with occupancy flags |
| Predicate flicker across frames | Temporal consistency loss + confidence thresholding; $r_\psi$ is explicitly designed to tolerate residual noise |
| Reviewer: "perception is trivial with a simulator" | Concede and reframe: perception is not the contribution; the oracle upper bound is the contribution's evidence, and the extractor is an engineering interface with a measured, reported gap |

### 2.7 One-sentence summary for the meeting

> We do not need a general scene-graph model: we need a **supervised predicate parser** — a frozen visual encoder with object and pairwise-predicate heads trained against engine ground truth, frozen thereafter so predicate semantics cannot drift, with the graph assembled only where SPSG needs relational composition — and the whole module is an interface whose gap from the oracle is a measured, reported result rather than an assumption.


# Symbolic Generation Noise vs. Regime Reliability — Hamid's Third Question

*Working notes for supervisor discussion, 31 July 2026*

---

## Q3. If the symbolic representation generator is deterministic, how do you address noise?

**Hamid's question (paraphrased):**

> If your symbolic representation generator is deterministic, how do you address the noise issue — false positives, false negatives? Could noise from the generation process make the symbolic representation progress incoherent? Note that this is separate from the $\alpha$-selection mechanism: choosing $\alpha$ does consider whether symbolic generation is stable, but the noise *coming from the generation process itself* is, in my view, a different thing.

### Short answer

The distinction is correct and we adopt it: there are two different noise sources — **regime-level validity** (is this state one where fine predicates are stable and meaningful?) and **extractor-level error** (the state is perfectly stable, yet the parser still misclassifies a predicate). The design treats them as distinct *conceptually* and *experimentally*, while absorbing both *mechanically*. Four layers of the answer: representation, absorption, architecture, and measurement.

---

### Layer 1 — Representation: deterministic mapping, probabilistic output

"Deterministic" describes the mapping, not the output format. Every predicate head emits a **calibrated probability**, not a hard label — so no downstream component ever consumes the symbolic state as certain. Ambiguous inputs (occlusion, fast contact, overlapping objects) naturally yield low confidence. If stronger epistemic estimates are needed, the heads are small: ensembling or MC-dropout is nearly free. Noise is not ignored; it is explicitly represented.

### Layer 2 — Absorption: three mechanisms that eat extractor noise

1. **The gate is trained on the real extractor, not an idealized one.** In the Tier-2 deployment version, $r_\psi$ takes the extractor's predicate confidences as input features, and its training data carries the extractor's error profile. It therefore learns *which states the parser tends to fail in* (occlusion, particle-scale contacts, fast motion) and down-weights micro-relational constraints there. Hamid's two noise sources are conceptually distinct — but they are **statistically absorbed by the same gate**, because the gate learns "are symbolic constraints useful in this state", regardless of whether the uselessness comes from the physics or the parser.
2. **Exploiting the FP/FN cost asymmetry.** For SPSG training constraints, a false positive (enforcing a relation that does not hold) is far more damaging than a false negative (a missing constraint degrades gracefully to the continuous baseline). Symbolic constraints are therefore applied only on **high-confidence positives**, with per-predicate thresholds set by dev-set precision targets and hinge margins on the negative side. Missing is harmless; wrong is costly — the design is built around this asymmetry.
3. **Temporal smoothing only where it belongs.** Macro-event predicates are smooth and absorbing by construction (`collapsed`, once true, stays true — this is deliberate vocabulary design: fine predicates are unstable *by physics*, coarse predicates are stable *by physics*). Short-window hysteresis or majority voting on the macro head removes residual flicker. Micro-predicate flicker needs no removal, because the regimes where it occurs are exactly the regimes that get gated off. Our Layer-2 novelty defense — *fine symbols fail while coarse symbols remain valid* — is simultaneously the answer to the noise-incoherence concern.

### Layer 3 — Architecture: symbolic discontinuity cannot propagate into the rollout

The fundamental point: **the rollout's state carrier is the continuous latent $z$, not the symbolic sequence.** The symbolic layer is a constraint/observation layer attached to $z$: micro predicates exist only as reliability-weighted training constraints when the gate is high, and are never integrated forward into the next frame's state. The only symbolic structure that persists across time is the macro-event chain $\mathcal{C}$, whose predicates are absorbing by design. Consequently, a one-frame misclassification of `contact(i,j)` does not enter the state, does not accumulate, and cannot make the rollout incoherent — at worst it adds noise to one frame's loss, which the gate down-weights. Hamid's "incoherent symbolic progress" is a genuine failure mode for systems that use symbols *as* the rollout state (e.g., purely symbolic world models); in our hierarchy it is structurally absent.

### Layer 4 — Measurement: robustness is measured, not asserted

- **Learned-vs-oracle gap** (Stage-2 exit evidence): directly quantifies how much extractor noise costs on final task metrics.
- **Noise-injection ablation** (already in the risk table — "train with noisy labels and calibrated confidence"): extend to a degradation curve — inject increasing label noise and report the slope of endpoint-accuracy decay.
- **Symbolic coherence metric**: per-regime predicate flip rate vs. engine ground-truth flip rate — a direct answer to the "incoherent progress" concern.
- **The falsification table already separates the two failure modes**: "symbol-extraction failure" and "controller failure" are independent rows. Hamid's point that generation noise is "a different thing" is honored by measuring it as a different thing.

---

### Suggested response (ready to say)

> You're right that these are two distinct noise sources — regime-level validity versus extractor-level error — and the design treats them as distinct, though it absorbs both. First, the generator is deterministic as a mapping, but its outputs are calibrated probabilities, not hard labels: no downstream component ever consumes the symbolic state as certain. Second, the reliability gate is trained *on the actual extractor's outputs*, including its confidence, so it learns the extractor's error profile — states where the parser tends to err produce low confidence and are gated off, statistically absorbing extractor noise through the same mechanism that handles regime instability. Third, we exploit the asymmetry: a false-positive constraint (enforcing a nonexistent relation) is far more damaging than a false negative (one missing constraint degrades gracefully to the continuous baseline), so symbolic constraints are applied only on high-confidence positives. And architecturally, the key point: the rollout's state carrier is the continuous latent $z$ — symbols are constraints and observations attached to it, never integrated forward in time, except macro events which are absorbing by design. So a one-frame symbolic mislabel cannot propagate or make the trajectory incoherent; it merely adds noise to that frame's loss, down-weighted by the gate. Empirically we don't assert robustness, we measure it: the learned-vs-oracle gap quantifies the cost of extractor noise on final metrics, we report predicate flip rates against engine ground truth as a coherence measure, and a noise-injection ablation gives the degradation curve.

---

## Cross-references in the project documents

- **Reliability gate trained on extractor confidence**: `symbolic_representation_QA_and_pipeline.md`, §2.3 step 5 (calibration) and §2.4 (interface contract — controller consumes calibrated confidence).
- **Noise-injection training**: `research_proposal.md`, §7 risk table ("Reliability estimator misclassifies a regime → train with noisy labels and use calibrated confidence").
- **Failure-mode separation**: `method_proposal_presentation.md`, falsification table (controller / symbol extraction / relational inductive bias as separable failures).
- **Fine-fails-while-coarse-survives asymmetry**: `method_innovation_in_one_paragraph.md`, Layer 2 mechanism defense.

## Action item

Add to the anticipated-questions list in `method_innovation_in_one_paragraph.md`:

**Q: "If the symbolic generator is deterministic, how do you handle extraction noise and symbolic incoherence?"**
Deterministic mapping, probabilistic outputs: every predicate carries calibrated confidence, and nothing consumes symbols as certain. The reliability gate is trained on the actual extractor's outputs and thus learns its error profile, absorbing extractor noise through the same mechanism that handles regime instability. Constraints apply only to high-confidence positives (FP/FN cost asymmetry), and the rollout state carrier is the continuous latent — symbols are constraints, not integrated state, so one-frame mislabels cannot propagate. Robustness is measured (learned-vs-oracle gap, flip-rate coherence, noise-injection degradation curve), not asserted.


# Do z and g Interact? The Point of Joint Generation, and the Mechanics of α — Hamid's Fourth Question

*Working notes for supervisor discussion, 31 July 2026*

---

## Q4. If the controller controls α but z and g don't interact, why not just generate both in parallel?

**Hamid's question (paraphrased):**

> The controller selects $\alpha$, but the raw continuous latent $z$ and the symbolic representation $g$ do not seem to interact. Then couldn't we just generate $z$ from observation and $g$ from observation in parallel? What is the point?

**Refined sub-questions (our own follow-up):**

1. When predicting the encoding of the next state, do we assume the next encoding has the same $\alpha$ as the one the controller just chose from the current $h$?
2. Is $h$ the same object as what Hamid calls $g$?
3. Is $\alpha_k$ influenced by the previous $\alpha_{k-1}$?

### Short answer

At the **perception front-end**, Hamid is right — $z$ and $g$ are extracted from the observation in parallel, by design. But the interaction happens at the **dynamics layer** and the **control layer**, not the perception layer — and the design already contains three explicit interactions. If $z$ and $g$ were truly parallel and non-interacting, the critique would hold: two redundant descriptions, no trust mechanism, no compute savings. The gain of the joint architecture comes not from *having* both representations, but from *choosing the prediction problem per step* — including the choice to not roll $z$ through quiescent stretches at all.

---

### Notation clarification: $h$ is not $g$

- **$g$** (Hamid's usage): the symbolic representation — the scene graph / predicate encoding ($z_{\text{sym}} = f_{\text{GINE}}(G_t)$, or the predicate states $S^\mu, S^M$). In our system this is a **prediction target** and a **constraint carrier**.
- **$h_t$**: the controller's decision-state summary:

$$h_t = \{z_t,\ u_t,\ r_\psi,\ p_\eta(E_t),\ \lambda_t\}$$

i.e., the continuous latent plus diagnostic scalars. $g$ contributes to $h$ only **indirectly**: extractor confidence enters $r_\psi$, which enters $h$. The embedding of $g$ itself is not in $h$ — the controller reads *whether symbols are reliable*, not *what they say*.

### The mechanics of α across a rollout

1. **Within one decision step:** the pair $(\Delta_k, \alpha_k)$ selected at $t_k$ *defines* that step's prediction task — predict the state at $t_k + \Delta_k$ **in representation $\alpha_k$**. The target encoding's $\alpha$ is the chosen one; this is not an assumption but the definition of the step.
2. **Across steps, no persistence assumption:** after prediction, the mode-specific head maps the result **back into the shared rollout state $z$** ("each head maps its prediction back to the shared rollout state $z$ before the next decision"). The next decision at $t_{k+1}$ re-selects $\alpha_{k+1}$ freely from the new $h_{t_{k+1}}$. Consecutive steps can therefore switch representation — $(1,\text{continuous}) \to (15,\text{macro})$ — which is exactly the coupled switch in the cascade timeline.
3. **Does $\alpha_{k-1}$ influence $\alpha_k$?** In the default design, only **indirectly through state**: the previous decision determines *where and when* the next decision is made (the semi-MDP structure). Feeding $(\Delta_{k-1}, \alpha_{k-1})$ explicitly into $h$ — a recurrent controller — is a legitimate architectural variant, but we treat it as an **ablation**, not a default: if $h$ is sufficiently informative, the Markov assumption is defensible.

In equation-free terms: the shared latent $z$ is the **common currency** between description levels — each step lifts into the chosen space, evolves, and restricts back to $z$. $\alpha$ is re-chosen every step; the state is what persists.

### Where Hamid is right, and where the parallel picture breaks

**Right:** the perception front-end is parallel —
$$o_t \to z_t, \qquad o_t \to G_t$$
— exactly as drawn in the Tier-2 pipeline. Conceding this costs nothing.

**Breaks:** three explicit interactions already exist at the dynamics and control layers:

1. **Cross-conditioned prediction.** The continuous backbone is not $F_\theta(z_t, a_t)$ but $F_\theta(z_t, a_t, S^\mu_t)$ — continuous dynamics *conditioned on the symbolic state*. The macro-event predictor takes both: $G_\omega(S^M_{\tau_i}, z_{\tau_i}, a_{\tau_i})$. Cross-inputs in both directions; structurally not parallel.
2. **Symbolic constraints shape $z$ during training.** SPSG losses and the closure-consistency maps $A$/$R$ pull $z$ toward relationally consistent regions — the *content* of $g$ directly changes the geometry of $z$.
3. **Control-layer interaction.** Symbolic confidence, via $r_\psi$, participates in deciding how $z$ itself is used (the subject of Q3).

### The positive answer to "what is the point?"

If $z$ and $g$ were generated in parallel without interaction:

- **(a)** You would still need a mechanism to decide which description to trust, when — and the trust signal $r_\psi$ itself requires the interaction. Parallelism does not remove the controller problem; it creates it.
- **(b)** You save no computation. The gains come precisely from *not* rolling $z$ through quiescent stretches: the macro head jumps 15 frames to a stable endpoint in a single decision, which a pure-$z$ rollout reaches only by paying 15 steps of compute and 15 steps of compounded drift.
- **(c)** You lose the error-correction channel: the rollout of $z$ is anchored by symbolic constraints at reliable times (SPSG, terminal anchor at equilibria).

Parallel generation yields two *redundant* descriptions; the joint architecture yields the ability to *select the prediction problem per step*. Every claimed benefit — the compute frontier, drift reduction, endpoint correctness — comes from the selection, not the redundancy.

### The pipeline in one view

```
Perception (parallel — Hamid is right):   o_t → z_t ,  o_t → G_t
Control (interaction):                    z_t + conf(G_t) → h_t → π_κ → (Δ_k, α_k)
Dynamics (interaction):                   F^{Δ_k,α_k}(z, a, S^μ) → predict → map back to shared z_{t_{k+1}}
```

---

### Suggested response (ready to say)

> At the perception front-end, yes — $z$ and $g$ are extracted in parallel, and that's by design. But the interaction happens at the dynamics and control layers, not the perception layer. Three concrete interactions: the continuous dynamics is conditioned on the symbolic state, $F_\theta(z_t, a_t, S^\mu_t)$, and the event predictor takes both, $G_\omega(S^M, z, a)$; the symbolic losses shape the geometry of $z$ during training through SPSG and the closure-consistency maps; and the symbolic confidence, via $r_\psi$, participates in deciding how $z$ itself is used. If the two were truly parallel and non-interacting, I'd agree there is no point — you would hold two redundant descriptions and still lack a trust mechanism. The gain comes from *not* rolling $z$ through quiescent stretches: the macro head jumps fifteen frames to a stable endpoint in one decision, which a pure-$z$ rollout can only reach by paying fifteen steps of compute and drift.
>
> On the mechanics of $\alpha$: within one decision step, the chosen $\alpha_k$ defines the target space of that step's prediction — predicting the state at $t_k{+}\Delta_k$ *in* representation $\alpha_k$. The result is then mapped back into the shared rollout state $z$, and the next decision re-selects $\alpha$ freely from the new $h$. So there is no persistence assumption across steps — consecutive decisions can switch representation, which is exactly the coupled switch in the cascade timeline. And $h$ is not $g$: $h$ is the controller's summary state — the latent plus diagnostics — while $g$ contributes only indirectly, through its confidence entering $r_\psi$. Whether the previous pair $(\Delta_{k-1}, \alpha_{k-1})$ should be fed into $h$ explicitly is a legitimate architectural question — we'd treat a recurrent controller as an ablation rather than a default.

---

## Cross-references in the project documents

- **Cross-conditioned predictors** $F_\theta(z_t, a_t, S^\mu_t)$ and $G_\omega(S^M, z, a)$: `research_proposal.md`, §4.3 (Component C).
- **Map-back to shared rollout state**: `method_proposal_presentation.md`, "Three state representations" slide.
- **Controller inputs / $h$ definition**: `research_proposal.md`, §4.4; notation table in the presentation backup.
- **Symbolic losses shaping $z$**: `research_proposal.md`, §4.2 and §5.1 (SPSG as structured regularizer).
- **Parallel perception front-end**: `symbolic_representation_QA_and_pipeline.md`, §2.2 (Tier-2 architecture).
- **Regime switching across steps**: cascade timeline slide ("the coupled switch").

## Action items

1. Add to the anticipated-questions list in `method_innovation_in_one_paragraph.md`:

**Q: "If the controller selects α but z and g never interact, why not generate both in parallel — what is the point?"**
The front-end is parallel by design; the interaction lives in the dynamics and the controller. The continuous predictor is conditioned on the symbolic state and the event predictor on both; symbolic losses shape the geometry of $z$ in training; symbolic confidence gates how $z$ is used. Parallel generation would still need a trust mechanism, would save no compute, and would lose symbolic anchoring of the rollout. The gains come from per-step selection of the prediction problem — including the decision to skip fine-grained rollout entirely — not from holding two redundant descriptions.

2. Consider adding the **recurrent-controller ablation** (feeding $(\Delta_{k-1}, \alpha_{k-1})$ into $h$) to the architectural ablation list in Stage 3 — low cost, and it preempts the follow-up question about cross-step dependence of $\alpha$.


# How Is the World Model Itself Trained? Teacher Forcing, Gradient Flow, and the State-Carrier Decision — Q5

*Working notes for supervisor discussion, 31 July 2026*

---

## Q5. If a rollout step moves from latent embedding to symbolic representation, are we forced into teacher forcing?

**The question (our own, surfaced during preparation):**

> If the middle of a rollout transitions from the latent embedding $z$ into a symbolic representation $g$, and the latent→symbolic step passes no gradient, does that force teacher-forced training rather than autoregressive training? We know how the controller is trained (Stage-1 scoring → distillation) and the three $\alpha$-parsers are pretrained — but it seems the training regime of the world model $F$ itself has not actually been pinned down. Is that right?

### Short answer

Yes — the diagnosis is correct: the proposal fixes controller training and extractor pretraining, but the world model's own training regime was left implicit in the single sentence "each head maps its prediction back to the shared rollout state $z$." That gap is now closed by two decisions:

1. **The state carrier is always $z$; symbolic heads are supervised readouts, never the carrier.** The gradient-break scenario only exists if symbols carry the rollout state — in this design, they do not.
2. **$F$ is trained teacher-forced, multi-task over the $(\Delta, \alpha)$ grid** — but this is a deliberate choice (the standard JEPA regime), not a concession forced by missing gradients. Autoregressive rollout lives at test time and in an optional fine-tuning phase, and it is differentiable throughout, because the carrier is $z$.

---

### Where the gradient could break — and why it doesn't

The only potential non-differentiable point is the moment a micro/macro-mode prediction **re-enters the rollout**. If "mapping back to $z$" meant argmax-ing predicted predicates into hard symbols, assembling a discrete graph, and re-encoding it into $z$, the argmax would sever the gradient.

The design eliminates this by construction:

> **Design decision (locked, not an ablation): whatever $\alpha$ the controller selects, $F$ always also outputs the continuous carrier $\hat{z}_{t+\Delta}$; the symbolic head is an additional supervised readout.**

Every prediction step is dual-output:

$$F_\theta^{\Delta,\alpha}(z_t, a_{[t_k,t_{k+1})}) \;\longrightarrow\; \underbrace{\hat{z}_{t+\Delta}}_{\text{rollout carrier — always produced}} \;+\; \underbrace{\text{mode-head output}}_{\hat{S}^\mu \;\text{or}\; (\hat{S}^M, \hat{\Delta}, \hat{e}) \text{ — readout, enters only the loss}}$$

The next decision continues from $\hat{z}_{t+\Delta}$. The path $z_t \to \hat{z}_{t+\Delta}$ is then an ordinary differentiable feed-forward computation — no argmax in the state path, no straight-through estimators, no Gumbel tricks required for the world model itself.

This is the same principle as the Q3/Q4 defense: **symbols are constraints and observations attached to $z$, never integrated forward in time.** The state-carrier decision makes that principle mechanically explicit.

### Why teacher forcing is a choice, not a compromise

In JEPA-style world models, the base training regime *is* teacher forcing. The duration-normalized objective unrolls into a collection of supervised sub-tasks:

$$\text{for each } (t, \Delta, \alpha) \text{ sample:} \qquad z^{\text{true}}_t \xrightarrow{\;F^{\Delta,\alpha}\;} \hat{z}_{t+\Delta} \;\;\text{aligned to}\;\; \mathrm{sg}(z^*_{t+\Delta})$$

Each sub-task is single-step supervision from a true encoded state — no gradient-through-time is ever required during base training, so the symbolic-break question never arises there. This is also exactly the regime the Stage-1 exhaustive scoring presupposes (score every pair on oracle trajectories), keeping the pipeline self-consistent.

### The training recipe (to be written into the proposal)

- **Phase A — $F$ pretraining (Stage 1).** Trajectories are encoded by the target encoder ($z^*$ sequence); engine state gives $S^{\mu*}, S^{M*}$. Sample $(t, \Delta, \alpha)$ uniformly over the grid; train the dual-output $F$ with $\mathcal{L}_{\text{pred}}$ (JEPA-style stop-gradient latent alignment) plus the selected mode's head loss (cross-entropy), weighted by $\Delta/T$. Fully teacher-forced, fully differentiable.
- **Phase B — controller-coupled training (Stage 3).** Same objective; the sampling distribution over $(\Delta, \alpha)$ switches to the controller's output (or stays uniform, per ablation design); the reliability gate $\omega_\psi$ modulates the symbolic losses.
- **Phase C — optional anti-exposure-bias fine-tuning.** Teacher forcing's standard cost is the train/test mismatch (train on true states, test on its own predictions). Three mitigations are already structural: multi-horizon supervision (direct targets at $\Delta{=}15$, not just composed one-step predictions), macro jumps that supervise endpoints directly, and the terminal anchor. If pilot drift is still unacceptable, add **short-window autoregressive fine-tuning / scheduled sampling through the $z$ carrier only** — feasible precisely because the carrier path is differentiable end-to-end.
- **Stage 4 — optional end-to-end controller.** The *only* place gradients must cross a discrete choice is the end-to-end controller variant (Gumbel-softmax / REINFORCE-style). That is one reason it is optional; by default only $\pi_\kappa$ would be touched there, not $F$.

### What is locked vs. still open

| Item | Status | Note |
|---|---|---|
| Controller training | Locked | Stage-1 exhaustive scoring → distillation into $\pi_\kappa$ |
| $\alpha$-parser / head pretraining | Locked | Supervised, then frozen |
| $F$'s training regime: teacher-forced multi-task over the $(\Delta,\alpha)$ grid | **Locked by this document** | To be written into proposal §4.1 / §6 |
| State carrier is always $z$; symbols are readouts (dual-output) | **Locked by this document** | Eliminates the gradient break by construction |
| $\mathcal{D}_0 = \{1, 5, 15\}$ | Locked provisionally | Rescale only if pilot capture rate requires |
| Phase C: scheduled sampling / short-window AR fine-tune | Open — decide after pilot | Depends on measured drift |
| Per-mode loss weights | Open | Tuned in Stage 3 |
| Stage 4 end-to-end: touches $F$ or only $\pi_\kappa$ | Optional | Default: only $\pi_\kappa$ |

### Suggested response (ready to say)

> Good catch — the controller and extractor training were pinned down, but the world model's own regime was implicit. It is now fixed by two decisions. First, the state carrier is always the continuous latent $z$: whatever $\alpha$ is selected, $F$ produces $\hat{z}_{t+\Delta}$ as the rollout state, and the symbolic head is a supervised readout that enters only the loss — so there is never an argmax in the state path and the gradient never actually breaks. Second, $F$ is trained teacher-forced as a multi-task problem over the $(\Delta, \alpha)$ grid — the standard JEPA regime of single-step supervision from true encoded states — which is a deliberate choice, not a workaround forced by non-differentiability. Autoregressive rollout exists at test time, and optionally as a short-window fine-tuning phase for exposure bias; both run through the differentiable $z$ carrier. The only place gradients must cross a discrete choice is the optional end-to-end controller variant, which is exactly why that variant is optional.

---

## Cross-references in the project documents

- **"Heads map back to shared $z$"** (the sentence this document makes precise): `method_proposal_presentation.md`, "Three state representations" slide.
- **Duration-normalized multi-task objective**: `research_proposal.md`, §4 (boxed objective); presentation, "Unified training objective" slide.
- **Stage-1 scoring regime consistency**: `research_proposal.md`, §6.1; presentation, "Staged implementation" slide.
- **Symbols as constraints, not state carrier** (Q3/Q4 principle): `symbolic_noise_QA.md` Layer 3; `z_g_interaction_and_alpha_mechanics_QA.md`.
- **Controller distillation**: `controller_router_QA.md`, Q2.
- **Optional end-to-end controller**: `controller_router_QA.md`, Q2 (Stage 4); presentation, "How $(\Delta,\alpha)$ trains the JEPA world model" slide.

## Action items

1. **Update `research_proposal.md` §4.1**: add the dual-output statement — $F^{\Delta,\alpha}$ always produces the carrier $\hat{z}_{t+\Delta}$; symbolic heads are supervised readouts entering only the loss.
2. **Update `research_proposal.md` §6**: add an explicit "World-model training regime" paragraph (Phase A/B/C as above) so the recipe no longer lives only in meeting notes.
3. **Update the presentation**: one sentence on the "Three state representations" slide — "the carrier is always $z$; symbolic heads are readouts, so no discrete step ever sits in the rollout path."
4. Add to anticipated questions in `method_innovation_in_one_paragraph.md`:

**Q: "Does the switch from latent to symbolic representation break gradients and force teacher forcing?"**
No discrete step sits in the rollout path: $z$ is always the state carrier, symbolic heads are supervised readouts that enter only the loss. Teacher forcing over the $(\Delta,\alpha)$ grid is the deliberate base regime (standard for JEPA), not a workaround; autoregressive rollout is differentiable throughout because the carrier is continuous, and exposure bias is handled by multi-horizon supervision, endpoint-level macro prediction, and — if needed — short-window fine-tuning through the carrier.
