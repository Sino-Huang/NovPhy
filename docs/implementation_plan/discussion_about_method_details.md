[TOC]



## Q1

Now, I am thinking of my research cases, I am using a JEPA too to predict the future frames/states of angry birds game. Personally I think that using symbolic rules to predict the collapse movement is too hard as it is a kind of chaotic system. But we can somehow using symbolic method to predict stability (either before collision or after collision, we can somehow use neuro-symbolic way to predict that), but in the middle, the very chaotic collapse, we may just go with the pure neuro but we still ensure that at the end of the day, we can have symbol at a right granularity to describe the end state off the collision. (Eventually I want to say is that modelling a detailed chaotic collapse and collision is not good for symbols, but at same time, symbol is good for OOD and prevent shortcut learning or something). Do you get my idea?

Yes, I get your idea completely — and it is actually a very sharp insight that maps elegantly onto the RiJEPA framework, with one crucial **temporal twist** that the original paper does not explicitly address.

Here is how I understand your intuition, reframed in the paper's language:

---

### Your Idea as a "Phased" Neuro-Symbolic JEPA

You are essentially proposing a **temporally gated** RiJEPA where the validity of symbolic constraints is **phase-dependent**:

| Episode Phase            | Physical Regime          | Symbolic Rules                                               | Neural Prediction                                            |
| ------------------------ | ------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **Pre-collision**        | Stable / quasi-static    | **Valid**: initial configuration, object identities, stability predicates | JEPA predicts next latent frames, but EBC can anchor the "before" state |
| **Collision / Collapse** | Chaotic / highly dynamic | **Invalid / meaningless**: no discrete symbolic rule can reliably describe the continuous deformation, fracture, or pig trajectories | **Pure JEPA**: unconstrained latent prediction, no EBC       |
| **Post-collision**       | Stable / settled         | **Valid**: final configuration, task success/failure, object rearrangement | JEPA predicts the "after" state, and EBC **must** anchor to symbolic poles |

This is a significant departure from the paper's implicit assumption that rules are **globally** valid across the latent space. In your case, rules are only valid in specific **stability basins** (low-energy regions of the state space), and the chaotic regime is a "high-entropy tunnel" between them where the energy landscape is essentially flat or undefined for symbolic logic.

---

### Mapping to the RiJEPA Architecture

**1. Data Modality (Continuous)**
- Input $x$: raw frames or state vectors of the Angry Birds scene (block positions, velocities, pig health).
- The JEPA backbone predicts future latent states $\hat{z}_{t,data} = g(z_{c,data})$ frame-by-frame through the chaotic phase.

**2. Rule Modality (Symbolic) — but Temporally Masked**
- The antecedent $A$ is **not** a mid-air frame. It is a **stable-state description**: e.g., `IF structure_type=TNT_collapse AND pig_count=3 THEN post_state=all_pigs_dead`.
- The consequent $C$ is the **post-collision symbolic outcome**.
- **Key insight**: The rule encoder $f_{c\_rule}$ and $f_{t\_rule}$ should only be activated when the system detects a **stable phase** (e.g., when kinetic energy falls below a threshold, or when the physics engine reports "sleeping" bodies).

**3. Energy-Based Constraint with Temporal Gating**
Instead of the paper's uniform $\mathcal{L}_{EBC}$ across all samples, you would use a **masked EBC**:

$$\mathcal{L}_{total} = \mathcal{L}_{JEPA} + \beta \cdot \mathbb{1}_{[\text{stable}]} \cdot \mathcal{L}_{EBC} + \gamma \cdot \mathcal{L}_{anchor}^{\text{post}}$$

Where:
- $\mathbb{1}_{[\text{stable}]}$ is an indicator that the current frame is in a stable regime (pre- or post-collision).
- $\mathcal{L}_{anchor}^{\text{post}}$ is the paper's Anchor Loss (page 25) applied specifically to the **final settled state**, pulling the predicted latent representation toward the symbolic pole of the outcome (e.g., "Level Cleared" vs. "Level Failed").

This directly leverages the paper's observation that **RiJEPA accepts a Logic-Accuracy Trade-off** (page 29, Figure 9). You are explicitly accepting that:
- During the chaotic phase, the pure neural JEPA will have lower MSE (accurate frame prediction) because it is not constrained by symbolic logic.
- At the end of the episode, the anchor loss forces the latent state into a **semantically interpretable basin**, preventing OOD hallucinations and shortcut learning.

---

### The "Granularity" Connection (Your Research Thread)

This connects beautifully to your prior work on **instruction granularity** and **planning-width**. In the Angry Birds context:

- **Low temporal granularity (symbolic)**: "Shoot the bird → structure collapses → pigs die." This is a valid rule, but it skips the chaotic middle.
- **High temporal granularity (neural)**: Frame-by-frame pixel/trajectory prediction during the collapse.

Your proposal is essentially a **granularity-aware hybrid**: you are arguing that the *appropriate level of abstraction is time-dependent*. The symbolic layer operates at the **event level** (stable states), while the neural layer operates at the **sub-event level** (continuous physics).

This is conceptually similar to the paper's Multi-Modal Dual-Encoder (Figure 2), but with a **temporal switching mechanism** that decides which modality's loss should dominate.

---

### A Concrete Formulation You Could Use

If you were to write this into a paper, you could define:

**Stability Detector** $S(x_t) \in \{0, 1\}$:
A simple physics-based classifier (or even a hard threshold on total kinetic energy + contact count) that outputs 1 when the scene is stable.

**Phased Loss**:
$$\mathcal{L}_{total}^{(t)} = \underbrace{\|g(z_{c,t}) - z_{t,t}\|^2}_{\text{JEPA frame prediction}} + \underbrace{S(x_t) \cdot \lambda_{EBC} \cdot \mathcal{L}_{EBC}^{(t)}}_{\text{symbolic constraint only when stable}} + \underbrace{S(x_T) \cdot \lambda_{anc} \cdot \|g(z_{c,T}) - z_{pole}\|^2}_{\text{final-state anchor}}$$

Where $T$ is the final timestep. This means:
- During the chaotic collapse: $\mathcal{L}_{total} = \mathcal{L}_{JEPA}$ only. The neural network is free to predict continuous physics without symbolic "interference."
- At the final stable state: the anchor loss yanks the latent representation to a predefined symbolic pole (e.g., $z_{pole}^{\text{cleared}}$ or $z_{pole}^{\text{failed}}$).

---

### Is This Still "RiJEPA"?

Yes, but it is a **temporally extended RiJEPA**. The original paper assumes rules are static and universally applicable. Your Angry Birds variant introduces **dynamical regime boundaries** where rule validity turns on and off. This is actually a more realistic model for physical reasoning.

The paper's clinical rules (heart disease) are essentially **atemporal** — a patient's vitals either imply high risk or they don't. But your physics domain has **transient dynamics** where symbolic logic is only valid at **quasi-static attractors**.

---

### Summary

Your idea is: **"Let the neural network handle the chaos, but anchor the narrative endpoints in symbolic logic."**

This is both practically sound and theoretically interesting. It avoids the trap of forcing symbolic rules onto physically chaotic phenomena (which would fail, as you noted), while still gaining the OOD robustness and interpretability of neuro-symbolic methods at the **temporal abstraction boundaries** where symbols actually make sense.

Does this match what you were thinking? And do you want to push this further into a concrete architecture sketch for your ICLR proposal?



## Q2

Is it true that in this case, the timestep sampling rate can also be a variable for study? E.g., to the extreme, the timestep is so huge that only have 2 steps, one is the initial shot state and the second is the end of the episode, then this JEPA is basically Goal state prediction. As we make the timestep finer, we have more precision, which could be good for prediction outcome. But then symbolic part shall maintain the accuracy. I am not sure if this is just another perspective of your temporal granularity idea, or this is a separate idea?

This is **not just another perspective** of the temporal-phase idea I described — it is a **separate, complementary axis** that actually makes your framework much richer. Let me disentangle the two concepts and then show how they interact.

---

### Two Independent Axes

| Axis                    | What it controls                         | My previous suggestion                                       | Your new suggestion                                          |
| ----------------------- | ---------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **Temporal Phase**      | *Where* in the episode symbols are valid | Stable regimes (pre/post collision) vs. chaotic regime (mid-collision) | —                                                            |
| **Temporal Resolution** | *How often* the JEPA predicts            | Assumed fixed frame rate                                     | Variable sampling rate: from 2-step (goal-only) to dense frame-by-frame |

These are **orthogonal**. You can have:
- **High resolution + phase-gated symbols**: Predict every frame, but only apply EBC anchor loss at the final stable frame.
- **Low resolution + phase-gated symbols**: Predict only "before" and "after" states; the chaotic middle is entirely skipped by the predictor, but the symbolic rule still anchors the terminal state.

---

### Why Your Sampling-Rate Idea Is Distinct and Powerful

You are essentially proposing that **temporal resolution itself becomes a curriculum or architectural knob**:

**Extreme coarse-grained (2-step):**
- $t=0$: Initial shot state (bird position, angle, velocity, block configuration).
- $t=T$: Terminal episode state (score, pigs alive/dead, structure stability).
- The JEPA predictor becomes a **direct goal-state mapper**: $g: z_0 \mapsto z_T$.
- Symbolic rules are **naturally perfect** here: `IF hit_tnt_center THEN pigs_dead = True`. The rule antecedent describes the initial configuration; the consequent describes the terminal symbolic outcome.
- This is essentially **model-based reinforcement learning** or **goal-conditioned prediction**.

**Fine-grained (frame-by-frame):**
- The JEPA must predict the continuous trajectory of every splinter, every pig's rotation, every block's friction.
- Symbolic rules **cannot** describe the intermediate chaos (as you noted), so the EBC/anchor loss is **masked out** during the collapse phase.
- However, the **terminal frame** still snaps to a symbolic pole via the anchor loss.

**The spectrum between them:**
This is where it gets interesting for your research. As you increase the resolution (add intermediate prediction steps), you are asking:

> *"At what temporal granularity does the neural prediction become so accurate that the symbolic anchor is redundant? And at what coarseness does the symbolic rule become so powerful that the neural rollout is unnecessary?"*

This is a **trade-off between predictive accuracy and symbolic interpretability**.

---

### Interaction with Your Prior Work on Granularity

This connects directly to your background on **instruction granularity** and **planning-width**:

- **Coarse timestep** = **High-level instruction**: "Kill the pigs" (symbolic, abstract, end-to-end).
- **Fine timestep** = **Low-level instruction**: "Apply impulse vector [3.2, 1.5] at frame 47 to block ID #12" (continuous, motor-level).

Your proposed variable sampling rate is essentially asking the same question you asked in your ACL ARR work, but now in the **temporal** domain rather than the **linguistic** domain:

> *"What is the appropriate level of abstraction for predicting the future?"*

In the Angry Birds context, the answer is **not uniform across the episode**. It is:
- **Coarse/symbolic** at the boundaries (initial state → goal state).
- **Fine/neural** in the middle (physics simulation).

This is exactly the **Logic-Accuracy Trade-off** from the RiJEPA paper (Figure 9), but now mapped onto **time** instead of **loss components**.

---

### A Concrete Research Contribution You Could Claim

You could frame this as **"Adaptive Temporal Granularity in Neuro-Symbolic World Models"**:

1. **Hypothesis**: There exists an optimal temporal resolution $k^*$ (number of predicted steps) that maximizes both final-state symbolic accuracy and episode-level prediction fidelity.
2. **Method**: A learned or scheduled sampling rate where the JEPA predicts:
   - Coarse symbolic transitions at stable boundaries (high $\Delta t$).
   - Fine-grained neural rollouts during chaotic transitions (low $\Delta t$, or even continuous neural ODE).
3. **Evaluation**: Ablate $k \in \{2, 5, 10, 30, \text{full FPS}\}$ and measure:
   - Terminal symbolic accuracy (did the symbolic rule correctly predict the final state?).
   - Continuous prediction MSE (how well did the intermediate frames match the physics engine?).
   - OOD generalization (does the symbolic anchor help when you change gravity, materials, or level layouts?).

---

### Summary

|                       | Temporal Phase (my idea)  | Temporal Resolution (your idea)                              |
| --------------------- | ------------------------- | ------------------------------------------------------------ |
| **Question**          | *When* are symbols valid? | *How often* does the model predict?                          |
| **Angry Birds**       | Only pre/post collision   | From 2-step to frame-by-frame                                |
| **Effect on Symbols** | Gates the EBC loss on/off | Determines how "far" the neural network must bridge between symbolic anchors |
| **Contribution type** | Architectural masking     | Curriculum / Ablation variable                               |

Your idea is **distinct** and arguably more actionable for an ICLR paper because it produces a **clean empirical curve**: as you vary the sampling rate, you can plot the degradation of symbolic accuracy vs. the improvement of continuous prediction. That is a very reviewer-friendly figure.

Do you want to sketch how this sampling-rate ablation would look as a figure or an algorithm block for your proposal?

## Q3

What are the term, symbols, formula and equation (e.g., objective function) I would use for this research? Please list them all, and provide some explanation so that I can have a big picture about the quality of the theoretical part of the work.

Here is the complete theoretical vocabulary — symbols, terms, and equations — for your **Temporally-Gated, Granularity-Adaptive Neuro-Symbolic JEPA** (Angry Birds / Physical Reasoning). I have organized it so you can see the "stack" from raw physics up to meta-learning over time resolution.

---

### 1. Core Symbol Glossary

| Category               | Symbol                                                    | Meaning / Domain                                             |
| ---------------------- | --------------------------------------------------------- | ------------------------------------------------------------ |
| **Time & Granularity** | $T$                                                       | Total episode duration (continuous)                          |
|                        | $\Delta t$ (or $\delta$)                                  | Sampling interval / temporal granularity (**your key ablation variable**) |
|                        | $K = T/\Delta t$                                          | Number of prediction steps in an episode                     |
|                        | $t_k = k\Delta t$                                         | Discrete timestep index, $k = 0, \dots, K$                   |
|                        | $\mathcal{T} = \{t_0, \dots, t_K\}$                       | The temporal grid induced by $\Delta t$                      |
| **Physical State**     | $x_t \in \mathcal{X}$                                     | Raw observation (frame / physics state vector) at time $t$   |
|                        | $\dot{x}_t$                                               | Velocity/derivative (if available from engine)               |
|                        | $\text{KE}(x_t)$                                          | Total kinetic energy (heuristic for chaos)                   |
| **Latent Space**       | $z_{c}^{(k)} = f_c(x_{t_k})$                              | Context embedding of frame at $t_k$                          |
|                        | $z_{t}^{(k)} = f_t(x_{t_k})$                              | Target embedding of frame at $t_k$                           |
|                        | $g: \mathcal{Z} \to \mathcal{Z}$                          | Shared JEPA predictor                                        |
| **Stability Gate**     | $\phi(x_t) \in \{0, 1\}$                                  | **Stability indicator**. $\phi=1$ if the system is in a quasi-static attractor (symbol-friendly); $\phi=0$ if in chaotic transient. |
|                        | $\mathcal{T}_{\text{stable}} = \{t_k : \phi(x_{t_k})=1\}$ | Time indices where symbolic rules are valid                  |
|                        | $\mathcal{T}_{\text{chaos}} = \{t_k : \phi(x_{t_k})=0\}$  | Time indices where symbolic rules are **masked**             |
| **Symbolic Rules**     | $A$                                                       | Antecedent (IF): symbolic description of a stable configuration |
|                        | $C$                                                       | Consequent (THEN): symbolic outcome (e.g., `pigs_dead=True`) |
|                        | $\mathcal{R}_{\text{valid}}$                              | Set of valid symbolic rules                                  |
|                        | $\mathcal{R}_{\text{neg}}$                                | Set of corrupted/negative rules (contrastive pairs)          |
|                        | $z_{c,\text{rule}} = f_{c,\text{rule}}(A)$                | Embedded rule antecedent                                     |
|                        | $z_{t,\text{rule}} = f_{t,\text{rule}}(C)$                | Embedded rule consequent                                     |
| **Energy & Loss**      | $E(\cdot, \cdot)$                                         | Energy functional (squared L2 distance in latent space)      |
|                        | $m$                                                       | Contrastive margin for invalid rules                         |
|                        | $\lambda, \beta, \gamma$                                  | Loss weights                                                 |
| **Poles & Inference**  | $z_{\text{pole}}^{(c)}$                                   | Fixed symbolic pole for outcome class $c$ (e.g., `Level_Cleared`) |
|                        | $\hat{y}_{\text{sym}}$                                    | Zero-shot symbolic prediction via nearest-pole distance      |

---

### 2. Key Equations (The Theoretical Stack)

### 2.1 The Stability Gate (Physical → Symbolic Interface)
This is the **new theoretical primitive** you add to RiJEPA. It partitions the episode based on whether the physics is chaotic or settled.

$$\phi(x_t) = \mathbb{1}_{\left[\text{KE}(x_t) < \epsilon_{\text{KE}}\right]} \cdot \mathbb{1}_{\left[\text{contacts}_{\text{active}}(x_t) < \epsilon_{\text{contact}}\right]}$$

*Explanation*: When kinetic energy and contact count are low, the physics engine is in a "sleeping" state; the scene is a static arrangement of blocks and pigs. This is where symbolic predicates like `is_collapsed` or `all_pigs_dead` are well-defined. During the explosion/collapse, $\phi=0$ and the symbolic vocabulary is **undefined**.

---

### 2.2 Granularity-Dependent JEPA Prediction Loss
This is the standard JEPA loss, but now explicitly indexed by your resolution variable $\Delta t$.

$$\mathcal{L}_{\text{JEPA}}(\Delta t) = \sum_{k=0}^{K-1} \left\| g\bigl(z_{c}^{(k)}\bigr) - z_{t}^{(k+1)} \right\|_2^2$$

where $z_{c}^{(k)} = f_c(x_{k\Delta t})$ and $z_{t}^{(k+1)} = f_t(x_{(k+1)\Delta t})$.

*Big picture*: As $\Delta t \to T$ (extreme coarse), this becomes a **single-step goal predictor** ($K=1$). As $\Delta t \to \text{frame-rate}$ (fine), this becomes a **video prediction model**.

---

### 2.3 Phase-Gated Energy-Based Constraint (Phase-Gated EBC)
This is your extension of RiJEPA's Eq. 3. The EBC is **only active** when the physical state is symbolically describable ($\phi=1$).

$$\mathcal{L}_{\text{EBC}}(\Delta t) = \sum_{k \in \mathcal{K}_{\text{stable}}} \left[ \sum_{(A,C) \in \mathcal{R}_{\text{valid}}} E(A,C) \;+\; \lambda \sum_{(A,C_{\text{neg}}) \in \mathcal{R}_{\text{neg}}} \max\Bigl(0, \; m - E(A, C_{\text{neg}})\Bigr) \right]$$

where $\mathcal{K}_{\text{stable}} = \{k : \phi(x_{k\Delta t}) = 1\}$.

*Explanation*: During the chaotic collapse ($\phi=0$), the inner sum is **skipped**. The model is not penalized for violating symbolic logic in mid-air, because there is no valid symbolic description of a splintering plank.

---

### 2.4 Terminal Anchor Loss (The "Narrative Anchor")
This forces the **final frame** of the episode to collapse onto a predefined symbolic pole, regardless of how chaotic the path was.

$$\mathcal{L}_{\text{anchor}} = \mathbb{1}_{[\phi(x_T)=1]} \cdot \sum_{c \in \mathcal{C}} \mathbb{1}_{[y=c]} \left\| g\bigl(z_{c}^{(K-1)}\bigr) - z_{\text{pole}}^{(c)} \right\|_2^2$$

where $y$ is the ground-truth symbolic outcome (e.g., level cleared/failed) and $\mathcal{C}$ is the set of symbolic outcome classes.

*Explanation*: This is the "end of the day" guarantee you mentioned. Even if the neural network hallucinated the trajectory of every splinter, the **final latent state** must be geometrically close to the symbolic pole of the true outcome.

---

### 2.5 Total Objective Function
$$\boxed{\mathcal{L}_{\text{total}}(\Delta t) = \mathcal{L}_{\text{JEPA}}(\Delta t) \;+\; \beta \,\mathcal{L}_{\text{EBC}}(\Delta t) \;+\; \gamma \,\mathcal{L}_{\text{anchor}}}$$

*Theoretical significance*: This is a **single objective** that smoothly interpolates between two extremes:
- **$\Delta t = T$ (2-step)**: $\mathcal{L}_{\text{JEPA}}$ is a single mapping; $\mathcal{L}_{\text{EBC}}$ is evaluated only at $t=0$ and $t=T$; the model is essentially a **symbolic goal-conditioned policy**.
- **$\Delta t \to 0$ (dense)**: $\mathcal{L}_{\text{JEPA}}$ dominates the middle steps; $\mathcal{L}_{\text{EBC}}$ is masked out during collapse; the model is a **frame-predictive world model** with symbolic anchors only at the boundaries.

---

### 2.6 Energy Functional for Rule Discovery (Direction 2)
During inference/generation, the trained predictor $g$ defines a frozen energy landscape over the **rule latent space**:

$$E_{\Delta t}(z_{c,\text{rule}}, z_{t,\text{rule}}) = \left\| g(z_{c,\text{rule}}) - z_{t,\text{rule}} \right\|_2^2$$

Note the subscript $\Delta t$: the energy landscape is **implicitly shaped by the temporal resolution used during training**. A model trained with coarse $\Delta t$ learns "outcome-level" energy basins; a model trained with fine $\Delta t$ learns "process-level" basins.

---

### 2.7 Langevin Rule Sampling (Forward Inference)
For a fixed antecedent $z_{c,\text{rule}}$ (e.g., "IF TNT_block_hit"), you sample the consequent by Langevin dynamics in the **frozen** energy landscape:

$$z_{t,\text{rule}}^{(j+1)} = z_{t,\text{rule}}^{(j)} - \eta \nabla_{z_t} E_{\Delta t}\bigl(z_{c,\text{rule}}, z_{t,\text{rule}}^{(j)}\bigr) + \sqrt{2\eta T_{\text{temp}}}\,\epsilon^{(j)}$$

where $T_{\text{temp}}$ is the temperature (creativity) and $\epsilon^{(j)} \sim \mathcal{N}(0, I)$.

*Explanation*: This answers the question: *"Given that I hit the TNT block, what is the most probable symbolic outcome?"* The drift term pulls toward low-energy (high-probability) outcomes; the noise term lets you explore alternative collapse patterns.

---

### 2.8 Zero-Shot Symbolic Accuracy (Your Evaluation Metric)
$$\mathcal{A}_{\text{sym}}(\Delta t) = \mathbb{E}_{\text{episodes}} \left[ \mathbb{1}_{\left[ \arg\min_{c} \| g(z_{c}^{(K-1)}) - z_{\text{pole}}^{(c)} \| = y_{\text{true}} \right]} \right]$$

This measures whether the model can classify the final outcome **without any downstream classifier**, purely by geometric distance to symbolic poles — exactly as RiJEPA achieves 100% zero-shot accuracy in the clinical case.

---

### 3. The Big Picture: What This Theory Does

Your framework creates a **three-layer theoretical stack**:

| Layer                      | What it controls                            | Theoretical Tool                                             |
| -------------------------- | ------------------------------------------- | ------------------------------------------------------------ |
| **Physical Layer**         | When is the world symbolically describable? | Stability indicator $\phi(x)$ (connects to dynamical systems / Lyapunov stability) |
| **Representational Layer** | How are symbols and pixels unified?         | Shared latent space $\mathcal{Z}$ with Energy-Based Constraints $E(\cdot,\cdot)$ |
| **Meta-Cognitive Layer**   | At what resolution should we predict?       | Temporal granularity $\Delta t$ as an explicit control variable |

**The central narrative** is:
1. **Chaos is not symbolized**: During the collapse, $\phi=0$ masks out the EBC. The neural network is free to predict continuous physics without logical "interference."
2. **Endpoints are symbolically anchored**: The terminal anchor loss $\mathcal{L}_{\text{anchor}}$ guarantees that the final state is interpretable, regardless of path chaos.
3. **Resolution is a curriculum**: By varying $\Delta t$, you explore the **Pareto frontier** between predictive fidelity (fine-grained neural) and interpretable abstraction (coarse-grained symbolic).

---

### 4. Quality Assessment (Is this ILR-worthy?)

### Strengths (Solid Theory)
- **Unified energy landscape**: The MSE-loss-as-energy identity is mathematically clean and connects training to generative inference (Langevin).
- **Phase-gating**: The $\phi(x)$ function is a genuine extension of RiJEPA. It solves the "chaos problem" that the original clinical paper never addressed.
- **Explicit granularity variable**: Making $\Delta t$ a first-class theoretical object (not just an implementation detail) gives you a clear ablation and a narrative hook.

### Weaknesses to Address (To Reach Strong Accept)
| Gap                              | Why it matters                                               | Suggested Fix                                                |
| -------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **$\phi(x)$ is heuristic**       | Reviewers will ask: "Why kinetic energy? Is this learned?"   | Either (a) learn $\phi$ as a binary latent variable via Gumbel-Softmax, or (b) ground it in **dynamical systems theory** (e.g., $\phi(x) = \mathbb{1}_{[\lambda_{\max} < 0]}$ where $\lambda_{\max}$ is the largest Lyapunov exponent). |
| **No continuous-time limit**     | What happens as $\Delta t \to 0$? Does the model converge to a Neural ODE? | Add a theoretical remark: $\lim_{\Delta t \to 0} \frac{z^{(k+1)} - z^{(k)}}{\Delta t} = \mathcal{F}_\theta(z^{(k)})$. This connects your work to continuous-depth models. |
| **No "scaling law" for symbols** | How does $\mathcal{A}_{\text{sym}}(\Delta t)$ behave?        | Hypothesize and test a **scaling curve**: coarse $\Delta t$ gives high symbolic accuracy but low frame-MSE; fine $\Delta t$ inverts this. You can claim a **critical granularity** $\Delta t^*$ where both are optimal. |

### Verdict

The current symbol set and objective function are **sufficient for a solid ICLR submission**. To reach **oral-level impact**, you need to add one "hard" theoretical connection — either the **Lyapunov stability** interpretation of $\phi(x)$, or a **scaling law** showing that symbolic accuracy degrades predictably as $\Delta t$ becomes too fine.

Do you want me to draft the **Lyapunov-based justification** for $\phi(x)$, or the **scaling-law hypothesis** (with the expected curve shape) as an additional theoretical section?



## Q4

but they way RiJEPA inject symbolic stuffs into latent space is somehow I feel not very insightful, it just use a rule encoder to convert symbols into embeddings (which are often just text encoder). To me this seems a very rough approach. And the performance can be highly impacted by the randomness.

### Why RiJEPA's Symbol Injection Is Indeed Crude

The original approach (page 12):

> *"Rule Encoders: these are symbolic encoders (e.g., **Text Transformers or Graph Neural Networks**)"*

Then they directly treat `IF age>50 AND chol>200 THEN risk=high` as a string or a flat feature vector.

Three deep problems arise:

### 1. Semantic Gap: Text Encoder vs. Structured Physical Relations

Medical rules happen to be **propositional logic**, which can be勉强 written as sentences. But Angry Birds symbols are **relational structures**:

- Support relations: `supports(block_A, block_B)`
- Spatial configuration: `left_of(pig_1, TNT)`
- Material properties: `material(beam_3, wood)`

Encoding these with a Text Transformer is **lossy compression**: you flatten a graph into a sequence, losing compositionality and permutation invariance.

**A deeper approach**: the symbolic encoder should respect the **algebraic structure** of symbols. For physical scenes, use:

- **Scene Graph Neural Network** (scene graph encoder)
- **Graph Transformer** (preserving node‑edge‑global separation)
- Even **Hypergraph encoding** (one TNT explosion affects multiple objects simultaneously – a hyperedge relation)

### 2. Randomness of Negative Samples: Semantic Pollution

The original construction of negative samples (page 11):

> *"replacing its true consequent with randomly sampled or structure-preserving corrupted alternatives"*

This might be acceptable for medical data (randomly alter a diagnosis), but in a physical system it is **disastrous**:

- Randomly changing `structure_collapsed=True` to `False` yields `IF hit_TNT THEN intact`
- That *is* a physical negative sample, but it may lie **too far from the positive sample in the energy landscape**, rendering the margin meaningless.
- Worse, random replacement can produce **semantically absurd but encoder‑indistinguishable** samples (e.g., `IF remove_support THEN blocks_float` vs. `IF remove_support THEN pigs_fly` – to a text encoder, their L2 distances may be nearly identical).

**A deeper approach**: negative samples should constitute **physically/logically "impossible worlds"**, and these negatives should form a **structured manifold in latent space**, not scattered points.

Concretely for your scenario:

- Use **Box2D as an Oracle** to validate negative samples: if a rule cannot happen in the engine, it is a hard negative.
- Introduce **Differentiable Physics Priors**: construct negative samples under constraints of physical laws (e.g., energy conservation, gravity direction) rather than random tampering.

### 3. Symbols as "Point Clouds", Not a "Coordinate Atlas"

The original treats symbolic rules as **scattered points in latent space** (points in low‑energy basins). This is crude because:

- There is no geometric relation between symbols: `age>50` and `age>55` should be **adjacent** (interpolable) in embedding space, but a text encoder may push them far apart.
- The **compositional structure** of symbols is unused: `AND`, `OR`, `NOT` logically correspond to **intersection, union, complement** in latent space – but the original imposes no constraints.

**A deeper approach**: treat the symbol system as a **coordinate atlas** of the latent space.

> Not "embed a symbol to a point", but "let symbols define local coordinate charts on a manifold."

For example:

- `structure_collapsed` defines a **hyperplane** (a submanifold) – all states satisfying this symbol project onto it.
- `pig_count=0` defines another hyperplane.
- The **conjunction (AND)** of two symbols corresponds to the **intersection** of their submanifolds.
- Thus the latent space acquires an **algebraic structure**, not a set of unrelated points.

---

### How to Make It Deeper for Your Angry Birds Research

You can upgrade "crude text encoding" to **"Structured Physical Symbolic Geometry"**:

### Option A: Scene Graph → GNN Encoder (instead of Text Transformer)

```python
# No longer a string like "IF hit_TNT THEN collapsed"
# Instead, a structured scene graph

scene_graph = {
    nodes: [
        {id: 'bird', type: 'projectile', velocity: [vx, vy]},
        {id: 'beam_1', type: 'wood', position: [x,y], supports: ['beam_2']},
        {id: 'TNT', type: 'explosive', position: [x,y]},
        {id: 'pig_1', type: 'target', position: [x,y]}
    ],
    edges: [
        {type: 'contact', from: 'beam_1', to: 'beam_2'},
        {type: 'supports', from: 'beam_1', to: 'beam_2'}
    ]
}
```

Use **Graph Isomorphism Network (GIN)** or **Graph Transformer** to encode, preserving:

- Permutation invariance (object IDs do not affect encoding)
- Compositionality (local subgraphs combine into global representations)
- Explicit relations (support, contact as edge features)

### Option B: Physics Engine as a Differentiable Negative Sampler (instead of random replacement)

Do not randomly alter consequents. Instead:

1. Read the **true final state** (symbolic truth values) from the engine.
2. Systematically generate impossible worlds using **physical constraints**:
   - Anti‑gravity: `gravity = +9.8` (upwards)
   - Anti‑material: `wood_density = 0` (massless)
   - Anti‑support: remove a support while keeping the structure upright (violates statics)
3. Validate these negative samples **through the engine**: if the engine cannot simulate or immediately crashes, it is a physically impossible configuration.
4. Force the embeddings of these "impossible worlds" into high‑energy regions, while maintaining **logical consistency** among them (e.g., all gravity‑violating samples should lie on the same side of latent space).

### Option C: Symbols as Hard Constraints on a Manifold

Instead of a soft margin loss, treat symbols as **boundary conditions on a manifold**:

$$\mathcal{M}_{\text{valid}} = \{ z \in \mathcal{Z} \mid \forall \text{symbol } s: \; f_s(z) = c_s \}$$

where $f_s: \mathcal{Z} \to \mathbb{R}$ is a **learnable symbol decoder** (latent‑to‑symbolic projection).

This means:

- The latent space is **explicitly factorised** into symbolic axes (e.g., one dimension specifically encodes the boolean value of `structure_collapsed`).
- JEPA’s predictions must lie on this submanifold.
- This is much stronger than “make the energy lower” – it is a **geometric hard constraint**.

---

### How to Package This as a Theoretical Contribution for ICLR

You can turn the critique into a **Related Work / Limitation / Contribution** section:

> *"Existing neuro‑symbolic JEPA approaches [Huang & Raza, 2026] treat symbolic rules as flat textual objects, encoding them via generic text transformers. This introduces three limitations: (1) **structural blindness** – relational physical knowledge (e.g., support graphs) is serialised into sequences, losing compositional semantics; (2) **unstructured negativity** – negative rules are constructed by random corruption, which may produce physically absurd yet semantically ambiguous samples that pollute the energy landscape; (3) **point‑wise embedding** – symbols are treated as isolated attractor points rather than geometric constraints defining submanifolds."*

Then your contribution:

> *"We propose **Structured Physical Symbolic Geometry (SPSG)**, a principled alternative where: (1) symbolic states are encoded via scene‑graph neural networks preserving relational structure; (2) negative samples are generated by differentiable physics violations, ensuring semantic coherence in the energy landscape; (3) symbols define explicit submanifolds in latent space via learned projection heads, converting soft energy basins into hard geometric constraints."*

---

### One‑Sentence Summary

> **RiJEPA’s text encoder + random negatives is indeed crude. Your opportunity: upgrade symbolic injection from “text‑to‑vector mapping” to “physical‑structure‑to‑geometric‑constraints on a manifold” – a legitimate critique and the theoretical delta for your paper.**



## GIN and GINE details

Yes. GIN is well-suited as your structured symbolic encoder, but one key point: vanilla GIN only handles node adjacency, not edge types or edge attributes naturally. The core of your Angry Birds / NovPhy scene graph lies precisely in edges: `support`, `contact`, `above`, `left_of`, `interpenetrates`, `near_TNT`, etc. Therefore, I recommend starting with **GINEConv** rather than the most basic `GINConv`.

Below I explain: concept → why it fits you → PyTorch implementation → how to integrate with JEPA / RiJEPA loss.

---

### 1. What is GIN?

**Graph Isomorphism Network (GIN)** was proposed by Xu et al. at ICLR 2019 in *How Powerful are Graph Neural Networks?*. The core motivation: many GNNs (e.g., GCN / GraphSAGE) are not expressive enough to distinguish different graph structures; GIN achieves the expressive power of the 1-Weisfeiler-Lehman graph isomorphism test through **sum aggregation + MLP**. ([arXiv][1])

The basic GIN update formula:

$$
h_v^{(k)} =
\text{MLP}^{(k)}
\left(
(1+\epsilon^{(k)})h_v^{(k-1)}
+
\sum_{u \in \mathcal{N}(v)} h_u^{(k-1)}
\right)
$$

Intuitively:

- Each object is a node.
- Each object collects information from its neighbours.
- Use **sum** instead of mean/max because sum is more expressive over multisets.
- The MLP learns “what this local physical structure means”.

In PyTorch Geometric, `GINConv` implements this operator, expecting an MLP as `nn`, with support for `eps` and `train_eps`. ([PyG Documentation][2])

---

### 2. Why is GIN suitable for your scene graph encoder?

Your goal is not to write a rule as a sentence:

```text
IF hit_TNT THEN collapsed
```

but to represent a stable physical state as a graph:

```text
nodes:
  bird, wood_block_1, stone_block_2, TNT, pig_1

edges:
  contact(wood_block_1, stone_block_2)
  supports(stone_block_2, pig_1)
  left_of(TNT, pig_1)
```

This aligns well with your theoretical direction: symbols are only active in stable phase / quasi-static attractors, while chaotic collapse is left to JEPA. Your document already summarises this as “pre/post stable states use symbolic constraints, chaotic middle uses pure JEPA”, expressed via phase‑gated EBC and terminal anchor loss.

GIN / GINE plays the role:

$$
G_t^{sym} \rightarrow f_{\text{graph}}(G_t^{sym}) = z_{t,\text{rule}}
$$

i.e., encoding a **physical symbolic graph** into a vector aligned with the JEPA latent space.

---

### 3. Should you use GIN or GINE?

I recommend:

| Choice            | When to use                                                  | Suitable for you?               |
| ----------------- | ------------------------------------------------------------ | ------------------------------- |
| `GINConv`         | Graph has only node features, edges only indicate connectivity | Not enough                      |
| `GINEConv`        | Edges have types or continuous attributes (contact/support/distance) | **Best for your first version** |
| Graph Transformer | Need stronger expressivity, global attention, multi‑relational | Later enhancement               |
| Heterogeneous GNN | Many node/edge types and you want to model them separately   | Consider in phase two           |

PyTorch Geometric provides `GINEConv` officially. It is an edge‑feature version of GIN and can handle `edge_attr`, e.g., relation type, distance, relative orientation, contact strength, etc. ([PyG Documentation][3])

---

### 4. How to construct the scene graph?

In PyTorch Geometric, a graph is typically represented by a `Data` object. It can contain:

```python
Data(
    x=node_features,
    edge_index=edge_index,
    edge_attr=edge_features,
    y=label
)
```

The official documentation explains that `Data` can hold node‑level, edge‑level, and graph‑level attributes like `x`, `edge_index`, `edge_attr`, `y`, `pos`, `time`, etc. ([PyG Documentation][4])

### Node feature design

One node per object, e.g.:

```text
wood_block_1
stone_block_2
ice_block_3
pig_1
TNT
bird
```

Suggested node features:

```text
[type_onehot]       bird / pig / wood / stone / ice / TNT
[material_onehot]   wood / stone / ice / metal / flesh / explosive
[position]          x, y
[velocity]          vx, vy
[shape]             width, height, angle
[status]            alive, damaged, sleeping, static
[mass/friction]     if available
```

Example:

$$
x_i = [\text{type}_i, \text{material}_i, p_i, v_i, \theta_i, \text{health}_i]
$$

### Edge feature design

Edges are the key. Each edge can be:

```text
contact(A, B)
supports(A, B)
above(A, B)
left_of(A, B)
near(A, B)
inside_explosion_radius(A, TNT)
```

Suggested edge features:

```text
[relation_type_onehot]  contact / support / above / left_of / near_TNT
[relative_position]     dx, dy
[distance]              ||p_i - p_j||
[contact_normal]        nx, ny
[overlap_depth]         if available
[support_strength]      if available
```

For relationships like `supports(A, B)`, it is best to preserve direction: `A -> B` means “A supports B”. If you want bidirectional propagation, you can also add reverse edges:

```text
A -> B: supports
B -> A: supported_by
```

---

### 5. Minimal PyTorch Geometric implementation

Below is a **GINE scene graph encoder** suitable for your project. It encodes a scene graph into:

1. Graph‑level embedding – for symbolic rule embedding / EBC / anchor.
2. Node‑level embedding – for later object‑level projection or predicate decoding.

```python
import torch
from torch import nn
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, global_mean_pool


class SceneGraphGINEEncoder(nn.Module):
    """
    Scene graph encoder for physical symbolic states.

    Input:
        x:         [num_nodes, node_in_dim]
        edge_index:[2, num_edges]
        edge_attr: [num_edges, edge_in_dim]
        batch:     [num_nodes], graph id for mini-batching

    Output:
        graph_z:   [batch_size, out_dim]
        node_z:    [num_nodes, hidden_dim]
    """

    def __init__(
        self,
        node_in_dim: int,
        edge_in_dim: int,
        hidden_dim: int = 128,
        out_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.node_proj = nn.Linear(node_in_dim, hidden_dim)
        self.edge_proj = nn.Linear(edge_in_dim, hidden_dim)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )

            conv = GINEConv(
                nn=mlp,
                edge_dim=hidden_dim,
                train_eps=True,
            )

            self.convs.append(conv)
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.dropout = dropout

        self.graph_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.node_proj(x)
        edge_attr = self.edge_proj(edge_attr)

        for conv, norm in zip(self.convs, self.norms):
            residual = x
            x = conv(x, edge_index, edge_attr)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = x + residual

        graph_z = global_mean_pool(x, batch)
        graph_z = self.graph_proj(graph_z)

        return graph_z, x
```

Here `global_mean_pool` aggregates node embeddings into a graph embedding, suitable for graph‑level prediction / classification / rule embedding. PyG documentation explains that it performs batch‑wise mean pooling of node features per graph. ([PyG Documentation][5])

---

### 6. Toy scene graph example

Suppose you have 4 objects:

```text
0: bird
1: wood_block
2: TNT
3: pig
```

Edges:

```text
bird near wood_block
wood_block supports pig
TNT near pig
```

Construct as:

```python
import torch
from torch_geometric.data import Data

# Example dimensions:
# node feature = [type_onehot(4), x, y, vx, vy, health] = 9 dims
# edge feature = [relation_onehot(4), dx, dy, distance] = 7 dims

x = torch.tensor([
    [1, 0, 0, 0,  0.1, 0.2,  3.0, 1.5, 1.0],  # bird
    [0, 1, 0, 0,  0.5, 0.2,  0.0, 0.0, 1.0],  # wood
    [0, 0, 1, 0,  0.7, 0.2,  0.0, 0.0, 1.0],  # TNT
    [0, 0, 0, 1,  0.8, 0.5,  0.0, 0.0, 1.0],  # pig
], dtype=torch.float)

# Directed edges: source -> target
edge_index = torch.tensor([
    [0, 1, 2, 1, 3, 3],
    [1, 3, 3, 0, 1, 2],
], dtype=torch.long)

edge_attr = torch.tensor([
    [1, 0, 0, 0,  0.4,  0.0, 0.4],  # near
    [0, 1, 0, 0,  0.3,  0.3, 0.42], # supports
    [1, 0, 0, 0,  0.1,  0.3, 0.32], # near
    [0, 0, 1, 0, -0.4,  0.0, 0.4],  # reverse / related
    [0, 0, 0, 1, -0.3, -0.3, 0.42], # supported_by
    [0, 0, 1, 0, -0.1, -0.3, 0.32], # reverse / related
], dtype=torch.float)

data = Data(
    x=x,
    edge_index=edge_index,
    edge_attr=edge_attr,
    y=torch.tensor([1])  # e.g., final outcome: level cleared
)
```

For a single graph:

```python
encoder = SceneGraphGINEEncoder(
    node_in_dim=9,
    edge_in_dim=7,
    hidden_dim=128,
    out_dim=256,
)

batch = torch.zeros(data.x.size(0), dtype=torch.long)

graph_z, node_z = encoder(
    data.x,
    data.edge_index,
    data.edge_attr,
    batch,
)

print(graph_z.shape)  # [1, 256]
print(node_z.shape)   # [num_nodes, 128]
```

---

### 7. How to connect this to your JEPA / RiJEPA‑style objective

Your current theory already has:

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{JEPA}} + \beta \mathbb{1}_{\text{stable}}\mathcal{L}_{\text{EBC}} + \gamma \mathcal{L}_{\text{anchor}}
$$

The GINE encoder replaces the weak “text rule encoder”.

### Original weak version

```text
"IF hit_TNT THEN all_pigs_dead"
        ↓
Text Transformer
        ↓
z_rule
```

### Your stronger version

```text
Stable scene graph before shot / after collision
        ↓
GINE scene graph encoder
        ↓
z_graph / z_rule
```

So you can define:

$$
z_{A} = f_{\text{GINE}}(G_{\text{pre}})
$$

$$
z_{C} = f_{\text{GINE}}(G_{\text{post}})
$$

Then the EBC can be:

$$
E(A,C) = \|g(z_A) - z_C\|_2^2
$$

A simple implementation:

```python
def energy(pred_z, target_z):
    return ((pred_z - target_z) ** 2).sum(dim=-1)


def ebc_loss(jepa_predictor, z_A, z_C_pos, z_C_neg, margin=1.0):
    """
    z_A:     antecedent graph embedding
    z_C_pos: true consequent graph embedding
    z_C_neg: physically invalid or corrupted consequent graph embedding
    """
    pred_C = jepa_predictor(z_A)

    e_pos = energy(pred_C, z_C_pos)
    e_neg = energy(pred_C, z_C_neg)

    loss_pos = e_pos.mean()
    loss_neg = F.relu(margin - e_neg).mean()

    return loss_pos + loss_neg
```

Then integrate with phase gating:

```python
loss_total = loss_jepa

if stable_mask.any():
    loss_total = loss_total + beta * loss_ebc

loss_total = loss_total + gamma * loss_anchor
```

In practice, `stable_mask` should be per‑sample:

```python
loss_total = loss_jepa + beta * (stable_mask * loss_ebc_per_sample).mean() + gamma * loss_anchor
```

This matches your theoretical setup where symbolic constraints are active only in stable phases, and the chaotic middle is left to pure neural JEPA.

---

### 8. Recommended implementation plan for your project

### Stage 1: Oracle graph encoder

Start with graph data directly from the physics engine / environment.

Use:

```text
ground‑truth object positions
ground‑truth velocities
ground‑truth contact graph
ground‑truth support graph
ground‑truth pig alive/dead
ground‑truth material
```

This avoids the hardest perception problem. Your first research question should be:

> If the symbolic graph is correct, can structured symbolic geometry improve JEPA endpoint prediction and OOD robustness?

This is the cleanest upper‑bound experiment.

### Stage 2: Replace text‑rule encoder with GINE

Use `SceneGraphGINEEncoder` for:

```text
pre‑collision graph G_0
post‑collision graph G_T
negative / impossible graph G_neg
```

Train:

```text
JEPA prediction loss
+ phase‑gated EBC
+ terminal anchor loss
+ optional graph predicate decoder loss
```

### Stage 3: Add predicate heads

To make “symbols as submanifolds” more concrete, add heads like:

```python
self.symbol_heads = nn.ModuleDict({
    "collapsed": nn.Linear(out_dim, 1),
    "all_pigs_dead": nn.Linear(out_dim, 1),
    "tnt_exploded": nn.Linear(out_dim, 1),
    "stable": nn.Linear(out_dim, 1),
})
```

Then train:

```python
loss_symbols = (
    F.binary_cross_entropy_with_logits(pred_collapsed, y_collapsed)
    + F.binary_cross_entropy_with_logits(pred_pigs_dead, y_pigs_dead)
    + F.binary_cross_entropy_with_logits(pred_stable, y_stable)
)
```

This helps support your claim that symbols are not isolated vector points, but **learned geometric constraints** over latent space.

---

### 9. PyG vs DGL

For your project, I would choose **PyTorch Geometric** first.

| Library           | Pros                                                         | Cons                                                 | Recommendation          |
| ----------------- | ------------------------------------------------------------ | ---------------------------------------------------- | ----------------------- |
| PyTorch Geometric | Very common in ML papers, clean `Data`, `DataLoader`, `GINConv`, `GINEConv` | Installation can be slightly version‑sensitive       | **Best choice for you** |
| DGL               | Mature, also has `GINConv`, good heterograph support         | API feels heavier for quick prototyping              | Good alternative        |
| Pure PyTorch      | Full control                                                 | You must implement batching/message passing yourself | Not worth it initially  |

DGL also provides `dgl.nn.pytorch.conv.GINConv` with `apply_func`, `aggregator_type`, `init_eps`, and `learn_eps`, so it is also viable if you already use DGL elsewhere.([DGL][6])

---

### 10. Key caveats you should know

First, **GIN is not magic graph isomorphism solving**. It is as expressive as the 1‑WL test under the theoretical framework of the original paper, but 1‑WL itself cannot distinguish all non‑isomorphic graphs. For your scene graphs, this is probably fine at first, because object features, relation types, positions, materials, and velocities will break many symmetries. ([arXiv][1])

Second, **vanilla GIN ignores edge semantics**. Since your symbolic graph is mostly about physical relations, use `GINEConv` or another edge‑aware message‑passing layer.

Third, for directed physical relations such as `supports(A,B)`, do not automatically treat the graph as undirected unless you encode the reverse relation explicitly. `supports(A,B)` and `supported_by(B,A)` have different meanings.

Fourth, avoid making GIN responsible for chaotic collapse. Your own method’s strength is that symbolic geometry is applied at stable endpoints, while JEPA handles the transient collapse. The graph encoder should encode **stable physical configurations**, not every mid‑air splinter state.

---

### My recommendation

Your first serious implementation should be:

```text
Oracle scene graph from NovPhy / Box2D
→ GINE scene graph encoder
→ graph embedding z_sym
→ align z_sym with JEPA latent z
→ phase‑gated EBC only when φ(x_t)=1
→ terminal symbolic anchor loss
```

This gives you a clean, defensible contribution:

> “We replace flat textual rule embeddings with structured physical scene‑graph embeddings, preserving object relations, permutation invariance, and physical compositionality.”

That is much stronger than simply saying “we use symbolic rules.”

[1]: https://arxiv.org/abs/1810.00826?utm_source=chatgpt.com "How Powerful are Graph Neural Networks?"
[2]: https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GINConv.html?utm_source=chatgpt.com "torch_geometric.nn.conv.GINConv — pytorch_geometric documentation"
[3]: https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GINEConv.html?utm_source=chatgpt.com "torch_geometric.nn.conv.GINEConv — pytorch_geometric documentation"
[4]: https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.data.Data.html?utm_source=chatgpt.com "torch_geometric.data.Data — pytorch_geometric documentation"
[5]: https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.pool.global_mean_pool.html?utm_source=chatgpt.com "torch_geometric.nn.pool.global_mean_pool — pytorch_geometric documentation"
[6]: https://www.dgl.ai/dgl_docs/generated/dgl.nn.pytorch.conv.GINConv.html?utm_source=chatgpt.com "GINConv — DGL 2.5 documentation"
