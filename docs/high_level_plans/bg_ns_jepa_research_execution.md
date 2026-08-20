# High-Level Research Execution Plan — BG-NS-JEPA (Regime-Adaptive World Models)

**Status:** corrected current roadmap, 2026-08-13. Supersedes the v1, 2026-08-07 roadmap. The physics
runtime workstream is closed (`repin_complete`); the world-model workstream has delivered the continuous
backbone and a continuous-only temporal grid; macro/oracle label work exists only as unaccepted
pre-repin scaffolding. **Publication and enriched-cohort collection are not authorized and have not
happened.**
**Canonical inputs:** `docs/research_proposal.md` (ICLR 2027 proposal) and
`docs/method_proposal_presentation/system_flow_and_training_overview.md` (consolidated system + training
picture). This plan derives its component vocabulary, stage structure, and acceptance evidence from
those two documents.

---

## 1. Destination and current position

**Destination:** BG-NS-JEPA — a world model that estimates a local degree of scale separation and
**jointly** selects how far ahead to predict ($\Delta$) and at which level of description to compute it
($\alpha \in \{continuous, micro, macro\}$), with the continuous latent $z$ as the sole rollout state
carrier and all symbolic output as reliability-gated readouts and constraints.

**One-sentence pitch (from the proposal):** BG-NS-JEPA is the learned analogue of a hybrid kinetic–fluid
solver.

**Non-negotiables that shape every milestone:**

1. **State-carrier principle.** $z$ is the only rollout state. No discrete step, graph assembly, or
   symbol decoding ever sits in the rollout path; symbolic heads are supervised readouts that enter only
   the loss. This is what keeps base training teacher-forced and fully differentiable, and it is the
   structural answer to the symbolic-noise critique.
2. **Training-time-oracle → test-time-amortized.** Everything deployable (controller $\pi_\kappa$,
   reliability gate $r_\psi$, predicate extractor) is distilled or supervised from oracle/engine labels
   during training, then used as a single amortized forward pass at test time. The oracle gate is the
   ceiling; the learned gate must approach it.
3. **Symbols are constraints and readouts, never the state carrier; extractor frozen after supervised
   training** to keep predicate semantics anchored to engine definitions (ontology reward-hacking guard).
4. **Physics grounding is algorithmic, not theoretical.** Every borrowed concept (Knudsen-like scale
   separation, restriction/lifting, closure consistency, equilibria) must map to a concrete, ablatable
   design decision. No rigorous derivation is claimed.
5. **Evidence quality bar.** Every milestone ends with machine-readable evidence and exit criteria
   before the next milestone starts.

**Current position:** the T1 physics-capture runtime is complete and its accepted smoke consumed;
nothing has been published and no enriched cohort has been collected. The JEPA backbone (1a), the
dual-output predictor/trainer (1b), and a continuous-only legacy temporal grid (1e/1f projection) exist.
SPSG/GINE (1c), the macro-event predictor and restriction/lifting maps (1d), and all micro/macro and
oracle-symbol scoring (full 1f) do not.

## 2. Current state of work

### 2.1 Physics runtime (T1) — complete

- **Runtime status:** `repin_complete`.
- **Accepted smoke:** one accepted smoke with four collisions, contact evidence present, speed
  `10.6197062`, request `1 -> 2`, zero refusals, and a stable listener, cleanup, and protected roots;
  the smoke was consumed.
- **Provenance:** archive `de59061350f78f79420d76ec33f1c506aa17c1cfc25d197cdd2f5f770874e838`;
  implementation `d5be336…`; closure commits `5a34ec9…` (record accepted T1 re-pin) and `2dfc439…`
  (hand off completed T1 re-pin).
- **Publication and enriched-cohort collection:** unauthorized and not done.

### 2.2 World model (Milestone 1) — partial

- **1a JEPA backbone: exists.** Context encoder, EMA/stop-grad target encoder, latent $z$ as the sole
  state carrier; the no-BatchNorm rule and the carrier/head independence are test-pinned
  (`world_model/model`, `world_model_jepa_backbone.md`).
- **1b Dual-output predictor/trainer: exists.** `DualOutputPredictor` emits the carrier $\hat z_{t+\Delta}$
  and mode-head readouts that enter the loss only; teacher-forced single-step loop with a seeded
  reproducibility manifest.
- **1e/1f continuous-only temporal grid: exists.** A legacy-RGB, teacher-forced projection over the
  requested horizon grid $\Delta=\{1,5,15\}$ with the abstraction axis closed to `continuous`
  (`world_model_jepa_pair_grid.md`). This is **not** an oracle-symbol or joint-$\alpha$ result.
- **1c SPSG relational mechanism: incomplete.** No GINE scene-graph encoder, predicate projection
  regularizers, or physics-validated negative sampling.
- **1d Macro-event predictor and restriction/lifting maps $A,R$: incomplete.**
- **Full micro/macro scoring and oracle-symbol 1f: blocked.** No enriched cohort exists to supply
  symbolic and physics supervision, so ADE/FDE, final-state accuracy, event F1, physical-violation
  rates, and the oracle-symbol ceiling remain unavailable.

**Continuous result (recorded exactly):**

| Field | Value |
|---|---|
| Grid | $\Delta \in \{1, 5, 15\}$, $\alpha$ = `continuous` only |
| micro / macro | unavailable, reason `symbolic_supervision_unavailable` |
| Runs | two CUDA runs, 3,600 updates each, seed `20260807`, batch 64, LR `3e-4`, EMA base momentum `0.996` |
| Budget | 1,200 updates per delta; 400 updates per `(delta, motion_regime)` key |
| States scored | 556,959 |
| Scores recorded | 1,670,877 |
| Checkpoints | matching declared checkpoint identities; metadata binds config, grid, catalog, run, and model configuration identities |
| Aggregate agreement | rtol ≤ 1e-2 |
| Best-pair agreement | 1.0 |
| Temporal-frontier verdict | `not_supported` for the continuous-only projection; this is **not** evidence for or against the joint $(\Delta,\alpha)$ controller |

### 2.3 Macro/oracle label derivation (Milestone 0a) — scaffolding only, not accepted

- Commits `6c70ebd` (derive macro, outcome, and oracle-gate labels) and `6bfbb8a` (derivation CLI and
  cohort health reporting) are **pre-repin scaffolding, not accepted M0a**.
- Reasons they are not accepted: an **event-render-frame clock bug** (event timestamps do not reliably
  align to `render_frame`); the work **combines 0a and 0b** instead of keeping the macro/outcome layer
  separate; and an **exact-`pig` assumption** in the derivation that is not engine-anchored.
- The **detailed M0a plan is current** and is the reference for the next step.

## 3. Gap analysis — from current state to the paper's Stage 1

| What the proposal/overview assumes | What exists today | The gap |
|---|---|---|
| Oracle per-frame states: continuous latent $z^*$, ground-truth predicates $S^{\mu*}, S^{M*}$, macro-event labels, outcome labels | Continuous carrier + readout heads; per-shot `physics_state.jsonl` / `physics_events.jsonl` sidecars | No accepted macro (structure-level) predicate layer, no outcome/equilibrium labels, no per-frame unified tensor cohort; macro/oracle derivations exist only as rejected scaffolding |
| Oracle gate label $\phi^*$ (KE + contact-activity thresholds) | Per-node kinetic energy and per-frame raw contacts captured | The $\phi^*$ computation is not accepted and was scaffolded only jointly with 0a |
| Model/training stack | JEPA backbone, dual-output predictor/trainer, continuous-only temporal grid | SPSG (1c), macro predictor + $A,R$ maps (1d), micro/macro scoring and oracle-symbol ceiling (1f) |
| Cross-domain eval (Physhion, CLEVRER) | Not present | Deferred to a later stage (not on this plan's critical path) |
| Baselines | None | Needed at Stage 3; LeWM/Sub-JEPA references are cited in the proposal but no code exists |

**Conclusion:** the physics instrumentation delivers the *micro-relational* and *event* descriptions the
proposal needs, and the continuous model path is proven on the legacy catalog, but two label layers are
still missing before Stage 1 can complete: the **macro (structure-level) event layer** (M0a) and the
**oracle gate + unified enriched cohort** (M0b). Until an authorized enriched cohort exists, all
symbolic and micro/macro evidence is blocked.

## 4. Milestone plan (high level)

Each milestone lists what it delivers, its acceptance evidence, and its dependency. Exit evidence gates
the next milestone. The horizon is the ICLR 2027 target.

### Milestone 0 — Research-ready enriched cohort and oracle labels (data track)

Two sub-tracks, in order:

- **0a — Macro/outcome label derivation.** Define the macro-event vocabulary
  (`structure-unstable`, `cascade-active`, `collapsed`, `pigs-cleared`, `steady-state`) as
  **deterministic derivations from existing physics_capture_v1 sidecars** (fixed-step clusters of
  destruction events, pig-removal counts, stability events, score/level outcome), recorded in a new
  derived-label schema (`physics_macro_labels_v1`). The derivation must be explicit, documented,
  engine-anchored, and **free of the exact-`pig` and event-frame-clock defects** that blocked the
  scaffolding. No vision, no learned model.
  - **Sequencing (updated):** the detailed M0a plan is current; **fixture-only M0a implementation may
    proceed before publication/cohort authorization** (schema/types/deriver/validator against faithful
    fixtures, no real cohort operations). Representative semantic acceptance of M0a remains **blocked**
    on authorization.
- **0b — Oracle reliability labels + unified tensor cohort.** Build the oracle gate label $\phi^*(x_t)$
  (KE and contact-activity thresholds per proposal §2.3). Consume the accepted 0a macro/outcome labels
  without redefining them, and produce a unified, per-frame, per-shot tensor dataset (or an on-the-fly
  collation) the existing lazy reader can consume, with the physics_capture_v1 supervision payload
  joined to RGB frames at exact `render_frame` alignment.
  - **Sequencing (updated):** 0b remains **blocked** — it requires the enriched cohort, whose collection
    is not authorized.

**Blocked-until-authorization set:** representative semantic acceptance of M0a, the 0b oracle/unified
cohort track, cohort health reporting on real data, and the full symbolic Milestone 1.

**Exit evidence:** a representative enriched cohort (a handful of levels across regimes, plus a
train/dev/test split note); a machine-readable dataset-health report; a macro-event and oracle-label
schema validated against the frozen sidecar contract; frame-exact alignment checks; the existing focused
data suite green on the new cohort.

**Dependency:** physics runtime (2.1), data pipeline (2.2), detailed M0a plan. **No model code is
written in this milestone.**

### Milestone 1 — World model + SPSG with oracle symbols (paper Stage 1, Weeks 1–4)

- **1a — JEPA backbone:** context encoder, EMA/stop-grad target encoder, latent $z$ as the sole state
  carrier. **Done.**
- **1b — Dual-output predictor** $F_\theta^{\Delta,\alpha}$: always emits the carrier $\hat z_{t+\Delta}$;
  mode-head readout $\hat S^\mu$ or $(\hat S^M, \hat\Delta, \hat e)$ enters the loss only. **Done.**
- **1c — SPSG relational mechanism:** GINE scene-graph encoder, predicate projection regularizers, and
  physics-validated negative sampling (reversed gravity, massless materials, anti-support) from proposal
  §4.2. **Incomplete.**
- **1d — Macro-event predictor** $G_\omega(S^M, z, a)$ and learned restriction/lifting maps $A, R$
  (§4.3). **Incomplete.**
- **1e — Teacher-forced multi-task training over the $(\Delta,\alpha)$ grid** (Phase A): sample
  $(t,\Delta,\alpha)$, predict from true $z_t$, align to $\mathrm{sg}(z^*_{t+\Delta})$, supervise the
  selected mode head, weight by $\Delta/T$. Terminal anchor loss (§4.5). Continuous-only legacy
  projection exists; full grid waits on the enriched cohort.
- **1f — Best-pair labels + oracle-symbol upper bound.** Exhaustively score every $(\Delta,\alpha)$ pair
  per state (duration-normalized prediction error + physical violations + compute) → the controller's
  future teacher, and the oracle-symbol ceiling. Continuous-only scores exist; full micro/macro and
  oracle-symbol scoring is **blocked** on the enriched cohort.

**Exit evidence:** per-pair grid metrics (ADE@H / FDE@H / final-state accuracy / event-prediction F1 on
oracle symbols); a recorded oracle-symbol upper bound; stored best-pair labels; a deterministic
reproducibility run (seeded, two runs comparable).

### Milestone 2 — Learned symbolic state (paper Stage 2, Weeks 5–8)

- **2a — Tier-2 visual predicate parser:** frozen pretrained ViT (V-JEPA/DINO-style) + DETR-style object
  decoder with Hungarian matching against engine objects + attribute/edge/unary/macro predicate heads,
  supervised by engine $G^*$; a light temporal-consistency loss within oracle-stable regimes.
- **2b — Learned reliability gate** $r_\psi$ trained on the extractor's confidence features against the
  oracle $\phi^*$.
- **2c — Freeze.** The extractor and visual encoder are frozen thereafter; predicate semantics stay
  anchored to engine definitions.

**Exit evidence:** learned-vs-oracle gap on controller performance; per-predicate F1 vs engine labels on
held-out levels; gate calibration (switch precision/recall vs oracle regimes); predicate flip-rate
coherence vs engine flip rate. Split by level, not by frame.

### Milestone 3 — Joint controller and ablations (paper Stage 3, Weeks 9–11)

- **3a — Distill $\pi_\kappa$** from the Milestone 1 best-pair labels (supervised classification over the
  pair grid).
- **3b — The central experiment:** joint vs factorized $\pi_\Delta\pi_\alpha$ vs single-axis (temporal-
  only, abstraction-only) vs fixed controllers, matched compute; shared-backbone vs separate-expert
  interfaces; optional recurrent controller feeding $(\Delta_{k-1},\alpha_{k-1})$ into $h$.
- **3c — Phase B training** of $F$ with the $(\Delta,\alpha)$ sampling distribution following the
  controller (or uniform, per ablation design); reliability gate $\omega_\psi$ modulating symbolic
  losses.

**Exit evidence (this is the claim the paper lives on):** the joint controller improves the
endpoint-correctness/physical-plausibility/compute frontier over fixed, single-axis, and factorized
alternatives; regime-aligned pair selection accuracy; physical-violation rates (penetration, floating,
illegal contact); OOD generalization (novel materials, gravity).

### Milestone 4 — Full system and optional variants (paper Stage 4, Weeks 12–14)

- Full BG-NS-JEPA evaluation on NovPhy standard + novelty scenarios; cross-domain (Physhion, CLEVRER)
  as stretch.
- **Optional variants, each gated by an acceptance criterion:** end-to-end controller relaxation
  (Gumbel-softmax — the only place gradients cross a discrete choice); short-window autoregressive
  fine-tuning of $F$ through the $z$ carrier (only if pilot drift demands it); extractor fine-tuning with
  anchored symbol loss + predicate-drift acceptance metric (rejected on significant drift).

### Milestone 5 — Paper and release

- Metrics table vs baselines (LeWM, Sub-JEPA, uniform micro/macro, ThinkJEPA, Causal-JEPA, temporal-only,
  abstraction-only, factorized, recurrent, and the oracle-gate / no-gate / no-SPSG / oracle-graph
  ablations from proposal §6.5).
- Reproducibility: seed, config, and evidence manifests; the data pipeline's named ablation presets are
  the mechanism.
- Paper write-up structured around the multiscale grounding and the joint-controller evidence.
- **Publication requires explicit authorization; nothing in this plan authorizes it.**

## 5. Sequencing rules, dependencies, and explicit non-goals

**Ordering:** accepted data/oracle dependencies gate research claims even though some model scaffolding was
built early. Fixture-only 0a implementation may proceed now. Representative 0a acceptance must precede
macro supervision; 0b must precede reliability-gated work; the full oracle-symbol Milestone 1 must precede
Milestones 2–5. Existing 1a/1b and continuous-only 1e/1f artifacts remain reusable but do not bypass these
gates.

**Dependencies:**

| Work | Depends on | Status of dependency |
|---|---|---|
| Runtime (T1) closed | — | Done (`repin_complete`, smoke consumed) |
| Data pipeline (`world_model/data`) | Runtime capture contract | Done |
| M0a fixture-only implementation | Detailed M0a plan; `physics_capture_v1` schema + opt-in reader | Current; may proceed before authorization |
| M0a semantic acceptance | Real enriched cohort + authorization | Blocked |
| M0b / 0b oracle + unified cohort | Enriched cohort collection + authorization | Blocked |
| 1c SPSG, 1d macro predictor, full 1f | M0a accepted labels + enriched cohort | Blocked (cohort) / Incomplete |
| M2–M5 | M1 complete | Not started |

**Explicitly out of scope for this plan** (each deferred or excluded deliberately):

- **No publication and no enriched-cohort collection** without explicit authorization; the fixture-only
  M0a path stops before real cohort operations.
- No training of a JEPA/world model beyond what exists (1a/1b/continuous 1e-1f projection) until the
  missing pieces (1c, 1d, full 1f) are unblocked; scene graphs, macro-event labels, and oracle labels
  remain in scope of Milestones 0–1.
- No retroactive annotation of legacy RGB-only episodes; enriched capture is opt-in under
  `capture_contract=physics_capture_v1`.
- No learned macro-event layer or oracle gate before Milestone 0a/0b — the derivations must be
  deterministic and engine-anchored first.
- No vision-based parser before Milestone 2 (Milestone 1 uses oracle symbols only).
- No cross-domain evaluation (Physhion, CLEVRER) before Milestone 4, and no claims about it before the
  NovPhy evidence exists.
- No changes to the production player, the active rollout root, or the canonical Unity project — the
  same protected-root guardrails as the completed work apply throughout.
- No claim of a rigorous Boltzmann-connection; the kinetic-theory mapping stays motivational and
  algorithmic.
- **No joint-$\alpha$ claim from the continuous-only grid**: it is `not_supported` and is not evidence
  for or against the joint controller.
- The `6c70ebd`/`6bfbb8a` macro/oracle scaffolding is **not** accepted M0a and is not treated as
  Stage-1 evidence.

## 6. Risks and mitigations (carried from the proposal, with this plan's additions)

| Risk | Mitigation |
|---|---|
| Learned symbolic extraction unstable | Start with oracle (M1); Tier-1 feature parser fallback; frozen supervised extractor (M2) |
| Symbolic noise corrupts rollouts | $z$ sole state carrier; constraints only on high-confidence positives; noise-injection degradation curve |
| Controller dismissed as error-based routing | Four structural differences (proposal §4.4); oracle-gate ablation is the ceiling, not an embarrassment |
| Macro-layer and oracle-label gaps block Stage 1 | Milestone 0 makes them deterministic engine-anchored derivations, not learned models; fixture-only path de-risks 0a before authorization |
| Unauthorized publication or cohort collection | Explicit gate: no publication/collection without authorization; fixture-only work stops before real cohort operations |
| "Hand-engineered vocabulary" critique | Vocabulary and deterministic oracle truth values are explicitly engine-anchored; learned extractors are evaluated against them later |
| Scaffolding defects recur (event-frame clock, combined 0a/0b, exact-`pig`) | M0a separates 0a/0b, projects event occurrence by fixed-step bracketing, and pins the engine pig-tag set; fixtures lock the contracts |
| Method too complex for the horizon | Oracle-symbol version (M0a+M1) is the minimum publishable unit |

## 7. Immediate next step

**Exact next step — fixture-only Milestone 0a implementation (no publication, no cohort operations):**

1. Implement the `physics_macro_labels_v1` **schema**, **types**, **deriver**, and **validator** for the
   deterministic macro/outcome derivations, separating 0a from 0b.
2. Build **faithful fixtures** from the frozen `physics_capture_v1` sidecar contract.
3. **Correct the opt-in reader event projection** so event records align by fixed-step bracketing while
   state/RGB records retain exact `render_frame` alignment.
4. Run **focused fixture tests only**, stopping before any real cohort operations.

This is the entire scope of the current step; representative semantic acceptance, 0b, cohort health, and
full symbolic Milestone 1 remain blocked on authorization.

---

*This high-level plan is deliberately implementation-agnostic on details such as exact hyperparameters,
model sizes, and training schedules; those belong to the milestone-level implementation plans.*
