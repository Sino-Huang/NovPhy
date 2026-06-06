# High-Level Research Project Plan

## Project Aim

Build and evaluate a **Temporally-Gated, Granularity-Adaptive Neuro-Symbolic JEPA / RiJEPA** world model for NovPhy / Science Birds. The central claim is that symbolic structure should not try to model chaotic collision dynamics directly. Instead, JEPA should handle the high-entropy collapse process, while symbolic constraints should anchor the stable pre-collision and post-collision states where object relations and outcomes are meaningful.

The first step remains the environment and data collection pipeline: generated level creation, engine loading, WebUI inspection, and reproducible trajectory collection. The updated implementation need is that this pipeline must expose enough frame, symbolic-state, and outcome data to support oracle scene graphs, phase detection, temporal-resolution ablations, and endpoint symbolic supervision.

## Target Outcome After 2-3 Months

By the end of the project window, the project should have:

1. A reproducible NovPhy / Science Birds trajectory pipeline for collecting frames, symbolic states, actions, phase labels, and final outcomes from generated and standard levels.
2. A JEPA / LeWM-style baseline showing how prediction quality changes across temporal resolutions, from 2-step goal-state prediction to dense frame-level rollout.
3. A temporally gated RiJEPA objective where symbolic constraints are active only in stable phases and masked during chaotic collapse.
4. An oracle scene graph upper-bound experiment using NovPhy / engine-provided symbolic information.
5. A first structured symbolic implementation based on an edge-aware scene graph encoder such as GINE, not flat text-rule embeddings.
6. An evaluation suite comparing pure JEPA, always-on symbolic constraints, phase-gated symbolic constraints, terminal anchors, and Delta t / sampling interval variants.
7. A paper-ready narrative, figures, ablations, and reviewer-risk responses.

## Core Research Questions

**RQ1: Phase Validity**
When are symbolic constraints valid in a physical cascade? Can stable pre-collision and post-collision regimes be separated from chaotic collision / collapse intervals using a stability detector phi(x_t)?

**RQ2: Temporal Granularity**
How does the sampling interval Delta t affect the trade-off between continuous prediction fidelity and terminal symbolic accuracy?

**RQ3: Baseline Failure Mode**
Do pure JEPA / LeWM rollouts fail more severely at particular temporal resolutions or physical phases, and is the failure visible as final-state error, physical violation, or planning degradation?

**RQ4: Phase-Gated Symbolic Guidance**
Does phase-gated EBC plus terminal anchor loss improve endpoint prediction and OOD robustness without damaging chaotic-collapse prediction?

**RQ5: Structured Symbolic Geometry**
Does a GINE-style scene graph encoder provide a stronger symbolic interface than flat symbolic text embeddings, especially for relation-heavy physical states such as support, contact, above, and near-TNT?

## Method Thesis

The method should be framed as:

> Let the neural predictor handle the chaos, but anchor the stable endpoints in structured symbolic geometry.

This means the project should not force symbolic rules onto the middle of the collapse. During collision and collapse, the world is too continuous, chaotic, and object-fragment-level for symbolic rules to be useful. Symbols are most useful at temporal abstraction boundaries: before collision, after the scene settles, and at goal/outcome states.

The high-level model stack is:

1. **JEPA / LeWM backbone** for latent future prediction.
2. **Stability detector phi(x_t)** for identifying symbol-friendly quasi-static states.
3. **Phase-gated EBC** that applies symbolic energy constraints only when phi(x_t)=1.
4. **Terminal anchor loss** that pulls the final predicted latent state toward symbolic poles such as level-cleared, failed, pigs-alive, or pigs-dead outcomes.
5. **Oracle scene graph source** from NovPhy / engine symbolic states for the first upper-bound experiment.
6. **GINE / edge-aware scene graph encoder** for structured symbolic embeddings z_sym.
7. **Optional predicate heads** for treating symbols as learned submanifolds rather than isolated vector points.

## Three-Month Execution Plan

### Month 1: Environment, Dataset, Baselines, and Phase Labels

Goal: establish the dataset and prove that temporal phase and temporal resolution matter before building the full symbolic module.

**Week 1: Reproducible Environment and Level Generation**

- Finalize Science Birds runtime setup and document exact launch commands.
- Use IratusAves and existing NovPhy levels to build a larger level pool.
- Confirm generated levels load through `sciencebirdsgames/Linux/config.xml`.
- Define the trajectory record format: initial frame, shot action, frame sequence, symbolic states, game state, score, pig survival, and final object layout.
- Deliverable: a small collected dataset from standard and generated levels.

**Week 2: Trajectory Collection and Oracle Graph Availability**

- Automate repeated level runs with controlled actions.
- Collect screenshots and symbolic states at fixed intervals after one shot.
- Store metadata for level id, shot parameters, success/failure, score, pig state, and final stable state.
- Convert symbolic states into oracle scene graph candidates with nodes for objects and directed edges for relations such as contact, support, above, left_of, and near_TNT.
- Deliverable: a stable dataset collection script and sample oracle scene graphs for pre-collision and post-collision states.

**Week 3: Temporal Resolution Baseline**

- Bring up the simplest JEPA / LeWM-style baseline that can train or evaluate on collected trajectories.
- Evaluate multiple temporal grids, such as K in {2, 5, 10, 30, full FPS}, where K is the number of prediction steps.
- Treat the 2-step setting as goal-state prediction and the dense setting as frame-level rollout.
- Measure continuous prediction error and terminal symbolic accuracy separately.
- Deliverable: first Delta t / K trade-off curves.

**Week 4: Stability Detection and Failure Analysis**

- Define a first stability detector phi(x_t), using available signals such as object velocity, low frame change, contact stability, or engine symbolic state stability.
- Partition trajectories into stable pre-collision, chaotic collapse, and stable post-collision phases.
- Identify failure types: wrong terminal outcome, wrong pig survival, missed collapse, floating unsupported objects, object penetration, or excessive latent drift.
- Decision gate: continue only if phase-specific or granularity-specific failure is measurable.
- Deliverable: baseline failure report with phase labels and locked evaluation metrics.

### Month 2: Phase-Gated RiJEPA and Structured Scene Graph Encoding

Goal: test the core implementation idea with oracle symbolic graphs first, then add a structured encoder.

**Week 5: Oracle Scene Graph Upper Bound**

- Use engine-provided symbolic states to build pre-collision graph G_pre and post-collision graph G_post.
- Include object type, material, position, approximate velocity, health/status, and directed relation edges.
- Use oracle graphs to ask the clean upper-bound question: if the symbolic graph is correct, can structured symbolic geometry improve endpoint prediction and OOD robustness?
- Deliverable: oracle graph extraction, validation examples, and first graph-level labels.

**Week 6: Phase-Gated EBC and Terminal Anchors**

- Add phase-gated symbolic loss at a high level: JEPA prediction loss remains active everywhere, while EBC is applied only when phi(x_t)=1.
- Add terminal anchor loss for final stable outcomes, using symbolic poles for classes such as level cleared/failed, all pigs dead, structure collapsed, or TNT exploded.
- Compare pure JEPA, always-on symbolic constraints, and phase-gated symbolic constraints.
- Decision gate: if always-on symbolic constraints damage chaotic-collapse prediction but phase-gated constraints preserve it, the main thesis is supported.
- Deliverable: phase-gating ablation and terminal-anchor ablation.

**Week 7: GINE Scene Graph Encoder**

- Replace weak text-rule embedding with a structured scene graph encoder.
- Start with GINE or another edge-aware GNN because physical relations are edge-heavy: support, contact, above, left_of, near_TNT, overlap, and relative position.
- Encode G_pre and G_post into graph embeddings z_A and z_C, then align these embeddings with the JEPA latent space.
- Keep graph encoding focused on stable configurations, not mid-collapse splinters.
- Deliverable: first GINE-based symbolic embedding and latent-alignment result.

**Week 8: Predicate Heads and Symbolic Submanifold Prototype**

- Add optional predicate heads for symbols such as collapsed, all_pigs_dead, tnt_exploded, and stable.
- Use predicate heads to support the stronger claim that symbols define learned latent submanifolds rather than scattered text-vector points.
- Compare oracle graph embeddings, GINE graph embeddings, and any simpler symbolic encoding baseline.
- Decision gate: if predicate heads are unreliable, keep them as a secondary analysis and preserve GINE + terminal anchors as the main implementation.
- Deliverable: predicate-head prototype or documented decision to keep it as future work.

### Month 3: Full Evaluation, Planning Link, and Paper Assembly

Goal: turn the method into a defensible research result with the right ablations.

**Week 9: Full Evaluation Matrix**

- Run experiments across standard NovPhy levels, generated IratusAves levels, and selected novelty levels.
- Report final-state accuracy, zero-shot symbolic accuracy, rollout error by horizon, physical violation rate, and runtime cost.
- Include baselines and ablations: pure JEPA, always-on symbolic constraints, phase-gated EBC, terminal anchor on/off, oracle graph upper bound, GINE encoder, and Delta t / K variants.
- Deliverable: complete evaluation tables and trade-off plots.

**Week 10: OOD and Adaptation Track**

- Test whether symbolic anchors help under changed level layouts, materials, gravity-like effects, or generated level distributions.
- If feasible, connect the world model to a simple planner or action selector.
- Compare action selection or adaptation with and without phase-gated symbolic guidance.
- Deliverable: OOD robustness result and small planning/adaptation result.

**Week 11: Ablations and Reviewer Defenses**

- Run ablations for temporal resolution, stability detector choice, EBC gating, terminal anchor loss, graph relation types, oracle vs GINE graph, and predicate heads.
- Prepare direct answers to likely reviewer objections: hand-engineered symbols, heuristic phi(x_t), DeepPHY overlap, NovPhy task mismatch, and comparison to rollout loss / MPC.
- Deliverable: ablation table and reviewer-risk memo.

**Week 12: Writing and Submission Package**

- Write the paper narrative around the concrete idea that symbols are valid at stable endpoints but not during chaotic collapse.
- Put the problem, why it matters, and the one-sentence method in the first 300 words.
- Prepare method diagram, temporal-phase diagram, Delta t trade-off curve, rollout examples, metric plots, and reproducibility appendix.
- Deliverable: complete paper draft and experiment checklist for remaining runs.

## Two-Month Compressed Plan

If the deadline is closer to two months, compress the project by reducing method ambition:

- Keep Month 1 mostly unchanged, because environment, trajectory collection, oracle graphs, phase labels, and baseline failure are mandatory.
- In Month 2, prioritize phase-gated EBC, terminal anchor loss, oracle scene graph upper bound, and Delta t ablations.
- Implement GINE only if the oracle graph experiment already shows value.
- Treat predicate heads, learned phi(x_t), and full planning/adaptation as secondary or future-work components.
- Submit the core story as: JEPA should model chaotic collapse, while temporally gated symbolic anchors improve stable endpoint prediction and OOD robustness.

## Minimum Viable Paper

The minimum viable publishable version is not a full learned perception-to-symbols system. It is:

1. A clear definition of phase-dependent symbol validity in prolonged action effect environments.
2. A benchmark protocol on NovPhy / Science Birds for temporal-resolution-dependent physical cascade prediction.
3. Evidence that pure JEPA / LeWM has different failure modes across coarse and fine temporal grids.
4. A phase-gated RiJEPA objective with symbolic constraints active only in stable phases.
5. A terminal symbolic anchor experiment showing improved final-state or OOD symbolic accuracy.
6. An oracle scene graph upper-bound experiment, with GINE as the first structured symbolic implementation if time allows.

This version is safer than trying to learn the entire symbolic pipeline end-to-end. It gives a defensible scientific contribution even if learned perception, learned stability detection, or predicate submanifolds are not fully mature within the project window.

## Metrics and Evidence

Use metrics that directly support the method claim:

- **Terminal symbolic accuracy**: predicted level-cleared status, pig survival, collapsed structures, TNT explosion, or other final symbolic outcomes.
- **Zero-shot symbolic accuracy**: nearest symbolic-pole classification without a downstream classifier.
- **Rollout error curve**: error across horizons and temporal grids K in {2, 5, 10, 30, full FPS}.
- **Phase-specific prediction error**: separate pre-collision, chaotic-collapse, and post-collision errors.
- **Physical violation rate**: unsupported floating objects, object penetration, impossible contacts, or broken support relations.
- **OOD robustness**: performance under changed layouts, materials, generated levels, or novelty settings.
- **Planning or adaptation gain**: improved score, success rate, or reduced recovery trials when the corrected world model is used downstream.

Avoid relying only on image reconstruction loss. The paper's claim is about temporal abstraction, phase validity, and endpoint symbolic correctness, so terminal symbolic accuracy, phase-specific error, and Delta t trade-off curves should be central.

## Key Ablations

The evaluation should include these ablations if time permits:

1. Pure JEPA / LeWM baseline.
2. Always-on symbolic constraints.
3. Phase-gated EBC only when phi(x_t)=1.
4. Terminal anchor loss on/off.
5. Oracle scene graph vs GINE scene graph encoder.
6. GINE vs simpler flat symbolic encoding.
7. Delta t / K variants: 2-step, 5-step, 10-step, 30-step, dense frame-level prediction.
8. Stability detector variants: heuristic phi(x_t), engine-derived stable state, or learned phi(x_t) if time allows.
9. Predicate heads on/off for symbols-as-submanifolds.

## Key Risks and Mitigations

**Risk: symbolic constraints hurt chaotic-collapse prediction.**
Mitigation: this is exactly why the method uses phase gating. Always-on symbolic constraints should be treated as an ablation, not the main method.

**Risk: phi(x_t) looks heuristic.**
Mitigation: start with an engine- or velocity-based detector for implementation speed, then discuss learned phi(x_t) or dynamical-systems justification as an extension.

**Risk: reviewers see symbolic relations as hand-engineering.**
Mitigation: distinguish fixed relation vocabulary from learned graph embeddings and learned predicate heads. Emphasize that GINE preserves relational structure rather than flattening rules into text.

**Risk: GINE becomes too large a side project.**
Mitigation: first run oracle scene graph and terminal-anchor experiments. Add GINE only after the symbolic upper bound is positive.

**Risk: NovPhy is seen as an adaptation benchmark, not a world model benchmark.**
Mitigation: explicitly define a prediction benchmark on top of NovPhy and then use planning/adaptation only as downstream validation.

**Risk: the project overpromises end-to-end learned symbols.**
Mitigation: keep oracle graphs and phase-gated endpoint anchoring as the main 2-3 month deliverables; learned perception and predicate submanifolds are secondary.

## Weekly Operating Rhythm

Each week should end with a binary checkpoint:

- A runnable command or notebook.
- A saved artifact: dataset, plot, table, trained checkpoint, or written memo.
- A short result summary: what worked, what failed, and what decision it forces.
- One locked next-week priority.

Do not let implementation work continue for more than one week without producing a measurable plot or table. The project succeeds only if the phase-gating and temporal-granularity claims become visible early.

## Immediate Next Steps

1. Finish the data collection pipeline around generated and standard Science Birds levels.
2. Define the stored trajectory schema with frames, symbolic states, final outcomes, and candidate phase labels.
3. Collect a pilot dataset that supports K in {2, 5, 10, 30, full FPS} sampling.
4. Implement the first baseline evaluation for terminal symbolic accuracy and phase-specific rollout error.
5. Build oracle scene graphs for stable pre-collision and post-collision states.
6. Start phase-gated EBC and terminal anchor experiments only after the baseline failure and phase labels are measurable.
