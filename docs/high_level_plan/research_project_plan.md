# High-Level Research Project Plan

## Project Aim

Build and evaluate a Neuro-Symbolic JEPA-style world model for prolonged action effect environments, using NovPhy / Science Birds as the primary testbed. The central claim is that action-conditioned latent world models accumulate severe rollout error when a single action triggers a long physical cascade, and that symbolic physical structure can reduce this error.

The first step is the environment and data collection pipeline: generated level creation, engine loading, WebUI inspection, and reproducible trajectory collection. After that, the project should move quickly from baseline measurement to an oracle-symbolic upper-bound experiment, then to learned symbolic guidance, and finally to evaluation and writing.

## Target Outcome After 2-3 Months

By the end of the project window, the project should have:

1. A reproducible NovPhy / Science Birds data pipeline for collecting image frames, symbolic states, actions, and final outcomes from generated and standard levels.
2. A baseline LeWM / JEPA-style rollout benchmark showing where and how prediction error accumulates in physical chain reactions.
3. An oracle-symbolic upper-bound experiment using NovPhy's symbolic API to test whether symbolic correction can reduce long-horizon drift.
4. A learned or partially learned symbolic module that approximates the oracle-symbolic correction.
5. A final evaluation suite comparing pure neural rollout, oracle-symbolic correction, learned symbolic correction, and planning / adaptation impact.
6. A paper-ready narrative, figures, ablations, and risk responses.

## Core Research Questions

**RQ1: Problem Definition**
Can we formally and empirically characterize prolonged action effect environments, where the action sparsity ratio is low but the physical effect persistence is high?

**RQ2: Baseline Failure Mode**
Do JEPA / LeWM-style latent rollouts fail systematically on NovPhy physical cascades, and is the failure visible as final-state error, physical violation, or planning degradation?

**RQ3: Symbolic Upper Bound**
If the symbolic scene graph is given by the environment, can symbolic projection or correction reduce long-horizon error accumulation?

**RQ4: Learned Symbolic Guidance**
Can a learned symbolic abstraction approximate the oracle symbolic graph well enough to improve rollout stability without relying on hand-coded physics?

**RQ5: Planning and Adaptation Impact**
Does the improved world model help action selection or novelty adaptation, rather than only improving prediction metrics?

## Three-Month Execution Plan

### Month 1: Environment, Dataset, and Baseline Failure

Goal: prove that the project has a measurable problem before building the full method.

**Week 1: Reproducible Environment and Level Generation**

- Finalize Science Birds runtime setup and document exact launch commands.
- Use IratusAves and existing NovPhy levels to build a larger level pool.
- Confirm generated levels load through `sciencebirdsgames/Linux/config.xml`.
- Define the trajectory record format: initial frame, action, frame sequence, symbolic states, game state, score, pig survival, final object layout.
- Deliverable: a small collected dataset from standard and generated levels.

**Week 2: Data Collection Pipeline**

- Automate repeated level runs with controlled actions.
- Collect screenshots and symbolic states at fixed intervals after one shot.
- Store metadata for level id, shot parameters, random seed if available, success/failure, and final score.
- Add integrity checks for missing frames, malformed symbolic states, and failed engine runs.
- Deliverable: a stable dataset collection script and a first dataset large enough for baseline debugging.

**Week 3: Baseline Rollout Benchmark**

- Bring up the simplest LeWM / JEPA-style baseline that can train or evaluate on collected trajectories.
- Define rollout horizons such as 10, 25, 50, and 100 steps.
- Measure image-level and symbolic-level errors.
- Separate event-dense segments from steady-state segments if possible.
- Deliverable: first error accumulation curves for pure neural rollout.

**Week 4: Failure Analysis and Scope Gate**

- Identify concrete failure types: object penetration, unsupported floating, missed collapse, wrong pig survival, wrong stable state.
- Decide the primary evaluation target: final-state prediction, physical violation rate, planning success, or adaptation gain.
- Write a short internal finding note with plots and example rollouts.
- Decision gate: continue only if baseline failure is clear and measurable.
- Deliverable: baseline failure report and locked metric definitions.

### Month 2: Oracle Symbolic Correction and Learned Symbolic Approximation

Goal: verify the neuro-symbolic idea with the easiest possible symbolic source before attempting full learned abstraction.

**Week 5: Oracle Symbolic Scene Graph**

- Convert NovPhy symbolic states into object graphs.
- Include object type, material, position, approximate velocity, contact/support relations, and pig survival state.
- Define physical plausibility checks that are measurable from symbolic states.
- Deliverable: oracle scene graph extraction and validation examples.

**Week 6: Oracle Symbolic Projection Upper Bound**

- Add a correction loop that periodically maps rollout state to a symbolic graph, corrects physically invalid graph states, and maps back to the rollout target representation or evaluation space.
- Start with simple correction targets: final pig survival, object stability, collision/contact consistency.
- Compare pure neural rollout against oracle-symbolic correction.
- Decision gate: if oracle correction does not improve metrics, revise the symbolic target before building learned modules.
- Deliverable: upper-bound experiment showing whether symbolic guidance is worth pursuing.

**Week 7: Learned Symbolic Module Prototype**

- Train a lightweight module to predict symbolic graph attributes from latent states or frames.
- Start with supervised prediction from environment-provided symbolic states.
- Focus on reliable object/relation prediction rather than full end-to-end elegance.
- Deliverable: learned symbolic graph prediction with accuracy metrics against oracle symbolic states.

**Week 8: Learned Correction Integration**

- Replace oracle symbolic inputs with the learned symbolic module where possible.
- Compare three systems: pure neural, oracle-symbolic, learned-symbolic.
- Add ablations for correction frequency and relation vocabulary.
- Decision gate: if learned symbolic correction is too weak, keep oracle-symbolic as the main scientific upper bound and frame learned symbols as partial progress.
- Deliverable: integrated learned-symbolic correction prototype.

### Month 3: Evaluation, Planning Link, and Paper Assembly

Goal: turn the prototype into a defensible research result.

**Week 9: Full Evaluation Matrix**

- Run experiments across standard NovPhy levels, generated IratusAves levels, and selected novelty levels.
- Report final-state accuracy, physical violation rate, rollout error by horizon, and runtime cost.
- Include baselines such as pure LeWM / JEPA rollout, rollout-loss baseline if available, oracle-symbolic correction, and learned-symbolic correction.
- Deliverable: complete evaluation tables and rollout plots.

**Week 10: Planning and Adaptation Track**

- Connect the world model to a simple planner or action selector.
- Compare action selection with and without symbolic correction.
- For novelty scenarios, measure whether the corrected world model improves adaptation speed or final success.
- Deliverable: planning/adaptation result, even if small-scale.

**Week 11: Ablations and Reviewer Defenses**

- Run ablations for relation types, correction interval, oracle vs learned graph, event-dense vs event-sparse periods, and generated vs original levels.
- Prepare direct answers to likely reviewer objections: hand-engineering, DeepPHY overlap, NovPhy task mismatch, and comparison to rollout loss / MPC.
- Deliverable: ablation table and reviewer-risk memo.

**Week 12: Writing and Submission Package**

- Write the paper narrative around a concrete Angry Birds / Science Birds failure case.
- Put the problem, why it matters, and the one-sentence solution in the first 300 words.
- Prepare method diagram, rollout examples, failure-case visuals, metric plots, and reproducibility appendix.
- Deliverable: complete paper draft and experiment checklist for any remaining runs.

## Two-Month Compressed Plan

If the deadline is closer to two months, compress the project by reducing method ambition:

- Keep Month 1 mostly unchanged, because the environment, dataset, and baseline failure are mandatory.
- In Month 2, prioritize oracle-symbolic correction and strong evaluation over a fully learned symbolic module.
- Treat learned symbolic abstraction as a secondary experiment or future-work extension if it does not become reliable quickly.
- Submit the core story as: prolonged action effects expose JEPA rollout failure, and environment-derived symbolic structure demonstrates a strong upper bound for correcting that failure.

## Minimum Viable Paper

The minimum viable publishable version is not the full AG-PDDL-JEPA system. It is:

1. A clear definition of prolonged action effect environments.
2. A benchmark protocol on NovPhy / Science Birds for long-horizon physical cascade prediction.
3. Evidence that pure JEPA / LeWM rollouts fail through measurable error accumulation.
4. An oracle-symbolic correction experiment showing that symbolic physical structure reduces this failure.
5. A limited learned-symbolic prototype or a precise path toward learning the symbolic layer.

This version is safer than attempting all components at once. It gives a defensible scientific contribution even if the learned symbolic module is not fully mature within the project window.

## Metrics and Evidence

Use metrics that directly support the project claim:

- **Final-state accuracy**: predicted pig survival, collapsed structures, and stable object configuration.
- **Rollout error curve**: error at horizons 10, 25, 50, and 100.
- **Physical violation rate**: unsupported floating objects, object penetration, impossible contacts, or broken support relations.
- **Event prediction quality**: collision onset, collapse event, steady-state detection.
- **Planning success**: success rate or score when using model-predicted outcomes for action selection.
- **Adaptation gain**: reduced trials to recover after novelty, or improved final success under novelty.

Avoid relying only on image reconstruction loss. The paper's claim is about physical consequence prediction, so final-state and physical-plausibility metrics should be central.

## Key Risks and Mitigations

**Risk: the method becomes too ambitious.**
Mitigation: use oracle-symbolic correction first and treat learned symbolic extraction as a second-stage approximation.

**Risk: reviewers see symbolic relations as hand-engineering.**
Mitigation: distinguish fixed relation vocabulary from learned relation content; compare oracle, learned, and ablated relation settings.

**Risk: NovPhy is seen as an adaptation benchmark, not a world model benchmark.**
Mitigation: explicitly define a prediction benchmark on top of NovPhy and then add planning/adaptation as a downstream validation.

**Risk: DeepPHY or VLM-based Angry Birds work seems too close.**
Mitigation: emphasize that this project predicts future states and supports planning; it is not screenshot-to-action agent evaluation.

**Risk: baseline training takes too long.**
Mitigation: begin with small models, short horizons, and symbolic-state prediction before scaling to full image rollouts.

## Weekly Operating Rhythm

Each week should end with a binary checkpoint:

- A runnable command or notebook.
- A saved artifact: dataset, plot, table, trained checkpoint, or written memo.
- A short result summary: what worked, what failed, and what decision it forces.
- One locked next-week priority.

Do not let implementation work continue for more than one week without producing a measurable plot or table. The project succeeds only if the research claim becomes visible early.

## Immediate Next Steps

1. Finish the data collection pipeline around generated and standard Science Birds levels.
2. Define the stored trajectory schema and collect a small pilot dataset.
3. Implement the first baseline evaluation that measures final-state and horizon-based rollout error.
4. Write a one-page baseline failure memo with examples.
5. Start the oracle-symbolic scene graph extraction only after the baseline failure is measurable.
