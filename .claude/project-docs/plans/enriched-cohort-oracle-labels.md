# enriched-cohort-oracle-labels - Work Plan

> Milestone 0 of `docs/high_level_plans/bg_ns_jepa_research_execution.md`.
> Sub-tracks 0a (macro/outcome label derivation) and 0b (oracle gate + unified per-frame tensor
> cohort), plus the first `physics_capture_v1` cohort collection.

## TL;DR (For humans)
**What you'll get:** The first real enriched cohort collected with the staged physics player, plus two
deterministic label layers derived from its frozen sidecars — a macro/structure layer
(`structure-unstable`, `cascade-active`, `collapsed`, `pigs-cleared`, `steady-state`) with outcome and
equilibrium labels, and the oracle scale-separation gate `phi*` — exposed to the existing lazy
`world_model/data` reader as per-frame tensors, with a machine-readable dataset-health report.

**Why this approach:** Every label is a documented, deterministic function of engine-exported facts that
already exist in `physics_state.jsonl` / `physics_events.jsonl`. No vision, no learned model, no change
to the frozen `physics_capture_v1` contract: labels land in a separate `physics_derived_labels_v1`
sidecar that is derived after collection and validated *against* the frozen sidecars, so a stale or
tampered label file fails closed instead of silently poisoning training.

**What it will NOT do:** No JEPA/world-model, controller, extractor, or training code (Milestone 1). No
learned macro layer or learned gate. No retroactive annotation of legacy RGB-only episodes. No writes to
the active cohort, the production player, or the canonical Unity project. No re-pin of the staged player.

**Effort:** Large
**Risk:** Medium — the collection path has never been executed end to end; five separate operational
gaps had to be closed before a single shot could be captured (see §Preconditions).

**Decisions made for you:** derived-label sidecar rather than a schema bump (the contract is frozen and
its verifier enforces it); `contacts_active` counts contacts whose relative speed clears an explicit
threshold, so a resting stack reads as quiescent rather than contact-saturated; absorbing macro states
stay absorbing by construction; thresholds are declared data recorded in every sidecar and report, never
implicit constants.

Your next move: review §Preconditions — two of them changed shared collection machinery.

---

> TL;DR (machine): Large, medium-risk deterministic label-derivation layer over frozen
> physics_capture_v1 sidecars, an oracle scale-separation gate, reader integration for per-frame label
> tensors, first enriched cohort collection, and a machine-readable dataset-health report.

## Scope
### Must have
- A `physics_derived_labels_v1` per-shot sidecar carrying, per render frame, the five macro-state
  predicates, the oracle gate `phi*`, and the scalar evidence each was derived from; plus one shot-level
  outcome/equilibrium record.
- A pure, non-Torch derivation module that consumes a parsed `PhysicsCapture` and returns immutable typed
  records, mirroring the existing `scripts/physics_capture_*.py` structure.
- A validator that re-derives labels from the sidecars and rejects any derived file that disagrees, is
  stale relative to its inputs, declares different thresholds, or violates the absorbing-state rules.
- An explicit `OracleGateSpec` (KE threshold, active-contact threshold, contact-activity speed
  threshold) recorded in every derived sidecar and every report; no implicit constants.
- Opt-in reader integration: `world_model/data` exposes the derived per-frame labels joined to RGB at
  exact `render_frame` equality, leaving the default RGB/action sample byte-identical.
- A CLI that derives labels for a shot, an episode, or a whole cohort root, and refuses to write inside
  the active cohort root.
- A machine-readable dataset-health report extending the existing inspection report with physics
  coverage, label coverage, oracle-gate distribution, macro-state distribution, and frame-exact
  alignment checks.
- The first `physics_capture_v1` cohort at `data/physics_capture_v1_cohort`, collected with the
  documented launcher from the physics worktree.

### Must NOT have (guardrails, anti-slop, scope boundaries)
- No edit to `docs/data_contracts/physics_capture_v1.md` / `.schema.json` semantics, the frozen event
  taxonomy, the declared capability list, or the `PHYSICS_CAPTURE_V1` descriptor's sidecar paths.
- No model, trainer, controller, extractor, encoder, or torch module beyond the existing reader surface.
- No learned or heuristic-visual label: every predicate must be a documented function of exported state
  and event records, and must cite the records it came from.
- No writes to `data/novphy_rollouts_dataset_20260708_171531`, `sciencebirdsgames/Linux` in the NovPhy
  checkout, the canonical Unity project, or the staged archive/receipt.
- No re-pin, rebuild, or republication of the staged player; the archive digest must stay
  `429cac1d748bed417b917d2838dc203d090668977dc8e56f5bac9a80ea95f2de`.
- No label written into a shot directory that would change the shot's accepted `physics_capture_v1`
  artifact validation, and nothing ever written under `frames/`.
- No claim that the cohort covers multiple regimes while the staged player ships one level.

## Preconditions (operational gaps closed before any shot can be captured)

All five were discovered by inspection this milestone; none was known to the handoff.

| # | Gap | Resolution |
|---|---|---|
| P1 | The canonical smoke marker `task-8-smoke.json` records archive `1c2a1bb…` (pre-final-repin) while the staged archive is `429cac1d…`; `resolve_physics_capture_provenance` fails closed. | Rerun `scripts/smoke_physics_capture.py` with `--report` at the documented marker path, per the contract's "verify the stage and then rerun the smoke test". |
| P2 | The generated collection script begins `source ~/cd_novphy`, which `cd`s to the NovPhy checkout and sets `PYTHONPATH` there; every relative `scripts/…` path then resolves to the legacy collector, which has no physics support at all. | Emit an explicit `cd` to the generation-time repo root plus `PYTHONPATH="$PWD"` in the generated script. Identical behaviour for legacy launches from the NovPhy checkout; correct behaviour from the worktree. |
| P3 | `build_collection_plan` asserts exactly 80 level buckets, but the staged physics player ships one level (`novelty_level_0/type2/Levels/3_9_6_1.xml`), and the worktree has no extracted `sciencebirdsgames/Linux` to inventory. | Add an explicit, opt-in scoped-inventory path so a plan may target a declared smaller bucket set; the 80-bucket invariant remains the default for the production inventory. |
| P4 | `CaptureBounds.resolve` caps a capture at 63 state records, but the launcher defaults (30 fps × 5 s = 150 frames) exceed it, so every physics shot would fail `state_limit`. | Collect with `ROLLOUT_FPS=30 ROLLOUT_DURATION=2` (60 frames). The bound is contract-fixed; the collection parameters must respect it. |
| P5 | `RESUME=1` requires `OUT_ROOT` to already exist. | Create `data/physics_capture_v1_cohort` before launching (`data/` is gitignored). |
| P6 | Collection produced no artifact and hung in the fresh-engine retry loop. The engine log showed `SocketException: Address already in use` from `PhysicsCaptureDirectSocket.StartListening`, caused by a previous run's engine still holding the port. Killing the collector's parent leaves the Java interface and Unity player alive, so a later run inherits an occupied port and never gets a physics listener. | Tear the whole process group down (collector, `game_playing_interface.jar`, `9001.x86_64`) and confirm the ports and X display are free before relaunching. A clean run binds the physics socket and captures normally. Not a port-derivation bug: `agent_port + 1` is served correctly once no stale engine holds it. |
| P7 | **Blocker.** The staged player emits `collision` events with an empty payload, but the frozen contract requires `contact_ids` and `relative_speed`. `validate_physics_shot_artifact` therefore rejects every shot containing a collision — which is every real gameplay shot. The instrumentation's accepted smoke shot contains no collision event, so this path was never exercised by the acceptance evidence. | Player-side defect; fixing it means changing the Unity exporter, rebuilding, republishing, and re-smoking (a full re-pin). Out of Milestone 0's scope, which must not re-pin the staged player. Recorded in `evidence/enriched-cohort-oracle-labels/task-7-collection-blocker.json` with the captured sidecars. |

**Cohort scope decision (owner-approved):** the first cohort is single-level. The staged player contains
exactly one level, so "a handful of levels across regimes" is not achievable without a rebuild/re-pin,
which is explicitly out of scope. Every report must state the covered bucket set and must not imply
regime coverage it does not have. Multi-regime collection is a follow-up gated on extending the staged
player.

## Label definitions (0a) — deterministic, engine-anchored

All predicates are evaluated per state record and cite the exported facts they derive from. `frame`
means the state record's `render_frame`; ordering uses the sidecar's monotonic
`(render_frame, render_time, fixed_step, fixed_time)` key.

Scalar evidence computed per state record:

- `total_kinetic_energy` — sum of `body.kinetic_energy_unity_units` over nodes with `body.present`.
  Absent bodies contribute nothing (they are explicitly null, never zero-massed).
- `active_contact_count` — count of raw contacts whose relative speed
  `|relative_velocity_a_to_b| >= contact_activity_speed` (default `0.01` unity units/second).
  A resting stack therefore has contacts but no *active* contacts.
- `support_edge_count`, `dynamic_node_count`, `pig_count` — direct counts over the record.

Macro predicates:

| Predicate | Definition | Absorbing |
|---|---|---|
| `steady-state` | The engine's own debounced stability interval: true from a `stable_entered` event's frame until the next `stable_exited` frame, and true before the first launch. | no |
| `structure-unstable` | Not `steady-state`, and either the support-edge set differs from the previous state record or `total_kinetic_energy >= kinetic_energy_threshold`. | no |
| `cascade-active` | Within a destruction burst: from the first frame carrying a `collision`, `explosion`, `entity_destroyed`, or `pig_removed` event after `bird_launched`, through the last such frame plus the debounce window, with no intervening `stable_entered`. | no |
| `collapsed` | A support edge present in an earlier state is absent later *and* its supported entity has an `entity_destroyed` / `pig_removed` event or has lost all support. Once true, stays true. | yes |
| `pigs-cleared` | No pig-class node remains in the state record, and the removals are accounted for by `pig_removed` events. Once true, stays true. | yes |

Outcome / equilibrium record (one per shot), the terminal macro state `z_pole`:

- `cleared` when a `level_cleared` event exists (carries its `score`);
- `failed` when a `level_failed` event exists (carries its `reason`);
- `settled` when the capture ends inside a `steady-state` interval with neither terminal event;
- `unsettled` otherwise — the capture window closed mid-cascade. This is an honest label, not a failure.

`level_cleared` and `level_failed` are mutually exclusive and at most one may appear; the validator
enforces that against the frozen taxonomy rather than trusting the derivation.

## Oracle gate (0b)

Per proposal §2.3, per state record:

```
phi*(x_t) = 1[ total_kinetic_energy(x_t) < kinetic_energy_threshold ]
          * 1[ active_contact_count(x_t) < active_contact_threshold ]
```

`OracleGateSpec` defaults: `kinetic_energy_threshold = 0.01`, `active_contact_threshold = 1`,
`contact_activity_speed = 0.01`. They are declared, validated positive, serialized into every derived
sidecar, echoed in the health report, and changing any of them changes the derivation digest — so a
cohort labelled under different thresholds can never be silently mixed with another.

`phi* = 1` means the fine relational ontology (`supports(A,B)`) is expected to hold over the coming
interval: the scene is at rest and nothing is actively interacting. This is the training-time ceiling the
learned gate `r_psi` is distilled against in Milestone 2; nothing here learns anything.

## Unified per-frame tensor cohort (0b)

The derived sidecar is already per-frame and rectangular, so no separate tensor cohort file is
introduced. The reader performs the join:

- `PhysicsSupervisionRequest` gains `include_derived_labels: bool = False`.
- `PhysicsFrameSupervision` gains `derived_labels: DerivedFrameLabels | None = None` (trailing default,
  so every existing construction and test is unaffected).
- The join key is exact `render_frame` equality against the same state record the RGB frame is bound to;
  a label file whose frames do not correspond one-to-one with the accepted states fails closed with a
  typed error, exactly like a malformed sidecar.
- `DerivedFrameLabels` exposes a fixed-order, documented numeric vector so a Milestone 1 collator can
  stack it without re-deriving semantics.

## Verification strategy
> Zero human intervention - all verification is agent-executed.
- Test decision: TDD with the repository's `unittest` suite over synthetic sidecar fixtures plus the
  committed golden fixture in `tests/fixtures/physics_capture_v1/`. Tests must not require a live game,
  Xvnc, or the collected cohort.
- Primary commands: `python -m unittest tests.test_physics_label_derivation`;
  `python -m unittest tests.test_prepare_rollout_dataset tests.test_world_model_data.WorldModelDataFixtureTests tests.test_world_model_data.RolloutArtifactValidatorTests`;
  `python -m unittest tests.test_physics_capture_contract tests.test_verify_physics_capture_docs tests.test_verify_physics_player`;
  `python scripts/verify_physics_capture_docs.py docs`.
- Evidence: `.claude/project-docs/evidence/enriched-cohort-oracle-labels/task-<N>.txt` per todo, holding
  the command, exit status, and the asserted excerpt.
- Required invariants: derivation is a pure function of the sidecars (same input → identical bytes);
  absorbing predicates never revert; `phi*` is exactly the conjunction of its two declared thresholds;
  every derived frame maps to exactly one accepted state record; a mutated sidecar invalidates its label
  file; the default reader sample is unchanged when labels are not requested; no report claims regime
  coverage beyond the collected buckets.

## Execution strategy
### Parallel execution waves

**Wave 1, preconditions:** Todos 1-2 refresh the smoke marker and close the collection runtime-path and
bucket-scope gaps. Independent of all label code.

**Wave 2, label layers:** Todos 3-5 build the derivation module, its validator + CLI, and the oracle
gate. Pure logic over fixtures; parallel with Wave 1.

**Wave 3, integration and data:** Todos 6-8 add reader integration, collect the cohort, derive its
labels, and emit the dataset-health report.

**Wave 4, final verification:** F1-F4 audit implementation, guardrails, evidence, and research-scope
fidelity.

### Dependency matrix
| Todo | Depends on | Blocks | Can parallelize with |
| --- | --- | --- | --- |
| 1 | None | 7 | 2, 3, 4, 5 |
| 2 | None | 7 | 1, 3, 4, 5 |
| 3 | None | 4, 5, 6 | 1, 2 |
| 4 | 3 | 6, 7 | 1, 2 |
| 5 | 3 | 6, 7 | 1, 2 |
| 6 | 3, 4, 5 | 8 | 1, 2 |
| 7 | 1, 2, 4, 5 | 8 | 6 |
| 8 | 6, 7 | F1-F4 | None |

## Todos
> Implementation + Test = ONE todo. Never separate.

- [x] 1. Refresh the canonical smoke marker for the current staged archive
  What to do / Must NOT do: Run `scripts/verify_physics_player.py --stage sciencebirdsgames/physics-v1
  --expect-sha <receipt>` (static path; the runtime probe needs the conda-env `Xvfb`) and then
  `scripts/smoke_physics_capture.py --stage sciencebirdsgames/physics-v1 --output-dir <tmp> --report
  .claude/project-docs/evidence/world-model-physics-instrumentation/task-8-smoke.json`, from the
  worktree with the `novphy` env active. Confirm the refreshed marker reports `status=accepted`,
  `phase=complete`, `protected_unchanged=true`, and `provenance.archive_sha256=429cac1d…`, and that
  `resolve_physics_capture_provenance` then succeeds. Must NOT hand-edit the marker, re-pin or rebuild
  the player, or run the smoke against anything other than the staged directory.
  Parallelization: Wave 1 | Blocked by: None | Blocks: 7
  Acceptance criteria: `resolve_physics_capture_provenance(archive, marker)` returns without raising and
  its `archive_sha256` equals the receipt; protected-root digests unchanged before/after.
  Commit: Y | `chore(physics): refresh canonical smoke marker for the published archive`

- [x] 2. Make the generated collection script CWD-safe and support a scoped level inventory
  What to do / Must NOT do: In `scripts/prepare_rollout_dataset.py`, emit an explicit `cd` to the
  generation-time repo root and `export PYTHONPATH="$PWD"` immediately after `source ~/cd_novphy` in the
  generated collection script, so relative `scripts/…`, plan, and data paths resolve against the repo
  the plan was generated in. Add an explicit opt-in scoped inventory so a plan may declare a smaller
  bucket set than the production 80; the 80-bucket assertion stays the default and must still fire for
  an unscoped production inventory. Must NOT change episode selection order, partition hashing, the
  active-cohort guard, resume semantics, or legacy generated-script behaviour when launched from the
  NovPhy checkout.
  Parallelization: Wave 1 | Blocked by: None | Blocks: 7
  Acceptance criteria: `python -m unittest tests.test_prepare_rollout_dataset` passes, including new
  tests proving the generated script re-enters the generation-time root, that a scoped inventory plans
  successfully, and that an unscoped production inventory still requires 80 buckets.
  Commit: Y | `fix(rollouts): resolve collection commands against the generating repo root`

- [x] 3. Derive macro-state and outcome labels from frozen sidecars
  What to do / Must NOT do: Add `scripts/physics_label_derivation.py` with frozen typed records
  (`OracleGateSpec`, `DerivedFrameLabel`, `ShotOutcome`, `DerivedLabels`) and a pure
  `derive_labels(capture, spec)` over a parsed `PhysicsCapture`. Implement the five macro predicates and
  the outcome/equilibrium label exactly as specified in this plan's label table, deriving only from
  exported state and event records. Absorbing predicates must latch. Must NOT import torch, open files,
  read RGB, infer anything from images, or mutate the parsed capture.
  Parallelization: Wave 2 | Blocked by: None | Blocks: 4, 5, 6
  Acceptance criteria: `python -m unittest tests.test_physics_label_derivation` passes with cases for a
  pre-launch quiescent frame, a mid-cascade frame, latch behaviour for `collapsed` / `pigs-cleared`,
  the four outcome classes, an event-free capture, and byte-identical repeat derivation.
  Commit: Y | `feat(physics-labels): derive macro-state and outcome labels`

- [x] 4. Serialize, validate, and re-derive the derived-label sidecar
  What to do / Must NOT do: Define `physics_derived_labels_v1`: a header record (schema version, shot
  identity, threshold spec, source sidecar digests, taxonomy) followed by one `frame_label` per accepted
  state and one `shot_outcome`. Add a validator that re-derives from the shot's sidecars and rejects
  disagreement, stale source digests, unknown fields, threshold mismatch, non-latching absorbing states,
  or frame sets that do not match the accepted states one-to-one. Add
  `scripts/derive_physics_labels.py` operating on a shot, episode, or cohort root, writing atomically
  and refusing the active cohort root. Must NOT write under `frames/`, alter `metadata.json`, or make a
  previously accepted shot fail `validate_physics_shot_artifact`.
  Parallelization: Wave 2 | Blocked by: 3 | Blocks: 6, 7
  Acceptance criteria: round-trip write→validate passes on the golden fixture; a mutated state sidecar,
  a mutated label row, and a threshold change each fail closed with a typed error; the shot still passes
  `validate_physics_shot_artifact` after the sidecar is written.
  Commit: Y | `feat(physics-labels): add validated derived-label sidecar and CLI`

- [x] 5. Implement the oracle gate and its per-frame tensor payload
  What to do / Must NOT do: Implement `phi*` as the declared conjunction, with `contacts_active` counted
  by the relative-speed threshold, and expose a fixed-order documented numeric vector per frame for
  downstream collation. Must NOT introduce a learned gate, a smoothed/hysteretic variant, or a
  threshold that is not carried in `OracleGateSpec`.
  Parallelization: Wave 2 | Blocked by: 3 | Blocks: 6, 7
  Acceptance criteria: threshold-boundary tests (at, just below, just above each threshold), a
  zero-contact frame, a resting-stack frame with many inactive contacts, and an absent-body frame all
  produce the specified `phi*`; the vector's field order is asserted explicitly.
  Commit: Y | `feat(physics-labels): add oracle scale-separation gate`

- [x] 6. Join derived labels into the lazy reader
  What to do / Must NOT do: Add `include_derived_labels` to `PhysicsSupervisionRequest` and a trailing
  `derived_labels` field to `PhysicsFrameSupervision`; load and validate the derived sidecar in
  `read_physics_shot` only when requested, joining on exact `render_frame`. Must NOT open the derived
  sidecar by default, change the default sample keys, or weaken the existing fail-closed behaviour.
  Parallelization: Wave 3 | Blocked by: 3, 4, 5 | Blocks: 8
  Acceptance criteria: `python -m unittest tests.test_world_model_data …` stays green; new tests prove
  labels are absent by default, present and frame-aligned when requested, and that a missing or
  misaligned label file raises a typed error.
  Commit: Y | `feat(world-model-data): expose derived physics labels to the reader`

- [ ] 7. Collect the first enriched cohort
  What to do / Must NOT do: Create `data/physics_capture_v1_cohort`, then run the documented launcher
  from the worktree with `PHYSICS_CAPTURE_V1=1`, the staged archive, the refreshed smoke marker,
  `RESUME=1 NOVPHY_YES=1 WORKERS=1 ROLLOUT_COUNT=1 ROLLOUT_FPS=30 ROLLOUT_DURATION=2`, and the scoped
  single-level inventory. Retain the plan, the failure ledger, and a redacted run log as evidence. Must
  NOT target the active cohort root, raise `WORKERS` above 1 without the isolation confirmation, exceed
  the 63-frame capture bound, or leave engine/Xvnc processes running.
  Parallelization: Wave 3 | Blocked by: 1, 2, 4, 5 | Blocks: 8
  Acceptance criteria: at least one accepted `physics_capture_v1` episode exists under the cohort root
  with `physics_state.jsonl`, `physics_events.jsonl`, frames, and a `metadata.json` recording the
  staged provenance digests; the active cohort and production player are unchanged.
  Commit: N (cohort data is gitignored; evidence only)

- [ ] 8. Emit the machine-readable dataset-health report
  What to do / Must NOT do: Extend the read-only inspection report with a physics section: capture
  coverage, derived-label coverage, oracle-gate distribution, macro-state distribution, outcome
  distribution, frame-exact alignment results, the threshold spec, and the covered bucket/split set with
  an explicit statement of what regimes are *not* covered. Run it against the collected cohort and store
  the JSON as evidence. Must NOT mutate the cohort, claim coverage that was not collected, or exit zero
  when a requested split has no accepted episodes.
  Parallelization: Wave 3 | Blocked by: 6, 7 | Blocks: F1-F4
  Acceptance criteria: the report validates against its declared shape on fixtures and on the real
  cohort; alignment checks pass for every accepted frame; the focused data suite is green against the
  new cohort.
  Commit: Y | `feat(world-model-data): report physics label and oracle-gate health`

## Final verification wave
> Runs in parallel after ALL todos. ALL must APPROVE.
- [ ] F1. Plan compliance audit — every scope-in deliverable present, every guardrail absent; frozen
  contract files unmodified; staged archive digest unchanged.
- [ ] F2. Determinism and fail-closed review — repeat derivation is byte-identical; mutated inputs,
  stale digests, and threshold drift all fail closed; no global random state; no torch in the
  derivation module.
- [ ] F3. Real data QA — labels derived on the actual cohort; reader join verified frame-exact on real
  shots; protected roots unchanged; report matches the artifacts on disk.
- [ ] F4. Research-scope fidelity — label semantics match proposal §2.2/§2.3 and the high-level plan's
  Milestone 0; no learned component; reports state single-level coverage honestly.

## Commit strategy
- Commit 1: preconditions (Todos 1-2).
- Commit 2: derivation, sidecar, CLI, oracle gate (Todos 3-5).
- Commit 3: reader integration and health report (Todos 6, 8).
- Do not commit cohort data, generated plans, or evidence output.

## Success criteria
- A collected `physics_capture_v1` cohort exists at `data/physics_capture_v1_cohort` with accepted
  episodes carrying engine-authoritative sidecars and staged provenance.
- Every accepted shot has a validated `physics_derived_labels_v1` sidecar whose labels re-derive exactly
  from its frozen sidecars and whose thresholds are explicit.
- The existing lazy reader returns frame-exact per-frame macro labels and `phi*` on request, and is
  unchanged when labels are not requested.
- A machine-readable dataset-health report states capture, label, gate, alignment, and split coverage,
  and names the regimes the cohort does not cover.
- The focused data suite, the physics contract suite, and the docs verifier are all green.
