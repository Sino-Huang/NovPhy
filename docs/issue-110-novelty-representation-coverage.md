# Issue #110 — representation coverage for all NovPhy novelty levels (design; no capture)

Parent: #108. Shared rules of #85 apply. Design only: **zero engine seconds**, no capture, no fit, no score.
Prior dispositions (#15–#112) are read-only inputs; nothing here edits the ICLR 2026 submission.

| Item | Location |
| --- | --- |
| Contract v2 (vocabulary, scene state, action contract, force laws, force-aware labels, objective) | `world_model/data/novelty_contract.py` |
| Runner (source verification, generator sweep, readiness, paired rosters, budget) | `scripts/prepare_novelty_coverage.py` |
| Frozen plan (2026-10-03T08:25:59Z, before any capture) | `.local-artifacts/issue-110-novelty-coverage-v1/plan.json` |
| Per-cell table / per-template table | `…/findings.md` / `…/readiness.csv` |
| Paired level files (3 920 instances) | `…/levels/` |
| Tests | `tests/test_issue_110_novelty_coverage.py` |

## Decision

**All 45 scenario × novelty cells go into the Phase-2 capture.** Under the frozen 18-slot contract (v1) only
2 / 45 cells are fully supported (appearance × single-force and × multiple-forces; the 5 normal cells are
supported on 25 / 40 of their templates). Under contract v2 all 45 are supported at design level. The capture
covers them through the 40 upstream normal/novel template pairs; the normal side of each pair covers the 5 normal
cells. Runtime readiness is decided by a rendered smoke gate before the campaign; a template that fails it makes
its cell `unsupported` with a typed reason.

## 1. Which engine realizes the novelties

Every #77/#109 campaign used one capture player: the original NovPhy game recovered with AssetRipper and rebuilt
with instrumentation (`Assembly-CSharp.dll` sha256 `82db3f42…`, identical in `issue-77-n1-v1`,
`issue-77-n2-appearance-v1`, `issue-109-capture-smoke-v1`). Its assembly contains `ABFan`, `AirTurbulence`,
`Magnet`, `InverseGravity`, `Storm`, `DieOnBirdHitCountPig`, `NoveltyData` and `ScenarioObjectIdentity`. Its
`resources.assets` contains every novelty prefab. The instrumentation assigns authored identities to novelty
objects, so a novelty source appears in the native trace as `runtime:novelty:0000`. The trace records
`world.gravity_vector` and per-body `gravity_scale` at every native step. It does not record forces.

The other player, `sciencebirdsgames/physics-v2` (built from `tasks/task_template_designer`), drops `<Novelty>`
nodes in `LevelLoader`. It is not the capture player and is not used.

## 2. Force laws (recovered source; every constant re-verified by the runner)

| level | source | law | constants | visible in RGB? |
| --- | --- | --- | --- | --- |
| 0 (type03xx/07xx), 3, 7 | Non/Novel/InverseAirTurbulence | `up × F` on non-pig, non-platform bodies inside a 0.64·scale box with speed > 0.1 | F = 1.5 / 4 / −10; box 4.48 × 6.4 at the template scale 7 × 10 | region yes; **the three variants share one sprite and colour**, so magnitude and sign are hidden |
| 2 | Fan | `−right × 25` on the launched bird while 0 < fan.x − bird.x < 2 and \|dy\| < 1.5 | thrust 25 | yes (own sprite, kinematic body) |
| 4 | Magnet | `15 × (magnet − body)` on birds and other-material blocks within radius 4; reversed on same-material (wood) blocks; a colliding body leaves the field lists until re-entry | k 15, r 4 | **looks like a normal wood Circle** (same sprite) |
| 6 | InverseGravity | `Physics2D.gravity = (0, 6)`; birds not on the slingshot get gravityScale −1 | g = +6 | no renderer |
| 8 | Storm | after the first bird is destroyed: `(3, 0)` on flying/dying birds and on blocks with speed > 0.1 | wind (3, 0) | no renderer until onset; a grey sprite (35.8 × 19.2 units, α 0.5) appears at onset |
| 5 | right slingshot (x = +12) | none | — | layout |
| 1 | Pink pig/blocks | none (PinkCircle mass 1.125, life 23.4) | — | yes (own sprites) |

Other source facts that bind the design:

- Bird death is ≥ 3 s after its first collision (`CheckVelocityToDie`), so storm onset needs a long enough window.
- BasicBig pigs need 3 distinct bird hits (life 1e8). BasicBigTwoShots and BasicBigFourShots need 2 and 4 hits
  (life 1e4). In single-shot capture, **all 9 multiple-forces cells have degenerate pig-removal verdicts**. They
  still serve rollout error; #107 should expect no mixed verdicts there.
- The clear condition (every pig dead) and the fail condition (birds exhausted) are identical on every level.

## 3. Contract v2 (`issue-110-representation-contract-v2`)

**Slots.** v2 has 28 slots: 4 birds, 8 blocks, 1 `novelty:0000`, 1 pig, 11 platforms, the slingshot, and 2
landscape slots. It is a superset of v1, and v1 slots keep their relative order. The capacities are the
maxima over the generator sweep (200 outcome-free seeds × 80 templates, including generator distractions and
the level-6 sky platform). Every template's generated levels fit v2 in 200 / 200 draws. Under v1, 48 templates
fit in 0 / 200.

**Carrier.** 2 header values, 4 scene values (`gravity_y_normalized`, `gravity_available`, `storm_active`,
`storm_available`), and 13 features × 28 slots: 370 values (v1: 236).

**Entities versus scene state.** A source with a body or renderer is an entity in `novelty:0000`: Fan,
turbulence region, Magnet, Pink blocks. A bodiless global effect is scene state: gravity and storm. Kinds name
only what is visually distinct: v1 kinds plus `fan` and `air-turbulence`. Magnet and Pink blocks are `block`.
InverseGravity and Storm declare the slot but are never present in it.

**Predicates (frozen before training).**
- micro: `force-on(source, body)` from the law above.
- macro: `external-force-active`.
- scene labels: `gravity-inverted`, `storm-active`.

A predicate requested at Δ is true if it holds at any frame in (t, t + Δ]. This matches the #113 per-Δ rule, and
#113's settle and structure-change redefinitions apply unchanged.

The labels are computed from the native trace plus the authored scenario. Where the trace lacks engine state,
the label uses an approximation (`LABEL_APPROXIMATIONS`): trigger overlap becomes the body centre being inside
the region, and the magnet's "removed after collision" becomes "not touching now". The smoke checks each
approximation against the residual acceleration of contact-free bodies.

**Action contract (work item 3).** Pull magnitude is 10–160 px, away from the targets. drag_x < 0 for the left
slingshot and > 0 for the right one. The engine's `TapShoot` → `DragBird` → `LaunchBird` path has no sign
restriction; the left-pull limit was Python-only. Level-5 candidates are the #93 grid mirrored (drag_x → −drag_x)
with the same suffixes. The model's action encoding is unchanged, so level 5 shows the model actions it has
never seen.

**Objective (work item 4), declared before any scoring.** Every level, level 7 included, uses one objective.
First-shot success is engine pig removal (the #85 `pig_removed` channel). The decision cost is the #99 tie-free
cost on `pig:0000`. Level 7's "changed goal" belongs to the turbulence agent: it pushes down with −10 instead of
up with +1.5. The engine's win and fail conditions do not change. Every template has exactly one pig, so the
single-pig cost machinery holds.

## 4. Per-cell readiness (45 cells; full table in `findings.md`)

| novelty level | cells | v1 blockers | v2 | identifiability (post-capture test) |
| --- | --- | --- | --- | --- |
| 0 normal | 5 (8 templates each) | 15/40 templates: authored platform overflow (falling type010104, rolling type010203/0403/0603, type010404) or NonNovelAirTurbulence | supported 40/40 | turbulence families: hidden magnitude |
| 1 appearance | 5 | rolling/falling/sliding: PinkCircle/PinkRectFat sit in `novelty:0000` | supported | not required |
| 2 fan | 5 | unrepresented entity/force | supported | source visible |
| 3 stronger turbulence | 5 | unrepresented force | supported | magnitude hidden |
| 4 magnet | 5 | unrepresented force | supported | source looks normal |
| 5 right slingshot | 5 | left-pull action contract | supported | none needed |
| 6 inverse gravity | 5 | unrepresented global force | supported | invisible global |
| 7 inverted turbulence | 5 | unrepresented force; objective undeclared | supported | sign hidden |
| 8 storm | 5 | unrepresented event | supported | onset hidden until the overlay appears |

No cell is `unsupported` at design level. The two typed risks that can still make one unsupported are both
smoke-gated: (a) a novelty entity missing from the trace or a launch-side failure; (b) storm onset outside the
capture window. Risk (b) is reported, not gated, and depends on the #113 window spec.

## 5. Capture membership (frozen rosters)

- **Units.** There are 40 pairs `pair-type010kSS`: the level-0 and level-k templates of the same upstream family.
  Each paired level has a normal and a novel instance with the same generation and engine seed. A screening
  attempt is accepted only if both sides pass.
- **Splits per pair.** `smoke` has 1 level (gate only, never fitted or scored). `fit` has 24: the normal side
  trains the zero-shot models and the novel side is few-shot adaptation data. `evaluation` has 24 and is never
  fitted.
- **Seeds.** Generation seeds are 768 000 001–768 279 999, engine seeds 768 500 001–768 779 999, and the generator
  sweep used 767 000 001–767 080 000. The seed audit checks these blocks against all 39 reserved ranges, including
  #109's blocks. The content audit checks against 84 prior-exposure sources, including #109's sealed cohort, and
  finds 0 collisions. Screening rejected 0 attempts.
- **Workbook restrictions.** These apply: Magnet levels get no wood-circle distractions.
- **Candidates.** The #93 Arm-B grid (16) is the primary inventory of #96/#104/Gate B, mirrored for the
  right-side slingshot. The fit splits carry the same grid, so the evaluation inventory lies inside training
  support.
- **Dispatch order.** Ordinal-major: every pair at ordinal k is captured before any pair at k + 1, so a stopped
  campaign keeps its cells balanced.

**Budget.** These numbers use the measured #109 throughput: 29.5 s/branch amortized at 6 workers, 84 MB/branch.
The 48-core server's worker count is unmeasured.

| paired levels per split per side | branches | wall h | GiB |
| --- | --- | --- | --- |
| 4 | 10 240 | 84 | 799 |
| **8 (recommended floor)** | **20 480** | **168** | **1 598** |
| 16 | 40 960 | 336 | 3 197 |
| 24 (full roster) | 61 440 | 504 | 4 795 |

The smoke gate is 160 branches, 1.3 h and 12.5 GiB. At 8 evaluation levels per cell, every per-cell interval has
≥ 8 clusters, and each novelty level has 40 levels for #107. Free space is 7.6 TiB on `/mnt/array`.

## 6. Declared post-capture protocols (in `plan.json` → `protocols`)

- **Encoder (work item 1).** The E99 `SpatialSlotParser` with 28 queries and the v2 kinds, on the E99 recipe. The
  hybrid and continuous predictor families share one encoder per condition, so every arm gets the same encoder
  budget.
- **Rollout error (acceptance).**
  - Requests: the nine F-Δ-α requests, from every evaluation decision state, at endpoints 15 / 60 / 225 frames.
  - Zero-shot: encoder and predictors fitted on normal fit data only.
  - Few-shot: per novelty level, 2 000 updates, batch 64, lr 1e-4, identical for every arm.
  - Zero-shot and few-shot are never pooled. The contrast is novel minus paired normal, with a DESCRIPTIVE
    level-clustered bootstrap.
- **Identifiability (work item 2).**
  - Probe: an MLP on k ∈ {1, 5, 15} frames of v2 carriers (primary k = 15), leave-one-level-out on the fit roster,
    then scored once on evaluation.
  - Verdicts: identifiable if AUROC ≥ 0.75 with lower bound > 0.65; not identifiable if the upper bound < 0.75.
  - Guards: ≥ 20 positive and ≥ 20 negative frames, plus an engine-feature ceiling.
  - Threshold rationale: #113's probes put recoverable predicates at 0.81–0.92 and unrecoverable ones at ≤ 0.74.
- **Smoke gate.** Per template, all of these must hold:
  - 0 typed capture failures;
  - `novelty:0000` present in the trace;
  - the expected gravity vector;
  - right-slingshot launches with vx < 0;
  - residual-acceleration signs that agree with the force-on labels.

## 7. Dependencies and handoff

- **#113.** Its capture-window spec must be merged before launch. Storm onset needs the first bird destroyed,
  which happens ≥ 3 s after its first collision, so levels 6 and 8 are window-sensitive.
- **Joint freeze with #104.** One joint #104 + #110 campaign freeze on `scripts/capture_pipeline_v2.py`. It fixes
  the prefix depth (recommended ≥ 8), the worker count, and the merge of normal fit data with #104's fit split.
- **#107.** Receives the cell list and rosters. Multiple-forces cells are expected to carry no decision headroom
  in single-shot capture.

## Reproduction

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate novphy && source env.sh
python -u -m scripts.prepare_novelty_coverage --validate   # re-derives plan, tables and 3 920 level files; exit 0
python -m pytest -q tests/test_issue_110_novelty_coverage.py
```

## Claim boundary

Readiness here comes from source, prefab, player-asset and generator evidence. No engine ran, no outcome was read,
and no model was trained or scored. "Supported" means that contract v2 can express the cell and the capture
player realizes it. Whether a hidden force is *identifiable* from observations is an open post-capture question
with a frozen test.

---

## Results (2026-10-16): contract v2 on the #104 captures (Work 1–5, Acceptance)

Everything above this line is the frozen design and is unchanged. This section reports the post-capture run.

### Where things are

| Item | Location |
| --- | --- |
| Runner (modes: repair, smoke, prepare, labels, encoder, fewshot-encoder, carriers, scene, assemble, train, fewshot-train, score, probe, publish, validate) | `scripts/run_novelty_coverage.py` |
| v2 labels and the repaired fan labeler | `world_model/data/novelty_labels_v2.py` |
| v2 models | `world_model/training/novelty_v2.py` (`SpatialSlotParserV2`, carrier builder, scene head, `HybridPredictorV2`, `ContinuousDynamicsV2`, probe) |
| Tests | `tests/test_issue_110_results.py` |
| Outputs | `.local-artifacts/issue-110-novelty-coverage-results-v1/` |

The v2 models extend the frozen E99 and #112 classes by subclassing. Earlier plans bind `spatial_slot_parser.py`, `cnn_hybrid.py` and `matched_dynamics.py` by hash, so none of them was edited. `novelty_contract.py` is also unchanged.

Output files:
- `plan.json` (frozen 2026-10-15T23:05:36Z, before any training or scoring), `smoke.json`, `repair/fan-repair.json`.
- `findings.md`, `summary.json`, `compute.json`, `readiness.csv`, `rollout-error.csv`.
- `handoff_107.json`, `records-manifest.json` (the raw-record hash manifest for 1 338 records).

Commits:
- Freeze: **f8986e73**.
- Publication: the commit that adds this section.

Validation: `python -u -m scripts.run_novelty_coverage --validate` exits 0. It regenerates every published file byte for byte from the retained records, checks the record manifest, and re-scores two rollout records exactly.

Guards: G1 isolation, G2 labels, G3 training and G4 scoring/manifest all pass.

### Sequence and compute

**Run sequence.** In order: repair → smoke (passed; projected wall 6.4 h on 4 GPUs) → freeze → labels → zero-shot encoder → few-shot encoders → carriers → scene heads → payloads → zero-shot training → few-shot training → probes → scoring (once) → publish → validate.

**Compute.** 35 444 GPU s and 37 337 summed process-wall s; 0 engine s. Wall clock was about 3.3 h after the freeze and about 1.5 h before it.

| Stage | GPU s |
| --- | --- |
| encoders (zero-shot + 8 few-shot) | 2 170 |
| carriers | 9 400 |
| zero-shot training | 6 871 |
| few-shot training | 13 224 |
| scoring | 1 857 |
| probes | 1 825 |
| scene heads | 99 |

Hardware: 4× RTX 5090 (the #85 rules name an RTX 3090).

### Work 1: slots, encoder, static fit

**Labels.** 34 348 complete branches were labelled with 0 typed label failures:
- issue110-fit: 10 208
- issue110-evaluation: 10 221
- issue109-fit: 13 919

On the v1 slots, the v2 labels equal the #112 retrain shot cache (20/20 shots: relations, masks, macros) and `s99.shot_labels` (presence identical, centers max |Δ| 0).

**Static fit.**
- v2: 80 / 80 templates. Every one of the 1 278 captured levels keeps every active bodied trace entity inside the 28 slots.
- v1: 19 / 80 templates keep every captured level inside the 18 v1 slots.

**Zero-shot encoder.**
- Architecture: SpatialSlotParserV2 (28 queries, 9 kinds, 3 relations, 3 macros).
- Recipe: E99, 12 epochs, batch 64, lr 1e-3.
- Data: 3 198 normal fit shots × 58 frames. Loss fell from 10.50 to 7.65 (epoch reports exported from the encoder checkpoint to the committed `encoder-training.json`: 10.49831 → 7.64830); E99 went from 10.89 to 7.18.

At the decision frame of the evaluation levels, the zero-shot parse reaches a mean slot-presence accuracy of 0.867 over the 80 templates. The lowest are 0.683 (type010603) and 0.737 (type010605). The worst template center RMSE is 0.341 in normalized image units. Per-template values are in `compute.json` → `static_fit`.

**Few-shot encoders.** Each novelty level adapts the zero-shot encoder with 2 000 updates (batch 64, lr 1e-4). Every level's loss fell, for example level 2 from 12.5 to 6.0 and level 4 from 14.0 to 8.1.

**Identical budgets.**
- Every condition uses one encoder, shared by the hybrid and continuous families.
- Every few-shot arm gets the same 2 000-update budget for its encoder, scene head and predictors.

### Work 2: force/agent descriptions, predicates, identifiability

**Fan labeler repair (before any training).**
- Cause: the Fan prefab runs `Fan.cs` (fanForce 50, windWidth 4, windHeight 1). `ABFan.cs` is attached but disabled.
- Law: `+transform.right × 50` on every non-platform rigidbody in the 8 × 1 trigger box.
- Label: the trace records the trigger box and the body colliders. `force-on` is true when a body's collider bounds, in the trigger frame, intersect that box.
- Residual acceleration of contact-free bodies:
  - Legacy (ABFan) labels disagree on every exercised fan template (medians 0, −50 and −100).
  - Repaired labels agree on every exercised template: the 5 fit templates and the 3 exercised smoke templates. The median projection is +100. Precision is 0.989–1.000 and recall 0.863–0.990.

**Predicates.** Frozen before training:
- micro: contact, supports, `force-on`
- macro: steady-state, structure-unstable, `external-force-active`
- scene: gravity-inverted, storm-active (the scene-head targets)

Per-Δ rule: the label is the OR over frames (t − Δ, t], and so is its mask. The smoke checked the batch gather against `window_any`.

**Identifiability.** 60 targets, k ∈ {1, 5, 15} with k = 15 primary. The verdict is taken on the evaluation roster, scored once. LOLO on the fit roster is reported alongside.

| verdict | targets |
| --- | --- |
| identifiable | 14 |
| not_identifiable | 20 |
| readiness_or_precision_insufficient | 19 |
| label_invalid (engine ceiling < 0.75) | 7 |

Which targets are identifiable:
- gravity-inverted: 4 / 5 (the fifth, sliding, is insufficient)
- storm-active: 3 / 5
- force-on: identifiable on 0/falling, 3/falling, 3/multiple_forces, 7/falling and 7/multiple_forces
- external-force-active: identifiable only on 0/falling and 3/falling. Elsewhere it is mostly not identifiable: hidden magnitude and sign, and a magnet that looks like a wood circle.

The level-0 targets use the normal sides of the turbulence families (pairs 03xx and 07xx). Under the frozen consequence rule, a non-identifiable force keeps its cell supported and its rollout error reported, and claims about reading that force are withheld.

### Work 3–4: action contract and level-7 objective

- **Level 5.** Candidates are the #93 grid mirrored (drag_x > 0) with the same suffixes. At assembly, every executed level-5 action equals its mirrored inventory action, and the v1 encoding is unchanged.
- **Level 7.** It uses the declared objective (engine pig removal; #99 tie-free cost on `pig:0000`) with no change. The objective is recorded in `handoff_107.json`.

### Work 5: rollout error on issue110-evaluation (Acceptance)

**Predictor recipe.**
- #112 v3 LC, through the retrain implementation, at 370 values.
- Families and seeds: hybrid and continuous (width 568, capacity-matched), seeds 20260908–10.
- Zero-shot training data: normal fit sides only: 19 032 shots, 5 355 828 positions, 796 ranking cells.
- Few-shot: per novelty level, from the zero-shot member, 2 000 updates at the LC end state. Level 8 has no mixed-verdict fit cell, so its ranking term is 0.

**Member usability.**
- All 6 zero-shot members are usable.
- 47 / 48 few-shot members are usable. few-shot-4 / continuous / seed 20260908 ended in an excursion (C-F-5 and C-F-15); it is a typed exclusion.

**What the tables report.**
- All 1 278 (condition, level) records were scored once, under `deterministic_scoring`.
- Metric: masked carrier MSE against the engine-projected v2 carrier, over engine-present slots plus gravity and storm.
- Times: 15, 60 and 225 frames. `rollout-error.csv` carries all three times, position MSE, the intervals, the C-F-Δ rows and every contrast. The tables below show 225 frames.
- Conditions: zero-shot (normal training) and few-shot (that level's adaptation) appear in separate rows and are never pooled.

**Descriptive readings.**
- A Δ = 15 request has the lowest error in 81 of the 85 (cell, condition) rows.
- In 42 / 45 cells, all three zero-shot F-1 requests are worse than persistence at 225 frames. Accumulated one-frame recursion error dominates.
- Few-shot lowers the F-15-macro error relative to zero-shot in 23 / 40 novel cells:
  - lower in all five cells of levels 2, 4, 6 and 8 (for example, 6/sliding goes from 0.106 to 0.025);
  - mostly higher on levels 1, 5 and 7;
  - mixed on level 3.
- Few-shot on level 7 makes F-1 rollouts diverge at 225 frames (masked MSE 1.1–2.5; persistence about 0.1), while F-15-macro rises by 0.005–0.051 over zero-shot. This is reported as is.
- Novel − normal contrast at 225 frames (F-15-macro):
  - zero-shot: the interval lies above 0 in 25 / 40 novel cells and below 0 in 2;
  - few-shot: the interval lies above 0 in 2 cells and below 0 in 15.
  - On 7/falling and 7/sliding the contrast is 0.0000 under both conditions, as is the persistence contrast. This is consistent with the inverted turbulence leaving those trajectories unchanged.

### Readiness (Acceptance): 45 / 45 supported, 0 unsupported

Every cell passed:
- the #104 smoke gate;
- v2 static fit;
- at least one complete evaluation level;
- finite zero-shot rollouts for all nine requests with usable hybrid seeds;
- every guard.

Two flags add caveats to cells that are still supported:
- `rollout_error_only`: the 9 multiple-forces cells. Their pigs need 2–4 distinct hits.
- `no_mixed_verdict_evaluation_level`: the evaluation grid has no level with mixed pig-removal verdicts, so #107 gets no decision headroom there. This applies to the 9 multiple-forces cells and to 3/single_force, 3/sliding, 4/sliding, 8/single_force, 8/rolling, 8/falling and 8/sliding.

The full per-cell readiness table and the identifiability verdicts are below.

### Handoff to #107: `.local-artifacts/issue-110-novelty-coverage-results-v1/handoff_107.json`

The handoff carries:
- the token (`supported`: every guard holds), the 45 supported cells, and per-cell flags, headroom and identifiability;
- contract v2: vocabulary, carrier layout, pig columns 175 / (180, 181), action encoding and bounds, the objective including level 7, and the repaired fan law;
- the v2 anchor rule: the condition's encoder parse of shot frame 0 (native step 30 000) with no prior, plus its scene head over the one-frame history (scene availability 0);
- loaders;
- the checkpoints with path, sha256 and (for predictors) usability: 9 encoders, 9 scene heads, and 54 predictor members (zero-shot and few-shot-1..8 × hybrid/continuous × 3 seeds);
- the evaluation inventories with verdicts: every issue110-evaluation grid branch per cell, with its action and engine `pig_removed`.

### Disclosures

- **Post-freeze runner edit.** After the freeze and before any scoring, two record lookups in the publication code (`rollout_tables`, the G1 side check) were changed from linear scans to dictionaries. The semantics are identical. No training, scoring or label code changed after f8986e73.
- **Labels stage restart.** The first labels launch was started with `nohup … &` from a shell call that then returned. It ended after 361 shots with no error logged. The stage was relaunched once. It resumed from the written files: label files are written atomically, and the labeler is deterministic and outcome-free. The ledger records only the relaunch's wall time (1 177 s). The interrupted first process's CPU time is not recorded, so the logged labels total is not exhaustive.
- **Zero-shot scene head.** Normal data have constant scene targets (gravity −1, no storm), so the zero-shot head is constant. Only the few-shot heads of levels 6 and 8 learn non-trivial scene state.
- **Excluded member.** One few-shot member is a typed exclusion: few-shot-4 / continuous / 20260908 (end-state excursion). The C-F-Δ rows are descriptive only.
- **Evaluation coverage.** The evaluation stream holds 639 levels with at least one complete branch (19 of 10 240 branches are campaign typed failures). One level of 2/rolling has no complete branch.
- **Phase-A review fixes (commit `7f689bdc`).** These are post-freeze runner edits for reporting, guard propagation and freeze/validation hardening. No evidence changed. `compute.json`, `handoff_107.json`, `records-manifest.json`, `rollout-error.csv` and `readiness.csv` are byte-identical (the #107 plan pins the first three by sha256). `summary.json` and `findings.md` gain the two reporting blocks below, and `encoder-training.json` is new. `--validate` exits 0.
  - *Freeze enforcement (B1).* `load_plan` now verifies every pinned input (as Gate B does), `smoke.json` against `smoke_sha256` and the labeler-repair record against `repair_sha256`, and refuses on drift. The runner's own entry is excluded, as in #102, because its post-freeze edits are reporting-only and disclosed. `--repair` refuses once `plan.json` exists. All pins match the retained files.
  - *Probe scoring (B2).* The 60 identifiability probe records were trained and scored by `--probe` without `deterministic_scoring()` and carry no scoring policy. The frozen G4 check covers the 1 278 rollout records only, so the plan's G4 text ("every record under deterministic_scoring") does not hold for the probe records. The probe models were not retained: each probe is trained inside the task, and the records keep only AUROCs and intervals. Re-scoring would therefore mean retraining, which the post-freeze rules forbid, so the probes were **not re-run**, and no value was re-scored or changed. A fresh process importing this runner has matmul TF32 off (as in `SCORING_POLICY`) and cuDNN deterministic off / cuDNN TF32 on; the probes use no cuDNN operation [INFERENCE about the original run, which recorded no flags]. Borderline verdicts are therefore not guaranteed to reproduce. `summary.json` `scoring_coverage` and the findings section "Scoring-policy coverage" now state this. `--probe` now scores under `deterministic_scoring()` and records the policy for any future probe record. It skips existing records, so nothing was rewritten. G4 and every token are as published.
  - *Validate docstring (B3).* It no longer claims a probe re-score; validate re-scores two rollout records.
  - *Guard propagation (F1).* A failed guard now turns every probe verdict, including each `by_k` verdict, into `readiness_or_precision_insufficient`, and every cell status into `readiness_or_precision_insufficient` instead of `unsupported`. The `summary.json` and handoff lists then separate insufficient from unsupported cells. All guards pass, so nothing changed.
  - *Encoder provenance.* `encoder-training.json` exports the encoder checkpoints' training summaries (zero-shot epoch reports, few-shot first/last-100-update losses, checkpoint sha256). No retraining was done.

### Claim boundary

- Rollout error, contrasts and identifiability are DESCRIPTIVE.
- Zero-shot and few-shot are separate conditions and are never pooled.
- "Supported" means that contract v2 represents the cell end to end (labels, encoder, carriers, predictors, finite rollout error). It makes no claim about decision quality or ranking; that is #107's question.
- A non-identifiable or `label_invalid` force keeps its cell supported, but claims about reading that force from observations are withheld.

### Reproduction

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate novphy && source env.sh
python -u -m scripts.run_novelty_coverage --validate          # exit 0
python -m pytest -q tests/test_issue_110_results.py tests/test_issue_110_novelty_coverage.py
```

### Tables

#### Readiness (45 cells)

| cell | status | flags | eval levels | mixed-verdict levels | identifiability, k = 15 |
| --- | --- | --- | --- | --- | --- |
| 0/single_force | supported | - | 64 | 26 | external-force-active: not_identifiable; force-on: readiness_or_precision_insufficient |
| 0/multiple_forces | supported | rollout_error_only; no_mixed_verdict_evaluation_level | 64 | 0 | external-force-active: not_identifiable; force-on: readiness_or_precision_insufficient |
| 0/rolling | supported | - | 64 | 33 | external-force-active: not_identifiable; force-on: not_identifiable |
| 0/falling | supported | - | 64 | 37 | external-force-active: identifiable; force-on: identifiable |
| 0/sliding | supported | - | 64 | 39 | external-force-active: not_identifiable; force-on: readiness_or_precision_insufficient |
| 1/single_force | supported | - | 8 | 5 | not required |
| 1/multiple_forces | supported | rollout_error_only; no_mixed_verdict_evaluation_level | 8 | 0 | not required |
| 1/rolling | supported | - | 8 | 4 | not required |
| 1/falling | supported | - | 8 | 3 | not required |
| 1/sliding | supported | - | 8 | 8 | not required |
| 2/single_force | supported | - | 8 | 1 | external-force-active: readiness_or_precision_insufficient; force-on: readiness_or_precision_insufficient |
| 2/multiple_forces | supported | rollout_error_only; no_mixed_verdict_evaluation_level | 8 | 0 | external-force-active: not_identifiable; force-on: readiness_or_precision_insufficient |
| 2/rolling | supported | - | 7 | 4 | external-force-active: not_identifiable; force-on: readiness_or_precision_insufficient |
| 2/falling | supported | - | 8 | 8 | external-force-active: not_identifiable; force-on: readiness_or_precision_insufficient |
| 2/sliding | supported | - | 8 | 4 | external-force-active: not_identifiable; force-on: not_identifiable |
| 3/single_force | supported | no_mixed_verdict_evaluation_level | 8 | 0 | external-force-active: not_identifiable; force-on: readiness_or_precision_insufficient |
| 3/multiple_forces | supported | rollout_error_only; no_mixed_verdict_evaluation_level | 8 | 0 | external-force-active: not_identifiable; force-on: identifiable |
| 3/rolling | supported | - | 8 | 4 | external-force-active: not_identifiable; force-on: label_invalid |
| 3/falling | supported | - | 8 | 2 | external-force-active: identifiable; force-on: identifiable |
| 3/sliding | supported | no_mixed_verdict_evaluation_level | 8 | 0 | external-force-active: label_invalid; force-on: readiness_or_precision_insufficient |
| 4/single_force | supported | - | 8 | 8 | external-force-active: label_invalid |
| 4/multiple_forces | supported | rollout_error_only; no_mixed_verdict_evaluation_level | 8 | 0 | external-force-active: label_invalid |
| 4/rolling | supported | - | 8 | 8 | external-force-active: not_identifiable |
| 4/falling | supported | - | 8 | 2 | external-force-active: not_identifiable |
| 4/sliding | supported | no_mixed_verdict_evaluation_level | 8 | 0 | external-force-active: label_invalid |
| 5/single_force | supported | - | 8 | 2 | not required |
| 5/multiple_forces | supported | rollout_error_only; no_mixed_verdict_evaluation_level | 8 | 0 | not required |
| 5/rolling | supported | - | 8 | 5 | not required |
| 5/falling | supported | - | 8 | 6 | not required |
| 5/sliding | supported | - | 8 | 4 | not required |
| 6/single_force | supported | - | 8 | 7 | gravity-inverted: identifiable |
| 6/multiple_forces | supported | rollout_error_only; no_mixed_verdict_evaluation_level | 8 | 0 | gravity-inverted: identifiable |
| 6/rolling | supported | - | 8 | 3 | gravity-inverted: identifiable |
| 6/falling | supported | - | 8 | 6 | gravity-inverted: identifiable |
| 6/sliding | supported | - | 8 | 6 | gravity-inverted: readiness_or_precision_insufficient |
| 7/single_force | supported | - | 8 | 6 | external-force-active: not_identifiable; force-on: readiness_or_precision_insufficient |
| 7/multiple_forces | supported | rollout_error_only; no_mixed_verdict_evaluation_level | 8 | 0 | external-force-active: readiness_or_precision_insufficient; force-on: identifiable |
| 7/rolling | supported | - | 8 | 5 | external-force-active: not_identifiable; force-on: not_identifiable |
| 7/falling | supported | - | 8 | 8 | external-force-active: readiness_or_precision_insufficient; force-on: identifiable |
| 7/sliding | supported | - | 8 | 3 | external-force-active: not_identifiable; force-on: readiness_or_precision_insufficient |
| 8/single_force | supported | no_mixed_verdict_evaluation_level | 8 | 0 | external-force-active: not_identifiable; storm-active: identifiable |
| 8/multiple_forces | supported | rollout_error_only; no_mixed_verdict_evaluation_level | 8 | 0 | external-force-active: label_invalid; storm-active: readiness_or_precision_insufficient |
| 8/rolling | supported | no_mixed_verdict_evaluation_level | 8 | 0 | external-force-active: readiness_or_precision_insufficient; storm-active: readiness_or_precision_insufficient |
| 8/falling | supported | no_mixed_verdict_evaluation_level | 8 | 0 | external-force-active: readiness_or_precision_insufficient; storm-active: identifiable |
| 8/sliding | supported | no_mixed_verdict_evaluation_level | 8 | 0 | external-force-active: label_invalid; storm-active: identifiable |

#### Rollout error at 225 frames: masked carrier MSE (level mean)

| cell | condition | F-1-continuous | F-1-micro | F-1-macro | F-5-continuous | F-5-micro | F-5-macro | F-15-continuous | F-15-micro | F-15-macro | persistence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0/falling | zero-shot | 0.194 | 0.197 | 0.229 | 0.059 | 0.057 | 0.060 | 0.038 | 0.037 | 0.038 | 0.102 |
| 0/multiple_forces | zero-shot | 0.195 | 0.198 | 0.229 | 0.094 | 0.095 | 0.096 | 0.077 | 0.077 | 0.076 | 0.130 |
| 0/rolling | zero-shot | 0.183 | 0.186 | 0.218 | 0.061 | 0.059 | 0.062 | 0.041 | 0.040 | 0.041 | 0.104 |
| 0/single_force | zero-shot | 0.109 | 0.111 | 0.146 | 0.038 | 0.038 | 0.043 | 0.023 | 0.023 | 0.024 | 0.093 |
| 0/sliding | zero-shot | 0.162 | 0.166 | 0.197 | 0.048 | 0.047 | 0.050 | 0.029 | 0.029 | 0.029 | 0.096 |
| 1/falling | few-shot-1 | 0.140 | 0.141 | 0.127 | 0.079 | 0.080 | 0.080 | 0.076 | 0.077 | 0.077 | 0.132 |
| 1/falling | zero-shot | 0.222 | 0.227 | 0.257 | 0.093 | 0.094 | 0.092 | 0.070 | 0.070 | 0.069 | 0.126 |
| 1/multiple_forces | few-shot-1 | 0.151 | 0.148 | 0.134 | 0.109 | 0.109 | 0.110 | 0.111 | 0.111 | 0.111 | 0.160 |
| 1/multiple_forces | zero-shot | 0.203 | 0.208 | 0.239 | 0.108 | 0.111 | 0.115 | 0.092 | 0.094 | 0.094 | 0.145 |
| 1/rolling | few-shot-1 | 0.124 | 0.122 | 0.117 | 0.062 | 0.062 | 0.066 | 0.059 | 0.060 | 0.060 | 0.119 |
| 1/rolling | zero-shot | 0.208 | 0.212 | 0.250 | 0.076 | 0.077 | 0.076 | 0.054 | 0.054 | 0.054 | 0.117 |
| 1/single_force | few-shot-1 | 0.158 | 0.154 | 0.123 | 0.076 | 0.076 | 0.072 | 0.073 | 0.073 | 0.070 | 0.132 |
| 1/single_force | zero-shot | 0.092 | 0.093 | 0.133 | 0.042 | 0.043 | 0.051 | 0.034 | 0.035 | 0.035 | 0.110 |
| 1/sliding | few-shot-1 | 0.132 | 0.132 | 0.116 | 0.066 | 0.067 | 0.067 | 0.059 | 0.060 | 0.059 | 0.121 |
| 1/sliding | zero-shot | 0.149 | 0.150 | 0.177 | 0.079 | 0.080 | 0.081 | 0.060 | 0.060 | 0.060 | 0.118 |
| 2/falling | few-shot-2 | 0.067 | 0.069 | 0.151 | 0.018 | 0.017 | 0.019 | 0.012 | 0.012 | 0.012 | 0.082 |
| 2/falling | zero-shot | 0.200 | 0.204 | 0.233 | 0.067 | 0.066 | 0.066 | 0.043 | 0.042 | 0.043 | 0.104 |
| 2/multiple_forces | few-shot-2 | 0.080 | 0.083 | 0.151 | 0.048 | 0.048 | 0.053 | 0.048 | 0.048 | 0.049 | 0.119 |
| 2/multiple_forces | zero-shot | 0.225 | 0.230 | 0.261 | 0.117 | 0.118 | 0.117 | 0.096 | 0.096 | 0.095 | 0.143 |
| 2/rolling | few-shot-2 | 0.113 | 0.121 | 0.254 | 0.043 | 0.043 | 0.052 | 0.039 | 0.040 | 0.043 | 0.101 |
| 2/rolling | zero-shot | 0.220 | 0.225 | 0.248 | 0.093 | 0.090 | 0.092 | 0.070 | 0.069 | 0.069 | 0.123 |
| 2/single_force | few-shot-2 | 0.059 | 0.061 | 0.158 | 0.026 | 0.027 | 0.033 | 0.026 | 0.026 | 0.027 | 0.099 |
| 2/single_force | zero-shot | 0.140 | 0.142 | 0.176 | 0.070 | 0.071 | 0.074 | 0.050 | 0.050 | 0.050 | 0.110 |
| 2/sliding | few-shot-2 | 0.086 | 0.090 | 0.195 | 0.033 | 0.034 | 0.040 | 0.030 | 0.033 | 0.036 | 0.097 |
| 2/sliding | zero-shot | 0.200 | 0.203 | 0.231 | 0.070 | 0.068 | 0.068 | 0.046 | 0.045 | 0.046 | 0.107 |
| 3/falling | few-shot-3 | 0.123 | 0.123 | 0.160 | 0.043 | 0.041 | 0.063 | 0.030 | 0.030 | 0.040 | 0.091 |
| 3/falling | zero-shot | 0.202 | 0.203 | 0.232 | 0.075 | 0.070 | 0.075 | 0.053 | 0.050 | 0.052 | 0.113 |
| 3/multiple_forces | few-shot-3 | 0.156 | 0.161 | 0.196 | 0.089 | 0.086 | 0.101 | 0.078 | 0.077 | 0.081 | 0.130 |
| 3/multiple_forces | zero-shot | 0.217 | 0.221 | 0.245 | 0.109 | 0.108 | 0.109 | 0.082 | 0.082 | 0.082 | 0.128 |
| 3/rolling | few-shot-3 | 0.127 | 0.127 | 0.163 | 0.046 | 0.044 | 0.064 | 0.036 | 0.036 | 0.044 | 0.100 |
| 3/rolling | zero-shot | 0.198 | 0.201 | 0.228 | 0.067 | 0.062 | 0.066 | 0.043 | 0.041 | 0.042 | 0.104 |
| 3/single_force | few-shot-3 | 0.114 | 0.122 | 0.176 | 0.032 | 0.029 | 0.065 | 0.015 | 0.014 | 0.030 | 0.081 |
| 3/single_force | zero-shot | 0.125 | 0.126 | 0.164 | 0.038 | 0.036 | 0.039 | 0.013 | 0.012 | 0.013 | 0.078 |
| 3/sliding | few-shot-3 | 0.115 | 0.118 | 0.162 | 0.041 | 0.039 | 0.052 | 0.027 | 0.027 | 0.029 | 0.091 |
| 3/sliding | zero-shot | 0.194 | 0.199 | 0.233 | 0.048 | 0.045 | 0.048 | 0.020 | 0.019 | 0.020 | 0.086 |
| 4/falling | few-shot-4 | 0.072 | 0.070 | 0.084 | 0.027 | 0.027 | 0.029 | 0.020 | 0.020 | 0.019 | 0.084 |
| 4/falling | zero-shot | 0.208 | 0.210 | 0.237 | 0.081 | 0.077 | 0.081 | 0.056 | 0.054 | 0.056 | 0.113 |
| 4/multiple_forces | few-shot-4 | 0.096 | 0.097 | 0.136 | 0.067 | 0.068 | 0.071 | 0.065 | 0.065 | 0.065 | 0.124 |
| 4/multiple_forces | zero-shot | 0.234 | 0.239 | 0.268 | 0.119 | 0.120 | 0.118 | 0.096 | 0.096 | 0.095 | 0.145 |
| 4/rolling | few-shot-4 | 0.081 | 0.080 | 0.094 | 0.040 | 0.040 | 0.039 | 0.036 | 0.036 | 0.033 | 0.100 |
| 4/rolling | zero-shot | 0.207 | 0.210 | 0.235 | 0.078 | 0.075 | 0.078 | 0.053 | 0.052 | 0.053 | 0.109 |
| 4/single_force | few-shot-4 | 0.068 | 0.070 | 0.114 | 0.043 | 0.043 | 0.050 | 0.041 | 0.041 | 0.042 | 0.105 |
| 4/single_force | zero-shot | 0.151 | 0.152 | 0.180 | 0.067 | 0.067 | 0.068 | 0.046 | 0.046 | 0.046 | 0.108 |
| 4/sliding | few-shot-4 | 0.064 | 0.063 | 0.080 | 0.022 | 0.023 | 0.023 | 0.015 | 0.016 | 0.013 | 0.079 |
| 4/sliding | zero-shot | 0.204 | 0.210 | 0.238 | 0.060 | 0.058 | 0.059 | 0.033 | 0.032 | 0.033 | 0.096 |
| 5/falling | few-shot-5 | 0.146 | 0.140 | 0.203 | 0.059 | 0.056 | 0.074 | 0.038 | 0.035 | 0.040 | 0.092 |
| 5/falling | zero-shot | 0.195 | 0.200 | 0.230 | 0.054 | 0.053 | 0.053 | 0.033 | 0.033 | 0.032 | 0.096 |
| 5/multiple_forces | few-shot-5 | 0.179 | 0.176 | 0.239 | 0.108 | 0.108 | 0.119 | 0.092 | 0.091 | 0.092 | 0.137 |
| 5/multiple_forces | zero-shot | 0.199 | 0.203 | 0.231 | 0.097 | 0.097 | 0.098 | 0.082 | 0.082 | 0.081 | 0.132 |
| 5/rolling | few-shot-5 | 0.139 | 0.132 | 0.196 | 0.054 | 0.051 | 0.068 | 0.036 | 0.032 | 0.038 | 0.093 |
| 5/rolling | zero-shot | 0.188 | 0.194 | 0.229 | 0.051 | 0.051 | 0.053 | 0.031 | 0.031 | 0.031 | 0.098 |
| 5/single_force | few-shot-5 | 0.138 | 0.132 | 0.197 | 0.061 | 0.063 | 0.070 | 0.059 | 0.061 | 0.059 | 0.125 |
| 5/single_force | zero-shot | 0.090 | 0.096 | 0.129 | 0.032 | 0.032 | 0.036 | 0.026 | 0.027 | 0.027 | 0.098 |
| 5/sliding | few-shot-5 | 0.137 | 0.133 | 0.189 | 0.066 | 0.067 | 0.077 | 0.056 | 0.056 | 0.058 | 0.114 |
| 5/sliding | zero-shot | 0.123 | 0.128 | 0.150 | 0.056 | 0.056 | 0.062 | 0.049 | 0.049 | 0.050 | 0.118 |
| 6/falling | few-shot-6 | 0.058 | 0.057 | 0.069 | 0.032 | 0.032 | 0.031 | 0.033 | 0.033 | 0.027 | 0.096 |
| 6/falling | zero-shot | 0.227 | 0.233 | 0.260 | 0.107 | 0.108 | 0.107 | 0.092 | 0.092 | 0.091 | 0.152 |
| 6/multiple_forces | few-shot-6 | 0.084 | 0.085 | 0.093 | 0.064 | 0.065 | 0.061 | 0.063 | 0.064 | 0.056 | 0.116 |
| 6/multiple_forces | zero-shot | 0.233 | 0.239 | 0.271 | 0.126 | 0.128 | 0.128 | 0.112 | 0.113 | 0.111 | 0.165 |
| 6/rolling | few-shot-6 | 0.091 | 0.090 | 0.098 | 0.064 | 0.064 | 0.061 | 0.061 | 0.061 | 0.057 | 0.124 |
| 6/rolling | zero-shot | 0.233 | 0.235 | 0.258 | 0.134 | 0.135 | 0.135 | 0.121 | 0.121 | 0.120 | 0.171 |
| 6/single_force | few-shot-6 | 0.041 | 0.041 | 0.057 | 0.029 | 0.028 | 0.027 | 0.029 | 0.028 | 0.022 | 0.094 |
| 6/single_force | zero-shot | 0.132 | 0.134 | 0.168 | 0.066 | 0.067 | 0.071 | 0.055 | 0.056 | 0.056 | 0.127 |
| 6/sliding | few-shot-6 | 0.066 | 0.065 | 0.077 | 0.033 | 0.033 | 0.032 | 0.029 | 0.031 | 0.025 | 0.094 |
| 6/sliding | zero-shot | 0.232 | 0.236 | 0.257 | 0.121 | 0.122 | 0.121 | 0.107 | 0.107 | 0.106 | 0.163 |
| 7/falling | few-shot-7 | 2.335 | 1.789 | 1.313 | 0.144 | 0.130 | 0.112 | 0.068 | 0.061 | 0.071 | 0.096 |
| 7/falling | zero-shot | 0.194 | 0.197 | 0.224 | 0.060 | 0.055 | 0.060 | 0.036 | 0.034 | 0.035 | 0.097 |
| 7/multiple_forces | few-shot-7 | 2.056 | 1.628 | 1.149 | 0.168 | 0.161 | 0.142 | 0.098 | 0.105 | 0.107 | 0.132 |
| 7/multiple_forces | zero-shot | 0.196 | 0.200 | 0.228 | 0.084 | 0.084 | 0.085 | 0.057 | 0.057 | 0.056 | 0.111 |
| 7/rolling | few-shot-7 | 2.396 | 1.824 | 1.347 | 0.130 | 0.119 | 0.100 | 0.043 | 0.049 | 0.045 | 0.091 |
| 7/rolling | zero-shot | 0.198 | 0.201 | 0.229 | 0.066 | 0.063 | 0.065 | 0.040 | 0.039 | 0.040 | 0.102 |
| 7/single_force | few-shot-7 | 2.099 | 1.593 | 1.217 | 0.108 | 0.104 | 0.095 | 0.048 | 0.052 | 0.057 | 0.100 |
| 7/single_force | zero-shot | 0.120 | 0.122 | 0.151 | 0.038 | 0.038 | 0.040 | 0.015 | 0.014 | 0.014 | 0.079 |
| 7/sliding | few-shot-7 | 2.508 | 1.899 | 1.346 | 0.134 | 0.122 | 0.102 | 0.046 | 0.051 | 0.047 | 0.092 |
| 7/sliding | zero-shot | 0.199 | 0.206 | 0.234 | 0.048 | 0.045 | 0.048 | 0.021 | 0.020 | 0.021 | 0.087 |
| 8/falling | few-shot-8 | 0.052 | 0.051 | 0.055 | 0.027 | 0.027 | 0.028 | 0.023 | 0.023 | 0.023 | 0.094 |
| 8/falling | zero-shot | 0.186 | 0.191 | 0.228 | 0.057 | 0.057 | 0.058 | 0.038 | 0.038 | 0.038 | 0.100 |
| 8/multiple_forces | few-shot-8 | 0.077 | 0.078 | 0.079 | 0.052 | 0.053 | 0.052 | 0.052 | 0.052 | 0.051 | 0.123 |
| 8/multiple_forces | zero-shot | 0.175 | 0.180 | 0.201 | 0.122 | 0.123 | 0.128 | 0.114 | 0.114 | 0.115 | 0.164 |
| 8/rolling | few-shot-8 | 0.037 | 0.038 | 0.040 | 0.024 | 0.024 | 0.026 | 0.026 | 0.026 | 0.026 | 0.103 |
| 8/rolling | zero-shot | 0.119 | 0.118 | 0.150 | 0.058 | 0.058 | 0.060 | 0.046 | 0.047 | 0.046 | 0.108 |
| 8/single_force | few-shot-8 | 0.042 | 0.043 | 0.044 | 0.031 | 0.031 | 0.031 | 0.032 | 0.032 | 0.032 | 0.105 |
| 8/single_force | zero-shot | 0.136 | 0.141 | 0.169 | 0.069 | 0.071 | 0.074 | 0.056 | 0.057 | 0.057 | 0.113 |
| 8/sliding | few-shot-8 | 0.041 | 0.043 | 0.045 | 0.036 | 0.036 | 0.037 | 0.036 | 0.037 | 0.036 | 0.108 |
| 8/sliding | zero-shot | 0.119 | 0.118 | 0.151 | 0.064 | 0.065 | 0.067 | 0.055 | 0.055 | 0.055 | 0.121 |

#### Novel minus paired normal at 225 frames (F-15-macro, the Gate-B unit; level-clustered bootstrap)

| cell | zero-shot | few-shot |
| --- | --- | --- |
| 1/falling | +0.007 [-0.025, +0.041] | -0.032 [-0.042, -0.024] |
| 1/multiple_forces | +0.020 [+0.009, +0.031] | +0.005 [-0.002, +0.014] |
| 1/rolling | +0.021 [+0.015, +0.027] | -0.021 [-0.027, -0.018] |
| 1/single_force | +0.002 [+0.001, +0.003] | -0.002 [-0.003, -0.001] |
| 1/sliding | +0.026 [+0.015, +0.037] | -0.014 [-0.022, -0.005] |
| 2/falling | +0.024 [+0.020, +0.028] | -0.013 [-0.024, -0.003] |
| 2/multiple_forces | +0.002 [-0.009, +0.012] | -0.018 [-0.038, +0.001] |
| 2/rolling | +0.013 [-0.005, +0.031] | +0.019 [+0.010, +0.028] |
| 2/single_force | +0.028 [+0.023, +0.033] | -0.006 [-0.014, -0.000] |
| 2/sliding | +0.020 [+0.012, +0.029] | +0.006 [-0.009, +0.024] |
| 3/falling | +0.013 [+0.009, +0.017] | -0.008 [-0.022, +0.006] |
| 3/multiple_forces | +0.010 [-0.003, +0.021] | +0.003 [-0.001, +0.007] |
| 3/rolling | -0.001 [-0.022, +0.018] | +0.000 [-0.017, +0.020] |
| 3/single_force | -0.001 [-0.003, -0.000] | +0.005 [-0.001, +0.012] |
| 3/sliding | +0.000 [-0.008, +0.006] | +0.001 [-0.005, +0.007] |
| 4/falling | +0.010 [+0.003, +0.015] | -0.002 [-0.009, +0.003] |
| 4/multiple_forces | +0.022 [+0.008, +0.036] | -0.010 [-0.020, +0.002] |
| 4/rolling | +0.014 [+0.007, +0.025] | -0.002 [-0.011, +0.007] |
| 4/single_force | +0.010 [-0.000, +0.020] | -0.021 [-0.050, +0.004] |
| 4/sliding | +0.011 [+0.000, +0.020] | -0.010 [-0.026, -0.001] |
| 5/falling | +0.010 [+0.005, +0.015] | -0.008 [-0.022, +0.003] |
| 5/multiple_forces | +0.019 [+0.012, +0.027] | +0.018 [-0.002, +0.037] |
| 5/rolling | +0.005 [+0.000, +0.009] | -0.020 [-0.030, -0.010] |
| 5/single_force | -0.004 [-0.015, +0.005] | +0.006 [-0.002, +0.017] |
| 5/sliding | +0.002 [-0.005, +0.011] | +0.005 [-0.011, +0.023] |
| 6/falling | +0.030 [+0.005, +0.054] | -0.034 [-0.057, -0.010] |
| 6/multiple_forces | +0.024 [+0.021, +0.027] | -0.021 [-0.034, -0.007] |
| 6/rolling | +0.050 [+0.031, +0.068] | -0.014 [-0.020, -0.007] |
| 6/single_force | +0.040 [+0.038, +0.043] | -0.038 [-0.040, -0.036] |
| 6/sliding | +0.067 [+0.055, +0.077] | -0.022 [-0.029, -0.016] |
| 7/falling | +0.000 [-0.000, +0.000] | +0.000 [-0.000, +0.000] |
| 7/multiple_forces | -0.018 [-0.030, -0.005] | -0.014 [-0.035, +0.009] |
| 7/rolling | -0.002 [-0.024, +0.020] | -0.006 [-0.012, +0.000] |
| 7/single_force | -0.003 [-0.008, +0.001] | -0.010 [-0.017, -0.002] |
| 7/sliding | +0.000 [-0.000, +0.000] | -0.000 [-0.000, +0.000] |
| 8/falling | +0.018 [+0.009, +0.027] | -0.004 [-0.018, +0.010] |
| 8/multiple_forces | +0.040 [+0.032, +0.049] | +0.014 [+0.004, +0.024] |
| 8/rolling | +0.025 [+0.020, +0.030] | -0.010 [-0.028, +0.004] |
| 8/single_force | +0.038 [+0.028, +0.046] | -0.010 [-0.018, -0.003] |
| 8/sliding | +0.029 [+0.020, +0.039] | +0.008 [-0.005, +0.021] |

#### Identifiability (k = 15, evaluation roster; carrier AUROC [interval], engine ceiling)

| task | verdict | carrier AUROC | engine ceiling | LOLO (fit) | positives / negatives |
| --- | --- | --- | --- | --- | --- |
| 0-falling--external-force-active | identifiable | 0.843 [0.774, 0.913] | 0.937 | 0.835 | 3763 / 68502 |
| 0-falling--force-on | identifiable | 0.847 [0.778, 0.912] | 0.935 | 0.855 | 4796 / 441817 |
| 0-multiple_forces--external-force-active | not_identifiable | 0.653 [0.585, 0.745] | 0.993 | 0.681 | 2000 / 71106 |
| 0-multiple_forces--force-on | readiness_or_precision_insufficient | 0.708 [0.631, 0.809] | 0.991 | 0.700 | 2000 / 400377 |
| 0-rolling--external-force-active | not_identifiable | 0.604 [0.482, 0.718] | 0.908 | 0.550 | 2932 / 69768 |
| 0-rolling--force-on | not_identifiable | 0.568 [0.444, 0.673] | 0.807 | 0.605 | 3545 / 447279 |
| 0-single_force--external-force-active | not_identifiable | 0.622 [0.539, 0.708] | 0.993 | 0.672 | 2608 / 70268 |
| 0-single_force--force-on | readiness_or_precision_insufficient | 0.684 [0.589, 0.779] | 0.992 | 0.763 | 2608 / 209518 |
| 0-sliding--external-force-active | not_identifiable | 0.646 [0.569, 0.719] | 0.967 | 0.589 | 4526 / 68173 |
| 0-sliding--force-on | readiness_or_precision_insufficient | 0.730 [0.634, 0.824] | 0.971 | 0.694 | 5245 / 373795 |
| 2-falling--external-force-active | not_identifiable | 0.649 [0.587, 0.730] | 0.965 | 0.612 | 1990 / 33568 |
| 2-falling--force-on | readiness_or_precision_insufficient | 0.723 [0.664, 0.809] | 0.970 | 0.694 | 2113 / 207115 |
| 2-multiple_forces--external-force-active | not_identifiable | 0.559 [0.459, 0.647] | 0.995 | 0.559 | 327 / 34626 |
| 2-multiple_forces--force-on | readiness_or_precision_insufficient | 0.617 [0.482, 0.769] | 0.993 | 0.602 | 327 / 234977 |
| 2-rolling--external-force-active | not_identifiable | 0.580 [0.434, 0.703] | 1.000 | 0.403 | 26449 / 3213 |
| 2-rolling--force-on | readiness_or_precision_insufficient | 0.724 [0.670, 0.841] | 0.990 | 0.732 | 27318 / 157375 |
| 2-single_force--external-force-active | readiness_or_precision_insufficient | 0.611 [0.507, 0.755] | 0.913 | 0.507 | 382 / 34303 |
| 2-single_force--force-on | readiness_or_precision_insufficient | 0.682 [0.597, 0.835] | 0.914 | 0.667 | 382 / 150956 |
| 2-sliding--external-force-active | not_identifiable | 0.642 [0.559, 0.748] | 0.936 | 0.583 | 253 / 35769 |
| 2-sliding--force-on | not_identifiable | 0.632 [0.565, 0.725] | 0.946 | 0.663 | 253 / 270854 |
| 3-falling--external-force-active | identifiable | 0.833 [0.756, 0.901] | 0.968 | 0.788 | 1322 / 34654 |
| 3-falling--force-on | identifiable | 0.829 [0.730, 0.920] | 0.958 | 0.798 | 1698 / 214324 |
| 3-multiple_forces--external-force-active | not_identifiable | 0.664 [0.629, 0.703] | 0.995 | 0.610 | 1721 / 34854 |
| 3-multiple_forces--force-on | identifiable | 0.785 [0.757, 0.811] | 0.994 | 0.735 | 1721 / 230519 |
| 3-rolling--external-force-active | not_identifiable | 0.651 [0.552, 0.745] | 0.777 | 0.527 | 2068 / 34401 |
| 3-rolling--force-on | label_invalid | 0.657 [0.551, 0.763] | 0.706 | 0.501 | 2259 / 214943 |
| 3-single_force--external-force-active | not_identifiable | 0.609 [0.525, 0.718] | 0.988 | 0.585 | 1641 / 34374 |
| 3-single_force--force-on | readiness_or_precision_insufficient | 0.717 [0.577, 0.826] | 0.985 | 0.617 | 1641 / 103948 |
| 3-sliding--external-force-active | label_invalid | 0.558 [0.448, 0.700] | 0.718 | 0.565 | 1490 / 34433 |
| 3-sliding--force-on | readiness_or_precision_insufficient | 0.630 [0.513, 0.763] | 0.767 | 0.590 | 1490 / 183015 |
| 4-falling--external-force-active | not_identifiable | 0.569 [0.502, 0.663] | 0.791 | 0.555 | 12207 / 20140 |
| 4-multiple_forces--external-force-active | label_invalid | 0.623 [0.573, 0.692] | 0.693 | 0.639 | 16408 / 20155 |
| 4-rolling--external-force-active | not_identifiable | 0.526 [0.455, 0.608] | 0.998 | 0.562 | 6387 / 27478 |
| 4-single_force--external-force-active | label_invalid | 0.520 [0.475, 0.577] | 0.748 | 0.616 | 7889 / 25605 |
| 4-sliding--external-force-active | label_invalid | 0.737 [0.573, 0.898] | 0.702 | 0.827 | 1375 / 31777 |
| 6-falling--gravity-inverted | identifiable | 0.965 [0.890, 1.000] | 1.000 | 0.833 | 36540 / 36412 |
| 6-multiple_forces--gravity-inverted | identifiable | 0.938 [0.856, 1.000] | 1.000 | 0.967 | 36365 / 36513 |
| 6-rolling--gravity-inverted | identifiable | 1.000 [0.999, 1.000] | 0.991 | 0.996 | 36605 / 35977 |
| 6-single_force--gravity-inverted | identifiable | 0.998 [0.988, 1.000] | 0.993 | 0.991 | 35658 / 36416 |
| 6-sliding--gravity-inverted | readiness_or_precision_insufficient | 0.733 [0.466, 0.965] | 0.889 | 0.884 | 36539 / 36139 |
| 7-falling--external-force-active | readiness_or_precision_insufficient | 0.767 [0.608, 0.914] | 0.850 | 0.765 | 1978 / 34219 |
| 7-falling--force-on | identifiable | 0.849 [0.712, 0.947] | 0.856 | 0.808 | 2156 / 220348 |
| 7-multiple_forces--external-force-active | readiness_or_precision_insufficient | 0.670 [0.577, 0.785] | 0.810 | 0.621 | 610 / 35958 |
| 7-multiple_forces--force-on | identifiable | 0.777 [0.720, 0.856] | 0.804 | 0.696 | 610 / 173599 |
| 7-rolling--external-force-active | not_identifiable | 0.571 [0.440, 0.646] | 0.905 | 0.536 | 3131 / 32934 |
| 7-rolling--force-on | not_identifiable | 0.635 [0.550, 0.706] | 0.937 | 0.621 | 3697 / 241274 |
| 7-single_force--external-force-active | not_identifiable | 0.579 [0.428, 0.735] | 0.995 | 0.782 | 648 / 35878 |
| 7-single_force--force-on | readiness_or_precision_insufficient | 0.641 [0.528, 0.785] | 0.989 | 0.681 | 648 / 104945 |
| 7-sliding--external-force-active | not_identifiable | 0.587 [0.496, 0.657] | 0.935 | 0.659 | 1752 / 34459 |
| 7-sliding--force-on | readiness_or_precision_insufficient | 0.677 [0.596, 0.777] | 0.928 | 0.679 | 1821 / 177908 |
| 8-falling--external-force-active | readiness_or_precision_insufficient | n/a n/a | n/a | 0.041 | 0 / 36291 |
| 8-falling--storm-active | identifiable | 0.939 [0.837, 0.996] | 0.993 | 0.565 | 114 / 36177 |
| 8-multiple_forces--external-force-active | label_invalid | 0.699 [0.450, 0.841] | 0.703 | n/a | 25 / 36393 |
| 8-multiple_forces--storm-active | readiness_or_precision_insufficient | 0.648 [0.296, 0.874] | 0.921 | 0.830 | 146 / 36272 |
| 8-rolling--external-force-active | readiness_or_precision_insufficient | 0.003 [0.002, 0.006] | 0.026 | 0.047 | 1 / 36301 |
| 8-rolling--storm-active | readiness_or_precision_insufficient | 0.631 [0.003, 0.891] | 0.853 | 0.670 | 18 / 36284 |
| 8-single_force--external-force-active | not_identifiable | 0.266 [0.123, 0.462] | 0.982 | n/a | 31 / 36163 |
| 8-single_force--storm-active | identifiable | 0.888 [0.657, 0.997] | 0.882 | 0.862 | 207 / 35987 |
| 8-sliding--external-force-active | label_invalid | 0.852 [0.699, 0.976] | 0.517 | 0.474 | 57 / 36463 |
| 8-sliding--storm-active | identifiable | 0.848 [0.700, 0.980] | 0.782 | 0.792 | 234 / 36286 |
