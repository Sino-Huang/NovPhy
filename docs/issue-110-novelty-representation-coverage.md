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
