# Issue #77 — Novelty-level impact experiments: executable runbook

**How to use this document:** it is self-contained. A fresh agent session can execute
the whole program by reading only this file (plus the repo's `AGENTS.md`). It tracks
[issue #77](https://github.com/Sino-Huang/NovPhy/issues/77); background/design
rationale lives in
[issue-76-novelty-level-evaluation-design.md](issue-76-novelty-level-evaluation-design.md)
and [issue-76-novelty-coverage.md](issue-76-novelty-coverage.md) — read them only if
you need the *why*, not the *how*.

Small-project mode (2026-09-20 reframing): no authorization gates or audit cycles.
Kept discipline, because it is cheap and keeps results interpretable:

1. Decide cell membership, templates, seeds and metrics **before** looking at model
   outcomes on any new cell.
2. Report every attempted cell including failures; unsupported/deferred cells are
   recorded explicitly; never report only the novelty on which a system wins.
3. Zero-shot and few-shot are separate claims; never mixed.
4. Both comparison arms always train with identical budgets.

## Objective

Measure whether and where novelty levels change dynamics-model performance: hybrid
(BG-NS-JEPA-style) vs genuine pure-continuous baseline under identical training
budgets. Claims: (a) normal-mechanics breadth (stage N1); (b) zero-shot novelty
generalization and few-shot adaptation on matched normal/novel pairs (stage N2).

## 0. Environment (verified on this host 2026-09-20)

```bash
conda activate novphy && source env.sh   # env.sh only sets PYTHONPATH=$PWD and XDG_DATA_HOME=$PWD/.cache/xdg
```

- Prebuilt game player: `.local-artifacts/issue-76-shared-history-smoke-v4/player/`
  (contains `9001-player.x86_64`, `9001.x86_64`, `game_playing_interface.jar`,
  `serverbackup`). **Hard prerequisite** — nothing in the repo rebuilds it
  automatically; rebuilding needs Unity 2019.4.41f2 batch mode (see `AGENTS.md`).
- System binaries: `java` (game bridge), `Xvnc` (per-attempt framebuffer; captures are
  NOT headless), `ffmpeg` (only for WebM audits). All present at `/usr/bin/`.
- GPU: RTX 3090 for training/eval (`--device cuda`; note `run_issue_74_matched_dynamics`
  defaults to cpu — always pass `--device cuda`).
- Disk: capture runner enforces ≥256 GiB free; 801 GiB free at check time. R3's tree
  grew to ~36 GiB for 616 branches.

## 1. How capture works (30 seconds)

A campaign runner (supervisor) dispatches *branches* = lineage × one fixed action to
isolated worker processes (pool of 8). Each worker clone-hardlinks the player into an
attempt dir, starts a private `Xvnc` display, launches the Java bridge + Unity binary,
pauses native physics at fixed step 30000, seals paired RGB views, fires one shot, and
persists native physics capture v2 + aligned observation traces. Wire protocol:
`src/webui/bridge.py`. Process hygiene is group-wise SIGTERM→SIGKILL
(`scripts/process_lifecycle.py`) — never kill individual processes by hand.

**Operational rule that matters most:** every runner freezes its own source + settings
into a `plan.json` at `--prepare` time and refuses to run after any source edit
(`load_plan` re-hashes). Therefore **each new campaign = new script modules + new
output root**, copied and adapted from the pattern below. Never edit the frozen v1
modules to "reuse" them.

The R3 reference implementation (read these, then copy-adapt):

| Role | Frozen v1 module | Copy to (new identity `issue-77-*`) |
| --- | --- | --- |
| plan/prepare | `scripts/prepare_issue_76_bounded_transfer.py` | `scripts/prepare_issue_77_n1.py` etc. |
| runner/supervisor | `scripts/run_issue_76_bounded_transfer.py` | `scripts/run_issue_77_n1.py` etc. |
| worker capture | `scripts/issue_76_bounded_transfer_capture.py` → `scripts/issue_76_shared_history_capture.py` | reuse as library, unchanged |
| level install/materialization | `scripts/issue_76_expansion.py` (`install_level`, `materialize`) | reuse as library, unchanged |
| cohort generation | `world_model/data/successor_cohort.py` | extend `GENERATOR_FAMILIES` (see §3) |

Campaign lifecycle commands (module names are the NEW ones you create):

```bash
python -u -m scripts.prepare_issue_77_n1 --write-plan     # plan.json + inventory.json
python -u -m scripts.run_issue_77_n1 --prepare            # copies player, writes campaign.json
python -u -m scripts.run_issue_77_n1 --run                # dispatch loop, 8 workers
python -u -m scripts.run_issue_77_n1 --validate           # writes coverage.json
```

Output layout per campaign root: `plan.json`, `inventory.json`, `campaign.json`,
`ledger.json`, `coverage.json`, `supervisor.log`, `markers/`, `results/`, `receipts/`,
`attempts/<branch>/` (runtime, paired-inputs, aligned traces, physics capture v2).
Resource limits to carry over: workers 8, per-worker RSS 4,096 MiB / aggregate 32,768
MiB, `attempt_seconds` backstop ~1,200 s (R3's nominal 420 never bound; real
per-attempt cost is non-stationary: early ≈383 s, late ≈774 s, p90 ≈1,180 s — size the
combined cap from the late rate), `technical_retries` 0, minimum free 256 GiB,
artifact cap 150 GiB. Keep terminal-incomplete ledgers refusing top-up; keep all
failure records.

## 2. Seed/lineage disjointness ledger (never reuse these ranges)

| Range | Used by |
| --- | --- |
| 6_300_000 / 63_000_000 offsets | issue-62 pilot/production cohorts |
| 761_700_001–761_700_700 / 761_800_001–761_800_700 | parked (§8 of R3 plan) |
| 761_900_000+ordinal / 762_000_000+ordinal | R3 generation/engine seeds |
| 763_100_000+ / 763_200_000+ | R3 readout/selector init seeds |
| 20260908–20260910 | #74 matched training seeds |

**Suggested new bases (pin in each new prepare module):** N1 generation
`764_000_000+ordinal`, engine `764_100_000+ordinal`; N2 generation `764_200_000+`,
engine `764_300_000+`; adaptation init seeds `764_400_000+`. Verify disjointness
programmatically in the prepare module (copy R3's self-check, extended to this table).

## 3. Stage N0 — representation readiness (per cell)

Frozen perception contract (`.local-artifacts/issue-71-hybrid-readiness-v1/plan.json`
→ `contract.vocabulary`): 18 slots = bird:0000-0002, block:0000-0004, pig:0000,
platform:0000-0005, slingshot:0000, world:landscape:0000-0001; carrier 236 dims
(2 globals + 18×13); action dim 5; action bounds drag-x ∈ [−160,−10], drag-y ∈
[−80,80], release 600 ms (`world_model/data/successor_cohort.py:53-58`).

| Cell | Verdict | What to do |
| --- | --- | --- |
| normal `type010101/02` | ready | existing corpora (#62 release, R3 retained) |
| normal rolling `type010103` | ready | template `tasks/task_templates/novelty_level_0/type010103/Levels/00001_0_1_010103_0_3.xml` |
| normal sliding `type010105` | ready | template `.../type010105/Levels/00001_0_1_010105_0_5.xml` (already an R3 family) |
| normal falling `type010104` | **needs amendment A** | template has 8 platforms > 6 platform slots; no ≤6-platform alternative exists locally |
| appearance (level 1) | ready + binder fix | pairs for `type010101/02` pass static fit; R3 binder asserts BirdRed-only (`prepare_issue_76_bounded_transfer.py:230`) — relax for PinkBigPig in the new prepare module |
| right-side slingshot (level 5) | **needs amendment B** | action contract is left-pull only |
| fan/turbulence/magnetic/gravity/storm (levels 2,3,4,6,8) | `unsupported` this round | need new represented entities / observation history / predicate semantics — record, don't drop |
| changed goal (level 7) | `unsupported` this round | task-objective semantics must be redeclared |

**Amendment A (platform slots 6→8; carrier 236→262)** — only if falling is included:
edit `SLOTS/DIM/CONFIG` in `world_model/training/cnn_hybrid.py:20-22` (heads
self-resize; `checkpoint_contract` follows), the `latent_dim` binding in
`world_model/data/deployment_temporal.py:573`, parser `max_entities`
(`scripts/run_issue_70_parser_repair.py:60-77,96-98`), and hardcoded 236/18 in
`scripts/run_issue_72_matched_grid.py:105` and
`scripts/run_issue_76_dynamics_diagnostic.py:274,298,371,435`. Consequence: **every
existing checkpoint is rejected** — retrain parser + both arms + controller from
scratch. All under new plan identities. Alternative: defer falling, run N1 with
rolling+sliding first (recommended for the first batch).

**Amendment B (right-pull)**: widen `ACTION_BOUNDS.drag_x`
(`world_model/data/successor_cohort.py:53-58`), re-author candidate grids
(`scripts/prepare_issue_76_bounded_transfer.py:248` pattern; #72 grid at
`scripts/run_issue_72_matched_grid.py:57-59`). Action dim stays 5 (no checkpoint
rejection), but all prior data is left-pull-only → versioned re-capture + retrain.
Defer unless level 5 is explicitly wanted.

## 4. Stage N1 — normal-mechanics expansion (first batch: rolling + sliding)

1. **Membership (pin before any rendered run):** families `type010103` (rolling) and
   `type010105` (sliding), 8 lineages each from the templates in §3, 13 fixed actions
   per lineage (copy `candidate_actions` from `prepare_issue_76_bounded_transfer.py:248-253`:
   reference drag (-80,10) + 12 angles 5°..82° at radius 80 px). Falling
   (`type010104`) joins as +8 lineages only after amendment A. Identity scheme
   `issue-77-n1-NNN`; seeds per §2.
2. **Capture:** copy-adapt the R3 modules per §1 into `prepare_issue_77_n1.py` /
   `run_issue_77_n1.py`, output root `.local-artifacts/issue-77-n1-v1/`. Set
   `FAMILIES=('type010103','type010105')`, `novelty_level=0`. Budget: 2×8×13 = 208
   branches ≈ 33 worker-h at the R3 mean rate (563 s/branch; up to ≈45 worker-h at
   the late-ordinal 774 s rate), ≈4–6 h wall at 8 workers; +104 branches ≈ 17
   worker-h when falling joins. Run the lifecycle commands from §1;
   keep `--validate` mandatory.
3. **Train both arms** on the expanded corpus (new identity, e.g.
   `.local-artifacts/issue-77-n1-train-v1/`), copied from
   `scripts/run_issue_74_matched_dynamics.py`: seeds (20260908, 20260909, 20260910) ×
   arms (continuous, hybrid), 9,000 steps, batch 64, AdamW 1e-4, identical
   lineage groups/minibatches per paired seed. Keep the capacity contract (continuous
   width matched to hybrid parameter count, 2% tolerance).
   ```bash
   python -u -m scripts.run_issue_77_n1_train --prepare
   python -u -m scripts.run_issue_77_n1_train --train --device cuda   # ~4 GPU-h
   ```
   (If amendment A was applied, first retrain the parser: `run_issue_70_parser_repair`
   pattern, new identity.)
4. **Evaluate** with the dynamics-diagnostic pattern (`scripts/run_issue_76_dynamics_diagnostic.py`
   copy-adapted): fixed systems `continuous_h{1,5,15}` +
   `hybrid_{continuous,micro,macro}_h{1,5,15}`, recursive carrier error at common
   endpoints t ∈ {15,30,60,150,225}, per-field errors, ranked-action regret vs REAL
   settled replay cost (#72 grid tooling `scripts/run_issue_72_matched_grid.py`
   pattern), ordinal09 prior retained. Publish tables like
   `scripts/run_issue_76_closeout.py` does (`summary.json`, `comparisons.csv`,
   `findings.md`).
5. **Verify:** `coverage.json` exists with all branches admissible-or-typed-failures;
   six checkpoints (3 seeds × 2 arms) load under the contract; `--validate` passes;
   run the ported unit tests (copy `tests/test_issue_76_bounded_transfer_{campaign,capture,inventory}.py`
   and `tests/test_issue_74_matched_dynamics.py` patterns to the new identities).
   Claim scope: **normal-mechanics breadth only.**

## 5. Stage N2 — matched normal/novel pairs (first batch: appearance)

1. **Membership (pin before any rendered run):** appearance (level 1) pairs for
   `type010101` and `type010102` — both pass static slot+action fit
   (`novelty-inventory.json` → `pairs`, `both_static_slot_action_fit: true`):
   - normal: `novelty_level_0/type010101/Levels/00001_0_1_010101_0_1.xml` ↔ novel:
     `novelty_level_1/type010101/Levels/00001_0_1_010101_1_1.xml`
   - normal: `novelty_level_0/type010102/Levels/00001_0_1_010102_0_2.xml` ↔ novel:
     `novelty_level_1/type010102/Levels/00001_0_1_010102_1_2.xml`
   8 lineages per side per family; same 13 actions; seeds per §2. Later batches add
   falling/sliding novelties as their N0 amendments land; levels 2,3,4,6,8 stay
   `unsupported` until their representation work is funded.
2. **Capture the novel side** (normal side already exists in the #62/R3 corpora):
   copy-adapt to `prepare_issue_77_n2.py` / `run_issue_77_n2.py`, output root
   `.local-artifacts/issue-77-n2-appearance-v1/`, `novelty_level=1`, relax the
   BirdRed-only binder check (§3). Budget: 2 families × 8 lineages × 13 = 208
   branches ≈ 33 worker-h.
3. **Zero-shot evaluation:** score frozen N1-era (or #74) checkpoints on both sides
   of each pair — no fitting on the novelty. Report normal vs novel separately, plus
   the paired change in the hybrid-minus-continuous effect with paired uncertainty
   (bootstrap over lineages/seeds).
4. **Few-shot adaptation:** adapt both arms on each novel cell with identical declared
   data/update budgets and paired seeds (suggested: same 9,000-step recipe at reduced
   data, or 2,000-step adaptation — pick one, declare it in the module header before
   running); evaluate on held-out novel lineages.
5. **Controls:** independent pure-continuous, same-hybrid-checkpoint fixed modes,
   ordinal09 prior; the appearance cell doubles as the perception control (it probes
   the shared CNN parser, not dynamics).
6. **Verify + report** as in §4.5, with zero-shot and few-shot tables separated.

## 6. Gotchas (all verified against code)

- Several template XMLs **declare `encoding="utf-16"` while containing ASCII bytes** —
  always read via `tasks/task_generator/canonical_materialization.py:70-81`
  (declaration rewrite + reparse), never naive `ET.parse`.
- Templates are bound at plan-freeze time, not read live at capture time; level
  install is data-driven (`issue_76_expansion.install_level` writes
  `9001_Data/StreamingAssets/Levels/novelty_level_{n}/{family}/Levels/{id}.xml`).
- `run_issue_74_matched_dynamics.py` defaults to `--device cpu`. Always pass cuda.
- Data role split when extending training corpora: lineages with 1-based index
  divisible by 5 supervise the controller only, never the predictor; calibration-role
  states are for development scoring only (keep this convention).
- A terminal-incomplete ledger (`combined_collection_limit`, etc.) refuses top-up
  forever — size caps generously up front instead.
- Worker teardown is process-group kill; a leaked Unity process keeps the physics
  port bound and poisons later attempts.
- Standard test invocation: `python -m unittest tests.<module> -q` (pytest also
  works); run the suites for every module you copy-adapt before first dispatch.

## 7. Done when

- N0: amendments A/B either landed symmetrically or cells recorded `unsupported`.
- N1: all scheduled branches executed or retained as typed failures; both arms
  retrained at identical budgets; comparison tables published and reproducible by
  rerunning the evaluation commands; claims limited to normal mechanics.
- N2: zero-shot and few-shot results reported separately with uncertainty; appearance
  perception control included; unsupported/deferred cells listed; no
  favorable-novelty selection.
- Readiness record updated: cells tested count moves off 0/45.
