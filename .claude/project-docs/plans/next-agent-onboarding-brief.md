# Onboarding brief — text-only coding agent, NovPhy

Written 2026-08-11, at the close of wave `runtime-repin-gate-20260810` (third session, verdict
`still_blocked`). This brief replaces ad-hoc context transfer. Read it first, then read the files it
names, then start at §5.

---

## 0. Your operating environment

| Fact | Value |
|---|---|
| Critical-path worktree | `/mnt/array/sukaih/Project/.novphy-worktrees/physics-unity-2019.4` |
| Branch | `physics-unity-2019.4` |
| HEAD at handoff | `0840da9abd17e6f1ff9a87b4db60fe170786bf2a` |
| Tracked drift at handoff | 0 |
| Unpushed commits | 8 (nothing has been pushed) |
| Main checkout | `/mnt/array/sukaih/Project/NovPhy`, branch `env` — **do not do critical-path work here** |
| Staged player pin | `sciencebirdsgames/physics-v1/` — **untracked / gitignored**, verified by `archive.sha256` = `429cac1d748bed417b917d2838dc203d090668977dc8e56f5bac9a80ea95f2de` |

**You do not need a screen.** No task in §5 requires viewing a desktop, a screenshot, or a rendered
frame. Pixels enter this system only as tensors (`world_model/data/dataset.py:324` → `:135`), and the
frozen capture contract structurally forbids screen-grabbed frames
(`docs/data_contracts/physics_capture_v1.schema.json:105` pins `"source": {"const":
"synchronized_endpoint"}`, enforced at `scripts/physics_capture_parsing.py:259`). If you ever find
yourself wanting to look at an image, you want a numeric check instead — see §7.

### Shell traps that have already cost real time

- Aliases shadow standard tools: `ps`→`procs`, `df`→`duf`, `ls`→`lsd`. **Use absolute `/bin/` or
  `/usr/bin/` paths for any command whose output becomes evidence.**
- The shell is `zsh`. Unquoted `--include=*.py` and `$B:path` git-tree refs fail with
  `bad substitution` / `no matches found`. Quote them.
- `pgrep -f 'Unity|9001-player'` matches **your own shell**, because the worktree path contains
  `physics-unity-2019.4`. **Never infer process identity from a name or a command substring.** Resolve
  `/proc/<pid>/exe` and bind through the listening socket inode. This is a standing rule, not advice.
- Set `PYTHONDONTWRITEBYTECODE=1` on every `python` invocation, or delete `scripts/__pycache__/` before
  packaging — `scripts/package_physics_player.py:105` aborts on untracked product source.

---

## 1. Read these, in this order

**Tier 1 — you cannot act correctly without these (~30 min):**

1. `.claude/project-docs/evidence/runtime-repin-gate-20260810/runtime-gate-result.md` — **lines 1–173
   only** (the third-session section, marked "current authority"). Everything below line 174 is
   superseded history; skip it unless you hit a contradiction.
2. `.claude/project-docs/evidence/runtime-repin-gate-20260810/finding-sidecar-array-order-violates-contract.json`
   — the blocker you are being asked to fix. Read all of it, including
   `candidate_fixes_for_the_next_wave` and `required_regression_tests_for_that_fix`.
3. `.claude/project-docs/evidence/runtime-repin-gate-20260810/handoff-next-session.md` — **Part A2
   only**. The banner at the top says so. A2.6 lists five traps that were each paid for once already.
4. `.claude/project-docs/evidence/runtime-repin-gate-20260810/session-3-plan.md` — the full method that
   produced the current state: phase ordering, RED/GREEN proof discipline, review remediation, and the
   error log. This is the template for how the next wave should be run.

**Tier 2 — read when you reach the milestone that needs them:**

5. `docs/high_level_plans/bg_ns_jepa_research_execution.md` — the research roadmap. **§2 (lines 40–75)
   is stale**; see §4 of this brief before you trust it.
6. `docs/data_contracts/physics_capture_v1.schema.json` — the frozen contract. Treat as immutable.
7. `scripts/physics_capture_parsing.py` — the consumer that defines what "valid" means. Lines 259
   (RGB source), 281 (`raw_contacts` order), 283 (`support_edges` order), 260 (`support_id` format).
8. `.claude/project-docs/evidence/runtime-repin-gate-20260810/finding-unrecorded-collision-callbacks-out-of-scope.json`
   — the ABEgg defect and the smoke-harness gap deliberately left open.

**Tier 3 — for Milestone work only:** `world_model/` (`data/`, `model/`, `training/`) and the 12
`tests/test_world_model_*.py` suites.

**Do not read** the pre-third-session sections of `runtime-gate-result.md`, `task_plan.md`, or
`notes.md` for current state. They are preserved receipts, not instructions, and they describe a world
that has since changed.

---

## 2. Standing constraints (carried forward, still in force)

These came from the user and survive the agent migration. Do not relax them on your own judgment.

**Authorization boundaries:**

- **Publication of the physics player is NOT authorized.** Stop and report before publishing.
- **Cohort collection is NOT authorized.** Do not collect the Milestone 0 enriched cohort.
- The **re-pin authorization is conditional**: overwrite `sciencebirdsgames/physics-v1/` only *after* a
  full smoke accepts against the rebuilt candidate. A re-pin before a passing smoke is not covered.
  No smoke has ever accepted. The pin is therefore unchanged and must stay unchanged until one does.

**Do not:**

- Weaken the frozen schema, the taxonomy, or the Python consumer to make a payload fit. If a payload
  does not satisfy the contract, the **producer** is wrong.
- Clean, reset, move, or incorporate: `.claude/logs/`, the knowledge-compression files, the F1–F7
  finding artifacts, or the staged production files.
- Touch the main checkout's Unity project or the protected rollout data.
- Reorder work around a blocker. If something is genuinely unreachable, finish everything that does not
  depend on it and state exactly what you left out and why.

**Method:**

- No-fallback, fast-fail. Do not paper over a failure to keep a pipeline moving.
- NUnit XML is the authority for test results. The Unity editor's exit code is not — `-6` from
  `CefBrowserMessageLoop` at shutdown is normal and means nothing. A compiler or editor failure
  *before* test discovery is `still_blocked`, not a pass.
- Run EditMode **per test class** via
  `.claude/project-docs/evidence/runtime-repin-gate-20260810/editmode_full_suite.py`. A single
  unfiltered run crashes before flushing XML.
- Prove every fix RED then GREEN, and record the NUnit XML sha256 at both stages.

---

## 3. Where the program actually stands

**Blocked at the gate.** Phase 5 (build determinism) passed: two isolated builds produced byte-identical
archives, `deterministic: true`, `drift: []`, 151 provenance files compared. Phase 6 (the single bounded
live smoke) was **deliberately not spent**, because F1 makes `validate-artifact` a known-failing
invariant and `require_collision` passes *first* — the one non-retryable run would have been fully
consumed before the failure surfaced, buying nothing the zero-cost probe already established.

**Milestone 0** — label-derivation code exists (`scripts/derive_physics_labels.py`,
`scripts/physics_label_derivation.py`) and the cohort infrastructure exists
(`world_model/data/{supervision,curriculum,sampling,catalog*,physics_health}.py`). What is missing is
**the data**. `world_model/data/catalog.py:231` still rejects `physics_capture_v1` episodes as
unsupported, while `world_model/data/dataset.py:209` *requires* that contract for supervision. Neither
side can be closed until a re-pinned player emits accepted shots.

**Milestone 1** — substantially built on this branch, and this is the fact most likely to surprise you:

| Sub-item | Status | Where |
|---|---|---|
| 1a JEPA backbone | done | `world_model/model/{encoder,ema,jepa}.py` |
| 1b dual-output predictor | done | `world_model/model/{predictor,heads}.py` |
| **1c SPSG / GINE relational** | **not started** | `world_model/model/encoder.py:8` says `EncoderOutput.tokens` is "reserved for the Milestone 1c SPSG" |
| **1d macro-event predictor $G_\omega$ + $A,R$** | **not started** | no module; `MacroReadoutHead` is 1b's readout, not a transition predictor |
| 1e teacher-forced (Δ,α) grid | done | `world_model/training/{loop,grid_data,grid_run,pair_grid}.py`, `scripts/run_jepa_pair_grid.py` |
| 1f best-pair labels + oracle ceiling | done | `world_model/training/{scoring*,frontier}.py`, `scripts/plot_jepa_pair_frontier.py` |

M1's real-data runs so far used the **legacy** cohort (`world_model/data/inspect.py:174` defaults
`legacy_rgb_v1`), which carries no supervision payload. The oracle-symbol ceiling and mode-head metrics
in M1's exit criteria are therefore **not yet obtainable**. M1 is provisional, not closed.

**Milestones 2–5** — untouched.

---

## 4. The roadmap is stale — correct it before you plan against it

`docs/high_level_plans/bg_ns_jepa_research_execution.md:42-44` states: *"no world-model, controller,
extractor, or training code exists yet — which is exactly the intended handoff state for this plan."*

That is false as of commit `aa31b31` (*JEPA backbone, dual-output predictor, teacher-forced loop*),
which is contained **only** in `physics-unity-2019.4` and `origin/physics-unity-2019.4` — not in `env`,
not in `main`. On `env`, `world_model/` holds only `__init__.py` and `requirements.txt`.

Two consequences you must handle:

1. The plan's §2 understates progress by roughly all of M1a/b/e/f. **Update §2 and the M1 checklist to
   match §3 of this brief** before using the roadmap for sequencing.
2. The M1 work is **stranded on the Unity migration branch**. Whether to merge or cherry-pick it toward
   `env` is a user decision, not yours — but record the divergence explicitly so it does not grow.

---

## 5. Your work, in dependency order

### T1 — Fix F1/F2 and clear the gate  ⚠️ REQUIRES NEW USER AUTHORIZATION BEFORE YOU EDIT

**Do not start T1 until the user grants authorization for the surface below.** The previous wave stopped
precisely here, on scope grounds, not on merit.

**The defect.** The emitter writes `raw_contacts` cumulative and step-major (sorted only *within* each
fixed step, then concatenated), while `scripts/physics_capture_parsing.py:281` requires the array
globally sorted by a key that **excludes** `fixed_step`. `rawContacts` is never cleared;
`PhysicalSnapshotRuntime.FixedUpdate` samples every collider every step; so on the smoke's level (bird +
block tower) every shot emits a violating array from the first step. `support_edges` (`:283`) has the
same defect class — appended in contact-pair order while the contract sorts by `supporter_id`, which is
whichever body is lower in y.

**The fix** (preferred option from the finding):

- Site: `tasks/task_template_designer/Assets/Scripts/GroundTruth/PhysicsShotRecorder.cs`,
  `CreateFinalizedSnapshot` at **lines 639–643**.
- Shape: before constructing `PhysicalShotRecorderSnapshot`, sort the cumulative `rawContacts` using
  `CompareContacts` (**lines 751–763**) *extended with a final `ContactId` tiebreak* — it currently ends
  at `Point.y` and the parser key does not — and sort `supportEdges` by `(SupporterEntityId,
  SupportedEntityId, SupportId)`.
- Cost: one sort per shot. Per-step ordering and `PointIndex` assignment stay untouched, so
  `contact_id` values do not change.
- **This is not weakening the contract.** It makes the producer conform to a contract that is already
  frozen. Do not touch `physics_capture_parsing.py` to accept the bad order.

**Regression tests required** (all three, RED before GREEN):

1. EditMode fixture: contacts for two pairs across three fixed steps → assert the finalized snapshot's
   `RawContacts` is globally sorted by the parser's key.
2. EditMode fixture with the F2 geometry (a pair whose *upper* member is entity A, sampled before a
   higher-numbered pair whose *lower* member is entity A) → assert `SupportEdges` is sorted by supporter.
3. Python: feed an emitter-shaped array through `_parse_state` and assert it parses.
   `.claude/project-docs/evidence/runtime-repin-gate-20260810/probe_raw_contact_order.py` is already
   that test in executable form — promote it into `tests/`.

**Bundle into the same wave** (all cheap once this file is open, none blocking on its own):

- **F3** — collision evidence looked up by entity pair, ignoring collider identity (minor on
  single-collider prefabs).
- **F4** — enforce "never throw in a physics callback" with an actual `try/catch` in
  `RecordCollisionCallback`; today it is argued, not enforced.
- **F5** — `PhysicsShotRecorder.cs` is **814 lines**, over the project's 800-line rule. Split it.
- **F6** — `currentStep` / `currentTime` are write-only.
- **F7** — the contact stream is unbounded in shot duration; ~180k contacts at the 120 s ceiling trips
  `RecordLimitExceeded`.
- **ABEgg** — `ABEgg.cs:10-13` carries the same unrecorded-collision defect as the two callbacks fixed
  in session 3. Unreachable on the smoke level, but **must** be fixed before any cohort using a white
  bird. See `finding-unrecorded-collision-callbacks-out-of-scope.json`.

**Then, in this order — do not skip or reorder:**

1. Full per-class EditMode suite green (`editmode_full_suite.py`; baseline was 53/53).
2. `mutation_check.py` 8/8 red, source restored byte-identical.
3. Commit. `git_revision` refuses to package tracked drift from HEAD.
4. **Rebuild and re-prove determinism** via `phase5_build_twice.py`. The previous archive
   `2bdd498a928204f5923ef84770b361b6ba31dfa5681867028870237cf048847e` **cannot be reused** — the fix
   changes `Assembly-CSharp.dll`.
5. Spend **exactly one** bounded full live smoke (`scripts/smoke_physics_capture.py`).
6. If and only if it accepts: the conditional re-pin fires. Overwrite
   `sciencebirdsgames/physics-v1/` and record the new `archive.sha256`.
7. **Stop and report before publishing.** Publication is a separate authorization.

Close the wave with an explicit verdict — `ready_for_repin_approval` or `still_blocked` — in the same
schema as `runtime-gate-verdict.json` (`novphy_runtime_repin_gate_verdict_v1`).

### T2 — Milestone 0 (blocked on T1)

Needs a passing smoke and a re-pinned player before any cohort exists. Also needs **separate
authorization** — cohort collection is currently forbidden. Once unblocked:

- Teach `world_model/data/catalog.py:231` to accept `physics_capture_v1` instead of rejecting it, so the
  catalog and `dataset.py:209` stop contradicting each other.
- Collect the enriched cohort; derive macro/outcome labels (0a) and the oracle gate φ* (0b).
- Produce the machine-readable dataset-health report and the frame-exact alignment checks named in the
  roadmap's M0 exit evidence.

### T3 — Milestone 1 completion (blocked on T2)

- Build **1c** (SPSG/GINE scene-graph encoder, predicate projection regularizers, physics-validated
  negative sampling) — `EncoderOutput.tokens` is already reserved for it.
- Build **1d** (macro-event predictor $G_\omega$ and the learned restriction/lifting maps $A, R$).
- Re-run 1e/1f against the *enriched* cohort to obtain the oracle-symbol ceiling and mode-head metrics
  that the legacy cohort cannot provide.

### T4 — Milestones 2–5

Gated on M1 exit evidence. Do not start. Note for whoever does: M2a's visual predicate parser is
supervised by engine ground truth $G^*$, so its metric is F1/mAP against the engine — not visual
inspection.

---

## 6. What "done" looks like for a report

Match the existing evidence discipline. Every wave ends with:

- a machine-readable verdict JSON with an explicit `what_did_not_happen` block naming, in plain words,
  every authorized-but-unused action (no re-pin, no publication, no cohort, no retry);
- a `finding-*.json` for every defect found and *not* fixed, with the reason it was left, the candidate
  fixes, and the regression tests those fixes would need;
- NUnit XML sha256 at RED and at GREEN for every behavioural claim;
- absolute paths and exact digests, never "verified" without the value that was compared.

If you stop, say exactly what blocked you and what the next unblock action is. A partial wave with an
honest boundary is worth more than a complete-looking one that guessed.

---

## 7. If you think you need to see an image

You don't. The two real cases and their text-only answers:

- **"Are the captured frames actually rendering, or is the headless player emitting black PNGs?"** The
  frozen contract validates frame *structure*, never *content* — this gap is real. Close it with a
  numeric frame-health check (per-frame mean, std, unique-colour count) in
  `world_model/data/physics_health.py`. That is a better check than an agent squinting at a thumbnail,
  because it runs on every frame of every episode.
- **"Has the JEPA latent collapsed?"** There is no pixel decoder, so there is nothing to look at.
  Collapse is numeric. Use **centred** rank — uncentred rank reports 1.03 where centred reports 5.35 on
  the same tensors, and compression is not collapse.
