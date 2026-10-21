## #125 implementer decisions (pre-trigger; posted before any #123 or #125 evaluation read)

Runner `scripts/run_hjepa_idm.py`; IDM module `world_model/hjepa/idm.py`; official-side G-port reference `scripts/hjepa_idm_port_reference.py` (hjepa env); tests `tests/test_issue_125_hjepa_idm.py`; scheduler jobs `scripts/gpu_window_issue_125.json`; doc `docs/issue-125-hjepa-idm.md`; artefacts `.local-artifacts/issue-125-hjepa-idm-v1/`. Nothing of #123 is edited: the runner imports `scripts/run_hjepa_comparison.py`, `world_model/hjepa/novphy.py` and the vendored code, and reads #123's data stores, checkpoints, frozen plan and (at publication) its validated records.

### Port of the remedy (only the IDM terms change)
- **Source of truth.** The vendored official code already contains the remedy: `loss.InverseDynamicsModel` / `loss.InverseDynamicsLoss` (loss.py:50-79) and the two terms of `hjepa_utils.hjepa_forward` (lines 153-165), driven by the per-level config keys `idm_coeff`, `idm.hidden_dim` and `action_sigreg_coeff`. #125 sets those keys on #123's arm configs and uses `hjepa_forward` unchanged.
- **Heads attached after construction.** `create_world_model` (lines 326-333) reads the IDM output width from `predictor.action_dim`, which only DROID's `causal_transformer` accepts; #123's arms use `ARPredictor`, which rejects that key. The heads are therefore built with the official classes and arguments (`state_dim = D`, `hidden_dim = 256`, `action_dim` = the action-embedding width = D) and attached as `jepa.idm_loss_fn`, in level order, after #123's `build_model`; the action-SIGReg modules (`SIGReg()`, `sigreg_action_level{l}`) are added to the loss modules as `create_world_model` (lines 353-354) adds them. With the same seed the shared parameters start bit-identically where the #123 arm starts (tested), so SI1/SI2 compare the same initialisation. The heads train in their level's AdamW group (official optimizer factory semantics).
- **Targets.** Exactly the official target, `action_l` (detached by the official loss): level 1 = the level-1 action embedding, i.e. the embedded launch on the launch transition and the embedded no-op elsewhere (#123's launch encoding through #123's `Embedder`); level ≥ 2 = the pooled macro-action embedding. Tested: the IDM loss equals mean ‖I(z_t, z_{t+1}) − a_t‖² per level, the target carries no gradient, and the level loss is embed loss + γ_l·L_idm (+ 0.005·action-SIGReg at l ≥ 2).
- **Weights.** γ₂ = γ₃ = 50, action-SIGReg 0.005 at levels 2 and 3 (DROID); γ₁ ∈ {100, 225} selected per input type. SIGReg λ = 0.18 (#123's frozen value for both input types).
- **LeWM arms.** One level: IDM at level 1 with the selected γ₁, no action-SIGReg (DROID LeWM+IDM is the base config with level-1 IDM only). LeWM-px-IDM / HJ-px-IDM backbones are #123's paper sizes.
- **Training.** #123's loader (`BatchSource`, `PCG64([seed, b])`), optimizer, schedule, clip, bf16 autocast and its A1 stop: SIGTERM/SIGUSR1 → `resume.pt` after the current optimizer step (atomic), exit 75, RNG restored after the loader iterator; nonfinite loss = typed failure, never retried. `model.pt` uses #123's model schema (backbone under `state_dict`, so #123's `load_trained`, `fit_readouts` and `score_arm_seed` run unchanged) and stores the training-only IDM heads under `idm_heads`. CPU test: N steps uninterrupted vs k steps + SIGTERM + resume + N − k steps → model (heads included), optimizer, scheduler, CPU RNG and the final `model.pt` bitwise identical.
- **Registration.** The four IDM arms are added to `novphy.ARMS` / `novphy.SIZES` in memory (novphy.py unchanged on disk) so #123's geometry, loader, readouts and scoring handle them by name.

### Matched-parameter rule
- HJ-car-m-IDM and LeWM-car-m-IDM count the IDM heads (U = 1 837 690, ±2 %). One head at D = 112 is 152 176 parameters; HJ has three (456 528, 25 % of the budget), LeWM one.
- Rule: #123's matched sizes with **only the predictor MLP width reduced, uniformly over levels**, to the 16-step value whose total is nearest U; D (112), encoders, upper encoders, heads, depth and attention unchanged; IDM heads at the official width 256.
- Result: **HJ-car-m-IDM** predictor MLP 448 → 96: backbone 1 379 972 + heads 456 528 = **1 836 500 (−0.06 %)**. **LeWM-car-m-IDM** MLP 480 → 384 (depth 7): 1 683 020 + 152 176 = **1 835 196 (−0.14 %)**. Tested that ±16 is farther from U. Disclosed confound for SI2/SI4: the carrier HJ predictor FF width is much smaller than in HJ-car-m.

### G-port-IDM (CPU, run 2026-10-19)
- Official side: `scripts/hjepa_idm_port_reference.py` in the hjepa env (official clone @ 97054ec, torch 2.7.1, CPU): official `create_world_model`, official IDM / SIGReg classes, official `hjepa_forward`, `main_hjepa._build_hjepa_optimizer_factory`, spt scheduler. Port: vendored modules + `idm.py` + #123's optimizer / scheduler (torch 2.13). #123's protocol: fixed inputs (batch 4, launches on clips 0 and 2 at frame 0, level-1 crop start 0 so the launch transition is an IDM target; AdaLN-zero modulation seeded N(0, 0.02) on both sides), the step at the step-0 LR, then a positive-LR step after warm-up (#123's `gport_step2`: loss, gradients, AdamW moments, updated parameters, movement).
- Tolerances: #123's declared `GPORT_TOLERANCE` unchanged. Additionally required: every expected IDM / action-SIGReg term present, head gradients compared, every head tensor in its level's optimizer group.
- **Result: PASS for all four arms** (`gport-idm.json`). Worst relative loss delta 2.9e-6 (HJ-px-IDM; tolerance 1e-4), worst per-tensor gradient rel-L2 4.4e-5 (tolerance 1e-3), IDM-head gradients ≤ 4.4e-6; positive-LR steps pass with both AdamW moments, parameter close share ≥ 0.9997, every group moving. Every IDM / action-SIGReg term within 1.2e-7 absolute of the official value (one float32 ulp at 1.7; e.g. HJ-car-m-IDM idm_loss_level1 0.0051941 vs 0.0051941, action-SIGReg level 2 1.68256 vs 1.68256). A test shows the check fails on a wrong γ₁ or a wrong action-SIGReg weight.

### Trigger computation (frozen in the ticket; evaluated by `--trigger`)
- Inputs: the trained #123 `models/<arm>/seed-<s>/model.pt` of HJ-px and HJ-car-m, seeds 20260908/09/10. `resume.pt` is never read; a missing `model.pt` raises (training not complete).
- Computation: #123's `fit_readouts` steps "fit probe frames" and "fit anchors" verbatim through #123's functions, under `deterministic_scoring()`: per-level effective rank and within-episode share χ on every 15th position of every fit shot (≈ 278 k frames), endpoint launch sensitivity on the 480 fit anchors × 16 grid launches. No R1/R2/R4 fit, no policy split, no evaluation file. Nothing is written into #123's directories (so #123's readouts are not pre-empted).
- Guards: #123's per-seed expressions and thresholds (`rc.arm_guards`, `rc.GUARD_THRESHOLDS`): G-collapse = top-level erank ≥ 0.1·D and χ ≥ 0.05; G-action = sensitivity ≥ 0.05. Tested equal to `rc.arm_guards` seed by seed at and around every threshold. A typed training failure fails both guards, as in `rc.arm_guards`.
- `triggered` iff any seed of either arm fails either guard. `trigger.json` records per-arm, per-seed values, the failing list, the part hashes, and #123's evaluation-read state at start and end (records scored, evaluation anchors read, position cache built).
- CPU dry run on #123's λ = 0.18 HJ-car-m selection model: erank 14.2233 / 112, χ 0.016724 (equal to its #123 readouts.json to 1e-8), sensitivity 1.6e-7 → fails both guards.

### γ₁ selection (frozen now)
- HJ-px-IDM and HJ-car-m-IDM at B1 (9000 × 64), seed 20260908, γ₁ ∈ {100, 225}; #123's readouts; policy-split pooled grid AUC under R1; ties at 4 d.p. → 100 (DROID). The selected γ₁ applies to every IDM arm of that input type (LeWM included). 4 runs in 3 slots (GPU 1: px γ 100 + car γ 225; GPU 2: px γ 225; GPU 3: car γ 100).
- `--selection` and `--smoke` exit 0 at once with a recorded `not_triggered` when `trigger.json` says false; `--prepare` refuses.

### Guards and tokens (from the ticket; consequences stated now)
- Every #123 guard per IDM arm via #123's `arm_guards`; global G-fid (#123's gfid.json), G-port (#123), G-port-IDM, G-isolation, G-chronology (trigger ≤ plan freeze ≤ readout freeze ≤ evaluation anchors read ≤ first record), G-scoring-policy, G-records.
- SI1/SI2 read #123's published records (never re-scored) and are gated by #123's published guard of HJ-px / HJ-car-m and #123's G-records. Because the trigger fires only if one of those arms fails G-collapse or G-action, **the SI token of a failing #123 arm will be `readiness_or_precision_insufficient` by the frozen rule**; the estimate is still reported as a number.
- Disposition 4 is evaluated alongside 1–3: true iff a primary IDM arm still fails G-collapse or G-action.

### Schedule (night window only)
- Enqueued 2026-10-19 17:15 AEDT (daytime; nothing launched): `125-trigger` (GPU 1, `shares_gpu`, after all four `123-train-q*`; it runs beside `123-readouts-g1` and is projected to finish before `123-score` starts), then `125-select-q0/q1/q2` (GPUs 1/2/3, after the trigger; they wait for #123's readouts on their GPUs by queue order) and `125-smoke` (GPU 1, after the three selection slots). No #125 job uses GPU 0, where #123 scores, publishes and validates. #123's jobs keep priority (earlier in the queue).
- Not enqueued: training, readouts, scoring, publication. They follow the #125 freeze (`--prepare`), which an agent commits after the smoke.

### Budget
- Cap 120 GPU-h of active compute (ledger in `ledger/`). Expected: trigger ≈ 1 GPU-h, selection ≈ 2, smoke < 1, HJ-px-IDM ≈ 3 × 16, LeWM-px-IDM ≈ 3 × 7, carriers < 1, readouts ≈ 4, scoring ≈ 3. The smoke re-projects from measured throughput and stops at > 120 GPU-h or > 48 h wall on 4 GPUs.

## Freeze (plan.json frozen 2026-10-21T07:26:34Z = 18:26:34 AEDT)

### Chronology disclosure
- `plan.json` was frozen **after #123's evaluation scoring** (#123 scored 18:03–18:10 AEDT and published at 18:15 on 2026-10-21; its `--validate` was running at the freeze). It was frozen **before any #123 result was read by the #125 implementer**: no #123 record, `summary.json`, `compute.json`, `findings.md`, `validation.json` or publication table was opened, and the #123 issue comments were not refreshed between #123's scoring and the push of this freeze commit.
- Every frozen #125 choice comes from three sources only:
  1. the ticket, pre-registered 2026-10-19 before any #123 evaluation read (arms, γ₂/γ₃ = 50, action-SIGReg 0.005, the γ₁ grid {100, 225} and its rule, contrasts, guards, dispositions, cap);
  2. the policy-split γ₁ selection;
  3. the fit-split trigger and the smoke (fit / policy data and timing only).
- `prepare` reads only #125's own `trigger.json`, `smoke.json` and `selection/*/readouts.json`, plus #123's pre-evaluation `plan.json` (the frozen λ and data manifest). It hashes #123's runner, module and plan.
- **Trigger.** It was evaluated 2026-10-20T18:59:24Z (05:59 AEDT on 21 Oct), before #123's first evaluation read. Its recorded #123 state, both at start and at end: 0 records, no evaluation anchors read, no position cache.
- **Smoke.** It wrote `smoke.json` at 2026-10-21T07:10:51Z (18:10 AEDT), while #123 was scoring. It read only fit / policy data, the #125 selection readouts and the throughput of scratch runs, and no #123 output.

### Trigger (fit split): triggered
All six #123 primary arm-seeds fail both G-collapse and G-action:

| arm | seeds | erank / D | χ | sensitivity |
| --- | --- | --- | --- | --- |
| HJ-car-m | 20260908 / 09 / 10 | 14.22 / 13.81 / 14.07 of 112 (threshold 11.2) | 0.0167 / 0.0167 / 0.0161 | 1.7e-7 |
| HJ-px | 20260908 / 09 / 10 | 111.7 / 112.7 / 113.0 of 192 | 0.0014 / 0.0015 / 0.0015 | 1.1e-6 |

- HJ-car-m passes the effective-rank half of G-collapse but fails χ < 0.05.
- HJ-px passes the effective-rank half (threshold 19.2) and fails χ.

### γ₁ selection (policy split, B1, seed 20260908, R1, 94 mixed grid cells; ties at 4 d.p. → 100)

| arm | γ₁ = 100 | γ₁ = 225 | selected |
| --- | --- | --- | --- |
| HJ-px-IDM | 0.4830 | **0.5170** | **225** (pixels) |
| HJ-car-m-IDM | 0.4275 | **0.4406** | **225** (carriers) |

- There is no tie. γ₁ = 225 applies to every IDM arm of both input types (HJ and LeWM).
- The selected AUCs are near or below chance. This is the pre-declared rule, not a quality claim.
- **Selection-run diagnostics** (fit / policy only, for information; nothing changed in response):
  - top-level erank: HJ-px-IDM 42 / 192, HJ-car-m-IDM 14 / 112;
  - χ: 0.0024 and 0.016;
  - endpoint launch sensitivity: 1e-4 to 2e-4 (HJ-px-IDM) and 2e-5 to 5e-5 (HJ-car-m-IDM), against #123's 2–5e-7 at B1;
  - the policy readout ceiling: 0.81–0.83 (px) and 0.57–0.60 (car).
- The ticket's guard thresholds stay unchanged.

### Plan
- **Arms** (λ 0.18, γ₁ 225; γ₂ = γ₃ = 50 and action-SIGReg 0.005 for the HJ arms):

  | arm | steps × batch | parameters |
  | --- | --- | --- |
  | HJ-px-IDM | 128 045 × 128 | 35 860 884 |
  | HJ-car-m-IDM | 9000 × 64 | 1 836 500 |
  | LeWM-px-IDM | 150 880 × 128 | 11 563 900 |
  | LeWM-car-m-IDM | 9000 × 64 | 1 835 196 |

- **Training queues.** LPT over the smoke throughput:

  | queue | contents | GPU-h |
  | --- | --- | --- |
  | q0 / q1 / q2 | HJ-px-IDM, HJ-car-m-IDM, LeWM-car-m-IDM of seed 20260908 / 09 / 10 | 13.1 each |
  | q3 | LeWM-px-IDM × 3 | 18.1 |

- **Projection.** 62.1 GPU-h in total against the 120 GPU-h cap: 57.4 training, 0.5 readouts, 2.8 scoring, 1.4 already spent. Wall time is 18.9 h, which is 2 nights.

### Main jobs (`scripts/gpu_window_issue_125.json`, enqueued after the freeze push)
- `125-train-q3` → GPU 1, `125-train-q1` → GPU 2, `125-train-q2` → GPU 3, `125-train-q0` → GPU 0 (once `123-validate` frees it). All are resumable: SIGTERM writes `resume.pt` at the next optimizer step and the job exits 75.
- `125-readouts-g0..g3` run after all four training queues. `125-freeze-readouts` runs after them.
- `125-score` runs after `125-freeze-readouts` **and** `123-validate`, because SI1 / SI2 read #123's validated publication. `125-publish` and `125-validate` follow on GPU 1.
- **Post-smoke edit to a frozen input.** `tests/test_issue_125_hjepa_idm.py` changed after the smoke: only its scheduler-job test, which now covers the main jobs. 18 / 18 tests pass on CPU at the freeze, and the frozen hash is of that file.

## Post-freeze amendment A1 (2026-10-21, night 1 of main training; compute ledger and plan loading only)
This amendment was made before any #125 readout or evaluation read, and after #123's publication agent reported the same bug in #123's `train()`. It is registered in `scripts/post_freeze_amendments.py`: `scripts/run_hjepa_idm.py` sha256 5e533591… → 56e7cc65….

- **Ledger double count.** A relaunched training queue re-visits arm-seeds that are already finished. `train_run` then returns their stored `report.json`, and the frozen `train()` added those seconds to `ledger/train-<arm>-seed<s>.json` a second time. With A1, `train()` marks an arm-seed only if its `model.pt` did not exist before the call.
  - The four queues running tonight loaded the frozen code. They mark only arm-seeds they train to completion, so night 1 cannot double count.
  - The first relaunch (night 2) runs the amended code, so no ledger correction is needed.
  - Test: `tests/test_issue_125_amendments.py` (night 1 / pause / night 2 / extra relaunch adds each arm-seed's seconds exactly once).
- **Registry pin.** The frozen plan pinned `scripts/post_freeze_amendments.py` by hash. That is the shared, append-only registry `matches()` reads, so any later registration by any ticket, including this one, would have made `load_plan` refuse at #125's relaunch, readouts and scoring. With A1, `load_plan` skips the hash check of that one file. Its frozen hash stays in `plan.json` as a record. Every other pinned input is still checked.
- Unchanged: objective, training, resume, RNG, data, readouts, scoring, tokens and dispositions. Registry test: `test_issue_125_ledger_amendment_is_registered` in `tests/test_issue_101_decision_aware_policy.py`.

## Inherited compute aggregation: #123 A3 (2026-10-21, `aa8e702f`; no #125 runner change)
`compute_tables` sums `rc.ledger_totals`, i.e. code in `scripts/run_hjepa_comparison.py`, a pinned input of the #125 plan (pinned at the #123 A2 bytes `045530a9…`). #123's post-freeze amendment A3 (`045530a9…` → `86e8ee9e…`, registered in `scripts/post_freeze_amendments.py`; `load_plan` accepts it because the frozen hash is in its `replaces`) changes two things there:
- `ledger_totals` counts each arm-seed's main training once: a `train-<arm>-seed<s>` phase whose `models/<arm>/seed-<s>/report.json` exists counts that report's seconds, the elapsed time `train_run` carries across resumes, which is exactly what one completion marks. A ledger entry that differs keeps its raw seconds as `ledger_gpu_seconds` / `ledger_wall_seconds` and is not counted. Every other phase counts its ledger seconds.
- #123's own `train()` gets the same fix as #125 A1.

#125's run layout is #123's (`run_dir` delegates to `rc.run_dir`), so the rule applies to #125's training phases unchanged. With #125 A1 the ledger should never differ from the reports; the rule is the second line of defence. Test: `test_aggregation_counts_each_arm_seed_once[issue-125]` in `tests/test_issue_123_compute_ledger.py`; `test_issue_123_a3_is_accepted_by_both_plans` in `tests/test_issue_101_decision_aware_policy.py` (both plans load). #125's `run_hjepa_idm.py` is unchanged (still the A1 bytes `56e7cc65…`). Objective, training, resume, RNG, data, readouts, scoring, tokens and dispositions unchanged.

## Post-freeze amendment A2 (2026-10-22, between main-training nights 1 and 2; validation only)
This amendment comes from the pre-scoring review (spec and standards notes, 2026-10-22). It was made before any #125 readout or evaluation read, and no current drift was alleged or found. It is registered in `scripts/post_freeze_amendments.py`: `scripts/run_hjepa_idm.py` 56e7cc65… → d53e44d0… (`replaces` keeps the frozen 5e533591… and A1 56e7cc65…).

- **Inherited pins (P2).** The frozen `load_plan` hashed #123's `plan.json` but did not run #123's `load_plan`. The inputs that plan pins were therefore unchecked, although #125 consumes them directly: the scoring harness, vendored H-JEPA code, the retrain fit anchors and verdicts, the Gate-B and #101 reference records, and the upstream runners. With A2, `load_plan` runs `rc.load_plan(SOURCE)` after #125's own pins. That accepts #123's registered amendments (A1–A3 of its runner, A1 of the Tier-T launcher and fidelity runner, A2 of its port reference and tests) and nothing else. #123's plan does not pin the shared registry, so the A1 exemption stays the only one.
  - Live-tree check, CPU, 2026-10-22 07:21 AEDT: through `main()` with the exact argv of all 12 jobs tonight (`--train --gpu 0..3`, `--readouts --gpu 0..3`, `--freeze-readouts`, `--score`, `--publish`, `--validate`), the new `load_plan` passed, #123's `load_plan` included, about 6 s each. No mode ran past its plan check.
  - Test: `test_load_plan_checks_inherited_123_inputs` simulates drift in `world_model/planning/scoring_harness.py`, which only #123's plan pins, and `load_plan` refuses. With the amended #123 runner in use, the real plan still loads.
- **Frozen guard reports (P2).** `--freeze-readouts` records `report_sha256` for every arm-seed's `readouts.json`, but nothing checked it. `rc.arm_guards` derives G-collapse, G-action and G-readout from the live reports, and `rc.load_manifest` checks only the `model.pt` / `readouts.pt` hashes. With A2, a #125 `load_manifest` wraps #123's and also verifies every report hash, typed failures included. It replaces the three `rc.load_manifest` calls: `--score`, `compute_tables` (`--publish`, `--validate`) and the spot check.
  - A report-only change fails closed. Scoring and publication refuse. `--validate` withdraws the published tables, writes a failed receipt and exits non-zero. It cannot re-derive stop-token tables, because those need the frozen reports as well.
  - Test: `test_report_only_drift_fails_closed`. A changed scored report and a changed typed-failure report are each caught while #123's check still passes, and score, `compute_tables`, publish, spot check and validate all refuse, leaving no publication.
- **Smoke re-run (P3).** `mark()` accumulates, so a smoke re-run would add its seconds again. A post-freeze re-run would also overwrite the pinned `smoke.json`. With A2, `--smoke` refuses once `plan.json` exists. Test: `test_smoke_refuses_after_the_freeze`. A pre-freeze re-run would add the seconds it really spent, which is correct for active compute. The smoke ran once: `ledger/smoke.json` holds 45.209 s, equal to `smoke.json`'s `wall_seconds`.
- Unchanged: objective, training, resume, RNG, data, readout fitting, scoring math, tokens and dispositions. The training resume path is unchanged except for the added plan check.
- Registry test: `test_issue_125_ledger_amendment_is_registered` in `tests/test_issue_101_decision_aware_policy.py`. It checks the chain frozen → A1 → A2, confirms #123's plan pins no registry, and loads the plan.
