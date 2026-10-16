# Issue #102: closed-loop evaluation with live engine execution

Parent: #108. Gate B passed. The binding #85 rules apply, and the #111 deterministic, tie-free scoring contract is adopted. Earlier records and dispositions (#101, #103, #104, #108 Gate B, #114) are read-only inputs. The ICLR submission is not edited.

| Deliverable | Location |
| --- | --- |
| Runner | `scripts/run_closed_loop_evaluation.py` |
| Tests | `tests/test_issue_102_closed_loop.py` (8 tests: ranking, exclusion, random arm, oracle, recorded replay, pass@k) |
| Artifacts | `.local-artifacts/issue-102-closed-loop-v1/`. Committed: `plan.json`, `smoke.json`, `findings.md`, `closed-loop-table.csv`, `compute.json`, `summary.json`, `manifest.json` (sha256 and size of every raw record), `ledger/`, `smoke/ledger/`, `logs/pids.txt`. Kept local: `records/` (decision and live-verdict records, post-hoc record, N2 anchors) and `engine/` (5,123 live attempts, 403 GB). |
| Rendered episodes | `data/issue-102-closed-loop/` (local, gitignored). It holds `cells/<execution>/{decision.png, final.png, shot.webm}` for every live execution, failed and typed-failure attempts included, plus `manifest.json` (executions and all 19,360 episodes with their shots), `index.html`, `levels/<level>.html` and `render-errors.json`. The smoke gallery is in `smoke/`. |
| Freeze | `06626829`, plan frozen 2026-10-15T00:19:38Z, after a passed rendered smoke |
| Validation | `python -u -m scripts.run_closed_loop_evaluation --validate` exits 0 (§ Validation) |

## Setup (user decision)

Every #109 and #110 level holds a single bird, so a level ends after one shot. The user chose **retry-from-start with live engine execution**:
- **Reset.** Shot k resets the same level to its start state.
- **Decision.** Before each shot the planner ranks the frozen 16-launch grid from the level's decision frame with its model or rule. It removes the candidates already executed and failed in this episode and picks the best remaining one.
- **Execution.** The chosen launch runs **live** in the #104 player (`capture_pipeline_v2.run_campaign`, the #104 limits, 16 isolated workers).
- **Stop.** The episode ends at the first pass (live engine pig removal) or after the pre-declared budget of **3 shots**.

The state never changes between shots, so "replanning" amounts to excluding failed candidates. The model still re-ranks at every shot. Guard G2 confirms that the re-rankings are identical: 45,360 re-rankings checked, 0 differences.

Recorded #104 outcomes are used only in two ways: as the oracle's inventory, and as a declared determinism cross-check. They are never the result.

## Result

### Closed-loop table: N1, the 400 #109 evaluation levels (primary)

Cells are level × seed (1,200). Intervals are level-clustered percentile bootstraps (10,000 draws, PCG64 7201), DESCRIPTIVE. "Decision s" is the host wall time of one decision, which ranks all 16 candidates on one RTX 5090. "MACs / decision" counts linear MACs over the 16 candidate rollouts.

| Arm | pass@1 | pass@3 | Shots to pass (passed) | Decision s | MACs / decision | pass@3 − prior | pass@3 − random |
| --- | --- | --- | --- | --- | --- | --- | --- |
| F-1-continuous | 0.070 [0.045, 0.095] | 0.334 [0.295, 0.375] | 2.40 | 0.148 | 5,157M | +0.012 [−0.028, +0.052] | +0.137 [+0.098, +0.177] |
| F-1-micro | 0.070 [0.045, 0.095] | 0.326 [0.287, 0.366] | 2.38 | 0.189 | 6,464M | +0.003 [−0.046, +0.052] | +0.128 [+0.092, +0.168] |
| F-1-macro | 0.112 [0.089, 0.135] | 0.343 [0.297, 0.390] | 2.01 | 0.165 | 5,273M | +0.020 [−0.015, +0.055] | +0.145 [+0.098, +0.193] |
| F-5-continuous | 0.180 [0.151, 0.210] | 0.453 [0.408, 0.499] | 2.03 | 0.031 | 1,031M | +0.131 [+0.087, +0.176] | +0.256 [+0.212, +0.302] |
| F-5-micro | 0.167 [0.138, 0.197] | 0.438 [0.394, 0.481] | 1.99 | 0.039 | 1,293M | +0.115 [+0.069, +0.162] | +0.240 [+0.200, +0.282] |
| F-5-macro | 0.107 [0.083, 0.132] | 0.413 [0.371, 0.457] | 2.16 | 0.034 | 1,055M | +0.091 [+0.051, +0.132] | +0.216 [+0.173, +0.261] |
| **F-15-continuous** | 0.189 [0.161, 0.218] | **0.478 [0.434, 0.523]** | 2.04 | 0.011 | 344M | +0.156 [+0.110, +0.203] | +0.281 [+0.238, +0.324] |
| F-15-micro | 0.221 [0.192, 0.251] | 0.458 [0.416, 0.502] | 1.83 | 0.014 | 431M | +0.136 [+0.089, +0.184] | +0.261 [+0.219, +0.303] |
| F-15-macro (Gate-B unit) | 0.151 [0.131, 0.172] | 0.424 [0.384, 0.465] | 1.98 | 0.012 | 352M | +0.102 [+0.057, +0.147] | +0.227 [+0.187, +0.268] |
| C* (C-F-15) | 0.148 [0.126, 0.172] | 0.466 [0.427, 0.505] | 1.96 | 0.008 | 445M | +0.143 [+0.099, +0.188] | +0.268 [+0.230, +0.309] |
| C-J λ 1e-4 (horizon-only adaptive) | 0.161 [0.136, 0.187] | 0.346 [0.315, 0.378] | 1.83 | 0.086 | 3,749M | +0.023 [−0.014, +0.062] | +0.148 [+0.116, +0.182] |
| ICLR J λ 1e-4 | 0.123 [0.098, 0.150] | 0.394 [0.352, 0.438] | 2.25 | 0.140 | 2,934M | +0.072 [+0.033, +0.110] | +0.197 [+0.154, +0.241] |
| J_dec λ 1e-1 (#101) | 0.139 [0.117, 0.162] | 0.445 [0.403, 0.487] | 2.07 | 0.017 | 363M | +0.122 [+0.077, +0.168] | +0.247 [+0.207, +0.289] |
| No-model action prior | 0.140 [0.107, 0.175] | 0.323 [0.278, 0.370] | 1.74 | 0.0005 | 0 | — | +0.125 [+0.078, +0.172] |
| Uniform random (3 seeds) | 0.068 [0.053, 0.082] | 0.198 [0.172, 0.223] | 1.93 | ~0 | 0 | −0.125 [−0.172, −0.078] | — |
| Oracle ceiling | 0.775 [0.733, 0.815] | 0.775 [0.733, 0.815] | 1.00 | — | 0 | +0.453 [+0.403, +0.502] | +0.578 [+0.539, +0.616] |

**Every contrast token is `readiness_or_precision_insufficient`**, because guard G4 failed (§ Guards). Under the frozen rule, a failed guard turns every token into that value. The estimates and intervals above are DESCRIPTIVE and unaffected.

Reading (descriptive, no δ-rule claim):
- **The oracle bounds what is achievable.** 89 of the 400 levels have no recorded passing grid launch, so the oracle reaches 0.775. One oracle shot was lost to a typed engine failure, against a recorded ceiling of 0.7775.
- **The best closed-loop arms are coarse-horizon fixed requests.** F-15-continuous, C* and F-15-micro reach 0.46–0.48 pass@3, about 60 % of the ceiling, at 0.01 s and about 350–450M MACs per decision.
- **Every model arm is above uniform random**, with intervals clear of 0.
- **Arms at or near the prior.** The prior reaches 0.323. The pass@3 − prior intervals of the Δ = 1 fixed arms and of C-J include 0.
- **Adaptive arms.** The interval of ICLR J − prior (+0.072) lies above 0. J − C-J is +0.048 [+0.028, +0.069], and J_dec − J is +0.051 [+0.028, +0.075]. Both J and C-J rank below the coarse fixed requests. This matches #103, where C-F-15 dominated both ICLR operating points.
- **Unit − C\*** is −0.042 [−0.060, −0.025], the same sign as Gate B's AUC contrast.
- **Shots to pass.** Among passed episodes it ranges from 1.74 (prior) to 2.40 (F-1-continuous). The second and third shots carry most of the gain: pass@3 is 2–5× pass@1 for every arm.

### Declared contrasts (pass@3)

| Set | Contrast | Estimate | Token |
| --- | --- | --- | --- |
| N1 | J − C-J | +0.048 [+0.028, +0.069] | readiness_or_precision_insufficient (G4) |
| N1 | J_dec − J | +0.051 [+0.028, +0.075] | readiness_or_precision_insufficient (G4) |
| N1 | unit − C* | −0.042 [−0.060, −0.025] | readiness_or_precision_insufficient (G4) |
| N1 | prior − random | +0.125 [+0.078, +0.172] | readiness_or_precision_insufficient (G4) |
| N1-mixed | J − C-J / J_dec − J / unit − C* / prior − random | +0.062 / +0.065 / −0.054 / +0.161 | readiness_or_precision_insufficient (G4) |
| N2-appearance | J − C-J / J_dec − J / unit − C* / prior − random | +0.025 [−0.033, +0.075] / +0.033 [−0.025, +0.100] / −0.008 [−0.033, +0.017] / +0.075 [−0.050, +0.208] | readiness_or_precision_insufficient (G4) |

The full per-arm table for the N1-mixed subset (311 levels with a recorded passing launch; the oracle reaches 0.997 there) is in `findings.md`. On that subset, F-15-continuous reaches 0.615 pass@3, J_dec 0.572, ICLR J 0.507, C-J 0.445, the prior 0.415 and random 0.254.

### N2-appearance: the 40 #110 evaluation appearance levels (reported separately)

The models were never fit on this novelty. Zero-shot, 120 cells.

| Arm | pass@3 | Arm | pass@3 |
| --- | --- | --- | --- |
| F-15-continuous | 0.233 [0.125, 0.358] | J_dec | 0.217 [0.117, 0.333] |
| C* | 0.225 [0.125, 0.333] | ICLR J | 0.183 [0.075, 0.308] |
| F-15-macro (unit) / F-15-micro | 0.217 | prior | 0.175 [0.075, 0.300] |
| F-5-* | 0.192–0.208 | C-J | 0.158 [0.083, 0.242] |
| F-1-* | 0.158–0.175 | random | 0.100 [0.042, 0.175] |
| oracle | 0.500 [0.350, 0.650] | | |

Only 20 of the 40 levels have a recorded passing launch. Every model − prior interval includes 0, so with 40 clusters the models are not separated from the prior.

### Request mix per shot (N1, adaptive arms)

The mix is the share of imagined rollout steps in the chosen candidate's rollout.

| Arm | Shot 1 | Shot 2 | Shot 3 |
| --- | --- | --- | --- |
| C-J λ 1e-4 | 1-continuous 0.94, 15-continuous 0.06 | 0.94 / 0.05 | 0.94 / 0.05 |
| ICLR J λ 1e-4 | 1-macro 0.70, 1-continuous 0.19, 15-micro 0.07, 1-micro 0.03 | 1-macro 0.79, 1-continuous 0.12 | 1-macro 0.67, 1-continuous 0.24 |
| J_dec λ 1e-1 | 15-continuous 0.46, 15-macro 0.38, 15-micro 0.16 | 0.45 / 0.42 / 0.13 | 0.47 / 0.41 / 0.12 |

- **C-J and J imagine almost entirely at Δ = 1.** This explains their high per-decision cost: 3.7G and 2.9G MACs and 0.09–0.14 s per decision.
- **J_dec stays at Δ = 15** and is 8–10× cheaper.
- **Within an arm, the mix barely moves across shots.** The state is identical, so only the chosen candidate changes. The per-shot and all-candidate mixes are in `findings.md`.
- **Fixed arms request one pair at every step.**

### Live executions and the determinism cross-check (declared; never the result)

- **Sharing.** 19,360 episodes (44 arm × seed rankings × 440 levels) and 49,374 shot choices shared **5,123 live executions** (out of at most 7,040 level × candidate pairs): 2,883 at shot 1, 1,175 new at shot 2 and 1,065 new at shot 3.
- **Verdicts.** 5,106 executions completed. Live pig removal equals the recorded #104 verdict of the same level and candidate **5,106/5,106**, and the stop kind does too, 5,106/5,106.
- **Physics trace.** The native trace (every step and event, run identifiers masked) is identical in 5,032/5,106 executions. The other 74 differ in the trace but not in the verdict.
- **Frames.** The live decision frame is byte-equal to the recorded one in 5,024/5,106 executions, and the final frame in 4,398/5,106. In the smoke, final-frame byte differences came with identical physics: 3–37 pixels in 4/27, consistent with the rebuilt Mesa (§ Disclosures). The renderer string matches the frozen one in 5,106/5,106.
- **Recorded replay.** Re-running the same rankings against the recorded verdicts gives pass@3 equal to the live value in every N2 cell. In N1 the outcome differs in at most 3 cells per arm, all of them typed-failure cells.

### Typed engine failures (recorded, never retried)

17 of the 5,123 executions are typed failures:
- **16 `supervisor_interrupted`** (shot-1 launch incident, § Execution).
- **1 `other`**: `ValueError: native contact has an unresolved participant` on `type010105-103-c21`.

Under the frozen rule, an episode that chose one of these ends as `typed_failure` and counts as not passed. That is 7–12 of the 1,200 N1 episodes per seeded arm, 3 for the prior and 1 for the oracle. All of these episodes sit on levels 055–058 of type010103 and level 103 of type010105.

## Protocol

- **Levels.** N1 is all 400 levels of the #104 `issue109-evaluation` stream (200 each of type010103 and type010105). The 311 levels with at least one recorded passing grid launch form the declared subset N1-mixed: the Gate-B cells. N2-appearance is the 40 `issue110-evaluation` levels at novelty level 1, novel side (5 families × 8). The inventory is the 16 grid launches that every level shares (`drag_x` ∈ {−40, −80, −120, −160} × `drag_y` ∈ {5, 30, 55, 80}, release 1000 ms).
- **Decision frame.** For N1 the planner uses the Gate-B anchor store: the E99 parse of shot frame 0 of the level's first complete recorded candidate. For N2 the same rule is parsed after the freeze (`--anchors`, 2 s).
- **Arms.** Every model arm is loaded through its source runner's hash-checked loader and scored with that runner's scoring block (#103 `load_arm` / `run_cell`), under `deterministic_scoring()`.
  - The nine fixed requests on the Gate-B hybrid LC predictor.
  - C* = C-F-15.
  - C-J and J at λ = 1e-4, the ICLR operating recipe (#114 / #101).
  - J_dec, the #101 selected `J-dec-1e-01`.
  - The no-model prior: the Gate-B rule, minus the fit-split success frequency of the identical action.
  - Uniform random: one `PCG64([102, seed, level index])` permutation per seed and level; taking the first remaining entry samples uniformly without replacement.
  - The oracle: the recorded passing launches, lowest ordinal first, executed live. It takes no shot when no launch is recorded as passing.
  - Model and random seeds are 20260908, 20260909 and 20260910. The prior and the oracle are seed-free and repeated per seed in the cell table, following the Gate-B convention.
- **Ranking.** The lowest #111 tie-free endpoint cost ranks first. Ties go to the lower ordinal, and nonfinite endpoints come last (none occurred).
- **Episode sharing (pre-declared).** There is one live execution per distinct level × candidate chosen by any arm, seed or shot. Every episode that chooses it reads the same live verdict.
- **Pass.** Pass is live engine pig removal (`capture_pipeline_v2.branch_outcome`, the #85 channel). Native level clear is reported in `findings.md`.
- **Statistics.** Reported per arm: pass@1/2/3, shots to pass among passed episodes, shots executed, native-clear pass@3, per-decision host seconds, CUDA-synchronized rollout seconds, linear MACs and calls (one process per GPU, warm-up discarded), and the request mix per shot. The paired contrasts on pass@3 are every arm − prior, every arm − random, J − C-J, J_dec − J, unit − C* and prior − random. They use the repo δ rule (δ = 0.02, `c2.token`). The random arm's seed s is paired with model seed s.
- **Guards** (a failure turns every token into `readiness_or_precision_insufficient`):
  - G1: the scoring policy on every decision record.
  - G2: re-ranking identity across shots, and shot-1 rows equal to the retained #101/#114 rows on N1-mixed.
  - G3: chronology.
  - G4: the live decision frame parses within 1e-3 of the anchor.
  - G5: execution integrity.
- **Smoke** (pre-freeze; 3 fit-split levels: two #109-fit, one #110-fit appearance; 7 arms; 27 live executions):
  - Verdicts, stop kinds and native traces equal #104 27/27, and decision frames are byte-equal 27/27.
  - The retained #101/#114 rows are reproduced on 2 evaluation cells for all 13 model arms.
  - Re-ranking is identical, and every decision ran under `deterministic_scoring()`.
  - The gallery rendered.

## Guards

| Guard | Result |
| --- | --- |
| G1 scoring policy | pass (132 decision records) |
| G2 decision determinism | pass: 45,360 re-rankings and 12,129 shot-1 rows equal to the retained #101/#114 records |
| G3 chronology | pass: freeze 00:19:38Z ≤ first decision 00:20:27Z ≤ first dispatch 01:10:49Z |
| **G4 decision frame** | **FAIL**: 25 of 5,106 complete executions, on 5 levels (N1 type010103-021, type010105-140, type010105-153; N2 type010104-novel-001, type010105-novel-008), parse their live decision frame up to 0.0032 from the anchor, against a tolerance of 1e-3. The other 5,081 parse within tolerance. |
| G5 execution integrity | pass: 5,123 executions = 5,123 chosen, each with a verdict, no candidate repeated in an episode, 0 incomplete episodes |

**Cause of the G4 failure.** On these levels the recorded #104 decision frames themselves differ between candidates. The anchor is the frame of the first recorded candidate (c00). Live frames that are byte-equal to their own candidate's recorded frame still parse 0.0013–0.0027 from the c00 anchor. The decision frame of these levels is therefore not pixel-stable across engine runs. This matches #109's finding that the blink phase at the decision step is a level property. The guard was frozen, so the failure stands.

**POST-HOC (declared after the failure; changes no token).** The grid was re-ranked from each distinct live decision frame on those 5 levels with every model arm and seed. The first pick was unchanged in 351/351 comparisons, and the first three picks were unchanged in 351/351 (`records/posthoc-decision-frames.json`). No closed-loop choice would have differed.

## Execution and disclosures

- **Launches.** Each stage was launched once; PIDs are in `logs/pids.txt`. The stages were the smoke, `--anchors`, `--decide` for shots 1/2/3 (4 GPUs, one process each), `--execute` for shots 1/2/3 (16 engine workers), `--posthoc-frames`, `--gallery`, `--publish` and `--validate`.
- **Shot-1 execution incident.** At 362 of 2,883 dispatches, the `capture_pipeline_v2` supervisor raised `worker … did not establish its process group`. The cause was the 5 s worker-start window: the spawned worker re-imported the runner's heavy `__main__` module under a load average of about 90.
  - The pipeline sealed the 14 live workers as `supervisor_interrupted`, plus one attempt without a result. These were never retried.
  - An orphan Xvnc from the failed worker was killed. No engine worker was alive afterwards, and the stage was resumed once.
  - The resume, and the shot-2 and shot-3 stages, used a `python -c` launcher, so spawned workers no longer import the runner. The runner code was unchanged. No further incident occurred.
- **Post-freeze runner changes** (publication and accounting only; no decision, execution or statistic rule changed):
  - Engine accounting reads the raw run-logs and receipts. The interrupted launch wrote a run-log but no ledger entry.
  - The engine-wall cap check reads the run-logs.
  - The post-hoc decision-frame mode and its report section were added.
- **#104 inputs.** `prepare_power_cohort.load_plan` fails its input re-hash on exactly two files: the #96 `plan.json` and the #104 capture-dev `summary.json`. Their diffs are machine-local home-directory path rewrites only, so the #104 plan is read directly; any other changed input refuses.
- **Renderer.** The Mesa 26.1.2 / LLVM 22.1.6 build was rebuilt on this host on 2026-10-14, with different embedded install paths. Three library hashes therefore differ from #104. The renderer string was checked on every execution, and frame and physics equality are reported above.
- **Hardware.** 4× RTX 5090 instead of the #85 RTX 3090, and 48 CPU threads (16 llvmpipe engine workers).
- **Known before the freeze.** Every #101/#103/#105/#114 grid ranking on the N1-mixed levels and the #104 verdict yields were known. #103's decision-only top-3 hit equals recorded-replay pass@3 under this semantics. No model output existed on the 89 N1 levels without a recorded pass, or on any #110 evaluation level.

## Compute

| Item | Seconds |
| --- | --- |
| GPU (decisions 3 shots × 44 rankings, N2 anchors, live-frame parses, post-hoc) | 3,379 (0.94 GPU-h) |
| Engine campaign wall (16 workers; 4 launches incl. the interrupted one) | 60,001 (16.7 h) |
| Engine worker seconds (sum of per-execution worker wall) | 953,176 (264.8 h) |
| Stage wall (ledger: decide, execute + verdicts, anchors, post-hoc) | 67,637, plus 4,479 for the interrupted launch |
| Smoke (pre-freeze) | GPU 10, engine 4,150, campaign wall 617 |
| Artifacts added | 403 GB (cap 1.5 TB) |

The worst-case projection was 7,040 executions and 23.4 h; the frozen cap was 48 h of campaign wall. The realised run was 5,123 executions and 16.7 h, so no scope was cut.

## Validation

`python -u -m scripts.run_closed_loop_evaluation --validate` regenerates `compute.json`, `summary.json`, `findings.md`, `closed-loop-table.csv`, `manifest.json` and the gallery `manifest.json` from the raw records and byte-compares them. It also checks:
- re-deciding the Gate-B unit (seed 20260908) on 20 levels reproduces the recorded row digests;
- recomputing 12 live verdicts (pig removal, stop kind, frame and native-trace comparison) from the retained engine attempts matches the records.

## Post-freeze runner edit (phase-A review)

Commit `b5ac0fad` adds one validation check: `--validate` now compares `smoke.json` with the sha256 pinned in `plan.json` (`smoke_matches`). Before this, smoke.json was the only published file whose later modification `--validate` could not detect. The retained smoke.json matches its pin. No evidence changed: every published file is byte-identical, and `--validate` exits 0.

The supervisor crash and resume are handled by recording 16 typed `supervisor_interrupted` failures, with no retries, plus the global G4 veto. These failures can affect the observed episode outcomes, so this handling is conservative; it does not show that the crash had no effect on the evidence.
