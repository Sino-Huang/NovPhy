# Issue #107: granularity under novelty

Parent: #108. The binding shared rules of #85 apply, and the run adopts the #111 scoring contract. The #110 contract-v2 results (`handoff_107.json`), the #101 request-policy recipe, Gate B and #109 are read-only inputs. Nothing here edits the ICLR 2026 submission.

| | |
| --- | --- |
| Runner | `scripts/run_granularity_under_novelty.py` |
| Tests | `tests/test_issue_107_granularity_under_novelty.py` (6 tests: tokens, view keys, share shifts, v2 cost columns, wrong-anchor rows, disposition rule) |
| Artifacts | `.local-artifacts/issue-107-granularity-under-novelty-v1/`. Tracked: `plan.json`, `smoke.json`, `findings.md`, `summary.json`, `compute.json`, `records-manifest.json` (raw-record hash manifest for 612 records), the training reports `models/*/*/seed-*/report.json` and `ledger/`. Not tracked: policies, pools, shot prefixes, records. |
| Freeze | Plan frozen 2026-10-16T02:47:55Z (commit f4e0e723), after a passed smoke and before any training or score. |
| Validation | `python -u -m scripts.run_granularity_under_novelty --validate` exits 0. It checks the record manifest, re-derives and byte-compares `compute.json`, `summary.json` and `findings.md`, re-scores two records exactly (rows and schedules), and re-parses two prefix shots exactly. |

Intervals are percentile bootstraps clustered by evaluation level (member): 10 000 draws, PCG64 7201, DESCRIPTIVE. The margin is δ = 0.02. Zero-shot and few-shot are separate sections, and no estimate pools their rows. Zero engine seconds.

## Result: disposition `not_supported_by_this_experiment`

The question is whether request selection is a cheap adaptation surface under novelty. The answer is the same for the primary policy J_dec and the secondary policy J. The population is the pooled novel levels 1–7: 122 mixed evaluation levels × 3 seeds. All six guards pass.

| policy | (a) policy-only retune − zero-shot | (b) policy-only − predictor-only (non-inferiority) | (c) policy retune cheaper | disposition |
| --- | --- | --- | --- | --- |
| **J_dec** (primary) | −0.007 [−0.021, +0.006]: upper < δ | −0.158 [−0.214, −0.104]: upper < −δ | yes: 841 vs 6 339 GPU s | **`not_supported_by_this_experiment`** |
| J (secondary) | −0.057 [−0.079, −0.034] | −0.170 [−0.233, −0.110] | yes: 740 vs 6 339 GPU s | `not_supported_by_this_experiment` |

Retuning only the request policy on the few-shot data is cheap: 49 417 policy parameters against 2.85 M for the LC hybrid predictor, and 841 GPU s against 6 339 GPU s for the predictor members (plus 829 GPU s of encoder adaptation). It is not, however, an adaptation surface here:
- the policy-only retune leaves the pooled J_dec AUC unchanged;
- retuning the predictor (the #110 few-shot world) raises it by +0.151 [+0.097, +0.206] (δ rule: `supported`);
- retuning both does not add a ≥ δ gain over retuning the predictor alone: both − predictor = −0.006 [−0.020, +0.009].

## Acceptance table: per novelty type, AUC per arm

Novelty types follow the #110 force-law table: 1 appearance (pink pig/blocks), 2 fan, 3 stronger air turbulence, 4 magnet, 5 right slingshot (mirrored layout), 6 inverse gravity, 7 inverse air turbulence. "Levels" counts the mixed evaluation levels; each one enters with 3 seeds.

### Zero-shot: normal-trained world and policies, frozen

| novelty type | levels | J | J_dec | best fixed in hindsight | no-model prior (normal fit) |
| --- | --- | --- | --- | --- | --- |
| N1 (#109 evaluation, Gate-B grid) | 311 | 0.72 [0.70, 0.74] | 0.72 [0.70, 0.74] | F-15-micro 0.72 [0.71, 0.74] | 0.70 [0.68, 0.73] |
| 0: paired normal sides | 135 | 0.63 [0.59, 0.67] | 0.65 [0.61, 0.68] | F-15-micro 0.66 [0.63, 0.70] | 0.76 [0.73, 0.79] |
| 1: appearance | 20 | 0.66 [0.54, 0.76] | 0.64 [0.53, 0.74] | F-15-micro 0.65 [0.54, 0.75] | 0.65 [0.56, 0.74] |
| 2: fan | 17 | 0.51 [0.42, 0.60] | 0.53 [0.42, 0.63] | F-15-continuous 0.54 [0.43, 0.64] | 0.74 [0.61, 0.85] |
| 3: stronger turbulence | 6 | 0.66 [0.51, 0.80] | 0.62 [0.52, 0.73] | F-1-continuous 0.67 [0.45, 0.88] | 0.72 [0.66, 0.78] |
| 4: magnet | 18 | 0.61 [0.53, 0.67] | 0.60 [0.52, 0.67] | F-1-macro 0.62 [0.55, 0.69] | 0.55 [0.49, 0.62] |
| 5: right slingshot | 17 | 0.60 [0.48, 0.71] | 0.60 [0.47, 0.71] | F-1-macro 0.66 [0.57, 0.75] | 0.86 [0.76, 0.94] |
| 6: inverse gravity | 22 | 0.38 [0.28, 0.48] | 0.33 [0.24, 0.43] | F-15-macro 0.36 [0.26, 0.45] | 0.21 [0.13, 0.30] |
| 7: inverse turbulence | 22 | 0.55 [0.41, 0.67] | 0.56 [0.41, 0.70] | F-15-micro 0.62 [0.46, 0.77] | 0.64 [0.51, 0.76] |
| **novel 1–7 pooled** | 122 | 0.55 [0.50, 0.60] | 0.54 [0.49, 0.59] | F-15-micro 0.55 [0.50, 0.60] | 0.60 [0.55, 0.65] |

### Few-shot: level-k novel fit sides

| novelty type | J_dec policy-only | J_dec predictor-only | J_dec both | J policy-only | J predictor-only | J both | best fixed, few-shot world | few-shot prior |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1: appearance | 0.65 [0.54, 0.75] | 0.66 [0.55, 0.75] | 0.62 [0.54, 0.70] | 0.58 [0.48, 0.68] | 0.62 [0.54, 0.69] | 0.67 [0.56, 0.76] | F-1-continuous 0.67 [0.57, 0.76] | 0.67 [0.58, 0.76] |
| 2: fan | 0.55 [0.44, 0.64] | 0.56 [0.45, 0.66] | 0.57 [0.47, 0.66] | 0.51 [0.42, 0.61] | 0.55 [0.48, 0.62] | 0.58 [0.47, 0.67] | F-1-continuous 0.58 [0.47, 0.68] | 0.80 [0.66, 0.92] |
| 3: stronger turbulence | 0.63 [0.51, 0.75] | 0.47 [0.18, 0.76] | 0.56 [0.30, 0.83] | 0.64 [0.54, 0.74] | 0.46 [0.17, 0.75] | 0.56 [0.30, 0.83] | F-1-continuous 0.51 [0.26, 0.78] | 0.82 [0.76, 0.89] |
| 4: magnet | 0.56 [0.48, 0.64] | 0.64 [0.53, 0.73] | 0.62 [0.52, 0.71] | 0.58 [0.51, 0.66] | 0.60 [0.52, 0.67] | 0.62 [0.52, 0.73] | F-15-macro 0.64 [0.54, 0.73] | 0.59 [0.47, 0.71] |
| 5: right slingshot | 0.62 [0.50, 0.72] | 0.68 [0.61, 0.75] | 0.69 [0.61, 0.76] | 0.56 [0.46, 0.66] | 0.63 [0.58, 0.70] | 0.73 [0.66, 0.79] | F-1-macro 0.76 [0.69, 0.81] | 0.59 [0.48, 0.69] |
| 6: inverse gravity | 0.32 [0.23, 0.41] | 0.79 [0.73, 0.85] | 0.79 [0.73, 0.85] | 0.26 [0.18, 0.34] | 0.82 [0.76, 0.87] | 0.74 [0.68, 0.80] | F-1-macro 0.82 [0.76, 0.88] | 0.79 [0.74, 0.84] |
| 7: inverse turbulence | 0.52 [0.39, 0.65] | 0.84 [0.76, 0.91] | 0.81 [0.73, 0.89] | 0.47 [0.35, 0.59] | 0.78 [0.72, 0.84] | 0.79 [0.71, 0.85] | F-1-micro 0.86 [0.79, 0.92] | 0.88 [0.83, 0.93] |
| **novel 1–7 pooled** | 0.53 [0.49, 0.58] | 0.69 [0.65, 0.73] | 0.69 [0.65, 0.72] | 0.49 [0.45, 0.54] | 0.66 [0.63, 0.70] | 0.69 [0.65, 0.72] | F-1-macro 0.71 [0.67, 0.75] | 0.73 [0.69, 0.77] |

`findings.md` has the remaining material: the nine fixed requests in each world with intervals, every contrast with its token, per-cell AUCs and wrong-anchor tables. The per-type tokens are DESCRIPTIVE. Only the pooled row carries the disposition.

## Acceptance table: request shares per arm (Work 3)

Shares are the percentage of rollout steps that choose each request. They are computed per (level, seed) over the 16 candidates and use every evaluation level of the in-scope cells (outcome-free).

Shift reference: the same kind's normal policy on the **paired normal side**, meaning the same pair and ordinal in the zero-shot world. TV is half the L1 norm of the shift.

**J_dec** stays at Δ = 15 in at least 91 % of steps in every condition. Δ < 15 takes 6–9 % of steps only in three arm/type combinations: policy-only on magnet, and both on appearance and on stronger turbulence. The table gives the α split at Δ = 15 as continuous / micro / macro.

| novelty type | zero-shot | policy-only | predictor-only | both |
| --- | --- | --- | --- | --- |
| #101 J_dec, N1 (E99 Gate-B predictor) | 52 / 10 / 38 | – | – | – |
| N1 (v2 zero-shot world) | 76 / 1 / 22 | – | – | – |
| 0: paired normal sides | 69 / 2 / 29 | – | – | – |
| 1: appearance | 76 / 2 / 22; TV 0.08 | 82 / 2 / 16; TV 0.13 | 82 / 1 / 18; TV 0.29 | 69 / 0 / 24 (+8 Δ5-cont); TV 0.35 |
| 2: fan | 64 / 2 / 35; TV 0.15 | 68 / 5 / 27; TV 0.20 | 77 / 0 / 23; TV 0.41 | 65 / 2 / 33; TV 0.43 |
| 3: stronger turbulence | 76 / 0 / 23; TV 0.12 | 56 / 10 / 34; TV 0.24 | 43 / 2 / 55; TV 0.37 | 53 / 0 / 37 (+9 Δ1/5-macro); TV 0.31 |
| 4: magnet | 65 / 2 / 32; TV 0.11 | 71 / 5 / 19 (+6 Δ1/5); TV 0.27 | 75 / 0 / 25; TV 0.37 | 60 / 3 / 37; TV 0.50 |
| 5: right slingshot | 89 / 2 / 9; TV 0.18 | 72 / 2 / 25; TV 0.20 | 53 / 2 / 45; TV 0.40 | 67 / 5 / 27; TV 0.39 |
| 6: inverse gravity | 88 / 2 / 9; TV 0.15 | 85 / 2 / 12; TV 0.16 | 77 / 1 / 23; TV 0.27 | 66 / 1 / 33; TV 0.42 |
| 7: inverse turbulence | 67 / 2 / 31; TV 0.08 | 59 / 5 / 36; TV 0.28 | 70 / 2 / 28; TV 0.38 | 65 / 3 / 32; TV 0.30 |
| **novel 1–7 pooled** | 75 / 2 / 23; TV 0.13 | 72 / 4 / 24; TV 0.20 | 70 / 1 / 29; TV 0.35 | 65 / 2 / 31; TV 0.39 |

**J (ICLR prediction-error DP)**: Δ = 1 / 5 / 15 and α = continuous / micro / macro shares.

| novelty type | zero-shot | policy-only | predictor-only | both |
| --- | --- | --- | --- | --- |
| #101 J, N1 (E99 Gate-B predictor) | Δ 94/0/6; α 12/8/80 | – | – | – |
| N1 (v2 zero-shot world) | Δ 69/3/29; α 68/30/2 | – | – | – |
| 0: paired normal sides | Δ 60/2/39; α 52/41/7 | – | – | – |
| 1: appearance | Δ 71/4/26; α 71/27/2; TV 0.09 | Δ 69/18/13; α 41/18/41; TV 0.62 | Δ 72/2/25; α 46/8/46; TV 0.66 | Δ 63/1/36; α 35/29/36; TV 0.68 |
| 2: fan | Δ 63/2/35; α 52/42/6; TV 0.14 | Δ 78/2/20; α 55/22/23; TV 0.42 | Δ 60/10/30; α 15/22/63; TV 0.73 | Δ 52/2/46; α 52/39/9; TV 0.47 |
| 3: stronger turbulence | Δ 39/0/60; α 23/70/7; TV 0.21 | Δ 57/2/41; α 39/29/32; TV 0.42 | Δ 63/2/35; α 5/34/61; TV 0.54 | Δ 55/3/42; α 35/40/25; TV 0.59 |
| 4: magnet | Δ 60/1/38; α 47/48/5; TV 0.10 | Δ 79/3/18; α 39/39/22; TV 0.46 | Δ 60/0/39; α 10/35/55; TV 0.70 | Δ 49/6/45; α 56/16/28; TV 0.76 |
| 5: right slingshot | Δ 53/1/46; α 46/53/2; TV 0.27 | Δ 69/15/17; α 45/28/27; TV 0.52 | Δ 84/5/11; α 38/3/59; TV 0.70 | Δ 63/0/36; α 43/37/21; TV 0.57 |
| 6: inverse gravity | Δ 67/4/29; α 67/32/1; TV 0.25 | Δ 52/23/26; α 42/10/47; TV 0.66 | Δ 69/1/30; α 12/31/57; TV 0.67 | Δ 54/2/44; α 18/22/61; TV 0.71 |
| 7: inverse turbulence | Δ 45/0/55; α 31/54/16; TV 0.15 | Δ 61/3/36; α 35/27/38; TV 0.49 | Δ 83/0/17; α 1/14/85; TV 0.73 | Δ 66/3/30; α 59/34/7; TV 0.67 |
| **novel 1–7 pooled** | Δ 58/2/40; α 50/44/5; TV 0.17 | Δ 66/10/23; α 43/24/34; TV 0.52 | Δ 71/3/26; α 20/19/61; TV 0.69 | Δ 58/2/40; α 42/31/27; TV 0.63 |

Shift intervals for every request and marginal are in `findings.md`. In the pooled novel row, "moves toward" means the shift interval lies above 0, in percentage points:
- **Zero-shot J_dec** moves toward (15, continuous): α continuous +5 [+3, +7] and macro −5 [−7, −3]. The largest per-type moves are on right slingshot (continuous +16 [+13, +19]) and inverse gravity (+11 [+4, +19]).
- **Zero-shot J** barely moves (TV 0.17; pooled horizon shift Δ 1 −2 [−4, −1] pp, Δ 15 +2 [+1, +4] pp). Its interval-excluding moves are toward (1, micro), (15, micro) and (15, macro), and away from (1, macro). The exception is right slingshot, where Δ 15 is +16 [+13, +20] and α micro +22 [+18, +26].
- **Predictor-only J** (retuned world, frozen policy) moves strongly toward macro (+54 [+53, +56]) and Δ = 1 (+11 [+8, +13]). On inverse turbulence it chooses (1, macro) in 82 % of steps. **Predictor-only J_dec** keeps its pooled α split (macro +1 [−2, +4]), but per type it moves to macro on stronger turbulence (+25) and right slingshot (+19), and to continuous on fan (+16).
- **Policy-only J** moves toward macro (+27) and Δ = 5 (+8). **Policy-only J_dec** moves little: macro −4 [−7, −1].
- **Both J_dec** moves toward macro (+4 [+2, +7]) and away from (15, continuous).

The request mix changes most when the world is retuned, and it changes for J far more than for J_dec. Those changes do not add ranking value: both − predictor ≈ 0 for J_dec.

## Readings

- **Novelty costs ranking value, and the zero-shot policy does not offset it.**
  - Pooled zero-shot J_dec drops from 0.65 on the paired normal sides to 0.54 on the novel sides.
  - On novel levels it trails the pre-declared F-15-macro by −0.011 [−0.020, −0.002] and the in-hindsight best fixed request by −0.011 [−0.022, −0.001].
  - On N1 the v2 J_dec ties the best fixed request: −0.001 [−0.003, +0.001].
  - Inverse gravity inverts the zero-shot ranking: J_dec 0.33, every fixed request 0.30–0.36, prior 0.21.
- **The world, not the request policy, carries the adaptation.** The predictor-only retune lifts J_dec on inverse gravity (+0.457 [+0.325, +0.581]) and inverse turbulence (+0.284 [+0.141, +0.445]), both `supported`. The policy-only retune changes no type by ≥ δ with a positive lower bound.
- **The no-model prior is hard to beat on these cells.** On the paired normal sides, the prior estimated from normal fit sides (0.76) ranks above every model arm (J_dec − prior −0.113 [−0.159, −0.067]). The few-shot prior (0.73 pooled) ranks above every few-shot arm (both J_dec − few-shot prior −0.046 [−0.093, +0.003]). On N1, J_dec − prior is +0.019 [−0.007, +0.045].
- **The wrong-anchor control.**
  - On N1, right − wrong is +0.146 [0.133, 0.159] for zero-shot J_dec (#101: +0.128).
  - On the paired normal sides it is +0.060 [0.034, 0.085].
  - On the novel sides it is +0.038 [0.002, 0.071], and +0.061 [0.031, 0.092] for both-retuned J_dec.
  - The v2 policies therefore use the anchor, but less on the issue110 cells than on N1.
- **J vs J_dec.**
  - On N1 the v2 J and J_dec tie: J_dec − J = +0.004 [−0.003, +0.012]. #101 reported +0.040 with the E99 predictor.
  - The v2 J is mostly a Δ = 1 policy on N1 (Δ 69 %, α continuous 68 % / micro 30 %), whereas #101's J was 79 % (1, macro). The v2 J_dec is a Δ = 15 policy, like #101's.

## Protocol

### Scope (#110 handoff)

**In scope.** A cell is in scope iff it is #110-supported, has mixed-verdict evaluation levels, and is not rollout-error-only. That gives 29 cells:
- 0/{single_force, rolling, falling, sliding};
- 1/*, 2/*, 5/*, 6/*, 7/* (all without multiple_forces);
- 3/{rolling, falling};
- 4/{single_force, rolling, falling}.

**Out of scope.** These 16 cells are `readiness_or_precision_insufficient`, with the #110 reason:
- the 9 multiple-forces cells: `rollout_error_only`, degenerate single-shot verdicts because the pigs need multiple hits;
- 3/single_force, 3/sliding, 4/sliding and 8/{single_force, rolling, falling, sliding}: `no_mixed_verdict_evaluation_level`.

Level 8 (storm) therefore gets no few-shot policy retune. Their ranking estimand stays out of scope. Their request shares (frozen normal policies on both worlds) are reported in the post-freeze DESCRIPTIVE supplement; see "Phase-A review fixes".

### Worlds

| world | what it is | arm |
| --- | --- | --- |
| zero-shot | the #110 zero-shot encoder, scene head and LC hybrid (normal fit sides only) | — |
| few-shot-k | the #110 few-shot-k encoder, scene head and LC hybrid (2 000 updates on the level-k novel fit sides) | "retune only the predictor" |

- Seeds are 20260908–10. Policy seed s uses predictor seed s.
- Every checkpoint hash equals the handoff, and every member used is usable.
- **Anchors** follow the #110 v2 rule: the world's parse of frame 0 of the level's first complete candidate in suffix order. On issue110 levels this is the #110 stored carriers.
- **N1 and policy-split shots** are parsed here with the #110 carrier rule over the whole shot, keeping positions 0..63. In the smoke these parses are bit-identical to the #110 store.

### Policies

- **Shared module.** `request_policy.py` runs as a private module with `DIM = 370`; its bytes are unchanged. In the smoke the copy equals the #101 module bit for bit at the v1 width. The network is Linear(376, 128)–SiLU–Linear(128, 9).
- **J** is the prediction-error DP at λ = 1e-4. **J_dec** is the decision-valued DP at λ = 1e-1, the #101 selection; no new sweep was run.
- **Normal policies.** Trained with the #101 recipe on the #109 policy split against the zero-shot world, then frozen:
  - recipe: 2 DAgger rounds × 1 800 steps, batch 128, AdamW 1e-3, #77 initialization;
  - pool: 120 levels, 3 480 shots, 167 mixed grid and angle cells;
  - dec continuation: F-15-macro, then the round-0 policy.
- **Few-shot retunes.** The normal policy of the same kind and seed is retuned with the same recipe on the level-k novel fit sides. Each level has 39–40 levels and 624–640 shots. pe labels use every shot's first window; dec labels use the level's 8–25 mixed grid cells.
  - **P** retunes against the zero-shot world ("policy only").
  - **B** retunes against the few-shot-k world ("both").
- **Training outcome.** There are 6 normal and 84 few-shot groups, with 0 typed failures.

### Scoring (once)

- **Process.** Scoring ran once, in processes that did not train, under `deterministic_scoring()`, through `rollout_endpoints` with the joint argmax.
- **Arms.**
  - Zero-shot world: the nine fixed requests, plus J and J_dec of N; on novel groups also those of P.
  - Few-shot-k world: the nine fixed requests, plus J and J_dec of N (predictor-only) and of B (both).
- **Records.** 612 records (group × world × arm × seed) cover every evaluation level of the in-scope cells (N1: 400 levels), with schedules for the policies.
- **Cost and AUC.** The cost is tie-free at the v2 pig columns 175 / (180, 181). The AUC is `m99.view` on the mixed levels.
- **Controls.**
  - *No-model prior:* cost = −(fit success frequency of the grid ordinal). Mirrored level-5 actions share ordinals. The zero-shot prior uses the normal fit sides; the few-shot prior uses the level-k novel fit sides.
  - *Wrong anchor:* the target's verdicts against the same arm's rollouts from every other level of the group.
  - *F_hindsight:* the best fixed request of the population; it is outcome-selected and DESCRIPTIVE.
- **Guards, all pass.**
  - G1 isolation;
  - G2 scoring policy;
  - G3 chronology (freeze 02:47:55 ≤ first training 02:54:27 ≤ last training 03:11:25 ≤ first score 03:11:38);
  - G4 checkpoint binding (world hashes = handoff, scored policy = report, retune init = normal policy hash);
  - G5 prefix-parse replication;
  - G6 scored inventories and verdicts equal the handoff (7 278 branches).

### Disposition rule (frozen)

The disposition uses J_dec as the primary policy (J secondary) on the pooled novel population. The member is the evaluation level.

**Components:**
- **(a) gain** = AUC(policy-only) − AUC(zero-shot), δ rule: supported iff the mean is ≥ 0.02 and the lower bound is > 0; not supported iff the upper bound is < 0.02.
- **(b) parity** = AUC(policy-only) − AUC(predictor-only), non-inferiority: supported iff the lower bound is > −0.02; not supported iff the upper bound is < −0.02.
- **(c) cheap**: the summed policy-only retune GPU s is below the summed #110 few-shot hybrid member training GPU s for the same levels and seeds.

**Outcome:**
- `supported` iff (a) and (b) are supported and (c) holds;
- `not_supported_by_this_experiment` iff (a) or (b) is not supported, or (c) fails;
- `readiness_or_precision_insufficient` otherwise;
- a failed guard turns every token into `readiness_or_precision_insufficient`.

## Decisions under delegation (posted on #107)

1. **Policies at the v2 carrier.** The #101 policies read 236 values and cannot read the 370-value v2 carrier. J and J_dec were therefore re-trained with the unchanged #101 recipe on the #109 policy split against the zero-shot v2 world, then frozen.
2. **Policy-split and N1 parses.** These use the #110 carrier rule.
3. **The three few-shot arms.** As above. The predictor-only arm is the whole #110 few-shot world.
4. **Scope.** 29 cells. Level 8 is not retuned.
5. **Shift reference.** The paired normal side, with N1 references from the v2 policies and from #101.
6. **Controls.** As above.
7. **Disposition rule and primary policy.** As above.
8. **Compute.** The projection was 1.0 h on 4 GPUs against a 48 h cap, so no scope cut was needed.

## Compute and disclosures

- **Compute.**
  - Totals: 7 287 GPU s (2.0 GPU-h) and 7 287 s of summed process wall time; 0 engine seconds.
  - By stage, in GPU s: data 1 049, assemble 43, normal training 614, few-shot training 3 121, scoring 2 442, smoke 18.
  - Elapsed wall after the freeze was about 35 minutes on 4 GPUs. Validate takes 23 s.
- **Hardware.** RTX 5090 instead of the #85 RTX 3090.
- **Post-freeze runner edits.** Neither touches training, scoring or the decision rule.
  - *Data task listing.* The first `--data` launch failed while listing tasks, before any parse: the Gate-B anchor branch of an N1 level is not always a grid candidate, so the lookup of its action failed. The lookup now reads the anchor branch's own action. This changes neither the anchor rule nor which branch is parsed. The stage was relaunched once.
  - *Cost rendering.* After the first publish, the adaptation-cost totals were corrected to count the shared per-level encoder and scene-head adaptation once per level instead of once per seed, and the "both" retune cost was added. These are publication code only; no record changed, and the disposition input (hybrid member GPU s against policy retune GPU s) was unchanged.
  - The runner hash at the freeze is recorded in `plan.json`.
- **Phase-A review fixes (commit `ddd9f427`).** These are post-freeze edits of reporting, guard propagation and validation, plus one declared outcome-free supplement. Every frozen table, estimate and disposition is unchanged: in `compute.json` and `summary.json`, every key that existed before is byte-equal, and `records-manifest.json` is unchanged.
  - *Guard propagation.* The frozen rule says that a failed guard turns every token into `readiness_or_precision_insufficient`. The runner as published applied this to the disposition only. It now also invalidates every population, cell and wrong-anchor token. All guards pass, so no value changed.
  - *Request shares on every supported cell (post-freeze, DESCRIPTIVE, outcome-free).* The frozen scope gave no request shares for the 16 out-of-scope cells, including all of level 8 (storm), because they have no mixed-verdict level or are rollout-error only. The ticket asks for request mix on every supported cell, and undefined AUC does not make policy request frequencies undefined. `--share-supplement` therefore rolled out only the frozen policies (N-pe/N-dec everywhere, plus P/B on levels 1–7 where they were trained) on those cells' evaluation levels. It used the #110 handoff worlds and anchor rule, under `deterministic_scoring()`, with zero engine seconds. That gave 186 records, 141 GPU s, in `supplement/` with their own ledger and `supplement/manifest.json`. No verdict is read, and no AUC, contrast or token is formed. `compute.json` `shares_supported_cells` (and the findings section "Request shares on every supported cell") reports request shares per cell, per novelty type (L0–L8) and pooled novel 1–8. In-scope cells use the retained records, and the other cells use the supplement. Example: on storm (level 8), zero-shot J_dec stays at Δ = 15 (100 %), α continuous 72 % / macro 26 %; zero-shot J is Δ = 1 at 69 %. `--validate` checks the supplement manifest and re-scores one storm supplement record exactly.
  - *Input check.* `load_plan` checks the pinned #101 and #110 runners through `scripts/post_freeze_amendments.py`, which accepts exactly their frozen or disclosed amended bytes.
  - *Not recoverable.* The first, pre-cost-fix publication was not preserved, so that earlier rendering change cannot be audited from the repository.
- **Known before the freeze.** Every #101 / Gate-B / #110 result, including the #110 rollout errors and the per-cell mixed-level counts. No request policy had been trained on a v2 world, and no policy output existed on an evaluation capture.
- **Cost accounting.** Policy-retune cost is the group's process wall time: label rollouts, distillation, the in-sample check and model loading. This is conservative toward "cheap". Predictor cost is the #110 ledger's hybrid member training only; the encoder (829 GPU s) and scene heads (22 GPU s) are reported separately.
- **Reproducibility.** Training is not claimed bit-reproducible (same caveat as #101). The retained policies and records are the published ones.
- **Small cells.** Level 3 has 6 mixed levels and several cells have 1–3, so per-type and per-cell intervals are wide and DESCRIPTIVE.

## Claim boundary

This run is one look at the issue110-evaluation grid cells, with the #109 evaluation split as the N1 reference. It covers request policies trained on normal data and few-shot retuned on the novel fit sides. Ranking uses the tie-free cost, and every interval is DESCRIPTIVE. There is no closed-loop claim.

"Not supported" means that retuning only the request policy did not recover ranking under this recipe and these worlds. It does not mean that request choice can never matter under novelty.

## Reproduction

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate novphy && source env.sh
python -u -m scripts.run_granularity_under_novelty --validate   # exit 0
python -m pytest -q tests/test_issue_107_granularity_under_novelty.py
```
