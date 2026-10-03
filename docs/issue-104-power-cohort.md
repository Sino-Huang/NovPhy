# Issue #104 — joint #104 + #110 cohort plan (frozen) and rendered smoke

Parent: #108. Shared rules of #85 apply. Prior dispositions (#15–#113) are read-only inputs; nothing here
edits the ICLR 2026 submission. No cohort level has been captured, fitted or scored.

| Item | Location |
| --- | --- |
| #113 capture-window spec (frozen 2026-10-03T09:03:55Z, zero engine seconds) | `scripts/derive_capture_window_spec.py`, `.local-artifacts/issue-113-capture-window-v1/` |
| Capture player (parent 82db3f42 + window rule) | `scripts/issue_104_window_player.py`; assembly `ac418250…` |
| Renderer (Mesa 26.1.2 / LLVM 22.1.6 llvmpipe, no root) | `scripts/build_capture_mesa.sh` |
| Frozen joint plan (2026-10-03T10:06:16Z, before any capture) | `scripts/prepare_power_cohort.py`, `.local-artifacts/issue-104-power-cohort-v1/` |
| Rendered smoke | `scripts/run_power_cohort_smoke.py`, `.local-artifacts/issue-104-capture-smoke-v1/`, gallery `data/issue-104-capture-smoke/index.html` |
| Engineering runs (EXPLORATORY, exposed #77 N1 branches, never scored) | `.local-artifacts/issue-104-capture-dev/summary.json` |
| Tests | `tests/test_issue_104_power_cohort.py`, `tests/test_native_segment_trace.py` |

## 1. #113 capture-window spec (merged into the plan)

Evidence: every native step of 498 retained single shots (#109 smoke, #77 N2 appearance, #77 n2n).

- At the 12 s cap, the structure is at engine rest in 282 of 284 censored shots. The only moving body is the
  launched bird rolling on the ground (281 / 284). A longer cap would add frames of a lone rolling bird:
  +23 % frames for 16 s, +102 % for 30 s.
- Shots that reach engine rest (20–48 % per level type) stop on the first rest step, so they have 0 settled
  frames after it. That is why #100 saw 25 steady-state frames in 24 731.
- Structure changes end well before the cap. Collapse yield (22–36 % of shots per level type) depends on
  whether the shot reaches the structure, not on the window.

**Rule.** Keep the cap at 30 000 native steps for shots that are not at rest. After `stable_entered` at or
before the cap, record 50 more frames (2 500 steps) and finish with `rest_tail_complete`. `stable_exited`
cancels the tail. `level_clear` and `level_fail` still finish immediately. The rule is engine-side
(`NOVPHY_NATIVE_MAX_SHOT_STEPS`, `NOVPHY_NATIVE_REST_TAIL_STEPS`), and every manifest declares
`maximum_shot_steps` and `rest_tail_steps`. Python readers read these declarations; older traces default to
(30000, 0). The spec does not freeze #113's item-1 predicate definitions.

## 2. Power analysis (work item 1)

The analysis uses the #96 member-level records. It is DESCRIPTIVE and based on development variance. The
design count of required mixed members is the larger of two estimates: the bootstrap half-width scaled by
1/√m, and (1.96·sd/δ)².

| inventory | primary J − F*: members for δ = 0.02 / 0.05 | max over the #96 contrast set, δ = 0.02 / 0.05 | mixed share (development / held-out) |
| --- | --- | --- | --- |
| grid | 208 / 34 | 293 / 47 | 11/11, 3/4 |
| offset | 194 / 32 | 194 / 32 | 6/11, 3/4 |
| angle | 60 / 10 | 344 / 55 | 5/10, 2/4 |
| power | 224 / 36 | 328 / 53 | 5/11, 3/4 |

Depth rule, declared before any capture:

| target | δ | contrasts | candidates | depth (levels per family) |
| --- | --- | --- | --- | --- |
| grid (primary) | 0.02 | every #96 grid contrast | 16 | all 200 |
| offset, angle, power | 0.05 | every #96 contrast | union of 60 | first 60 |

- The share used is the conservative one, min(development, held-out). Depth is rounded up to 20.
- Expected mixed members: grid 300, offset 65, angle 60, power 55.

## 3. Frozen membership (49 560 branches, dispatch order)

| stream | role | levels | branches |
| --- | --- | --- | --- |
| #109 evaluation (grid; union for ordinal ≤ 60) | final_evaluation | 400 | 11 680 |
| #110 evaluation (40 pairs × 8 × 2 sides × 16) | final_evaluation | 640 | 10 240 |
| #110 fit | training (normal: zero-shot, novel: few-shot) | 640 | 10 240 |
| #109 fit (240 / family, 29 candidates) | training | 480 | 13 920 |
| #109 policy (60 / family) | training | 120 | 3 480 |

- Dispatch is ordinal-major within each stream.
- Limits: 16 workers, attempt cap 1 800 s, no retries.
- Stop rules: typed-failure share > 0.05 after 200 branches; < 500 GiB free; artifact cap 1.25 × projection;
  wall cap 302.8 h.
- Projection: 151 h, 4.5 TiB.
- The plan also binds the Gate-B unit (#112 v3: LC : F-15-macro, hybrid, E99) and its target, its
  pre-declared contrast against C*, and its two controls. It does not execute them.

## 4. Capture stack findings (engineering runs, before the freeze)

- **Renderer.** The server's Mesa 25.2.8 / LLVM 20.1.2 changes 228–872 edge pixels on every decision frame
  (0/20 equal to the retained frames). A user-local Mesa 26.1.2 / LLVM 22.1.6 reproduces the retained frames
  byte for byte. The plan pins this renderer, and any other renderer fails typed as `renderer_mismatch`.
- **Player.** On this server, the parent and window players produce identical full native prefixes on 19/20
  branches. The one remaining difference starts late in a level-clear shot.
- **Cross-machine physics.** Physics is not bit-identical across machines. From the bird's first collision
  the float digits differ from the retained traces, and event sequences agree on 16/20.
- **Window.** In single-bird levels the tail is usually cut by `level_fail` once the launched bird dies
  after rest.

## 5. Rendered smoke (168 branches: #110 smoke gate 160 + 8 exposed replays)

Captured once each in 35.5 min on 16 workers.

| gate | result |
| --- | --- |
| S1 capture reliability | 0 / 168 typed failures |
| S2 renderer | 168 / 168 on the frozen renderer |
| S3 pixel identity | 8 / 8 replay decision frames equal the retained #109 frames outside Bird/Pig boxes |
| S4 window rule | 0 violations: 89 censored at the cap, 22 `rest_tail_complete`, 35 clear/fail inside a tail, 22 clear/fail before rest |

**Campaign launch: `supported`.**

#110 per-template criteria: all 80 templates pass, so **45 / 45 cells are supported**. Every template had 0
typed failures, its authored novelty entity in the trace, the expected gravity ((0, 6) on level 6), and a
leftward launch on the right slingshot. The residual-acceleration check:

| templates | force sign | meaning |
| --- | --- | --- |
| magnet, turbulence (incl. inverse) | agree | labels match the observed acceleration |
| fan (level 2), 4 templates | disagree | median projection 0 or −100 on the labelled direction; the fan label's sign or zone needs repair |
| others | not exercised | no contact-free labelled step |

Under the #110 rule, the fan result is a labeler disagreement, repaired in the derivation before any
training; it does not make the cell unsupported. Storm onset fell inside the window on 7 of 10 storm branches.

Settle and collapse per novelty level, and the full template table, are in `findings.md`. The tail produced
917 rest frames over 80 level-0 shots (27 shots with rest), against at most 1 per shot under the old window.

Throughput: 12.7 s/branch amortized (single pass, including the cold-start ramp), branch wall 191.5 s,
88.6 MB/branch. Re-projected campaign: 174 h, 3.99 TiB, inside the frozen caps.

## Deviations and disclosures

- `SERVER_SETUP.md` byte-rewrote `/p/Project/NovPhy` in frozen records, so the #109 sealed-cohort
  `--validate` fails on its #96 input hash. Inputs here are bound to the current bytes, and every level file
  is checked against its frozen sha256 at dispatch.
- After the capture pass, `run_power_cohort_smoke.py` was changed only in its publish path: analysis and media
  now run in a process and thread pool. `run-manifest.json` keeps the runner hash from capture time. The
  capture code, membership and gates did not change.
- `findings.md` of the plan was regenerated once from the frozen `plan.json`, minutes after the freeze and
  before any capture. Only the rendering changed (key order); the plan bytes did not.
- The #85 single-RTX-3090 hardware gate is carried as a single-GPU gate on the 4 × RTX 5090 server. Capture
  uses no GPU.

## Reproduction

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate novphy && source env.sh
python -u -m scripts.derive_capture_window_spec --validate
python -u -m scripts.prepare_power_cohort --validate
python -u -m scripts.run_power_cohort_smoke --validate
python -m pytest -q tests/test_issue_104_power_cohort.py tests/test_native_segment_trace.py
```
