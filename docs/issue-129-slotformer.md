# Issue #129: SlotFormer-E99 (#127 arm B1) on NovPhy (reproduction, guards, and the frozen disposition)

Parent: #127 (external baselines program). Ticket: [#129](https://github.com/Sino-Huang/NovPhy/issues/129). The
binding shared rules of #85 and the #127 shared protocol (`scripts/external_baselines.py`, `eb`) apply. The Gate-B
(#108), #101, #104, #109, #112, #117 and #123 records and stores are read-only inputs. Nothing here edits the ICLR 2026
submission.

| | |
| --- | --- |
| Paper and code | Wu, Dvornik, Greff, Kipf, Garg, *SlotFormer: Unsupervised Visual Dynamics Simulation with Object-Centric Models*, ICLR 2023, arXiv 2210.05861; [pairlab/SlotFormer](https://github.com/pairlab/SlotFormer) @ `10a6d9b` (MIT); its toolbox [Wuziyi616/nerv](https://github.com/Wuziyi616/nerv) v0.1.0 @ `e83ac66` (MIT) |
| Runner | `scripts/run_issue_129_slotformer.py` (selection / selection-readouts / smoke / prepare / train / readouts / freeze-readouts / score / score-novelty / bench-train / bench-score / publish / validate) |
| Modules | `world_model/baselines/slotformer/novphy.py` (NovPhy adapter); `world_model/baselines/slotformer/vendor/` (official `SlotRollouter` / `SlotFormer` / `SingleStepSlotFormer`, the PHYRE readout `PHYREReadout`, nerv `BaseModel`, `CosineAnnealingWarmupRestarts`, `build_pos_enc`; per-file upstream and vendored sha256 in `VENDOR.json`; edits: package-relative imports and one unused-import stub, `StoSAVi = None`) |
| Fidelity | `scripts/b127_slotformer_fidelity.py` (Tier E: the official `test_vp.py`, released weights), `scripts/b127_slotformer_port_reference.py` (G-port reference and comparison) |
| Tests | `tests/test_issue_129_slotformer.py` (20 tests, CPU) |
| Artifacts | `.local-artifacts/issue-129-slotformer-v1/`: `plan.json`, `smoke.json`, `selection.json`, `decisions.md`, `findings.md`, `summary.json`, `compute.json`, `records-manifest.json`, `validation.json`, `readouts-manifest.json`, `gport/gport.json` (+ `gport/torch271/`, the earlier torch-2.7.1 comparison), `fidelity/{band,gfid}.json`, `analysis/g-action-calibration.json`, `envs/`, `ledger/`. Checkpoints, readouts, records and the external downloads (`data/external/slotformer/`, `MANIFEST.json`) stay local; records and readouts are hashed in the manifests |
| Freeze | pre-freeze implementation `1c1494ff`; `plan.json` frozen 2026-10-24T20:06:47Z (07:06 AEDT, 25 Oct), commit `bd5c5577` (sha256 `b7bc3e3a…`, 115 pins), after a passed smoke and before any main-arm training or evaluation-split read. No post-freeze amendment |
| Readout freeze | `readouts-manifest.json` frozen 2026-10-25T08:55:28Z (19:55 AEDT; sha256 `368be95c…`), before any evaluation anchor was read |
| Scoring | once, in a process that did not train: scheduler run `129-score#1`, 2026-10-25 20:00:20–20:06:48 AEDT, exit 0 (9 arm-seeds); P-novelty pass `129-score-novelty#1`, 20:10:20–20:13:31, exit 0 (no arm passed its guards; references only) |
| Validation | `python -u -m scripts.run_issue_129_slotformer --validate`, scheduler run `129-validate#1`, 2026-10-25 20:30:20–20:36:30 AEDT, **exit 0**: 22 records hash-checked, B1 seed 20260908 re-scored on every inventory with exact row equality, the publication re-derived twice byte-identically. `validation.json` binds `records-manifest.json` sha256 `a178bb1f…`. Publication commit `238ad763` |

Engine time is zero seconds. The primary metric is the Gate-B pooled grid AUC over the 311 mixed grid cells × 3 seeds
of the #104 sealed #109 evaluation split, tie-free cost, under `deterministic_scoring()`. Intervals are level-clustered
percentile bootstraps (10 000 draws, PCG64 7201) and are DESCRIPTIVE. The margin is δ = 0.02.

## Outcome

**Disposition 4: per-contrast tokens only. No scoped ranking claim either way.**

- Every global guard passes: G-isolation, G-chronology (including the novelty pass), G-scoring-policy and G-records.
  G-port and G-fid pass.
- B1 (primary, program set) and B1-m (matched secondary) fail **G-action on every seed**: endpoint action sensitivity
  0.0122–0.0138 (B1) and 0.00007–0.00026 (B1-m) against the frozen 0.05. Every other arm guard (G-rank, G-nonfinite,
  G-trained) passes on every seed.
- By the frozen rule every token involving B1 or B1-m is `readiness_or_precision_insufficient`. Neither disposition 2
  ("U ranks launches better than B1 (scoped)") nor disposition 1 applies, and B1 vetoes the #127 program-level claim
  (guard failed).
- Descriptively, B1 ranks the grid at 0.5476 [0.5264, 0.5692] against U's 0.7444 [0.7269, 0.7620]; this carries no
  token.
- The G-action threshold was frozen on #123's latent states. A fit-split calibration (descriptive, after the score,
  changes nothing) shows that the engine truth itself reaches only 0.0395 in carrier space (see "G-action calibration
  context"). The frozen guard, its failure and the disposition stand.
- Exploratory (no disposition): SlotFormer's own PHYRE success head trained on **U's** predicted fit endpoints
  (U+head) gives U+head − U = +0.0679 [+0.0583, +0.0773], `supported`.

## Design summary

The design is frozen in `plan.json`; every implementer decision was posted on #129 before the step it governs and is in
`decisions.md`.

**Model.** The official `SingleStepSlotFormer`, SlotFormer's PHYRE planning model (burn-in 1, iterative overlapping
over a 6-frame `cond_len` window, sine temporal P.E., no slot P.E., pre-norm Transformer encoder), used unchanged
through the vendored code. The adapter only fixes the data plumbing
([adapter decision](https://github.com/Sino-Huang/NovPhy/issues/129#issuecomment-6382359850)):

- tokens = the 18 E99 vocabulary slots × 13 `ENTITY_FEATURES` (carrier columns 2–235), raw; absent objects stay as slots;
- the 2 header columns are not tokens; a predicted carrier gets the header (1, 0.02) of every non-anchor fit carrier;
- the launch is a 19th token `[launch5, 0 × 8]` at t0 (SlotFormer's PHYRE recipe: inject the intervention, then roll
  out); its target at every step is the same token;
- frames 0, 15, …, 225 (1 burn-in + 15 rollout steps); shots ending before frame 225 (601 of 13 919) use the official
  PHYRE `fix_video_len` padding and the `vid_len` loss truncation (1.1 % of target frames masked);
- loss: the official slot reconstruction MSE; optimizer: the official recipe (Adam, lr 2e-4, weight decay 0, no
  clipping, nerv `CosineAnnealingWarmupRestarts` from lr/100 with 5 % warm-up, per step), batch 64, fp16 autocast with
  `GradScaler` (the official `--fp16`);
- score: tie-free `EndpointCosts` on the predicted frame-225 carrier, no readout.

**Arms** (fit split only, seeds 20260908/09/10):

| arm | model | schedule | role | trainable parameters |
| --- | --- | --- | --- | --- |
| **B1** | native PHYRE encoder: d_model 256, 8 layers, 8 heads, ffn 1024 | selected `C3-s15-long`: stride 15, lr 2e-4, 250 epochs = 54 250 × 64 | primary, program set | 6 325 005 |
| B1-m | d_model 136, 8 layers, 8 heads, ffn 544 | U's B1 recipe 9 000 × 64, stride / lr of the selected configuration | secondary (matched budget and parameters; U 1 837 690, −2.4 %) | 1 793 445 |
| B1-head | SlotFormer's PHYRE readout (`PHYREReadout`; d_model 128, 4 layers, 8 heads) on [anchor, B1's predicted frame-225 state] | official readout recipe: Adam 1e-3, 10 % warm-up, batch 256, 50 epochs = 2 700 steps, BCE on the fit verdicts | exploratory native success head | 811 649 |
| U+head | the same head on U's predicted fit endpoints (Gate-B unit, the request's fixed pair from each fit shot's level anchor under its executed launch) | as B1-head | exploratory control | 811 649 |

**Selection** ([decision](https://github.com/Sino-Huang/NovPhy/issues/129#issuecomment-6382590719); #127 rule:
policy-split pooled grid AUC of the primary score, 94 mixed policy cells, seed 20260908, full budget each; the selected
run is reused as main seed 20260908):

| configuration | stride (rollout steps) | lr | epochs (steps) | policy grid AUC |
| --- | --- | --- | --- | --- |
| C1-s15 | 15 (15) | 2e-4 | 50 (10 850) | 0.4361 |
| C2-s5 | 5 (45) | 2e-4 | 50 (10 850) | 0.4744 |
| **C3-s15-long** | 15 (15) | 2e-4 | 250 (54 250) | **0.5505** (selected) |

The native schedule is SlotFormer's PHYRE schedule in epochs (50) scaled to the 13 919 fit sequences; the third
configuration (5 × longer) addresses under-training, the most likely objection to a retrained baseline on a fit split
about 28 × smaller than SlotFormer's PHYRE training set. U had no policy-split tuning of this kind (#127), disclosed.

## G-fid and G-port

**G-fid, Tier E: PASS.** The released weights (`slotformer_clevrer_params/model_80.pth`, frozen SAVi decoder
`stosavi_clevrer_params/model_12.pth`, released `clevrer_slots.pkl`) evaluated once by the unmodified official
`slotformer/video_prediction/test_vp.py` give **CLEVRER mIoU 49.42** (49.4203) against the README's post-fix 49.42
(PR #7 / #8) and the band [48.92, 49.92] frozen before the run
([band](https://github.com/Sino-Huang/NovPhy/issues/129#issuecomment-6382402612); `band.json` sha256 `02ff5b00…`).
Other metrics of the same run: ARI 0.6345, FG-ARI 0.6300, PSNR 30.54, SSIM 0.8996, LPIPS 0.0996, AR 0.5314. The
official stack ran unmodified in env `b127-slotformer` (torch 2.7.1 + cu128, RTX 5090), with a raising `phyre` import
stub ([P-route](https://github.com/Sino-Huang/NovPhy/issues/129#issuecomment-6382402424)). The optional Tier T (PHYRE
planning AUCCESS 82.0 ± 1.1) was not run.

**G-port: PASS under #123's exact rule, bit-identical.** On fixed CPU float32 inputs, six arms (the official PHYRE
configuration, native stride 15, matched stride 15, native stride 5, the official readout configuration and the #129
head) were compared with the official sources run on the **same torch build as the adapter** (2.13.0 + cu130, CPU):
forward, loss, gradients and two official optimizer steps (step-0 LR and the post-warm-up LR). Every delta is exactly
0: loss, gradient tensors, Adam moments, predictions, parameters (|Δ| ≤ 1e-7 share 1.000000, max |Δ| 0); parameter
counts, optimizer and scheduler are identical, and the adapter's packed batches equal the official `fix_video_len`
batches. See "Deviations and disclosures" for the history of the reference torch.

## Results

### Guards

| arm | seed | G-rank (mean tied-pair share) | G-action (≥ 0.05) | G-nonfinite | G-trained | policy grid AUC (fit/policy only) |
| --- | --- | --- | --- | --- | --- | --- |
| B1 | 20260908 | pass (0.0009) | **fail (0.0130)** | pass (0.0) | pass | 0.5505 |
| B1 | 20260909 | pass (0.0007) | **fail (0.0138)** | pass | pass | 0.4439 |
| B1 | 20260910 | pass (0.0009) | **fail (0.0122)** | pass | pass | 0.5800 |
| B1-m | 20260908 | pass (0.0251) | **fail (0.00007)** | pass (0.0) | pass | 0.6665 |
| B1-m | 20260909 | pass (0.0183) | **fail (0.00026)** | pass | pass | 0.4721 |
| B1-m | 20260910 | pass (0.0070) | **fail (0.00019)** | pass | pass | 0.4665 |

G-collapse and G-readout do not apply (carrier-space model, no readout). G-port and G-fid pass. Global guards: G-isolation,
G-chronology (plan 2026-10-24T20:06:47Z ≤ readouts 2026-10-25T08:55:28Z ≤ evaluation anchors 09:00:46Z ≤ first record
09:01:46Z; novelty anchors 09:11:11Z after the last primary record 09:06:47Z), G-scoring-policy and G-records all pass.

### Grid rankers (311 mixed cells × 3 seeds)

| ranker | pooled AUC [CI] | per seed (08 / 09 / 10) | top-1 (chance 0.089) |
| --- | --- | --- | --- |
| U | 0.7444 [0.7269, 0.7620] | 0.7141 / 0.8011 / 0.7181 | 0.195 |
| C\* | 0.7834 [0.7676, 0.7992] | 0.8262 / 0.7334 / 0.7905 | 0.192 |
| P | 0.7203 [0.6965, 0.7442] | — | 0.180 |
| J | 0.7057 [0.6846, 0.7267] | 0.6690 / 0.7012 / 0.7470 | 0.160 |
| J_dec | 0.7460 [0.7277, 0.7637] | 0.7205 / 0.7933 / 0.7242 | 0.180 |
| **B1** | 0.5476 [0.5264, 0.5692] | 0.5778 / 0.4591 / 0.6060 | 0.079 |
| B1-m | 0.5348 [0.5176, 0.5523] | 0.6701 / 0.4708 / 0.4637 | 0.032 |
| B1-head (exploratory) | 0.6069 [0.5813, 0.6328] | 0.6136 / 0.5856 / 0.6216 | 0.092 |
| U+head (exploratory control) | 0.8123 [0.7948, 0.8303] | 0.7991 / 0.8155 / 0.8225 | 0.348 |

No ranker has tied pairs or nonfinite rows except P (tied-pair share 0.002).

### Contrasts (grid, paired per (level, seed))

| contrast | estimate [CI] | token | reverse token |
| --- | --- | --- | --- |
| B1 − U | −0.1968 [−0.2228, −0.1705] | `readiness_or_precision_insufficient` | `readiness_or_precision_insufficient` |
| B1 − C\* | −0.2358 [−0.2614, −0.2098] | `readiness_or_precision_insufficient` | `readiness_or_precision_insufficient` |
| B1 − P | −0.1727 [−0.1973, −0.1479] | `readiness_or_precision_insufficient` | `readiness_or_precision_insufficient` |
| B1 − J_dec | −0.1984 [−0.2254, −0.1708] | `readiness_or_precision_insufficient` | `readiness_or_precision_insufficient` |
| B1-m − U | −0.2096 [−0.2303, −0.1899] | `readiness_or_precision_insufficient` | `readiness_or_precision_insufficient` |
| B1-m − C\* | −0.2485 [−0.2687, −0.2286] | `readiness_or_precision_insufficient` | `readiness_or_precision_insufficient` |
| B1-m − P | −0.1855 [−0.2213, −0.1504] | `readiness_or_precision_insufficient` | `readiness_or_precision_insufficient` |
| B1-m − J_dec | −0.2111 [−0.2308, −0.1925] | `readiness_or_precision_insufficient` | `readiness_or_precision_insufficient` |

Every token is the stop token because B1 and B1-m fail G-action. The estimates are reported as measured; they license
no statement in either direction.

**Within family.**

| contrast | estimate [CI] | token | reverse token |
| --- | --- | --- | --- |
| B1 − B1-m | +0.0128 [−0.0196, +0.0447] | `readiness_or_precision_insufficient` | `readiness_or_precision_insufficient` |
| B1-head − B1 | +0.0593 [+0.0227, +0.0965] | `readiness_or_precision_insufficient` | `readiness_or_precision_insufficient` |
| B1-head − U+head | −0.2054 [−0.2303, −0.1797] | `readiness_or_precision_insufficient` | `readiness_or_precision_insufficient` |
| U+head − U (exploratory, global guards only) | **+0.0679 [+0.0583, +0.0773]** | **`supported`** | `not_supported_by_this_experiment` |

### Wrong-anchor control (right − wrong, 480 fit anchors, grid)

| arm | right | wrong | right − wrong | ranking stability (Spearman) | token |
| --- | --- | --- | --- | --- | --- |
| B1 | 0.5476 [0.5264, 0.5692] | 0.5541 [0.5400, 0.5682] | −0.0065 [−0.0239, +0.0113] | 0.3189 | `readiness_or_precision_insufficient` |
| B1-m | 0.5348 [0.5176, 0.5523] | 0.5252 [0.5074, 0.5433] | +0.0097 [+0.0030, +0.0160] | 0.8596 | `readiness_or_precision_insufficient` |

### Descriptive inventories (no tokens)

| inventory (cells) | U | C\* | P | J | J_dec | B1 | B1-m | B1-head | U+head |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| offset (228) | 0.6670 | 0.7748 | 0.5000 | 0.7522 | 0.6904 | 0.6259 | 0.5625 | 0.6904 | 0.8604 |
| angle (234) | 0.6791 | 0.7590 | 0.8119 | 0.7765 | 0.6988 | 0.6340 | 0.5328 | 0.7318 | 0.8666 |
| power (141) | 0.8058 | 0.8068 | 0.5000 | 0.8076 | 0.7943 | 0.5002 | 0.7184 | 0.8085 | 0.8150 |

B1 − U: offset −0.0410 [−0.0820, −0.0009], angle −0.0451 [−0.0811, −0.0100], power −0.3055 [−0.3598, −0.2524]
(stop tokens).

### Engine position error (#100 metric; DESCRIPTIVE)

Mean slot-centre MSE of the predicted carrier against the engine centres on present slots, every complete grid
candidate of every mixed grid cell, from the Gate-B anchors (member-clustered bootstrap):

| t | 15 | 30 | 60 | 150 | 225 |
| --- | --- | --- | --- | --- | --- |
| B1 | 0.0527 | 0.0526 | 0.0509 | 0.0486 | 0.0508 [0.0462, 0.0555] |
| B1-m | 0.0535 | 0.0542 | 0.0526 | 0.0481 | 0.0495 [0.0449, 0.0543] |
| U | 0.0505 | 0.0501 | 0.0484 | 0.0459 | 0.0474 [0.0440, 0.0508] |
| C\* | 0.0547 | 0.0545 | 0.0543 | 0.0594 | 0.0670 [0.0617, 0.0723] |

**Tier-C trigger note.** #127 makes DDLP a Tier-C arm "if B1 fails G-readout or its position error at frame 225
exceeds U's". B1 has no readout (G-readout does not apply). Its frame-225 position error (0.0508) exceeds U's (0.0474)
as a point estimate; the intervals overlap. The trigger condition as written is met on the point estimate; whether DDLP
is funded is decided under P-budget by the orchestrator.

### Native success heads (exploratory)

The head is SlotFormer's PHYRE readout, trained per the official recipe on the predicted fit endpoints with the engine
verdicts (838 positives of 13 919); B1-head reads B1's, U+head reads U's. Policy grid AUC (fit/policy only): B1-head
0.6166 / 0.5878 / 0.6231, U+head 0.8017 / 0.8161 / 0.8209 (seeds 08 / 09 / 10). On the sealed grid U+head − U is
+0.0679 [+0.0583, +0.0773], `supported` (exploratory within-family contrast, global guards only; not a disposition):
the PHYRE success head on U's predicted endpoints ranks launches better than U's tie-free cost on this grid. B1-head − B1
is +0.0593 (stop token, B1 guard).

### CLEVRER / Physion-Dominoes (EXPLORATORY; no ranking, no tokens)

Action-free B1 (the selected configuration's stride and lr, native size, 9 000 × 64 as #123's bench arms, null launch)
on the #117 item-5 annotation carriers (12 scenes each, fold all, in-sample); #79 `field_errors` position_mse at
t ∈ {15, 30, 60, 120} from contexts 0–3, mean over scenes × contexts × 3 seeds, beside #117 item-5 J (LC, full, fold all).

| CLEVRER | t = 15 | t = 30 | t = 60 | t = 120 |
| --- | --- | --- | --- | --- |
| B1-bench | 0.005404 | 0.004412 | 0.004158 | 0.002526 |
| J (#117 item 5, LC, full) F-15-continuous | 0.001795 | 0.002058 | 0.001214 | 0.006567 |
| J (#117 item 5, LC, full) F-5-continuous | 0.002574 | 0.004234 | 0.004886 | 0.01537 |

| Physion-Dominoes | t = 15 | t = 30 | t = 60 | t = 120 |
| --- | --- | --- | --- | --- |
| B1-bench | 0.000077 | 0.0001457 | 0.0002473 | 0.0004823 |
| J (#117 item 5, LC, full) F-15-continuous | 0.0001865 | 0.0002947 | 0.0005389 | 0.001101 |
| J (#117 item 5, LC, full) F-5-continuous | 0.0004014 | 0.0008574 | 0.001726 | 0.002862 |

B1-bench has lower in-sample position error than J at every horizon on Physion-Dominoes and at t = 120 on CLEVRER, and
higher error at t ≤ 60 on CLEVRER. These rows involve no candidate actions and rank nothing.

### G-action calibration context (DESCRIPTIVE; fit split only, measured after the score; changes nothing)

G-action (endpoint action sensitivity ≥ 0.05) was frozen on #123's latent states; #129 applies it, as frozen, to the
236-d predicted endpoint carrier. With the same `eb.action_sensitivity` call on the 480 fit anchors × 16 grid launches
(`analysis/g-action-calibration.json`, script `analysis/g_action_calibration.py`):

| source | per seed (08 / 09 / 10) | mean | fraction of engine truth |
| --- | --- | --- | --- |
| engine truth: true carriers at min(225, end) of the 16 executed grid shots of 479 fit levels | — | **0.0395** | 100 % |
| U (Gate-B checkpoints, request pair from the fit anchor) | 0.0240 / 0.0229 / 0.0220 | 0.0229 | 58 % |
| C\* | 0.0363 / 0.0304 / 0.0336 | 0.0334 | 85 % |
| B1 (frozen readouts) | 0.0130 / 0.0138 / 0.0122 | 0.0130 | 33 % |
| B1-m | 0.00007 / 0.00026 / 0.00019 | 0.00018 | 0.4 % |

The engine truth itself is below 0.05: **the frozen threshold is not attainable in carrier space even by a perfect
model.** U and C\* would also fail it in this space (they are reference rankers, not subject to it in #129). B1's
endpoints vary with the launch about one third as much as the true endpoints do (about half as much as U's), and B1-m's
are nearly launch-invariant. This is reported as context only: by the integrity rule nothing is changed after the
evaluation score, so the frozen guard, its failure and disposition 4 stand.

### P-novelty (#110 novelty pairs)

Not scored for B1 or B1-m: P-novelty scores each arm that passes its guards, and both fail G-action
(`novelty/arms.json`); U+head is a control without arm guards. The U / C\* / P novelty references were computed and are
hashed in `records-manifest.json` (`novelty/references/`), but no novelty table is published for this child.

## Disposition (δ rule)

- Disposition 1 (B1 ranks launches better than U, scoped): **no** — B1 fails G-action; B1 − U carries the stop token.
- Disposition 2 (U ranks launches better than B1, scoped): **no** — requires every B1 guard to pass.
- Disposition 3 (no AUC difference of 0.02 or more): **no** — guards fail.
- **Disposition 4: per-contrast tokens only**, every one `readiness_or_precision_insufficient`.
- #127 program claim: B1 is a program-set member that fails a guard, so it **vetoes** the intersection–union claim.
- Budget safeguard: does not fire (B1 − U is not `supported`); B1's mean training compute is 10.1 × U's.

## Deviations and disclosures

1. **Selection runs relaunched once (P-crash).** The three selection jobs first launched together on GPU 1 and died
   with CUDA out-of-memory within their first steps; nothing was written. They were relaunched once with the same
   seeds and batch streams, C1 and C3 together, C2 alone afterwards
   ([decision](https://github.com/Sino-Huang/NovPhy/issues/129#issuecomment-6382755036)).
2. **G-port criterion correction and reference torch, before the freeze.** The first G-port compared the adapter
   (torch 2.13) with the official code on torch 2.7.1 and was described as following #123's tolerance; it actually
   gated on a per-element `isclose` share, a loosened variant, and #123's exact rule failed for one arm-step
   (`b1-native-s5` step 2, share 0.99202). This was corrected on the ticket
   ([correction](https://github.com/Sino-Huang/NovPhy/issues/129#issuecomment-6383196400)). The orchestrator then
   removed the cause, not the tolerance: the frozen G-port runs the official sources on the adapter's torch 2.13 and
   gates on #123's exact rule, which passes bit-identically
   ([follow-up](https://github.com/Sino-Huang/NovPhy/issues/129#issuecomment-6383281200)). The torch-2.7.1 comparison
   is kept as a diagnostic (`gport/torch271/gport.json`).
3. **Numeric choices.** Training uses fp16 autocast with `GradScaler` (the official `--fp16`) and cuDNN benchmark (the
   official `--cudnn`); TF32 flags are left at the torch defaults during training. Every readout and scoring path runs
   under `deterministic_scoring()` (no TF32, deterministic cuDNN, float32).
4. **Native schedule.** SlotFormer's PHYRE config fixes 50 epochs at batch 64; its "~300k steps" holds for its own data
   size. Scaled to the fit split this is 10 850 steps; the selected configuration is the pre-declared 5 × longer
   schedule (54 250 steps).
5. **Native head input.** The official readout reads slots at frames [0, 3] of a 1-fps rollout; per #127 the head reads
   [anchor, predicted frame 225]. Its d_model keeps the official 128 (the official `d_model = slot_size` cannot split
   our 13-wide slots into 8 heads). It trains on all fit shots with unweighted BCE (PHYRE's `pos_ratio` is a
   subsampling rule over its 100 k action cache, not applied).
6. **Equal tuning allowance.** B1 chose among three configurations on the policy split; U had no such tuning (#127).
7. **Novelty references computed, no arm scored** (P-novelty; see above).
8. **P-novelty scoring mode, CLEVRER / Physion benchmark modes, peak-memory reset and the parameter-criterion
   correction** were added before the freeze after the pre-freeze review (Review129); the training semantics did not
   change while selection runs were in flight (the runner was edited on a copy and swapped atomically, with the training
   path byte-identical; the peak-memory reset changes only the reported `peak_gib`).

## Compute

GPU seconds are each process's wall time on its GPU (`eb.mark`, each arm-seed's training counted once from its
`report.json`). Jobs ran with `shares_gpu`, so process time on a shared GPU exceeds the device time the work would need
alone.

| item | process GPU-h |
| --- | --- |
| selection C1-s15 / C2-s5 / C3-s15-long (C3 = B1 seed 20260908) | 0.480 / 1.005 / 7.372 |
| B1 seeds 20260909 / 20260910 | 1.803 / 1.681 |
| B1-m × 3 | 1.074 |
| CLEVRER / Physion benchmarks (6 runs) | 0.981 |
| Tier-E fidelity | 0.131 |
| readouts (9 arm-seeds), scoring (9), smoke, selection readouts | 0.214 |
| **recorded** (`compute.json`) | **14.74** |
| unrecorded: score-novelty 191 s, bench-score 19 s, publish 153 s, validate 370 s, OOM launches 36 s | 0.21 |
| **active total** | **≈ 14.95 (child cap 88; smoke projection 33.5)** |

**Device vs process.** B1 seed 20260908 (the reused C3 selection run) ran at 0.489 s/step while sharing GPU 1 (7.37
process GPU-h); its siblings ran the same 54 250 steps at 0.112–0.120 s/step (1.68–1.80 h). At the sibling rate its
device-equivalent cost is about 1.75 GPU-h, which puts the child's device-equivalent total at about 9.3 GPU-h
[estimate]. The ledger keeps the process seconds.

| | B1 | B1-m | U | C\* |
| --- | --- | --- | --- | --- |
| trainable parameters | 6 325 005 | 1 793 445 | 1 837 690 | 1 862 396 |
| training GPU-s per seed (08 / 09 / 10; process) | 26 538 / 6 490 / 6 050 | 1 190 / 1 492 / 1 184 | mean 1 294 | mean 1 024 |
| inference MACs per candidate (to frame 225) | 5.69 M | 3.02 M | 20.99 M | 27.26 M |
| wall per decision (16 candidates, GPU, median) | 0.046 s | 0.038 s | — | — |

Active wall time with a #129 GPU process running was about 11.8 h over two windows (2026-10-24 20:15 → 25 05:27;
2026-10-25 18:00 → 20:37 AEDT), against the 48 h-on-4-GPUs rule. Engine seconds: 0. No scope was cut.

## Reproduction

```
conda activate novphy && source env.sh
M="python -u -m scripts.run_issue_129_slotformer"
# Phase 0
/srv/home/shua0171/miniconda3/envs/b127-slotformer/bin/python -u scripts/b127_slotformer_fidelity.py   # fidelity/gfid.json (GPU)
CUDA_VISIBLE_DEVICES="" PYTHONPATH=/mnt/array/sukaih/novphy-baselines-src/nerv@e83ac66 \
  /srv/home/shua0171/venvs/b127-slotformer-ref213/bin/python scripts/b127_slotformer_port_reference.py reference .local-artifacts/issue-129-slotformer-v1/gport
CUDA_VISIBLE_DEVICES="" python scripts/b127_slotformer_port_reference.py compare .local-artifacts/issue-129-slotformer-v1/gport   # exit 0
# selection, smoke, freeze
CUDA_VISIBLE_DEVICES=g $M --selection --gpu q        # q = 0, 1, 2 (C1, C2, C3); night window only (scheduler)
$M --selection-readouts && $M --smoke && $M --prepare
# main runs (scripts/gpu_window_issue_129_main.json)
CUDA_VISIBLE_DEVICES=g $M --train --gpu q            # q = 0..3
CUDA_VISIBLE_DEVICES=g $M --bench-train --gpu q      # q = 0, 1
CUDA_VISIBLE_DEVICES=g $M --readouts --gpu q && $M --freeze-readouts
CUDA_VISIBLE_DEVICES=0 $M --score                    # once
CUDA_VISIBLE_DEVICES=0 $M --score-novelty && $M --bench-score
$M --publish
CUDA_VISIBLE_DEVICES=0 $M --validate                 # exit 0
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 python .local-artifacts/issue-129-slotformer-v1/analysis/g_action_calibration.py   # calibration context (CPU, fit only)
CUDA_VISIBLE_DEVICES="" pytest tests/test_issue_129_slotformer.py    # 20 passed
```
