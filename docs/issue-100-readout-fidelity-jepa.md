# Issue #100 — description levels the model can actually read: readout fidelity and a jointly trained JEPA encoder

Parent: #108. Shared rules of #85 apply. Prior dispositions (#15–#99, #109) are read-only inputs;
nothing here edits the ICLR 2026 submission.

| Version | Runner | Artifacts |
| --- | --- | --- |
| v1 (frozen 2026-09-27T07:09:13Z, before training and scoring) | `scripts/run_readout_fidelity_jepa.py`, `world_model/training/jepa_slot_encoder.py` | `.local-artifacts/issue-100-readout-fidelity-jepa-v1/` |
| v2 guard revision (terminal; frozen 2026-09-27T11:03:45Z, **after** v1) | `scripts/run_readout_fidelity_guard_revision.py` | `.local-artifacts/issue-100-guard-revision-v2/` |

Validation (all exit 0, `conda activate novphy && source env.sh`):

```
python -u -m scripts.run_readout_fidelity_jepa --validate
python -u -m scripts.run_readout_fidelity_guard_revision --validate   # also runs the v1 --validate
pytest tests/test_issue_100_readout_fidelity.py
```

Zero engine seconds: every frame and verdict is a retained #77/#87/#89/#93/#94 capture. Every interval is
a lineage- or member-clustered percentile bootstrap (10 000 draws, PCG64 7201), DESCRIPTIVE. Review
gallery (engine vs decoded contact/supports/macro on held-out frames per phase):
`data/issue-100-readout-fidelity/index.html`.

## Encoders

| | encoder | predictor | controller |
| --- | --- | --- | --- |
| F77 | frozen issue-70 parser v6 (96x64, 290 702 params) | frozen #77 hybrid (1 837 690) | frozen #77 |
| E99 | frozen #99 spatial slot parser (320x240, 506 414) | #99 E hybrid (#77 recipe, 745 fit shots) | trained here, #77 recipe |
| JEPA | context encoder (spatial slot parser architecture, 506 414) + EMA target encoder (0 trainable) | CNNHybridPredictor, trained jointly | trained here, #77 recipe |

JEPA recipe (`jepa_slot_encoder.py`): #77 training_loss structure with prediction targets from the EMA
target encoder (stop-gradient, momentum 0.996 → 1.0); micro and macro head losses on context and
predicted carriers for every request; the issue-70 parser loss on the context frames (keeps the fixed
236-value carrier layout readable by costs and metrics); 9000 steps x 64 windows, the #77 update budget;
encoder lr 1e-3, predictor lr 1e-4. The carrier is built differentiably (`carrier_from_outputs`, equal
to the adapter within 2.4e-5, guard G6), so predictor gradients reach the encoder. Inference wraps the
context encoder in the standard adapter.

## Dispositions

| question | v1 | v2 (terminal) |
| --- | --- | --- |
| Q2 readout target: macro F1 ≥ 0.7 and micro edge F1 ≥ 0.5 on held-out lineages, some encoder | readiness_or_precision_insufficient (G1, G3) | **not_supported_by_this_experiment** |
| Q4 J − T, J − D per inventory, better encoder (E99), held-out, tie-free | readiness_or_precision_insufficient (G1, G3) | grid J−T and J−D, offset J−D, angle J−T **supported**; the other four readiness_or_precision_insufficient |

v2 replaces exactly the two guards that failed for float-noise reasons:

- **G1** (audit carriers = #77 shard carriers within 1e-3): every exceedance was a motion value, center
  noise from a different parse batch divided by the frame interval (worst 0.036 on the few-ms terminal
  frame of an early-ending shot). v2: non-motion values within 1e-3 (observed 7.2e-4), motion within
  2e-3 / elapsed (observed scaled 1.5e-5).
- **G3** (F77 arms = #96 count costs within 1e-3): the run process had cuDNN autotuning on (set for JEPA
  training), so anchor parses used differently rounded kernels; Δ = 1 rollouts over 225 steps and the
  count cost's 1000x pig weight amplified it (worst 87.7). v2: a fresh deterministic process reproduces
  all 34 242 #96 candidate costs within 4.8e-7. The v1 records stay the scored records; their per-cell
  count-cost AUC differs from #96 on 0–9 of 114 mixed cells per arm (F-15 and T arms: 0). A post-v2
  deterministic recomputation of the E99 held-out arms (diagnostic, not scored) returns identical Q4
  estimates except angle J − D (0.0556 → 0.0694 [0.0000, 0.1389]; token unchanged).

## 1. Probe reproduction (F77, seed 20260908, capture issue-77-n1-001-a06)

| claim (2026-09-25 probe) | observed | verdict |
| --- | --- | --- |
| bird presence < 0.5 at frame 80 | 0.231 / 0.093 / 0.168 (engine: bird:0000 present) | reproduced |
| pig presence < 0.5 at frame 80 | 0.056 (engine: present) | reproduced |
| gated micro never above 0.059 over 555 frames | max 0.0589 (frame 101) | reproduced |
| macro at frame 80: steady 0.188, unstable 0.041 | 0.649, 0.110 | **corrected** |
| only lineage 009 has gated micro edges > 0.5 | 009 only (max 0.983; next 012 at 0.409) | reproduced |
| lineage 009 positions mislocalised | mean center error 0.091 on its edge frames | reproduced |

The probe's macro numbers come from a single-frame carrier without a prior frame (v2 recomputes
0.188 / 0.041 exactly); the deployment carrier, which carries motion from frame 79 as in training,
reads 0.649 / 0.110. The micro maximum (0.0589) is identical across the three seeds; [INFERENCE] the gate
multiplies by the parser's presence of both slots, which caps the value regardless of the head.

## 2. Readout fidelity (held-out lineages 007/008/014/016, 52 shots, encoder readout, all phases)

| predicate | F77 | E99 | JEPA |
| --- | --- | --- | --- |
| presence: bird recall | 0.000 | 1.000 | 1.000 |
| presence: pig recall | 0.000 | 1.000 | 0.917 |
| presence: block F1 | 0.531 | 0.635 | 0.730 |
| contact F1 | 0.000 | 0.988 | 0.897 |
| supports F1 | 0.000 | 0.990 | 0.901 |
| steady-state F1 | 0.005 | 0.000 | 0.000 |
| structure-unstable F1 | 0.000 | 0.000 | 0.000 |
| **macro F1 (target 0.7)** | 0.0025 [0.0012, 0.0045] | 0.0000 | 0.0000 |
| **micro edge F1 (target 0.5)** | 0.0000 | 0.9886 [0.9718, 0.9995] | 0.8988 [0.8113, 0.9723] |

- Micro is readable once perception is fixed: both 320x240 encoders clear the micro target; F77's gate
  multiplies by a near-zero carrier presence and reads nothing.
- Macro is not readable by any encoder. The engine rarely labels either predicate positive on these
  frames: steady-state is true on 25 of 24 731 available held-out frames (88 of 55 589 on fit lineages),
  structure-unstable on 115. E99 and JEPA never predict a macro positive; F77 predicts steady-state on
  9 930 held-out frames (precision 0.0025), inherited from its synthetic-lineage training.
- Phases: held-out audited frames split 11 617 pre-contact / 1 070 cascade / 1 settled (only 7/52 shots
  have a bird contact with the structure; the ground is not a labelled slot). Contact F1 in the cascade:
  E99 0.951, JEPA 0.852.
- After one Δ ∈ {1, 5, 15} prediction the all-phase readouts match the encoder readouts to within 0.01 F1
  for every encoder: the predictor preserves what the encoder makes readable and adds nothing.
- JEPA − E99 micro edge F1: −0.090 [−0.163, −0.024]; macro equal (both 0).

The full table (presence per kind, four predicates, encoder and 9 post-prediction requests, 4 phases,
fit and held-out, precision/recall/F1) is in v1 `findings.md` and `comparisons.csv`.

## 3. JEPA versus the frozen encoders: rollout error (#77 diagnostic protocol, held-out, t = 225)

| request | F77 own-parse | E99 own-parse | JEPA own-parse | F77 engine | E99 engine | JEPA engine |
| --- | --- | --- | --- | --- | --- | --- |
| (1, cont) | 3141 | 3644 | 4527 | 2479 | 3773 | 4690 |
| (5, cont) | 0.200 | 0.276 | 0.834 | 0.368 | 0.343 | 0.872 |
| (15, cont) | 0.0296 | 0.0158 | 0.0517 | 0.166 | 0.060 | 0.072 |
| (15, micro) | 0.0313 | 0.0150 | 0.0459 | 0.173 | 0.060 | 0.066 |
| (15, macro) | 0.0397 | 0.0101 | 0.0550 | 0.187 | 0.051 | 0.075 |

Own-parse is the tab:res:rollout estimand (F77 reproduces the published 3141 / 0.200 / 0.0296 / 0.0313 /
0.0397, guard G2, relative error ≤ 1.1e-4). "Engine" measures the same rollouts against engine-projected
centers, which does not reward an encoder for predicting its own blind parse.

- JEPA − F77 (engine, Δ = 15): −0.095 [−0.117, −0.071] (cont), −0.107 (micro), −0.111 (macro); at
  Δ = 1 and 5 JEPA is worse (e.g. (5, cont) +0.503 [0.416, 0.579]).
- JEPA − E99 (engine, Δ = 15): +0.012 [−0.017, 0.041]; at Δ = 5 +0.53 [0.40, 0.66].
- Joint training does not beat the matched frozen encoder (E99, same architecture, data and budget) on
  rollout error or readout F1.

## 4. Integrated effect of α (#96 arms rerun; better encoder by the frozen rule: E99)

| inventory | J − T | J − D | cells / members |
| --- | --- | --- | --- |
| grid | 0.1630 [0.0889, 0.2444] supported | 0.2000 [0.1556, 0.2222] supported | 9 / 3 |
| offset | 0.0222 [−0.0667, 0.1667] insufficient | 0.1556 [0.1333, 0.1667] supported | 9 / 3 |
| angle | 0.0694 [0.0278, 0.1111] supported | 0.0556 [−0.0278, 0.1389] insufficient | 6 / 2 |
| power | 0.0526 [−0.1404, 0.2281] insufficient | 0.0117 [−0.1228, 0.0877] insufficient | 9 / 3 |

Limits: 2–3 held-out member clusters per inventory (a 3-cluster percentile bootstrap has 10 distinct
resamples); no multiplicity adjustment across 8 contrasts; E99's controller was trained on the 3 #77
controller lineages only. On F77 (all members, tie-free) J − T and J − D span zero on every set except
offset J − T (−0.111 [−0.200, −0.022]).
On JEPA the description-only arm beats J on offset, angle and power (J − D −0.20, −0.23, −0.15, all
members).

## Reading

1. Micro descriptions are readable from carriers once the encoder sees the objects (E99 0.99, JEPA 0.90
   held-out edge F1); the frozen #77 carriers are unreadable at every level because the presence gate
   is near zero.
2. Macro descriptions are not readable by any encoder here, and the N1 windows barely contain the
   events: settling within 600 frames is rare. The macro target is not a perception problem on these
   captures; it needs frames where scenes settle or collapse.
3. The jointly trained JEPA encoder is comparable to the frozen spatial parser on presence (better block F1,
   lower pig recall) and loses 0.09 micro F1 and Δ = 5 rollout accuracy; the "JEPA" framing does not by
   itself buy readout fidelity.
4. With readable micro carriers (E99), the policy's joint choice beats its temporal-only and
   description-only restrictions on the drag grid and on one axis of offset and angle, on 2–3 held-out
   members. Unlike #96 on the F77 checkpoint, these intervals exclude zero; they rest on few clusters and
   need the #104 sealed cohort.

Compute: v1 12 203 GPU s (JEPA 3 x ~3190 s), wall 13 572 s; v2 397 GPU s; engine seconds 0.
Checkpoints, carriers, labels and the 25.9 GB frame cache stay local under the existing ignore policy.
