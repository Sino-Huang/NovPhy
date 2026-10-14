# Issue #108 Gate B: the #112 v3 unit on the #104 sealed evaluation split

Plan frozen 2026-10-14T09:36:20Z after a passed smoke, before any evaluation score; scored once in a process that did not train, under `deterministic_scoring()`. Validation: `python -u -m scripts.run_gate_b --validate`.

**Gate B: pass.** Unit LC : F-15-macro (hybrid, E99), pooled grid AUC **0.7444 [0.7269, 0.7620]** over 933 cells (311 levels x 3 seeds); target >= 0.65 with lower bound > 0.55.

## Evaluation set (stream `issue109-evaluation`)

- 400 levels, 11680 branches, 11680 complete; typed failures 0, incomplete branches 0.
- Mixed-verdict cells (levels with the inventory): grid 311 (400), offset 76 (120), angle 78 (120), power 47 (120).
- Anchors: E99 batch-1 parse of shot frame 0 of each level's first complete candidate in suffix order (the training rule).

## Grid (Gate B)

| ranker | pooled AUC | seed 20260908 | seed 20260909 | seed 20260910 | tied pair share | nonfinite share | typed failures |
| --- | --- | --- | --- | --- | --- | --- | --- |
| unit LC : F-15-macro | 0.7444 [0.7269, 0.7620] | 0.7141 [0.6917, 0.7362] | 0.8011 [0.7826, 0.8199] | 0.7181 [0.6961, 0.7404] | 0.0000 | 0.0000 | 0 |
| C* LC : C-F-15 | 0.7834 [0.7676, 0.7992] | 0.8262 [0.8091, 0.8431] | 0.7334 [0.7127, 0.7536] | 0.7905 [0.7688, 0.8120] | 0.0000 | 0.0000 | 0 |
| no-model action prior | 0.7203 [0.6965, 0.7442] | seed-free | seed-free | seed-free | 0.0022 | 0.0000 | 0 |

- Pre-declared contrast unit − C* (paired by level and seed): -0.0390 [-0.0442, -0.0338] over 933 cells, 311 levels; not_supported_by_this_experiment.
- Unit − no-model action prior (paired): 0.0241 [-0.0008, 0.0487]; readiness_or_precision_insufficient. The prior ranks by each grid action's fit-split success frequency (action overlap 1.00).
- Wrong-anchor control of the unit (480 fit-split anchors per cell): right 0.7444 [0.7269, 0.7620], wrong 0.6122 [0.5966, 0.6275], right − wrong 0.1322 [0.1216, 0.1430] (supported); rank stability (Spearman right vs wrong) 0.5918 [0.5802, 0.6035].

## Other inventories (DESCRIPTIVE; not part of the gate)

| inventory | cells | unit | C* | action prior (overlap) | unit − C* |
| --- | --- | --- | --- | --- | --- |
| offset | 228 | 0.6670 [0.6503, 0.6827] | 0.7748 [0.7524, 0.7962] | 0.5000 [0.5000, 0.5000] (0.00) | -0.1078 [-0.1217, -0.0937] |
| angle | 234 | 0.6791 [0.6663, 0.6912] | 0.7590 [0.7422, 0.7752] | 0.8119 [0.7778, 0.8450] (1.00) | -0.0799 [-0.0929, -0.0673] |
| power | 141 | 0.8058 [0.7729, 0.8396] | 0.8068 [0.7711, 0.8440] | 0.5000 [0.5000, 0.5000] (0.00) | -0.0011 [-0.0096, 0.0072] |

## Guards

- G1_isolation: True
- G2_scoring_policy: True
- G3_chronology: True
- G4_checkpoint_binding: True

Compute: 64 GPU s, 64 wall s (1 x RTX 5090 (one scoring process)); 0 engine seconds; 0 training.
