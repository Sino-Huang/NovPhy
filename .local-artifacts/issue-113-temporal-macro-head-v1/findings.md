# #113 item 3: temporal macro head (`issue-113-temporal-macro-head-v1`)

Frozen 2026-10-16T13:54:57Z (item-1 definitions frozen 2026-10-16T13:31:48Z). Zero engine seconds. Every interval is a level-clustered percentile bootstrap (10 000 draws, PCG64 11302), DESCRIPTIVE.

Selected by grouped leave-out: temporal head **K4-mix50**, single-frame head **K1-mix50** (K = stride-Delta history length; mix = share of training rows read from Gate-B one-step predictions). Current head: threshold 0.5. Thresholds per source (engine / predicted) in `selection.json`.

## Guards

- G1_carrier_replication: pass — the fresh fit parse (deterministic_scoring policy) agrees with the v3 retrain carriers on positions 0..285: >= 0.9 of rows within the scoring_harness parse tolerance (a pipeline error fails every row; the v3 data mode ran under torch's default flags, cuDNN TF32 allowed, so slots near a presence or kind boundary differ numerically: the smoke saw 95 % of rows within tolerance and one kind-index flip)
- G2_gate_b_checkpoints: pass — the Gate-B unit checkpoints match the handoff sha256
- G3_evaluation_isolation: pass — no evaluation-split trace, carrier or label was read before the item-3 freeze
- G4_frame_alignment: pass — every shot's carrier rows equal its trace-summary frames
- G5_deterministic_rescore: pass — each scoring worker's first record scored twice is identical
- G6_finite: pass — every held-out readout is finite
- G7_compute_caps: pass — summed GPU s <= 144000 and summed process wall s <= 172800

## Grouped leave-out (fit split, 5 level-grouped folds; heads seed 20260908; current head and predictor in-sample): F1 per redefined predicate

| head | source | macro F1 | settle Δ1 | settle Δ5 | settle Δ15 | structure-change Δ1 | structure-change Δ5 | structure-change Δ15 |
|---|---|---|---|---|---|---|---|---|
| K1-mix0 | engine | 0.085 [0.079, 0.090] | 0.064 | 0.063 | 0.069 | 0.037 | 0.101 | 0.176 |
| K1-mix0 | predicted | 0.019 [0.016, 0.021] | 0.021 | 0.011 | 0.013 | 0.005 | 0.024 | 0.040 |
| K1-mix50 | engine | 0.076 [0.071, 0.082] | 0.059 | 0.058 | 0.066 | 0.033 | 0.081 | 0.160 |
| K1-mix50 | predicted | 0.080 [0.076, 0.085] | 0.073 | 0.075 | 0.059 | 0.043 | 0.114 | 0.117 |
| K2-mix0 | engine | 0.108 [0.105, 0.112] | 0.068 | 0.073 | 0.080 | 0.047 | 0.145 | 0.238 |
| K2-mix0 | predicted | 0.015 [0.014, 0.016] | 0.021 | 0.001 | 0.002 | 0.005 | 0.020 | 0.040 |
| K2-mix50 | engine | 0.099 [0.096, 0.103] | 0.066 | 0.065 | 0.072 | 0.041 | 0.133 | 0.219 |
| K2-mix50 | predicted | 0.083 [0.078, 0.088] | 0.086 | 0.072 | 0.061 | 0.048 | 0.117 | 0.115 |
| K3-mix0 | engine | 0.158 [0.153, 0.164] | 0.077 | 0.137 | 0.248 | 0.052 | 0.162 | 0.274 |
| K3-mix0 | predicted | 0.014 [0.013, 0.015] | 0.018 | 0.001 | 0.000 | 0.005 | 0.020 | 0.039 |
| K3-mix50 | engine | 0.139 [0.135, 0.144] | 0.073 | 0.107 | 0.202 | 0.047 | 0.148 | 0.259 |
| K3-mix50 | predicted | 0.096 [0.091, 0.101] | 0.083 | 0.077 | 0.083 | 0.046 | 0.128 | 0.158 |
| K4-mix0 | engine | 0.194 [0.188, 0.201] | 0.087 | 0.189 | 0.374 | 0.053 | 0.179 | 0.285 |
| K4-mix0 | predicted | 0.014 [0.013, 0.016] | 0.020 | 0.003 | 0.000 | 0.004 | 0.020 | 0.039 |
| K4-mix50 | engine | 0.165 [0.159, 0.171] | 0.081 | 0.145 | 0.278 | 0.049 | 0.166 | 0.269 |
| K4-mix50 | predicted | 0.117 [0.111, 0.124] | 0.086 | 0.084 | 0.157 | 0.045 | 0.145 | 0.187 |
| as_is | engine | 0.000 [0.000, 0.000] | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| as_is | predicted | 0.000 [0.000, 0.000] | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

## Held-out (#109 evaluation split, scored once; seed mean of 3): F1 per redefined predicate

| head | source | macro F1 | settle Δ1 | settle Δ5 | settle Δ15 | structure-change Δ1 | structure-change Δ5 | structure-change Δ15 |
|---|---|---|---|---|---|---|---|---|
| temporal | engine | 0.157 [0.151, 0.164] | 0.086 | 0.145 | 0.226 | 0.050 | 0.170 | 0.266 |
| temporal | predicted | 0.113 [0.107, 0.119] | 0.069 | 0.071 | 0.149 | 0.052 | 0.140 | 0.196 |
| temporal | rollout | 0.015 [0.014, 0.015] | 0.000 | 0.007 | 0.009 | 0.000 | 0.021 | 0.050 |
| single | engine | 0.082 [0.078, 0.087] | 0.070 | 0.065 | 0.073 | 0.034 | 0.084 | 0.167 |
| single | predicted | 0.075 [0.070, 0.080] | 0.065 | 0.062 | 0.047 | 0.048 | 0.111 | 0.116 |
| single | rollout | 0.018 [0.017, 0.019] | 0.010 | 0.015 | 0.009 | 0.000 | 0.016 | 0.057 |
| as_is | engine | 0.000 [0.000, 0.000] | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| as_is | predicted | 0.000 [0.000, 0.000] | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| as_is | rollout | 0.000 [0.000, 0.000] | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

## Held-out contrasts (macro F1 difference, paired level bootstrap; margin 0.05)

| contrast | source | difference | token |
|---|---|---|---|
| temporal-as_is (temporal head - current head (Acceptance 3/4)) | engine | 0.157 [0.151, 0.164] | supported |
| temporal-single (temporal-context effect (same labels, same width)) | engine | 0.075 [0.068, 0.082] | supported |
| single-as_is (label effect (same single-frame architecture)) | engine | 0.082 [0.078, 0.087] | supported |
| temporal-as_is (temporal head - current head (Acceptance 3/4)) | predicted | 0.113 [0.107, 0.119] | supported |
| temporal-single (temporal-context effect (same labels, same width)) | predicted | 0.038 [0.034, 0.042] | not_supported_by_this_experiment |
| single-as_is (label effect (same single-frame architecture)) | predicted | 0.075 [0.070, 0.080] | supported |
| temporal-as_is (temporal head - current head (Acceptance 3/4)) | rollout | 0.015 [0.014, 0.015] | descriptive |
| temporal-single (temporal-context effect (same labels, same width)) | rollout | -0.003 [-0.004, -0.002] | descriptive |
| single-as_is (label effect (same single-frame architecture)) | rollout | 0.018 [0.017, 0.019] | descriptive |

## Grouped leave-out contrasts (descriptive)

| contrast | source | difference |
|---|---|---|
| temporal-as_is | engine | 0.165 [0.159, 0.171] |
| temporal-single | engine | 0.088 [0.081, 0.096] |
| single-as_is | engine | 0.076 [0.071, 0.082] |
| temporal-as_is | predicted | 0.117 [0.111, 0.124] |
| temporal-single | predicted | 0.037 [0.033, 0.042] |
| single-as_is | predicted | 0.080 [0.076, 0.085] |

## Item-4 readout target: held-out readout counts per phase (temporal head; seed mean)

| source | phase | predicate | Δ | engine positives | predicted positives | F1 |
|---|---|---|---|---|---|---|
| engine | all | settle | 1 | 63894 | 14039.3 | 0.086 |
| engine | all | settle | 5 | 54684 | 53116.3 | 0.145 |
| engine | all | settle | 15 | 36086 | 165933.0 | 0.226 |
| engine | all | structure-change | 1 | 16161 | 172348.3 | 0.050 |
| engine | all | structure-change | 5 | 54549 | 117285.0 | 0.170 |
| engine | all | structure-change | 15 | 101693 | 145702.0 | 0.266 |
| engine | pre_contact | settle | 1 | 0 | 5059.3 | 0.000 |
| engine | pre_contact | settle | 5 | 0 | 18514.7 | 0.000 |
| engine | pre_contact | settle | 15 | 0 | 40059.7 | 0.000 |
| engine | pre_contact | structure-change | 1 | 0 | 82676.0 | 0.000 |
| engine | pre_contact | structure-change | 5 | 0 | 30366.7 | 0.000 |
| engine | pre_contact | structure-change | 15 | 0 | 13470.3 | 0.000 |
| engine | cascade | settle | 1 | 0 | 5610.3 | 0.000 |
| engine | cascade | settle | 5 | 0 | 26191.0 | 0.000 |
| engine | cascade | settle | 15 | 0 | 101322.7 | 0.000 |
| engine | cascade | structure-change | 1 | 16157 | 88539.3 | 0.090 |
| engine | cascade | structure-change | 5 | 54491 | 85301.0 | 0.209 |
| engine | cascade | structure-change | 15 | 101556 | 128692.3 | 0.286 |
| engine | settled | settle | 1 | 63894 | 3369.7 | 0.100 |
| engine | settled | settle | 5 | 54684 | 8410.7 | 0.248 |
| engine | settled | settle | 15 | 36086 | 24550.7 | 0.754 |
| engine | settled | structure-change | 1 | 4 | 1133.0 | 0.000 |
| engine | settled | structure-change | 5 | 58 | 1617.3 | 0.000 |
| engine | settled | structure-change | 15 | 137 | 3539.3 | 0.001 |
| predicted | all | settle | 1 | 63894 | 302015.7 | 0.069 |
| predicted | all | settle | 5 | 54684 | 290233.0 | 0.071 |
| predicted | all | settle | 15 | 36086 | 75305.3 | 0.149 |
| predicted | all | structure-change | 1 | 16161 | 68155.7 | 0.052 |
| predicted | all | structure-change | 5 | 54549 | 87680.3 | 0.140 |
| predicted | all | structure-change | 15 | 101693 | 78764.3 | 0.196 |
| predicted | pre_contact | settle | 1 | 0 | 132910.7 | 0.000 |
| predicted | pre_contact | settle | 5 | 0 | 116688.0 | 0.000 |
| predicted | pre_contact | settle | 15 | 0 | 23778.0 | 0.000 |
| predicted | pre_contact | structure-change | 1 | 0 | 28691.7 | 0.000 |
| predicted | pre_contact | structure-change | 5 | 0 | 24044.7 | 0.000 |
| predicted | pre_contact | structure-change | 15 | 0 | 7161.7 | 0.000 |
| predicted | cascade | settle | 1 | 0 | 156494.7 | 0.000 |
| predicted | cascade | settle | 5 | 0 | 159906.0 | 0.000 |
| predicted | cascade | settle | 15 | 0 | 42589.0 | 0.000 |
| predicted | cascade | structure-change | 1 | 16157 | 39001.7 | 0.080 |
| predicted | cascade | structure-change | 5 | 54491 | 62853.3 | 0.170 |
| predicted | cascade | structure-change | 15 | 101556 | 70379.7 | 0.205 |
| predicted | settled | settle | 1 | 63894 | 12610.3 | 0.329 |
| predicted | settled | settle | 5 | 54684 | 13639.0 | 0.356 |
| predicted | settled | settle | 15 | 36086 | 8938.3 | 0.367 |
| predicted | settled | structure-change | 1 | 4 | 462.3 | 0.004 |
| predicted | settled | structure-change | 5 | 58 | 782.3 | 0.003 |
| predicted | settled | structure-change | 15 | 137 | 1223.0 | 0.015 |
| rollout | all | settle | 1 | 63894 | 114832.0 | 0.000 |
| rollout | all | settle | 5 | 10882 | 139055.0 | 0.007 |
| rollout | all | settle | 15 | 2394 | 148406.0 | 0.009 |
| rollout | all | structure-change | 1 | 16161 | 165919.3 | 0.000 |
| rollout | all | structure-change | 5 | 10830 | 266991.0 | 0.021 |
| rollout | all | structure-change | 15 | 6794 | 134943.3 | 0.050 |
| rollout | pre_contact | settle | 1 | 0 | 113752.3 | 0.000 |
| rollout | pre_contact | settle | 5 | 0 | 39731.3 | 0.000 |
| rollout | pre_contact | settle | 15 | 0 | 25614.3 | 0.000 |
| rollout | pre_contact | structure-change | 1 | 0 | 165174.0 | 0.000 |
| rollout | pre_contact | structure-change | 5 | 0 | 77180.3 | 0.000 |
| rollout | pre_contact | structure-change | 15 | 0 | 24164.7 | 0.000 |
| rollout | cascade | settle | 1 | 0 | 1079.7 | 0.000 |
| rollout | cascade | settle | 5 | 0 | 98527.3 | 0.000 |
| rollout | cascade | settle | 15 | 0 | 121426.0 | 0.000 |
| rollout | cascade | structure-change | 1 | 16157 | 745.3 | 0.001 |
| rollout | cascade | structure-change | 5 | 10816 | 187932.7 | 0.034 |
| rollout | cascade | structure-change | 15 | 6785 | 109374.3 | 0.064 |
| rollout | settled | settle | 1 | 63894 | 0.0 | 0.000 |
| rollout | settled | settle | 5 | 10882 | 796.3 | 0.109 |
| rollout | settled | settle | 15 | 2394 | 1365.7 | 0.369 |
| rollout | settled | structure-change | 1 | 4 | 0.0 | 0.000 |
| rollout | settled | structure-change | 5 | 14 | 1878.0 | 0.002 |
| rollout | settled | structure-change | 15 | 9 | 1404.3 | 0.003 |

Reference: the #100 macro readout bar was F1 >= 0.7; it is reported, not tested.

Compute: 23921 GPU s, 26776 s summed process wall, 120945 CPU process-s, 0 engine s.
