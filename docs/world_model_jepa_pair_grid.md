# Legacy Temporal Pair Grid (Todo 8)

Todo 8 is the first real-data Phase-A projection of the JEPA backbone. It is a
teacher-forced, continuous-only experiment over the fixed requested horizon
grid `delta={1,5,15}`. The abstraction axis is intentionally closed to
`continuous`; `micro` and `macro` are declared unavailable with reason
`symbolic_supervision_unavailable` because the legacy RGB catalog has no engine
symbol supervision.

## Data and partitions

The dev catalog is built in a fresh process from the legacy RGB root. Its
identity is catalog digest
`8265809a528e41eaae646cb1cae9d577d7f34fd99b85b859bb14f07a479c6beb`, with 463
accepted episodes, 5,556 shots, 562,515 frames, and 1,137 rejected
`missing_artifact` episodes. Episodes are deterministically partitioned into
`controller-train`, `calibration`, and `evaluation`. Motion regimes are
calibrated from target-aware delta-15 shot motion using deterministic P50/P90
thresholds; the calibration metadata and exact assignments are serialized in
the sweep manifest.

## Training identity

Both fresh CUDA runs use seed `20260807`, 3,600 updates, batch size 64, learning
rate `3e-4`, weight decay `0.05`, warmup `0`, gradient clipping `1.0`, EMA base
momentum `0.996`, device `cuda`, and the approved grid above. The matching
experiment digests are:

| Field | Digest |
|---|---|
| config | `de9197990222350d9e0413d83f7dbe0f8e263d071cb13707fd80a055a1e80e38` |
| grid | `fa8035ba918f885c1e8e4e505b25086fd2e260673862c48ea9b98491747075dc` |
| run identity | `6b1d5b18fd45175ad3a6a03f31d80b2ffdf4c624d1493ce14bc67dd05fd403b9` |

Each run records exactly 1,200 updates per delta and exactly 400 updates per
`(delta,motion_regime)` key. Checkpoint bytes are hash-bound and the checkpoint
metadata binds the catalog and run identity.

## Scoring contract

The scorer evaluates every declared state and all three requested deltas. It
uses the online encoder for `z_t`, the EMA target encoder for the stop-gradient
target, and the predictor carrier only. Inference is shot-batched with bounded
frame batches; compact per-shot latent caches avoid a state-by-state PNG scan.

For terminal-edge states, scoring clamps only the target lookup:
`effective_delta=min(requested_delta,T-t)`. Labels retain requested delta,
effective delta, target frame index, and `terminal_clamp=true` when clamping
occurred. Training windows remain strict and never use this scoring-only clamp.

The score artifact root contains canonical atomic shards, `per_pair_metrics`, a
temporal oracle ceiling, and `unavailable_metrics`. Fresh validation checks all
partitions, state/score counts, canonical state digest, checkpoint/catalog/
config provenance, terminal metadata, finite numeric values, and the explicit
unavailability records before frontier publication.

## Claim boundary

The frontier is a temporal prediction-versus-compute witness for the legacy
continuous carrier. It is not an oracle-symbol result and does not establish
ADE/FDE, final-state accuracy, event F1, physical-violation rates, or a learned
joint controller. Those metrics remain unavailable until an enriched
`physics_capture_v1` cohort supplies symbolic and physics supervision.
