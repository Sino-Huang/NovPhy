# Legacy Temporal Pair Grid (Todo 8)

Todo 8 is the first real-data Phase-A projection of the JEPA backbone. It is a
teacher-forced, continuous-only experiment over the fixed requested horizon
grid `delta={1,5,15}`. The abstraction axis is intentionally closed to
`continuous`; `micro` and `macro` are declared unavailable with reason
`symbolic_supervision_unavailable` because the legacy RGB catalog has no engine
symbol supervision.

## Data and partitions

The dev catalog is built in a fresh process from the legacy RGB root. Its plain
`episode-catalog-v1` identity serializes the declared cohort, collection-plan,
split, and capture-contract fields. The catalog has 463 accepted episodes, 5,556
shots, 562,515 frames, and 1,137 rejected `missing_artifact` episodes. Episodes
are deterministically partitioned into
`controller-train`, `calibration`, and `evaluation`. Motion regimes are
calibrated from target-aware delta-15 shot motion using deterministic P50/P90
thresholds; the calibration metadata and exact assignments are serialized in
the sweep manifest.

## Training identity

Both fresh CUDA runs use seed `20260807`, 3,600 updates, batch size 64, learning
rate `3e-4`, weight decay `0.05`, warmup `0`, gradient clipping `1.0`, EMA base
momentum `0.996`, device `cuda`, and the approved grid above. Artifacts use plain
declared identities: the config identity serializes those Phase-A fields and the
grid identity; the grid identity serializes the grid version, approved deltas,
continuous abstraction, and exclusions; and the real-run identity serializes
the catalog, config, grid, and model-config identities.

Each run records exactly 1,200 updates per delta and exactly 400 updates per
`(delta,motion_regime)` key. The checkpoint identity is declared from the run
identity and completed step; checkpoint metadata binds config, grid, catalog,
run, and model-config identities.

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
partitions, state/score counts, the state-set identity recomputed from sorted
state identities, checkpoint/catalog/config/partition bindings, terminal
metadata, finite numeric values, and the explicit unavailability records before
frontier publication.

## Claim boundary

The frontier is a temporal prediction-versus-compute witness for the legacy
continuous carrier. It is not an oracle-symbol result and does not establish
ADE/FDE, final-state accuracy, event F1, physical-violation rates, or a learned
joint controller. Those metrics remain unavailable until an enriched
`physics_capture_v1` cohort supplies symbolic and physics supervision.
