# #76 prospective native training-footprint amendment

The first refit dry-run failed before any research or synthetic optimizer ran:
the frozen350-member generator schedule exceeds the earlier18-slot visual
schema. Training metadata alone contains `block:0005`, `block:0006`,
`platform:0006` and `platform:0007`, affecting41 training lineages. All calibration
and model-selection authored slots are contained in that training-derived union;
neither role adds vocabulary. No image inspection or outcome screening selected
these slots, and no assigned seed/layout/role is removed or replaced.

This amendment supersedes only the18-slot/236-visual/300-total dimensions and
associated numeric parameter/MAC counts in the earlier censored-development and
native-refit design documents. Their training roles, seeds, updates, objectives,
stopping/failure/censoring rules and resource limits remain unchanged. The C4
capture source/player/protocol and its already completed engineering evidence
remain immutable; this is a separately versioned unfitted representation.

The new schema has22 slots: three birds, seven blocks, one pig, eight platforms,
one slingshot and two landscape slots. Thus the visual carrier is2+13*22=288
values and the shared64-value observed-action memory gives352 total values.
Both independent predictors receive and predict exactly this same state.
Earlier236/300-value models/checkpoints are neither resized nor reinterpreted.

Metadata-only module construction established these costs before implementation:

| Module | Trainable parameters |
| --- | ---: |
| Common visual-only convolutional parser | 241,418 |
| Common observation/action GRU | 69,504 |
| Training-only next-visual auxiliary readout | 101,952 |
| Common deployed representation | 310,922 |
| Common fitting, including auxiliary readout | 412,874 |
| Hybrid predictor, width384 | 2,120,750 |
| Independent pure predictor, width480 | 2,139,040 |

Pure width480 is selected by the same nearest-total-count rule over384..512,
step8; its difference is0.8625%, below2%. No dead padding, transferred hybrid
weights or outcome-based architecture sweep. Shared fitting/deployment costs
are charged identically to both methods. The common transition-MAC normalization
reference becomes2,063,232, the full-width hybrid micro transition; it remains
one reference for both arms, not separate per-arm normalization. Continuous and
macro hybrid transitions have1,521,536 and1,568,640 linear MACs, respectively;
these are not complete FLOP or matched-latency claims.

The original common encoder keeps its default236-value interface; the native
352-state successor supplies an explicit288-value visual dimension. Shared
history masks, causal ordering and episode resets remain unchanged and are
regression-tested for both dimensions. Static shape labels use enabled collider
geometry when a body position is absent, without exposing that geometry as a
runtime agent input. Native contact/support and macro labels remain hybrid-only
training supervision, not common perception/history features.

All changes stay within the already approved3h common+6h predictor+3h controller
GPU-active ceilings,12h collection and256GiB working-data cap. A larger state is
not permission to extend a job, select favorable seeds or open fresh data.
The broader fresh/novel population still requires its own exact supported
footprint, viable candidate, archived assets and statistical freeze.
