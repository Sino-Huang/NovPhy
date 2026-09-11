# #76 prospective censored-segment collection and symmetric refit design

The operator approved the censored-segment revision on 2026-09-11 local time.
This new collection does not relabel, replay or fit on C1/C2/native-continuation
engineering data. All earlier failed attempts and source freezes remain intact.
Source-bound `plan.json` preparation is the executable freeze; the descriptions
below alone are not permission to launch an unfrozen plan.

## Collection membership and stopping

Use the unchanged canonical native player, original 0.0004s physics timestep,
stride-50 RGB, 250-sample lossless chunks and 30,000-step/12s maximum shot window.
Require every native sample, exact observation alignment and one actual launch.
Only `native_time_window_limit` with all 30,001 native samples and 601 scheduled
observations is eligible as a censored segment. Its raw failure and null terminal
remain unchanged. Corruption, missing frames, other failures and short prefixes
are not censored learning examples. Only exact observed endpoints are targets.

Stop an episode at actual won/lost state, the first censored segment, a capture
failure or its fixed shot allowance. A stable segment is not a win. Timeout,
capture failure and unsuccessful shot-budget exhaustion each get gameplay
success 0 and failure penalty 1; success requires observed native WON before a
timeout. Keep every assigned episode in outcome denominators, including errors
and resource-unattempted cases. Never draw a replacement.

Three new engineering lineages consume at most the three remaining capture
slots: normal type010101, type010102 and type010105, one shot each, generator
seeds 760810001–3 and Unity seeds 760820001–3. Use the corresponding exact
templates/constraint rows from C1's metadata freeze, with new generated XML and
lineage identities. These engineering assignments are excluded from all fitting
and fresh data. Charge the earlier C2/native 355.26103935716674s and 626,492,528
bytes against the same 1800s/2GiB/3GiB engineering allowance.

After that engineering pipeline passes, the operator may freeze the production
development collection: normal type010101, type010102, type010103, type010204,
type010105, 70 base lineages each. Within each family, ordinals 1–50 are training,
51–60 calibration and 61–70 model selection: 250/50/50 total. No target-novelty
data is fitted. Generator seeds are 760900001–760900350 and Unity seeds are
760910001–760910350, in family-major order. Exact template/constraint/XML,
identities, role and prior-exposure projections are frozen before rendering.
Known prior fitting/development/final metadata and all prior #76 plan lineages
are checked without opening sealed outcomes. External unregistered data is
not silently certified disjoint.

Use at most three authored red birds per episode, with fixed successive actions
(-80,+10), (-60,+45), (-10,+80), tap0ms, transport release600ms; the actual native
drag handler's hold remains one second. Single-bird levels receive one action.
The public slingshot anchor executes these actions; pig/block oracle geometry
does not choose them. No abilities, action tuning, retries or solvability filter.
Episode state/clock/history continues between shots; gaps stay unobserved.

One isolated graphics-enabled worker is selected, below the approved maximum
of four. Private runtime/display/ports/XDG; no `-nographics`. Development caps:
43,200s active wall, 12GiB aggregate RSS, 256GiB new working data, 420s per episode
and 180s per shot manifest wait. Progress shows episode/chunk/frame counts,
elapsed time, RSS, artifact bytes and ETA; resume retains completed/interrupted
attempts and never replays them. Budget-unattempted members remain explicit.

## Common representation and paired training costs, before implementation

Keep the previously declared 18-slot/236-value visual carrier plus 64-value
observation/action GRU memory. Both arms independently predict all 300 values.
The existing small RGB convolutional parser is newly fitted, with no pretrained
weights or oracle runtime inputs. Static declared slot names come from the
training/template schema, not from test-image object enumeration. Runtime
ground extensions remain in full engine evidence; predicates outside the
representable slot set must be masked, never silently declared false/valid.

Use three paired model seeds 760930001–760930003. For each seed, fit common
perception/history once and freeze the identical checkpoint for both arms.
Charge its full standalone fit/deployment cost symmetrically to each method and
report actual shared work separately. Common fitting has at most 3600 GPU-active
seconds per seed (3h total); each predictor arm/seed has at most 3600s (6h total);
controller/readiness work has at most 1800s per arm/seed (3h total). Do not borrow
unused time from another arm or choose a favorable seed. Hard-limit interruptions
remain incomplete fits, not extra optimization or resumable new allowances.

Perception uses RGB64x96, fixed image/127.5-1 scaling and the existing 128-wide
slot parser; all weights train from scratch. The common history auxiliary
objective predicts the next actually observed visual carrier, conditioned on
its observed time difference, using only causal earlier observation/action
memory. Auxiliary readout weights are training-only, but their fitting cost
is included. No engine hit count, future observation or symbolic mode is a
common encoder input. The pure arm has no symbolic modules or mode embedding.

Native horizons are 50/250/750 steps (20/100/300ms); equal-time evaluation is at
11,250 steps (4.5s), only where observed endpoints exist. Local and recursive
losses mask unavailable endpoints. Both arms use paired data/update/horizon
exposure; symbolic losses/adapters exist only in the hybrid arm. Width selection
uses only the existing parameter-count rule, not outcome search. Matching total
parameters does not assert equal active MACs, latency or deployment resources.

Concrete optimizer counts, data transformations, controller objectives and
readiness thresholds must be source-bound in the fitting plan before any real
fit. Test fixtures/synthetic optimizer smokes are not research fitting and are
kept outside candidate checkpoints. No fresh inventory or final outcome access
is needed for these implementation tests.

## Later scientific gates

A collection pass means usable engineering/data transport, not gameplay success,
power or model superiority. Production data and fitted weights are prerequisites,
not a supported candidate. Before fresh access, verify candidate/baselines,
legal search/re-observation semantics, complete training/deployment cost, useful
mode/gameplay/validity gates and durable source/checkpoint/repair/player archival.
Fresh numerical margins, confidence/multiplicity/decision hierarchy and paired
clustered power must be frozen with actual candidate/development evidence.
Normal/novel variants share their lineage partition/analysis cluster. Zero-shot
and few-shot protocols remain separate; deferred cells remain untested.
#64/#65 stay unauthorized until an exactly supported disposition passes every
required gate. Missing archival destination cannot be replaced with a flag.
