# JEPA Backbone (Milestone 1a + 1b)

`world_model.model` implements the BG-NS-JEPA continuous backbone and its
dual-output predictor.  `world_model.training` implements a teacher-forced
single-step training loop with a seeded reproducibility manifest.

The backbone section below documents Milestones 1a and 1b plus the exclusive
transition adapters introduced for issue #4. Todo 8 additionally
delivers the legacy temporal projection of 1e/1f, documented in
[`world_model_jepa_pair_grid.md`](world_model_jepa_pair_grid.md). It still does
**not** implement the SPSG/GINE relational encoder (1c), the macro-event
predictor or restriction/lifting maps (1d), or any learned symbolic controller.

## The state-carrier principle

The proposal's first non-negotiable is that the continuous latent `z` is the
**sole** rollout state carrier: no discrete step, graph assembly, or symbol
decode ever sits in the rollout path. Three structural guarantees implement it,
and all are pinned by tests rather than left to convention.

1. `DualOutputPredictor.carrier()` computes `ẑ` without touching either mode
   head, so no head parameter can influence the rollout state.
   `test_the_carrier_is_graph_independent_of_the_mode_heads` asserts that
   `autograd.grad(carrier.sum(), head_parameters)` is all-`None` for every
   `(Δ, α)` pair in the grid.
2. `DualOutputPredictor.rollout()` chains carrier to carrier and never
   constructs a head.  `test_the_rollout_path_never_constructs_a_mode_head`
   monkeypatches call counters onto both heads and asserts zero invocations
   across a four-step rollout.
3. Each decision executes exactly one continuous, micro, or macro transition
   adapter. Micro and macro inputs fail closed when their selected typed
   symbolic content is absent; continuous calls retain the original
   `(latent, action, pair)` interface.

Head gradients *do* reach the latent — symbols shape `z` through the loss, which
is the point — but a symbol decode is never *inside* a rollout step.

## Components

| Module | What it is |
|---|---|
| `world_model/model/config.py` | Validated, identity-bearing configuration; `Abstraction`, `PredictionPair`, `EncoderConfig`, `PredictorConfig`, `JepaConfig` |
| `world_model/model/encoder.py` | `ContextEncoder` + `build_encoder()` registry |
| `world_model/model/ema.py` | `EmaTargetEncoder` — deep-copied, no-grad, cosine momentum ramp |
| `world_model/model/predictor.py` | `PairConditioner`, transition adapters, `FiLMBlock`, `DualOutputPredictor`, `PredictorOutput` |
| `world_model/model/heads.py` | `MicroReadoutHead`, `MacroReadoutHead`, `mode_weight()` (`ω_ψ`) |
| `world_model/model/jepa.py` | `JepaBackbone` — online encoder + EMA target + predictor |

### Encoder (1a)

A strided convolutional trunk: `conv7s2 → 64`, then three stages of two convs
each to `128 / 256 / 512`, **GroupNorm + SiLU throughout, no BatchNorm**,
followed by an attention pool over the resulting token grid to `z ∈ R^512`.

The no-BatchNorm rule is load-bearing, not stylistic: batch statistics would
couple samples within a batch and make the stop-grad target branch depend on
batch composition.  `test_the_encoder_contains_no_batch_normalization` and
`test_the_backbone_contains_no_batch_normalization` enforce it structurally.

`EncoderOutput.tokens` is the pre-pool grid.  It is a **side output** reserved
for Milestone 1c (SPSG) and is never the state carrier.

`build_encoder(config)` resolves the backbone by name from a registry, so
Milestone 2's frozen pretrained ViT registers as a new entry and needs no change
to the predictor, the EMA wrapper, or the training loop.

### EMA / stop-grad target encoder (1a)

`EmaTargetEncoder` deep-copies the online encoder, sets `requires_grad=False` on
every parameter, wraps its forward in `torch.no_grad()`, and updates in place as
`t ← m·t + (1−m)·o` for float parameters (non-float parameters and all buffers
are copied, never interpolated).  Momentum follows a cosine ramp from a base
value to `1.0` over the run.

The run's `TrainingConfig.ema_base_momentum` is the source of truth for the
schedule during training — it is what the manifest records — so the overfit
driver can use a faster target (0.99) than the default training schedule
(0.996) without editing the backbone.

### Dual-output predictor (1b)

`F_θ^{Δ,α}` **always** emits the carrier `ẑ_{t+Δ}` of shape `[B, 512]`.  The
mode-head readout is emitted only for the abstraction the controller selected:

| `α` | `carrier` | `micro_readout` | `macro_readout` |
|---|---|---|---|
| `continuous` | yes | `None` | `None` |
| `micro` | yes | `[B, n_micro]` | `None` |
| `macro` | yes | `None` | `(S^M, Δ̂, ê)` |

The selected adapter receives typed content separately from the pair
conditioner. `build_cohort_v2_transition_request()` carries available labels
from validated v5 oracle windows without turning unavailable labels into empty
or false values. The vocabulary is exactly `contact` and directed `supports`
for micro, and `steady-state` and `structure-unstable` for macro. Material/damage
labels and the excluded legacy macro predicates are not model inputs. Only the
selected adapter and selected readout execute for a decision.

`mode_weight(α, r_ψ)` is the proposal's `ω_ψ`: `0` for a continuous step, `r_ψ`
for micro relational constraints, `1` for macro-event supervision.  A masked
term contributes exactly zero gradient rather than being silently dropped.

### `(Δ, α)` conditioning

A single shared trunk.  Sinusoidal features of `Δ` (so unseen horizons in
Milestone 1e interpolate rather than requiring a new embedding row) are
concatenated with an `α` embedding and fused by an MLP into one **joint** code,
which modulates every trunk block through zero-initialized adaptive LayerNorm.

The conditioning is joint rather than additively factorized — proposal §4.4 —
and `test_the_pair_conditioning_is_joint_rather_than_additive` pins this by
checking that the code violates the parallelogram identity that any additive
conditioner would satisfy.

The adaptive scale/shift are **zero-initialized**, so at construction the predictor
is a plain residual MLP and the conditioning is deliberately inert; it becomes
live as soon as the modulation weights move off zero.  The conditioning tests
perturb the modulation explicitly rather than depending on that initialization
choice.

`PairConditioner` and the three transition adapters are injected into
`DualOutputPredictor`, so later symbolic-interface and conditioning ablations
are constructor swaps.

## Training loop

```
EpisodeCatalog.build(root, split="dev", capture_contract=LEGACY_RGB_V1)
  → TemporalWindowDataset(catalog, TemporalWindowRequest(prediction_steps=1,
                          stride_frames=Δ), transform=resize_transform(240, 320))
  → EpochSampler(seed, draw_count) → TemporalWindowCollator → DataLoader
```

`TemporalWindowRequest(prediction_steps=1, stride_frames=Δ)` *is* the
teacher-forced single-step setup at horizon `Δ`: context at `t`, one target at
`t+Δ`.  No new dataset code was needed.

One step: encode the context online, encode the target through the EMA/stop-grad
branch, predict the carrier, and regress `ẑ` to the detached `z*`.  AdamW
(β 0.9/0.999), lr 3e-4 with warmup + cosine decay, grad clip 1.0.

## What is deliberately inert, and why

**Symbolic supervision is wired but not trained by the legacy loop.** The
legacy RGB cohort carries no symbolic labels, so that loop optimizes the
carrier MSE only. The validated cohort-v2 reader now supplies the accepted
symbolic inputs; issues #5 and #6 own their training and scoring paths. Every
legacy run manifest records
`"symbolic_loss_active": false` so no run can be mistaken for one that trained
symbols.

**The enriched cohort does not exist yet.**  `data/physics_capture_v1_cohort` is
empty: the staged Unity player emits `collision` events with an empty payload
where the frozen contract requires `contact_ids` and `relative_speed`, so every
real gameplay shot fails artifact validation.  That is a player-side defect
requiring a full re-pin and is out of scope here; it does not block Milestone 1.

**The legacy action is shot-constant.**  Each shot carries one `[5]` action
vector, so single-step prediction *inside* a shot is largely autonomous
dynamics.  This is the action-sparse persistent-effect regime the proposal
targets (§2.1), not a defect — but action conditioning is therefore exercised
only weakly by this cohort, and that should not be mistaken for evidence that
the action pathway is well trained.

## Reproducibility

`RunManifest` is written to `<output-dir>/<run-id>/manifest.json`. Its `identity`
is a plain namespaced serialization of the declared experiment fields: seed,
git revision and dirty flag, torch/CUDA/device declarations, dataset root and
split, `catalog_identity`, `sampled_index_identity`, `model_config_identity`,
window selection, symbolic-loss status, and optimizer settings.

The identity names the **experiment, not its outcome**. It deliberately excludes
wall-clock timing, the timestamped `run_id`, and **the measured metrics**: CUDA
float reductions are not bitwise reproducible across processes, so two runs of
the same experiment differ around the 5th significant digit (measured: final
loss 3.9308e-08 vs 3.9300e-08). Compare `identity` for exact experiment identity;
compare the metrics numerically with a tolerance.

`world_model.data.catalog_identity(catalog)` is the public entry point for the
catalog's plain declared provenance identity.

## Overfit acceptance

A teacher-forced MSE against a stop-grad target can reach zero by representation
collapse: a constant encoder yields a constant prediction and zero loss.  Loss
alone is therefore not evidence.  `run_overfit` requires **all four**:

| Criterion | Threshold |
|---|---|
| final MSE | `< 1e-3` |
| mean per-dim std of `ẑ` | `> 1e-2` |
| effective rank of `ẑ` (spectral entropy, uncentred) | `>= 4` of 8 |
| retrieval accuracy (`ẑ_i` nearest to its own `z*_i`) | `== 1.0` |

All four land in the manifest alongside a `"pass"` / `"fail"` verdict, and the
CLI exits non-zero on a failed acceptance.

## What the legacy cohort is actually like (measured)

These numbers drove several design decisions and are worth not re-deriving.

| Quantity | Measured |
|---|---|
| dev catalog build | 31 s → 463 accepted, 1137 `missing_artifact` |
| eligible windows, Δ=1 / Δ=4 | 556,959 / 540,291 |
| context→target MSE, median over 400 random Δ=1 windows | **2.5e-05** |
| fraction of Δ=1 windows with MSE > 1e-3 | **0.000** |
| fraction of Δ=4 windows with MSE > 1e-3 | 0.007 |
| target-embedding separation, 8 uniformly drawn windows | **7.4e-05** |
| image separation, 8 windows from 8 distinct episodes | L2 ≥ 20.5 |

The cohort is overwhelmingly quiescent — which is precisely the action-sparse
persistent-effect regime the proposal describes (§2.1), now measured rather than
assumed.

The consequence for an overfit demo is sharp: **uniformly drawn windows are
frequently near-duplicates.** Their target embeddings land 7.4e-05 apart, so
"is each prediction nearest to its own target" is a coin flip no matter how good
the predictor is. The same backbone, same seed, same steps scores **1/8 on
near-duplicate windows and 8/8 on windows from distinct episodes.**

`--window-selection` therefore defaults to `diverse` (one window per episode).
`motion` (rank a seeded pool by inter-frame change) and `uniform` remain
available; `uniform` keeps the degenerate baseline reproducible.

### Warmup is wrong for a short overfit

Measured on the 8-window subset at 1500 steps, holding everything else fixed:

| warmup | weight decay | retrieval |
|---|---|---|
| 0 | 0.0 | **1.000** |
| 0 | 0.05 | **1.000** |
| 100 | 0.05 | 0.250 |

The cosine decay is measured against the *full* run, so after a 100-step warmup
the LR is already well into decay and the model settles on the trivial solution
first. `--warmup-steps` therefore defaults to **0 for `--mode overfit`** and 100
for `--mode train`.

## Evidence from the recorded runs

Seed 20260807, Δ=4, 8 windows, 1500 steps, one RTX 5090, ~120 s per run:

```
initial_loss      0.360247
final_loss        3.93e-08
relative_spread   0.005114      (> 1e-3)
effective_rank    2.916         (>= 2.0)
retrieval_acc     1.000         (8/8)
acceptance        pass
```

Two runs at the same seed produced **identical manifest identities**, with zero
differing configuration fields and metrics agreeing to ~2e-04 relative.

A 200-step `--mode train` smoke over 12,800 distinct dev windows ran without
stalling: loss 0.356163 → 0.000134. That is a stability smoke, not a training
claim.

## Reproducing

```bash
# Overfit evidence on 8 real dev windows (read-only dataset access).
# Exits 0 on a passing acceptance, 2 on a failing one.
python scripts/train_jepa_backbone.py --mode overfit --split dev \
    --seed 20260807 --steps 1500 --window-count 8 --delta 4 \
    --candidate-count 4096 --output-dir runs

# Reproducibility: same seed, second directory, identities must match
python scripts/train_jepa_backbone.py --mode overfit --split dev \
    --seed 20260807 --steps 1500 --window-count 8 --delta 4 \
    --candidate-count 4096 --output-dir runs-repro

# Short stability smoke over full dev windows (not a training claim)
python scripts/train_jepa_backbone.py --mode train --split dev \
    --seed 20260807 --steps 200 --batch-size 64 --delta 4 --output-dir runs

# Tests (113 across the two new suites; 271 with the existing data suites)
python -m unittest tests.test_world_model_model tests.test_world_model_training
```

Building a **dev** catalog takes ~31 s (463 accepted, 1137 `missing_artifact`
rejections — incomplete collector output, designed behaviour).  A **train**
catalog validates 10,328 episodes and takes minutes; use `dev` while iterating.

`runs/` and `checkpoints/` are gitignored.

## Todo 8 temporal projection

The approved legacy experiment trains the continuous carrier on
`delta={1,5,15}` with `abstraction=continuous` only. The real dev catalog is
read-only and has a plain `episode-catalog-v1` identity declared from its cohort,
collection-plan, split, and capture-contract fields (463 episodes, 5,556 shots,
562,515 frames, and 1,137 `missing_artifact` rejections). Training and scoring
artifacts bind the checkpoint, config, grid, catalog, run, partition, and
state-set identities. The primary and reproduction runs use seed `20260807`,
3,600 steps, batch 64, learning rate `3e-4`, weight decay `0.05`, zero warmup,
gradient clip `1.0`, and EMA base momentum `0.996`.

Exhaustive scoring enumerates every nonterminal state in each deterministic
episode partition and evaluates all three requested deltas. A state at a
terminal edge uses `effective_delta=min(requested_delta, T-t)` for scoring only;
the serialized label retains both requested and effective delta, plus explicit
terminal-clamp metadata. Shards are atomic and validators recompute the declared
state-set identity from sorted state identities, score count, per-pair
aggregates, and provenance before frontier generation.

This is temporal-only evidence. `micro` and `macro` remain unavailable with
the exact reason `symbolic_supervision_unavailable`; no symbolic, ADE/FDE,
final-state, event, penetration, floating, or illegal-contact result is
fabricated. The resulting frontier therefore supports only the scoped
continuous temporal comparison and makes no claim about the joint `(delta,
alpha)` controller or oracle-symbol ceiling.
