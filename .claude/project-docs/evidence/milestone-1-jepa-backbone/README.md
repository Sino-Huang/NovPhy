# Milestone 1a/1b Evidence — JEPA backbone overfit acceptance

Produced 2026-08-07 by the session that implemented Milestone 1a + 1b (commit `aa31b31`
on `physics-unity-2019.4`).

`overfit-pass-seed20260807.manifest.json` is the machine-readable acceptance record for the
passing overfit run, copied out of the gitignored `runs/` directory so the evidence survives
(plan non-negotiable #5: every milestone ends with machine-readable evidence).

## What it records

Seed 20260807, Δ=4, 8 episode-diverse dev windows, 1500 steps, one RTX 5090, ~120 s:

| Metric | Value | Threshold | Verdict |
|---|---|---|---|
| `final_loss` | 3.93e-08 | `< 1e-3` | pass |
| `relative_spread` | 0.005114 | `> 1e-3` | pass |
| `effective_rank` (centred) | 2.916 | `>= 2.0` | pass |
| `retrieval_accuracy` | 1.000 | `== 1.0` | pass |
| `acceptance` | **pass** | | |

`symbolic_loss_active: false` — the legacy RGB cohort carries no symbolic labels, so only
the carrier MSE trained.  See `docs/world_model_jepa_backbone.md`.

## Reproducing

```bash
python scripts/train_jepa_backbone.py --mode overfit --split dev \
    --seed 20260807 --steps 1500 --window-count 8 --delta 4 \
    --candidate-count 4096 --output-dir runs
```

Exit code 0 on pass, 2 on fail.  A second run at the same seed must produce an identical
`digest` (verified: `6029c6e7…`, zero differing configuration fields).  The metrics
themselves differ around the 5th significant digit — CUDA float reductions are not bitwise
reproducible across processes, which is why the digest deliberately excludes them.

## Caveats recorded with the evidence

- **Δ=4, not Δ=1.**  No Δ=1 dev window in 400 draws exceeds 1e-3 context→target MSE; the
  Δ=1 task is too static to be non-degenerate.
- **Episode-diverse selection is required.**  Uniformly drawn windows are near-duplicates
  (target embeddings 7.4e-05 apart); the identical backbone scores 1/8 retrieval there.
- **The action is shot-constant**, so action conditioning is exercised only weakly.

`git_dirty: true` in the manifest reflects the untracked staged physics player
(`sciencebirdsgames/physics-v1/`), which is untracked by design.
