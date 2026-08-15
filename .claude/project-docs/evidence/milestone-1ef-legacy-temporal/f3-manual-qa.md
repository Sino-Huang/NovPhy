# F3 Manual QA Retry

Verdict: PASS

Reviewed detached source SHA: `95e532e164d9d49a0cecdb9514b9abca8be1e24a`.
The ignored `m1ef-primary` and `m1ef-repro` trees were read directly from the
physics worktree. This report is the captured artifact for all scenarios.

## Fresh Score-Tree Validation

Scenario: validate each complete ignored score tree from a fresh process.

Invocation:

```sh
PYTHONPATH=. python scripts/run_jepa_pair_grid.py validate --output-dir /mnt/array/sukaih/Project/.novphy-worktrees/physics-unity-2019.4/runs/m1ef-primary
PYTHONPATH=. python scripts/run_jepa_pair_grid.py validate --output-dir /mnt/array/sukaih/Project/.novphy-worktrees/physics-unity-2019.4/runs/m1ef-repro
```

Binary observable: both exit statuses were `0` and emitted:

```text
validated states=556959 scores=1670877 manifest=fa189bd625dc89777f52c1f102d2c87fe4b740dccc9156b9bd37b6f000c1983e
validated states=556959 scores=1670877 manifest=01ad3fbc820537b7721547502992e8795518d6db5c6cab48fe0cf321b241cf2b
```

Captured artifact: this file.

## Fresh Frontier Render And Witnesses

Scenario: render the primary canonical input into the unique detached-tree
temporary directory `.f3-qa.j2mO69`. The copied 460-byte input retained its
relative `score_artifacts` reference via a read-only symlink to the real
primary score tree; no source run artifact was written.

Invocation:

```sh
PYTHONPATH=. python scripts/plot_jepa_pair_frontier.py --input .f3-qa.j2mO69/frontier_input.json --output-dir .f3-qa.j2mO69/frontier --seed 20260807
```

Binary observable: the fresh CLI completed and its post-render verifier ran
under `set -e`. The output source digest was
`851f4b8257578277c3d1073cdce31e1516b1ed4e33be93a872e0ee390e2259ea`, with
`state_count=62064`, verdict `not_supported`, and
`global/high_motion/quiescent/transitional=[1,5,15]`.

Witness inspection: `frontier.json` is 143725 bytes; `frontier.md` is 399
bytes and contains the digest, unavailable scope, verdict, and all regime rows;
`frontier.svg` is 46551 bytes and `file` identifies it as SVG; `frontier.pdf`
is 17457 bytes and `file`/`pdfinfo` identify a PDF 1.4 with one 460.8 x 345.6
pt page.

Captured artifact: this file. The temporary directory is removed after this
report is written.

## Independent Raw-Record Checks

Scenario: independently recompute one state label and one regime aggregate
from raw evaluation JSONL records, not from the frontier output.

Invocation: fresh `python -` reader calculated
`weighted_error / error_scale + compute_cost`, applied the canonical tie break
`(weighted_error, delta)`, and separately averaged every transitional record.

Binary observables:

```text
STATE_LABEL PASS state_id=84342f419047d17299afbb84a36093feae64a9efc88e71957f37539ef0987b18 selected_delta=5 objective=0.373601680380498
TRANSITIONAL_AGGREGATE PASS states=24197
delta=1 mean_weighted_error=5.084646221045924e-09 mean_compute_cost=1.0
delta=5 mean_weighted_error=2.5293773402802632e-08 mean_compute_cost=0.2
delta=15 mean_weighted_error=7.684341757981684e-08 mean_compute_cost=0.06666666666666667
```

No transitional candidate dominated another; the recomputed frontier was
`[1,5,15]`, matching the fresh render.

Captured artifact: this file.

## Provenance And Unavailable Metrics

Scenario: exact field assertions over both manifests/sweep receipts, score
unavailable metrics, and fresh rendered-frontier unavailable metrics.

Invocation: fresh `python -` JSON reader.

Binary observable: both runs have catalog digest
`8265809a528e41eaae646cb1cae9d577d7f34fd99b85b859bb14f07a479c6beb`, run
identity `6b1d5b18fd45175ad3a6a03f31d80b2ffdf4c624d1493ce14bc67dd05fd403b9`,
and `checkpoint_step=3600`. Primary checkpoint digest is
`8fec18218a7647a8d448fc3021080d6834b7ed3e993725da447a047a4e809843`; repro
checkpoint digest is
`c0b8a28e3d34af98b0e85d8d63b103712cc87ac5774ab9b403d268c60769b0ab`.

The seven score metrics `ade`, `fde`, `final_state`, `event`, `penetration`,
`floating`, and `illegal_contact` are exactly `unavailable` with reason
`required supervision is unavailable`. Fresh frontier `alpha` and `physical`
have that reason; `micro` and `macro` have
`symbolic_supervision_unavailable`.

Captured artifact: this file.
