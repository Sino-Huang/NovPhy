# Full-duration recursive objective: implementation and resource probe

The boundary-balanced experiment repaired the initial-state deficit but left
severe short-horizon drift and failed its overall qualification. Its weights
and results remain unchanged. This preparation implements a longer recursive
objective without running an optimizer or accessing fresh data.

`scripts/issue_76_full_duration_loss.py` constructs exact observed targets up
to 11,250 native fixed steps: 225 / 45 / 15 recursive transitions at horizons
50 / 250 / 750. Targets stay inside their source shot segments; missing exact
endpoints remain zero-filled and masked. As in the original loss, fitting stops
when no batch row has an available adjacent target pair. The admitted intact
observed windows supply contiguous horizon-grid prefixes, not invented tails.

The local loss still uses at most four observed-context transitions. The
recursive component now averages over its full available trajectory, without
truth resets or gradient truncation, and retains the original .01 carrier-bound
penalty. Local and recursive averages receive equal weight. Hybrid symbolic
supervision still uses only the original observed offsets 0–4; extending the
continuous trajectory does not add direct symbolic-label supervision. Pure
fitting accepts only carriers, availability masks, actions, and its pair.

Three new tests verify four-step loss/gradient equivalence with the original
objectives, exact prefix/action/symbolic-target matching and shot boundaries,
and a known recursive chain that would fail if predictions were truth-reset
or masked tail targets were used. Fourteen focused tests pass together.

## No-update resource evidence

`scripts/probe_issue_76_full_duration_resources.py` measured one full
forward/backward pass for each of the nine hybrid and three pure pairs, batch
32, on the first preassigned training lineage and seed 760930001. Both models
were loaded from the completed boundary-balanced checkpoints. Every loss and
gradient was finite, and exact post-probe parameter comparisons confirmed no
weights changed. No optimizer was constructed or called.

| Arm | Horizon | Recursive transitions | Forward/backward seconds | Peak allocated GPU MiB |
| --- | ---: | ---: | ---: | ---: |
| Hybrid continuous / micro / macro | 50 | 225 | .937 / .998 / .761 | 303.9 / 418.7 / 321.1 |
| Hybrid continuous / micro / macro | 250 | 45 | .140 / .191 / .165 | 88.7 / 114.6 / 92.2 |
| Hybrid continuous / micro / macro | 750 | 15 | .055 / .071 / .062 | 52.8 / 63.9 / 54.1 |
| Pure continuous | 50 | 225 | .635 | 359.2 |
| Pure continuous | 250 | 45 | .127 | 101.9 |
| Pure continuous | 750 | 15 | .049 | 59.0 |

The complete probe consumed 5.069 active seconds under a 180-second ceiling,
with peak CPU RSS 1,425.7 MiB. The artifact
`data/issue-76-full-duration-resource-probe/report.json` contains all 12 rows,
source text, source checkpoint-plan binding, unchanged-weight declaration,
and completed resource accounting. No concurrent GPU research job was running.

These are single-pass resource measurements, not estimates of model quality
or guaranteed production throughput. They support retaining batch 32 and the
full endpoint rather than reducing either for memory reasons.

## Next stage to freeze

A practical next experiment is a separately identified continuation of all
six boundary-balanced checkpoints with this objective, preserving optimizer
state and charging the complete prior fitting costs. A fixed 1,800-update stage
at absolute schedule positions 6000–7799 has exactly 1,786 applied positions,
14 preserved skips, and 57,152 sampled starts per arm/seed. Keep the existing
half-boundary/half-uniform sampler and within-pair mixing; preserve absolute
pair and sampler indexing. This is a new objective/stage, not a retroactive
extension or changed outcome for the failed boundary experiment.

The measured pair averages imply approximately 676 seconds per hybrid fit
and 487 seconds per pure fit before allowing for production overhead. Numeric
stage limits, initialization bindings, complete comparison controls, and
qualification criteria must still be prospectively frozen and published
before any optimizer work. The stronger older pure checkpoints remain
eligible controls. No fitting-stage plan has yet been frozen or executed by
this preparation, and no advancement or fresh-access permission is implied.

Initialize `novphy` and source `env.sh` before the tests or probe. The published
probe is immutable; this document reports its completed run, not a repeated
resource allowance.
