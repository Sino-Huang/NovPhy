# Validator boundedness red/green

## Red

Command: fresh `python` process calling
`validate_score_artifacts(runs/m1ef-primary/score_artifacts)` before the
streaming validator change.

Observed: the process disappeared before emitting a receipt; the wrapper output
and error files were empty, and `ps` showed no validator process. The old
implementation retained every parsed shard label in `scored` and then built
additional aggregate/oracle tuples.

## Green

Command: fresh process calling the same validator after the change.

Observed receipt: `state_count=556959`, `score_count=1670877`, manifest
`fa189bd625dc89777f52c1f102d2c87fe4b740dccc9156b9bd37b6f000c1983e`.

The reproduction validator now streams shard records, retains only bounded
scalar accumulators, percentile samples, duplicate IDs, and incremental state
digests, and passes for both primary and reproduction artifacts. The focused
scoring/frontier suite remains green.
