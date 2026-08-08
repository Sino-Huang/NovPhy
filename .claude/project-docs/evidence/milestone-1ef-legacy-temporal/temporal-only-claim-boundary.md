# Temporal-Only Claim Boundary

Todo 8 evaluates the continuous JEPA carrier on the legacy RGB dev catalog at
requested deltas 1, 5, and 15. It establishes an auditable temporal
prediction-versus-compute artifact: deterministic partitions, target-aware
motion regimes, terminal-clamp metadata, checkpoint/config/catalog provenance,
per-pair metrics, and a bootstrap frontier.

It does not establish a joint `(delta, alpha)` controller. The legacy cohort has
no engine symbolic supervision, so `micro` and `macro` are unavailable with
reason `symbolic_supervision_unavailable`. ADE, FDE, final-state accuracy, event
F1, penetration, floating, and illegal-contact metrics are consequently also
unavailable. No value for those metrics appears in the evidence.

The two fresh CUDA runs share experiment identity and update-count contracts,
but fail the requested reproducibility thresholds: aggregate metrics are not
within `rtol <= 1e-2`, and best-pair agreement is `0.9695291753971118` versus the
required `>= 0.99`. The acceptance status is therefore
`inconclusive_reproducibility_failure`; this is recorded rather than hidden or
repaired by threshold changes.
