# Cohort-v2 controller validation diagnostics

The intermittent source-bound controller validation failure is unresolved. It did
not reproduce on 2026-08-25 in the accepted issue #10 validator or the exact issue
#11 preflight. No inference policy, numeric computation, serialization, provenance,
or publication behavior was changed.

The reproduction matrix used fresh Python processes with Torch CPU thread counts 1
and 4 and `PYTHONHASHSEED` values 0 and 17. It covered repeated fixture publication
and validation, accepted production checkpoint inference over the frozen held-out
release inputs, and issue #11 validation preflight. The environment was Python
3.13.9 and Torch 2.13.0+cu130; deterministic algorithms were disabled. All matrix
entries passed, and fixture decisions, scores, and manifests were byte-identical.

Run the matrix with:

```sh
python -u -m scripts.reproduce_cohort_v2_controller_validation
```

The validator retains byte-exact recomputation and has no retry or floating-point
tolerance. Failures identify one of these components:

- `canonical_manifest`
- `stored_artifact_identities`
- `recomputed_decisions`
- `recomputed_scores`
- `recomputed_manifest_provenance`

Each mismatch includes short expected and actual identities, the first differing
record or field, Python and Torch runtime settings, and artifact size/mtime metadata.
Use that first difference to reduce a future reproduction to one controller, state,
and field before changing controller behavior.

The accepted issue #10 and issue #11 artifacts remain authoritative and must not be
rewritten. If a future reproduced fix changes controller result bytes or provenance,
publish versioned issue #10 controller artifacts and regenerate issue #11 under a
new output root.
