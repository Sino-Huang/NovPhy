# Representative cohort release — 2026-08-20

Issue: #31
Disposition: accepted, scoped to the lightweight representative cohort authorized by #39

The production collection executed frozen collection plan v4 through production parameter plan v1 at collector revision `91e974e5720a8d8f5843871815704953067b12c8`. Four fresh non-fixture Unity rollouts were accepted across two legacy-static scenario lineages and both planned intervention sources. The accepted run has 4 accepted, 0 rejected, 0 failed, and 0 quarantined attempts, with no unmet slots, coverage shortfalls, retries, or systematic exporter defects. Realized coverage is four collisions, four destructions, four stability-transition traces, and four level-fail terminations.

Immutable identities:

- Publication: `representative-cohort-publication-v1:sha256:a6daf82d47f7001e8731068c68a91e83487fd2c26926b35ab2974bc75a93ecf8`
- Cohort release: `representative-cohort-release-v1:sha256:40b997354a256f889ef7dd007888b5ad8d84b5266883f3611500086f22b62ed2`
- Authoritative derivations: `authoritative-cohort-derivations-v1:sha256:64cff842534beb40ece23ec11673903dddc9a08881e4646779daae973434e187`
- Partition manifest: `cohort-partition-manifest-v1:sha256:d2ab1b531e27cb2028c3e51064dfb53648b11176334bd230417ffbd2ba11f111`
- Production plan: `production-parameter-plan-v1:sha256:caf33ef3c05a92c8508473c99d91a1f30ff69dca087eff5323d1085e9b439f78`
- Collection plan: `collection-plan-v1:sha256:8e95f1085331ed47e6dff1d1d6319b39ca227d12983ef71fca89354fca847769`

The release inventories and digests every accepted primary trace, both scenario manifests, the production partition manifest, the frozen plan copies, and the production quality report. Separately versioned derivation artifacts are source-bound to the cohort release. Available derivations are `physics_relational_supervision_v1` and the accepted `steady-state` and `structure-unstable` predicates from `physics_macro_labels_v1`. Fifteen unavailable or excluded capabilities and labels are explicit in the authoritative derivation index, including material, damage, physical-violation, physical-regime, illegal-contact, template-held-out, access-separated observation, bounded-negative, replay, and the three predicates rejected by #40.

Pre-publication execution history is retained rather than rewritten. One port-conflict preflight failed before a plan attempt entered the ledger; two misrouted physics-port runs quarantined four transport failures each; and one reused post-pilot disposable runtime run quarantined one transport failure before interruption. All were pre-intervention and admitted no primary trace. Their manifests and reports are copied by digest into the final production quality history. The successful run restored four accepted wrapper/player layouts and bound one disposable runtime to each frozen attempt.

Authoritative entry points:

- `production-v1/release/cohort_publication_a6daf82d47f7001e8731068c68a91e83487fd2c26926b35ab2974bc75a93ecf8_v1.json`
- `production-v1/release/cohort_release_40b997354a256f889ef7dd007888b5ad8d84b5266883f3611500086f22b62ed2_v1.json`
- `production-v1/release/authoritative_derivations_64cff842534beb40ece23ec11673903dddc9a08881e4646779daae973434e187_v1.json`
- `production-v1/release/production_quality_report.json`

This release completes #31. It does not claim the broader #18 definition of done: capabilities explicitly unavailable in this scoped cohort remain unavailable rather than inferred or silently mapped to false.

## Downstream ingestion — issue #32

The immutable publication and all required authoritative sidecars were smoke-ingested through the public partition, scenario-manifest, physics-capture, macro-label, relational-label, and world-model supervision readers. The proof preserves all 24 frame-record identities, all 36 fixed-step event assignments, all four terminal observations, and all 608 explicitly unavailable relational labels. Missing capabilities, malformed artifacts, unknown sidecar-reference fields, and derivations bound to another cohort release fail closed.

Evidence: `downstream-ingestion-v2.json`
