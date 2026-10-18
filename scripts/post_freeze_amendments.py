"""Disclosed post-freeze amendments of hash-pinned runners and modules: the one sanctioned registry.

A frozen plan binds runners and modules by sha256: the upstream runners it imports (``plan["inputs"]``)
and, in some plans, its own runner and modules (``runner_sha256_at_freeze``, ``inputs.modules``).
``load_plan`` refuses to run on drift of an enforced pin. After its freeze a runner or module may still
change, but only for digest repair, reporting, guard propagation or validation hardening, never for
evidence (the phase-A and phase-B/C review fixes of the #108 after-ICLR program), or for the disclosed
pre-scoring implementation repair listed below. Every such change is listed here: the file, the frozen
sha256 values it replaces (earlier disclosed amendments of it included), the sha256 of its amended bytes
and what changed; the ticket documents name the commit. ``matches`` accepts a pinned file only if its
current bytes are the frozen bytes or exactly the listed amendment of them, so any other change still
refuses. Every ``--validate`` still regenerates its published records byte for byte through the amended
file, so an amendment that changed evidence would fail validation.
"""
from __future__ import annotations

AMENDMENTS = {
    "scripts/run_decision_aware_policy.py": {
        "issue": 101, "replaces": ("sha256:8d743b07483afad73d935b8454ae2540f6341eab9f03ba85bfc66827ca2f1a5b",),
        "sha256": "sha256:00415b79fb0f4485269ff2107c9828770c51e9268cd629edd0a2ba3bd0c74ea4",
        "change": ("a failed guard invalidates every inventory and wrong-anchor token (frozen guards.failure); "
                   "records-manifest.json of the 81 raw records published and byte-checked by --validate; F* "
                   "cross-inventory qualification in findings.md")},
    "scripts/run_novelty_coverage.py": {
        "issue": 110, "replaces": ("sha256:86307b1c3bf45dfb806d934e1c19ce7adc2d60275cb19fb14ec0e4026e8683fa",),
        "sha256": "sha256:57b2e1415c366d7c950fbd9972be82e14a9d09550cea73beafca3de812754bfe",
        "change": ("load_plan verifies every pinned input, smoke.json and the repair record; --repair refuses after "
                   "the freeze; a failed guard invalidates every probe verdict and readiness status; scoring-policy "
                   "coverage and encoder-training summary published; --probe records its scoring policy; "
                   "--probe-reproducibility (owner decision 2026-10-16) retrains every probe under "
                   "deterministic_scoring() into probe-reproducibility/ and the publication reports it beside the "
                   "published probes with the G4 literal-text failure")},
    "scripts/run_horizon_vs_joint.py": {
        "issue": 114, "replaces": ("sha256:d14edbeed0377b606e9897074b1cbd01ef21905ce9430bbd2866eb29af715bc5",),
        "sha256": "sha256:e632248e8be86763a52f873cc6c02cf8e84880bdefeae9761f2d5d95c1e478cb",
        "change": ("a failed guard is reported with the statement that every secondary token is not interpretable "
                   "(frozen rule unchanged); load_plan uses this registry")},
    "scripts/run_hierarchy_baselines.py": {
        "issue": 105, "replaces": ("sha256:ca218c194d853516de746fab85d2b1af28248a031312a33bcc3ce6ff0f8539ad",),
        "sha256": "sha256:609e30af50798e7ecb50b905de195658fd2c77cd4a98a473284606ad4e921721",
        "change": ("a failed guard is reported with the statement that every secondary token is not interpretable "
                   "(frozen rule unchanged); load_plan uses this registry")},
    "world_model/data/pixel_benchmarks.py": {
        "issue": 106, "replaces": ("sha256:b495dfe77dc4c274fe8ce32a1576348b1794ac2ac7bb65fb66da5a0ecc0a9a09",),
        "sha256": "sha256:4c2b8e5359e75a3df7dbc8db3757d847655cf8f07b83dff41fd75959ed04346c",
        "change": ("A1 (parse stage, before any predictor, policy or score): the 512 x 512 Physion testing frames are "
                   "resized to the frozen 256 x 256 parser input with the PIL box filter and the segmentation presence "
                   "threshold is scaled to the stored area; testing labels re-derived")},
    "world_model/training/pixel_carrier.py": {
        "issue": 106, "replaces": ("sha256:bf34939649e8cc5d5d5357450a9e28de249755e5c42cac5e9a20b08138ee4a62",),
        "sha256": "sha256:5efd891e635fcddac1aeb951fdc46b031ff760ea4acae70b807d6f15e7d7d2a3",
        "change": ("A2 (after scoring): slot_attributes gains min_frames (default 1 = the frozen behaviour) for the "
                   "EXPLORATORY readout; frozen records unchanged")},
    "scripts/run_pixel_benchmarks.py": {
        "issue": 106, "replaces": ("sha256:60e9371d95f0e285583008c298647742ee3a2fc3dcf68050d512bb639edaf84a",
                                   "sha256:cc868cba940c72acc6e282809a02974e13323c610281f888cf24527ebb910d27",
                                   "sha256:7d8f18fd1c43b62fe56467394793e2899dcf463ab8515d9ce2aa96c06fed34cf"),
        "sha256": "sha256:6ab660e2988028b7fd55c95db4c728d7f2127e607fea715359073fd753e20878",
        "change": ("A1 amendment listing, A2 EXPLORATORY readout mode; A3 (phase-B/C review): the Physion static "
                   "digest hashes canonical string / number contents instead of object pointers, mode leakage-audit "
                   "re-derives G1 on the frozen membership with it and reports the colour-excluded physical overlap, "
                   "G1 reads that audit and --validate regenerates it, load_plan verifies data_manifest_sha256 and "
                   "accepts module drift only through this registry, publication wrapped in deterministic_scoring()")},
    "scripts/run_predicate_feedback_robustness.py": {
        "issue": 115, "replaces": ("sha256:87655498f700bfd773d1a16ff3e4dfc9d6508841c49f031ca44310b6fb3dbe04",),
        "sha256": "sha256:28483409fce5f18fe6b5704d7439d9be7780958bd7f2063318e2f33b86ae7c64",
        "change": ("a failed guard also invalidates each disposition's contrast_tokens map and the reference J - T "
                   "token (apply_guard_failure; frozen rule 'every token'); publication unchanged")},
}


def matches(artifact, frozen_sha256, actual_sha256):
    """True iff the pinned input's bytes are the frozen bytes or the disclosed amendment of them."""
    if actual_sha256 == frozen_sha256:
        return True
    entry = AMENDMENTS.get(artifact)
    return entry is not None and frozen_sha256 in entry["replaces"] and actual_sha256 == entry["sha256"]
