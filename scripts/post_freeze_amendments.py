"""Disclosed post-freeze amendments of runners that later frozen plans pin as inputs.

A frozen plan binds every upstream runner it imports by sha256 (``plan["inputs"]``), and ``load_plan``
refuses to run on drift. After its own freeze a runner may still change, but only for reporting,
guard propagation or validation hardening, never for evidence (the phase-A review fixes of the #108
after-ICLR program). Each such change is listed here: the runner, the frozen sha256 values it replaces,
the sha256 of its amended bytes and what changed; the ticket documents name the commit. ``matches``
accepts a pinned input only if its current bytes are the frozen bytes or exactly the listed amendment
of them, so any other change still refuses. Every downstream ``--validate`` still regenerates its
published records byte for byte and re-scores spot checks through the amended module, so an amendment
that changed evidence would fail validation.
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
}


def matches(artifact, frozen_sha256, actual_sha256):
    """True iff the pinned input's bytes are the frozen bytes or the disclosed amendment of them."""
    if actual_sha256 == frozen_sha256:
        return True
    entry = AMENDMENTS.get(artifact)
    return entry is not None and frozen_sha256 in entry["replaces"] and actual_sha256 == entry["sha256"]
