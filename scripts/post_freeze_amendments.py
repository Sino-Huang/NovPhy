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
    "scripts/run_request_interference_benchmarks.py": {
        "issue": 117, "replaces": ("sha256:746ce9d132f209c2cabb311c6d5885a469a3fab5f0af9ec489bfcbcb333b024b",),
        "sha256": "sha256:5a7620fba0e33a7bba856c4e36611c8ded8d0595bf934c21eb27f5d67889ce2c",
        "change": ("item 5: the frozen caps guard enters global_pass, guards_pass and the disposition stop condition; a "
                   "guard-forced scope also turns its per-request tokens into the stop token; validate log says "
                   "compute.json is compared by value; publication unchanged")},
    "scripts/run_remaining_frames_input.py": {
        "issue": 119, "replaces": ("sha256:fb8efa5b6fcb0313299d3039dceb379bc1c8a680100bbc4d894ee925dd36c8db",),
        "sha256": "sha256:def0276923fa014543686f9aeeb6ddec9a1c313e701e6aa2c81bd67784bbaedc",
        "change": ("compute_tables re-verifies the per-shot sha256 values and shot order embedded in windows-225.pt "
                   "against the #101 pool shots and refuses on any difference; publication unchanged")},
    "scripts/run_cost_alignment.py": {
        "issue": 120, "replaces": ("sha256:ac8613a8b4d7c450e4c1d431d07b805d103ca91aef1bdad1f54f8f5750a5a520",),
        "sha256": "sha256:396704c7567bfc495d1f43d8b22f26d7e3996afb16d981bd135f3ca56e20a70d",
        "change": ("apply_guards gates every token of the assembled publication, the inventories view included (global "
                   "guards and each cost's mixed-cell guard); --validate regenerates verdicts.json from the evaluation "
                   "branch file; publication unchanged")},
    "scripts/run_request_code_factorisation.py": {
        "issue": 121, "replaces": ("sha256:2410017a4ce2514420a5d0d5cd0ffe680c5fd037a7f500f72d8fbe79bece36f7",
                                   "sha256:cf8ef0e728e58cc992e23aa709ed599e363299dcd304781d20cc8cb3a8a83a49"),
        "sha256": "sha256:f9bd6279334ba8ea1f9fcaa59cd5405e62aedd4a166f48bb7b45384fe1008106",
        "change": ("A1 (validation only): the spot check re-scores each record's first seed that is not a typed failure; "
                   "A2 (reporting, phase-B/C review): the second disposition carries its demonstrated claim (trained "
                   "fused vs trained additive parameterization) and the qualification that interaction-specific "
                   "decision value is unidentified; frozen claim, rule, token and numbers unchanged")},
    "scripts/run_heldout_96_contrasts.py": {
        "issue": 104, "replaces": ("sha256:7c9a8d2843b9af6232c6f2e6beea71b731436558f8fde8c90ea543542f8d996e",),
        "sha256": "sha256:8cf796f01cc98f7c61b10e5757181828454ad8990ff7579fa66e745ff775a3a8",
        "change": ("A1 (reporting / guard propagation, after scoring): the frozen guard_failure rule ('any failed guard "
                   "turns every token of that cost into the stop token') also gates the grid secondaries J - M and "
                   "J - OL, which the frozen code had gated by the #96 secondary rule (own half-width in place of G6); "
                   "that rule and the #96 literal G6 bound stay as DESCRIPTIVE readings; records, estimates and "
                   "intervals unchanged")},
    "scripts/run_hjepa_comparison.py": {
        "issue": 123, "replaces": ("sha256:db7f87ae101bd3ab3c3c23243b05cd814db06b443029fbd1dfacd8e488426177",
                                   "sha256:b3390059a7d83c8433c5dc13fc211c25c6522d466cc11baf56f75268dfe4f6f4"),
        "sha256": "sha256:045530a99823ae3402ee48087db763ed09caa9f0b00543303f822921641f5cab",
        "change": ("A1 (scheduling only; owner GPU-window rule 2026-10-19, during main training, before any readout "
                   "or any evaluation feature / label read): --train handles SIGTERM / SIGUSR1: the running arm-seed "
                   "writes resume.pt (atomic tmp + os.replace, as the 2000-step checkpoints) after its current "
                   "optimizer step and the queue exits 75; the relaunch resumes there; the CPU / CUDA RNG states are "
                   "restored after the loader iterator is created (its worker-seed draw shifted a resumed run's CPU "
                   "stream) and the CUDA RNG is saved only when CUDA exists; objective, optimizer, schedule and "
                   "batches unchanged; tests/test_issue_123_training_resume.py: stop-and-resume bit-identical to the "
                   "uninterrupted run. A2 (pre-scoring fidelity review 2026-10-19, before any readout or evaluation "
                   "feature / label read; publication, validation, guards and the G-port check only): G-records gates "
                   "every token (fail closed until --validate writes validation.json bound to the records-manifest "
                   "sha256 with a passing re-score; a failed validation withdraws the publication and re-derives it "
                   "with every token the stop token); an absent paired estimate is the stop token (reverse(None) no "
                   "longer crashes) and the unguarded raw_token / raw_reverse_token fields are removed (estimates "
                   "kept as numbers); G-nonfinite counts the actual R1 and R2 cost values separately and either "
                   "share above 10 % fails the arm (S9 included); G-port also requires gport/a2/gport.json, whose "
                   "check adds a positive-LR step (AdamW state, updated parameters, movement); training, resume, "
                   "RNG, data, readouts and scoring unchanged")},
    "scripts/hjepa_train_ckpt.py": {
        "issue": 123, "replaces": ("sha256:ae6c27b830b8bd4bde08d852de8b27db8d5f7ac5384178984bd730baf0e74543",),
        "sha256": "sha256:f07eed6a3f03aea47561ae1a9153697cb7e239c2fe643a49b7c1eabd89e585be",
        "change": ("A1 (scheduling only; owner GPU-window rule): on SIGTERM the Tier-T run saves the official "
                   "ResumeCheckpoint state (its own _save: same file, format, atomic replace) at the end of the "
                   "current train batch and exits 75; Lightning's own SIGTERM stop also exits 75; a relaunch from a "
                   "finished resume file runs the train-end hooks Lightning skips (final object dump, planning eval "
                   "resuming from its per-task chunks) once; official resume semantics otherwise (step, optimizer, "
                   "LR continue; the interrupted epoch runs its remaining batches in a new shuffle order); "
                   "main_hjepa.py and every hyperparameter unchanged")},
    "scripts/run_hjepa_fidelity.py": {
        "issue": 123, "replaces": ("sha256:0e78d456eba8d29f9f36af82b43e9cd689da6e701f354afcb38b067db449a166",),
        "sha256": "sha256:05bbd271873b044cac1f9db21139e91def98ff76bba729e770df79dae86aba5e",
        "change": ("A1 (scheduling only; owner GPU-window rule): the --queue driver forwards SIGTERM / SIGUSR1 to "
                   "its running job, records the job's exit in the ledger as before, starts no further job and "
                   "returns 75; otherwise it returns 1 if a job exited non-zero or a Tier-T job left no "
                   "planning-eval metrics (was always 0); commands, overrides, --table and the G-fid rules "
                   "unchanged")},
    "scripts/hjepa_port_reference.py": {
        "issue": 123, "replaces": ("sha256:7e2db8db18c842e71e7d8b89eb93cadb15322aae246b0b0ad28f869c7e31da84",),
        "sha256": "sha256:c30878693388960bf529cc5128e853d1e3593abb43887fd8915299c2b631afde",
        "change": ("A2 (pre-scoring fidelity review): after the frozen step (unchanged, LR 0) the official scheduler "
                   "is advanced to the end of warm-up and a second step at every group's base LR is recorded with "
                   "its gradients, AdamW state (step, exp_avg, exp_avg_sq), parameters and LRs; re-run on CPU into "
                   "gport/a2/ (the frozen gport/ references and gport.json are kept)")},
    "tests/test_issue_123_hjepa_comparison.py": {
        "issue": 123, "replaces": ("sha256:d4f22c42e774e7a834a816d0ac6f45018ae0cd084d089ca255685fb00988aa57",),
        "sha256": "sha256:749ed45dabe3068be696b3ac100796893dcfd3a75b3a288f8a18c8d1ef4c4930",
        "change": ("A2 (pre-scoring fidelity review): tests of the G-records gate and failed validation, failed-arm "
                   "publication with empty estimates, per-readout G-nonfinite with S9 gating, the positive-LR G-port "
                   "record and its sensitivity to a wrong gradient clip; the frozen tests unchanged")},
    "scripts/run_hjepa_idm.py": {
        "issue": 125, "replaces": ("sha256:5e533591c02d471e02bd8178cf9e0415e8f2009ccc69a94f97d8c5cd0647dd7b",),
        "sha256": "sha256:56e7cc6588a430fb63f69ea31ac6011f8b7c3372ae2500f6a0241e785f82dba7",
        "change": ("A1 (compute ledger and plan loading only; during main training night 1, before any #125 readout "
                   "or evaluation read): --train marks an arm-seed's training seconds in the ledger only if its "
                   "model.pt did not exist before the call (a relaunched queue re-visits finished arm-seeds, whose "
                   "report.json the frozen code added a second time; the bug reported by #123's publication); "
                   "load_plan no longer hash-checks this shared registry (pinned in the frozen #125 plan, but it is "
                   "the append-only file matches() reads, so any later registration by any ticket would have made "
                   "#125's relaunch refuse); objective, training, resume, data, readouts and scoring unchanged")},
}


def matches(artifact, frozen_sha256, actual_sha256):
    """True iff the pinned input's bytes are the frozen bytes or the disclosed amendment of them."""
    if actual_sha256 == frozen_sha256:
        return True
    entry = AMENDMENTS.get(artifact)
    return entry is not None and frozen_sha256 in entry["replaces"] and actual_sha256 == entry["sha256"]
