# Balanced full-rollout event engineering findings

The frozen full-rollout engineering stage completed cleanly. All six event
heads and six selectors were complete and finite, with 39,000 applied updates
and no invented targets or dropped assigned captures. Full fixed features and
adaptive scores were finite for the retained 2,600-row inventory.

The exposed model-selection engineering result was not promising: hybrid
adaptive, same-hybrid fixed, and primary pure each hit 1/33 groups (10 groups
had an available clear action), versus 3/33 for the selected no-model training
family prior. Calibration hybrid hit zero; primary pure selected fixed-50
continuous with one hit. No readiness claim follows.

The genuinely repeated hybrid selector was non-degenerate: its second mode
fraction was 0.3065 and second horizon fraction 0.4038. It used continuous,
micro, and macro modes and all three horizons. Its actual model/selector/head
linear MAC was 0.1125 times the calibration-selected primary pure policy.
Thus compute and adaptation were no longer the failing criteria; action
utility was.

A training-only score audit found hybrid selected ordinal 1 in every one of
the 68 admissible training groups and hit zero of 22 informative groups.
Every fixed hybrid pair also hit zero. This indicates failure to learn
training action ranking, not primarily a held-out generalization failure.
Within-training-lineage endpoint mean standard deviation was 0.00306 for
hybrid and 0.00593 for pure; action variation was finite and present.

Training-only differential probes used the same seed-1 fixed-continuous-50
features and no calibration/model-selection outcomes. At 500 full-batch
updates, both per-lineage normalized weighted cross-entropy and globally
balanced weighted loss still hit zero informative training groups. At 1,000
updates with globally balanced loss, stronger action scaling learned 9/22,
explicit state-action multiplicative conditioning learned 6/22, and both
together learned 22/22. The combined probe loss was 0.5652. These are fitting
diagnostics, not utility/generalization evidence.

The next prospective engineering intervention is therefore a conditioned
readout and globally balanced fitting recipe, reusing the exact completed
dynamics/features and unchanged selector/comparator/screen rules. Its new
held-out engineering result must be frozen and scored separately. A new
disjoint development cohort remains required before readiness claims.
