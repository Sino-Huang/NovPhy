# Issue-95: #94 follow-up controls (release-inert rescoring, within-power ordering) — findings

- identity `issue-95-launch-power-followup-v1`, plan v1 frozen 2026-09-24T08:21:37Z; designed after the #94 publication (post-#94 follow-up, #94 dispositions {'C24': 'supported', 'C25': 'not_supported_by_this_experiment'} known); frozen before any release-600 cost or within-power statistic was computed; the power-only baseline (engine verdicts only, no ranker input) was printed by a synthetic-cost dry run of this runner before the freeze
- validation command: `python -u -m scripts.run_launch_power_followup --validate`; zero engine seconds; every interval DESCRIPTIVE
- release inertness: ScienceBirdsBridge.shoot sends (x, y, release_time, tap_time); the jar stores them as Shot(x, y, t_shot, t_tap); ShootAndTapSchema.shoot reads only getX, getY and getT_tap into ProxyTapShootMessage(x, y, tap_time), and AIBirdsConnection.TapShoot reads only x, y and tap_time. The release time never reaches the engine, so a #94 candidate at release 600 is the identical engine action and its retained #94 verdict applies.

## Dispositions (frozen rules) vs stated predictions

| claim | outcome | stated prediction |
|---|---|---|
| C26 | **supported** | supported |
| C27 | **not_supported_by_this_experiment** | no prediction |
| C28 | **readiness_or_precision_insufficient** | no prediction |

- release_control reading: "The #94 dissociation survives moving the release input into the rankers' dominant training regime; the release-1000 extrapolation does not explain it. The remaining training-range confound is the pull radius alone."
- power_control reading: "Report descriptively in the appendix only."

## Release control (release 600 vs #94 release 1000; identical engine actions and verdicts)

- pooled top-1: release 600 **0/72** vs chance 0.0521, paired difference -0.0521 [-0.0563, -0.0500] (release 1000: 0/72)
  - continuous-fixed-h1: 0/24
  - continuous-fixed-h5: 0/24
  - hybrid-fixed-h1: 0/24
- AUC_m: release 600 **0.7456 [0.6959, 0.8061]** (release 1000: 0.7441 [0.6923, 0.8026]); cell-unit 0.7456 [0.7247, 0.7670]
  - continuous-fixed-h1: 0.7641 [0.7243, 0.8199]
  - continuous-fixed-h5: 0.7553 [0.7127, 0.8099]
  - hybrid-fixed-h1: 0.7173 [0.6491, 0.7918]
- chosen ordinals at release 600: {"9": 2, "14": 65, "19": 68}; saturated-column share 1.0000
- rank agreement release 1000 vs 600: {"cells": 135, "median_rho": 1.0, "min_rho": 0.9909774436090225}
- within-state structure at release 600, median rho cost vs realized speed within angle: -1.0000

## Power-only control

- within-power AUC (release_1000): member-clustered **0.5949 [0.4444, 0.7454]** over 8 members / 72 cells; cell-unit 0.5949 [0.5347, 0.6597]
  - continuous-fixed-h1: 0.6736 [0.5278, 0.8194]
  - continuous-fixed-h5: 0.6042 [0.4028, 0.8056]
  - hybrid-fixed-h1: 0.5069 [0.3958, 0.6250]
- within-power AUC (release_600): member-clustered **0.5995 [0.4583, 0.7361]** over 8 members / 72 cells; cell-unit 0.5995 [0.5417, 0.6597]
  - continuous-fixed-h1: 0.6736 [0.5278, 0.8194]
  - continuous-fixed-h5: 0.6181 [0.4306, 0.8056]
  - hybrid-fixed-h1: 0.5069 [0.4167, 0.5972]
- power-only baseline (cost = -expected speed): top-1 0/8 vs chance 0.0521; AUC_m 0.6889 [0.6424, 0.7599]; within-power AUC 0.5 by construction

## Claim boundary

controls on the #94 inventory and its retained engine verdicts only: the release control moves one engine-inert ranker input into the training regime and leaves the pull-radius extrapolation (5-17 px, mostly outside training support) untouched; no retraining, no new engine slot, no other ranker family; #94 dispositions are inputs and are not re-opened.
