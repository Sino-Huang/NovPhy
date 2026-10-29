## Result (2026-10-29): U-native: A4 − U-native is not `supported`, so A4's wording stays budget-qualified

Every #140 scheduler job exited 0 on its first launch (no P-crash, no pause, no amendment). `140-validate#1` (GPU 1, 19:35:20–19:37:30 AEDT) exited 0 after these checks:
- the 3 records match `records-manifest.json` (sha256 `4dbc9884…`);
- seed 20260908 re-scored on every grid cell through Gate-B's own spot-check rule (`harness.endpoint_replication`, max |Δ| 0);
- the publication was re-derived twice and was byte-identical both times.

Full write-up: `docs/issue-140-u-native.md`. Artefacts: `.local-artifacts/issue-140-u-native-v1/` (`summary.json` sha256 `586daf73…`, `compute.json` `7f8bfe46…`, `findings.md` `9506b568…`, `validation.json` `aec62745…`).

**Recipe.** As decided in [Decision P-adapter](https://github.com/Sino-Huang/NovPhy/issues/140#issuecomment-6479886189), frozen in `plan.json` sha256 `2cab0ce7…`:
- U's frozen #112 v3 retrain code and data;
- 16 929 updates per seed (1.881 × 9 000);
- schedule constants 941 / 1 881 / 1 881;
- seeds 20260908–10; no selection.

G-port passed bit-exactly against U's retrain code over the first 18 updates.

**Guards.**
- Global guards (G-isolation, G-chronology, G-scoring-policy, G-records) and G-port pass.
- U-native **fails G-action on seed 20260908** (0.0389 < 0.05). Seeds 20260909 / 20260910 pass (0.0567 / 0.0658).
- G-rank, G-nonfinite and G-trained pass on every seed. Every seed is usable under U's end-state rule, with 0 skipped updates.
- By the frozen rule every U-native token is `readiness_or_precision_insufficient`. U-native's disposition is **4** (per-contrast tokens only).

| ranker / contrast (grid, 311 cells × 3 seeds) | estimate [CI] (DESCRIPTIVE) | token | reverse token |
| --- | --- | --- | --- |
| U-native | 0.8132 [0.7977, 0.8287] | — | — |
| U | 0.7444 [0.7269, 0.7620] | — | — |
| A4 (#128 records) | 0.9945 [0.9922, 0.9965] | — | — |
| **A4 − U-native** | +0.1813 [+0.1658, +0.1967] | `readiness_or_precision_insufficient` | `readiness_or_precision_insufficient` |
| U-native − U | +0.0688 [+0.0579, +0.0795] | `readiness_or_precision_insufficient` | `readiness_or_precision_insufficient` |
| U-native − C\* | +0.0298 [+0.0197, +0.0395] | `readiness_or_precision_insufficient` | `readiness_or_precision_insufficient` |
| U-native − P | +0.0929 [+0.0711, +0.1145] | `readiness_or_precision_insufficient` | `readiness_or_precision_insufficient` |
| U-native − J_dec | +0.0672 [+0.0553, +0.0790] | `readiness_or_precision_insufficient` | `readiness_or_precision_insufficient` |

A4's side of the A4 − U-native guard passes:
- its arm guards, #128's global guards and #128's G-records;
- the three A4 records match #128's records manifest.

The stop token comes from U-native's G-action failure alone.

**Reading (pre-registered).** "If A4 − U-native is still `supported`, A4's budget qualification is dropped from its wording (an architectural reading is permitted); otherwise the wording stays budget-qualified." A4 − U-native is not `supported`, so **A4's wording stays budget-qualified**: "A4 at its native budget (1.88 × U's training GPU-h) ranks better than U at B1". No architectural reading of A4 − U is made.

**Realised budget.**
- U-native trained for 2 189.4 / 2 203.1 / 2 028.4 GPU-s (mean 2 140.3).
- That is 1.654 × U's 1 294.19 and **0.879 × A4's 2 434.73**.
- The rule scaled updates, not seconds. Each seed ran alone on an RTX 5090 at about 0.126 s per update, against the 0.144 s in U's reports, so it came in under the projected 2 434 s.

**Context (descriptive; changes nothing).**
- #129's calibration measured the carrier-space action sensitivity of the engine truth at 0.0395, and of U's Gate-B checkpoints at 0.022–0.024. U-native reaches 0.039–0.066.
- Wrong-anchor right − wrong is +0.2020 [+0.1886, +0.2152].
- Position error at t = 225 is 0.0528 for U-native against U's 0.0474.

**Compute.** 6 507.8 GPU-s recorded (1.81 GPU-h), plus validate's 131 s ≈ 1.84 GPU-h active, against the projected ≈ 2.2. Engine seconds: 0.
