## Decision P-other (2026-10-28): selection result — `native` (pre-declared rule); main seeds 20260909 / 20260910 train `native`

**Trigger.** The three pre-declared configurations (Decision P-adapter, issuecomment-6426541110) finished their seed-20260908 runs at the full budget; the rule (`eb.select_configuration`: largest policy-split pooled grid AUC at 4 d.p., ties → fewer parameters → declaration order) picks the configuration whose run is reused as main seed 20260908. Fit / policy data only; the evaluation split has not been read.

**Result** (`selection.json`, `deterministic_scoring()`, 94 policy grid mixed cells, 16 candidates × 8 paths each, the frozen readout):

| configuration | updates | policy grid AUC | training GPU-h (shared GPUs) | typed failure |
| --- | --- | --- | --- | --- |
| **native** | 409 600 | **0.6863** | 8.82 | none |
| ar4 | 409 600 | 0.6802 | 15.40 | none |
| native-2x | 819 200 | 0.6523 | 12.85 | none |

All three have 4 406 019 parameters. **Selected: `native`** (the Atari 100k recipe exactly).

**Smoke** (`smoke.json`, pass): accounting (13 919 fit shots, 480 fit anchors, 94 policy cells, 220 887 readout frames), G-port record current (adapter sha unchanged), G-fid / readiness / funding gates pass, CPU tests 21 passed. Projection of what is left at the measured rates: two seeds 17.6 GPU-h (8.8 each), readouts 3.1, scoring + novelty 9.4 → **child total 67.2 GPU-h** (37.6 already in the ledger incl. timing and selection readouts) against the funded 80.3 GPU-h and the 136 GPU-h cap; serial wall 19.2 h on 4 GPUs (cap 48 h). Next: the orchestrator's `--prepare` freeze, then phase B (`scripts/gpu_window_issue_137_main.json`).
