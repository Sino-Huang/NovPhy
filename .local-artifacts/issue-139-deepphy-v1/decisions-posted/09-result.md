## #139 result (2026-10-28): T7 DeepPHY-VLM not run, typed `fidelity_not_reproduced`

**Outcome.** T7, the last P-coverage arm, is **not run**. G-fid failed in both logged attempts, so under P-fid it is dropped with `fidelity_not_reproduced` ([decision](https://github.com/Sino-Huang/NovPhy/issues/139#issuecomment-6463275977)).
- The not-run child summary is published (`summary.json`, `extra.pending_arms` {T7: `fidelity_not_reproduced`}; `findings.md`).
- No NovPhy policy or evaluation data was scored, no selection ran, and the evaluation split was never read.

**G-port: pass.**
- DeepPHY's Angry Birds prompt, message layout and answer parser are byte-identical to the official code, executed from the vendored source (4 prompt cases, 26 parse cases).
- Model inputs equal exactly what the official pipeline passes to `generate`.
- Teacher-forced log p matches generate's logits within 2.3e-6 nats on the released Qwen2.5-VL-3B (fp32).

**G-fid: fail.** Quantity: PHYRE, VLA prompt, Qwen2.5-VL-7B, success within 10 attempts over the 1 000 fold-0 test tasks, run with DeepPHY's official runner. Paper 9.62 %; band [7.47, 11.77] %, frozen before any run.

| k (attempts) | 1 | 3 | 5 | 7 | 10 |
| --- | --- | --- | --- | --- | --- |
| Paper Table 6 | 0.60 | 4.10 | 5.50 | 7.16 | **9.62** |
| Attempt 0 | 0.5 | 2.0 | 3.3 | 4.6 | **6.0** |
| Repair 1 (DeepPHY's pinned matplotlib / pillow) | 0.4 | 2.0 | 3.2 | 3.9 | **6.1** |

**Diagnosis (CPU).**
- All 9 723 actions of attempt 0 replay to identical PHYRE statuses under DeepPHY's pinned phyre 0.2.2 + numpy 1.26.4 and under our build. INVALID_INPUT (3 190 attempts) is a legitimate agent outcome counted as a failure by the official runner, and the `np.float` shim changes nothing.
- DeepPHY's pinned transformers never forwards its `temperature=0.1`, so the agent is effectively greedy, for them as for us.
- The only diagnosed deviation was the rendering of the grid-label glyphs in the VLM's gridded image (matplotlib 3.11 against the pinned 3.9.4). It was repaired in attempt 1 without effect.
- The MOCK agent through the same runner gives 6.20 % (paper 7.64 %); the replay shows this is sampling noise, not an environment offset.

**Compute.** 3.23 GPU-h used: timing probe 0.03, G-fid attempt 0 1.61, repair 1.59 (including a run SIGKILLed at 07:00 by a since-fixed stop bug). The funding decision allowed 8.8 GPU-h, so about 5.6 GPU-h return to the remainder.

Doc: `docs/issue-139-deepphy.md`. T7 is a coverage arm: it never entered the program claim, and it is listed as not run in the program summary.
