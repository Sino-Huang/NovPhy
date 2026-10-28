# Issue #139: DeepPHY-VLM (#127 coverage arm T7) on NovPhy (port, fidelity, and the typed not-run outcome)

Parent: #127 (external baselines program). Ticket: [#139](https://github.com/Sino-Huang/NovPhy/issues/139). The binding
shared rules of #85 and the #127 shared protocol (`scripts/external_baselines.py`, `eb`) apply. Nothing here edits the
ICLR 2026 submission.

T7 is the last **coverage arm** under P-coverage: "DeepPHY with open-weight VLMs run locally (no paid API calls)". It
is secondary and is reported, and it never enters the #127 program claim.

| | |
| --- | --- |
| Paper and code | Xu et al., *DeepPHY: Benchmarking Agentic VLMs on Physical Reasoning*, AAAI-26, arXiv 2508.05405 v1; [XinrunXu/DeepPHY](https://github.com/XinrunXu/DeepPHY) @ `0562104` (MIT) |
| Models | `Qwen/Qwen2.5-VL-7B-Instruct` @ `cc59489` (Apache-2.0; primary) and `Qwen/Qwen2.5-VL-3B-Instruct` @ `6628554` (Qwen Research Licence, non-commercial research), fp16, local; `data/external/deepphy/MANIFEST.json` |
| Runner | `scripts/run_issue_139_deepphy.py` (gport-inputs / project / timing / selection / selection-readouts / smoke / prepare / train / readouts / freeze-readouts / score / score-novelty / publish / validate) |
| Modules | `world_model/baselines/deepphy/novphy.py` (DeepPHY prompt / action code / parser, the answer-likelihood score, and the scoring worker that runs in env `b127-deepphy`); `vendor/` (8 upstream files, verbatim, no edits; `VENDOR.json`) |
| Fidelity | `scripts/b127_deepphy_port_reference.py` (G-port); `scripts/b127_deepphy_fidelity.py` (G-fid: DeepPHY's official PHYRE runner, the MOCK check and the repair attempt) |
| Tests | `tests/test_issue_139_deepphy.py` (17 tests, CPU) |
| Envs | `b127-deepphy` (DeepPHY's transformers 4.52.4 stack, torch 2.7.1+cu128, phyre build of #128); `b127-deepphy-gfid` (the same plus DeepPHY's pinned matplotlib 3.9.4 / pillow 11.3.0; repair attempt 1); `b127-phyre022` (py3.9, released phyre 0.2.2, numpy 1.26.4, matplotlib 3.9.4: diagnosis only). Package lists in `envs/` |
| Artifacts | `.local-artifacts/issue-139-deepphy-v1/`: `decisions.md`, `gport/gport.json`, `fidelity/{band,gfid-attempt0,gfid,mock}.json`, `fidelity/{runs,runs-r1,mock}/` (official per-task logs), `timing.json`, `projection.json`, `funded.json`, `not-run.json`, `summary.json`, `findings.md`, `ledger/` |
| Outcome | **Not run, typed `fidelity_not_reproduced`** ([Decision P-fid](https://github.com/Sino-Huang/NovPhy/issues/139#issuecomment-6463275977)). The not-run child summary was published with `--publish` (`extra.pending_arms` {T7: `fidelity_not_reproduced`}). There is no `--validate` for a not-run child: no plan, records or manifest exist |

## Outcome

- **G-port passes.** DeepPHY's prompt construction and answer parsing are byte-identical in the adapter. The model call matches the official inference path.
- **G-fid fails in both logged attempts**, against the band frozen before any run:
  - quantity: PHYRE, VLA prompt, Qwen2.5-VL-7B, success within 10 attempts;
  - paper value 9.62 %, band [7.47, 11.77] %;
  - attempt 0: **6.0 %**; repair attempt 1: **6.1 %**.
- One environment deviation was diagnosed and repaired: the matplotlib glyph rendering in the VLM's gridded image. The repair changed nothing (the two attempts agree). No other port or environment defect was found.
- Under P-fid, T7 is therefore dropped with `fidelity_not_reproduced` and listed as not run.
- No NovPhy policy or evaluation data was scored, no selection ran, and the evaluation split was never read.

## Design (frozen before any NovPhy result)

Decisions, all posted on #139 before the step they govern, are listed at the end. The design below was never exercised on NovPhy data, because the G-fid gate blocked selection.

- **Input.** The NovPhy decision frame is frame 0 of the anchor branch, uint8 3 × 240 × 320, the same frame #129, #130 and #137 score. It is 2× pixel-replicated to the engine's 640 × 480 screenshot and passed by path, as DeepPHY passes its screenshot; there is no crop, because NovPhy frames have no iPhone-mirroring title bar. The prompt is DeepPHY's first-shot Angry Birds prompt, byte for byte: current bird `red`, queue `None`, no history.
- **Action.** A launch is rendered in DeepPHY's action code `[shoot(angle=X, power=Y)]`:
  - angle = atan2(drag_y, −drag_x) in degrees, rounded to an integer;
  - power = min(1, pull / 18.49 px), the BirdRed drag radius, rendered with 2 decimals.
  
  Grid ordinals 8 and 12 both render as `angle=2` and tie exactly.
- **Score.** The teacher-forced log-probability of that answer followed by `<|im_end|>`. Row value −score.
- **Configurations.** `q7b` (T = 1), `q7b-t01` (repetition penalty 1.05, T = 0.1) and `q3b`.
- **Runs.** One deterministic pass, fp16, batch 16, written as the three seeds.
- **Zero-shot adaptations:** `--train` is a no-op, and G-action, G-collapse and G-readout do not apply. Arm checks are G-rank, G-nonfinite and G-trained, plus G-port and G-fid.

## G-port (`gport/gport.json`, pass)

The reference executes the official functions from the vendored source in env `b127-deepphy`, on CPU: `get_llm_action`, `parse_and_execute_action`, the parsing statements of `play_level_with_llm` and `get_qwen_model_response`.

- **Prompts and messages.** 4 cases, including multi-bird and history prompts. System prompt, user prompt, image list and chat messages are byte-identical.
- **Parser.** 26 answers (the 16 grid answers plus code fences, multiple brackets, `success()` and malformed strings) all agree.
- **Model call.** On a fit-anchor frame, our processor outputs (`input_ids`, `attention_mask`, `pixel_values`, `image_grid_thw`; 1 202 prompt tokens) equal exactly what the official pipeline passes to `generate`.
- **Log-probabilities.** Teacher-forced log p of the generated tokens against generate's logits (tolerance 1e-3 nats):
  - released Qwen2.5-VL-3B, fp32: 2.3e-6 nats;
  - a tiny random Qwen2.5-VL loaded through DeepPHY's own `initialize_qwen_vlm` in fp16: 4.9e-4;
  - the same tiny model in fp32: 9.5e-7.
- **Batching.** Candidate scores are batch-invariant within 3.8e-6.
- The 7B in fp32 needs about 33 GB of host RAM, over the shared host's budget. The 3B runs the same model class with released weights.

**Decoding finding** ([addendum](https://github.com/Sino-Huang/NovPhy/issues/139#issuecomment-6437519414)):
- DeepPHY calls `pipe(messages, max_new_tokens=200, do_sample=True, temperature=0.1, return_full_text=False)`. In transformers 4.52.4, DeepPHY's pin, the image-text-to-text pipeline routes `do_sample` and `temperature` into the preprocess parameters, and the chat branch ignores them. `generate` therefore decodes with the checkpoint's `generation_config`: temperature 1e-6, repetition penalty 1.05, top-k 50. That is effectively greedy.
- The same holds in transformers 4.49, 4.50, 4.51, 4.53 and 4.55, so DeepPHY's own runs decoded the same way.
- The paper's "temperature 0.1" is therefore not what the released code does locally.

## G-fid

**Frozen design** ([Decision P-fid](https://github.com/Sino-Huang/NovPhy/issues/139#issuecomment-6437344622); `fidelity/band.json` sha256 `ad5bfb65…3c30`, frozen before any run):
- DeepPHY's Qwen-7B number is only reproducible here on PHYRE. Angry Birds and Cut the Rope need the iPhone and Mac apps; I-PHYRE has only 40 games.
- Run: the official `deepphy.runner.01_phyre_runner` of the pinned clone, unchanged, with released `conf/env_config_01_phyre.json` (8 × 8 grid, 8 radius levels, 10 attempts, 5 image-history attempts), `--format VLA`, on the 1 000 fold-0 test tasks of `ball_within_template` + `ball_cross_template`; one seeded run.
- Band: paper 9.62 % ± 2 SE of the difference between one 1 000-task run and the paper's 3-run mean.
- Disclosed: the paper's random MOCK agent (7.64 %) lies inside the band.
- Only instrumentation: per-task resume through the official `--reuse_log_dir`, a stop between tasks (exit 75), and `np.float = float` for `phyre/vis.py`. That alias is needed under DeepPHY's own numpy pin too, because phyre 0.2.2 uses it.

**Results**, success within k attempts in %:

| k | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Paper Table 6, Qwen-7B VLA (3-run mean) | 0.60 | 2.10 | 4.10 | 4.30 | 5.50 | 6.40 | 7.16 | 8.16 | 8.82 | **9.62** |
| Attempt 0 (`139-gfid`, env `b127-deepphy`) | 0.5 | 1.0 | 2.0 | 2.2 | 3.3 | 3.8 | 4.6 | 4.9 | 5.4 | **6.0** |
| Repair 1 (`139-gfid-r1`, env `b127-deepphy-gfid`) | 0.4 | 1.0 | 2.0 | 2.3 | 3.2 | 3.6 | 3.9 | 4.9 | 5.4 | **6.1** |
| MOCK (random agent, CPU, descriptive) | 0.7 | 1.4 | 2.1 | 2.6 | 3.0 | 4.2 | 4.4 | 5.1 | 5.8 | 6.20 (paper 7.64) |

Split by setup (within / cross template): attempt 0 6.8 / 5.2 %, repair 1 6.4 / 5.8 %. In both attempts there were no empty responses (no OOM or call failures) and no malformed answers. INVALID_INPUT: 3 190 of 9 723 attempts (attempt 0) and 3 159 of 9 733 (repair 1). Attempt 1 matches the paper; our agent falls behind over the later attempts, where the near-greedy model mostly sweeps neighbouring cells at radius 3 (the most common first answer is `Cell: 50, Radius: 3`).

**Diagnosis** ([repair decision](https://github.com/Sino-Huang/NovPhy/issues/139#issuecomment-6450386391)):
1. **INVALID_INPUT replay.** All 9 723 parsed actions of attempt 0 were replayed through PHYRE twice:
   - DeepPHY's pinned environment (py3.9, the released phyre 0.2.2 wheel, numpy 1.26.4);
   - our build.
   
   Both reproduce every recorded status: NOT_SOLVED 6 473, INVALID_INPUT 3 190, SOLVED 60, with the same task order. INVALID_INPUT is PHYRE's own verdict for a ball that overlaps an object or leaves the scene. The official runner counts it as a failed attempt, as in the paper's protocol. Neither the PHYRE build nor the shim changes any outcome. The MOCK gap (6.20 % against 7.64 %, ≈ 1.5 SE) is therefore sampling noise, not an environment offset.
2. **Decoding** is the same as DeepPHY's (above).
3. **VLM images.** We rendered the official initial scenes and gridded images of 20 tasks in DeepPHY's pinned stack (matplotlib 3.9.4, pillow 11.3.0, numpy 1.26.4).
   - Initial scenes, and therefore the keyframes: pixel-identical to ours.
   - Gridded 8 × 8 images: the cell-number **glyphs are rasterised differently** by our matplotlib 3.11.2 (up to 7 972 px per image, labels only).
   
   Repair 1 reran G-fid with the pinned matplotlib and pillow, whose renders are pixel-identical to the pinned stack (40 / 40). Result: 6.1 %, so the glyphs did not cause the miss.
4. Remaining unavoidable differences: the GPU (RTX 5090) and torch's cu128 build (cu126 has no sm_120 kernels).

**Infrastructure incident** (no effect on results):
- Repair run 1 was SIGKILLed at 07:00 because our wrapper's stop patch on the official per-task loop had been lost when the `np.float` shim was added.
- Fixed in commit `7a40a506`, with a refusal to resume on an unparsable summary. Tested: SIGTERM gives exit 75 in 1.7 s, and the resume is an exact continuation of the test-fold order.
- The resumed run redid the in-flight task from attempt 1 and processed the remaining 91 tasks. The killed run's 5 080 s are in the ledger.

## Compute

| item | GPU-h |
| --- | --- |
| Timing probe (`139-timing`, RTX 5090: 7B 3.78 s / decision + 34 s load, 3B 3.11 s + 17 s) | 0.03 |
| G-fid attempt 0 | 1.61 |
| G-fid repair 1 (1.41 h killed run + 0.18 h resume) | 1.59 |
| **Total** | **3.23** |

The funding decision allowed 8.8 GPU-h (the measured projection,
[#127 issuecomment-6443206040](https://github.com/Sino-Huang/NovPhy/issues/127#issuecomment-6443206040)); about 5.6
GPU-h return to the program remainder. CPU-side work: G-port (3B fp32, 11 min), MOCK check (22 min), replay and
rendering diagnosis.

## Deviations and disclosures

- DeepPHY's agent emits one action and never ranks candidates. T7's per-candidate score (the likelihood DeepPHY's agent assigns to emitting each launch in its own action language) was the declared adaptation. It was never run on NovPhy.
- The band is binomial and does not model run-to-run variation from hardware numerics under near-greedy decoding. The paper does not publish its 3-run spread.
- The G-fid environments differ from DeepPHY's `requirements_py39.txt` in python (3.11), torch's CUDA build (cu128) and numpy (2.4.6; outcomes shown identical). The pinned file cannot run the official PHYRE runner unshimmed.
- The Qwen2.5-VL-3B weights are under a non-commercial research licence: local use only, never redistributed.

## Decisions (posted on #139, also in `decisions.md`)

- P-license: [issuecomment-6437037536](https://github.com/Sino-Huang/NovPhy/issues/139#issuecomment-6437037536)
- P-route: [issuecomment-6437058547](https://github.com/Sino-Huang/NovPhy/issues/139#issuecomment-6437058547)
- P-adapter: [issuecomment-6437241396](https://github.com/Sino-Huang/NovPhy/issues/139#issuecomment-6437241396) and its addendum [issuecomment-6437519414](https://github.com/Sino-Huang/NovPhy/issues/139#issuecomment-6437519414)
- P-fid: [issuecomment-6437344622](https://github.com/Sino-Huang/NovPhy/issues/139#issuecomment-6437344622)
- Phase 0 gates: [issuecomment-6437774684](https://github.com/Sino-Huang/NovPhy/issues/139#issuecomment-6437774684)
- Timing probe (orchestrator): [issuecomment-6437800465](https://github.com/Sino-Huang/NovPhy/issues/139#issuecomment-6437800465)
- Funding: [#127 issuecomment-6443206040](https://github.com/Sino-Huang/NovPhy/issues/127#issuecomment-6443206040)
- P-fid repair 1: [issuecomment-6450386391](https://github.com/Sino-Huang/NovPhy/issues/139#issuecomment-6450386391)
- P-fid drop: [issuecomment-6463275977](https://github.com/Sino-Huang/NovPhy/issues/139#issuecomment-6463275977)
- Result comment: [issuecomment-6463317951](https://github.com/Sino-Huang/NovPhy/issues/139#issuecomment-6463317951)

## Reproduction

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate b127-deepphy && python scripts/b127_deepphy_port_reference.py --run          # G-port (CPU)
conda activate b127-deepphy && python scripts/b127_deepphy_fidelity.py --run                # G-fid attempt 0 (GPU)
conda activate b127-deepphy-gfid && python scripts/b127_deepphy_fidelity.py --run --repair 1 # repair 1 (GPU)
python scripts/b127_deepphy_fidelity.py --finish [--repair 1]                               # fidelity/gfid.json
conda activate novphy && source env.sh && python scripts/run_issue_139_deepphy.py --publish  # not-run summary
CUDA_VISIBLE_DEVICES="" python -m pytest -q tests/test_issue_139_deepphy.py
```
