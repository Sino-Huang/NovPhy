## #139 Phase 0 gate results (2026-10-27): G-port passes (incl. the released Qwen2.5-VL-3B); G-fid band frozen, GPU spec prepared; MOCK environment check 6.20 % (paper 7.64 %); CPU projection 5.9–12.3 GPU-h [INFERENCE]

**G-port** (`gport/gport.json`, schema `issue_127_gport_v1`, **pass**; reference `scripts/b127_deepphy_port_reference.py` in env `b127-deepphy`, transformers 4.52.4, torch 2.7.1, CPU):
- **Prompts and parser.** The official `get_llm_action`, `parse_and_execute_action`, the parsing statements of `play_level_with_llm` and `get_qwen_model_response` were executed from the vendored source (GUI calls, crop and model stubbed). On 4 prompt cases, system and user prompts, image lists and chat messages are **byte-identical** to the adapter's. On 26 answers (the 16 grid answers plus code fences, multiple brackets, `success()` and malformed strings), the parser agrees in all 26 cases. The 16 grid answers round-trip.
- **Model call.** On a fit-anchor decision frame (`issue-109-sc-fit-type010103-001`), the adapter's processor outputs (`input_ids`, `attention_mask`, `pixel_values`, `image_grid_thw`; 1 202 prompt tokens) **equal exactly** what the official pipeline passes to `generate`. Teacher-forced log p of the generated tokens against generate's logits:
  - released **Qwen2.5-VL-3B** (fp32): official answer `[shoot(angle=45, power=0.8)]` (14 tokens); max |Δ log p| **2.3e-6** nats; raw logits 3.4e-5; argmax agreement 1.0;
  - tiny random-init model run through the verbatim official `initialize_qwen_vlm` (fp16): 4.9e-4;
  - tiny model in fp32: 9.5e-7.
  
  All are within the 1e-3 tolerance.
- **Logits processing** at the official effective decoding config (addendum issuecomment-6437519414): relative error 6.9e-7 (3B). Candidate scores are batch-invariant within 3.8e-6 (tolerance 1e-4), and ordinals 8 and 12 tie exactly.
- Peak RSS of the 3B run was 27.7 GB, briefly over the ~20 GB host guideline: loading the fp32 model. No further CPU runs of the released models are planned.

**G-fid** (Decision P-fid issuecomment-6437344622). Band **[7.47, 11.77] %** (PHYRE VLA Qwen-7B within 10 attempts, paper 9.62 %), `fidelity/band.json` `sha256:ad5bfb65…3c30`. GPU job `139-gfid` (`scripts/gpu_window_issue_139_gfid.json`, resumable, est. 300 min, 2.1–4.5 GPU-h [INFERENCE]); not CPU-feasible; not enqueued.
- **Environment check** (descriptive, CPU): DeepPHY's MOCK agent through the same official runner on the same 1 000 tasks reaches **6.20 %** within 10 attempts (within 5.8 %, cross 6.6 %; `fidelity/mock.json`). The paper's MOCK value is 7.64 %; the gap is 1.4 pp, about 1.5 SE of the difference.
- Environment shim (P-fid): `np.float = float` for `phyre/vis.py`.

**Projection** (`projection.json`, [INFERENCE]):
- Forward FLOPs per decision from the configs (≤ 16 distinct answers × one 1 216-token sequence with 1 564 vision patches): 7B 0.31 PFLOP, 3B 0.16 PFLOP. These are mapped to an RTX 5090 at 100 / 40 sustained fp16 TFLOP/s, with model loads.
- Totals: selection 0.2–0.4, readouts 0.5–1.1, score 0.9–2.2, novelty 0.4–1.0 and validate 0.9–2.2 GPU-h; G-fid 2.1–4.5; timing probe 1.0. **Child total 5.9–12.3 GPU-h**, against the ≈ 73 GPU-h remainder and the 18 GPU-h child cap.
- CPU reference: the released 3B in fp32 on 4 threads scores one grid decision in 538 s.
- A ≤ 1 GPU-h timing probe is prepared (`scripts/gpu_window_issue_139_timing.json`, `shares_gpu` false, est. 20 min), with its decision drafted for the orchestrator.

Nothing has been scored on NovPhy policy or evaluation data, and no GPU time has been used. The evaluation split has not been read.
