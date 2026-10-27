## Decision P-adapter (2026-10-27): addendum. DeepPHY's `temperature=0.1` never reaches `generate` in its pinned stack; the configurations stay as declared; G-port checks the logits processing at the official effective decoding config

**Trigger.** The G-port reference was developing on a tiny random-init Qwen2.5-VL, before any released model ran and before any NovPhy frame was scored. It showed that the official call `pipe(messages, max_new_tokens=200, do_sample=True, temperature=0.1, return_full_text=False)` reaches `generate` with **only** `max_new_tokens`. This contradicts the decoding stated in Decision P-route (issuecomment-6437058547) and in P-adapter item 5 (issuecomment-6437241396).

**Facts** (transformers 4.52.4, DeepPHY's pin, `pipelines/image_text_to_text.py`):
- `_sanitize_parameters` forwards only `max_new_tokens` / `generate_kwargs` to `generate`. Every other keyword, here `do_sample` and `temperature`, goes into the preprocess parameters.
- The chat branch of `preprocess` ignores those parameters.
- `generate` therefore decodes with the checkpoint's `generation_config.json`: `do_sample` true, **`temperature` 1e-6**, `repetition_penalty` 1.05, `top_k` 50 (default), `top_p` 1.0. That is effectively greedy decoding with a repetition penalty.
- Measured on the tiny model: generate's processed scores have magnitude ~10^5 to 10^6 (logits / 1e-6). The adapter's `processed_rows` at (T 1e-6, rep 1.05) matches them with relative error 4.5e-7 in fp32 and 5.9e-4 in fp16.
- The paper (Section 5.1) and the code state temperature 0.1. Whatever DeepPHY's own runs did is reproduced by running its pinned code unchanged, which is what G-fid does.

**Options.** (a) Keep the three configurations as declared. `q7b-t01` stays DeepPHY's *stated* decoding: T 0.1 plus the checkpoint's repetition penalty 1.05. (b) Replace `q7b-t01` by the effective T = 1e-6. A likelihood at that temperature degenerates into a sum of logit gaps to the greedy token, scaled by 10^6, which is a different score family. (c) Patch DeepPHY's call: an edit of the official path.

**Choice.** (a); nothing that is already declared changes.
1. The configurations (`q7b`, `q7b-t01`, `q3b`), the score and the selection rule are unchanged.
2. G-port verifies the adapter's logits processing against generate's processed scores **at the official effective config** (T 1e-6, rep 1.05). The pass criterion is relative error ≤ 1e-3 against max(1, max |score|), at generate's finite (top-50) entries. The same function then computes `q7b-t01` at (T 0.1, rep 1.05). The teacher-forced log p check (≤ 1e-3 nats, T = 1) and the exact-input check are unchanged.
3. G-fid runs the official runner unchanged, so it uses this effective decoding. That is what DeepPHY's pinned code does; the frozen band (issuecomment-6437344622) is unchanged.
4. Correction to P-route: "temperature overridden to 0.1" is wrong under the pinned library. The effective local decoding is the checkpoint's generation config. This is disclosed in the plan.

**Reason.** The finding concerns how DeepPHY's own stack decodes, not how T7 scores. Keeping the declared configurations avoids any post-hoc change. G-port must verify the computation the official path actually performs, and the G-fid run executes exactly that path.
