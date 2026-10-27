## Decision P-license (2026-10-27): vendor XinrunXu/DeepPHY @ 0562104 (MIT) verbatim; Qwen2.5-VL-7B-Instruct (Apache-2.0) and Qwen2.5-VL-3B-Instruct (Qwen Research Licence, non-commercial research) weights used locally and never redistributed

**Trigger.** #139 Phase 0 needs the official DeepPHY code (the Angry Birds prompt and action interface, its answer parsing and the local Qwen2.5-VL inference path) for the adapter and G-port. It also needs the official PHYRE runner for G-fid, and the open-weight VLM weights DeepPHY reports.

**Facts.**
- Code: `XinrunXu/DeepPHY` `main` @ `056210492bca103ce8aff11aa225c40bf8e7c797` (2025-11-16, "Minor updates for GUI"; HEAD on 2026-10-27). The root `LICENSE` is plain MIT, "Copyright (c) 2025 Xinrun Xu". The repository has no other licence text. The Angry Birds and Cut the Rope apps are third-party iPhone/Mac games and are not part of the repository; T7 uses neither.
- Weights (Hugging Face, the models DeepPHY's local provider `deepphy/provider/llm/qwen.py` loads; paper Table 4):
  - `Qwen/Qwen2.5-VL-7B-Instruct` @ `cc594898137f460bfe9f0759e9844b3ce807cfb5`: card licence `apache-2.0`, 16.6 GB bf16 safetensors.
  - `Qwen/Qwen2.5-VL-3B-Instruct` @ `66285546d2b821cf421d4f5eb2576359d3770cd3`: `LICENSE` = **Qwen RESEARCH LICENSE AGREEMENT** (2024-09-19). It grants use "FOR NON-COMMERCIAL PURPOSES ONLY", where Non-Commercial means "for research or evaluation purposes only"; redistribution must carry the agreement and a notice. 7.5 GB.
  - Qwen2.5-VL-32B / 72B (DeepPHY also reports these) are not downloaded; Decision P-route gives the reason.

**Options.** (a) Vendor verbatim, as MIT allows. (b) Re-implement under G-port: not needed. (c) Drop with `license`: there are no grounds.

**Choice.** (a).
1. Vendor @ 0562104 into `world_model/baselines/deepphy/vendor/` with `VENDOR.json` (schema `issue_127_vendor_v1`, per-file upstream and vendored sha256, **no edits**):
   - `LICENSE` and `README.md`.
   - `deepphy/planner/prompt_strategy_05_angrybirds.py`: the adapter imports it by file location; it has no imports.
   - Reference copies whose functions the G-port reference executes from source: `deepphy/runner/05_angrybirds_runner.py` (`get_llm_action`, `parse_and_execute_action`, the parsing lines of `play_level_with_llm`), `deepphy/provider/llm/{qwen,base}.py`, `deepphy/utils/img_utils.py` and `conf/env_config_05_angrybirds.json`.
   - The PHYRE G-fid runs the official `deepphy.runner.01_phyre_runner` from the pinned clone `/mnt/array/sukaih/novphy-baselines-src/DeepPHY@0562104/`.
2. Weights go to `data/external/deepphy/` (gitignored) with `MANIFEST.json` (repo, revision, url, bytes, sha256, downloaded_at, licence). They are used only for this NovPhy research evaluation and are never committed or redistributed. The 3B model's non-commercial terms are satisfied by this use.
3. A dedicated env `b127-deepphy` holds the official stack. Its package list goes to artefact `envs/b127-deepphy.txt`. Decision P-route gives the details.

**Reason.** MIT permits use, modification and redistribution with the notice, which is kept verbatim. Apache-2.0 permits use of the 7B weights. The 3B licence permits non-commercial research evaluation, which is exactly T7's use, provided nothing is redistributed.
