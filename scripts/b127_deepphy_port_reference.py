"""#139 T7 G-port reference (env b127-deepphy; Decision P-adapter item 8): runs the official DeepPHY code and compares
the adapter (world_model/baselines/deepphy/novphy.py) with it on fixed inputs (CPU).

1. Prompts / messages: the official ``get_llm_action`` and ``parse_and_execute_action`` are executed from the vendored
   ``deepphy/runner/05_angrybirds_runner.py`` (AST-extracted; GUI calls, the iPhone-mirroring crop and the model call
   stubbed), the response-parsing statements of ``play_level_with_llm`` likewise, and ``get_qwen_model_response`` from
   the vendored ``provider/llm/qwen.py`` with a recording pipe; byte equality with the adapter's prompts, image list,
   messages and parser.
2. Model call: the official pipeline (``initialize_qwen_vlm`` executed verbatim on a tiny random-init Qwen2.5-VL with
   the 7B processor, float16; and the same ``pipeline(...)`` call in float32 on the tiny model and on the released
   Qwen2.5-VL-3B) answers DeepPHY's prompt for a fit-anchor decision frame through ``get_qwen_model_response``
   (``generate`` instrumented to keep its inputs, raw logits and processed scores). The adapter's ``Scorer`` (its own
   ``AutoProcessor``) must give the same model inputs exactly, the same log p of the generated tokens (teacher-forced,
   <= 1e-3 nats), the same processed scores at generate's finite entries (repetition penalty 1.05, temperature 0.1;
   <= 1e-3) and batch-invariant candidate scores (<= 1e-4).

Inputs: ``gport/inputs.pt`` written by ``scripts/run_issue_139_deepphy.py --gport-inputs`` (env novphy). Output:
``gport/gport.json`` (schema issue_127_gport_v1).
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import re
import shutil
import sys
import tempfile
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
ADAPTER = ROOT / "world_model/baselines/deepphy/novphy.py"
VENDOR = ROOT / "world_model/baselines/deepphy/vendor"
OUTPUT = ROOT / ".local-artifacts/issue-139-deepphy-v1"
TOL_LOGP = 1e-3
TOL_BATCH = 1e-4
GEN_SEED = 20260908


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


dn = load("b139_adapter", ADAPTER)


def sha(path):
    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


def log(message):
    print(f"[issue-139 gport] {message}", flush=True)


# ---------------------------------------------------------------------------------------------------------------
# 1. the official prompt / parse code, executed from the vendored source
# ---------------------------------------------------------------------------------------------------------------

class Official:
    """get_llm_action / parse_and_execute_action / the play_level_with_llm parsing statements from the vendored runner."""

    def __init__(self):
        path = VENDOR / "deepphy/runner/05_angrybirds_runner.py"
        source = path.read_text()
        tree = ast.parse(source)
        functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
        pm = dn.prompt_module()
        self.calls = {}
        ns = {"re": re, "random": random, "time": time, "print": lambda *a, **k: None,
              "system_prompt_angry_birds": pm.system_prompt_angry_birds,
              "get_user_prompt_angrybirds": pm.get_user_prompt_angrybirds,
              "MODEL_NAME": "Qwen2.5-VL-7B-Instruct",
              "crop_image_by_cm": self._crop, "get_model_response": self._model,
              "perform_action_swipe": lambda *a, **k: None, "perform_action_shoot": self._shoot}
        for name in ("get_llm_action", "parse_and_execute_action"):
            exec(compile(ast.Module(body=[functions[name]], type_ignores=[]), str(path), "exec"), ns)
        play = functions["play_level_with_llm"]
        wanted = {"cleaned_response", "matches", "final_action_str"}
        statements = [ast.get_source_segment(source, node) for node in ast.walk(play)
                      if isinstance(node, ast.Assign) and len(node.targets) == 1
                      and isinstance(node.targets[0], ast.Name) and node.targets[0].id in wanted]
        if len(statements) != 3:
            raise ValueError(f"expected the three parsing statements, found {len(statements)}")
        self.parse_source = statements
        body = ("def official_parse(raw_llm_response):\n    " + statements[0] + "\n    " + statements[1]
                + "\n    if not matches:\n        return None\n    " + statements[2] + "\n    return final_action_str\n")
        exec(compile(body, str(path), "exec"), ns)
        self.ns = ns

    def _crop(self, path, crop_top_cm=0, **kwargs):
        self.calls.setdefault("crop", []).append({"path": path, "crop_top_cm": crop_top_cm})
        return True

    def _model(self, image_descriptions, system_prompt, user_prompt, model_name):
        self.calls["model"] = (image_descriptions, system_prompt, user_prompt, model_name)
        return "[shoot(angle=45, power=0.8)]"

    def _shoot(self, app_name, start_point, angle_degrees, power_ratio):
        self.calls["shoot"] = (angle_degrees, power_ratio)

    def prompt(self, paths, level_data, history):
        self.calls.pop("model", None)
        self.ns["get_llm_action"](paths, level_data, history, {})
        return self.calls["model"]

    def parse(self, raw):
        """(final action string or None, terminal flag, (angle, power) or None)."""
        final = self.ns["official_parse"](raw)
        if final is None:
            return None, None, None
        self.calls.pop("shoot", None)
        terminal = self.ns["parse_and_execute_action"](final, "app", {"slingshot_pos": {"x": 0.4, "y": 0.75}})
        return final, terminal, self.calls.get("shoot")


PROMPT_CASES = [
    {"bird_type": ["red"], "history": [], "paths": ["/tmp/frame-0.png"]},
    {"bird_type": ["red", "red", "red"], "history": [], "paths": ["/tmp/frame-0.png"]},
    {"bird_type": ["red", "yellow", "blue", "black"], "history": ["shoot(angle=45, power=0.8)"],
     "paths": ["/tmp/a.png", "/tmp/b.png"]},
    {"bird_type": None, "history": [], "paths": ["/tmp/frame-0.png"]},
]


def parse_cases(grid):
    answers = [dn.answer_text(item["action"]) for item in grid]
    return answers + [
        "[shoot(angle=45, power=0.8)]", "```python\n[shoot(angle=30, power=1.0)]\n```",
        "Plan: [shoot(angle=10, power=0.5)] then [shoot(angle=20, power=0.9)]", "[success()]", "no action here",
        "[shoot(angle = 45, power=0.8)]", "[shoot(angle=-5.5, power=1)]", "[jump()]", "", "[ shoot( angle=12,power=0.25 ) ]"]


def prompt_parse_gport(grid):
    official = Official()
    qwen = load("deepphy_vendor_qwen", VENDOR / "deepphy/provider/llm/qwen.py")
    prompts = []
    for case in PROMPT_CASES:
        level = {} if case["bird_type"] is None else {"bird_type": case["bird_type"]}
        descriptions, system_prompt, user_prompt, _ = official.prompt(list(case["paths"]), level, list(case["history"]))
        ours_system, ours_user = dn.prompts(case["bird_type"] or ["red"], case["history"])
        ours_desc = dn.image_descriptions(list(case["paths"]), case["history"])
        recorded = {}

        def pipe(msgs, **kwargs):
            recorded.update(messages=msgs, kwargs=kwargs)
            return [{"generated_text": "[shoot(angle=45, power=0.8)]"}]

        qwen.get_qwen_model_response(pipe, descriptions, system_prompt, user_prompt)
        prompts.append({
            "case": case, "system_equal": ours_system == system_prompt, "user_equal": ours_user == user_prompt,
            "images_equal": ours_desc == descriptions,
            "messages_equal": dn.messages(ours_desc, ours_system, ours_user) == recorded["messages"],
            "official_call_kwargs": recorded["kwargs"],
            "system_sha256": "sha256:" + hashlib.sha256(system_prompt.encode()).hexdigest(),
            "user_sha256": "sha256:" + hashlib.sha256(user_prompt.encode()).hexdigest()})
    crops = official.calls.get("crop", [])
    parses = []
    for raw in parse_cases(grid):
        final, terminal, shoot = official.parse(raw)
        ours = dn.parse_response(raw)
        if final is None:
            equal = ours is None
        else:
            equal = (ours is not None and ours["action"] == final and ours["success"] == bool(terminal)
                     and (None if ours["angle"] is None else (ours["angle"], ours["power"])) == shoot)
        parses.append({"raw": raw, "official": None if final is None else {"action": final, "terminal": terminal,
                                                                           "shoot": shoot},
                       "ours": ours, "equal": bool(equal)})
    roundtrip = [{"ordinal": item["ordinal"], "answer": dn.answer_text(item["action"]),
                  "parsed": official.parse(dn.answer_text(item["action"]))[2],
                  "rendered": [round(dn.launch_angle_power(item["action"])[0]),
                               round(dn.launch_angle_power(item["action"])[1], 2)]} for item in grid]
    ok = (all(p["system_equal"] and p["user_equal"] and p["images_equal"] and p["messages_equal"] for p in prompts)
          and all(p["equal"] for p in parses)
          and all(list(r["parsed"]) == [float(v) for v in r["rendered"]] for r in roundtrip))
    return {"prompts": prompts, "official_crop_calls": crops,
            "crop_note": "the official runner crops 1.10 cm off the current screenshot (iPhone-mirroring title bar); "
                         "not applied to NovPhy frames (Decision P-adapter item 1)",
            "parsing_statements": official.parse_source, "parses": parses, "grid_roundtrip": roundtrip,
            "pass": bool(ok)}


# ---------------------------------------------------------------------------------------------------------------
# 2. the model call
# ---------------------------------------------------------------------------------------------------------------

def tiny_model(directory):
    """A random-init Qwen2_5_VLForConditionalGeneration (seed 0, 2 text + 2 vision layers) with the 7B's processor,
    tokenizer, chat template and generation config."""
    from transformers import AutoConfig, AutoModelForImageTextToText
    src = dn.model_dir("Qwen2.5-VL-7B-Instruct")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name in ("chat_template.json", "merges.txt", "preprocessor_config.json", "tokenizer.json",
                 "tokenizer_config.json", "vocab.json", "generation_config.json"):
        shutil.copy(src / name, directory / name)
    cfg = AutoConfig.from_pretrained(src)
    for c in (cfg, cfg.text_config):
        c.hidden_size, c.intermediate_size, c.num_hidden_layers = 64, 128, 2
        c.num_attention_heads, c.num_key_value_heads, c.max_window_layers = 4, 2, 2
        c.rope_scaling = {**c.rope_scaling, "mrope_section": [2, 3, 3]}
    v = cfg.vision_config
    v.depth, v.hidden_size, v.intermediate_size, v.num_heads, v.out_hidden_size = 2, 32, 64, 2, 64
    v.fullatt_block_indexes = [1]
    torch.manual_seed(0)
    AutoModelForImageTextToText.from_config(cfg).save_pretrained(directory)
    shutil.copy(src / "generation_config.json", directory / "generation_config.json")  # save_pretrained wrote defaults
    return directory


def instrument(pipe):
    captured = {}
    original = pipe.model.generate

    def generate(**kwargs):
        captured["inputs"] = {k: v.detach().clone() for k, v in kwargs.items() if torch.is_tensor(v)}
        captured["kwargs"] = {k: v for k, v in kwargs.items() if not torch.is_tensor(v) and k != "generation_config"}
        config = kwargs.get("generation_config")
        captured["generation_config"] = {k: getattr(config, k, None) for k in
                                         ("do_sample", "temperature", "top_k", "top_p", "repetition_penalty",
                                          "max_new_tokens")} if config is not None else None
        out = original(**kwargs, return_dict_in_generate=True, output_logits=True, output_scores=True)
        captured["out"] = out
        return out.sequences

    pipe.model.generate = generate
    return captured


def model_case(name, pipe, path, frame_path, grid, batch_check):
    qwen = load("deepphy_vendor_qwen", VENDOR / "deepphy/provider/llm/qwen.py")
    system_prompt, user_prompt = dn.prompts()
    descriptions = dn.image_descriptions([frame_path])
    captured = instrument(pipe)
    torch.manual_seed(GEN_SEED)
    t = time.monotonic()
    text = qwen.get_qwen_model_response(pipe, descriptions, system_prompt, user_prompt)
    generate_seconds = time.monotonic() - t
    from transformers import AutoProcessor
    scorer = dn.Scorer(path, device="cpu", batch=1, model=pipe.model, processor=AutoProcessor.from_pretrained(str(path)))
    ours = scorer.inputs(dn.messages(descriptions, system_prompt, user_prompt))
    theirs = captured["inputs"]
    inputs_equal = {k: bool(k in theirs and ours[k].shape == theirs[k].shape and torch.equal(ours[k], theirs[k]))
                    for k in ours}
    out = captured["out"]
    start = int(ours["input_ids"].shape[1])
    generated = out.sequences[0, start:]
    (ids, rows), = scorer.rows_for_ids(ours, [generated])
    gen_logits = torch.stack([lg[0] for lg in out.logits]).float()
    gen_scores = torch.stack([sc[0] for sc in out.scores]).float()
    ref_logp = torch.log_softmax(gen_logits, -1).gather(1, generated[:, None])[:, 0]
    our_logp = dn.answer_token_logprobs(rows, ids, start)
    passed = captured["generation_config"]  # None: the pipeline passes no generation_config, generate uses the model's
    gc = pipe.model.generation_config
    effective = {k: getattr(gc, k) for k in ("do_sample", "temperature", "top_k", "top_p", "repetition_penalty")}
    processed = dn.processed_rows(rows, ids, start, temperature=effective["temperature"],
                                  repetition_penalty=effective["repetition_penalty"])
    finite = torch.isfinite(gen_scores)
    logp_delta = float((our_logp - ref_logp).abs().max())
    processed_abs = float((processed[finite] - gen_scores[finite]).abs().max())
    processed_rel = processed_abs / max(1.0, float(gen_scores[finite].abs().max()))
    raw_delta = float((rows - gen_logits).abs().max())
    result = {
        "model": name, "dtype": str(pipe.model.dtype), "device": str(pipe.device), "prompt_tokens": start,
        "official_text": text, "official_parse": dn.parse_response(text), "generated_tokens": int(len(generated)),
        "generate_kwargs": {k: (v if isinstance(v, (int, float, bool, str, type(None))) else str(v))
                            for k, v in captured["kwargs"].items()},
        "generation_config_kwarg": passed, "effective_generation_config": effective,
        "finding": ("transformers 4.52.4's image-text-to-text pipeline routes get_qwen_model_response's do_sample=True / "
                    "temperature=0.1 into the preprocess parameters, which the chat branch ignores; generate receives "
                    "only max_new_tokens and decodes with the checkpoint's generation_config (effective_generation_config)"),
        "generate_seconds": generate_seconds, "inputs_equal": inputs_equal,
        "max_abs_raw_logit_delta": raw_delta, "max_abs_logp_delta": logp_delta,
        "processed_max_abs_delta": processed_abs, "processed_rel_delta": processed_rel,
        "processed_finite_entries": int(finite.sum()),
        "argmax_agree": float((rows.argmax(-1) == gen_logits.argmax(-1)).float().mean()),
    }
    ok = all(inputs_equal.values()) and logp_delta <= TOL_LOGP and processed_rel <= TOL_LOGP
    if batch_check:
        variants = {"q7b": {"temperature": 1.0, "repetition_penalty": None},
                    "q7b-t01": {"temperature": 0.1, "repetition_penalty": 1.05}}
        actions = [item["action"] for item in grid]
        t = time.monotonic()
        one, info = scorer.score(frame_path, actions, variants)
        result["cpu_seconds_per_decision_batch1"] = time.monotonic() - t
        scorer.batch = 16
        many, _ = scorer.score(frame_path, actions, variants)
        delta = max(abs(a - b) for v in variants for a, b in zip(one[v], many[v]))
        ties = [(i, j) for i in range(len(actions)) for j in range(i + 1, len(actions))
                if info["answers"][i] == info["answers"][j]]
        result.update(batch_invariance={"max_abs_delta": delta, "tolerance": TOL_BATCH, "pass": delta <= TOL_BATCH},
                      grid_scores=one, answers=info["answers"],
                      exact_ties={"pairs": ties, "equal": all(one[v][i] == one[v][j] for v in variants
                                                               for i, j in ties)})
        ok = ok and delta <= TOL_BATCH and result["exact_ties"]["equal"]
    elif grid:
        t = time.monotonic()
        scores, info = scorer.score(frame_path, [item["action"] for item in grid],
                                    {"q3b": {"temperature": 1.0, "repetition_penalty": None}})
        result.update(cpu_seconds_per_decision_batch1=time.monotonic() - t, grid_scores=scores,
                      answers=info["answers"])
    result["pass"] = bool(ok)
    return result


def run(output, real=True):
    from transformers import pipeline
    output = Path(output)
    inputs = torch.load(output / "gport/inputs.pt", map_location="cpu", weights_only=False)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    grid = inputs["grid"]
    began = time.monotonic()
    record = {"schema": "issue_127_gport_v1", "issue": 139, "arm": "T7 DeepPHY-VLM",
              "G_port_rule": ("prompts, image list, messages and parser byte-identical to the official code executed from "
                              "the vendored source; model inputs exactly equal to the official pipeline's generate "
                              f"inputs; teacher-forced log p of the generated tokens <= {TOL_LOGP} nats from generate's "
                              "logits; the adapter's logits processing (repetition penalty, temperature) at the "
                              "official effective decoding config within "
                              f"{TOL_LOGP} relative to max(1, max |score|) of generate's processed scores at their "
                              "finite entries; "
                              f"candidate scores batch-invariant <= {TOL_BATCH}; exact ties for equal answers"),
              "inputs": {"fit_anchor_member": inputs["member"], "sha256": sha(output / "gport/inputs.pt")},
              "cases": {}}
    record["cases"]["prompt_parse"] = prompt_parse_gport(grid)
    log(f"prompt / parse: {record['cases']['prompt_parse']['pass']}")
    with tempfile.TemporaryDirectory(prefix="b139-gport-") as tmp:
        frame_path = dn.write_screenshot(inputs["frame"].numpy(), Path(tmp) / "frame.png")
        tiny = tiny_model(Path(tmp) / "tiny")
        qwen = load("deepphy_vendor_qwen", VENDOR / "deepphy/provider/llm/qwen.py")
        os.environ["QWEN_MODEL_7B_PATH"] = str(tiny)
        pipe = qwen.initialize_qwen_vlm("Qwen2.5-VL-7B-Instruct")  # official, verbatim (float16)
        record["cases"]["tiny_fp16_official_init"] = model_case("tiny random Qwen2.5-VL (official initialize_qwen_vlm, "
                                                                "float16)", pipe, tiny, frame_path, grid, False)
        del pipe
        pipe = pipeline(model=str(tiny), task="image-text-to-text", model_kwargs={"torch_dtype": torch.float32},
                        trust_remote_code=True)
        record["cases"]["tiny_fp32"] = model_case("tiny random Qwen2.5-VL (float32)", pipe, tiny, frame_path, grid, True)
        del pipe
        log(f"tiny: fp16 {record['cases']['tiny_fp16_official_init']['pass']}, "
            f"fp32 {record['cases']['tiny_fp32']['pass']}")
        if real:
            path = dn.model_dir("Qwen2.5-VL-3B-Instruct")
            pipe = pipeline(model=str(path), task="image-text-to-text", model_kwargs={"torch_dtype": torch.float32},
                            trust_remote_code=True)
            record["cases"]["qwen3b_fp32"] = model_case("Qwen2.5-VL-3B-Instruct (released, float32)", pipe, path,
                                                        frame_path, grid, False)
            del pipe
            log(f"3B: {record['cases']['qwen3b_fp32']['pass']}")
    import transformers
    record.update(
        tolerance={"logp_nats": TOL_LOGP, "processed": TOL_LOGP, "batch": TOL_BATCH},
        notes=["The 7B in float32 needs ~33 GB RSS, over the shared host's 20 GB CPU budget; the 3B (15 GB) runs the "
               "same Qwen2_5_VLForConditionalGeneration code with released weights. The tiny model covers the official "
               "initialize_qwen_vlm verbatim (float16) and batch padding.",
               "Official inference on GPU runs in float16 (initialize_qwen_vlm); the CPU G-port compares float32 "
               "(the official pipeline call with torch_dtype float32) besides the verbatim float16 tiny case."],
        adapter_sha256=sha(ADAPTER), reference_script_sha256=sha(Path(__file__)),
        vendor_sha256=sha(VENDOR / "VENDOR.json"), torch=torch.__version__, transformers=transformers.__version__,
        seconds=time.monotonic() - began, written_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    record["pass"] = bool(all(c["pass"] for c in record["cases"].values()))
    (output / "gport").mkdir(parents=True, exist_ok=True)
    tmp = output / "gport/gport.json.tmp"
    tmp.write_text(json.dumps(record, indent=1, sort_keys=True, default=str) + "\n")
    os.replace(tmp, output / "gport/gport.json")
    log(f"G-port pass {record['pass']} ({record['seconds']:.0f} s)")
    return 0 if record["pass"] else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", action="store_true", required=True)
    parser.add_argument("--no-real", action="store_true", help="skip the released 3B case (development only)")
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args(argv)
    return run(args.output, real=not args.no_real)


if __name__ == "__main__":
    sys.exit(main())
