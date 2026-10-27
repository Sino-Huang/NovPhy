"""#139 T7 DeepPHY-VLM: NovPhy adapter of DeepPHY's Angry Birds agent (XinrunXu/DeepPHY @ 0562104, MIT; ``vendor/``)
and the scoring worker (Decisions P-route issuecomment-6437058547, P-adapter on #139).

Prompt: DeepPHY's own Angry Birds VLA prompt, exactly as ``get_llm_action`` of ``deepphy/runner/05_angrybirds_runner.py``
builds it for the first shot of a level (system prompt ``system_prompt_angry_birds``, user prompt
``get_user_prompt_angrybirds`` with the current bird ``red`` and an empty queue, no history images), and the message
layout of ``deepphy/provider/llm/qwen.py`` (system text; per image its label then ``{"type": "image", "url": path}``;
then the user prompt). The image is the NovPhy decision frame (uint8 [3, 240, 320]) at the engine's 640 x 480
screenshot size (2x pixel replication), written as PNG and passed by path as DeepPHY passes its screenshot.

Action: DeepPHY's action code ``shoot(angle=X, power=Y)``. A NovPhy launch (drag_x, drag_y) px from the slingshot,
screen y down, maps to angle = atan2(drag_y, -drag_x) in degrees (0 = horizontal right, 90 = up) and power = the pull
as a fraction of the BirdRed drag radius (18.491484 px; launch speed is proportional to the clamped pull, #94), capped
at 1. The angle is rendered as an integer (the system prompt's "An integer from 0 to 90") and the power with two
decimals (``round(power, 2)``, DeepPHY's mock-agent rendering); candidates with the same rendering tie exactly.

Score: the teacher-forced log-probability of the candidate's answer ``[shoot(angle=X, power=Y)]<|im_end|>`` after the
prompt, summed over its tokens, under the configuration's logits processing (``temperature`` and optional
``repetition_penalty``, HF's order; truncation warpers excluded). Higher = preferred (``eb.score_rows`` value = -score).

The worker (``python world_model/baselines/deepphy/novphy.py --worker REQUEST RESPONSE``) runs in env b127-deepphy
(DeepPHY's pinned transformers 4.52.4): it loads the checkpoint as DeepPHY's pipeline does (``AutoProcessor`` +
``AutoModelForImageTextToText``, float16 on GPU), builds the inputs as the pipeline's ``preprocess`` does and calls the
model's forward. This module imports nothing outside the standard library at import time except numpy, so the novphy
env imports it for the prompt / action / parse helpers.
"""
from __future__ import annotations

import argparse
import importlib.util
import math
import os
from pathlib import Path
import re
import signal
import sys
import tempfile
import time

import numpy as np

HERE = Path(__file__).resolve().parent
VENDOR = HERE / "vendor"
ROOT = HERE.parents[2]
WEIGHTS = ROOT / "data" / "external" / "deepphy"
WORKER_PYTHON = Path(os.environ.get("B139_PYTHON", Path.home() / "miniconda3/envs/b127-deepphy/bin/python"))
EXIT_PAUSED = 75

CLAMP_RADIUS_PX = 18.491484  # .local-artifacts/issue-94-launch-power-v1/engine_action_check.json clamp_radius_px
SCREENSHOT_SCALE = 2  # 320 x 240 frame -> the engine's 640 x 480 screenshot (as #128 A3-img)
BIRD_TYPES = ("red",)  # every NovPhy bird is BirdRed; one launch is ranked (DeepPHY's first-shot prompt)
CURRENT_LABEL = "This is the current game state. Plan your next shot based on this image."
END_OF_TURN = "<|im_end|>"

MODELS = {
    "Qwen2.5-VL-7B-Instruct": {"repo": "Qwen/Qwen2.5-VL-7B-Instruct",
                               "revision": "cc594898137f460bfe9f0759e9844b3ce807cfb5",
                               "parameters": 8_292_166_656, "license": "apache-2.0"},
    "Qwen2.5-VL-3B-Instruct": {"repo": "Qwen/Qwen2.5-VL-3B-Instruct",
                               "revision": "66285546d2b821cf421d4f5eb2576359d3770cd3",
                               "parameters": 3_754_622_976, "license": "Qwen RESEARCH LICENSE AGREEMENT"},
}
# Decision P-adapter: <= 3 configurations in declaration order (model, score processing)
CONFIGS = {
    "q7b": {"model": "Qwen2.5-VL-7B-Instruct", "temperature": 1.0, "repetition_penalty": None},
    "q7b-t01": {"model": "Qwen2.5-VL-7B-Instruct", "temperature": 0.1, "repetition_penalty": 1.05},
    "q3b": {"model": "Qwen2.5-VL-3B-Instruct", "temperature": 1.0, "repetition_penalty": None},
}
DTYPE = "float16"  # DeepPHY's initialize_qwen_vlm torch_dtype
GPU_BATCH = 16  # candidate sequences per forward on the GPU (one decision)


def model_dir(name):
    return WEIGHTS / name


# ---------------------------------------------------------------------------------------------------------------
# DeepPHY prompt, messages, action code, parsing
# ---------------------------------------------------------------------------------------------------------------

_PROMPT_MODULE = None


def prompt_module():
    """The vendored deepphy/planner/prompt_strategy_05_angrybirds.py (no imports), loaded by file location."""
    global _PROMPT_MODULE
    if _PROMPT_MODULE is None:
        path = VENDOR / "deepphy/planner/prompt_strategy_05_angrybirds.py"
        spec = importlib.util.spec_from_file_location("deepphy_vendor_prompt_05_angrybirds", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _PROMPT_MODULE = module
    return _PROMPT_MODULE


def prompts(bird_types=BIRD_TYPES, history=()):
    """(system_prompt, user_prompt) as get_llm_action builds them for the shot after ``history`` (action strings)."""
    pm = prompt_module()
    bird_types = list(bird_types)
    index = len(history)
    if index < len(bird_types):
        current_bird, upcoming_birds = bird_types[index], bird_types[index + 1:]
    else:
        current_bird, upcoming_birds = "None", []
    state = "\n".join([f"*   **Current Bird on Slingshot:** `{current_bird}`",
                       f"*   **Remaining Birds in Queue:** `{upcoming_birds if upcoming_birds else 'None'}`"])
    instructions = "\n".join([
        "*   `shoot(angle=X, power=Y)`: Launch the current bird. `X` is an angle in degrees (0 to 90). `Y` is a power "
        "level (0.0 to 1.0).",
        "*   `success()`: Call this ONLY if you are certain all pigs have been destroyed and no more shots are needed."])
    if history:
        history_text = "\n".join([f"- Shot {i + 1}: `{h}`" for i, h in enumerate(history)])
    else:
        history_text = "This is the first shot of the level. No actions have been taken yet."
    user = pm.get_user_prompt_angrybirds(state_analysis_prompt=state, history_prompt_text=history_text,
                                         action_instructions_prompt=instructions)
    return pm.system_prompt_angry_birds, user


def image_descriptions(paths, history=()):
    """get_llm_action's image list: one labelled entry per earlier screenshot, then the current one."""
    out = [{"path": path, "label": f"This was the game state before Shot {i + 1}. The action taken was `{history[i]}`."}
           for i, path in enumerate(paths[:-1])]
    out.append({"path": paths[-1], "label": CURRENT_LABEL})
    return out


def messages(descriptions, system_prompt, user_prompt):
    """get_qwen_model_response's chat messages."""
    content = []
    for item in descriptions:
        content.append({"type": "text", "text": item["label"]})
        content.append({"type": "image", "url": item["path"]})
    content.append({"type": "text", "text": user_prompt})
    return [{"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": content}]


def launch_angle_power(action):
    """(angle in degrees, power in [0, 1]) of a NovPhy launch in DeepPHY's shoot() convention."""
    dx, dy = float(action["drag_x"]), float(action["drag_y"])
    return math.degrees(math.atan2(dy, -dx)), min(1.0, math.hypot(dx, dy) / CLAMP_RADIUS_PX)


def action_code(action):
    angle, power = launch_angle_power(action)
    return f"shoot(angle={round(angle)}, power={round(power, 2)})"


def answer_text(action):
    """The answer DeepPHY's prompt requests: only the action code in brackets."""
    return f"[{action_code(action)}]"


SHOOT = re.compile(r"shoot\s*\(\s*angle=(-?\d+\.?\d*)\s*,\s*power=(\d+\.?\d*)\s*\)")


def parse_response(raw):
    """DeepPHY's parsing of a raw answer (play_level_with_llm + parse_and_execute_action): the last bracketed action
    after removing code fences; {"action", "success", "angle", "power"} or None when no bracketed action exists."""
    cleaned = re.sub(r'```(?:python)?\n(.*?)\n```', r'\1', raw, flags=re.DOTALL).strip()
    matches = re.findall(r'\[(.*?)\]', cleaned)
    if not matches:
        return None
    final = matches[-1].strip()
    if "success" in final:
        return {"action": final, "success": True, "angle": None, "power": None}
    shoot = SHOOT.search(final)
    return {"action": final, "success": False, "angle": float(shoot.group(1)) if shoot else None,
            "power": float(shoot.group(2)) if shoot else None}


def screenshot(frame):
    """uint8 [3, H, W] decision frame -> uint8 [2H, 2W, 3] screenshot (2x pixel replication)."""
    hwc = np.ascontiguousarray(np.asarray(frame, dtype=np.uint8).transpose(1, 2, 0))
    return hwc.repeat(SCREENSHOT_SCALE, axis=0).repeat(SCREENSHOT_SCALE, axis=1)


def write_screenshot(frame, path):
    from PIL import Image
    Image.fromarray(screenshot(frame)).save(path, format="PNG")
    return str(path)


# ---------------------------------------------------------------------------------------------------------------
# score definition (pure torch; teacher-forced answer log-probability)
# ---------------------------------------------------------------------------------------------------------------

def processed_rows(rows, ids, start, temperature=1.0, repetition_penalty=None):
    """Processed next-token scores [n, V] (float32) from the raw logit rows [n, V] that predict ids[start:] (row j is
    the model output at position start - 1 + j): HF RepetitionPenaltyLogitsProcessor over the prefix ids[:start + j]
    (if a penalty is given), then TemperatureLogitsWarper. Truncation warpers (top-k / top-p) are not applied."""
    import torch
    rows = rows.float().clone()
    if repetition_penalty is not None:
        for j in range(len(rows)):
            prefix = ids[:start + j]
            score = rows[j].gather(0, prefix)
            score = torch.where(score < 0, score * repetition_penalty, score / repetition_penalty)
            rows[j] = rows[j].scatter(0, prefix, score)
    return rows / temperature


def answer_token_logprobs(rows, ids, start, temperature=1.0, repetition_penalty=None):
    """log p of each answer token ids[start:] under the processed distribution (float32 [n])."""
    import torch
    processed = processed_rows(rows, ids, start, temperature, repetition_penalty)
    return torch.log_softmax(processed, dim=-1).gather(1, ids[start:, None])[:, 0]


def answer_score(rows, ids, start, temperature=1.0, repetition_penalty=None):
    """The per-candidate score: the summed answer log-probability (a Python float; -inf / nan stay nonfinite)."""
    return float(answer_token_logprobs(rows, ids, start, temperature, repetition_penalty).double().sum())


# ---------------------------------------------------------------------------------------------------------------
# the scorer (env b127-deepphy)
# ---------------------------------------------------------------------------------------------------------------

def torch_dtype(name):
    import torch
    return {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}[name]


class Scorer:
    """One DeepPHY checkpoint: processor + model loaded as DeepPHY's pipeline loads them; scores candidate answers."""

    def __init__(self, path, device=None, dtype=DTYPE, batch=None, model=None, processor=None):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = processor or AutoProcessor.from_pretrained(str(path))
        if model is None:
            model = AutoModelForImageTextToText.from_pretrained(str(path), torch_dtype=torch_dtype(dtype))
            model = model.to(self.device)
        self.model = model.eval()
        self.batch = batch or (GPU_BATCH if str(self.device).startswith("cuda") else 1)
        self.tokenizer = self.processor.tokenizer

    def inputs(self, msgs):
        """The pipeline's preprocess of a chat (image-text-to-text, transformers 4.52.4)."""
        return self.processor.apply_chat_template(msgs, add_generation_prompt=True, continue_final_message=False,
                                                  return_tensors="pt", tokenize=True, return_dict=True)

    def answer_ids(self, answer):
        return self.tokenizer(answer + END_OF_TURN, add_special_tokens=False)["input_ids"]

    def answer_rows(self, model_inputs, answers):
        """[(ids [L], rows [n, V]) per answer string]: rows_for_ids on its tokens (answer + <|im_end|>)."""
        return self.rows_for_ids(model_inputs, [self.answer_ids(a) for a in answers])

    def rows_for_ids(self, model_inputs, continuations):
        """[(ids [L], rows [n, V]) per continuation token list]: the model forward on prompt + continuation
        (right-padded batches of ``self.batch``); rows = the float32 logits at positions start - 1 .. L - 2 (they
        predict the continuation tokens), kept on the model's device."""
        import torch
        prompt = model_inputs["input_ids"][0]
        start = len(prompt)
        seqs = [torch.cat([prompt, torch.as_tensor(c, dtype=prompt.dtype)]) for c in continuations]
        out = []
        pad = self.tokenizer.pad_token_id
        for i in range(0, len(seqs), self.batch):
            chunk = seqs[i:i + self.batch]
            width = max(len(s) for s in chunk)
            ids = torch.full((len(chunk), width), pad, dtype=prompt.dtype)
            mask = torch.zeros((len(chunk), width), dtype=model_inputs["attention_mask"].dtype)
            for b, s in enumerate(chunk):
                ids[b, :len(s)] = s
                mask[b, :len(s)] = 1
            with torch.no_grad():
                logits = self.model(input_ids=ids.to(self.device), attention_mask=mask.to(self.device),
                                    pixel_values=model_inputs["pixel_values"].repeat(len(chunk), 1).to(self.device),
                                    image_grid_thw=model_inputs["image_grid_thw"].repeat(len(chunk), 1).to(self.device),
                                    use_cache=False).logits
            for b, s in enumerate(chunk):
                out.append((s.to(self.device), logits[b, start - 1:len(s) - 1].float()))
            del logits
        return out

    def score(self, image_path, actions, variants):
        """{variant: [score per action]} of one decision; variants: {name: {"temperature", "repetition_penalty"}}.
        Each distinct answer string is scored once, so candidates with the same rendering tie exactly."""
        system_prompt, user_prompt = prompts()
        model_inputs = self.inputs(messages(image_descriptions([image_path]), system_prompt, user_prompt))
        start = int(model_inputs["input_ids"].shape[1])
        answers = [answer_text(a) for a in actions]
        unique = list(dict.fromkeys(answers))
        pairs = dict(zip(unique, self.answer_rows(model_inputs, unique)))
        scores = {}
        for name, v in variants.items():
            by_answer = {a: answer_score(rows, ids, start, v["temperature"], v["repetition_penalty"])
                         for a, (ids, rows) in pairs.items()}
            scores[name] = [by_answer[a] for a in answers]
        return scores, {"prompt_tokens": start, "answers": answers}

# ---------------------------------------------------------------------------------------------------------------
# worker protocol (runner in env novphy <-> worker in env b127-deepphy)
# ---------------------------------------------------------------------------------------------------------------

REQUEST_SCHEMA = "issue_139_worker_request_v1"
RESPONSE_SCHEMA = "issue_139_worker_response_v1"
_STOP = {"requested": False}


def _request_stop(signum, frame):
    _STOP["requested"] = True


def _set_policy(policy):
    import torch
    torch.backends.cudnn.benchmark = policy["cudnn_benchmark"]
    torch.backends.cudnn.deterministic = policy["cudnn_deterministic"]
    torch.backends.cudnn.allow_tf32 = policy["cudnn_allow_tf32"]
    torch.backends.cuda.matmul.allow_tf32 = policy["matmul_allow_tf32"]


def _save(path, payload):
    import torch
    tmp = Path(str(path) + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def _flops(scorer, image_path, actions, variants):
    from torch.utils.flop_counter import FlopCounterMode
    with FlopCounterMode(display=False) as counter:
        scorer.score(image_path, actions, variants)
    return counter.get_total_flops()


def run_worker(request_path, response_path):
    """Score every request item (resumable per item: SIGTERM / SIGUSR1 -> save after the current item -> exit 75)."""
    import torch
    import transformers
    request = torch.load(request_path, map_location="cpu", weights_only=False)
    if request.get("schema") != REQUEST_SCHEMA:
        raise ValueError("worker request schema differs")
    for signum in (signal.SIGTERM, signal.SIGUSR1):
        signal.signal(signum, _request_stop)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    _set_policy(request["policy"])
    response_path = Path(response_path)
    partial = Path(str(response_path) + ".partial")
    state = (torch.load(partial, map_location="cpu", weights_only=False) if partial.is_file()
             else {"scores": {}, "answers": {}, "seconds": {}, "prompt_tokens": {}})
    began = time.monotonic()
    t = time.monotonic()
    scorer = Scorer(request["model_dir"], request.get("device"), request.get("dtype", DTYPE), request.get("batch"))
    load_seconds = time.monotonic() - t
    variants = request["variants"]
    flops = state.get("flops")
    with tempfile.TemporaryDirectory(prefix="b139-") as tmp:
        for k, item in enumerate(request["items"]):
            if item["id"] in state["scores"]:
                continue
            path = write_screenshot(item["frame"].numpy(), Path(tmp) / f"frame-{k}.png")
            if request.get("count_flops") and flops is None:
                flops = {"decision_flops": _flops(scorer, path, item["actions"], variants),
                         "candidates": len(item["actions"])}
                state["flops"] = flops
            if scorer.device.startswith("cuda"):
                torch.cuda.synchronize()
            t = time.monotonic()
            scores, info = scorer.score(path, item["actions"], variants)
            if scorer.device.startswith("cuda"):
                torch.cuda.synchronize()
            state["seconds"][item["id"]] = time.monotonic() - t
            state["scores"][item["id"]] = scores
            state["answers"][item["id"]] = info["answers"]
            state["prompt_tokens"][item["id"]] = info["prompt_tokens"]
            if _STOP["requested"] or (k + 1) % 32 == 0:
                _save(partial, state)
            if _STOP["requested"]:
                print(f"[b139-worker] paused after {len(state['scores'])} items", flush=True)
                return EXIT_PAUSED
    device_name = torch.cuda.get_device_name(0) if scorer.device.startswith("cuda") else "cpu"
    response = {"schema": RESPONSE_SCHEMA, "tag": request["tag"], "model_dir": str(request["model_dir"]),
                "variants": variants, "dtype": request.get("dtype", DTYPE), "batch": scorer.batch,
                "device": scorer.device, "device_name": device_name, "torch": torch.__version__,
                "transformers": transformers.__version__, "policy": request["policy"],
                "load_seconds": load_seconds, "seconds_total": time.monotonic() - began, **state}
    _save(response_path, response)
    if partial.is_file():
        partial.unlink()
    print(f"[b139-worker] {request['tag']}: {len(state['scores'])} items on {device_name}", flush=True)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--worker", nargs=2, metavar=("REQUEST", "RESPONSE"), required=True)
    args = parser.parse_args(argv)
    return run_worker(*args.worker)


if __name__ == "__main__":
    sys.exit(main())
