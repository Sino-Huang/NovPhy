"""Issue #139 (#127 coverage arm T7, rule P-coverage): DeepPHY-VLM, an open-weight vision-language model run locally with
DeepPHY's own Angry Birds prompt and action interface, as a zero-shot launch scorer on the sealed NovPhy launch grid.

Arm (Decisions on #139: P-license / P-route / P-adapter / P-fid, see DECISIONS):
- **T7 DeepPHY-VLM** (coverage, secondary; never in the program claim): the NovPhy decision frame (2x replicated to the
  640 x 480 screenshot) and DeepPHY's first-shot Angry Birds prompt go to Qwen2.5-VL (DeepPHY's local transformers
  pipeline stack, fp16, env b127-deepphy, through the scoring worker ``world_model/baselines/deepphy/novphy.py``); each
  candidate launch is rendered as DeepPHY's action code ``[shoot(angle=X, power=Y)]`` and scored by the
  teacher-forced log-probability of that answer (row value -score). Configuration selected on the policy split among
  q7b / q7b-t01 / q3b. No NovPhy training (zero-shot); scoring is one deterministic pass, replicated as the three seeds.

Contract adaptations (Decision P-adapter): ``--selection`` scores the policy cells under the three configurations (no
training); ``--selection-readouts`` computes their policy AUCs and selects; ``--train`` is a no-op that records
"zero-shot"; ``--readouts`` measures G-rank / nonfinite / MACs on the 480 fit anchors; G-action, G-collapse and G-readout
do not apply; ``--score`` writes the primary and wrong-anchor rows (no position error).

Gates: ``--selection`` / ``--readouts`` / ``--score`` / ``--score-novelty`` refuse without the orchestrator's
``funded.json``; ``--selection`` and ``--score`` also need G-port and G-fid (``fidelity/gfid.json``) to have passed.
``--timing`` (<= 1 GPU-h) needs no funded.json but its own posted decision. Without plan.json, ``--publish`` writes the
not-run child summary from the orchestrator's ``not-run.json`` ``{"typed_reason": "cap" | "fidelity_not_reproduced",
"decision": url}``.

Modes: --gport-inputs (CPU), --project (CPU, [INFERENCE]), --timing, --selection, --selection-readouts, --smoke,
--prepare, --train, --readouts, --freeze-readouts, --score, --score-novelty, --publish, --validate. GPU work only
through the night-window scheduler (scripts/gpu_window_issue_139*.json; prepared, not enqueued by the child); env
``novphy`` for this runner, ``b127-deepphy`` for the worker; tests use CUDA_VISIBLE_DEVICES="".
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import external_baselines as eb  # noqa: E402
from world_model.baselines.deepphy import novphy as dn  # noqa: E402
from world_model.planning import scoring_harness as harness  # noqa: E402

ISSUE = 139
IDENTITY = "issue-139-deepphy-v1"
OUTPUT = ROOT / ".local-artifacts" / IDENTITY
RUNNER = "scripts/run_issue_139_deepphy.py"
TESTS = "tests/test_issue_139_deepphy.py"
ADAPTER = "world_model/baselines/deepphy/novphy.py"
SCHEMA_PLAN = "issue_139_plan_v1"
SCHEMA_RECORD = "issue_139_record_v1"
SCHEMA_SMOKE = "issue_139_smoke_v1"
SCHEMA_SELECTION = "issue_139_selection_v1"
SCHEMA_READOUTS_MANIFEST = "issue_139_readouts_manifest_v1"
SCHEMA_PROJECTION = "issue_139_projection_v1"
SCHEMA_TIMING = "issue_139_timing_v1"
SCHEMA_TRAIN = "issue_139_zero_shot_report_v1"
SCHEMA_GPORT_INPUTS = "issue_139_gport_inputs_v1"
DECISIONS = {
    "license": "https://github.com/Sino-Huang/NovPhy/issues/139#issuecomment-6437037536",
    "route": "https://github.com/Sino-Huang/NovPhy/issues/139#issuecomment-6437058547",
    "adapter": "https://github.com/Sino-Huang/NovPhy/issues/139#issuecomment-6437241396",
    "fid": "https://github.com/Sino-Huang/NovPhy/issues/139#issuecomment-6437344622",
}

CONFIGS = dn.CONFIGS  # declaration order (Decision P-adapter)
ARMS = {"T7": {"role": "coverage (secondary, P-coverage; never in the program claim)", "program_set": False}}
ARM = "T7"
ARM_CHECKS = ["G_rank", "G_nonfinite", "G_trained"]  # G-action / G-collapse / G-readout do not apply (P-adapter)
GRID = "grid"
KEY = "neg_score"  # eb.score_rows: value = -log p(answer); lower ranks first
CHILD_CAP_GPU_HOURS = 18.0  # upper end of the #127 scout estimate for DeepPHY (4-18)
WALL_CAP_HOURS = 48.0  # on 4 GPUs
REMAINDER_GPU_HOURS = 73.0  # #127 issuecomment-6436095504 and #138's cap decision: program remainder


def log(message):
    eb.log(ISSUE, message)


def rel(path):
    path = Path(path)
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def models_of(names):
    """{model: [config names]} in declaration order (configurations of one model share a forward pass)."""
    out = {}
    for name in names:
        out.setdefault(CONFIGS[name]["model"], []).append(name)
    return out


def variants(names):
    return {n: {"temperature": CONFIGS[n]["temperature"], "repetition_penalty": CONFIGS[n]["repetition_penalty"]}
            for n in names}


def parameters(name):
    return dn.MODELS[CONFIGS[name]["model"]]["parameters"]


def selection_dir(output, name):
    return Path(output) / "selection" / name


def readouts_dir(output, arm, seed):
    return Path(output) / "readouts" / f"{arm}--seed{seed}"


def model_dir(output, arm, seed):
    return Path(output) / "models" / f"{arm}--seed{seed}"


def record_path(output, arm, seed):
    return Path(output) / "records" / f"{arm}--seed{seed}.json"


def arm_seeds():
    return [(arm, seed) for arm in ARMS for seed in eb.SEEDS]


# ---------------------------------------------------------------------------------------------------------------
# the scoring worker (env b127-deepphy)
# ---------------------------------------------------------------------------------------------------------------

def score_items(output, tag, model, names, items, device=None, count_flops=False, model_path=None, dtype=None,
                batch=None):
    """Scores of items [(id, frame uint8 [3, 240, 320], [actions])] under the configurations ``names`` of one model:
    the worker's response {"scores": {id: {name: [score per action]}}, "seconds": {id: s}, ...}. The request and the
    response live in output/work/<tag>/ (a finished response is reused; a paused worker resumes per item)."""
    work = Path(output) / "work" / tag
    response = work / "response.pt"
    if response.is_file():
        return torch.load(response, map_location="cpu", weights_only=False)
    work.mkdir(parents=True, exist_ok=True)
    request = work / "request.pt"
    if not request.is_file():
        eb.atomic_torch(request, {
            "schema": dn.REQUEST_SCHEMA, "tag": tag, "model_dir": str(model_path or dn.model_dir(model)),
            "variants": variants(names), "policy": dict(harness.SCORING_POLICY), "device": device,
            "dtype": dtype or dn.DTYPE, "batch": batch, "count_flops": count_flops,
            "items": [{"id": i, "frame": torch.as_tensor(np.asarray(f), dtype=torch.uint8), "actions": list(a)}
                      for i, f, a in items]})
    env = dict(os.environ, PYTHONNOUSERSITE="1")
    code = subprocess.run([str(dn.WORKER_PYTHON), str(ROOT / ADAPTER), "--worker", str(request), str(response)],
                          cwd=ROOT, env=env).returncode
    if code == eb.EXIT_PAUSED:
        raise eb.TrainingPaused(tag)
    if code != 0:
        raise RuntimeError(f"scoring worker {tag} exited {code}")
    return torch.load(response, map_location="cpu", weights_only=False)


def worker_meta(response):
    return {k: response[k] for k in ("model_dir", "dtype", "batch", "device", "device_name", "torch", "transformers",
                                      "policy", "load_seconds", "seconds_total")}


def rows_of(entries, response, name, prefix=""):
    """{member: rows} (eb.score_rows, key neg_score) of {member: [{ordinal, action, ...}]}."""
    return {member: eb.score_rows(items, response["scores"][f"{prefix}{member}"][name], KEY)
            for member, items in entries.items()}


def grid_actions():
    items = eb.grid_items()
    return items, [item["action"] for item in items]


def wall_of(response, ids):
    seconds = [response["seconds"][i] for i in ids]
    return float(np.median(seconds)) if seconds else None


# ---------------------------------------------------------------------------------------------------------------
# gates
# ---------------------------------------------------------------------------------------------------------------

def funded(output):
    """The orchestrator's funding marker (Decision P-coverage funding): present, a JSON object, not ``funded: false``."""
    path = Path(output) / "funded.json"
    if not path.is_file():
        return False
    try:
        marker = eb.read_json(path)
    except ValueError:
        return False
    return isinstance(marker, dict) and marker.get("funded", True) is True


def gport_pass(output):
    path = Path(output) / "gport/gport.json"
    return bool(path.is_file() and eb.read_json(path).get("pass") is True)


def gfid_pass(output):
    path = Path(output) / "fidelity/gfid.json"
    return bool(path.is_file() and eb.read_json(path).get("pass") is True)


def refuse_unfunded(output, mode):
    if funded(output):
        return False
    log(f"{mode} refused: funded.json absent (no posted funding decision); no GPU work")
    return True


def ladder(output):
    """Funding marker, G-port and G-fid (PHYRE VLA, Decision P-fid)."""
    out = {"funded.json": funded(output), "gport/gport.json": gport_pass(output), "fidelity/gfid.json": gfid_pass(output)}
    out["pass"] = all(out.values())
    return out


def refuse_ladder(output, mode):
    gate = ladder(output)
    if gate["pass"]:
        return False
    log(f"{mode} refused: {json.dumps(gate)}")
    return True


# ---------------------------------------------------------------------------------------------------------------
# --gport-inputs (CPU): the fixed inputs of the G-port reference (a fit-split decision frame + the grid)
# ---------------------------------------------------------------------------------------------------------------

def gport_inputs(output):
    output = Path(output)
    frames = eb.fit_anchor_frames()
    member = sorted(frames)[0]
    eb.atomic_torch(output / "gport/inputs.pt", {"schema": SCHEMA_GPORT_INPUTS, "member": member,
                                                  "frame": torch.as_tensor(frames[member]), "grid": eb.grid_items()})
    log(f"gport inputs: fit anchor {member}")
    return 0


# ---------------------------------------------------------------------------------------------------------------
# --selection / --selection-readouts
# ---------------------------------------------------------------------------------------------------------------

def policy_items():
    cells = eb.policy_cells(GRID)
    frames = eb.policy_anchor_frames()
    return cells, [(m, frames[m], [e["action"] for e in cells[m]["inventory"]]) for m in sorted(cells)]


def selection(output, device=None):
    """The 94 policy grid cells under every configuration (one worker pass per model): selection/<config>/."""
    output = Path(output)
    if refuse_ladder(output, "selection"):
        return 1
    cells, items = policy_items()
    for model, names in models_of(CONFIGS).items():
        if all((selection_dir(output, n) / "scores.json").is_file() for n in names):
            continue
        began = time.monotonic()
        response = score_items(output, f"selection-{model}", model, names, items, device)
        seconds = time.monotonic() - began
        on_gpu = str(response["device"]).startswith("cuda")
        for name in names:
            sdir = selection_dir(output, name)
            sdir.mkdir(parents=True, exist_ok=True)
            eb.write_json(sdir / "report.json", {
                "config": name, "spec": CONFIGS[name], "parameters": parameters(name), "decisions": len(items),
                "gpu_seconds": response["seconds_total"] if on_gpu else 0.0,
                "seconds_per_decision": wall_of(response, [i for i, _, _ in items]), "worker": worker_meta(response),
                "shared_pass": names, "written_at": eb.utc_now()})
            eb.write_json(sdir / "scores.json", {m: response["scores"][m][name] for m, _, _ in items})
        eb.mark(output, f"selection-{model}", seconds, response["seconds_total"] if on_gpu else 0.0)
        log(f"selection scored {model} ({names}) in {seconds:.0f}s")
    return 0


def selection_readouts(output):
    """Per configuration: the policy-split pooled grid AUC of the primary score; selection.json (CPU)."""
    output = Path(output)
    cells = eb.policy_cells(GRID)
    results, details = {}, {}
    for name in CONFIGS:
        path = selection_dir(output, name) / "scores.json"
        if not path.is_file():
            raise ValueError(f"selection scores of {name} missing")
        scores = eb.read_json(path)
        rows = {m: eb.score_rows(cells[m]["inventory"], scores[m], KEY) for m in sorted(cells)}
        results[name] = {"auc": eb.policy_grid_auc(rows, KEY), "parameters": parameters(name)}
        details[name] = {"scores_sha256": eb.sha256_of(path), "cells": len(rows),
                         "nonfinite": sum(r[KEY] is None for v in rows.values() for r in v)}
    choice = eb.select_configuration(results)
    eb.write_json(output / "selection.json", {
        "schema": SCHEMA_SELECTION, "plan_identity": IDENTITY, "configs": CONFIGS, **choice, "details": details,
        "policy_cells": len(cells), "written_at": eb.utc_now()})
    log(f"selection: {choice['selected']} ({json.dumps(results)})")
    return 0 if choice["selected"] is not None else 1


def selected_config(output):
    path = Path(output) / "selection.json"
    if not path.is_file():
        raise ValueError("selection.json missing; run --selection-readouts")
    selected = eb.read_json(path)["selected"]
    if selected is None:
        raise ValueError("no configuration was selected")
    return selected


# ---------------------------------------------------------------------------------------------------------------
# --project (CPU, [INFERENCE]) / --timing (GPU, measured)
# ---------------------------------------------------------------------------------------------------------------

PROMPT_TOKENS = 1202  # DeepPHY's first-shot Angry Birds prompt with one 640 x 480 screenshot (391 image tokens)
ANSWER_TOKENS = 14  # [shoot(angle=37, power=1.0)]<|im_end|>
IMAGE_PATCHES = 1564  # 644 x 476 / 14^2 (before the 2 x 2 merge)
SEQUENCES_PER_DECISION = 16  # an upper bound on distinct answers per decision (grid: 15)
LOAD_SECONDS = {"Qwen2.5-VL-7B-Instruct": 120.0, "Qwen2.5-VL-3B-Instruct": 60.0}  # [INFERENCE] per worker pass
# decisions (one anchor x its candidates): selection = 94 policy cells per model; readouts = 480 fit anchors;
# scoring = the mixed cells of the four inventories (311 + 76 + 78 + 47) + 480 wrong anchors; novelty = 29 x 8 x 2
# levels; validate = one more scoring pass; the three seeds are one deterministic pass (P-adapter item 6)
POLICY_DECISIONS = 94
READOUT_DECISIONS = 480
SCORING_DECISIONS = (311 + 76 + 78 + 47) + 480
NOVELTY_DECISIONS = 29 * 8 * 2
VALIDATE_DECISIONS = SCORING_DECISIONS
WORKER_PASSES = {"selection": 2, "readouts": 1, "score": 1, "novelty": 1, "validate": 1}
# [INFERENCE] RTX 5090 mapping of the forward FLOPs (sustained fp16 tensor throughput of the HF forward: lower =
# optimistic 100 TFLOP/s, upper = pessimistic 40 TFLOP/s)
GPU_MODEL = {"lower": {"tflops": 100.0}, "upper": {"tflops": 40.0}}
GFID_GPU_H = {"lower": 2.1, "upper": 4.5}  # Decision P-fid (PHYRE, ~9 600 official calls)
TIMING_PROBE_GPU_H = 1.0


def forward_flops(config):
    """FLOPs of one forward over a prompt + answer sequence of a Qwen2.5-VL config.json (dict): language-model linear
    layers and lm_head at every position, causal attention, and the vision tower on the screenshot's patches."""
    t = config.get("text_config", config)
    d, inter, layers = t["hidden_size"], t["intermediate_size"], t["num_hidden_layers"]
    heads, kv = t["num_attention_heads"], t["num_key_value_heads"]
    head = d // heads
    per_layer = 2 * d * d + 2 * d * kv * head + 3 * d * inter
    seq = PROMPT_TOKENS + ANSWER_TOKENS
    lm = 2 * seq * (layers * per_layer + d * t.get("vocab_size", config.get("vocab_size")))
    attention = 2 * layers * d * seq * seq  # QK^T + AV, causal (half of 4 L^2 d)
    v = config["vision_config"]
    vh, vi, vd = v["hidden_size"], v["intermediate_size"], v["depth"]
    merge = (vh * 4) * (vh * 4) + (vh * 4) * v["out_hidden_size"]
    vision = 2 * IMAGE_PATCHES * (vd * (4 * vh * vh + 3 * vh * vi)) + 2 * (IMAGE_PATCHES // 4) * merge
    return {"language": lm, "attention": attention, "vision": vision, "total": lm + attention + vision}


def decision_flops(model):
    config = json.loads((dn.model_dir(model) / "config.json").read_text())
    per = forward_flops(config)
    return {"per_sequence": per, "sequences": SEQUENCES_PER_DECISION,
            "decision_flops": per["total"] * SEQUENCES_PER_DECISION}


def child_projection(seconds_per_decision, load_seconds, selected_model, gfid_gpu_h):
    """GPU-h of the child from s/decision and s/load per model: selection (every model on the policy cells), then the
    selected (worst-case: the most expensive) model on readouts, scoring, novelty and validation; G-fid and the
    timing probe."""
    sel = sum(POLICY_DECISIONS * s + load_seconds[m] for m, s in seconds_per_decision.items())
    s, load = seconds_per_decision[selected_model], load_seconds[selected_model]
    parts = {"selection": sel, "readouts": READOUT_DECISIONS * s + load, "score": SCORING_DECISIONS * s + load,
             "score_novelty": NOVELTY_DECISIONS * s + load, "validate": VALIDATE_DECISIONS * s + load}
    total = sum(parts.values()) + (gfid_gpu_h + TIMING_PROBE_GPU_H) * 3600
    wall = total  # the passes are sequential (each needs the previous one's freeze); one GPU at a time
    return {"seconds_per_decision": seconds_per_decision, "load_seconds": load_seconds,
            "projected_model": selected_model, **{f"{k}_gpu_h": v / 3600 for k, v in parts.items()},
            "gfid_gpu_h": gfid_gpu_h, "timing_probe_gpu_h": TIMING_PROBE_GPU_H, "total_gpu_h": total / 3600,
            "wall_h_sequential_one_gpu": wall / 3600, "child_cap_gpu_h": CHILD_CAP_GPU_HOURS,
            "wall_cap_h": WALL_CAP_HOURS, "remainder_gpu_h": REMAINDER_GPU_HOURS,
            "fits_remainder": bool(total / 3600 <= REMAINDER_GPU_HOURS),
            "within_caps": bool(total / 3600 <= CHILD_CAP_GPU_HOURS and wall / 3600 <= WALL_CAP_HOURS)}


def worst_model(rates):
    return max(rates, key=lambda m: rates[m])


def project(output):
    """projection.json: [INFERENCE] forward FLOPs per decision of each model (config.json; the measured CPU decision of
    the released 3B in gport.json beside it) mapped to an RTX 5090 by GPU_MODEL; timing.json (GPU) is primary when
    present."""
    output = Path(output)
    models = list(models_of(CONFIGS))
    flops = {m: decision_flops(m) for m in models}
    result = {"schema": SCHEMA_PROJECTION, "plan_identity": IDENTITY, "measured_at": eb.utc_now(),
              "label": "[INFERENCE] forward FLOPs of one decision (<= 16 distinct answers x one prompt + answer "
                       "sequence of 1 216 tokens with a 1 564-patch screenshot) from each model's config.json, mapped "
                       "to an RTX 5090 at 100 (lower) / 40 (upper) sustained fp16 TFLOP/s, plus a model load per "
                       "worker pass; G-fid and the timing probe added",
              "flops": flops, "gpu_model": GPU_MODEL, "load_seconds": LOAD_SECONDS,
              "decisions": {"selection_per_model": POLICY_DECISIONS, "readouts": READOUT_DECISIONS,
                            "score": SCORING_DECISIONS, "score_novelty": NOVELTY_DECISIONS,
                            "validate": VALIDATE_DECISIONS}}
    gport_path = output / "gport/gport.json"
    if gport_path.is_file():
        case = eb.read_json(gport_path)["cases"].get("qwen3b_fp32", {})
        result["cpu_reference"] = {"model": "Qwen2.5-VL-3B-Instruct", "dtype": "float32", "threads": 4,
                                   "seconds_per_decision_batch1": case.get("cpu_seconds_per_decision_batch1"),
                                   "source": rel(gport_path)}
    for tag, gpu in GPU_MODEL.items():
        rates = {m: flops[m]["decision_flops"] / (gpu["tflops"] * 1e12) for m in models}
        result[f"projection_{tag}"] = child_projection(rates, LOAD_SECONDS, worst_model(rates), GFID_GPU_H[tag])
    result["primary"] = "projection_upper"
    timing_path = output / "timing.json"
    if timing_path.is_file() and str(eb.read_json(timing_path).get("device", "")).startswith("cuda"):
        result["projection_measured"] = measured_projection(eb.read_json(timing_path))
        result["timing_sha256"] = eb.sha256_of(timing_path)
        result["primary"] = "projection_measured"
    eb.write_json(output / "projection.json", result)
    lo, up = (result[f"projection_{t}"]["total_gpu_h"] for t in ("lower", "upper"))
    log(f"projection [INFERENCE]: {lo:.1f}-{up:.1f} GPU-h (remainder {REMAINDER_GPU_HOURS}, cap "
        f"{CHILD_CAP_GPU_HOURS}); primary {result['primary']}")
    return 0


def measured_projection(record):
    rates = {m: v["seconds_per_decision_median"] for m, v in record["models"].items()}
    loads = {m: v["load_seconds"] for m, v in record["models"].items()}
    return {"source": "timing.json (measured)", **child_projection(rates, loads, worst_model(rates), GFID_GPU_H["upper"])}


def timing(output, decisions=8, device=None, model_paths=None, dtype=None):
    """--timing probe (<= 1 GPU-h; enqueued only after its own posted decision; no funded.json needed), exclusive GPU:
    per model, one worker pass over ``decisions`` fit-anchor frames x the 16 grid launches (the first decision is a
    warm-up; FLOPs counted on it), its load time and the median s/decision of the rest. Fit split only; keeps nothing
    but timing.json."""
    output = Path(output)
    frames = eb.fit_anchor_frames()
    members = sorted(frames)[:decisions]
    _, actions = grid_actions()
    items = [(m, frames[m], actions) for m in members]
    began = time.monotonic()
    result = {"schema": SCHEMA_TIMING, "plan_identity": IDENTITY, "decisions": decisions, "candidates": len(actions),
              "fit_anchors": members, "models": {}}
    for model, names in models_of(CONFIGS).items():
        path = (model_paths or {}).get(model)
        response = score_items(output, f"timing-{model}", model, names, items, device, count_flops=True,
                               model_path=path, dtype=dtype)
        seconds = [response["seconds"][m] for m in members[1:]] or [response["seconds"][members[0]]]
        result["models"][model] = {
            "configs": names, "load_seconds": response["load_seconds"], "worker_seconds": response["seconds_total"],
            "seconds_per_decision_median": float(np.median(seconds)), "seconds_per_decision": seconds,
            "warmup_seconds": response["seconds"][members[0]], "flops": response.get("flops"),
            "prompt_tokens": sorted(set(response["prompt_tokens"].values())), "worker": worker_meta(response)}
        result["device"] = response["device"]
        result["device_name"] = response["device_name"]
    result["seconds"] = time.monotonic() - began
    result["written_at"] = eb.utc_now()
    eb.write_json(output / "timing.json", result)
    on_gpu = str(result["device"]).startswith("cuda")
    eb.mark(output, "timing", result["seconds"], result["seconds"] if on_gpu else 0.0)
    log("timing: " + ", ".join(f"{m} {v['seconds_per_decision_median']:.2f} s/decision (load {v['load_seconds']:.0f}s)"
                               for m, v in result["models"].items()))
    if on_gpu and (output / "projection.json").is_file():
        projection = eb.read_json(output / "projection.json")
        projection.update(projection_measured=measured_projection(result), primary="projection_measured",
                          timing_sha256=eb.sha256_of(output / "timing.json"))
        eb.write_json(output / "projection.json", projection)
    return 0


# ---------------------------------------------------------------------------------------------------------------
# --smoke / --prepare
# ---------------------------------------------------------------------------------------------------------------

def smoke(output):
    output = Path(output)
    cells = eb.policy_cells(GRID)
    accounting = {"fit_anchors": len(eb.fit_anchor_frames()), "policy_grid_mixed_cells": len(cells),
                  "policy_anchor_frames": len(eb.policy_anchor_frames()), "grid_candidates": len(eb.grid_items()),
                  "grid_answers": [dn.answer_text(it["action"]) for it in eb.grid_items()]}
    gport = eb.read_json(output / "gport/gport.json")
    gport_check = {"pass": gport.get("pass") is True,
                   "adapter_current": gport.get("adapter_sha256") == eb.sha256_of(ROOT / ADAPTER)}
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    tests = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", TESTS], cwd=ROOT, env=env,
                           capture_output=True, text=True)
    tests_record = {"command": f"CUDA_VISIBLE_DEVICES= python -m pytest -q {TESTS}", "exit_code": tests.returncode,
                    "tail": tests.stdout.strip().splitlines()[-1:] if tests.stdout else []}
    selection_record = eb.read_json(output / "selection.json")
    selected = selected_config(output)
    timing_record = eb.read_json(output / "timing.json")
    model = CONFIGS[selected]["model"]
    rate = eb.read_json(selection_dir(output, selected) / "report.json")["seconds_per_decision"]
    load = timing_record["models"][model]["load_seconds"]
    remaining = ((READOUT_DECISIONS + SCORING_DECISIONS + NOVELTY_DECISIONS + VALIDATE_DECISIONS) * rate
                 + 4 * load) / 3600
    _, spent_gpu_s, _ = eb.ledger_totals(output, training_report_of(output))
    projection = {"selected_model": model, "seconds_per_decision": rate, "load_seconds": load,
                  "spent_gpu_h": spent_gpu_s / 3600, "remaining_gpu_h": remaining,
                  "total_gpu_h": spent_gpu_s / 3600 + remaining, "child_cap_gpu_h": CHILD_CAP_GPU_HOURS,
                  "remainder_gpu_h": REMAINDER_GPU_HOURS}
    projection["within_caps"] = bool(projection["total_gpu_h"] <= CHILD_CAP_GPU_HOURS)
    record = {"schema": SCHEMA_SMOKE, "plan_identity": IDENTITY, "accounting": accounting, "gport": gport_check,
              "gate": ladder(output), "tests": tests_record, "selection": selection_record, "selected": selected,
              "timing": {"sha256": eb.sha256_of(output / "timing.json"), "device_name": timing_record["device_name"]},
              "projection": projection,
              "pass": bool(gport_check["pass"] and gport_check["adapter_current"] and tests.returncode == 0
                           and projection["within_caps"] and accounting["fit_anchors"] == 480
                           and accounting["policy_grid_mixed_cells"] == 94 and gfid_pass(output)),
              "written_at": eb.utc_now()}
    eb.write_json(output / "smoke.json", record)
    log(f"smoke: pass {record['pass']}; projection {projection['total_gpu_h']:.1f} GPU-h; selected {selected}")
    return 0 if record["pass"] else 1


OWN_FILES = [RUNNER, TESTS, "world_model/baselines/deepphy/__init__.py", ADAPTER,
             "scripts/b127_deepphy_port_reference.py", "scripts/b127_deepphy_fidelity.py"]


def own_pins(output):
    output = Path(output)
    paths = [ROOT / p for p in OWN_FILES]
    paths += sorted(p for p in (ROOT / "world_model/baselines/deepphy/vendor").rglob("*") if p.is_file())
    paths += [output / "gport/gport.json", output / "gport/inputs.pt", output / "fidelity/band.json",
              output / "fidelity/gfid.json", output / "smoke.json", output / "selection.json", output / "funded.json",
              output / "timing.json", dn.WEIGHTS / "MANIFEST.json"]
    paths += [selection_dir(output, n) / f for n in CONFIGS for f in ("scores.json", "report.json")]
    return [{"name": f"t7_{rel(p)}", "artifact": rel(p), "sha256": eb.sha256_of(p)}
            for p in paths if p.is_file() and "__pycache__" not in str(p)]


def prepare(output):
    output = Path(output)
    path = output / "plan.json"
    if path.is_file():
        load_plan(output)
        log("plan.json is frozen; pins re-verified")
        return 0
    record = eb.read_json(output / "smoke.json")
    if not record["pass"]:
        raise SystemExit("smoke did not pass; no freeze (P-cap / fix first)")
    selected = record["selected"]
    plan = {
        "schema": SCHEMA_PLAN, "plan_identity": IDENTITY, "issue": ISSUE, "program_issue": eb.PROGRAM_ISSUE,
        "frozen_at": eb.utc_now(), "pins": eb.shared_inputs() + own_pins(output),
        "selected": selected, "selection_rule": eb.SELECTION_RULE, "configs": CONFIGS, "models": dn.MODELS,
        "arms": {arm: {**ARMS[arm], "config": selected, "spec": CONFIGS[selected], "seeds": list(eb.SEEDS)}
                 for arm in ARMS},
        "scoring_policy": harness.SCORING_POLICY, "precision": dn.DTYPE, "gpu_batch": dn.GPU_BATCH,
        "score": "teacher-forced log p of DeepPHY's answer [shoot(angle=X, power=Y)]<|im_end|> after its first-shot "
                 "Angry Birds prompt and the decision screenshot; row value neg_score = -log p; equal renderings tie",
        "seeds_rule": "one deterministic pass (no sampling) written as the three seeds (replicated_from)",
        "queues": {"train": "no-op (zero-shot)", "readouts": "GPU 0, one pass", "score": "GPU 0, one process"},
        "seeds": list(eb.SEEDS), "guards": eb.GUARD, "arm_guards": ARM_CHECKS, "extra_guards": ["G_port", "G_fid"],
        "global_guards": ["G_isolation", "G_chronology", "G_scoring_policy", "G_records"],
        "contrasts": {"per_arm": list(eb.CONTRAST_REFERENCES), "within_family": [], "wrong_anchor": list(ARMS)},
        "dispositions": eb.DISPOSITION_LABELS, "delta": eb.DELTA, "decisions": DECISIONS,
        "disclosures": [
            "T7 is a coverage arm (#127 rule P-coverage; funded by the measured projection): secondary, reported, never "
            "in the program claim.",
            "Zero-shot: no NovPhy training; the configuration was selected on the policy split among three pre-declared "
            "configurations (q7b, q7b-t01, q3b); U had no such tuning (#127).",
            "DeepPHY's agent emits one action; T7 scores each candidate by the likelihood DeepPHY's agent assigns to "
            "emitting it in DeepPHY's action language (integer angle, power = pull / drag radius).",
            "Ordinals 8 and 12 of the grid render to the same action code (angle 2, power 1.0) and tie exactly.",
            "G-fid reproduces DeepPHY's PHYRE VLA Qwen-7B number (the Angry Birds number needs the iPhone app); the "
            "band contains the paper's random-agent value.",
            "G-action, G-collapse and G-readout do not apply (no dynamics model, latent or readout); no position error; "
            "no native success head and no U + head control.",
            "P-novelty: after the primary scoring pass, T7 (if its guards pass) is scored on the novelty levels from "
            "each level's decision frame (--score-novelty); descriptive only, no tokens.",
        ],
    }
    eb.write_json(path, plan)
    log(f"plan.json frozen ({len(plan['pins'])} pins); selected {selected}")
    return 0


def load_plan(output):
    path = Path(output) / "plan.json"
    if not path.is_file():
        raise ValueError("plan.json missing; run --smoke and --prepare first")
    plan = eb.read_json(path)
    if plan.get("schema") != SCHEMA_PLAN or plan.get("plan_identity") != IDENTITY:
        raise ValueError("plan.json binding differs")
    eb.verify_pins(plan["pins"])
    return plan


# ---------------------------------------------------------------------------------------------------------------
# --train (zero-shot no-op) / --readouts / --freeze-readouts
# ---------------------------------------------------------------------------------------------------------------

def train(output):
    output = Path(output)
    if refuse_unfunded(output, "train"):
        return 1
    plan = load_plan(output)
    for arm, seed in arm_seeds():
        path = model_dir(output, arm, seed) / "report.json"
        if path.is_file():
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        eb.write_json(path, {"schema": SCHEMA_TRAIN, "arm": arm, "seed": seed, "zero_shot": True, "steps": 0,
                             "gpu_seconds": 0.0, "config": plan["selected"], "model": CONFIGS[plan["selected"]]["model"],
                             "note": "DeepPHY evaluates VLMs zero-shot; T7 has no NovPhy training (Decision P-adapter)",
                             "written_at": eb.utc_now()})
    log("train: zero-shot (no training); reports written")
    return 0


def training_report_of(output):
    def report_of(phase):
        return None
    return report_of


def readouts(output, device=None):
    """G-rank and the fit-anchor nonfinite share on the 480 fit anchors x 16 grid launches (selected configuration),
    MACs per candidate and wall per decision; the policy AUC from selection.json. One pass, written per seed."""
    output = Path(output)
    if refuse_unfunded(output, "readouts"):
        return 1
    plan = load_plan(output)
    if all((readouts_dir(output, a, s) / "readouts.json").is_file() for a, s in arm_seeds()):
        return 0
    name = plan["selected"]
    frames = eb.fit_anchor_frames()
    members = sorted(frames)
    items, actions = grid_actions()
    began = time.monotonic()
    with harness.deterministic_scoring() as policy:
        response = score_items(output, "readouts-fit", CONFIGS[name]["model"], [name],
                               [(m, frames[m], actions) for m in members], device, count_flops=True)
    seconds = time.monotonic() - began
    scores = np.asarray([response["scores"][m][name] for m in members], dtype=np.float64)
    costs = -scores
    selection_record = eb.read_json(output / "selection.json")
    flops = response.get("flops") or {}
    base = {"plan_identity": IDENTITY, "config": name, "scoring_policy": policy, "typed_failure": None,
            "G_rank": eb.g_rank(costs), "G_action": None, "G_collapse": None, "policy_ceiling_auc": None,
            "not_applied": {"G_action": "no dynamics model", "G_collapse": "no latent", "G_readout": "no readout"},
            "fit_anchor_nonfinite_share": float((~np.isfinite(scores)).mean()),
            "policy_auc": selection_record["results"][name]["auc"], "trainable_parameters": 0,
            "model_parameters": parameters(name),
            "macs_per_candidate": (flops["decision_flops"] / 2 / flops["candidates"]) if flops else None,
            "wall_per_decision": wall_of(response, members), "worker": worker_meta(response),
            "seconds": seconds, "written_at": eb.utc_now()}
    for arm, seed in arm_seeds():
        out = readouts_dir(output, arm, seed)
        out.mkdir(parents=True, exist_ok=True)
        eb.write_json(out / "readouts.json", {**base, "arm": arm, "seed": seed,
                                              "replicated_from": None if seed == eb.SELECTION_SEED else eb.SELECTION_SEED})
    on_gpu = str(response["device"]).startswith("cuda")
    eb.mark(output, "readouts", seconds, response["seconds_total"] if on_gpu else 0.0)
    log(f"readouts: G-rank {base['G_rank']}; policy AUC {base['policy_auc']}")
    return 0


def weight_pins(plan):
    model = dn.model_dir(CONFIGS[plan["selected"]]["model"])
    return [dn.WEIGHTS / "MANIFEST.json", model / "model.safetensors.index.json", model / "config.json"]


def freeze_readouts(output):
    output = Path(output)
    plan = load_plan(output)
    path = output / "readouts-manifest.json"
    if path.is_file():
        load_manifest(output)
        log("readouts are frozen; manifest re-verified")
        return 0
    if eb.evaluation_anchors_path(output).is_file():
        raise SystemExit("evaluation anchors already read: G-chronology would fail")
    entries = {}
    for arm, seed in arm_seeds():
        rpath = readouts_dir(output, arm, seed) / "readouts.json"
        if not rpath.is_file():
            raise SystemExit(f"readouts of {arm} seed {seed} missing")
        entries[f"{arm}--seed{seed}"] = {"readouts": rel(rpath), "readouts_sha256": eb.sha256_of(rpath)}
    weights = {rel(p): eb.sha256_of(p) for p in weight_pins(plan)}
    eb.write_json(path, {"schema": SCHEMA_READOUTS_MANIFEST, "plan_identity": IDENTITY, "frozen_at": eb.utc_now(),
                         "entries": entries, "weights": weights})
    log(f"readouts frozen ({len(entries)} arm-seeds)")
    return 0


def load_manifest(output):
    path = Path(output) / "readouts-manifest.json"
    if not path.is_file():
        raise ValueError("readouts are not frozen; run --freeze-readouts first")
    manifest = eb.read_json(path)
    for name, entry in manifest["entries"].items():
        if eb.sha256_of(ROOT / entry["readouts"]) != entry["readouts_sha256"]:
            raise ValueError(f"{name}: frozen readouts changed")
    for artifact, digest in manifest["weights"].items():
        if eb.sha256_of(ROOT / artifact) != digest:
            raise ValueError(f"{artifact}: frozen weights pin changed")
    return manifest


# ---------------------------------------------------------------------------------------------------------------
# --score (once; GPU 0; deterministic; only after G-fid passed) / --score-novelty
# ---------------------------------------------------------------------------------------------------------------

def score_pass(output, plan, tag, anchors, cells, device=None):
    """One deterministic pass of the selected configuration: every mixed cell of every inventory from its Gate-B
    anchor frame, and the wrong-anchor rows (the 480 fit anchor frames x the 16 grid launches)."""
    name = plan["selected"]
    items = []
    for inventory in eb.INVENTORIES:
        for member, entry in sorted(cells[inventory]["mixed"].items()):
            items.append((f"{inventory}/{member}", anchors["frames"][member], [e["action"] for e in entry["inventory"]]))
    fit = eb.fit_anchor_frames()
    grid, actions = grid_actions()
    items += [(f"wrong/{m}", fit[m], actions) for m in sorted(fit)]
    response = score_items(output, tag, CONFIGS[name]["model"], [name], items, device)
    inventories = {inventory: {member: {"rows": eb.score_rows(entry["inventory"],
                                                              response["scores"][f"{inventory}/{member}"][name], KEY)}
                               for member, entry in sorted(cells[inventory]["mixed"].items())}
                   for inventory in eb.INVENTORIES}
    wrong = {m: {"rows": eb.score_rows(grid, response["scores"][f"wrong/{m}"][name], KEY)} for m in sorted(fit)}
    grid_ids = [f"{GRID}/{m}" for m in sorted(cells[GRID]["mixed"])]
    return {"inventories": inventories, "wrong_anchor": wrong, "wall_per_decision": wall_of(response, grid_ids),
            "worker": worker_meta(response)}, response


def score(output, device=None):
    output = Path(output)
    if refuse_unfunded(output, "score"):
        return 1
    if not gfid_pass(output):
        log("score refused: fidelity/gfid.json has not passed (#139 gate); evaluation not read")
        return 1
    plan = load_plan(output)
    manifest = load_manifest(output)
    if all(record_path(output, a, s).is_file() for a, s in arm_seeds()):
        return 0
    with harness.deterministic_scoring() as policy:
        path = eb.evaluation_anchors_path(output)
        anchors = (torch.load(path, map_location="cpu", weights_only=True) if path.is_file()
                   else eb.read_evaluation_anchors(output, policy))
        cells = eb.evaluation_cells()
        began = time.monotonic()
        body, response = score_pass(output, plan, "score", anchors, cells, device)
        seconds = time.monotonic() - began
        for arm, seed in arm_seeds():
            record = {"schema": SCHEMA_RECORD, "plan_identity": IDENTITY, "arm": arm, "seed": seed,
                      "typed_failure": None, "config": plan["selected"], **body, "scoring_policy": policy,
                      "replicated_from": None if seed == eb.SELECTION_SEED else eb.SELECTION_SEED,
                      "scored_at": eb.utc_now(), "readouts_frozen_at": manifest["frozen_at"],
                      "anchors_read_at": anchors["read_at"], "score_seconds": seconds}
            eb.write_json(record_path(output, arm, seed), record)
    on_gpu = str(response["device"]).startswith("cuda")
    eb.mark(output, "score", seconds, response["seconds_total"] if on_gpu else 0.0)
    log(f"scored {ARM} ({seconds:.0f}s; one deterministic pass written as {len(eb.SEEDS)} seeds)")
    return 0


SCHEMA_NOVELTY_RECORD = "issue_139_novelty_record_v1"
NOVELTY_ARMS = tuple(ARMS)


def novelty_dir(output):
    return Path(output) / "novelty"


def novelty_record_path(output, arm, seed):
    return novelty_dir(output) / "records" / f"{arm}--seed{seed}.json"


def score_novelty(output, device=None, cells=None):
    """P-novelty: every guard-passing arm on the mixed novelty levels (grid) from each level's decision frame, as the
    primary scoring path scores a cell; refuses before the primary records and the readouts freeze exist."""
    output = Path(output)
    if refuse_unfunded(output, "score-novelty"):
        return 1
    plan = load_plan(output)
    load_manifest(output)
    missing = [f"{a}--seed{s}" for a, s in arm_seeds() if not record_path(output, a, s).is_file()]
    if missing:
        log(f"score-novelty refused: primary records missing ({missing})")
        return 1
    records = load_records(output)
    guards_of, extra = arm_guards(output, records), guard_extras(output)
    status = {arm: {"guards_pass": arm_guard_pass(guards_of[arm], extra),
                    "failed": sorted(k for k, v in {**guards_of[arm], **extra}.items()
                                     if k.startswith("G_") and v is not True)} for arm in NOVELTY_ARMS}
    name = plan["selected"]
    with harness.deterministic_scoring() as policy:
        cells = eb.novelty_cells() if cells is None else cells
        path = eb.novelty_anchors_path(output)
        anchors = (torch.load(path, map_location="cpu", weights_only=True) if path.is_file()
                   else eb.read_novelty_anchors(output, policy, cells))
        eb.novelty_reference_records(output, policy, cells, anchors["carriers"])
        eb.write_json(novelty_dir(output) / "arms.json", {"schema": "issue_139_novelty_arms_v1", "arms": status,
                                                          "anchors_read_at": anchors["read_at"]})
        for arm in NOVELTY_ARMS:
            if not status[arm]["guards_pass"]:
                log(f"score-novelty: {arm} not scored (failed guards {status[arm]['failed']})")
                continue
            if all(novelty_record_path(output, arm, s).is_file() for s in eb.SEEDS):
                continue
            levels = sorted(cells["mixed"])
            items = [(level, anchors["frames"][level], [e["action"] for e in cells["levels"][level]["inventory"]])
                     for level in levels]
            began = time.monotonic()
            response = score_items(output, f"novelty-{arm}", CONFIGS[name]["model"], [name], items, device)
            seconds = time.monotonic() - began
            grid = {level: {"rows": eb.score_rows(cells["levels"][level]["inventory"], response["scores"][level][name],
                                                  KEY)} for level in levels}
            for seed in eb.SEEDS:
                eb.write_json(novelty_record_path(output, arm, seed), {
                    "schema": SCHEMA_NOVELTY_RECORD, "arm": arm, "seed": seed, "config": name,
                    "model_sha256": eb.sha256_of(dn.WEIGHTS / "MANIFEST.json"), "inventories": {GRID: grid},
                    "replicated_from": None if seed == eb.SELECTION_SEED else eb.SELECTION_SEED,
                    "scoring_policy": policy, "scored_at": eb.utc_now(), "anchors_read_at": anchors["read_at"]})
            on_gpu = str(response["device"]).startswith("cuda")
            eb.mark(output, f"score-novelty-{arm}", seconds, response["seconds_total"] if on_gpu else 0.0)
            log(f"novelty scored {arm} ({seconds:.0f}s)")
    return 0


def novelty_section(output, records, cells=None):
    """DESCRIPTIVE novelty table (no tokens, never in a disposition) when the novelty pass has run, else None."""
    output = Path(output)
    arms_path = novelty_dir(output) / "arms.json"
    if not arms_path.is_file():
        return None
    status = eb.read_json(arms_path)["arms"]
    scored = [arm for arm in NOVELTY_ARMS if status[arm]["guards_pass"]]
    cells = eb.novelty_cells() if cells is None else cells
    anchors_read = torch.load(eb.novelty_anchors_path(output), map_location="cpu", weights_only=True)["read_at"]
    views = {arm: ({seed: eb.read_json(novelty_record_path(output, arm, seed)) for seed in eb.SEEDS}, KEY)
             for arm in scored}
    refs = eb.novelty_reference_records(output, None, cells)
    last_primary = max(r["scored_at"] for recs in records.values() for r in recs.values())
    return {"label": "DESCRIPTIVE (P-novelty)", "scored_arms": scored,
            "not_scored": {arm: status[arm]["failed"] for arm in NOVELTY_ARMS if arm not in scored},
            "chronology": {"pass": bool(anchors_read >= last_primary), "novelty_anchors_read_at": anchors_read,
                           "last_primary_scored_at": last_primary},
            "table": eb.novelty_table({**views, **refs}, cells)}


# ---------------------------------------------------------------------------------------------------------------
# --publish / --validate
# ---------------------------------------------------------------------------------------------------------------

RECORD_PATTERNS = ["records/*.json", "novelty/arms.json", "novelty/records/*.json", "novelty/references/*.json"]


def load_records(output):
    return {arm: {seed: eb.read_json(record_path(output, arm, seed)) for seed in eb.SEEDS} for arm in ARMS}


def global_guards(output, plan, records):
    output = Path(output)
    manifest = eb.read_json(output / "readouts-manifest.json")
    anchors_read = min(r["anchors_read_at"] for recs in records.values() for r in recs.values())
    first_scored = min(r["scored_at"] for recs in records.values() for r in recs.values())
    chronology = bool(plan["frozen_at"] <= manifest["frozen_at"] <= anchors_read <= first_scored)
    policies = [r["scoring_policy"] for recs in records.values() for r in recs.values()]
    return {
        "G_isolation": {"pass": True, "rule": "no NovPhy training; the policy split selects; evaluation anchors are "
                        "read only by --score after --freeze-readouts (G_chronology) and after G-fid passed"},
        "G_chronology": {"pass": chronology, "plan_frozen_at": plan["frozen_at"],
                         "readouts_frozen_at": manifest["frozen_at"], "anchors_read_at": anchors_read,
                         "first_scored_at": first_scored},
        "G_scoring_policy": {"pass": all(p == harness.SCORING_POLICY for p in policies)},
        "G_records": eb.records_guard(output, IDENTITY),
    }


def arm_compute(output, arm):
    rd = eb.read_json(readouts_dir(output, arm, eb.SELECTION_SEED) / "readouts.json")
    walls = [eb.read_json(record_path(output, arm, s)).get("wall_per_decision") for s in eb.SEEDS]
    walls = [w for w in walls if w is not None]
    return {"trainable_parameters": 0, "training_gpu_seconds": {str(s): 0.0 for s in eb.SEEDS},
            "macs_per_candidate": rd.get("macs_per_candidate"),
            "wall_per_decision": float(np.median(walls)) if walls else None}


def guard_extras(output):
    return {"G_port": gport_pass(output), "G_fid": gfid_pass(output)}


def arm_guards(output, records):
    """{arm: eb.arm_guard_block} of the arm checks from the readouts and the primary records."""
    guards_of = {}
    for arm in ARMS:
        per_seed = {}
        for seed in eb.SEEDS:
            rd = eb.read_json(readouts_dir(output, arm, seed) / "readouts.json")
            per_seed[seed] = {"G_rank": rd.get("G_rank"), "G_action": None, "G_collapse": None,
                              "policy_ceiling_auc": None,
                              "typed_failure": rd.get("typed_failure") or records[arm][seed].get("typed_failure")}
        guards_of[arm] = eb.arm_guard_block(per_seed, ARM_CHECKS, nonfinite=eb.nonfinite_share(records[arm].values(), KEY))
    return guards_of


def arm_guard_pass(guards, extra_guards):
    merged = {**guards, **extra_guards}
    names = [k for k in merged if k.startswith("G_")]
    return bool(names) and all(merged[k] is True for k in names)


def publication(output):
    """Every published table, derived from the frozen plan, readouts and records (deterministic)."""
    output = Path(output)
    plan = load_plan(output)
    records = load_records(output)
    cells = eb.evaluation_cells()
    refs = eb.references()
    guards_global = global_guards(output, plan, records)
    global_pass = all(g["pass"] is True for g in guards_global.values())
    extra_guards = guard_extras(output)
    views = {arm: (records[arm], KEY) for arm in ARMS}
    blocks, tables = {}, {}
    guards_of = arm_guards(output, records)
    for inventory in eb.INVENTORIES:
        rows = eb.view_rows(cells, inventory, views, refs)
        tables[inventory] = {"rankers": eb.ranker_table(rows, ["U", "C*", "P", "J", "J_dec", *views])}
        if inventory == GRID:
            for arm in ARMS:
                blocks[arm] = eb.arm_block(rows, arm, records[arm], KEY, guards_of[arm], global_pass,
                                           arm_compute(output, arm), ARMS[arm]["role"], ARMS[arm]["program_set"],
                                           extra_guards=extra_guards)
        else:
            tables[inventory]["contrasts"] = {
                f"{arm} - U": eb.contrast(rows, arm, "U", global_pass and blocks[arm]["guards"]["pass"])
                for arm in ARMS}
    wrong = {arm: eb.wrong_anchor_table(cells, records[arm], arm, KEY, global_pass and blocks[arm]["guards"]["pass"])
             for arm in ARMS}
    gfid_path = output / "fidelity/gfid.json"
    gfid_block = {ARM: {"pass": extra_guards["G_fid"], "record": rel(gfid_path),
                        "sha256": eb.sha256_of(gfid_path) if gfid_path.is_file() else None}}
    ledger, gpu_total, wall_total = eb.ledger_totals(output, training_report_of(output))
    readouts_view = {f"{a}--seed{s}": {k: v for k, v in eb.read_json(readouts_dir(output, a, s) / "readouts.json").items()
                                       if k in ("policy_auc", "G_rank", "fit_anchor_nonfinite_share", "typed_failure",
                                                "replicated_from")} for a, s in arm_seeds()}
    summary = eb.child_summary(ISSUE, IDENTITY, plan["frozen_at"], blocks, {}, guards_global, gfid_block, {
        "inventories": tables, "wrong_anchor": wrong, "positions": None, "selection": plan["selected"],
        "coverage_arm": True, "zero_shot": True, "seeds_rule": plan["seeds_rule"], "readouts": readouts_view,
        "novelty": novelty_section(output, records), "native_success_head": None,
        "u_plus_head": "not applicable: DeepPHY's VLM agent has no success head"})
    compute = {"schema": "issue_139_compute_v1", "plan_identity": IDENTITY, "ledger": ledger,
               "gpu_seconds_total": gpu_total, "wall_seconds_total": wall_total,
               "arms": {arm: blocks[arm]["compute"] for arm in ARMS}, "reference": eb.reference_compute()}
    return summary, compute


def findings_text(summary):
    lines = [f"# #{ISSUE} T7 DeepPHY-VLM (coverage arm): findings", "",
             f"Global guards pass: {summary['global_guards_pass']}; selected configuration "
             f"{summary['extra']['selection']} (zero-shot; one deterministic pass written as three seeds).", "",
             *eb.format_arm_rows(summary["arms"]), "", "## Wrong anchor (right - wrong, grid)", ""]
    for arm, block in summary["extra"]["wrong_anchor"].items():
        lines.append(f"- {arm}: {eb.rc.fmt_block(block['right_minus_wrong'])} ({block['token']})")
    novelty = summary["extra"]["novelty"]
    if novelty is not None:
        lines += ["", f"## Novelty pairs ({novelty['table']['label']})", "",
                  f"Not scored (guards): {novelty['not_scored'] or 'none'}; novelty anchors read after the primary "
                  f"records: {novelty['chronology']['pass']}.", "", *eb.format_novelty_rows(novelty["table"])]
    lines += ["", "T7 is a coverage arm: never in the program claim. Intervals are descriptive (member-clustered "
              "bootstrap); tokens follow the #127 delta rule.", ""]
    return "\n".join(lines)


def not_run_reason(output):
    """The typed reason T7 is not run (no plan.json): the orchestrator's not-run.json marker (e.g. {"typed_reason":
    "cap", "decision": url}); None while T7 may still run."""
    marker = Path(output) / "not-run.json"
    return eb.read_json(marker)["typed_reason"] if marker.is_file() else None


def publish_not_run(output, reason):
    """summary.json of a T7 that was not run (typed): no arm block, extra.pending_arms {T7: reason}, with the G-port,
    G-fid and projection records that exist."""
    output = Path(output)
    gfid_path = output / "fidelity/gfid.json"
    ppath = output / "projection.json"
    gfid_block = {ARM: {"pass": gfid_pass(output), "record": rel(gfid_path),
                        "sha256": eb.sha256_of(gfid_path) if gfid_path.is_file() else None}}
    projection = None
    if ppath.is_file():
        full = eb.read_json(ppath)
        projection = {"primary": full["primary"], "sha256": eb.sha256_of(ppath),
                      "total_gpu_h": full[full["primary"]]["total_gpu_h"],
                      "range_gpu_h": [full[f"projection_{t}"]["total_gpu_h"] for t in ("lower", "upper")
                                      if f"projection_{t}" in full],
                      "remainder_gpu_h": REMAINDER_GPU_HOURS}
    marker = eb.read_json(output / "not-run.json") if (output / "not-run.json").is_file() else None
    summary = eb.child_summary(ISSUE, IDENTITY, None, {}, {}, {}, gfid_block, {
        "pending_arms": {ARM: reason}, "coverage_arm": True, "G_port": gport_pass(output), "projection": projection,
        "not_run_marker": marker, "decisions": DECISIONS})
    eb.write_json(output / "summary.json", summary)
    proj = "" if projection is None else (
        f" Projection ({projection['primary']}): {projection['total_gpu_h']:.1f} GPU-h against a remainder of "
        f"{REMAINDER_GPU_HOURS:.0f} GPU-h.")
    (output / "findings.md").write_text(
        f"# #{ISSUE} T7 DeepPHY-VLM (coverage arm): not run\n\nTyped reason: `{reason}`.{proj} G-port "
        f"{gport_pass(output)}; G-fid {gfid_pass(output)}. No records; T7 is listed as not run in the program "
        f"summary.\n")
    log(f"published not-run summary ({reason})")
    return 0


def publish(output):
    output = Path(output)
    if not (output / "plan.json").is_file():
        reason = not_run_reason(output)
        if reason is None:
            raise SystemExit("no plan.json and no typed not-run reason: nothing to publish")
        return publish_not_run(output, reason)
    eb.write_json(output / "records-manifest.json", eb.records_manifest(output, IDENTITY, RECORD_PATTERNS))
    summary, compute = publication(output)
    eb.write_json(output / "compute.json", compute)
    eb.write_json(output / "summary.json", summary)
    (output / "findings.md").write_text(findings_text(summary))
    log(f"published; G-records {summary['global_guards']['G_records']['pass']}")
    return 0


def validate(output, device=None):
    """Hash-check the records against the manifest, re-score the primary pass (exact row equality, every inventory
    and the wrong anchors), write the receipt, then re-derive the publication twice and require identical bytes."""
    output = Path(output)
    plan = load_plan(output)
    load_manifest(output)
    manifest = eb.read_json(output / "records-manifest.json")
    error, spot = None, None
    if manifest != eb.records_manifest(output, IDENTITY, manifest["patterns"]):
        error = "records differ from records-manifest.json"
    else:
        stored = eb.read_json(record_path(output, ARM, eb.SELECTION_SEED))
        anchors = torch.load(eb.evaluation_anchors_path(output), map_location="cpu", weights_only=True)
        with harness.deterministic_scoring():
            fresh, _ = score_pass(output, plan, "validate", anchors, eb.evaluation_cells(), device)
        fresh = json.loads(eb.json_text(fresh))
        same = (all(fresh["inventories"][inv] == stored["inventories"][inv] for inv in eb.INVENTORIES)
                and fresh["wrong_anchor"] == stored["wrong_anchor"])
        spot = {"record": f"{ARM}--seed{eb.SELECTION_SEED}", "pass": bool(same),
                "rule": "exact row equality, every inventory and the wrong anchors"}
    eb.write_receipt(output, IDENTITY, spot, error)
    publish(output)
    first = {name: (output / name).read_bytes() for name in ("summary.json", "compute.json", "findings.md")}
    publish(output)
    second = {name: (output / name).read_bytes() for name in first}
    stable = first == second
    receipt = eb.read_json(output / eb.VALIDATION_RECEIPT)
    ok = bool(receipt["pass"] and stable)
    log(f"validate: receipt {receipt['pass']}, publication stable {stable}")
    return 0 if ok else 1


# ---------------------------------------------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    modes = parser.add_mutually_exclusive_group(required=True)
    for mode in ("gport-inputs", "project", "timing", "selection", "selection-readouts", "smoke", "prepare", "train",
                 "readouts", "freeze-readouts", "score", "score-novelty", "publish", "validate"):
        modes.add_argument(f"--{mode}", action="store_true")
    parser.add_argument("--gpu", type=int, default=0, help="queue index (the job's CUDA_VISIBLE_DEVICES picks the GPU)")
    parser.add_argument("--device", default=None, help="worker device (default: cuda if visible, else cpu)")
    parser.add_argument("--output", default=str(OUTPUT))
    parser.add_argument("--timing-decisions", type=int, default=8, help="--timing: fit anchors scored per model")
    args = parser.parse_args(argv)
    output = Path(args.output)
    torch.set_num_threads(int(os.environ.get("B139_CPU_THREADS", "4")))
    try:
        if args.gport_inputs:
            return gport_inputs(output)
        if args.project:
            return project(output)
        if args.timing:
            return timing(output, args.timing_decisions, args.device)
        if args.selection:
            return selection(output, args.device)
        if args.selection_readouts:
            return selection_readouts(output)
        if args.smoke:
            return smoke(output)
        if args.prepare:
            return prepare(output)
        if args.train:
            return train(output)
        if args.readouts:
            return readouts(output, args.device)
        if args.freeze_readouts:
            return freeze_readouts(output)
        if args.score:
            return score(output, args.device)
        if args.score_novelty:
            return score_novelty(output, args.device)
        if args.publish:
            return publish(output)
        return validate(output, args.device)
    except eb.TrainingPaused as paused:
        log(f"paused ({paused}); exit {eb.EXIT_PAUSED}")
        return eb.EXIT_PAUSED


if __name__ == "__main__":
    raise SystemExit(main())
