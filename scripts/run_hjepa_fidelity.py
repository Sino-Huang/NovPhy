"""Issue-123 Stage 0: the H-JEPA FourRoom fidelity check (G-fid).

Binding runner module: scripts/run_hjepa_fidelity.py

Reproduces the paper's own FourRoom-Distractors hierarchy gain with the official code
(kevinghst/H-JEPA @ 97054ec, cloned at HJEPA_REF) in the separate ``hjepa`` conda env, so that a
NovPhy loss of H-JEPA cannot be our stack's bug:

- Tier E (eval-only): the released checkpoints of LeWM, H-JEPA-2 and H-JEPA-3, seeds 42/43/44, on
  the 50 released tasks (``eval.py``; planner seed = model seed).
- Tier T (retrain): LeWM, H-JEPA-2 and H-JEPA-3 retrained from scratch for seeds 42/43/44 with
  ``main_hjepa.py`` (their training method) through ``scripts/hjepa_train_ckpt.py``, which only enables
  the ViT's activation checkpointing (batch 128 does not fit a 32 GB RTX 5090 otherwise; gradients are
  unchanged); the end-of-training planning eval gives the success rate. The post-hoc probing/decoding
  eval is disabled (it does not touch the trained weights and is not part of G-fid); the loader
  workers/prefetch are lowered to fit 4 concurrent runs in host memory.

Pass rules (frozen in #123 before any run):
- Tier E passes iff every 3-seed mean lies in mean_paper +- 2 max(SE_paper, sqrt(p(1-p)/150)),
  LeWM < H-JEPA-2 < H-JEPA-3 holds on the means, and H-JEPA-3 - LeWM >= +30 pp on every seed.
- Tier T passes iff every 3-seed mean lies in mean_paper +- 2 sqrt(SE_paper^2 + SE_ours^2), and
  H-JEPA-3 - LeWM paired by seed has mean >= +30 pp with every seed positive.
- G-fid passes iff Tier E and Tier T both pass.

Modes (stdlib only; the H-JEPA processes run under the hjepa env's interpreter):
  --queue tier-e|tier-t --gpu G --jobs model:seed,...   run the jobs sequentially on one GPU
  --table                                              collect metrics, apply the rules, write gfid.json/gfid.md
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / ".local-artifacts/issue-123-hjepa-comparison-v1/fidelity"
HJEPA_REF = Path("/mnt/array/sukaih/hjepa-ref/H-JEPA")
HJEPA_HOME = Path("/mnt/array/sukaih/hjepa-home")
HJEPA_PYTHON = Path.home() / "miniconda3/envs/hjepa/bin/python"
COMMIT = "97054ec"
SEEDS = (42, 43, 44)
MODELS = {"lewm": "flat", "hjepa_l2": "l2", "hjepa_l3": "l3"}
PAPER = {"lewm": (40.7, 6.8), "hjepa_l2": (81.3, 6.7), "hjepa_l3": (96.0, 1.1)}  # docs/RESULTS.md, Fig. 6 bottom
TASKS = 50
MARGIN_PP = 30.0
TRAIN_OVERRIDES = ("final_probing_decoding_eval.enabled=false", "num_workers=8", "loader.prefetch_factor=2")
# Tier-T runs share 32 GB GPUs: allocator without fragmentation, cuDNN autotune workspace capped at 2 GiB
# (algorithm choice only; the math is unchanged). The first hjepa_l3 seed-42 launch without them hit a CUDA OOM
# at step 541 (26 GB allocated against a measured 9.4 GiB steady-state peak); that infrastructure failure is
# relaunched once with these settings and disclosed.
MEMORY_ENV = {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "CUDNN_CONV_WSCAP_DBG": "2048"}


def log(message):
    print(f"[issue-123-fidelity] {message}", flush=True)


def eval_dir(model, seed):
    return HJEPA_HOME / "fidelity" / "tier-e" / f"fourroom_{model}" / f"seed{seed}"


def train_subdir(model, seed):
    return f"fidelity/tier-t/fourroom_{model}/seed{seed}"


def train_dir(model, seed):
    return HJEPA_HOME / "ckpts" / train_subdir(model, seed)


def released_ckpt(model, seed):
    return HJEPA_HOME / "ckpts" / "fourroom" / f"fourroom_{model}" / f"seed{seed}" / f"fourroom_{model}_object.ckpt"


def read_success(metrics_path):
    """success_rate (%) from a planning-eval metrics.yaml (flat 'key: value' YAML)."""
    for line in Path(metrics_path).read_text().splitlines():
        if line.startswith("success_rate:"):
            return float(line.split(":", 1)[1])
    raise ValueError(f"no success_rate in {metrics_path}")


def tier_t_metrics(model, seed):
    found = sorted((train_dir(model, seed) / "planning_eval").glob("epoch_*/metrics.yaml"))
    return found[-1] if found else None


def ledger_path():
    return OUTPUT / "ledger.jsonl"


def record(entry):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with ledger_path().open("a") as handle:
        handle.write(json.dumps(entry, sort_keys=True) + "\n")


def command(tier, model, seed):
    if tier == "tier-e":
        out = eval_dir(model, seed)
        return [str(HJEPA_PYTHON), "eval.py", "--config-name", f"fourroom_{MODELS[model]}", f"seed={seed}",
                f"policy={released_ckpt(model, seed)}", f"output.dir={out}"], out / "metrics.yaml"
    return [str(HJEPA_PYTHON), str(ROOT / "scripts" / "hjepa_train_ckpt.py"), "--config-name", f"fourroom_{model}",
            f"seed={seed}", f"subdir={train_subdir(model, seed)}", *TRAIN_OVERRIDES], None


def queue(tier, gpu, jobs):
    """Run the (model, seed) jobs one after another on one GPU; a job whose metrics exist is skipped
    (typed failures are recorded and never retried)."""
    env = {**os.environ, "HJEPA_HOME": str(HJEPA_HOME), "CUDA_VISIBLE_DEVICES": str(gpu), "MUJOCO_GL": "egl",
           "PYTHONUNBUFFERED": "1", **(MEMORY_ENV if tier == "tier-t" else {})}
    logs = OUTPUT / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    for model, seed in jobs:
        done = (eval_dir(model, seed) / "metrics.yaml") if tier == "tier-e" else tier_t_metrics(model, seed)
        if done is not None and Path(done).is_file():
            log(f"{tier} {model} seed {seed}: metrics exist; skipped")
            continue
        argv, _ = command(tier, model, seed)
        began, wall0 = time.time(), time.monotonic()
        with (logs / f"{tier}--{model}--seed{seed}.log").open("a") as handle:
            proc = subprocess.Popen(argv, cwd=HJEPA_REF / "h_jepa", env=env, stdout=handle, stderr=subprocess.STDOUT)
            (logs / f"{tier}--{model}--seed{seed}.pid").write_text(f"{proc.pid}\n")
            log(f"{tier} {model} seed {seed} on GPU {gpu}: worker pid {proc.pid}")
            code = proc.wait()
        wall = time.monotonic() - wall0
        record({"tier": tier, "model": model, "seed": seed, "gpu": gpu, "pid": proc.pid, "exit_code": code,
                "started_unix": began, "wall_seconds": wall, "gpu_seconds": wall})
        log(f"{tier} {model} seed {seed}: exit {code} after {wall:.0f}s")
    return 0


def mean_se(values):
    return statistics.fmean(values), (statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0)


def tier_table(tier):
    rows = {}
    for model in MODELS:
        per_seed = {}
        for seed in SEEDS:
            path = (eval_dir(model, seed) / "metrics.yaml") if tier == "tier-e" else tier_t_metrics(model, seed)
            per_seed[seed] = read_success(path) if path is not None and Path(path).is_file() else None
        rows[model] = per_seed
    complete = all(v is not None for r in rows.values() for v in r.values())
    out = {"complete": complete, "models": {}}
    for model, per_seed in rows.items():
        paper_mean, paper_se = PAPER[model]
        entry = {"per_seed": {str(s): v for s, v in per_seed.items()}, "paper_mean": paper_mean, "paper_se": paper_se}
        if complete:
            mean, se = mean_se(list(per_seed.values()))
            if tier == "tier-e":
                p = paper_mean / 100.0
                half = 2 * max(paper_se, 100.0 * math.sqrt(p * (1 - p) / (len(SEEDS) * TASKS)))
            else:
                half = 2 * math.sqrt(paper_se ** 2 + se ** 2)
            band = [paper_mean - half, paper_mean + half]
            entry.update({"mean": mean, "se": se, "band": band, "in_band": band[0] <= mean <= band[1]})
        out["models"][model] = entry
    if not complete:
        out["pass"] = None
        return out
    diffs = [rows["hjepa_l3"][s] - rows["lewm"][s] for s in SEEDS]
    means = {m: out["models"][m]["mean"] for m in MODELS}
    out["hjepa3_minus_lewm"] = {"per_seed": {str(s): d for s, d in zip(SEEDS, diffs)}, "mean": statistics.fmean(diffs)}
    bands = all(out["models"][m]["in_band"] for m in MODELS)
    if tier == "tier-e":
        out["ordering"] = means["lewm"] < means["hjepa_l2"] < means["hjepa_l3"]
        out["gain_every_seed_ge_30"] = all(d >= MARGIN_PP for d in diffs)
        out["pass"] = bool(bands and out["ordering"] and out["gain_every_seed_ge_30"])
    else:
        out["gain_mean_ge_30_all_positive"] = statistics.fmean(diffs) >= MARGIN_PP and all(d > 0 for d in diffs)
        out["pass"] = bool(bands and out["gain_mean_ge_30_all_positive"])
    return out


def compute_ledger():
    if not ledger_path().is_file():
        return {}
    out = {}
    for line in ledger_path().read_text().splitlines():
        entry = json.loads(line)
        block = out.setdefault(entry["tier"], {"runs": 0, "gpu_seconds": 0.0})
        block["runs"] += 1
        block["gpu_seconds"] += entry["gpu_seconds"]
    return out


def fmt(value, digits=1):
    return "n/a" if value is None else f"{value:.{digits}f}"


def table(_args=None):
    tiers = {tier: tier_table(tier) for tier in ("tier-e", "tier-t")}
    gfid = None
    if all(t["pass"] is not None for t in tiers.values()):
        gfid = bool(tiers["tier-e"]["pass"] and tiers["tier-t"]["pass"])
    freeze = HJEPA_HOME / "env-freeze.txt"
    result = {"schema": "issue_123_gfid_v1", "commit": COMMIT, "environment": "FourRoom Distractors, 50 released tasks",
              "rules": __doc__.split("Pass rules (frozen in #123 before any run):")[1].split("Modes")[0].strip(),
              "tiers": tiers, "G_fid": gfid, "compute": compute_ledger(),
              "env_freeze_sha256": __import__("hashlib").sha256(freeze.read_bytes()).hexdigest() if freeze.is_file() else None,
              "train_overrides": list(TRAIN_OVERRIDES)}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "gfid.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    lines = ["# #123 G-fid: FourRoom Distractors fidelity check", "",
             f"Official code {COMMIT}; success % on the 50 released tasks; seeds 42/43/44.", ""]
    for tier, name in (("tier-e", "Tier E (released checkpoints)"), ("tier-t", "Tier T (retrained with main_hjepa.py)")):
        t = tiers[tier]
        lines += [f"## {name}: {'PASS' if t['pass'] else ('FAIL' if t['pass'] is False else 'incomplete')}", "",
                  "| model | seed 42 | seed 43 | seed 44 | mean ± SE | paper | band | in band |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- |"]
        for model, e in t["models"].items():
            ps = e["per_seed"]
            band = f"[{e['band'][0]:.1f}, {e['band'][1]:.1f}]" if "band" in e else "n/a"
            lines.append(f"| {model} | {fmt(ps['42'])} | {fmt(ps['43'])} | {fmt(ps['44'])} | "
                         f"{fmt(e.get('mean'))} ± {fmt(e.get('se'))} | {e['paper_mean']} ± {e['paper_se']} | {band} | "
                         f"{e.get('in_band', 'n/a')} |")
        if "hjepa3_minus_lewm" in t:
            d = t["hjepa3_minus_lewm"]
            lines.append("")
            lines.append(f"H-JEPA-3 − LeWM per seed {d['per_seed']}, mean {d['mean']:.1f} pp; "
                         + (f"ordering {t['ordering']}, every seed ≥ 30: {t['gain_every_seed_ge_30']}" if tier == "tier-e"
                            else f"mean ≥ 30 with every seed positive: {t['gain_mean_ge_30_all_positive']}"))
        lines.append("")
    lines.append(f"**G-fid: {'PASS' if gfid else ('FAIL' if gfid is False else 'incomplete')}**")
    (OUTPUT / "gfid.md").write_text("\n".join(lines) + "\n")
    log(f"G-fid {gfid}; Tier E {tiers['tier-e']['pass']}; Tier T {tiers['tier-t']['pass']}")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--queue", choices=("tier-e", "tier-t"))
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--jobs", help="model:seed,...")
    parser.add_argument("--table", action="store_true")
    args = parser.parse_args()
    if args.table:
        return table()
    jobs = [(item.split(":")[0], int(item.split(":")[1])) for item in args.jobs.split(",")]
    for model, _ in jobs:
        if model not in MODELS:
            raise ValueError(f"unknown model {model}")
    return queue(args.queue, args.gpu, jobs)


if __name__ == "__main__":
    raise SystemExit(main())
