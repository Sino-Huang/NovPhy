"""#128 A1 Tier-E G-fid: the shipped PHYRE DQN checkpoints on PHYRE-B within-template, 10 folds, official evaluation.

Run in conda env ``b127-phyre`` on one GPU (``DQNAgent.real_eval`` calls ``model.cuda()``), through the night-window
scheduler (job ``128-gfid-a1-tier-e``):

    /srv/home/shua0171/miniconda3/envs/b127-phyre/bin/python scripts/b127_phyre_dqn_fidelity.py

For each fold k (resumable per fold: a fold with ``fidelity/runs/fold-k/results.json`` is skipped) the unmodified
official ``agents/train.py`` from the pinned clone runs in-process with the argv frozen in ``fidelity/band.json``
(``--dqn-load-from`` = the shipped ``dqn_10k/ball_within_template/k`` checkpoint folder) and writes its own
``results.json``. Environment only: ``PHYRE_CACHE_DIR`` (the downloaded 100k cache) and
``TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`` (torch >= 2.6 loads with weights_only by default; the 2020 checkpoints pickle
the numpy sampler state). SIGTERM / SIGINT -> exit 75 (the interrupted fold reruns on relaunch). After the 10th fold
``fidelity/gfid.json`` (schema ``issue_127_gfid_v1``) records per-fold AUCCESS x 100, the mean, the frozen band (its
sha256 checked against decisions.md) and pass.

``--smoke N`` (CPU check of the pipeline, never the gate): fold 0, the first N test tasks, ``Module.cuda`` the
identity and checkpoints mapped to CPU; writes ``fidelity/smoke/`` only.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / ".local-artifacts/issue-128-direct-predictors-v1"
FID = ART / "fidelity"
EXTERNAL = ROOT / "data/external/phyre_dqn"
CKPTS = EXTERNAL / "results/finals/dqn_10k/ball_within_template"
PHYRE_REF = Path(os.environ.get("PHYRE_REF", "/mnt/array/sukaih/novphy-baselines-src/phyre@d0765dd"))
FOLDS = range(10)
EXIT_PAUSED = 75


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def sha256_of(path):
    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fold_argv(fold, output_dir):
    return ["agents/train.py", "--use-test-split=1", "--eval-setup-name=ball_within_template", f"--fold-id={fold}",
            "--agent-type=dqn", f"--dqn-load-from={CKPTS / str(fold)}", "--dqn-rank-size=10000",
            "--dqn-eval-batch-size=128", "--dqn-finetune-iterations=0", "--dqn-refine-iterations=0",
            "--max-test-attempts-per-task=100", f"--output-dir={output_dir}"]


def run_fold(train, fold, output_dir):
    sys.argv = fold_argv(fold, output_dir)
    train.main(**vars(train.parse_and_log_args()))


def band_checked():
    band_path = FID / "band.json"
    digest = sha256_of(band_path)
    if digest not in (ART / "decisions.md").read_text():
        raise SystemExit(f"band.json {digest} is not the sha256 recorded in decisions.md")
    return json.loads(band_path.read_text()), digest


def write_gfid(band, digest, walls):
    folds = {}
    for fold in FOLDS:
        result = json.loads((FID / "runs" / f"fold-{fold}" / "results.json").read_text())
        shipped = json.loads((CKPTS / str(fold) / "results.json").read_text())["target_metric"]
        folds[str(fold)] = {"auccess": 100 * result["target_metric"], "num_eval_tasks": result["num_eval_tasks"],
                            "shipped_results_json_auccess": 100 * shipped,
                            "delta_vs_shipped": 100 * (result["target_metric"] - shipped),
                            "checkpoint_sha256": sha256_of(CKPTS / str(fold) / "ckpt.00100000"),
                            "wall_s": walls.get(str(fold))}
    values = [f["auccess"] for f in folds.values()]
    mean = statistics.mean(values)
    lo, hi = band["band"]
    import torch
    gfid = {"schema": "issue_127_gfid_v1", "issue": 128, "arm": "A1", "tier": "E", "written_at": utc_now(),
            "band_file": "fidelity/band.json", "band_sha256": digest, "band": band["band"], "paper": band["paper"],
            "metric": band["metric"], "per_fold": folds, "mean": mean, "sd": statistics.stdev(values),
            "shipped_mean": statistics.mean(f["shipped_results_json_auccess"] for f in folds.values()),
            "pass": bool(lo <= mean <= hi), "repair_attempts": 0,
            "env": {"python": sys.version.split()[0], "torch": torch.__version__,
                    "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                    "phyre_ref": str(PHYRE_REF)}}
    (FID / "gfid.json").write_text(json.dumps(gfid, indent=2) + "\n")
    print(f"[phyre-gfid] mean AUCCESS {mean:.2f} (band {band['band']}), pass={gfid['pass']}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", type=int, default=0, help="CPU pipeline check on the first N fold-0 test tasks")
    args = parser.parse_args()
    os.environ.setdefault("PHYRE_CACHE_DIR", str(EXTERNAL / "phyre_cache"))
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

    def stop(signum, frame):
        print(f"[phyre-gfid] signal {signum}: pausing (exit {EXIT_PAUSED})", flush=True)
        sys.exit(EXIT_PAUSED)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    os.chdir(PHYRE_REF)
    sys.path.insert(0, str(PHYRE_REF / "agents"))
    import torch
    import train  # official agents/train.py

    if args.smoke:
        torch.nn.Module.cuda = lambda self, *a, **k: self
        original_load, original_split = torch.load, train.get_train_test
        torch.load = lambda *a, **k: original_load(*a, **{**k, "map_location": "cpu"})
        train.get_train_test = lambda *a: (lambda tr, te: (tr, te[:args.smoke]))(*original_split(*a))
        out = FID / "smoke" / "fold-0"
        run_fold(train, 0, out)
        print(f"[phyre-gfid] smoke: {json.loads((out / 'results.json').read_text())['target_metric']:.4f} on "
              f"{args.smoke} tasks", flush=True)
        return 0

    band, digest = band_checked()
    if not torch.cuda.is_available():
        raise SystemExit("the official DQNAgent.real_eval needs a GPU (model.cuda()); run via the night-window scheduler")
    walls_path = FID / "runs" / "walls.json"
    walls = json.loads(walls_path.read_text()) if walls_path.is_file() else {}
    for fold in FOLDS:
        out = FID / "runs" / f"fold-{fold}"
        if (out / "results.json").is_file():
            continue
        start = time.time()
        run_fold(train, fold, out)
        walls[str(fold)] = time.time() - start
        walls_path.write_text(json.dumps(walls, indent=2) + "\n")
        print(f"[phyre-gfid] fold {fold}: "
              f"{100 * json.loads((out / 'results.json').read_text())['target_metric']:.2f} "
              f"in {walls[str(fold)]:.0f} s", flush=True)
    write_gfid(band, digest, walls)
    return 0


if __name__ == "__main__":
    sys.exit(main())
