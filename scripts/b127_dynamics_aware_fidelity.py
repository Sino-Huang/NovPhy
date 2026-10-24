"""#128 A4 G-fid Tier T: retrain the official hand-crafted dynamics-aware DQN on PHYRE-1B within-template folds 0-2.

Env b127-dynaware (Python 3.11, torch 2.13+cu130, upstream PHYRE d0765dd built from source), the vendored official
code (world_model/baselines/dynamics_aware/vendor, VENDOR.json). One fold per process, through the official entry
point ``agents/train.py`` (``parse_and_log_args`` + ``main``) with exactly the command line that
``agents/run_sweep_file.py`` builds from ``06_hand_crafted_test_run_table_1.conf`` (the paper's Table-1 run) for
``ball_within_template``: train on train + dev of the fold, 100 000 updates x 512 (64 tasks x 8 actions, aux loss on
the per-batch PHYRE rollouts from 20 simulator workers), evaluate AUCCESS@100 on the fold's test tasks by ranking the
first 10 000 cached actions (``results.json["target_metric"]``, x 100 = the paper's unit).

    PY=/srv/home/shua0171/miniconda3/envs/b127-dynaware/bin/python
    $PY scripts/b127_dynamics_aware_fidelity.py --fold 0            # train + eval fold 0 (resumable; GPU job)
    $PY scripts/b127_dynamics_aware_fidelity.py --probe-cpu         # CPU throughput -> fidelity/a4/probe-cpu.json
    $PY scripts/b127_dynamics_aware_fidelity.py --probe-gpu         # GPU timing (600 updates) -> probe-gpu.json
    $PY scripts/b127_dynamics_aware_fidelity.py --table             # gfid.json from the three folds vs band.json

Resumability (night window): SIGTERM / SIGUSR1 set the vendored ``neural_agent.STOP_REQUESTED`` (vendor edit E3);
after the current optimizer step the official checkpoint dict (model, Adam, cosine scheduler, numpy sampler RNG,
done batches) is written atomically and the process exits 75. A relaunch resumes through the official
``get_latest_checkpoint`` path; training has no torch RNG draws, so the resumed batch sequence and state are the
uninterrupted run's (tests/test_issue_128_dynamics_aware.py checks this bitwise on CPU). A finished fold
(``results.json`` present) exits 0 without work. A stop during the final evaluation is not honoured (it takes
minutes); the 07:00 kill would make the relaunch redo only the evaluation from ``ckpt.00100000``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import multiprocessing
import os
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / ".local-artifacts" / "issue-128-direct-predictors-v1" / "fidelity" / "a4"
FAMILY = ROOT / "world_model" / "baselines" / "dynamics_aware"
PHYRE_CACHE = ROOT / "data" / "external" / "dynamics_aware" / "phyre_cache"
EVAL_SETUP = "ball_within_template"
FOLDS = (0, 1, 2)
EXIT_PAUSED = 75
# torch intra-op CPU threads of the training process (the aux distance labels run on the CPU); torch's default (one per
# core, 48) under the shared host's load slowed the official loop ~7x (0.3 vs 2.2 updates/s); a runtime setting only
CPU_THREADS = int(os.environ.get("B127_A4_CPU_THREADS", "8"))
CONF = FAMILY / "vendor" / "agents" / "experiments" / "hand_crafted_experiments" / \
    "06_hand_crafted_test_run_table_1.conf"


def official_argv(fold, out_dir, overrides=None):
    """``run_sweep_file.build_command`` for conf 06 at EVALSETUP=ball_within_template, SEED=fold (mandatory
    parameters substituted into its base command, then the conf's other lines appended in conf order; rank size
    'optimal' -> 10000 for ball_within_template), with basedir paths mapped into ``out_dir``. ``overrides``
    (tests / probes only) replace a flag's value or append it."""
    out_dir = Path(out_dir)
    argv = [
        "--use-test-split=1", f"--output-dir={out_dir / 'output'}", f"--eval-setup-name={EVAL_SETUP}",
        f"--fold-id={fold}", "--dqn-save-checkpoints-every=5000", "--dqn-updates=100000", "--dqn-cosine-scheduler=1",
        "--dqn-learning-rate=3e-4", "--dqn-train-batch-size=512", "--dqn-balance-classes=1", "--dqn-rank-size=10000",
        "--dqn-num-auccess-actions=10000", "--dqn-eval-every=5000", "--dqn-num-auccess-actions=10000",
        f"--dqn-tensorboard-dir={out_dir / 'tensorboard'}", "--dqn-network-type=resnet18",
        "--agent-type=dqn", "--dqn-n-samples-per-task=8", "--dqn-use-sample-distance-aux-loss=1",
        "--dqn-lhp-n-regression-bins=20", "--dqn-lhp-distance-cutoff=1.414", "--dqn-lhp-regression-type=binned",
        "--dqn-lhp-repr-merging-method=outer", "--dqn-lhp-embeddor-type=linear", "--dqn-lhp-time-slice=0",
        "--dqn-eval-batch-size=1024", f"--dqn-checkpoint-dir={out_dir / 'checkpoints'}",
    ]
    for flag, value in (overrides or {}).items():
        hits = [i for i, a in enumerate(argv) if a.split("=", 1)[0] == flag]
        if hits:
            for i in hits:
                argv[i] = f"{flag}={value}"
        else:
            argv.append(f"{flag}={value}")
    return argv


def sha256(path):
    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=1, sort_keys=True) + "\n")
    os.replace(tmp, path)


def import_official():
    """The vendored official package, imported as top-level ``vendor`` (world_model/__init__ is not needed here)."""
    os.environ.setdefault("PHYRE_CACHE_DIR", str(PHYRE_CACHE))
    sys.path.insert(0, str(FAMILY))
    import vendor.agents.train as dtrain
    from vendor.agents import neural_agent
    return dtrain, neural_agent


def clean_checkpoints(ckpt_dir):
    """Drop a stop checkpoint's leftover .tmp and a truncated latest checkpoint (a kill during a periodic save)."""
    import torch
    ckpt_dir = Path(ckpt_dir)
    removed = []
    for tmp in ckpt_dir.glob("ckpt.*.tmp"):
        tmp.unlink()
        removed.append(tmp.name)
    known = sorted(p for p in ckpt_dir.glob("ckpt.*") if not p.name.endswith(".corrupt"))
    if known:
        try:
            torch.load(known[-1], weights_only=False, map_location="cpu")
        except Exception as error:  # noqa: BLE001 - any unreadable checkpoint is set aside, never resumed
            known[-1].rename(known[-1].with_name(known[-1].name + ".corrupt"))
            removed.append(f"{known[-1].name} -> .corrupt ({type(error).__name__})")
    return removed


def run_fold(fold, out_dir, overrides=None):
    """Train (or resume) and evaluate one fold with the official entry point; returns 0 (done) or 75 (paused)."""
    out_dir = Path(out_dir)
    results = out_dir / "output" / "results.json"
    if results.is_file():
        print(f"fold {fold}: results.json present, nothing to do")
        return 0
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    cleaned = clean_checkpoints(out_dir / "checkpoints")
    dtrain, neural_agent = import_official()
    import torch
    torch.set_num_threads(CPU_THREADS)

    def request_stop(signum, _frame):
        logging.warning("signal %s: stop requested (checkpoint after the current optimizer step)", signum)
        neural_agent.STOP_REQUESTED.set()

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGUSR1, request_stop)
    logging.basicConfig(format="%(asctime)s %(levelname)-8s {%(module)s:%(lineno)d} %(message)s",
                        level=logging.DEBUG, datefmt="%Y-%m-%d %H:%M:%S")
    argv = official_argv(fold, out_dir, overrides)
    began, started = time.time(), time.strftime("%Y-%m-%dT%H:%M:%S%z")
    previous = sorted(p.name for p in (out_dir / "checkpoints").glob("ckpt.*") if not p.name.endswith(".corrupt"))
    sys.argv = ["train.py", *argv]
    outcome, done = "completed", None
    try:
        dtrain.main(**vars(dtrain.parse_and_log_args()))
    except neural_agent.TrainingPaused as paused:
        outcome, done = "paused", int(paused.args[0])
    finally:
        # the official ParallelPhyreSimulator workers are non-daemon processes closed only by a weakref.finalize
        # that holds the simulator alive; at interpreter exit multiprocessing would join them forever. They are
        # forked after our SIGTERM handler is installed, so terminate() would only set the stop flag: SIGKILL.
        for worker in multiprocessing.active_children():
            worker.kill()
            worker.join(10)
    row = {"fold": fold, "started": started, "wall_seconds": time.time() - began, "outcome": outcome,
           "resumed_from": previous[-1] if previous else None, "paused_at_batch": done, "cleaned": cleaned,
           "gpu_window_run": os.environ.get("GPU_WINDOW_RUN"), "cuda_visible_devices":
               os.environ.get("CUDA_VISIBLE_DEVICES"), "argv": argv}
    with open(out_dir / "runs.jsonl", "a") as stream:
        stream.write(json.dumps(row) + "\n")
    if outcome == "paused":
        print(f"fold {fold}: paused after batch {done}; exit {EXIT_PAUSED}")
        return EXIT_PAUSED
    target = json.loads(results.read_text())["target_metric"]
    print(f"fold {fold}: AUCCESS {100 * target:.2f}")
    return 0


def probe_cpu(out_path, batches=5):
    """CPU timings of the parts of one official update (batch 512): the 20-worker PHYRE rollouts, the aux-loss
    distance labels, and forward + backward of the model; FLOPs of one update for the GPU projection."""
    import numpy as np
    import torch
    from torch.utils.flop_counter import FlopCounterMode
    os.environ.setdefault("PHYRE_CACHE_DIR", str(PHYRE_CACHE))
    _, neural_agent = import_official()
    import phyre
    from vendor.agents import psim
    torch.set_num_threads(16)
    train, dev, _ = phyre.get_fold(EVAL_SETUP, 0)
    tasks = train + dev
    cache = phyre.get_default_100k_cache("ball")
    data = cache.get_sample(tasks)
    statuses = data["simulation_statuses"]
    rng = np.random.RandomState(0)
    sim = psim.ParallelPhyreSimulator(tasks, "ball", 20, neural_agent.MAX_LEN, 512, requires_imgs=False)
    simulator = phyre.initialize_simulator(tasks, "ball")
    hp = {"regression_type": "binned", "repr_merging_method": "outer", "distance_cutoff": 1.414,
          "n_regression_bins": 20, "time_slice": 0, "embedding_dim": 256, "embeddor_type": "linear", "weight": 1.0}
    model = neural_agent.build_model("resnet18", action_space_dim=3, fusion_place="last", action_hidden_size=256,
                                     action_layers=1, embedding_dim=256, embeddor_type="linear",
                                     repr_merging_method="outer", n_regressor_outputs=20)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    t_sim, t_aux_labels, t_step, flops = [], [], [], None
    for b in range(batches):
        task_idx = rng.choice(len(tasks), 64, replace=False)
        rows, acts, picked = [], [], []
        for t in task_idx:
            pos = np.flatnonzero(statuses[t] == 1)
            neg = np.flatnonzero(statuses[t] == -1)
            rows += [t] * 8
            picks = np.concatenate([rng.choice(neg, 4), rng.choice(pos if len(pos) else neg, 4)])
            picked.append(statuses[t][picks] == 1)
            acts.append(cache.action_array[picks])
        rows, acts = np.array(rows), np.concatenate(acts).astype(np.float32)
        t0 = time.perf_counter()
        _, _, feats, masks = sim.simulate_parallel(rows, acts, need_images=False, need_featurized_objects=True)
        t_sim.append(time.perf_counter() - t0)
        feats, masks = torch.from_numpy(feats), torch.from_numpy(masks)
        t0 = time.perf_counter()
        for i in range(0, 512, 8):
            neural_agent.rollout_dist.measure_all_distances(feats[i:i + 8], masks[i:i + 8], 1.414)
        t_aux_labels.append(time.perf_counter() - t0)
        obs = torch.LongTensor(simulator.initial_scenes)[torch.from_numpy(rows)]
        labels = torch.from_numpy(np.concatenate(picked))
        t0 = time.perf_counter()
        counter = FlopCounterMode(display=False) if flops is None else None
        if counter is not None:
            counter.__enter__()
        logits, emb = model(obs, torch.from_numpy(acts), get_embeddings=True)
        aux = neural_agent.sample_distance_loss(model, emb, rows, feats, masks, None, 8, hp)
        loss = model.ce_loss(logits, labels).mean() + aux
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if counter is not None:
            counter.__exit__(None, None, None)
            flops = counter.get_total_flops()
        t_step.append(time.perf_counter() - t0)
        print(f"batch {b}: sim {t_sim[-1]:.3f}s aux-labels {t_aux_labels[-1]:.3f}s cpu-step {t_step[-1]:.1f}s")
    sim.close()
    med = lambda xs: float(np.median(xs[1:] if len(xs) > 2 else xs))  # noqa: E731
    per_update_cpu = med(t_sim) + med(t_aux_labels)
    projection = {}
    for tflops in (25.0, 50.0):  # effective RTX 5090 fp32/TF32-conv throughput assumed for ResNet-18 fwd+bwd
        gpu = flops / (tflops * 1e12)
        hours = 100_000 * (per_update_cpu + gpu) / 3600
        projection[f"{tflops:g}_tflops"] = {"gpu_compute_s_per_update": gpu, "update_s": per_update_cpu + gpu,
                                            "train_h_per_fold": hours, "three_folds_h": 3 * hours}
    record = {"schema": "issue_128_a4_probe_cpu_v1", "batches": batches, "threads": 16, "sim_workers": 20,
              "sim_s_per_update": med(t_sim), "aux_label_s_per_update": med(t_aux_labels),
              "cpu_fwd_bwd_s_per_update": med(t_step), "flops_per_update": flops, "projection": projection,
              "note": "the official loop is synchronous: update time = PHYRE rollouts (20 CPU workers) + aux labels "
                      "(CPU) + GPU step; in-training evals (every 5000 updates) and the final test eval are extra "
                      "(measured by --probe-gpu)"}
    write_json(out_path, record)
    print(json.dumps(record, indent=1))
    return 0


class _Timeline(logging.Handler):
    def __init__(self):
        super().__init__()
        self.events = []

    def emit(self, record):
        self.events.append((record.created, record.getMessage()))


def probe_gpu(out_dir):
    """Official fold-0 run shortened to 600 updates in a scratch dir (timing only, never a fidelity result): the
    initial in-training eval, the update rate, and the final test evaluation."""
    import shutil
    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    logging.basicConfig(format="%(asctime)s %(levelname)-8s {%(module)s:%(lineno)d} %(message)s",
                        level=logging.DEBUG, datefmt="%Y-%m-%d %H:%M:%S")
    timeline = _Timeline()
    logging.getLogger().addHandler(timeline)
    began = time.time()
    code = run_fold(0, out_dir, {"--dqn-updates": 600, "--dqn-save-checkpoints-every": 0})
    end = time.time()
    logging.getLogger().removeHandler(timeline)
    ev = timeline.events

    def first(predicate):
        return next((t for t, m in ev if predicate(m)), None)

    import torch
    marks = {"iter100": first(lambda m: m.startswith("Iter: 100,")),
             "iter600": first(lambda m: m.startswith("Iter: 600,")),
             "eval0_start": first(lambda m: m == "Start eval"), "eval0_end": first(lambda m: m.startswith("__log__")),
             "final_eval_start": first(lambda m: m == "Starting eval")}
    record = {"schema": "issue_128_a4_probe_gpu_v1", "exit": code, "wall_s": end - began, "marks": marks,
              "max_memory_allocated_gb": torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else None,
              "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}
    if None not in marks.values():
        rate = 500 / (marks["iter600"] - marks["iter100"])
        per_eval = marks["eval0_end"] - marks["eval0_start"]
        final = end - marks["final_eval_start"]
        fold_h = (100_000 / rate + 21 * per_eval + final) / 3600
        record.update({"updates_per_s": rate, "in_training_eval_s": per_eval, "final_eval_s": final,
                       "projected_h_per_fold": fold_h, "projected_h_three_folds": 3 * fold_h})
    write_json(OUT / "probe-gpu.json", record)
    print(json.dumps(record, indent=1))
    return code


def table():
    band = json.loads((OUT / "band.json").read_text())
    folds, missing = {}, []
    for k in FOLDS:
        path = OUT / f"fold{k}" / "output" / "results.json"
        if not path.is_file():
            missing.append(k)
            continue
        folds[str(k)] = 100.0 * json.loads(path.read_text())["target_metric"]
    if missing:
        print(f"folds {missing} not finished")
        return 1
    mean = sum(folds.values()) / len(folds)
    lo, hi = band["band"]
    runs = [json.loads(line) for k in FOLDS for line in (OUT / f"fold{k}" / "runs.jsonl").read_text().splitlines()]
    record = {"schema": "issue_127_gfid_v1", "issue": 128, "arm": "A4", "tier": "T",
              "band_sha256": sha256(OUT / "band.json"), "paper": band["paper"], "band": band["band"],
              "rule": band["rule"], "per_fold": folds, "mean": mean, "in_band": lo <= mean <= hi,
              "runs": len(runs), "wall_seconds": sum(r["wall_seconds"] for r in runs),
              "vendor_json_sha256": sha256(FAMILY / "vendor" / "VENDOR.json"),
              "runner_sha256": sha256(Path(__file__))}
    record["G_fid"] = record["pass"] = record["in_band"]
    write_json(OUT / "gfid.json", record)
    print(json.dumps(record, indent=1))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--fold", type=int, choices=FOLDS)
    group.add_argument("--probe-cpu", action="store_true")
    group.add_argument("--probe-gpu", action="store_true")
    group.add_argument("--table", action="store_true")
    args = parser.parse_args()
    if args.fold is not None:
        if not (OUT / "band.json").is_file():
            raise SystemExit("fidelity/a4/band.json must be frozen (and posted on #128) before any Tier-T run")
        return run_fold(args.fold, OUT / f"fold{args.fold}")
    if args.probe_cpu:
        return probe_cpu(OUT / "probe-cpu.json")
    if args.probe_gpu:
        return probe_gpu(OUT / "probe-gpu-scratch")
    return table()


if __name__ == "__main__":
    raise SystemExit(main())
