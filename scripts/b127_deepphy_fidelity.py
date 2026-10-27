"""#139 T7 G-fid (env b127-deepphy; Decision P-fid on #139): DeepPHY's PHYRE benchmark with Qwen2.5-VL-7B-Instruct,
VLA prompt format, run by the official ``deepphy.runner.01_phyre_runner`` of the pinned clone (XinrunXu/DeepPHY @
0562104; released ``conf/env_config_01_phyre.json``: 8 x 8 grid, 8 radius levels, 10 attempts, 5 image-history
attempts) on the test tasks of fold 0 of ``ball_within_template`` and ``ball_cross_template`` (500 + 500 = the paper's
1 000 tasks). Published quantity: Table 6 (VLA), Qwen-7B success within 10 attempts = 9.62 %.

Modes:
- ``--run`` (GPU, scheduler): the official ``main()`` per setup with ``--reuse_log_dir fidelity/runs/<setup>`` (its own
  per-task resume). The only instrumentation: ``process_single_task_with_retries`` is wrapped so that a SIGTERM /
  SIGUSR1 stops before the next task (after the official summary of the previous one is written) with exit 75, and
  ``torch`` / ``random`` / ``numpy`` are seeded at process start (seed + tasks already done) since DeepPHY samples
  (temperature 0.1) from the global RNG.
- ``--mock`` (CPU): the same official runner with ``--model mock`` (DeepPHY's random agent, ``random`` seeded) on the
  same 1 000 tasks into ``fidelity/mock/<setup>`` (descriptive environment check against the paper's MOCK 7.64 %).
- ``--finish`` (CPU): fidelity/gfid.json from the official summaries and the frozen band.json.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import random
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
CLONE = Path("/mnt/array/sukaih/novphy-baselines-src/DeepPHY@0562104")
OUTPUT = ROOT / ".local-artifacts/issue-139-deepphy-v1"
MODEL = "Qwen2.5-VL-7B-Instruct"
SETUPS = ("ball_within_template", "ball_cross_template")
SEED = 20260908
EXIT_PAUSED = 75
_STOP = {"requested": False}


def log(message):
    print(f"[issue-139 gfid] {message}", flush=True)


def sha(path):
    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _request_stop(signum, frame):
    _STOP["requested"] = True


class Paused(Exception):
    pass


def done_tasks(run_dir):
    path = Path(run_dir) / "summary_results.json"
    if not path.is_file():
        return 0
    return sum(1 for r in json.loads(path.read_text())[1:] if "seed" in r)


def run_setup(setup, model, run_dir):
    """The official 01_phyre_runner.main() for one setup (resumes from run_dir/summary_results.json)."""
    os.chdir(CLONE)
    if str(CLONE) not in sys.path:
        sys.path.insert(0, str(CLONE))
    runner = importlib.import_module("deepphy.runner.01_phyre_runner")
    official = runner.process_single_task_with_retries

    def guarded(*args, **kwargs):
        if _STOP["requested"]:
            raise Paused()
        return official(*args, **kwargs)

    seed = SEED + done_tasks(run_dir)
    random.seed(seed)
    import numpy as np
    if not hasattr(np, "float"):
        np.float = float  # phyre/vis.py's np.float (the removed alias of the builtin float, i.e. float64 arrays)
    np.random.seed(seed % 2**32)
    if model != "mock":
        import torch
        torch.manual_seed(seed)
    sys.argv = ["01_phyre_runner", "--model", model, "--eval_setup", setup, "--format", "VLA", "--eval_type", "test",
                "--reuse_log_dir", str(run_dir)]
    runner.main()


def mark(output, name, seconds, gpu):
    """ledger/<name>.json: wall / GPU seconds accumulated over runs (the rc.mark format)."""
    path = Path(output) / "ledger" / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    block = json.loads(path.read_text()) if path.is_file() else {"wall_seconds": 0.0, "gpu_seconds": 0.0}
    block.update(wall_seconds=block["wall_seconds"] + seconds, gpu_seconds=block["gpu_seconds"] + (seconds if gpu else 0.0),
                 last_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(block, indent=1, sort_keys=True) + "\n")
    os.replace(tmp, path)


def run(output, model, kind):
    output = Path(output)
    for signum in (signal.SIGTERM, signal.SIGUSR1):
        signal.signal(signum, _request_stop)
    if model != "mock":
        os.environ["QWEN_MODEL_7B_PATH"] = str(ROOT / "data/external/deepphy" / MODEL)
    began = time.monotonic()
    code = 0
    for setup in SETUPS:
        run_dir = (output / "fidelity" / kind / setup).resolve()
        if done_tasks(run_dir) >= 500:
            continue
        try:
            run_setup(setup, model, run_dir)
        except Paused:
            log(f"paused in {setup} after {done_tasks(run_dir)} tasks")
            code = EXIT_PAUSED
            break
        log(f"{setup}: {done_tasks(run_dir)} tasks")
    mark(output, "gfid" if model != "mock" else "gfid-mock", time.monotonic() - began, model != "mock")
    return code


def success_curve(results, attempts=10):
    """Cumulative success rate (%) within k attempts, k = 1..attempts, over the tasks of the summaries."""
    solved_at = [r["total_attempts"] if r.get("is_solved") else None for r in results]
    n = len(results)
    return [100.0 * sum(1 for a in solved_at if a is not None and a <= k) / n for k in range(1, attempts + 1)]


def finish(output, kind="runs"):
    output = Path(output)
    band = json.loads((output / "fidelity/band.json").read_text())
    results, per_setup = [], {}
    for setup in SETUPS:
        path = output / "fidelity" / kind / setup / "summary_results.json"
        rows = [r for r in json.loads(path.read_text())[1:] if "seed" in r]
        per_setup[setup] = {"tasks": len(rows), "summary_sha256": sha(path), "curve_pct": success_curve(rows)}
        results += rows
    curve = success_curve(results)
    value = curve[-1]
    complete = len(results) == band["tasks"]
    attempts = [a for r in results for a in r.get("attempt_history", [])]
    diagnostics = {"attempts": len(attempts),
                   "empty_responses": sum(1 for a in attempts if not str(a.get("vlm_response") or "").strip()),
                   "invalid_format": sum(1 for a in attempts if a.get("simulation_status") == "INVALID_ACTION_FORMAT"),
                   "invalid_input": sum(1 for a in attempts if a.get("simulation_status") == "INVALID_INPUT"),
                   "note": "descriptive; an empty response is what the official get_qwen_model_response returns after "
                           "3 failed calls (e.g. CUDA OOM) and would be a diagnosed environment defect under P-fid"}
    record = {"schema": "issue_127_gfid_v1", "issue": 139, "arm": "T7 DeepPHY-VLM", "kind": kind,
              "paper": band["paper"], "band_pct": band["band_pct"], "band_sha256": sha(output / "fidelity/band.json"),
              "tasks": len(results), "complete": complete, "success_within_10_pct": value, "curve_pct": curve,
              "per_setup": per_setup, "diagnostics": diagnostics,
              "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if kind.startswith("runs"):
        record["pass"] = bool(complete and band["band_pct"][0] <= value <= band["band_pct"][1])
        if kind != "runs":  # a logged repair attempt (P-fid); the earlier attempt's record is kept beside it
            record["repair_attempt"] = int(kind.removeprefix("runs-r"))
            record["previous_attempts"] = {p.name: sha(p) for p in sorted((output / "fidelity").glob("gfid-attempt*.json"))}
        target = output / "fidelity/gfid.json"
    else:
        record["label"] = "DESCRIPTIVE environment check (DeepPHY MOCK agent; paper Table 6 MOCK 7.64 %)"
        target = output / "fidelity/mock.json"
    target.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
    log(f"{kind}: {value:.2f} % within 10 attempts over {len(results)} tasks; band {band['band_pct']}")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--run", action="store_true")
    modes.add_argument("--mock", action="store_true")
    modes.add_argument("--finish", action="store_true")
    parser.add_argument("--kind", default="runs", choices=("runs", "mock"), help="--finish: which runs to summarise")
    parser.add_argument("--repair", type=int, choices=(1, 2), default=None,
                        help="a logged P-fid repair attempt: --run / --finish use fidelity/runs-r<N>/")
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args(argv)
    kind = f"runs-r{args.repair}" if args.repair else "runs"
    if args.run:
        return run(args.output, MODEL, kind)
    if args.mock:
        return run(args.output, "mock", "mock")
    return finish(args.output, kind if args.kind == "runs" else args.kind)


if __name__ == "__main__":
    sys.exit(main())
