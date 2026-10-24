"""Issue-130 (C1 DINO-WM) Tier-E G-fid: the official PushT planning evaluation with the released checkpoint.

Modes
    --run      (``b127-dino_wm`` env, one GPU, through the night-window scheduler) runs the official ``plan.py`` of
               gaoyuezhou/dino_wm @ 0a9492f unmodified, with the released ``plan_pusht.yaml`` (50 evaluation cases,
               seed 99, goal_source dset, goal_H 5, MPC-CEM: horizon 5, 300 samples, top-30, 30 CEM steps) on the
               released ``outputs/pusht`` checkpoint; only ``planner.max_iter`` is set (12, frozen in band.json)
               and the output directory is ours. Then writes ``fidelity/gfid.json``.
    --record   re-derives ``fidelity/gfid.json`` from the run directory's ``logs.json`` (CPU).
    --smoke    CPU pipeline check (2 cases, 4 samples, 2 CEM steps, --max-iter MPC iterations); nothing is recorded.
               With --run-dir / --stop-after it exercises the resume path (tests/test_issue_130_dino_wm.py).

Resumable (decision on #130): the MPC loop checkpoints its complete state at every MPC-iteration boundary
(install_mpc_checkpointing); SIGTERM -> exit 75; the next --run resumes the unfinished run directory from its last
completed iteration (the interrupted iteration is recomputed from the restored RNG state).

Infrastructure shims (disclosed in gfid.json): ``import env`` of the official package registers PointMaze, whose module
imports ``mujoco_py`` and ``d4rl`` at import time; PushT never constructs a MuJoCo or D4RL object. Instead of
installing MuJoCo 2.1 + mujoco-py, empty stand-ins for those two modules are placed in ``sys.modules`` before the
official code is imported (forked SubprocVectorEnv workers inherit them). DINOv2 is resolved by the official
``torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")`` from a pre-populated, pinned ``TORCH_HOME``
(facebookresearch/dinov2 @ 7764ea0 and the official ``dinov2_vits14_pretrain.pth``), so nothing is fetched at run
time. wandb runs in ``disabled`` mode (the official DummyWandbRun path).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import types

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / ".local-artifacts/issue-130-dino-wm-v1"
FIDELITY = OUTPUT / "fidelity"
OFFICIAL = Path(os.environ.get("DINO_WM_REF", "/mnt/array/sukaih/novphy-baselines-src/dino_wm@0a9492f"))
EXTERNAL = ROOT / "data/external/dino_wm"
CKPT_BASE = EXTERNAL / "checkpoints"
DATASET_DIR = EXTERNAL / "datasets"
TORCH_HOME = EXTERNAL / "torch_hub"
COMMIT = "0a9492fa12044b852ae9e001cc74604b79c8bb0c"
SCHEMA = "issue_127_gfid_v1"
SMOKE = ["n_evals=2", "planner.sub_planner.num_samples=4", "planner.sub_planner.topk=2",
         "planner.sub_planner.opt_steps=2", "n_plot_samples=1"]


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def utc_now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def install_import_stubs():
    """Empty stand-ins for the two modules the PointMaze registration imports (never used by PushT)."""
    mujoco_py = types.ModuleType("mujoco_py")
    d4rl = types.ModuleType("d4rl")
    offline_env = types.ModuleType("d4rl.offline_env")

    class OfflineEnv:  # base class named by env/pointmaze/maze_model.py; never instantiated for PushT
        pass

    offline_env.OfflineEnv = OfflineEnv
    d4rl.offline_env = offline_env
    sys.modules.setdefault("mujoco_py", mujoco_py)
    sys.modules.setdefault("d4rl", d4rl)
    sys.modules.setdefault("d4rl.offline_env", offline_env)


def install_decode_chunking(chunk=4):
    """Plot-path memory guard: the official evaluator decodes every predicted frame of all 50 cases at once with the
    VQ-VAE decoder (only for its rollout videos; upstream issue #23 reports OOM after a few MPC iterations). Decoding
    in chunks of cases gives the same tensors; planning, the objective and the success metric never use it."""
    import torch
    from models import visual_world_model as vwm
    original = vwm.VWorldModel.decode_obs

    def decode_obs(self, z_obs):
        b = z_obs["visual"].shape[0]
        if b <= chunk:
            return original(self, z_obs)
        parts = [original(self, {k: v[i:i + chunk] for k, v in z_obs.items()}) for i in range(0, b, chunk)]
        obs = {key: torch.cat([p[0][key] for p in parts]) for key in parts[0][0]}
        return obs, sum(p[1] for p in parts) / len(parts)

    vwm.VWorldModel.decode_obs = decode_obs


CHECKPOINT = "mpc_checkpoint.pt"
FINAL = "mpc_final.pt"


def install_mpc_checkpointing(run_dir, stop_after=None):
    """Resumability (scheduling only; decision on #130): ``planning.mpc.MPCPlanner.plan`` is replaced by the
    official loop body verbatim (0a9492f, planning/mpc.py lines 73-139) with two hooks at the MPC-iteration boundary:
    after each iteration (after the official ``dump_logs``) every piece of loop state is saved atomically -- is_success,
    action_len, planned_actions, memo_actions, cur_obs_0, the evaluator's init condition, iter, the logs.json length
    and the torch CPU / CUDA, numpy and python RNG states; at the start, a saved state is restored (and logs.json is
    truncated to the saved length). Everything else the loop reads is rebuilt identically by the official
    PlanWorkspace (seed 99, the same targets; every env rollout re-seeds and re-runs from state_0), so a resumed run
    takes the same path as an uninterrupted one. ``stop_after`` (tests only) exits 75 right after that iteration's
    checkpoint."""
    import random

    import numpy as np
    import torch
    from planning import mpc
    from utils import slice_trajdict_with_t

    run_dir = Path(run_dir)
    path = run_dir / CHECKPOINT

    def save(planner, cur_obs_0, memo_actions):
        logs = run_dir / "logs.json"
        state = {"is_success": planner.is_success, "action_len": planner.action_len,
                 "planned_actions": [a.cpu() for a in planner.planned_actions],
                 "memo_actions": None if memo_actions is None else memo_actions.cpu(), "cur_obs_0": cur_obs_0,
                 "evaluator_obs_0": planner.evaluator.obs_0, "evaluator_state_0": planner.evaluator.state_0,
                 "iter": planner.iter, "logs_bytes": logs.stat().st_size if logs.is_file() else 0,
                 "torch_rng": torch.get_rng_state(),
                 "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                 "numpy_rng": np.random.get_state(), "python_rng": random.getstate(), "saved_at": utc_now()}
        tmp = path.with_suffix(".tmp")
        torch.save(state, tmp)
        os.replace(tmp, path)

    def plan(self, obs_0, obs_g, actions=None):
        n_evals = obs_0["visual"].shape[0]
        self.is_success = np.zeros(n_evals, dtype=bool)
        self.action_len = np.full(n_evals, np.inf)
        init_obs_0, init_state_0 = self.evaluator.get_init_cond()

        cur_obs_0 = obs_0
        memo_actions = None
        if path.is_file():  # resume hook
            state = torch.load(path, map_location="cpu", weights_only=False)
            device = next(self.wm.parameters()).device
            self.is_success, self.action_len, self.iter = state["is_success"], state["action_len"], state["iter"]
            self.planned_actions = [a.to(device) for a in state["planned_actions"]]
            memo_actions = None if state["memo_actions"] is None else state["memo_actions"].to(device)
            cur_obs_0 = state["cur_obs_0"]
            self.evaluator.assign_init_cond(obs_0=state["evaluator_obs_0"], state_0=state["evaluator_state_0"])
            with open(run_dir / "logs.json", "r+") as f:
                f.truncate(state["logs_bytes"])
            torch.set_rng_state(state["torch_rng"])
            if state["cuda_rng"] is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(state["cuda_rng"])
            np.random.set_state(state["numpy_rng"])
            random.setstate(state["python_rng"])
            print(f"[issue-130] resumed the MPC loop after iteration {self.iter}", flush=True)
        while not np.all(self.is_success) and self.iter < self.max_iter:
            self.sub_planner.logging_prefix = f"plan_{self.iter}"
            actions, _ = self.sub_planner.plan(
                obs_0=cur_obs_0,
                obs_g=obs_g,
                actions=memo_actions,
            )  # (b, t, act_dim)
            taken_actions = actions.detach()[:, : self.n_taken_actions]
            self._apply_success_mask(taken_actions)
            memo_actions = actions.detach()[:, self.n_taken_actions :]
            self.planned_actions.append(taken_actions)

            print(f"MPC iter {self.iter} Eval ------- ")
            action_so_far = torch.cat(self.planned_actions, dim=1)
            self.evaluator.assign_init_cond(
                obs_0=init_obs_0,
                state_0=init_state_0,
            )
            logs, successes, e_obses, e_states = self.evaluator.eval_actions(
                action_so_far,
                self.action_len,
                filename=f"plan{self.iter}",
                save_video=True,
            )
            new_successes = successes & ~self.is_success  # Identify new successes
            self.is_success = (
                self.is_success | successes
            )  # Update overall success status
            self.action_len[new_successes] = (
                (self.iter + 1) * self.n_taken_actions
            )  # Update only for the newly successful trajectories

            print("self.is_success: ", self.is_success)
            logs = {f"{self.logging_prefix}/{k}": v for k, v in logs.items()}
            logs.update({"step": self.iter + 1})
            self.wandb_run.log(logs)
            self.dump_logs(logs)

            # update evaluator's init conditions with new env feedback
            e_final_obs = slice_trajdict_with_t(e_obses, start_idx=-1)
            cur_obs_0 = e_final_obs
            e_final_state = e_states[:, -1]
            self.evaluator.assign_init_cond(
                obs_0=e_final_obs,
                state_0=e_final_state,
            )
            self.iter += 1
            self.sub_planner.logging_prefix = f"plan_{self.iter}"
            save(self, cur_obs_0, memo_actions)  # checkpoint hook
            if stop_after is not None and self.iter == stop_after:
                os._exit(75)

        planned_actions = torch.cat(self.planned_actions, dim=1)
        self.evaluator.assign_init_cond(
            obs_0=init_obs_0,
            state_0=init_state_0,
        )
        torch.save({"planned_actions": planned_actions.cpu(), "action_len": self.action_len,
                    "is_success": self.is_success, "iterations": self.iter}, run_dir / FINAL)
        return planned_actions, self.action_len

    mpc.MPCPlanner.plan = plan


def overrides(band, run_dir, smoke=False, max_iter=None):
    out = ["--config-name", "plan_pusht.yaml", "model_name=pusht", f"ckpt_base_path={CKPT_BASE}",
           f"planner.max_iter={band['protocol']['max_iter'] if max_iter is None else max_iter}",
           f"hydra.run.dir={run_dir}"]
    return out + (SMOKE if smoke else [])


def run_official(argv, run_dir, stop_after=None):
    """Run the official plan.py exactly as `python plan.py ARGV` (as __main__; hydra parses argv), with the MPC loop's
    iteration-boundary checkpoint (install_mpc_checkpointing)."""
    os.environ["DATASET_DIR"] = str(DATASET_DIR)
    os.environ["TORCH_HOME"] = str(TORCH_HOME)
    os.environ["WANDB_MODE"] = "disabled"
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"  # torch >= 2.6 default; the release pickles whole modules
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # allocator only (P-crash on #130)
    install_import_stubs()
    sys.path.insert(0, str(OFFICIAL))
    os.chdir(OFFICIAL)
    install_decode_chunking()
    install_mpc_checkpointing(run_dir, stop_after)
    sys.argv = [str(OFFICIAL / "plan.py"), *argv]
    import runpy
    runpy.run_path(str(OFFICIAL / "plan.py"), run_name="__main__")


def run_dir_for_run():
    """The unfinished run to resume (it has an MPC checkpoint and no final record), else a new tier-e-N."""
    base = FIDELITY / "runs"
    base.mkdir(parents=True, exist_ok=True)
    runs = sorted(base.glob("tier-e-*"), key=lambda p: int(p.name.split("-")[-1]))
    if runs and (runs[-1] / CHECKPOINT).is_file() and not (runs[-1] / FINAL).is_file():
        return runs[-1]
    n = 1 + max([int(p.name.split("-")[-1]) for p in runs] or [0])
    return base / f"tier-e-{n}"


def add_wall(run_dir, seconds):
    path = Path(run_dir) / "wall.json"
    block = read_json(path) if path.is_file() else {"seconds": 0.0, "launches": 0}
    block["seconds"] += seconds
    block["launches"] += 1
    write_json(path, block)
    return block


def parse_logs(run_dir):
    """MPC iteration curve (cumulative success, sticky per case) and the official final evaluation."""
    entries = [json.loads(line) for line in (Path(run_dir) / "logs.json").read_text().splitlines() if line.strip()]
    curve = [{"mpc_iteration": int(e["step"]), "success_rate": float(e["mpc/success_rate"])}
             for e in entries if "mpc/success_rate" in e]
    final = [e for e in entries if "final_eval/success_rate" in e]
    return curve, (float(final[-1]["final_eval/success_rate"]) if final else None), final[-1] if final else None


def record(run_dir, wall_seconds=None, exit_code=None):
    import platform
    band_path = FIDELITY / "band.json"
    band = read_json(band_path)
    curve, success, final = parse_logs(run_dir)
    n = band["protocol"]["n_evals"]
    lo, hi = band["band"]
    previous = read_json(FIDELITY / "gfid.json") if (FIDELITY / "gfid.json").is_file() else {}
    runs = [r for r in previous.get("runs", []) if r["run_dir"] != str(Path(run_dir).relative_to(ROOT))]
    entry = {"run_dir": str(Path(run_dir).relative_to(ROOT)), "recorded_at": utc_now(),
             "success_rate": success, "successes": None if success is None else round(success * n),
             "n_evals": n, "mpc_curve": curve, "mpc_iterations_run": len(curve),
             "all_cases_succeeded_before_cap": bool(curve and curve[-1]["success_rate"] == 1.0
                                                    and len(curve) < band["protocol"]["max_iter"]),
             "final_eval": final, "wall_seconds": wall_seconds, "exit_code": exit_code,
             "in_band": None if success is None else bool(lo <= success <= hi)}
    runs.append(entry)
    import torch
    result = {"schema": SCHEMA, "issue": 130, "arm_family": "dino_wm", "tier": "E",
              "paper": band["paper"], "band": band["band"], "band_rule": band["band_rule"],
              "band_sha256": "sha256:" + sha256_of(band_path), "protocol": band["protocol"],
              "official": {"repo": "https://github.com/gaoyuezhou/dino_wm", "commit": COMMIT,
                           "command": "python plan.py " + " ".join(overrides(band, "<run_dir>"))},
              "checkpoint": {"path": str((CKPT_BASE / "outputs/pusht/checkpoints/model_latest.pth").relative_to(ROOT)),
                             "sha256": "sha256:" + sha256_of(CKPT_BASE / "outputs/pusht/checkpoints/model_latest.pth"),
                             "hydra_yaml_sha256": "sha256:" + sha256_of(CKPT_BASE / "outputs/pusht/hydra.yaml")},
              "shims": ["sys.modules stand-ins for mujoco_py and d4rl.offline_env (import-only, PointMaze registration)",
                        "TORCH_HOME pinned to data/external/dino_wm/torch_hub (dinov2 @ 7764ea0 + official weights)",
                        "WANDB_MODE=disabled", "SDL_VIDEODRIVER=dummy",
                        "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (allocator only)",
                        "VWorldModel.decode_obs chunked over 4 cases (plot-only decoder path; identical tensors; OOM guard, upstream #23)",
                        "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 (torch 2.7 default weights_only=True; the released checkpoint pickles whole modules, as torch 2.3 loaded it)"],
              "environment": {"conda_env": "b127-dino_wm", "python": platform.python_version(),
                              "torch": torch.__version__,
                              "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"},
              "runs": runs}
    scored = [r for r in runs if r["success_rate"] is not None]
    result["pass"] = bool(scored) and bool(scored[-1]["in_band"])
    result["decided_by_run"] = scored[-1]["run_dir"] if scored else None
    write_json(FIDELITY / "gfid.json", result)
    print(f"[issue-130] G-fid Tier E: success {success} band {band['band']} -> pass {result['pass']}", flush=True)
    return result


def ledger(phase, seconds):
    path = OUTPUT / "ledger" / f"{phase}.json"
    block = read_json(path) if path.is_file() else {"wall_seconds": 0.0, "gpu_seconds": 0.0}
    block["wall_seconds"] += seconds
    block["gpu_seconds"] += seconds
    block["updated_at"] = utc_now()
    write_json(path, block)


def launch(argv, run_dir, stop_after=None):
    """Run the official evaluation in a forked child (its exit path is the official one); SIGTERM / SIGUSR1 are
    forwarded and turn into exit 75 (resume from the last MPC-iteration checkpoint)."""
    import signal
    stopped = []
    pid = os.fork()
    if pid == 0:
        try:
            run_official(argv, run_dir, stop_after)
            os._exit(0)
        except BaseException as exc:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            print(f"[issue-130] official plan.py failed: {type(exc).__name__}: {exc}", flush=True)
            os._exit(0 if isinstance(exc, SystemExit) and not exc.code else 1)

    def forward(signum, _frame):
        stopped.append(signum)
        os.kill(pid, signal.SIGTERM)

    for signum in (signal.SIGTERM, signal.SIGUSR1):
        signal.signal(signum, forward)
    while True:
        try:
            _, status = os.waitpid(pid, 0)
            break
        except InterruptedError:
            continue
    code = os.waitstatus_to_exitcode(status)
    return 75 if stopped and not (Path(run_dir) / FINAL).is_file() else code


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--record", metavar="RUN_DIR")
    mode.add_argument("--smoke", action="store_true")
    parser.add_argument("--run-dir", default=None, help="--smoke only: the run directory (resumed if it has a checkpoint)")
    parser.add_argument("--max-iter", type=int, default=1, help="--smoke only")
    parser.add_argument("--stop-after", type=int, default=None, help="--smoke only: exit 75 after this MPC iteration")
    args = parser.parse_args(argv)
    band_path = FIDELITY / "band.json"
    if not band_path.is_file():
        raise SystemExit("fidelity/band.json missing: the band is frozen and posted before any run")
    band = read_json(band_path)
    if args.record:
        run_dir = Path(args.record).resolve()
        wall = read_json(run_dir / "wall.json")["seconds"] if (run_dir / "wall.json").is_file() else None
        record(run_dir, wall)
        return 0
    if args.smoke:
        run_dir = Path(args.run_dir or Path("/tmp/issue-130-gfid-smoke") / time.strftime("%Y%m%d-%H%M%S")).resolve()
        code = launch(overrides(band, run_dir, smoke=True, max_iter=args.max_iter), run_dir, args.stop_after)
        if code == 0:
            curve, success, _ = parse_logs(run_dir)
            print(f"[issue-130] smoke ok: curve {curve} final {success}", flush=True)
        return code
    run_dir = run_dir_for_run()
    began = time.monotonic()
    code = launch(overrides(band, run_dir), run_dir)
    wall = time.monotonic() - began
    ledger("fidelity-tier-e", wall)
    total = add_wall(run_dir, wall)
    if code == 0 and (run_dir / FINAL).is_file():
        record(run_dir, total["seconds"], code)
    return code


if __name__ == "__main__":
    sys.exit(main())
