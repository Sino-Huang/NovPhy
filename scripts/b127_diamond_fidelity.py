"""#137 G-fid (Decision P-fid on #137): Tier E, the released DIAMOND Atari 100k agents (Hugging Face
eloialonso/diamond @ 5d4abca, ``data/external/diamond/``) under the paper's evaluation protocol, against the band
frozen in ``.local-artifacts/issue-137-diamond-v1/fidelity/band.json`` before any run. Env ``b127-diamond``; the
official code runs from the vendored ``vendor/src`` (eloialonso/diamond ``main`` @ 5bcd159).

Protocol = ``Trainer.collect_test(final=True)``: the official ``make_atari_env`` test env (``config/env/atari.yaml``
``test``: done_on_life_loss False, 64 x 64, no step cap), the official collector / env loop with epsilon 0.0 (actions
sampled from the actor-critic), 100 episodes, mean return. The agent is built and loaded exactly as ``play.py
--pretrained`` (trainer config composed with Hydra, the checkpoint's ``config/agent`` / ``config/env`` overrides,
``Agent(instantiate(cfg.agent, num_actions=...))``, ``agent.load``).

Resumable per episode: episode i of a game is one ``collector.send(NumToCollect(episodes=1))`` (the official
``reset_every_collect`` collector re-creates its env loop for every collect) after seeding ``random`` / numpy / torch
with ``seed + 1000 * game index + i``, so an episode interrupted by SIGTERM is replayed identically on resume; the
runner checks the stop flag between episodes and exits 75. Disclosure: the official final collect draws 100 episodes in
one call (episodes after the first follow the vector env's autoreset); here every episode starts from its own seeded
reset (same per-episode distribution: up to 30 no-op starts from the env's seeded RNG).

Modes: ``--run [--games G ...]`` (CPU by default) -> ``fidelity/run/<game>.json``; ``--finish`` -> ``fidelity/gfid.json``
(schema ``issue_127_gfid_v1``).
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "world_model" / "baselines" / "diamond" / "vendor"
OUTPUT = ROOT / ".local-artifacts" / "issue-137-diamond-v1"
HF = ROOT / "data" / "external" / "diamond" / "hf-5d4abca" / "atari_100k"
EXIT_PAUSED = 75
STOP = []


def sha256(path) -> str:
    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_json(path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    os.replace(tmp, path)


def log(message) -> None:
    print(f"[issue-137 gfid] {message}", flush=True)


def load_agent_and_env(game: str, device):
    """The official objects of ``play.py --pretrained`` for ``game`` (test env, agent with the released weights)."""
    sys.path.insert(0, str(VENDOR / "src"))
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)
    from agent import Agent
    from envs import make_atari_env
    with initialize_config_dir(config_dir=str(VENDOR / "config"), version_base="1.3"):
        cfg = compose(config_name="trainer")
    cfg.agent = OmegaConf.load(HF / "config/agent/default.yaml")
    cfg.env = OmegaConf.load(HF / "config/env/atari.yaml")
    cfg.env.train.id = cfg.env.test.id = f"{game}NoFrameskip-v4"
    test_env = make_atari_env(num_envs=1, device=device, **cfg.env.test)
    agent = Agent(instantiate(cfg.agent, num_actions=test_env.num_actions)).to(device).eval()
    agent.load(HF / "models" / f"{game}.pt")
    return agent, test_env, cfg


def run_game(output: Path, band: dict, game: str, gi: int, device) -> int:
    path = output / "fidelity" / "run" / f"{game}.json"
    spec = band["games"][game]
    ckpt = ROOT / spec["checkpoint"]
    if sha256(ckpt) != spec["checkpoint_sha256"]:
        raise SystemExit(f"{game}: checkpoint sha256 differs from band.json")
    state = json.loads(path.read_text()) if path.is_file() else {
        "game": game, "env_id": spec["env_id"], "checkpoint": spec["checkpoint"],
        "checkpoint_sha256": spec["checkpoint_sha256"], "device": str(device), "torch": torch.__version__,
        "episodes": [], "started_at": utc_now()}
    if len(state["episodes"]) >= band["episodes"]:
        return 0
    agent, env, _ = load_agent_and_env(game, device)
    from coroutines.collector import make_collector, NumToCollect
    from data import Dataset
    dataset = Dataset(output / "fidelity" / "run" / f"{game}-dataset", "test_dataset", cache_in_ram=True,
                      save_on_disk=False)
    collector = make_collector(env, agent.actor_critic, dataset, epsilon=0.0, reset_every_collect=True, verbose=False)
    with torch.no_grad():
        while len(state["episodes"]) < band["episodes"]:
            if STOP:
                write_json(path, state)
                log(f"{game}: paused after {len(state['episodes'])} episodes")
                return EXIT_PAUSED
            i = len(state["episodes"])
            seed = band["seed"] + 1000 * gi + i
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            began = time.monotonic()
            to_log = collector.send(NumToCollect(episodes=1))
            episodes = [x for x in to_log if "return" in x and "length" in x]
            if len(episodes) != 1:
                raise RuntimeError(f"{game}: expected one finished episode, got {episodes}")
            dataset.clear()
            state["episodes"].append({"index": i, "seed": seed, "return": float(episodes[0]["return"]),
                                      "length": int(episodes[0]["length"]),
                                      "seconds": time.monotonic() - began})
            write_json(path, state)
            if (i + 1) % 10 == 0:
                log(f"{game}: {i + 1}/{band['episodes']} episodes")
    env.close()
    state["finished_at"] = utc_now()
    write_json(path, state)
    return 0


def run(output: Path, games, device) -> int:
    band = json.loads((output / "fidelity" / "band.json").read_text())
    for signum in (signal.SIGTERM, signal.SIGUSR1):
        signal.signal(signum, lambda *_: STOP.append(1))
    order = sorted(band["games"])
    for game in games or order:
        code = run_game(output, band, game, order.index(game), device)
        if code != 0:
            return code
    return 0


def finish(output: Path) -> int:
    band_path = output / "fidelity" / "band.json"
    band = json.loads(band_path.read_text())
    games, ok = {}, True
    for game, spec in sorted(band["games"].items()):
        path = output / "fidelity" / "run" / f"{game}.json"
        state = json.loads(path.read_text())
        returns = [e["return"] for e in state["episodes"]]
        if len(returns) != band["episodes"]:
            raise SystemExit(f"{game}: {len(returns)} of {band['episodes']} episodes")
        mean = float(np.mean(returns))
        inside = bool(spec["band"][0] <= mean <= spec["band"][1])
        ok = ok and inside
        games[game] = {"ours_mean_return": mean, "ours_sd_episodes": float(np.std(returns, ddof=1)),
                       "ours_se": float(np.std(returns, ddof=1) / np.sqrt(len(returns))),
                       "episodes": len(returns), "mean_length": float(np.mean([e["length"] for e in state["episodes"]])),
                       "paper_mean": spec["paper_mean"], "paper_table1": spec["paper_table1"],
                       "paper_seed_returns": spec["paper_seed_returns"], "band": spec["band"], "pass": inside,
                       "run": str(path.relative_to(ROOT)), "run_sha256": sha256(path),
                       "seconds": float(sum(e["seconds"] for e in state["episodes"]))}
    record = {"schema": "issue_127_gfid_v1", "issue": 137, "arm": "T5 DIAMOND-px", "tier": band["tier"],
              "target": band["target"], "protocol": band["protocol"], "rule": band["rule"],
              "band": str(band_path.relative_to(ROOT)), "band_sha256": sha256(band_path), "games": games,
              "repair_attempts": 0, "pass": ok, "written_at": utc_now()}
    write_json(output / "fidelity" / "gfid.json", record)
    log(f"G-fid pass {ok}: " + ", ".join(f"{g} {v['ours_mean_return']:.2f} in {[round(x, 2) for x in v['band']]}: "
                                         f"{v['pass']}" for g, v in games.items()))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--finish", action="store_true")
    parser.add_argument("--games", nargs="*", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args(argv)
    torch.set_num_threads(int(os.environ.get("B137_CPU_THREADS", "4")))
    output = Path(args.output)
    if args.run:
        return run(output, args.games, torch.device(args.device))
    return finish(output)


if __name__ == "__main__":
    raise SystemExit(main())
