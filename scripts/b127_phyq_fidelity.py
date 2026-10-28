"""#128 A3 native-task fidelity: Phy-Q's released within-template training run (template 3_02_01) in Phy-Q's own game.

Decision P-engine 2026-10-25 (design, band, cost; issuecomment-6388391126) and 2026-10-28 (funding,
issuecomment-6464530188). Band: ``a3/fidelity-band.json`` (sha256 84b17d09...077f), frozen before any run.

What runs (released code, phy-q/benchmark @ 514491c, vendored verbatim under world_model/baselines/phyq/vendor):
``TrainLearningAgent.py --template 3_02_01 --mode within_template --level_path fifth_generation --game_version Linux``
(the ``TrainLearningAgent.sh within_template`` command line) executed unchanged through ``runpy`` in conda env
b127-phyq, against the benchmark's own game build (``sciencebirdsgames/Linux.zip``, its ``game_playing_interface.jar``)
and its ``fifth_generation`` levels: 100 tasks, train = 1..80, test = 81..100; 16 update steps (step 0: every training
task 10 x; steps 1-15: one task per worker); AdamW 3e-4, batch 32; greedy test pass on the 20 test tasks. Metric = the
script's own "percent of testing tasks solved" (``level_winning_rate`` over the 20 test tasks).

Settings the README tells users to edit in ``Utils/Parameters.py`` are applied by wrapping ``Parameters.__init__``
(no file edit): ``agent = DQNDiscreteAgent``, ``network = DQNRelationalSymbolic`` / ``DQNRelationalImage``,
``state_repr_type = "symbolic"`` / ``"image"``; plus ``device = "cpu"`` (engine runs are CPU only).

Harness (process plumbing only; every learning, sampling, environment-step and evaluation line is the released code):
- one game server per agent: each of the 10 official workers talks to its own jar (``--dev``, plus ``--headless`` for
  the symbolic agent exactly as ``SBEnvironmentWrapper.__start_server`` starts it) in its own hard-linked copy of the
  game, on private ports; ``SBEnvironmentWrapper.make`` starts the agent's jar instead of the official
  ``gnome-terminal``/``nohup`` launch, ``close`` no longer runs the official host-wide ``ps | kill -9``;
- at most ``--concurrency`` (default 5, <= 8) game instances at a time: ``MultiThreadTrajCollection`` runs the 10 official
  agents in balanced waves (all agents of a collection step use the same frozen network and only append to the replay
  data, so a wave split leaves the data distribution unchanged);
- the Unity player is launched through a renaming wrapper (as scripts/9001-player-wrapper.sh) with
  ``-force-glcore -screen-fullscreen 0 -screen-width 640 -screen-height 480`` (Phy-Q's 640 x 480 frame), Mesa llvmpipe,
  a private Xvnc display;
- instrumentation: one event per environment step, jar start/stop, the first raw ground-truth JSON (dev labels check).

Modes (supervisor: novphy or any Python 3.11; child: b127-phyq):
  --prepare                  build the shared player copy and check inputs (no engine)
  --run sym[,img]            daytime engine window only (07:05-17:30 Australia/Melbourne; waits for it); A3 smoke gate on
                             the first 50 episodes; Relational-Image only after its funding decision is posted
  --record                   rewrite fidelity/a3/gfid.json from the attempt records
  --child VARIANT CONFIG     internal (b127-phyq python, cwd = the attempt's sciencebirdsagents directory)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHILD_OUT = ROOT / ".local-artifacts/issue-128-direct-predictors-v1"
OUT = CHILD_OUT / "fidelity/a3"
BAND = CHILD_OUT / "a3/fidelity-band.json"
BAND_SHA256 = "84b17d0914791dcde3ae54899b0373064bd564c77c7de73f7c53a90eaa99077f"
LEDGER = CHILD_OUT / "ledger/engine-a3-fidelity.json"
DECISIONS = CHILD_OUT / "decisions.md"
PHYQ_SRC = Path("/mnt/array/sukaih/novphy-baselines-src/benchmark@514491c")
GAME_ZIP = PHYQ_SRC / "sciencebirdsgames/Linux.zip"
LEVELS_ZIP = PHYQ_SRC / "sciencebirdslevels/generated_levels/fifth_generation.zip"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
GAME_ZIP_SHA256 = "d556c0e8cd82f25a908e52a25c310e219009529edf081cdf1e8d63c0f63f8e70"
JAR_SHA256 = "b8ebda1806e8e440eda2e810c7c2892bf4a7251252bb79f9fb827a1c7af81e7a"
SERVERBACKUP = ROOT / ".local-artifacts/issue-104-power-cohort-v1/player/serverbackup"  # as the E1 stock player
VENDOR = ROOT / "world_model/baselines/phyq/vendor"
PHYQ_PYTHON = Path.home() / "miniconda3/envs/b127-phyq/bin/python"
MESA = Path.home() / ".cache/novphy-mesa/mesa-26.1.2-llvm22.1.6"
TEMPLATE = "3_02_01"
VARIANTS = {
    "sym": {"name": "Relational-Symbolic", "network": "DQNRelationalSymbolic", "state": "symbolic",
            "jar_flags": ["--headless", "--dev"], "port_base": 39600},
    "img": {"name": "Relational-Image", "network": "DQNRelationalImage", "state": "image",
            "jar_flags": ["--dev"], "port_base": 39700},
}
ORDER = ("sym", "img")  # fixed in the 2026-10-25 P-engine decision
OFFICIAL_ARGV = ["TrainLearningAgent.py", "--template", TEMPLATE, "--mode", "within_template",
                 "--level_path", "fifth_generation", "--game_version", "Linux"]
# released code, level_path != 'MattewLevel': 16 update steps; step 0 = 80 tasks x 10; steps 1-15 = 10 workers x 1;
# test = 20 tasks x 1 (a3/harness-check.json, 'code' variant); one game session per worker per collection call
EPISODES = 800 + 15 * 10 + 20
SESSIONS = 10 * 16 + 10
ALLOWANCE_S = 36.343 * 3600  # R = 190 - E1's realised 153.657 worker-h (Decision P-engine 2026-10-28)
MARGIN = 1.25
SMOKE_EPISODES = 50
STALL_S = 20 * 60
ENGINE_WINDOW = ((7, 5), (17, 30))
ENGINE_TZ = "Australia/Melbourne"
SCHEMA_ATTEMPT = "issue_128_a3_fidelity_attempt_v1"
SCHEMA_GFID = "issue_127_gfid_v1"
EXIT_HARNESS = 4


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def log(message):
    print(f"[{utc_now()}] b127_phyq_fidelity: {message}", flush=True)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=1, sort_keys=True) + "\n")
    os.replace(tmp, path)


def listening(port):
    target = f":{port:04X}"
    for table in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            rows = Path(table).read_text().splitlines()[1:]
        except OSError:
            continue
        if any(r.split()[1].endswith(target) and r.split()[3] == "0A" for r in rows):
            return True
    return False


def kill_group(pid):
    try:
        os.killpg(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


# ---------------------------------------------------------------------------
# preparation (no engine)
# ---------------------------------------------------------------------------

WRAPPER = """#!/usr/bin/env bash
# A3 fidelity launcher (the renaming scheme of scripts/9001-player-wrapper.sh): one game per directory
set -euo pipefail
root="$(dirname "$0")"
self="$root/9001.x86_64"
binary="$root/9001-player.x86_64"
wrapper="$root/.9001-wrapper"
mv "$self" "$wrapper"
mv "$binary" "$self"
restore() {
  mv "$self" "$binary"
  mv "$wrapper" "$self"
}
trap restore EXIT
"$self" -force-glcore -screen-fullscreen 0 -screen-width 640 -screen-height 480 "$@"
"""


def player_dir():
    return OUT / "player"


def prepare():
    """Benchmark game build (bundled level library left out; the official Config installs the template's levels),
    the launcher, and the template's 100 level XMLs; input hashes in player/player.json."""
    root = player_dir()
    if (root / "player.json").is_file():
        return read_json(root / "player.json")
    if sha256_file(GAME_ZIP) != GAME_ZIP_SHA256:
        raise ValueError("benchmark Linux.zip differs")
    if sha256_file(BAND) != BAND_SHA256:
        raise ValueError("frozen band file differs")
    tmp = root.with_name("player.tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    game = tmp / "Linux"
    with zipfile.ZipFile(GAME_ZIP) as archive:
        for info in archive.infolist():
            name = info.filename
            if name == "Linux/" or ("/StreamingAssets/Levels/" in name and not name.endswith("/Levels/")):
                continue
            target = tmp / name
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, open(target, "wb") as sink:
                shutil.copyfileobj(source, sink)
    (game / "9001_Data/StreamingAssets/Levels/novelty_level_1/type1/Levels").mkdir(parents=True, exist_ok=True)
    if sha256_file(game / "game_playing_interface.jar") != JAR_SHA256:
        raise ValueError("benchmark jar differs")
    (game / "9001.x86_64").rename(game / "9001-player.x86_64")
    (game / "9001.x86_64").write_text(WRAPPER)
    for name in ("9001.x86_64", "9001-player.x86_64"):
        os.chmod(game / name, 0o755)
    shutil.copy2(SERVERBACKUP, game / "serverbackup")
    levels = tmp / "levels" / "fifth_generation/3/02/01"
    levels.mkdir(parents=True)
    names = []
    with zipfile.ZipFile(LEVELS_ZIP) as archive:
        for info in archive.infolist():
            if info.filename.startswith("fifth_generation/3/02/01/") and info.filename.endswith(".xml"):
                (levels / Path(info.filename).name).write_bytes(archive.read(info))
                names.append(Path(info.filename).name)
    if len(names) != 100:
        raise ValueError(f"template {TEMPLATE} has {len(names)} level files, expected 100")
    meta = {"game_zip": str(GAME_ZIP), "game_zip_sha256": GAME_ZIP_SHA256, "jar_sha256": JAR_SHA256,
            "levels_zip": str(LEVELS_ZIP), "levels_zip_sha256": sha256_file(LEVELS_ZIP), "template": TEMPLATE,
            "level_files": sorted(names), "launcher_sha256": hashlib.sha256(WRAPPER.encode()).hexdigest(),
            "serverbackup_sha256": sha256_file(SERVERBACKUP), "band_sha256": BAND_SHA256,
            "excluded": "9001_Data/StreamingAssets/Levels/* (bundled level library; Config installs the template)",
            "built_at": utc_now()}
    write_json(tmp / "player.json", meta)
    tmp.rename(root)
    return meta


def build_workspace(attempt_dir):
    """The relative layout TrainLearningAgent.py / Utils/Config.py expect, with cwd = sciencebirdsagents:
    ../sciencebirdsgames/Linux (game, config.xml), ../sciencebirdslevels/generated_levels/fifth_generation/..."""
    from scripts.issue_76_shared_player_storage import clone_player
    agents = attempt_dir / "sciencebirdsagents"
    (agents / "Utils").mkdir(parents=True)
    (agents / "LearningAgents").mkdir()
    for name in ("model", "target_class", "server_client_config.json"):
        shutil.copy2(VENDOR / "Utils" / name, agents / "Utils" / name)
    games = attempt_dir / "sciencebirdsgames"
    games.mkdir()
    clone_player(player_dir() / "Linux", games / "Linux")
    shutil.copytree(player_dir() / "levels", attempt_dir / "sciencebirdslevels/generated_levels")
    return agents


# ---------------------------------------------------------------------------
# child (b127-phyq): the released TrainLearningAgent.py with harness hooks
# ---------------------------------------------------------------------------

class Events:
    def __init__(self, path):
        self.handle = open(path, "a", buffering=1)
        self.lock = threading.Lock()

    def emit(self, kind, **fields):
        with self.lock:
            self.handle.write(json.dumps({"t": time.time(), "kind": kind, **fields}) + "\n")


class Jars:
    """One game server per slot (its own copy of the game, private ports), started on demand, killed per wave."""

    def __init__(self, cfg, events, games_dir):
        self.cfg, self.events, self.games_dir = cfg, events, Path(games_dir)
        self.procs = {}
        self.lock = threading.Lock()

    def ports(self, slot):
        base = self.cfg["port_base"] + 10 * slot
        return base, base + 1

    def slot_dir(self, slot):
        from scripts.issue_76_shared_player_storage import clone_player
        path = self.games_dir / f"slot-{slot}"
        if not path.is_dir():  # cloned after the official Config wrote config.xml and the levels into Linux/
            clone_player(self.games_dir / "Linux", path)
        return path

    def start(self, slot):
        with self.lock:
            if slot in self.procs and self.procs[slot].poll() is None:
                return
            agent_port, game_port = self.ports(slot)
            for port in (agent_port, game_port):
                if listening(port):
                    raise RuntimeError(f"port {port} already in use")
            game = self.slot_dir(slot)
            env = dict(os.environ)
            env.pop("XDG_RUNTIME_DIR", None)
            env.update(DISPLAY=self.cfg["display"], XDG_DATA_HOME=str(game / ".xdg"),
                       XDG_CONFIG_HOME=str(game / ".config"), PULSE_SERVER="unix:/nonexistent",
                       CUDA_VISIBLE_DEVICES="", LD_LIBRARY_PATH=str(MESA / "lib"),
                       LIBGL_DRIVERS_PATH=str(MESA / "lib/dri"), __GLX_VENDOR_LIBRARY_NAME="mesa")
            log_file = open(game / f"jar-{int(time.time())}.log", "wb")
            proc = subprocess.Popen(["java", "-jar", "./game_playing_interface.jar", *self.cfg["jar_flags"],
                                     "--agent-port", str(agent_port), "--game-start-port", str(game_port)],
                                    cwd=game, env=env, stdout=log_file, stderr=subprocess.STDOUT,
                                    stdin=subprocess.DEVNULL, start_new_session=True)
            self.procs[slot] = proc
            self.events.emit("jar_start", slot=slot, pid=proc.pid, agent_port=agent_port)
            deadline = time.monotonic() + 60
            while not listening(agent_port):
                if proc.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError(f"slot {slot}: jar did not open its agent port")
                time.sleep(0.2)

    def stop_all(self):
        with self.lock:
            for slot, proc in sorted(self.procs.items()):
                kill_group(proc.pid)
                try:
                    proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    pass
                self.events.emit("jar_stop", slot=slot, pid=proc.pid)
            self.procs = {}


def balanced_waves(items, size):
    count = max(1, math.ceil(len(items) / size))
    per, extra = divmod(len(items), count)
    out, start = [], 0
    for i in range(count):
        stop = start + per + (1 if i < extra else 0)
        out.append(items[start:stop])
        start = stop
    return out


def child(variant, cfg_path):
    import runpy
    import numpy as np
    import torch
    cfg = read_json(cfg_path)
    spec = VARIANTS[variant]
    events = Events(cfg["events"])
    np.float = float  # removed in NumPy 1.24; the vendored reader uses it (VENDOR.json compatibility)
    torch.set_num_threads(cfg["torch_threads"])
    sys.path.insert(0, str(ROOT))
    jars = Jars(cfg, events, Path.cwd().parent / "sciencebirdsgames")
    state = {"agents": None, "calls": 0, "gt_saved": False, "shape_saved": False}

    from world_model.baselines.phyq.vendor.Utils import Parameters as parameters
    from world_model.baselines.phyq.vendor.LearningAgents.DQNDiscreteAgent import DQNDiscreteAgent
    from world_model.baselines.phyq.vendor.LearningAgents.RLNetwork import DQNRelationalImage as image_net
    from world_model.baselines.phyq.vendor.LearningAgents.RLNetwork import DQNRelationalSymbolic as symbolic_net
    from world_model.baselines.phyq.vendor.SBEnvironment import SBEnvironmentWrapper as wrapper
    from world_model.baselines.phyq.vendor.LearningAgents import LearningAgentThread as threads
    from world_model.baselines.phyq.vendor.Client import agent_client
    network = {"DQNRelationalSymbolic": symbolic_net.DQNRelationalSymbolic,
               "DQNRelationalImage": image_net.DQNRelationalImage}[spec["network"]]

    original_init = parameters.Parameters.__init__

    def parameters_init(self, *args, **kwargs):  # the README's Parameters.py settings, plus CPU
        original_init(self, *args, **kwargs)
        self.param.update(agent=DQNDiscreteAgent, network=network, state_repr_type=spec["state"], device="cpu")
        events.emit("parameters", **{k: (v.__name__ if hasattr(v, "__name__") else v)
                                     for k, v in self.param.items() if not isinstance(v, (list, dict))})
    parameters.Parameters.__init__ = parameters_init

    env_cls = wrapper.SBEnvironmentWrapper
    original_make, original_step = env_cls.make, env_cls.step

    def make(self, agent, if_first_server=True, start_level=1, state_representation_type='symbolic'):
        jars.start(agent.a3_slot)
        return original_make(self, agent, if_first_server=False, start_level=start_level,
                             state_representation_type=state_representation_type)

    def step(self, action, batch_gt=False):
        out = original_step(self, action, batch_gt)
        events.emit("shot", call=state["calls"] - 1, agent=self.agent.id, level=self.current_level,
                    reward=float(out[1]), done=bool(out[2]), win=bool(self.did_win))
        return out

    env_cls.make, env_cls.step = make, step
    env_cls.close = lambda self: None  # waves stop their own jars; the official close kills every jar on the host

    client_cls = agent_client.AgentClient
    original_gt, original_gt_shot = (client_cls.get_symbolic_state_without_screenshot,
                                     client_cls.get_symbolic_state_with_screenshot)

    def gt_without(self, **kwargs):
        gt = original_gt(self, **kwargs)
        if not state["gt_saved"]:
            state["gt_saved"] = True
            write_json(Path(cfg["attempt_dir"]) / "first-ground-truth.json", gt)
        return gt

    def gt_with(self):
        image, gt = original_gt_shot(self)
        if not state["shape_saved"]:
            state["shape_saved"] = True
            events.emit("screenshot", shape=list(np.asarray(image).shape))
        return image, gt

    client_cls.get_symbolic_state_without_screenshot = gt_without
    client_cls.get_symbolic_state_with_screenshot = gt_with

    original_run = threads.AgentThread.run

    def thread_run(self):
        try:
            original_run(self)
        except Exception as error:  # the official thread would die silently; the supervisor aborts the attempt
            events.emit("agent_error", agent=self.agent.id, error=f"{type(error).__name__}: {error}")
            raise

    threads.AgentThread.run = thread_run

    def connect_and_run_agents(self, mode='train'):
        call = state["calls"]
        state["calls"] += 1
        state["agents"] = self.agents
        if cfg.get("dry"):  # CPU check of everything before the first game session (no engine)
            events.emit("dry_stop", call=call, agents=[a.id for a in self.agents],
                        levels=[list(a.level_list) for a in self.agents])
            raise SystemExit("dry run: stopped before the first collection")
        results = []
        for wave, members in enumerate(balanced_waves(list(self.agents), cfg["concurrency"])):
            events.emit("wave_start", call=call, wave=wave, agents=[a.id for a in members])
            running = []
            for slot, agent in enumerate(members):
                agent.a3_slot = slot
                agent.ar.server_port = jars.ports(slot)[0]
                worker = threads.AgentThread(
                    agent, agent.env, self.lock, mode=mode, memory_saving_path=self.memory_saving_path,
                    simulation_speed=self.simulation_speed, if_first=slot == 0,
                    if_save_local=agent.if_save_local if hasattr(agent, 'if_save_local')
                    else agent.network.if_save_local)
                worker.start()
                running.append(worker)
                time.sleep(10 if slot == 0 else 1)  # the official start-up pacing
            for worker in running:
                worker.join()
            jars.stop_all()
            results.extend(w.result for w in running)
            events.emit("wave_end", call=call, wave=wave)
        return results

    threads.MultiThreadTrajCollection.connect_and_run_agents = connect_and_run_agents

    result = {"variant": variant, "status": None, "failure": None}
    sys.argv = list(OFFICIAL_ARGV)
    try:
        g = runpy.run_module("world_model.baselines.phyq.vendor.TrainLearningAgent", run_name="__main__",
                             alter_sys=False)
        test_levels = list(g["c"].test_level_list)
        solved = {int(k): int(v) for k, v in g["level_winning_rate"].items()}
        result.update(status="complete", test_levels=test_levels, level_winning_rate=solved,
                      pass_rate=float(np.average(list(solved.values()))),
                      average_testing_winning_rate=float(np.average(g["winning_rate"])),
                      train_levels=len(g["c"].train_level_list), update_steps=int(g["c"].num_update_steps))
    except BaseException as error:
        result.update(status="failed", failure=f"{type(error).__name__}: {error}")
        agents = state["agents"] or []
        result["last_call_did_win"] = {str(a.id): {str(k): v for k, v in a.did_win.items()} for a in agents}
    finally:
        jars.stop_all()
        write_json(Path(cfg["attempt_dir"]) / "child-result.json", result)
    return 0 if result["status"] == "complete" else 1


# ---------------------------------------------------------------------------
# supervisor (window, smoke gate, allowance, ledger, records)
# ---------------------------------------------------------------------------

def window_minutes_left(now=None):
    from zoneinfo import ZoneInfo
    now = now or datetime.now(ZoneInfo(ENGINE_TZ))
    start = now.replace(hour=ENGINE_WINDOW[0][0], minute=ENGINE_WINDOW[0][1], second=0, microsecond=0)
    end = now.replace(hour=ENGINE_WINDOW[1][0], minute=ENGINE_WINDOW[1][1], second=0, microsecond=0)
    return 0.0 if now < start or now >= end else (end - now).total_seconds() / 60


def seconds_until_window(now=None):
    from datetime import timedelta
    from zoneinfo import ZoneInfo
    now = now or datetime.now(ZoneInfo(ENGINE_TZ))
    start = now.replace(hour=ENGINE_WINDOW[0][0], minute=ENGINE_WINDOW[0][1], second=0, microsecond=0)
    if now >= start:
        start += timedelta(days=1)
    return (start - now).total_seconds()


def read_events(path):
    if not Path(path).is_file():
        return []
    out = []
    for line in Path(path).read_text().splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass  # a partially written last line
    return out


def engine_seconds(events, now=None):
    """Worker-seconds of game servers: sum over jar lifetimes (start -> stop, or -> now while running)."""
    now = now or time.time()
    started, total = {}, 0.0
    for e in events:
        if e["kind"] == "jar_start":
            started[e["pid"]] = e["t"]
        elif e["kind"] == "jar_stop" and e["pid"] in started:
            total += e["t"] - started.pop(e["pid"])
    return total + sum(now - t for t in started.values())


def cost_model(events, now=None):
    """(boot s per session, s per episode) from the sessions so far: boot = jar start -> the agent's first shot."""
    now = now or time.time()
    shots = [e for e in events if e["kind"] == "shot"]
    starts = [e for e in events if e["kind"] == "jar_start"]
    if not shots or not starts:
        return None
    first_shot = {}
    for e in shots:
        first_shot.setdefault((e["call"], e["agent"]), e["t"])
    boots = []
    waves = [e for e in events if e["kind"] == "wave_start"]
    for wave in waves:
        wave_starts = [s for s in starts if s["t"] >= wave["t"]][:len(wave["agents"])]
        for slot_start, agent in zip(wave_starts, wave["agents"]):
            if (wave["call"], agent) in first_shot:
                boots.append(first_shot[(wave["call"], agent)] - slot_start["t"])
    boot = sum(boots) / len(boots) if boots else 0.0
    sessions = len(starts)
    episode = max(0.0, (engine_seconds(events, now) - sessions * boot)) / len(shots)
    return boot, episode


def projection_seconds(boot, episode):
    return SESSIONS * boot + EPISODES * episode


def ledger():
    return read_json(LEDGER) if LEDGER.is_file() else {"wall_seconds": 0.0, "gpu_seconds": 0.0,
                                                       "engine_seconds": 0.0, "attempts": {}}


def update_ledger(attempt_id, engine_s, wall_s):
    block = ledger()
    block["attempts"][attempt_id] = {"engine_seconds": engine_s, "wall_seconds": wall_s}
    block["engine_seconds"] = sum(a["engine_seconds"] for a in block["attempts"].values())
    block["wall_seconds"] = sum(a["wall_seconds"] for a in block["attempts"].values())
    block["gpu_seconds"] = 0.0
    block.update(last_at=utc_now(), allowance_seconds=ALLOWANCE_S,
                 note="A3 native-task fidelity engine worker-seconds (game-server lifetimes), CPU only; shared "
                      "190 CPU worker-h allowance with E1 (#132 realised 153.657 h)")
    write_json(LEDGER, block)
    return block


def start_display(log_path):
    import select
    reader, writer = os.pipe()
    with open(log_path, "ab") as handle:
        proc = subprocess.Popen(["Xvnc", "-displayfd", str(writer), "-geometry", "1024x768", "-depth", "24",
                                 "-SecurityTypes", "None", "-rfbport", "0", "-localhost"],
                                pass_fds=(writer,), stdout=handle, stderr=subprocess.STDOUT)
    os.close(writer)
    reply, deadline = b"", time.monotonic() + 15
    try:
        while not reply.endswith(b"\n"):
            if not select.select([reader], [], [], max(0.0, deadline - time.monotonic()))[0]:
                raise TimeoutError("Xvnc did not allocate a display")
            chunk = os.read(reader, 32)
            if not chunk:
                raise RuntimeError("Xvnc closed its display pipe")
            reply += chunk
    finally:
        os.close(reader)
    return ":" + reply.decode().strip(), proc


def run_attempt(variant, concurrency, torch_threads, remaining_s, initial_projection_s):
    attempts = sorted((OUT / variant).glob("attempt-*")) if (OUT / variant).is_dir() else []
    attempt_id = f"{variant}-attempt-{len(attempts) + 1}"
    attempt_dir = OUT / variant / attempt_id.split("-", 1)[1]
    attempt_dir.mkdir(parents=True)
    agents_dir = build_workspace(attempt_dir)
    record = {"schema": SCHEMA_ATTEMPT, "attempt": attempt_id, "variant": VARIANTS[variant]["name"],
              "template": TEMPLATE, "started_at": utc_now(), "status": None, "failure": None,
              "concurrency": concurrency, "remaining_allowance_s_at_start": remaining_s,
              "official_argv": OFFICIAL_ARGV, "jar_flags": VARIANTS[variant]["jar_flags"]}
    display, xvnc = start_display(attempt_dir / "display.log")
    cfg = {"display": display, "port_base": VARIANTS[variant]["port_base"], "jar_flags": VARIANTS[variant]["jar_flags"],
           "concurrency": concurrency, "torch_threads": torch_threads, "events": str(attempt_dir / "events.jsonl"),
           "attempt_dir": str(attempt_dir)}
    write_json(attempt_dir / "config.json", cfg)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONUNBUFFERED="1", OMP_NUM_THREADS=str(torch_threads),
               TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="1")  # torch 1.8 torch.load semantics (MemoryDatasetMemmap.update)
    began = time.time()
    child_proc = subprocess.Popen([str(PHYQ_PYTHON), str(Path(__file__).resolve()), "--child", variant,
                                   str(attempt_dir / "config.json")], cwd=agents_dir, env=env,
                                  stdout=open(attempt_dir / "child.log", "wb"), stderr=subprocess.STDOUT,
                                  start_new_session=True)
    smoke_done, stop_reason = False, None
    last_progress = time.time()
    seen_shots = 0
    while child_proc.poll() is None:
        time.sleep(10)
        events = read_events(attempt_dir / "events.jsonl")
        now = time.time()
        shots = sum(e["kind"] == "shot" for e in events)
        jars_running = engine_seconds(events, now) > 0 and any(e["kind"] == "jar_start" for e in events) and \
            sum(e["kind"] == "jar_start" for e in events) > sum(e["kind"] == "jar_stop" for e in events)
        if shots > seen_shots or not jars_running:
            last_progress, seen_shots = now, shots
        spent = engine_seconds(events, now)
        if any(e["kind"] == "agent_error" for e in events):
            stop_reason = ("harness_failure", next(e for e in events if e["kind"] == "agent_error")["error"])
        elif window_minutes_left() <= 0:
            stop_reason = ("window_closed", "the daytime engine window closed before the run finished")
        elif spent >= remaining_s:
            stop_reason = ("engine_budget", "the shared engine allowance is exhausted")
        elif now - last_progress > STALL_S:
            stop_reason = ("engine_stall", f"no environment step for {STALL_S // 60} minutes with game servers up")
        elif not smoke_done and shots >= SMOKE_EPISODES:
            smoke_done = True
            boot, episode = cost_model(events, now)
            projected = projection_seconds(boot, episode)
            gt_path = attempt_dir / "first-ground-truth.json"
            labels = sorted({f["properties"].get("label") for f in read_json(gt_path)[0]["features"]}) \
                if gt_path.is_file() else None
            screenshot = next((e["shape"] for e in events if e["kind"] == "screenshot"), None)
            smoke = {"schema": "issue_128_a3_fidelity_smoke_v1", "attempt": attempt_id, "at": utc_now(),
                     "episodes": shots, "engine_seconds": spent, "boot_seconds_per_session": boot,
                     "seconds_per_episode": episode, "sessions_per_run": SESSIONS, "episodes_per_run": EPISODES,
                     "projected_engine_seconds": projected, "projected_worker_hours": projected / 3600,
                     "initial_projection_worker_hours": initial_projection_s / 3600, "margin": MARGIN,
                     "remaining_allowance_s": remaining_s, "funded": MARGIN * projected <= remaining_s,
                     "ground_truth_labels": labels, "platform_label_present": None if labels is None
                     else "Platform" in labels, "screenshot_shape": screenshot}
            write_json(attempt_dir / "smoke.json", smoke)
            record["smoke"] = smoke
            log(f"{attempt_id} smoke: {projected / 3600:.2f} worker-h projected, funded={smoke['funded']}, "
                f"labels={labels}")
            if not smoke["funded"]:
                stop_reason = ("engine_budget", "the A3 smoke projection x 1.25 exceeds the remaining allowance")
        if stop_reason:
            break
    if stop_reason:
        kill_group(child_proc.pid)
        for e in read_events(attempt_dir / "events.jsonl"):
            if e["kind"] == "jar_start":
                kill_group(e["pid"])
        now = time.time()
        with open(attempt_dir / "events.jsonl", "a") as handle:  # close the books of killed servers
            started = {}
            for e in read_events(attempt_dir / "events.jsonl"):
                if e["kind"] == "jar_start":
                    started[e["pid"]] = e
                elif e["kind"] == "jar_stop":
                    started.pop(e["pid"], None)
            for pid, e in started.items():
                handle.write(json.dumps({"t": now, "kind": "jar_stop", "slot": e["slot"], "pid": pid,
                                         "killed_by": stop_reason[0]}) + "\n")
        child_proc.wait(timeout=60)
    xvnc.kill()
    xvnc.wait(timeout=10)
    events = read_events(attempt_dir / "events.jsonl")
    spent = engine_seconds(events)
    result_path = attempt_dir / "child-result.json"
    child_result = read_json(result_path) if result_path.is_file() else None
    record.update(finished_at=utc_now(), wall_seconds=time.time() - began, engine_seconds=spent,
                  episodes=sum(e["kind"] == "shot" for e in events),
                  sessions=sum(e["kind"] == "jar_start" for e in events), child_exit=child_proc.returncode,
                  child_result=child_result, cost_model=cost_model(events))
    if stop_reason:
        record.update(status=stop_reason[0], failure=stop_reason[1])
    elif child_result and child_result["status"] == "complete":
        record["status"] = "complete"
    else:
        record.update(status="harness_failure", failure=(child_result or {}).get("failure", "child ended"))
    write_json(attempt_dir / "attempt.json", record)
    update_ledger(attempt_id, spent, record["wall_seconds"])
    log(f"{attempt_id}: {record['status']} ({spent / 3600:.2f} worker-h, {record['episodes']} episodes)")
    return record


def band():
    if sha256_file(BAND) != BAND_SHA256:
        raise ValueError("frozen band file differs")
    return read_json(BAND)


def write_gfid():
    frozen = band()
    variants = {}
    for variant in ORDER:
        name = VARIANTS[variant]["name"]
        records = [read_json(p) for p in sorted((OUT / variant).glob("attempt-*/attempt.json"))] \
            if (OUT / variant).is_dir() else []
        complete = [r for r in records if r["status"] == "complete"]
        entry = {"band": frozen["bands"][name]["band"], "published": frozen["bands"][name]["published"],
                 "attempts": [{k: r.get(k) for k in ("attempt", "status", "failure", "engine_seconds", "episodes",
                                                     "wall_seconds")} for r in records]}
        if complete:
            run = complete[0]["child_result"]
            low, high = entry["band"]
            entry.update(pass_rate=run["pass_rate"], test_tasks=len(run["test_levels"]),
                         solved=sum(run["level_winning_rate"].values()),
                         level_winning_rate=run["level_winning_rate"],
                         **{"pass": bool(low <= run["pass_rate"] <= high)}, status="complete")
        elif any(r["status"] == "engine_budget" for r in records):
            entry.update(status="engine_budget", **{"pass": False})
        else:
            entry.update(status="not_run" if not records else records[-1]["status"], **{"pass": None})
        variants[name] = entry
    block = ledger()
    gfid = {"schema": SCHEMA_GFID, "issue": 128, "arm": "A3", "tier": "native-task (P-engine)",
            "written_at": utc_now(), "band_file": "a3/fidelity-band.json", "band_sha256": "sha256:" + BAND_SHA256,
            "template": TEMPLATE, "statistic": frozen["statistic"], "paper": frozen["source"],
            "procedure": "released TrainLearningAgent.py within_template run (scripts/b127_phyq_fidelity.py)",
            "engine_seconds": block["engine_seconds"], "allowance_seconds": ALLOWANCE_S, "variants": variants,
            "disclosures": [
                "one game server per worker and at most 5 game instances at a time (waves of the 10 official "
                "workers); Phy-Q runs one jar with 10 instances",
                "Relational-Image runs the released image pipeline, which stores the normalised image in a uint8 "
                "memmap (Decision P-other 2026-10-24 'upstream training defects', defect 2); the NovPhy A3-img "
                "training uses the clean transform",
                "device cpu; torch 2.13 / torchvision 0.28 (b127-phyq) instead of torch 1.8.1",
                "single run per variant on 20 test tasks; the band covers test sampling noise only",
                "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1: torch >= 2.6 defaults torch.load to weights_only=True, which "
                "rejects the numpy states MemoryDatasetMemmap.update reloads (torch 1.8 loads them); environment only "
                "(sym attempt 1 stopped on it after step-0 collection; its engine time counts)",
                "the released Linux/config.xml caps attempts at 5 per level per game session "
                "(attempt_limit_per_level=\"5\"): in step 0 each worker gets 5 shots per training task and the other 5 "
                "of RepeatPlay(10) end as 'not valid state' episodes without a transition, as in the released setup"]}
    write_json(OUT / "gfid.json", gfid)
    return gfid


def post_image_decision(sym_record, remaining_s, projected_s):
    funded = MARGIN * projected_s <= remaining_s
    text = f"""## Decision P-engine ({datetime.now().date().isoformat()}): A3 Relational-Image native-task fidelity — {"funded" if funded else "not funded (engine_budget)"}

**Trigger.** Rule of the 2026-10-25 / 2026-10-28 P-engine decisions: Relational-Image follows only if the remainder after Relational-Symbolic's realised cost still covers its projection with the 25 % margin.

**Numbers (fidelity/a3/, ledger/engine-a3-fidelity.json).** Relational-Symbolic realised {sym_record['engine_seconds'] / 3600:.3f} worker-h ({sym_record['episodes']} episodes, {sym_record['sessions']} game sessions, status {sym_record['status']}); A3 engine total so far {ledger()['engine_seconds'] / 3600:.3f} worker-h; remainder {remaining_s / 3600:.3f} worker-h. Relational-Image projection = Relational-Symbolic's realised run cost (same protocol: {EPISODES} episodes, {SESSIONS} sessions) = {projected_s / 3600:.3f} worker-h; x 1.25 = {MARGIN * projected_s / 3600:.3f}.

**Choice.** {"Relational-Image's run starts now (daytime engine window), with its own 50-episode smoke gate against the remainder." if funded else "Relational-Image is not run; A3-img stays tokens-only, typed `engine_budget`."}

**Reason.** The frozen rule applied mechanically; no A3 NovPhy result enters it (Relational-Symbolic's band outcome is not used).
"""
    path = OUT / "decision-image-funding.md"
    path.write_text(text)
    posted = subprocess.run(["gh", "issue", "comment", "128", "--repo", "Sino-Huang/NovPhy", "--body-file", str(path)],
                            capture_output=True, text=True)
    if posted.returncode != 0:
        raise RuntimeError(f"posting the Relational-Image funding decision failed: {posted.stderr}")
    url = posted.stdout.strip()
    with open(DECISIONS, "a") as handle:
        handle.write("\n" + text + f"\nPosted: {url}\n")
    log(f"Relational-Image funding decision posted: {url}")
    return funded, url


def run(variants, concurrency, torch_threads):
    if concurrency > 8:
        raise SystemExit("at most 8 game instances")
    prepare()
    band()
    realised = {}
    for variant in variants:
        if variant == "img":
            sym = [read_json(p) for p in sorted((OUT / "sym").glob("attempt-*/attempt.json"))]
            sym_done = [r for r in sym if r["status"] == "complete"]
            if not sym_done:
                log("Relational-Symbolic has no complete run; Relational-Image is not started")
                break
            remaining = ALLOWANCE_S - ledger()["engine_seconds"]
            if not (OUT / "decision-image-funding.md").is_file():
                funded, _ = post_image_decision(sym_done[0], remaining, sym_done[0]["engine_seconds"])
                if not funded:
                    break
            initial = sym_done[0]["engine_seconds"]
            expected_minutes = sym_done[0]["wall_seconds"] / 60 * 1.3
        else:
            initial = 4.0 * 3600
            expected_minutes = 240.0
        while True:
            done = [p for p in sorted((OUT / variant).glob("attempt-*/attempt.json"))
                    if read_json(p)["status"] == "complete"] if (OUT / variant).is_dir() else []
            if done:
                realised[variant] = read_json(done[0])
                break
            remaining = ALLOWANCE_S - ledger()["engine_seconds"]
            if MARGIN * initial > remaining:
                log(f"{variant}: projection x 1.25 exceeds the remaining allowance; typed engine_budget")
                write_gfid()
                return 0
            left = window_minutes_left()
            if left < expected_minutes:
                wait = seconds_until_window() if left <= 0 else seconds_until_window()
                log(f"{variant}: {left:.0f} min left in the engine window (< {expected_minutes:.0f}); "
                    f"waiting {wait / 3600:.2f} h for the next window")
                time.sleep(wait + 5)
                continue
            record = run_attempt(variant, concurrency, torch_threads, remaining, initial)
            if record["status"] in ("harness_failure", "engine_stall"):
                write_gfid()
                log(f"{variant}: {record['status']} ({record['failure']}); stopping for diagnosis")
                return EXIT_HARNESS
            if record["status"] == "engine_budget":
                break
            if record.get("smoke"):
                initial = record["smoke"]["projected_engine_seconds"]
    gfid = write_gfid()
    log("gfid: " + json.dumps({k: (v.get("pass_rate"), v.get("pass"), v["status"])
                              for k, v in gfid["variants"].items()}))
    return 0


def dry(variant, torch_threads):
    """No engine: the released script up to its first collection call (Parameters, Config level install, network)."""
    prepare()
    attempt_dir = OUT / "dry" / variant
    shutil.rmtree(attempt_dir, ignore_errors=True)
    attempt_dir.mkdir(parents=True)
    agents_dir = build_workspace(attempt_dir)
    cfg = {"display": ":0", "port_base": VARIANTS[variant]["port_base"], "jar_flags": VARIANTS[variant]["jar_flags"],
           "concurrency": 5, "torch_threads": torch_threads, "events": str(attempt_dir / "events.jsonl"),
           "attempt_dir": str(attempt_dir), "dry": True}
    write_json(attempt_dir / "config.json", cfg)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONUNBUFFERED="1", OMP_NUM_THREADS=str(torch_threads),
               TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="1")
    code = subprocess.run([str(PHYQ_PYTHON), str(Path(__file__).resolve()), "--child", variant,
                           str(attempt_dir / "config.json")], cwd=agents_dir, env=env,
                          stdout=open(attempt_dir / "child.log", "wb"), stderr=subprocess.STDOUT).returncode
    events = read_events(attempt_dir / "events.jsonl")
    stop = next((e for e in events if e["kind"] == "dry_stop"), None)
    parameters = next((e for e in events if e["kind"] == "parameters"), None)
    config = (agents_dir.parent / "sciencebirdsgames/Linux/config.xml").read_text()
    levels = sorted(p.name for p in (agents_dir.parent / "sciencebirdsgames/Linux/9001_Data/StreamingAssets/Levels/"
                                     "novelty_level_1/type1/Levels").iterdir())
    summary = {"variant": variant, "child_exit": code, "parameters": parameters, "dry_stop": stop,
               "config_levels": config.count("<game_levels"), "installed_levels": len(levels),
               "first_level": levels[0] if levels else None, "last_level": levels[-1] if levels else None}
    write_json(attempt_dir / "dry.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--run")
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--dry")
    parser.add_argument("--child", nargs=2, metavar=("VARIANT", "CONFIG"))
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--torch-threads", type=int, default=4)
    args = parser.parse_args(argv)
    if args.child:
        return child(*args.child)
    if args.prepare:
        print(json.dumps(prepare(), indent=1))
        return 0
    if args.dry:
        print(json.dumps(dry(args.dry, args.torch_threads), indent=1))
        return 0
    if args.record:
        print(json.dumps(write_gfid(), indent=1))
        return 0
    if args.run:
        variants = [v for v in ORDER if v in args.run.split(",")]
        return run(variants, args.concurrency, args.torch_threads)
    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
