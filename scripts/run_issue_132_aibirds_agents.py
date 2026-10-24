"""Issue-132 (E1 of #127): AIBIRDS heuristic agents acting natively on the NovPhy engine.

Binding runner module: scripts/run_issue_132_aibirds_agents.py

Agents (licenses: .local-artifacts/issue-132-aibirds-agents-v1/licenses.json; nothing is copied or modified):
Datalab (sciencebirdsagents/HeuristicAgents/data_lab_agent_v5.jar), Eagle's Wing (eagle_wings_agent_v6.jar), both
Java clients of the stock Science Birds game_playing_interface.jar, and PigShooter (Phy-Q, vendored verbatim under
world_model/baselines/phyq/vendor/, run in env b127-phyq through Phy-Q's own Python client and environment loop).

One execution = one agent shot on one level from its start state, in two stages:

* Stage A (choice, the agents' native environment). A fresh copy of the stock Science Birds build
  (sciencebirdsgames/Linux.zip, byte-identical to phy-q/NovPhy@0da1258, launched through the #104 player wrapper
  scripts/9001-player-wrapper.sh) and the stock jar, one level installed with Phy-Q's own config.xml attributes,
  a private Xvnc display with the #104 Mesa llvmpipe, PulseAudio disabled. The agent connects through a relay that
  parses its requests, records its first shot request verbatim (code, fx, fy, t1, t2), injects two ground-truth
  requests (62) just before forwarding it (the slingshot anchor, as capture_pipeline_v2 computes it, and a noise
  check), and lets the engine execute it. The engine's own evaluation CSV gives the native level status. The
  Java agents connect to a hard-coded localhost:2004; each worker maps "localhost" to its own loopback address
  with the JVM option -Djdk.net.hosts.file, so workers never share a port.
* Stage B (verdict, the #102 channel). The recorded shot, converted to the slingshot-relative drag with the
  stage-A anchor, is executed once on the frozen #104 player through capture_pipeline_v2.run_campaign (the #102
  executor); the verdict is capture_pipeline_v2.branch_outcome (pig removal, #85 channel), as for every #102 row.
  The release time is engine-inert (#95 release_inertness); the tap time is passed through.

Modes:
  --levels                 print the N1 400 list (metadata only) and the smoke list
  --smoke                  20 fit-split levels (never evaluation) x 3 agents x 2 resets (stage A), then stage B for
                           every distinct recorded shot; writes smoke/smoke.json (determinism, seconds/execution,
                           anchor and verdict agreement, projection against the 190 CPU worker-hour allowance)
  --project                recompute the projection from smoke/smoke.json
  --pigshooter-driver PORT internal: the Phy-Q PigShooter loop (run with the b127-phyq python, cwd = vendor dir)

Engine runs are CPU only; worker count is capped at 8.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import os
import shutil
import socket
import struct
import subprocess
import sys
import threading
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IDENTITY = "issue-132-aibirds-agents-v1"
OUTPUT = ROOT / ".local-artifacts" / IDENTITY
COHORT = ROOT / ".local-artifacts/issue-109-sealed-cohort-v1"
STOCK_ZIP = ROOT / "sciencebirdsgames/Linux.zip"
STOCK_JAR = ROOT / "sciencebirdsgames/Linux/game_playing_interface.jar"
PLAYER_104 = ROOT / ".local-artifacts/issue-104-power-cohort-v1/player"
WRAPPER = ROOT / "scripts/9001-player-wrapper.sh"
AGENT_DIR = ROOT / "sciencebirdsagents/HeuristicAgents"
VENDOR = ROOT / "world_model/baselines/phyq/vendor"
PHYQ_PYTHON = Path.home() / "miniconda3/envs/b127-phyq/bin/python"
MESA = Path.home() / ".cache/novphy-mesa/mesa-26.1.2-llvm22.1.6"

SCHEMA_ATTEMPT = "issue_132_stage_a_attempt_v1"
SCHEMA_VERDICT = "issue_132_stage_b_verdict_v1"
SCHEMA_SMOKE = "issue_132_smoke_v1"
SHA = {"stock_zip": "c45c8b44078ebcf1ba3168d7606a6e2e739cd70da6bf1e0400f66f12d0ba03f0",
       "jar": "b8ebda1806e8e440eda2e810c7c2892bf4a7251252bb79f9fb827a1c7af81e7a",
       "datalab": "c11ddd6e8d6f129eb8f88cf4f1f4e8f69f7ade0da4cf1c310d1454f6d4b43354",
       "eaglewing": "5759b9c99a3bfc2f05d89e156ad6beb6ad120f9c8e41c6c0b2e8ab0358914ae1"}
AGENTS = {
    "datalab": {"kind": "java", "jar": AGENT_DIR / "data_lab_agent_v5.jar", "args": ["1"]},
    "eaglewing": {"kind": "java", "jar": AGENT_DIR / "eagle_wings_agent_v6.jar", "args": ["1"]},
    "pigshooter": {"kind": "python"},
}
# Jar flags: Phy-Q starts the jar with --dev for its Python agents (SBEnvironmentWrapper.__start_server) and
# without flags for the Java agents (README 4.1); both are kept.
JAR_FLAGS = {"datalab": [], "eaglewing": [], "pigshooter": ["--dev"]}
# Phy-Q's config.xml attributes (phy-q/benchmark Linux.zip config.xml); Phy-Q lists the last level twice
# ("a repeated level for the weird not loading last level bug", Utils/PrepareTestConfig.py).
PHYQ_TRIAL = {"id": "0", "number_of_executions": "0", "checkpoint_time_limit": "9999999",
              "checkpoint_interaction_limit": "9999999", "notify_novelty": "False"}
PHYQ_SET = {"mode": "training", "time_limit": "9999999", "total_interaction_limit": "9999999",
            "attempt_limit_per_level": "5", "allow_level_selection": "True"}

SHOOT_CODES = {31: "cshoot", 32: "pshoot", 41: "cfastshoot", 42: "pfastshoot", 38: "gtshoot"}
# Agent -> server request payload sizes (bytes after the 1-byte code), from Phy-Q's client and the jar protocol.
REQUEST_PAYLOAD = {1: 5, 2: 4, 11: 0, 12: 0, 14: 0, 15: 0, 23: 0, 31: 16, 32: 16, 34: 0, 35: 0, 36: 0, 41: 16,
                   42: 16, 51: 4, 52: 0, 53: 0, 61: 0, 62: 0, 63: 0, 64: 0, 65: 0, 68: 0, 69: 0}
GT_WITHOUT_SCREENSHOT = 62
FRAME_HEIGHT = 480
SMOKE_LEVELS_PER_FAMILY = 10
SMOKE_REPEATS = 2
MAX_WORKERS = 8
STAGE_A_SHOT_DEADLINE = 420.0   # seconds from launch until the first shot request
STAGE_A_VERDICT_DEADLINE = 180.0  # seconds from the shot until the engine's evaluation row
ALLOWANCE_WORKER_HOURS = 190.0
N1_LEVELS = 400
PORT_BASE = 39400


def utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(message):
    print(f"[{IDENTITY}] {message}", flush=True)


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


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
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


# ---------------------------------------------------------------------------
# levels (#109 sealed cohort plan; metadata only)
# ---------------------------------------------------------------------------

def cohort_plan():
    return read_json(COHORT / "plan.json")


def split_levels(split, per_family=None, plan=None):
    """Level records of one #109 split, family by family in plan order, each family in ordinal order."""
    plan = plan or cohort_plan()
    out = []
    for family, roster in plan["splits"][split].items():
        rows = sorted(roster, key=lambda level: level["ordinal"])
        out.extend(rows if per_family is None else rows[:per_family])
    return out


def n1_levels(plan=None):
    """The #102 N1 set: all 400 #109 evaluation levels (#104 stream issue109-evaluation)."""
    levels = split_levels("evaluation", plan=plan)
    if len(levels) != N1_LEVELS or len({level["identity"] for level in levels}) != N1_LEVELS:
        raise ValueError("N1 must be the 400 distinct #109 evaluation levels")
    return levels


def smoke_levels(plan=None):
    """20 fit-split levels (ordinals 1-10 of each family): the smoke never touches the evaluation split."""
    return split_levels("fit", SMOKE_LEVELS_PER_FAMILY, plan=plan)


def bird_slots(level):
    return [slot for slot in level["generated_slots"] if slot.startswith("bird:")]


def level_member(level):
    """The level record with its XML text, hash-checked against the frozen plan."""
    xml = (COHORT / level["xml_path"]).read_bytes()
    if sha256_bytes(xml) != level["xml_sha256"]:
        raise ValueError(f"{level['identity']}: level xml differs from its frozen sha256")
    return {**level, "xml": xml.decode()}


# ---------------------------------------------------------------------------
# stage A runtime: stock build, Phy-Q config, display, jar, agent
# ---------------------------------------------------------------------------

def stock_player(root=OUTPUT / "stock-player"):
    """Stock Linux build without its bundled level library, the stock jar, and the #104 player wrapper."""
    root = Path(root)
    if (root / "player.json").is_file():
        return root
    if sha256_file(STOCK_ZIP) != SHA["stock_zip"] or sha256_file(STOCK_JAR) != SHA["jar"]:
        raise ValueError("stock build or jar differs from the hashes recorded in licenses.json")
    tmp = root.with_name(root.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    with zipfile.ZipFile(STOCK_ZIP) as archive:
        for info in archive.infolist():
            name = info.filename
            if name == "Linux/" or ("/StreamingAssets/Levels/" in name and not name.endswith("/Levels/")):
                continue
            target = tmp / Path(name).relative_to("Linux")
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, open(target, "wb") as sink:
                shutil.copyfileobj(source, sink)
    (tmp / "9001.x86_64").rename(tmp / "9001-player.x86_64")
    shutil.copy2(WRAPPER, tmp / "9001.x86_64")
    for name in ("9001.x86_64", "9001-player.x86_64"):
        os.chmod(tmp / name, 0o755)
    shutil.copy2(STOCK_JAR, tmp / "game_playing_interface.jar")
    shutil.copy2(PLAYER_104 / "serverbackup", tmp / "serverbackup")
    write_json(tmp / "player.json", {"stock_zip_sha256": SHA["stock_zip"], "jar_sha256": SHA["jar"],
                                     "wrapper_sha256": sha256_file(WRAPPER),
                                     "excluded": "9001_Data/StreamingAssets/Levels/* (bundled level library)",
                                     "built_at": utc_now()})
    tmp.rename(root)
    return root


def phyq_config(relative_xml):
    """config.xml with Phy-Q's attributes and the one level (listed twice, as Phy-Q does)."""
    import xml.etree.ElementTree as ET
    evaluation = ET.Element("evaluation")
    ET.SubElement(evaluation, "novelty_detection_measurement",
                  {"step": "1", "measure_in_training": "False", "measure_in_testing": "False"})
    trial = ET.SubElement(ET.SubElement(evaluation, "trials"), "trial", dict(PHYQ_TRIAL))
    level_set = ET.SubElement(trial, "game_level_set", dict(PHYQ_SET))
    for _ in range(2):
        ET.SubElement(level_set, "game_levels", {"level_path": relative_xml})
    ET.indent(evaluation, space="  ")
    return ET.tostring(evaluation, encoding="utf-8", xml_declaration=True) + b"\n"


def install_level(game, member):
    relative = (Path("9001_Data/StreamingAssets/Levels") / f"novelty_level_{member['novelty_level']}"
                / member["generator_family"] / "Levels" / f"{member['identity']}.xml")
    (game / relative).parent.mkdir(parents=True, exist_ok=True)
    (game / relative).write_text(member["xml"])
    (game / "config.xml").write_bytes(phyq_config(relative.as_posix()))
    return relative.as_posix()


def engine_environment(display, home):
    env = dict(os.environ)
    env.pop("XDG_RUNTIME_DIR", None)  # a stale user PulseAudio server blocks Unity's audio init
    env.update(DISPLAY=display, XDG_DATA_HOME=str(home / "xdg"), XDG_CONFIG_HOME=str(home / "config"),
               PULSE_SERVER="unix:/nonexistent", CUDA_VISIBLE_DEVICES="",
               LD_LIBRARY_PATH=str(MESA / "lib"), LIBGL_DRIVERS_PATH=str(MESA / "lib/dri"),
               __GLX_VENDOR_LIBRARY_NAME="mesa")
    env.pop("NOVPHY_PHYSICS_CAPTURE_PORT", None)
    return env


def listening(port):
    """True when a local TCP socket listens on ``port`` (read from /proc; never connects)."""
    target = f":{port:04X}"
    for table in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            rows = Path(table).read_text().splitlines()[1:]
        except OSError:
            continue
        for row in rows:
            fields = row.split()
            if fields[1].endswith(target) and fields[3] == "0A":
                return True
    return False


def slot_ports(slot):
    base = PORT_BASE + 10 * slot
    return {"agent": base, "game": base + 1, "relay": base + 5, "loopback": f"127.0.1.{slot + 1}"}


# ---------------------------------------------------------------------------
# relay: parse agent requests, record the first shot, inject the anchor ground truth
# ---------------------------------------------------------------------------

def parse_requests(buffer):
    """Split complete agent requests off ``buffer``; returns ([(code, payload)], rest).

    Fixed-size requests use REQUEST_PAYLOAD; any other code (variable-length novelty reports) takes the rest of
    the buffer, because the synchronous clients write one request per send.
    """
    out = []
    while buffer:
        code = buffer[0]
        size = REQUEST_PAYLOAD.get(code)
        if size is None:
            out.append((code, bytes(buffer[1:])))
            return out, b""
        if len(buffer) < 1 + size:
            break
        out.append((code, bytes(buffer[1:1 + size])))
        buffer = buffer[1 + size:]
    return out, bytes(buffer)


def decode_shot(code, payload):
    if code == 38:
        values = struct.unpack("!" + "i" * (len(payload) // 4), payload[:4 * (len(payload) // 4)])
    else:
        values = struct.unpack("!iiii", payload)
    return {"code": code, "name": SHOOT_CODES[code], "polar": code in (32, 42),
            "fx": values[0], "fy": values[1], "t1": values[2], "t2": values[3],
            "wire_hex": (bytes([code]) + payload).hex()}


def decode_ground_truth(body):
    """Phy-Q's client strips the 5-byte trailer; raw_decode ignores any trailer without assuming its length."""
    text = body.decode("utf-8", errors="replace")
    value, _ = json.JSONDecoder().raw_decode(text)
    return value


def slingshot_anchor(ground_truth):
    from scripts.slingshot_readiness import slingshot_observation_from_symbolic_state
    return slingshot_observation_from_symbolic_state(ground_truth, FRAME_HEIGHT)


def slingshot_vertices(ground_truth):
    for item in ground_truth if isinstance(ground_truth, list) else []:
        for feature in item.get("features", []) if isinstance(item, dict) else []:
            if feature.get("properties", {}).get("label") == "Slingshot":
                return feature.get("geometry", {}).get("coordinates")
    return None


class Relay:
    """One agent connection, relayed to the jar's agent port."""

    def __init__(self, host, port, upstream_port, log_path):
        self.upstream_port = upstream_port
        self.log = open(log_path, "w")
        self.lock = threading.Lock()   # owns upstream reads (pump vs. injection)
        self.shot = None
        self.anchor = None
        self.injections = []
        self.counts = {}
        self.shot_at = None
        self.error = None
        self.closed = threading.Event()
        self.server = socket.socket()
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind((host, port))
        self.server.listen(1)
        self.server.settimeout(1.0)
        threading.Thread(target=self._accept, daemon=True).start()

    def _note(self, direction, data):
        self.log.write(json.dumps({"t": round(time.time(), 3), "d": direction, "n": len(data),
                                   "head": data[:32].hex()}) + "\n")
        self.log.flush()

    def _accept(self):
        while not self.closed.is_set():
            try:
                client, _ = self.server.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            try:
                upstream = socket.create_connection(("127.0.0.1", self.upstream_port), timeout=30)
            except OSError as error:
                self.error = f"upstream connect: {error}"
                client.close()
                return
            upstream.settimeout(None)
            threading.Thread(target=self._pump, args=(upstream, client), daemon=True).start()
            self._requests(client, upstream)
            return

    def _pump(self, upstream, client):
        import select
        while not self.closed.is_set():
            with self.lock:
                try:
                    ready = select.select([upstream], [], [], 0.05)[0]
                    data = upstream.recv(1 << 16) if ready else None
                except (OSError, ValueError):  # closed by the request side
                    break
            if data is None:
                continue
            if not data:
                break
            self._note("S", data)
            try:
                client.sendall(data)
            except OSError:
                break

    def _read_exact(self, sock, size):
        data = b""
        while len(data) < size:
            chunk = sock.recv(size - len(data))
            if not chunk:
                raise ConnectionError("upstream closed during injection")
            data += chunk
        return data

    def _inject_ground_truth(self, upstream):
        with self.lock:
            upstream.sendall(bytes([GT_WITHOUT_SCREENSHOT]))
            size = struct.unpack("!I", self._read_exact(upstream, 4))[0]
            body = self._read_exact(upstream, size)
        return decode_ground_truth(body)

    def _requests(self, client, upstream):
        buffer = b""
        try:
            while not self.closed.is_set():
                data = client.recv(1 << 16)
                if not data:
                    break
                self._note("A", data)
                requests, buffer = parse_requests(buffer + data)
                for code, payload in requests:
                    self.counts[code] = self.counts.get(code, 0) + 1
                    if code in SHOOT_CODES:
                        if self.shot is not None:
                            self.closed.set()   # one attempt per reset: a second shot is never forwarded
                            return
                        shot = decode_shot(code, payload)
                        truths = [self._inject_ground_truth(upstream) for _ in range(2)]
                        self.injections = [slingshot_vertices(gt) for gt in truths]
                        self.anchor = slingshot_anchor(truths[0])
                        self.shot, self.shot_at = shot, time.time()
                    upstream.sendall(bytes([code]) + payload)
        except Exception as error:  # recorded as a typed relay failure
            self.error = f"{type(error).__name__}: {error}"
        finally:
            for sock in (client, upstream):
                try:
                    sock.close()
                except OSError:
                    pass

    def close(self):
        self.closed.set()
        try:
            self.server.close()
        except OSError:
            pass
        self.log.close()


def evaluation_rows(game):
    rows = []
    for path in sorted(game.glob("Agent_*_Evaluation_*/*EvaluationData.csv")):
        with open(path, newline="") as handle:
            reader = csv.DictReader(handle, skipinitialspace=True)
            rows.extend({k.strip(): (v or "").strip() for k, v in row.items() if k} for row in reader)
    return rows


def drag_details(game):
    """The game's own record of every received shot (AIBirdsConnection 'DragDetails' line)."""
    out = []
    for path in sorted(game.glob("sciencebirds_*.log")):
        for line in path.read_text(errors="replace").splitlines():
            if line.startswith("DragDetails:"):
                fields = [field.strip() for field in line.split(":")]
                values = {}
                for key in ("dragx", "dragy", "dragDX", "DragDY", "TapTime", "Score"):
                    if key in fields:
                        raw = fields[fields.index(key) + 1].split(",")[0].strip()
                        values[key] = float(raw)
                out.append(values)
    return out


def kill_group(process):
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, 9)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass


def agent_command(agent, ports, work):
    spec = AGENTS[agent]
    if spec["kind"] == "java":
        hosts = work / "hosts"
        hosts.write_text(f"{ports['loopback']} localhost\n")
        return (["java", f"-Djdk.net.hosts.file={hosts}", "-jar", str(spec["jar"]), *spec["args"]], work)
    return ([str(PHYQ_PYTHON), str(Path(__file__).resolve()), "--pigshooter-driver", str(ports["relay"])], VENDOR)


def stage_a_attempt(agent, level, repeat, slot, root):
    """One agent attempt on a fresh stock runtime; writes and returns its record (typed failures included)."""
    from scripts import issue_76_display_start as display_start
    from scripts.issue_76_shared_player_storage import clone_player
    work = Path(root) / agent / f"{level['identity']}-r{repeat}"
    if (work / "record.json").is_file():
        return read_json(work / "record.json")
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    ports = slot_ports(slot)
    record = {"schema": SCHEMA_ATTEMPT, "agent": agent, "level": level["identity"], "repeat": repeat,
              "slot": slot, "ports": ports, "jar_flags": JAR_FLAGS[agent], "started_at": utc_now(),
              "status": None, "failure": None, "shot": None, "anchor": None, "native": None}
    began = time.monotonic()
    jar = agent_process = display_process = relay = None
    game = work / "runtime"
    try:
        for port in (ports["agent"], ports["game"], ports["relay"]):
            if listening(port):
                raise RuntimeError(f"port {port} already in use")
        clone_player(stock_player(), game)
        record["level_path"] = install_level(game, level_member(level))
        display, display_process = display_start.start_display(work / "display.log")
        env = engine_environment(display, work)
        jar = subprocess.Popen(["java", "-jar", "./game_playing_interface.jar", *JAR_FLAGS[agent],
                                "--agent-port", str(ports["agent"]), "--game-start-port", str(ports["game"])],
                               cwd=game, env=env, stdout=open(work / "jar.log", "wb"), stderr=subprocess.STDOUT,
                               stdin=subprocess.DEVNULL, start_new_session=True)
        deadline = time.monotonic() + 60
        while not listening(ports["agent"]):
            if jar.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError("jar did not open its agent port")
            time.sleep(0.2)
        host, port = ((ports["loopback"], 2004) if AGENTS[agent]["kind"] == "java"
                      else ("127.0.0.1", ports["relay"]))
        relay = Relay(host, port, ports["agent"], work / "relay.jsonl")
        command, cwd = agent_command(agent, ports, work)
        launched, launched_wall = time.monotonic(), time.time()
        agent_process = subprocess.Popen(command, cwd=cwd, env=env, stdout=open(work / "agent.log", "wb"),
                                         stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                         start_new_session=True)
        record["agent_command"] = [str(part) for part in command]
        while relay.shot is None:
            if relay.error or agent_process.poll() is not None or relay.closed.is_set():
                raise RuntimeError(f"agent ended before shooting (relay error: {relay.error}; "
                                   f"agent exit: {agent_process.poll()})")
            if time.monotonic() - launched > STAGE_A_SHOT_DEADLINE:
                raise TimeoutError("no shot within the stage-A shot deadline")
            time.sleep(0.2)
        record["decision_seconds"] = round(relay.shot_at - launched_wall, 3)
        shot_deadline = time.monotonic() + STAGE_A_VERDICT_DEADLINE
        rows = []
        while not rows:
            rows = evaluation_rows(game)
            if rows or time.monotonic() > shot_deadline:
                break
            time.sleep(0.5)
        record["shot"] = relay.shot
        record["anchor"] = relay.anchor
        record["anchor_noise_free"] = (relay.injections[0] is not None
                                       and relay.injections[0] == relay.injections[1])
        record["game_drag_details"] = drag_details(game)
        record["native"] = rows[0] if rows else None
        record["request_counts"] = {str(k): v for k, v in sorted(relay.counts.items())}
        record["status"] = "complete" if rows else "no_native_verdict"
    except Exception as error:
        record["status"] = "failed"
        record["failure"] = f"{type(error).__name__}: {error}"
        if relay is not None and relay.shot is not None:
            record["shot"], record["anchor"] = relay.shot, relay.anchor
    finally:
        for process in (agent_process, jar):
            kill_group(process)
        if relay is not None:
            relay.close()
        if display_process is not None:
            display_process.kill()
            display_process.wait(timeout=10)
        for name in ("sciencebirds_*.log", "Agent_*_Evaluation_*"):
            for path in game.glob(name):
                target = work / "engine-logs" / path.name
                target.parent.mkdir(exist_ok=True)
                (shutil.copytree if path.is_dir() else shutil.copy2)(path, target)
        if record["status"] == "complete":
            shutil.rmtree(game, ignore_errors=True)
        record["wall_seconds"] = round(time.monotonic() - began, 3)
        record["finished_at"] = utc_now()
        write_json(work / "record.json", record)
    return record


def run_stage_a(jobs, root, workers):
    """``jobs`` = [(agent, level, repeat)]; slots are reused only after their attempt finished."""
    workers = min(workers, MAX_WORKERS)
    stock_player()  # built once, before any worker clones it
    free = list(range(workers))
    lock = threading.Lock()

    def run(job):
        with lock:
            slot = free.pop()
        try:
            return stage_a_attempt(*job, slot, root)
        finally:
            with lock:
                free.append(slot)

    with concurrent.futures.ThreadPoolExecutor(workers) as pool:
        records = list(pool.map(run, jobs))
    return records


# ---------------------------------------------------------------------------
# stage B: the recorded shot through the #102 executor on the #104 player
# ---------------------------------------------------------------------------

def shot_action(shot, anchor):
    """Slingshot-relative drag reproducing the agent's wire release point under ``anchor``.

    Inverse of slingshot_readiness._anchored_action_and_command: x = int(gameX) + drag_x and
    y = frame_height - 1 - (int(gameY) - drag_y).
    """
    if shot["polar"]:
        raise ValueError("polar shots are not transferable without the jar's polar conversion")
    start_x, start_y = int(anchor["gameX"]), int(anchor["gameY"])
    return {"drag_x": int(shot["fx"]) - start_x,
            "drag_y": int(shot["fy"]) - (FRAME_HEIGHT - 1) + start_y,
            "tap_time_ms": int(shot["t2"]), "release_time_ms": int(shot["t1"])}


def execution_suffix(action):
    text = json.dumps(action, sort_keys=True)
    return "e1-" + sha256_bytes(text.encode())[:10]


def dispatch_record(stream, level, action):
    from scripts import prepare_power_cohort as cohort
    template = cohort.templates_109()[level["generator_family"]]
    candidate = {"suffix": execution_suffix(action), "action": dict(action), "inventory_entries": []}
    return cohort.dispatch_record(stream, level, level["identity"], candidate, template)


def run_stage_b(engine, records, workers, label):
    from scripts import capture_pipeline_v2 as pipeline
    from scripts import run_closed_loop_evaluation as closed_loop
    engine = Path(engine)
    binding = closed_loop.prepare_engine(engine)
    limits = {**closed_loop.engine_limits(engine), "workers": min(workers, MAX_WORKERS)}
    environment = {"PULSE_SERVER": os.environ.get("PULSE_SERVER"), "XDG_RUNTIME_DIR": os.environ.get("XDG_RUNTIME_DIR")}
    os.environ.pop("XDG_RUNTIME_DIR", None)
    os.environ["PULSE_SERVER"] = "unix:/nonexistent"
    try:
        wall = pipeline.run_campaign(engine, IDENTITY, records, limits, label) if records else 0.0
    finally:
        for key, value in environment.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    return {**binding, "campaign_wall_seconds": wall, "workers": limits["workers"]}


def stage_b_verdict(engine, identity, expected_wire):
    from scripts import capture_pipeline_v2 as pipeline
    entry = pipeline.branch_entry(engine, identity)
    out = {"schema": SCHEMA_VERDICT, "execution": identity, "status": entry["status"],
           "failure": entry.get("failure"), "failure_class": entry.get("failure_class"),
           "wall_seconds": entry.get("wall_seconds"), "pig_removed": None, "stop_kind": None,
           "native_clear": None, "wire_reproduced": None, "game_drag_details": None}
    if entry["status"] == "complete":
        result = read_json(Path(engine) / "results" / f"{identity}.json")
        outcome = pipeline.branch_outcome(result["segments"][0])
        details = drag_details(Path(engine) / "attempts" / identity / "runtime")
        out.update(pig_removed=outcome["pig_removed"], stop_kind=outcome["stop_kind"],
                   native_clear=outcome["native_clear"], game_drag_details=details,
                   stage_seconds=result.get("stage_seconds"))
        if details:
            out["wire_reproduced"] = (int(details[0]["dragDX"]) == expected_wire["fx"]
                                      and int(details[0]["DragDY"]) == expected_wire["fy"])
    return out


# ---------------------------------------------------------------------------
# smoke and projection
# ---------------------------------------------------------------------------

def shot_key(shot):
    return None if shot is None else (shot["code"], shot["fx"], shot["fy"], shot["t1"], shot["t2"])


def determinism(records):
    """Per agent: levels whose two resets both shot, and how many of them repeated the identical wire shot."""
    by = {}
    for record in records:
        by.setdefault(record["agent"], {}).setdefault(record["level"], {})[record["repeat"]] = record
    out = {}
    for agent, levels in sorted(by.items()):
        compared = identical = 0
        for repeats in levels.values():
            keys = [shot_key(repeats[r]["shot"]) for r in sorted(repeats)]
            if len(keys) >= 2 and all(key is not None for key in keys):
                compared += 1
                identical += len(set(keys)) == 1
        out[agent] = {"levels_compared": compared, "identical": identical,
                      "deterministic": compared > 0 and identical == compared}
    return out


def projection(seconds_a, seconds_b, deterministic, levels=N1_LEVELS, allowance=ALLOWANCE_WORKER_HOURS):
    """Worker-hours of the full E1 run: pass@1 for deterministic agents, pass@3 for agents whose shots vary.

    ``seconds_a`` / ``seconds_b``: mean worker seconds per stage-A attempt / stage-B execution, per agent.
    Every attempt is costed with its own stage-B execution (no sharing credit), an upper bound.
    """
    rows, total = {}, 0.0
    for agent in sorted(deterministic):
        attempts = levels * (1 if deterministic[agent] else 3)
        hours = attempts * (seconds_a[agent] + seconds_b[agent]) / 3600
        rows[agent] = {"attempts_per_level": 1 if deterministic[agent] else 3, "executions": attempts,
                       "stage_a_seconds": seconds_a[agent], "stage_b_seconds": seconds_b[agent],
                       "worker_hours": round(hours, 2)}
        total += hours
    return {"agents": rows, "worker_hours": round(total, 2), "allowance_worker_hours": allowance,
            "remainder_for_a3_worker_hours": round(allowance - total, 2), "within_allowance": total <= allowance}


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def smoke(workers):
    out = OUTPUT / "smoke"
    levels = smoke_levels()
    jobs = [(agent, level, repeat) for repeat in range(SMOKE_REPEATS) for level in levels for agent in AGENTS]
    log(f"smoke stage A: {len(jobs)} attempts on {len(levels)} fit levels, {min(workers, MAX_WORKERS)} workers")
    began = time.monotonic()
    records = run_stage_a(jobs, out / "stage-a", workers)
    stage_a_wall = time.monotonic() - began
    by_level = {level["identity"]: level for level in levels}
    executions = {}
    for record in records:
        if record["shot"] is None or record["anchor"] is None or record["shot"]["polar"]:
            continue
        action = shot_action(record["shot"], record["anchor"])
        identity = f"{record['level']}-{execution_suffix(action)}"
        executions.setdefault(identity, {"level": record["level"], "action": action, "wire": record["shot"],
                                         "attempts": []})["attempts"].append(
            f"{record['agent']}/{record['level']}-r{record['repeat']}")
    dispatch = [dispatch_record("issue109-fit", by_level[e["level"]], e["action"]) for e in executions.values()]
    log(f"smoke stage B: {len(dispatch)} distinct shots")
    began = time.monotonic()
    binding = run_stage_b(out / "engine", dispatch, workers, f"{IDENTITY}:smoke")
    stage_b_wall = time.monotonic() - began
    verdicts = {}
    for identity, execution in executions.items():
        verdicts[identity] = {**stage_b_verdict(out / "engine", identity, execution["wire"]), **execution}
        write_json(out / "verdicts" / f"{identity}.json", verdicts[identity])
    summary = summarize(records, verdicts)
    summary.update(schema=SCHEMA_SMOKE, identity=IDENTITY, levels=[level["identity"] for level in levels],
                   repeats=SMOKE_REPEATS, workers=min(workers, MAX_WORKERS), stage_a_wall_seconds=stage_a_wall,
                   stage_b_wall_seconds=stage_b_wall, stage_b_binding=binding, written_at=utc_now())
    write_json(out / "smoke.json", summary)
    log(json.dumps(summary["projection"], indent=2))
    return summary


def summarize(records, verdicts):
    by_attempt = {}
    for identity, verdict in verdicts.items():
        for attempt in verdict["attempts"]:
            by_attempt[attempt] = verdict
    agents = {}
    for agent in AGENTS:
        rows = [r for r in records if r["agent"] == agent]
        b = [by_attempt.get(f"{agent}/{r['level']}-r{r['repeat']}") for r in rows]
        native = [r["native"]["LevelStatus"] == "Pass" for r in rows if r.get("native")]
        agree = [(r["native"]["LevelStatus"] == "Pass") == v["pig_removed"] for r, v in zip(rows, b)
                 if r.get("native") and v and v["pig_removed"] is not None]
        agents[agent] = {
            "attempts": len(rows),
            "status": {s: sum(r["status"] == s for r in rows) for s in sorted({r["status"] for r in rows})},
            "failures": sorted({r["failure"] for r in rows if r["failure"]}),
            "shots": sum(r["shot"] is not None for r in rows),
            "polar_shots": sum(bool(r["shot"] and r["shot"]["polar"]) for r in rows),
            "anchor_noise_free": sum(bool(r.get("anchor_noise_free")) for r in rows),
            "stage_a_seconds_mean": mean([r["wall_seconds"] for r in rows]),
            "decision_seconds_mean": mean([r.get("decision_seconds") for r in rows]),
            "native_pass_rate": mean(native),
            "stage_b_complete": sum(bool(v and v["status"] == "complete") for v in b),
            "stage_b_pig_removed_rate": mean([v["pig_removed"] for v in b if v and v["pig_removed"] is not None]),
            "stage_b_wire_reproduced": sum(bool(v and v["wire_reproduced"]) for v in b),
            "native_vs_stage_b_agreement": mean(agree),
        }
    distinct_b = [v for v in verdicts.values() if v["status"] == "complete"]
    seconds_b = mean([v["wall_seconds"] for v in distinct_b])
    det = determinism(records)
    seconds_a = {a: agents[a]["stage_a_seconds_mean"] or 0.0 for a in AGENTS}
    proj = projection(seconds_a, {a: seconds_b or 0.0 for a in AGENTS},
                      {a: det[a]["deterministic"] for a in AGENTS})
    return {"agents": agents, "determinism": det, "stage_b_executions": len(verdicts),
            "stage_b_complete": len(distinct_b), "stage_b_seconds_mean": seconds_b, "projection": proj}


# ---------------------------------------------------------------------------
# PigShooter driver (b127-phyq env; Phy-Q's AgentThread loop for one level, one attempt)
# ---------------------------------------------------------------------------

def pigshooter_driver(port):
    sys.path.insert(0, str(ROOT))
    from world_model.baselines.phyq.vendor.Client.agent_client import AgentClient
    from world_model.baselines.phyq.vendor.HeuristicAgents.PigShooter import PigShooter
    from world_model.baselines.phyq.vendor.SBEnvironment.SBEnvironmentWrapper import SBEnvironmentWrapper
    from world_model.baselines.phyq.vendor.Utils.LevelSelection import LevelSelectionSchema

    class NovPhyAgentClient(AgentClient):
        """Phy-Q's client; only the level-load transport differs.

        On the NovPhy build, request 51 (selectlevel) sent from the main menu raises IndexOutOfRangeException in
        SymbolicGameState.GetGTJson (no slingshot in the menu scene) and never answers. The runtime holds exactly
        one level, so the environment's load_level(1) is sent as request 53 (load next available level), the
        request the Java agents and the #104 pipeline use. The agent's policy code is untouched.
        """

        def load_level(self, level_number):
            return self.load_next_available_level()

    # MultiAgentTestOnly.py: SBEnvironmentWrapper(reward_type='passing', speed=100), RepeatPlay(1);
    # HeuristicAgentThread.AgentThread.run: make -> reset -> select_action/step until done.
    env = SBEnvironmentWrapper(reward_type="passing", speed=100)
    agent = PigShooter(env=env, level_list=[1], id=1,
                       level_selection_function=LevelSelectionSchema.RepeatPlay(1).select)
    agent.ar = NovPhyAgentClient("127.0.0.1", port)  # Utils/server_client_config.json points at the shared 2004
    env.make(agent=agent, if_first_server=False, start_level=1, state_representation_type=agent.state_representation_type)
    state, _, is_done, info = env.reset()
    while not is_done:
        state, _, is_done, info = env.step(agent.select_action(state))
    print("pigshooter done", info[:2], flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--levels", action="store_true")
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--project", action="store_true")
    mode.add_argument("--pigshooter-driver", type=int, metavar="PORT")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    args = parser.parse_args(argv)
    if args.pigshooter_driver:
        pigshooter_driver(args.pigshooter_driver)
        return 0
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise SystemExit('engine runs are CPU only: set CUDA_VISIBLE_DEVICES=""')
    if args.levels:
        print(json.dumps({"n1": [level["identity"] for level in n1_levels()],
                          "smoke": [level["identity"] for level in smoke_levels()]}, indent=2))
        return 0
    if args.smoke:
        smoke(args.workers)
        return 0
    summary = read_json(OUTPUT / "smoke" / "smoke.json")
    print(json.dumps(summary["projection"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
