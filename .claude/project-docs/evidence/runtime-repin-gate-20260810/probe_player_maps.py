#!/usr/bin/env python3
"""Capture what the live physics-port owner actually maps, and nothing else.

The listener-only diagnostic bound the port to the right process and then
rejected it with `0 mapped Assembly-CSharp.dll images`. That is either a gate
defect (the assembly is loaded in a way `/proc/<pid>/maps` does not show as a
file-backed mapping) or a candidate defect (the running player is not the code
we staged). Only the live map table distinguishes them, so this probe captures
it and stops.

It performs no gameplay: it starts the display and jar, completes the agent
handshake that spawns the player, reads `/proc/<pid>/{exe,cwd,maps}` for the
port owner, and tears everything down. It loads no level and fires no shot, so
it cannot consume the single final full smoke. It writes one JSON receipt and
mutates nothing outside its own output directory.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.smoke_physics_capture import (  # noqa: E402
    _listening_sockets,
    _socket_owner_scan,
    archive_details,
    free_port,
    start_display,
    terminate,
)
from scripts.manual_agent import connect_with_retry  # noqa: E402
from src.webui.bridge import PlayingMode  # noqa: E402


def file_backed_mappings(pid: int) -> list[dict[str, object]]:
    """Return every distinct file-backed mapping of the process, deduplicated."""
    seen: dict[tuple[str, str, int], dict[str, object]] = {}
    for line in (Path("/proc") / str(pid) / "maps").read_text(encoding="utf-8").splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) != 6:
            continue
        pathname = fields[5].strip()
        if not pathname or pathname.startswith("["):
            continue
        key = (pathname, fields[3], int(fields[4]))
        seen.setdefault(key, {"path": pathname, "name": PurePosixPath(pathname).name, "device": fields[3], "inode": int(fields[4])})
    return sorted(seen.values(), key=lambda entry: str(entry["path"]))


def main() -> int:
    """Launch, bind, dump the owner's mappings, tear down, and report."""
    stage = ROOT / "sciencebirdsgames/physics-v1"
    output = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(tempfile.mkdtemp(prefix="novphy-maps-probe-"))
    output.mkdir(parents=True, exist_ok=True)
    receipt: dict[str, object] = {"schema": "novphy_player_mapping_probe_v1", "stage": str(stage)}
    engine = xvfb = bridge = None
    with tempfile.TemporaryDirectory(prefix="novphy-maps-clone-") as clone_dir:
        clone = Path(clone_dir)
        try:
            _archive, archive_sha, player_sha, _protocol_sha, assembly_sha = archive_details(stage, clone)
            receipt.update({"clone": str(clone), "archive_sha256": archive_sha, "staged_player_sha256": player_sha, "staged_assembly_sha256": assembly_sha})
            display, xvfb = start_display(output / "xvfb.log")
            # The physics port is a parameter here because the first probe run
            # passed a random free port through `--physics-port` and never saw a
            # listener, while the smoke -- which used the 2004 default -- bound
            # one immediately. Treat the flag as unproven and drive the port
            # explicitly rather than assuming the player honours it.
            physics_port = int(sys.argv[2]) if len(sys.argv) > 2 else 2004
            agent_port, game_port = free_port(), free_port()
            receipt["ports"] = {"agent": agent_port, "game": game_port, "physics": physics_port}
            with (output / "engine.log").open("wb") as stream:
                engine = subprocess.Popen(
                    ["java", "-jar", "./game_playing_interface.jar", "--agent-port", str(agent_port), "--game-start-port", str(game_port), "--physics-port", str(physics_port), "--dev"],
                    cwd=clone, env={**os.environ, "DISPLAY": display}, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True,
                )
            bridge = connect_with_retry("127.0.0.1", agent_port, timeout=10.0, deadline_seconds=90.0)
            bridge.configure(agent_id=28701, mode=PlayingMode.TRAINING)
            deadline = time.monotonic() + 180.0
            owners: tuple[tuple[int, str], ...] = ()
            while time.monotonic() < deadline:
                sockets = _listening_sockets(physics_port)
                if sockets:
                    scan = _socket_owner_scan(tuple(sock.inode for sock in sockets))
                    if scan.owners:
                        owners = scan.owners
                        break
                time.sleep(1.0)
            receipt["owners"] = [list(owner) for owner in owners]
            if not owners:
                receipt["result"] = "no owner resolved before deadline"
                return 1
            pid = owners[0][0]
            mappings = file_backed_mappings(pid)
            receipt.update({
                "owner_pid": pid,
                "owner_exe": os.readlink(f"/proc/{pid}/exe"),
                "owner_cwd": os.readlink(f"/proc/{pid}/cwd"),
                "mapping_count": len(mappings),
                "mapped_dll_names": sorted({str(entry["name"]) for entry in mappings if str(entry["name"]).endswith(".dll")}),
                "mapped_names": sorted({str(entry["name"]) for entry in mappings}),
                "assembly_csharp_mappings": [entry for entry in mappings if entry["name"] == "Assembly-CSharp.dll"],
                "mappings": mappings,
            })
            receipt["result"] = "captured"
            return 0
        finally:
            if bridge is not None:
                try:
                    bridge.disconnect()
                except OSError as error:
                    receipt["disconnect_error"] = str(error)
            receipt["cleanup"] = {"engine": terminate(engine), "xvfb": terminate(xvfb)}
            (output / "player-mapping-probe.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            print(json.dumps({"result": receipt.get("result"), "owner_pid": receipt.get("owner_pid"), "mapped_dll_count": len(receipt.get("mapped_dll_names") or []), "receipt": str(output / "player-mapping-probe.json")}, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
