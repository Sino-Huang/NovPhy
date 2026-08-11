"""Phase 5: build the committed source twice into isolated non-production stages.

Determinism is the gate. Both builds run the same script against the same commit,
differing only in `NOVPHY_PHYSICS_STAGE`, and every named digest must match. Any
drift ends the wave `still_blocked` — it would mean the archive that gets pinned
cannot be reproduced from the source it claims to come from.

Neither build's output is ever piped. Unity's detached `UnityPackageManager`
daemon inherits stdout and stderr and outlives the editor, so a captured pipe
never reaches EOF and the caller hangs after a build that already succeeded. Both
streams go to a file, and any daemon left pointing at a dead editor is reaped.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "scripts/build_physics_player.sh").is_file():
            return candidate
    raise SystemExit("cannot locate the worktree root")


ROOT = _repo_root()
HERE = Path(__file__).resolve().parent
OUT = HERE / "phase5-builds"
BUILD = ROOT / "scripts/build_physics_player.sh"
COMPARE = HERE / "compare_builds.py"
PRODUCTION = Path("/mnt/array/sukaih/Project/NovPhy/sciencebirdsgames/Linux")
STAGED_PIN = ROOT / "sciencebirdsgames/physics-v1"


def reap_orphan_package_managers() -> list[int]:
    """Kill `UnityPackageManager` daemons whose `-s <pid>` editor no longer exists.

    Keyed on the daemon's own argument, never on the process name: a name match
    would also kill a daemon belonging to a live editor that is not ours.
    """
    reaped: list[int] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            argv = (entry / "cmdline").read_bytes().decode("utf-8", "replace").split("\0")
        except OSError:
            continue
        if not argv or not argv[0].endswith("UnityPackageManager") or "-s" not in argv:
            continue
        target = argv[argv.index("-s") + 1 :][:1]
        if not target or not target[0].isdigit():
            continue
        if Path(f"/proc/{target[0]}").exists():
            continue
        try:
            os.kill(int(entry.name), 15)
            reaped.append(int(entry.name))
        except OSError:
            pass
    return reaped


def run_build(label: str) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    stage = OUT / label
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)

    # Refuse to write anywhere that matters, before invoking anything.
    resolved = stage.resolve()
    if resolved == PRODUCTION.resolve() or resolved == STAGED_PIN.resolve():
        raise SystemExit(f"refusing to build into a protected root: {resolved}")

    log_path = OUT / f"{label}.log"
    env = dict(os.environ)
    env["NOVPHY_PHYSICS_STAGE"] = str(stage)
    with log_path.open("wb") as log:
        completed = subprocess.run(
            ["bash", str(BUILD)], cwd=str(ROOT), env=env,
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            timeout=7200,
        )
    reaped = reap_orphan_package_managers()
    return {"label": label, "stage": str(stage), "exit": completed.returncode,
            "log": log_path.name, "reaped_orphan_package_managers": reaped}


def main() -> int:
    first = run_build("build-a")
    print(json.dumps(first))
    if first["exit"] != 0:
        print(f"FAILURE: build-a exited {first['exit']}; see {first['log']}")
        return 1

    second = run_build("build-b")
    print(json.dumps(second))
    if second["exit"] != 0:
        print(f"FAILURE: build-b exited {second['exit']}; see {second['log']}")
        return 1

    receipt = OUT / "determinism-receipt.json"
    compare = subprocess.run(
        [sys.executable, str(COMPARE), first["stage"], second["stage"], str(receipt)],
        capture_output=True, text=True, timeout=1800,
    )
    print(compare.stdout.strip())
    if compare.stderr.strip():
        print(compare.stderr.strip(), file=sys.stderr)
    (OUT / "phase5-runs.json").write_text(
        json.dumps({"builds": [first, second], "compare_exit": compare.returncode,
                    "receipt": str(receipt.relative_to(HERE))}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    if compare.returncode != 0:
        print("FAILURE: the two builds are not byte-identical; the wave ends still_blocked")
    return compare.returncode


if __name__ == "__main__":
    raise SystemExit(main())
