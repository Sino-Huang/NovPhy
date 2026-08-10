"""Prove each new gate assertion bites, by breaking it and requiring a red suite.

A test that passes against both the fixed and the broken source pins nothing.
Each mutation below is the specific defect the corresponding remediation closed;
the source is restored and digest-checked afterwards.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts/smoke_physics_capture.py"
BASELINE = "f61ccfa52ef5bed5e156e8480f32f407c13286276df00d8ec676e9671fc84978"

MUTATIONS: tuple[tuple[str, str, str, str], ...] = (
    (
        "process-tree-direct-children-only",
        "MAJOR 7: transitive descendants",
        """    tree = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, parent in parents.items():
            if parent in tree and pid not in tree:
                tree.add(pid)""",
        """    tree = {root_pid}
    changed = False
    if changed:
        for pid, parent in parents.items():
            if parent in tree and pid not in tree:
                tree.add(pid)""",
    ),
    (
        "process-view-uses-host-path",
        "MAJOR 5: read through the observed root",
        '    return Path("/proc") / str(pid) / "root" / path.relative_to("/")',
        "    return path",
    ),
    (
        "terminate-guarded-on-the-leader",
        "MAJOR 1: group-keyed teardown",
        """    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass""",
        """    try:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass""",
    ),
    (
        "terminate-never-escalates",
        "MAJOR 1: SIGKILL escalation on the group",
        """                os.killpg(process.pid, signal.SIGKILL)
                escalated = True""",
        """                escalated = True""",
    ),
    (
        "launch-environment-passes-everything-through",
        "MAJOR 4: interposition refusal",
        """    present = sorted(name for name in INTERPOSITION_ENV_VARS if environ.get(name))
    if present:
        raise SmokeError(f"refusing to launch with code-interposition environment set: {', '.join(present)}")""",
        """    present = ()""",
    ),
    (
        "no-owner-reconfirmation",
        "MAJOR 3: starttime + descriptor re-confirmation",
        """        if starttime_after != starttime_before:
            raise ListenerBindingError(f"port listener owner pid={pid} was replaced during observation")
        if socket_inode not in held_inodes:
            raise ListenerBindingError(f"port listener owner pid={pid} no longer holds socket inode {socket_inode}")""",
        """        del starttime_after, held_inodes""",
    ),
    (
        "wildcard-bind-accepted",
        "loopback-only enforcement",
        """            foreign = sorted({entry.local_address for entry in sockets if entry.local_address.rsplit(":", 1)[0].upper() not in LOOPBACK_BIND_ADDRESSES})
            if foreign:
                raise ListenerBindingError(f"port-{physics_port} is bound to a non-loopback address: {foreign}")""",
        """            foreign = ()""",
    ),
    (
        "unchecked-observation-fields-tolerated",
        "field-completeness guard",
        """    if uncovered:
        raise SmokeError(f"listener observation fields are neither checked nor declared unchecked: {', '.join(uncovered)}")""",
        """    del uncovered""",
    ),
)


def main() -> int:
    original = SOURCE.read_text(encoding="utf-8")
    if hashlib.sha256(original.encode("utf-8")).hexdigest() != BASELINE:
        print("source digest does not match the recorded baseline; refusing to mutate")
        return 2
    failures: list[str] = []
    try:
        for name, closes, before, after in MUTATIONS:
            if original.count(before) != 1:
                failures.append(f"{name}: anchor matched {original.count(before)} times")
                continue
            SOURCE.write_text(original.replace(before, after), encoding="utf-8")
            result = subprocess.run(
                [sys.executable, "-m", "unittest", "tests.test_smoke_physics_capture"],
                cwd=ROOT, capture_output=True, text=True, timeout=900,
            )
            verdict = "RED (assertion bites)" if result.returncode != 0 else "GREEN (NOT COVERED)"
            print(f"{name}: {verdict}  [{closes}]")
            if result.returncode == 0:
                failures.append(f"{name}: suite stayed green under mutation")
    finally:
        SOURCE.write_text(original, encoding="utf-8")
    restored = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    print(f"restored sha256: {restored} ({'identical' if restored == BASELINE else 'DRIFTED'})")
    if restored != BASELINE:
        failures.append("source was not restored byte-identically")
    for failure in failures:
        print(f"FAILURE: {failure}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
