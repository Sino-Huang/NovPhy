"""Run every EditMode fixture class and collect authoritative NUnit XML.

A single unfiltered `-runTests` invocation is not usable on this host: the editor
races its own CEF shutdown, and on the unfiltered run the crash (SIGSEGV in
`CefBrowserMessageLoop`) landed *before* the result file was flushed, leaving no
XML at all even though every test had executed. Per-class runs win that race
consistently, and each writes its own XML, so coverage stays complete and every
verdict stays anchored to a result file rather than to an exit code.

Editor stdout and stderr go to DEVNULL, never to a pipe: Unity's detached
`UnityPackageManager` daemon inherits both and outlives the editor, so a captured
pipe never reaches EOF and the caller hangs after a correct run.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path


def _repo_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "scripts/build_physics_player.sh").is_file():
            return candidate
    raise SystemExit("cannot locate the worktree root")


ROOT = _repo_root()
HERE = Path(__file__).resolve().parent
OUT = HERE / "editmode-full"
PROJECT = ROOT / "tasks/task_template_designer"

EDITOR = Path(os.environ.get(
    "UNITY_2019_4_41F2",
    str(Path.home() / ".local/share/novphy-unity/2019.4.41f2-6b23d448b533/editor/Editor/Unity"),
))
COMPAT_LIBS = os.environ.get(
    "UNITY_LTS_LIBS",
    "/tmp/opencode/unity-2019.4-libssl1.1/root/usr/lib/x86_64-linux-gnu"
    ":/tmp/opencode/unity-2019.3.4f1-libs/root/usr/lib/x86_64-linux-gnu"
    ":/tmp/opencode/unity-2019.3.4f1-libs/root/usr/lib",
)

CLASSES = [
    "ABBirdLaunchTests",
    "ABGameWorldLifecycleTests",
    "LegacyGroundTruthTests",
    "PhysicalEntityRegistryTests",
    "PhysicalShotRecorderTests",
    "PhysicalSnapshotExporterTests",
    "PhysicsCaptureProtocolTests",
    "PhysicsShotRecorderTests",
]


def reap_package_manager(editor_pid: int) -> list[int]:
    """Kill the daemon this editor spawned, identified by its own `-s <pid>`."""
    reaped: list[int] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            argv = (entry / "cmdline").read_bytes().decode("utf-8", "replace").split("\0")
        except OSError:
            continue
        if not argv or not argv[0].endswith("UnityPackageManager"):
            continue
        if "-s" in argv and argv[argv.index("-s") + 1 :][:1] == [str(editor_pid)]:
            try:
                os.kill(int(entry.name), 15)
                reaped.append(int(entry.name))
            except OSError:
                pass
    return reaped


def run_class(name: str) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    xml_path = OUT / f"{name}.xml"
    log_path = OUT / f"{name}.log"
    if xml_path.exists():
        xml_path.unlink()
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = COMPAT_LIBS + ":" + env.get("LD_LIBRARY_PATH", "")
    editor = subprocess.Popen(
        [str(EDITOR), "-batchmode", "-nographics", "-projectPath", str(PROJECT),
         "-runTests", "-testPlatform", "EditMode", "-testFilter", name,
         "-testResults", str(xml_path), "-logFile", str(log_path)],
        env=env, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        exit_code = editor.wait(timeout=1800)
    finally:
        reaped = reap_package_manager(editor.pid)

    result = {"class": name, "process_exit": exit_code, "editor_pid": editor.pid,
              "reaped_package_managers": reaped, "nunit_xml": xml_path.name}
    if not xml_path.exists():
        result["error"] = "no NUnit XML written; compiler or editor failure before discovery"
        return result
    root = ET.parse(xml_path).getroot()
    result["nunit_xml_sha256"] = hashlib.sha256(xml_path.read_bytes()).hexdigest()
    result["total"] = int(root.get("total", "0"))
    result["passed"] = int(root.get("passed", "0"))
    result["failed"] = int(root.get("failed", "0"))
    result["skipped"] = int(root.get("skipped", "0"))
    result["non_pass"] = [
        {"test": case.get("fullname"), "result": case.get("result")}
        for case in root.iter("test-case") if case.get("result") != "Passed"
    ]
    return result


def main() -> int:
    runs = [run_class(name) for name in CLASSES]
    for run in runs:
        print(json.dumps(run))
    failures = [r["class"] for r in runs if r.get("error") or r.get("failed", 1) != 0]
    receipt = {
        "authority": "per-class NUnit XML under editmode-full/; the editor exit code is not authoritative",
        "why_per_class": "the single unfiltered run crashed in CefBrowserMessageLoop before flushing its result file, producing no XML despite executing every test",
        "editor": {"path": str(EDITOR), "sha256": hashlib.sha256(EDITOR.read_bytes()).hexdigest()},
        "runs": runs,
        "totals": {
            "tests": sum(r.get("total", 0) for r in runs),
            "passed": sum(r.get("passed", 0) for r in runs),
            "failed": sum(r.get("failed", 0) for r in runs),
            "skipped": sum(r.get("skipped", 0) for r in runs),
        },
        "classes_with_failures": failures,
        "verdict": "all_editmode_green" if not failures else "editmode_failures",
    }
    (HERE / "editmode-full.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt["totals"]), receipt["verdict"])
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
