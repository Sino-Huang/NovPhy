"""Prove the new EditMode fixture bites the exact defect the staged player has.

The 2026-08-06 staged player (`git_head e2d19ae`) hardcoded an empty collision
payload, so request-70 serialized `"payload":{}` — a value that matches no branch
of the frozen `event_payload` oneOf, and which the Python consumer rejects. The
fix is already committed on this branch; what was missing was a fixture at the
recorder -> wire seam that fails when the payload is empty.

This harness injects an empty collision payload at the point the payload is
built, runs the fixture, and requires it RED; then restores the source
byte-identically and requires it GREEN. Note the mutation reproduces the staged
player's *observable wire output*, not its mechanism: the staged emitter went
empty upstream, in `PhysicalSnapshotRuntime`, by querying a store that was not
yet populated when the callback fired. Both end in `"payload":{}`, which is what
this fixture exists to catch, but the mutation is a wire-level stand-in.

NUnit XML is the authority — every EditMode invocation on this host exits from a
signal inside CEF shutdown, reached after the run finished and after the XML was
written.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def _repo_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "scripts/build_physics_player.sh").is_file():
            return candidate
    raise SystemExit("cannot locate the worktree root")


ROOT = _repo_root()
HERE = Path(__file__).resolve().parent
SOURCE = ROOT / "tasks/task_template_designer/Assets/Scripts/GroundTruth/PhysicsShotRecorder.cs"
PROJECT = ROOT / "tasks/task_template_designer"
BASELINE = "0987580887fb07603d1b4174effbdfc83c6cfef8b8554d4d8eaed7934cc879d6"

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

TEST_CLASS = "PhysicsCaptureProtocolTests"
TEST_NAME = "Request70CollisionPayloadCarriesTheContactEvidenceTheContractRequires"

# The staged defect's observable wire output: evidence is computed and validated,
# then discarded at the point the payload is built. See the module docstring —
# the staged player reached the same empty payload by a different route.
ANCHOR = "new PhysicalMacroEventPayload(contactIds: evidence, relativeSpeed: relativeSpeed)"
MUTATION = "new PhysicalMacroEventPayload(contactIds: new string[0], relativeSpeed: 0f)"

OUT = HERE / "collision-payload-redgreen"


def reap_package_manager(editor_pid: int) -> list[int]:
    """Kill the `UnityPackageManager` daemon this editor invocation spawned.

    It is identified by its own `-s <editor pid>` argument, never by name alone:
    a name match would also hit a daemon belonging to somebody else's editor.
    """
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


def run_editmode(label: str) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    xml_path = OUT / f"{label}.xml"
    log_path = OUT / f"{label}.log"
    if xml_path.exists():
        xml_path.unlink()
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = COMPAT_LIBS + ":" + env.get("LD_LIBRARY_PATH", "")
    # Never capture the editor's stdout/stderr. Unity spawns a detached
    # `UnityPackageManager` daemon that inherits both descriptors and outlives
    # the editor, so a captured pipe never reaches EOF: subprocess.run blocks in
    # communicate() forever while its editor child sits defunct, and the harness
    # hangs *after* a correct run, holding the mutated source in the tree. The
    # editor already writes everything to -logFile, so the pipes bought nothing.
    editor = subprocess.Popen(
        [str(EDITOR), "-batchmode", "-nographics", "-projectPath", str(PROJECT),
         "-runTests", "-testPlatform", "EditMode", "-testFilter", TEST_CLASS,
         "-testResults", str(xml_path), "-logFile", str(log_path)],
        env=env, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        exit_code = editor.wait(timeout=3600)
    except subprocess.TimeoutExpired:
        # Never leave a batchmode editor alive: it holds Temp/UnityLockfile on the
        # project, and every later editor invocation on it then fails.
        editor.kill()
        editor.wait()
        raise
    finally:
        reaped = reap_package_manager(editor.pid)
    result = {"label": label, "process_exit": exit_code, "editor_pid": editor.pid,
              "reaped_package_managers": reaped,
              "nunit_xml": str(xml_path.relative_to(HERE)), "log": str(log_path.relative_to(HERE))}
    if not xml_path.exists():
        result["error"] = "no NUnit XML was written; treat as a compiler or editor failure before discovery"
        return result
    root = ET.parse(xml_path).getroot()
    result["nunit_xml_sha256"] = hashlib.sha256(xml_path.read_bytes()).hexdigest()
    result["suite_total"] = int(root.get("total", "0"))
    result["suite_failed"] = int(root.get("failed", "0"))
    result["suite_passed"] = int(root.get("passed", "0"))
    for case in root.iter("test-case"):
        if case.get("name") == TEST_NAME or (case.get("name") or "").endswith("." + TEST_NAME):
            result["fixture_result"] = case.get("result")
            message = case.find(".//message")
            if message is not None and message.text:
                result["fixture_message"] = message.text.strip()[:1200]
            break
    else:
        result["error"] = f"fixture {TEST_NAME} was not discovered"
    return result


def main() -> int:
    original = SOURCE.read_text(encoding="utf-8")
    if hashlib.sha256(original.encode("utf-8")).hexdigest() != BASELINE:
        print("recorder digest does not match the recorded baseline; refusing to mutate")
        return 2
    if original.count(ANCHOR) != 1:
        print(f"mutation anchor matched {original.count(ANCHOR)} times; refusing to mutate")
        return 2

    failures: list[str] = []
    try:
        SOURCE.write_text(original.replace(ANCHOR, MUTATION), encoding="utf-8")
        red = run_editmode("red-staged-emitter")
        print(f"RED pass: {json.dumps(red)}")
        if red.get("fixture_result") != "Failed":
            failures.append(f"fixture did not fail against the staged emitter: {red.get('fixture_result')}")
    finally:
        SOURCE.write_text(original, encoding="utf-8")

    restored = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    print(f"restored sha256: {restored} ({'identical' if restored == BASELINE else 'DRIFTED'})")
    if restored != BASELINE:
        failures.append("recorder source was not restored byte-identically")
        red_green = {"restored": restored}
    else:
        green = run_editmode("green-head")
        print(f"GREEN pass: {json.dumps(green)}")
        if green.get("fixture_result") != "Passed":
            failures.append(f"fixture did not pass at HEAD: {green.get('fixture_result')}")
        if green.get("suite_failed", 1) != 0:
            failures.append(f"{green.get('suite_failed')} tests failed in {TEST_CLASS} at HEAD")
        red_green = {"red": red, "green": green, "restored": restored}

    receipt = {
        "authority": "NUnit XML result attributes; the editor process exit code is not authoritative",
        "fixture": f"{TEST_CLASS}.{TEST_NAME}",
        "source": str(SOURCE.relative_to(ROOT)),
        "baseline_sha256": BASELINE,
        "mutation": {"anchor": ANCHOR, "replacement": MUTATION,
                     "reproduces": "the e2d19ae staged emitter's empty collision payload"},
        "passes": red_green,
        "failures": failures,
        "verdict": "red_then_green" if not failures else "inconclusive",
    }
    (HERE / "collision-payload-redgreen.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for failure in failures:
        print(f"FAILURE: {failure}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
