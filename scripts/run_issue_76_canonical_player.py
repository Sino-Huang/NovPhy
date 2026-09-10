"""Prepare and compile a separate recovered reference player; never open a cohort."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
RECOVERED = ROOT / ".local-artifacts/issue-76-canonical-recovery-v1/ExportedProject"
WORK = ROOT / ".local-artifacts/issue-76-canonical-player-v1"
UNITY = Path("/home/sukaih/.local/share/novphy-unity/2019.4.41f2-6b23d448b533/editor/Editor/Unity")
VERSION = "m_EditorVersion: 2019.4.41f2\nm_EditorVersionWithRevision: 2019.4.41f2 (6b23d448b533)\n"
REQUIRED = (
    "Assets/Scripts/Assembly-CSharp/DieOnBirdHitCountPig.cs",
    "Assets/Scripts/Assembly-CSharp/ABGameObject.cs",
    "Assets/Resources/prefabs/gameworld/characters/pigs/BasicBig.prefab",
    "Assets/Resources/prefabs/gameworld/benchmarknovelties/PinkBigPig.prefab",
    "ProjectSettings/EditorBuildSettings.asset",
)


def plan(recovered=RECOVERED, work=WORK):
    for name in REQUIRED:
        if not (recovered / name).is_file():
            raise ValueError(f"incomplete reference recovery: {name}")
    return {"schema": "issue_76_reference_player_preparation_v1",
            "recovered": str(recovered), "work": str(work),
            "source_player": "sciencebirdsgames/Linux",
            "asset_recovery_tool": "AssetRipper 2.0.0",
            "source_unity_version": "2019.3.4f1", "build_unity_version": "2019.4.41f2",
            "version_change_requires_runtime_validation": True,
            "graphics_captures": 0, "fresh_access": False,
            "canonical_equivalence_established": False,
            "recovered_bundle_resource_root": "retained outside Assets; original StreamingAssets bundles retained in player",
            "required_recovered_assets": list(REQUIRED)}


def prepare(recovered=RECOVERED, work=WORK):
    contract = plan(recovered, work)
    if work.exists():
        raise ValueError("working player already exists; do not overwrite its engineering evidence")
    project = work / "project"
    print("[canonical-player] copying recovered project; original and C1 remain untouched", flush=True)
    shutil.copytree(recovered, project)
    retain_bundle_exports(project)
    (project / "ProjectSettings/ProjectVersion.txt").write_text(VERSION)
    manifest = project / "Packages/manifest.json"
    packages = json.loads(manifest.read_text())
    packages["dependencies"]["com.unity.test-framework"] = "1.1.31"
    manifest.write_text(json.dumps(packages, indent=2) + "\n")
    editor = project / "Assets/Editor"
    editor.mkdir(exist_ok=True)
    shutil.copy2(ROOT / "tasks/task_template_designer/Assets/Scripts/Editor/NovPhyBuild.cs", editor)
    shutil.copy2(ROOT / "tasks/issue_76_canonical/CanonicalPigAssetTests.cs", editor)
    (work / "preparation.json").write_text(json.dumps(contract, indent=2) + "\n")
    print(f"[canonical-player] prepared {project}", flush=True)


def retain_bundle_exports(project):
    # AssetRipper also exports StreamingAssets bundles under lowercase resources.
    # Unity ignores BOTH case-colliding roots. Keep those derived exports outside
    # the build; the unchanged original runtime bundles remain in StreamingAssets.
    bundle_root = project / "Assets/resources"
    if bundle_root.exists():
        bundle_root.rename(project.parent / "recovered-bundle-exports")
        metadata = project / "Assets/resources.meta"
        if metadata.exists():
            metadata.rename(project.parent / "recovered-bundle-exports.meta")


def run_editor(mode, work=WORK, unity=UNITY):
    if not (work / "preparation.json").exists():
        raise ValueError("run --prepare before invoking the editor")
    attempt = 1 + len(list(work.glob(f"{mode}-*.log")))
    editor = work / "project/Assets/Editor"
    shutil.copy2(ROOT / "tasks/issue_76_canonical/CanonicalPigAssetTests.cs", editor)
    instrumented = (work / "instrumentation.json").exists()
    if instrumented:
        shutil.copy2(ROOT / "tasks/issue_76_canonical/CanonicalPigBehaviorTests.cs", editor)
    log = work / f"{mode}-{attempt:02}.log"
    command = [str(unity), "-batchmode", "-projectPath", str(work / "project"), "-logFile", str(log)]
    if mode == "test":
        command += ["-runTests", "-testPlatform", "EditMode", "-testFilter", "CanonicalPig",
                    "-testResults", str(work / f"tests-{attempt:02}.xml")]
    else:
        command += ["-quit", "-executeMethod", "NovPhyBuild.BuildPhysicsLinux"]
    environment = dict(os.environ, NOVPHY_BUILD_OUTPUT=str(work / f"player-build-{attempt:02}/9001.x86_64"))
    # env.sh redirects game data, but the editor license is in the user's normal
    # XDG location. Do not make the editor activate a second repository-local seat.
    environment.pop("XDG_DATA_HOME", None)
    environment["DOTNET_SYSTEM_GLOBALIZATION_INVARIANT"] = "1"
    from scripts.smoke_physics_capture import start_display
    display, display_process = start_display(work / f"display-{mode}-{attempt:02}.log")
    environment["DISPLAY"] = display
    print(f"[canonical-player] {mode} starting; detailed log: {log}", flush=True)
    started = time.monotonic()
    try:
        with subprocess.Popen(command, env=environment) as process:
            while process.poll() is None:
                elapsed = time.monotonic() - started
                print(f"[canonical-player] {mode} elapsed={elapsed:.0f}s limit=1800s", flush=True)
                if elapsed >= 1800:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                    raise TimeoutError("editor engineering command exceeded 30 minutes; log retained")
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    pass
    finally:
        display_process.terminate()
        display_process.wait(timeout=10)
    print(f"[canonical-player] {mode} finished exit={process.returncode} elapsed={time.monotonic()-started:.1f}s", flush=True)
    if process.returncode:
        raise RuntimeError(f"editor failed; inspect {log}")
    if mode == "test":
        import xml.etree.ElementTree as ET
        result = ET.parse(work / f"tests-{attempt:02}.xml").getroot()
        if int(result.attrib.get("passed", "0")) < (5 if instrumented else 3) or int(result.attrib.get("failed", "0")):
            raise ValueError("canonical asset fixtures did not all pass")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    for mode in ("dry-run", "prepare", "instrument", "test", "build"):
        modes.add_argument(f"--{mode}", action="store_true")
    parser.add_argument("--work-dir", type=Path, default=WORK,
                        help="isolated working/output directory; use a native filesystem for Unity asset import")
    args = parser.parse_args()
    work = args.work_dir.resolve()
    if args.dry_run:
        print(json.dumps(plan(work=work), indent=2))
    elif args.prepare:
        prepare(work=work)
    elif args.instrument:
        from scripts.issue_76_canonical_instrumentation import instrument
        instrument(work / "project")
    else:
        run_editor("test" if args.test else "build", work=work)


if __name__ == "__main__":
    main()
