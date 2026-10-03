"""Build the #104 capture player: the shared-history-v3 player plus the #113 window rule.

Two source changes, nothing else (physics, gameplay, rendering untouched):

* ``CanonicalNativeTrace.MaximumShotSteps`` becomes environment-configurable
  (``NOVPHY_NATIVE_MAX_SHOT_STEPS``, default 30000 = the old constant) and a new
  ``RestTailSteps`` (``NOVPHY_NATIVE_REST_TAIL_STEPS``, default 0) is added; both are
  written into ``native-manifest.json`` (``maximum_shot_steps``, ``rest_tail_steps``).
* ``PhysicalSnapshotRuntime``: with a positive tail, ``stable_entered`` at a step no
  later than the cap no longer finalizes the shot; recording continues for the tail and
  then finalizes with terminal reason ``rest_tail_complete``. ``stable_exited`` cancels
  the tail. A shot that is not in a tail when the cap is reached is censored there, as
  before. With both defaults the player behaves exactly like its parent.

The parent project (``~/.cache/novphy-shared-history-v3``) is never modified; the work
copy includes its import cache so a build is an incremental script compile.
"""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import shutil

from scripts import run_issue_76_canonical_player as base
from scripts.issue_76_canonical_instrumentation import replace

PARENT = Path.home() / ".cache/novphy-shared-history-v3"
WORK = Path.home() / ".cache/novphy-window-player-v1"
UNITY = Path.home() / ".local/share/novphy-unity/2019.4.41f2-6b23d448b533/editor/Editor/Unity"
PARENT_ASSEMBLY_SHA256 = "82db3f42011f1b6768f08f2bf554f34a6573043ea96e66fbd94b7cc086338250"
REFERENCE_PLAYER = base.ROOT / ".local-artifacts/issue-109-capture-smoke-v1/player"
REBUILT_FILES = ("9001_Data/Managed/Assembly-CSharp.dll", "9001_Data/globalgamemanagers")
TRACE = "Assets/Scripts/CanonicalCapture/CanonicalNativeTrace.cs"
RUNTIME = "Assets/Scripts/CanonicalCapture/PhysicalSnapshotRuntime.cs"
MAX_STEPS_VARIABLE = "NOVPHY_NATIVE_MAX_SHOT_STEPS"
TAIL_VARIABLE = "NOVPHY_NATIVE_REST_TAIL_STEPS"
TAIL_TERMINAL = "rest_tail_complete"


def trace_source(text):
    text = replace(text, "    public const long MaximumShotSteps = 30000;\n", f'''    public const long DefaultMaximumShotSteps = 30000;
    public const string MaximumShotStepsVariable = "{MAX_STEPS_VARIABLE}";
    public const string RestTailStepsVariable = "{TAIL_VARIABLE}";
    public const string RestTailTerminalReason = "{TAIL_TERMINAL}";
    public static readonly long MaximumShotSteps = StepsFromEnvironment(MaximumShotStepsVariable, DefaultMaximumShotSteps, ObservationStride, 150000);
    public static readonly long RestTailSteps = StepsFromEnvironment(RestTailStepsVariable, 0, 0, 30000);
''')
    method = '''    // Window lengths are frozen per campaign in its environment; an unparsable,
    // off-grid or out-of-range value fails the player instead of guessing.
    private static long StepsFromEnvironment(string name, long fallback, long minimum, long maximum)
    {
        string text = Environment.GetEnvironmentVariable(name);
        if (string.IsNullOrEmpty(text)) return fallback;
        long value;
        if (!long.TryParse(text, NumberStyles.None, CultureInfo.InvariantCulture, out value)
            || value < minimum || value > maximum || value % ObservationStride != 0)
            throw new InvalidOperationException(name + " must be a multiple of 50 in [" + minimum + ", " + maximum + "]");
        return value;
    }

'''
    text = replace(text, "    public CanonicalNativeTrace(string root, string captureId, long firstStep)\n",
                   method + "    public CanonicalNativeTrace(string root, string captureId, long firstStep)\n")
    return replace(text, '            .Append(",\\"fixed_delta_seconds\\":0.0004,\\"observation_stride\\":50")\n',
                   '            .Append(",\\"fixed_delta_seconds\\":0.0004,\\"observation_stride\\":50")\n'
                   '            .Append(",\\"maximum_shot_steps\\":").Append(MaximumShotSteps)\n'
                   '            .Append(",\\"rest_tail_steps\\":").Append(RestTailSteps)\n')


def runtime_source(text):
    text = replace(text, "    private long nativeLastRenderedStep = -1;\n",
                   "    private long nativeLastRenderedStep = -1;\n    private long nativeRestTailEndStep = -1;\n")
    text = replace(text, "        nativeFirstStep = Clock.FixedStep;\n        nativeLastRenderedStep = -1;\n",
                   "        nativeFirstStep = Clock.FixedStep;\n        nativeLastRenderedStep = -1;\n"
                   "        nativeRestTailEndStep = -1;\n")
    text = replace(text, '''        ObserveV2Stability();
        if (nativeTrace != null && !nativeTrace.Closed)
        {
            if (Clock.FixedStep - nativeFirstStep >= CanonicalNativeTrace.MaximumShotSteps)
''', '''        ObserveV2Stability();
        if (v2Recorder.IsFinalized) return;
        if (nativeRestTailEndStep >= 0 && Clock.FixedStep >= nativeRestTailEndStep)
        {
            FinalizeV2(CanonicalNativeTrace.RestTailTerminalReason);
            return;
        }
        if (nativeTrace != null && !nativeTrace.Closed)
        {
            if (Clock.FixedStep - nativeFirstStep >= CanonicalNativeTrace.MaximumShotSteps && nativeRestTailEndStep < 0)
''')
    return replace(text, '''            if (v2StabilityCandidate && !levelClearPending)
                FinalizeV2("stable_entered");
''', '''            if (!v2StabilityCandidate)
                nativeRestTailEndStep = -1;
            else if (!levelClearPending)
            {
                if (CanonicalNativeTrace.RestTailSteps > 0 && nativeTrace != null
                    && Clock.FixedStep - nativeFirstStep <= CanonicalNativeTrace.MaximumShotSteps)
                    nativeRestTailEndStep = Clock.FixedStep + CanonicalNativeTrace.RestTailSteps;
                else
                    FinalizeV2("stable_entered");
            }
''')


def prepare(parent=PARENT, work=WORK):
    if work.exists():
        raise ValueError("window-player work already exists; preserve its source and evidence")
    builds = sorted(parent.glob("player-build-*"))
    if not builds or assembly_sha256(builds[-1]) != PARENT_ASSEMBLY_SHA256:
        raise ValueError("parent project is not the source of the campaign player 82db3f42")
    changes = {}
    for relative, transform in ((TRACE, trace_source), (RUNTIME, runtime_source)):
        before = (parent / "project" / relative).read_text()
        changes[relative] = {"before": before, "after": transform(before)}
    shutil.copytree(parent / "project", work / "project",
                    ignore=lambda directory, names: [n for n in ("Temp", "Logs") if n in names])
    for name in ("preparation.json", "instrumentation.json", "shader-restoration.json"):
        shutil.copy2(parent / name, work / name)
    for relative, value in changes.items():
        (work / "project" / relative).write_text(value["after"])
    (work / "window-player-source.json").write_text(json.dumps({
        "schema": "issue_104_window_player_source_v1", "parent": str(parent),
        "parent_assembly_sha256": PARENT_ASSEMBLY_SHA256, "changes": changes,
        "environment": {MAX_STEPS_VARIABLE: "default 30000", TAIL_VARIABLE: "default 0"},
        "defaults_reproduce_parent_behavior": True, "physics_and_gameplay_rules_changed": False,
        "spec": ".local-artifacts/issue-113-capture-window-v1/spec.json",
    }, indent=2) + "\n")
    print(f"[window-player] prepared {work}; no captures started", flush=True)


def assembly_sha256(build):
    return sha256((build / "9001_Data/Managed/Assembly-CSharp.dll").read_bytes()).hexdigest()


def file_hashes(root):
    return {str(path.relative_to(root)): sha256(path.read_bytes()).hexdigest()
            for path in sorted(root.rglob("*")) if path.is_file()}


def assemble(destination, reference=REFERENCE_PLAYER, parent=PARENT, work=WORK):
    """Campaign player = the #109 campaign player with exactly the files the rebuild changed.

    The raw parent and window builds are compared file by file; only the managed assembly
    and the build-settings blob may differ. Those two files replace the reference player's.
    """
    destination = Path(destination)
    if destination.exists():
        raise ValueError(f"{destination} exists; a campaign player is never overwritten")
    old_build, new_build = sorted(parent.glob("player-build-*"))[-1], sorted(work.glob("player-build-*"))[-1]
    old, new = file_hashes(old_build), file_hashes(new_build)
    if set(old) != set(new):
        raise ValueError("the window build has a different file set than its parent build")
    changed = sorted(name for name in old if old[name] != new[name])
    if set(changed) != set(REBUILT_FILES):
        raise ValueError(f"unexpected rebuilt files: {changed}")
    if assembly_sha256(reference) != PARENT_ASSEMBLY_SHA256:
        raise ValueError("reference campaign player is not 82db3f42")
    shutil.copytree(reference, destination, symlinks=True)
    for name in changed:
        shutil.copy2(new_build / name, destination / name)
    identity = {"schema": "issue_104_campaign_player_v1", "reference_player": str(reference),
                "reference_assembly_sha256": PARENT_ASSEMBLY_SHA256, "window_build": str(new_build),
                "replaced_files": {name: {"before": old[name], "after": new[name]} for name in changed},
                "assembly_sha256": assembly_sha256(destination)}
    (destination.parent / "player-identity.json").write_text(json.dumps(identity, indent=2) + "\n")
    return identity


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    modes = parser.add_mutually_exclusive_group(required=True)
    for mode in ("prepare", "test", "build"):
        modes.add_argument(f"--{mode}", action="store_true")
    modes.add_argument("--assemble", type=Path, metavar="DESTINATION",
                       help="write a campaign player directory (reference player + rebuilt files)")
    parser.add_argument("--work-dir", type=Path, default=WORK)
    args = parser.parse_args()
    if args.prepare:
        prepare(work=args.work_dir)
    elif args.assemble:
        print(json.dumps(assemble(args.assemble, work=args.work_dir), indent=2), flush=True)
    else:
        base.run_editor("test" if args.test else "build", work=args.work_dir, unity=UNITY)


if __name__ == "__main__":
    main()
