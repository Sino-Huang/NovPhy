"""Build an isolated RGB-history/barrier player; never alter a frozen parent."""
import argparse
import json
from pathlib import Path
import shutil

from scripts import run_issue_76_canonical_player as base
from scripts.issue_76_canonical_instrumentation import replace

PARENT = Path("/home/sukaih/.cache/novphy-canonical-native-v1")
WORK = Path("/home/sukaih/.cache/novphy-shared-history-v1")
BARRIER = "tasks/issue_76_canonical/NativeDecisionBarrier.cs"


def runtime_source(text):
    text = replace(text, '        v2InterventionObserved = false;\n    }',
                   '        v2InterventionObserved = false;\n        NativeDecisionBarrier.Reset(this);\n    }')
    text = replace(text, '    public void BeginV2Shot()\n    {',
                   '    public void BeginV2Shot()\n    {\n        NativeDecisionBarrier.RequireReady(this);')
    return replace(text, '        CaptureNativeObservation();\n    }\n\n    public PhysicalCaptureResult',
                   '        CaptureNativeObservation();\n        NativeDecisionBarrier.ReleaseAfterShotSnapshot(this);\n    }\n\n    public PhysicalCaptureResult')


def world_source(text):
    text = replace(text, '\tprivate void FixedUpdate()\n\t{',
                   '\tprivate void FixedUpdate()\n\t{\n\t\tif (NativeDecisionBarrier.Paused) return;')
    return replace(text, '\t\t\tManageBirds();\n\t\t\tTakeAction();',
                   '\t\t\tManageBirds();\n\t\t\tTakeAction();\n\t\t\tNativeDecisionBarrier.AfterBookkeeping(runtime);')


def connection_source(text):
    # Public readiness requests must not unpause physics. Scene navigation cancels
    # the old barrier, while each shot releases it after its initial RGB snapshot.
    text = text.replace('Time.timeScale = 1f;', 'Time.timeScale = NativeDecisionBarrier.Paused ? 0f : 1f;')
    for name in ('SelectNextAvailableLevel', 'SelectLevel'):
        marker = f'\tprivate IEnumerator {name}(JSONNode data)\n\t{{'
        text = replace(text, marker, marker + '\n\t\tNativeDecisionBarrier.Cancel();')
    return text


def prepare(parent=PARENT, work=WORK):
    if work.exists():
        raise ValueError("shared-history work already exists; preserve its source and evidence")
    changes = {}
    for relative, transform in (
        ('Assets/Scripts/CanonicalCapture/PhysicalSnapshotRuntime.cs', runtime_source),
        ('Assets/Scripts/Assembly-CSharp/ABGameWorld.cs', world_source),
        ('Assets/Scripts/Assembly-CSharp/AIBirdsConnection.cs', connection_source),
    ):
        before = (parent / 'project' / relative).read_text()
        changes[relative] = {'before': before, 'after': transform(before)}
    # Include the valid import cache to avoid repeating a full recovered-asset
    # import. These are private copies, not links into the parent project.
    shutil.copytree(parent / 'project', work / 'project',
                    ignore=lambda directory, names: [n for n in ('Temp', 'Logs') if n in names])
    for name in ('preparation.json', 'instrumentation.json', 'shader-restoration.json'):
        shutil.copy2(parent / name, work / name)
    for relative, value in changes.items():
        (work / 'project' / relative).write_text(value['after'])
    shutil.copy2(base.ROOT / BARRIER, work / 'project/Assets/Scripts/CanonicalCapture')
    (work / 'shared-history-source.json').write_text(json.dumps({
        'schema': 'issue_76_shared_history_player_source_v1', 'parent': str(parent),
        'changes': changes, 'barrier_source': (base.ROOT / BARRIER).read_text(),
        'canonical_physics_and_gameplay_rules_changed': False,
        'agent_input': 'RGB, observation time, accepted past actions only',
        'fresh_access': False, 'advancement_authorized': False,
    }, indent=2) + '\n')
    print(f"Shared-history source prepared at {work}; no captures started.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--prepare', action='store_true')
    modes.add_argument('--build', action='store_true')
    parser.add_argument('--work-dir', type=Path, default=WORK)
    args = parser.parse_args()
    if args.prepare:
        prepare(work=args.work_dir)
    else:
        base.run_editor('build', work=args.work_dir)


if __name__ == '__main__':
    main()
