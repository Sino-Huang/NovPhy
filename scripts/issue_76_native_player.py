"""Prepare a separate native-time streaming player; preserve the C2 freeze."""
import argparse
import json
from pathlib import Path
import shutil

from scripts import run_issue_76_canonical_player as base
from scripts.issue_76_canonical_instrumentation import replace

WORK = Path("/home/sukaih/.cache/novphy-canonical-native-v1")
PARENT = Path("/home/sukaih/.cache/novphy-canonical-player-v2")
ROOT = base.ROOT


def native_protocol(text):
    sample_start = text.index("        for (int i = 0; i < snapshot.FixedStepSamples.Count; i++)")
    sample_end = text.index('        json.Append("],\\\"minimum_contact_separation', sample_start)
    event_start = text.index("        for (int eventIndex = 0; eventIndex < snapshot.Events.Count; eventIndex++)")
    event_end = text.index("        if (snapshot.Events.Count == 0)", event_start)
    method = '''    public static string BuildNativeChunkJson(PhysicsCaptureV2EngineSnapshot snapshot)
    {
        StringBuilder json = new StringBuilder("{\\"schema\\":\\"canonical_native_chunk_v1\\",\\"capture_id\\":");
        AppendString(json, snapshot.CaptureId);
        json.Append(",\\"shot_id\\":"); AppendString(json, snapshot.ShotId);
        json.Append(",\\"fixed_step_samples\\":[");
''' + text[sample_start:sample_end] + '''        json.Append("],\\"events\\":[");
''' + text[event_start:event_end] + '''        return json.Append("]}").ToString();
    }

'''
    text = replace(text, "    private static void AppendEntity(", method + "    private static void AppendEntity(")
    start = text.index("    public static byte[] BuildCaptureEnvelope()")
    end = text.index("    public static byte[] BuildCaptureEnvelope(PhysicsCaptureV2EngineSnapshot", start)
    # A drained native trace is NOT a monolithic legacy request-71 capture.
    return text[:start] + '''    public static byte[] BuildCaptureEnvelope()
    {
        return BuildFailureEnvelope(PhysicsCaptureV2EngineFailureCode.CaptureUnavailable,
            "native streaming profile: read its native-manifest.json, not request 71");
    }

''' + text[end:]


def native_recorder(text):
    text = replace(text, "    private int stride;", "    private int nativeEventOrdinal;\n    private int stride;")
    text = replace(text, "        events.Clear();", "        events.Clear();\n        nativeEventOrdinal = 0;")
    text = replace(text, "        int ordinal = events.Count;", "        int ordinal = nativeEventOrdinal++;")
    text = replace(text, "        AddSample(fixedStep, causalObjects, contacts, completeContactEnumeration);\n        if ((fixedStep",
                   "        AddSample(fixedStep, causalObjects, contacts, completeContactEnumeration);\n        if (Failure != null) return;\n        if ((fixedStep")
    method = '''    public int NativeBufferedSampleCount { get { return fixedStepSamples.Count; } }

    public PhysicsCaptureV2EngineSnapshot NativeMetadataSnapshot()
    {
        return new PhysicsCaptureV2EngineSnapshot(captureId, shotId, stride, preInterventionFixedStep,
            terminalFixedStep, new PhysicsCaptureV2FixedStepSample[0], frameRecords,
            new PhysicsCaptureV2EventSnapshot[0], terminalReason, terminalEventId);
    }

    public PhysicsCaptureV2EngineSnapshot NativeChunkSnapshot(bool final)
    {
        int count = final ? fixedStepSamples.Count : Math.Min(CanonicalNativeTrace.ChunkSamples, fixedStepSamples.Count - 1);
        if (count <= 0) return null;
        long end = fixedStepSamples[count - 1].FixedStep;
        return new PhysicsCaptureV2EngineSnapshot(captureId, shotId, stride, preInterventionFixedStep,
            terminalFixedStep, fixedStepSamples.GetRange(0, count), new PhysicsCaptureV2FrameRecord[0],
            events.FindAll(item => item.FixedStep <= end), terminalReason, terminalEventId);
    }

    public void CommitNativeChunk(int count, long throughStep)
    {
        fixedStepSamples.RemoveRange(0, count);
        events.RemoveAll(item => item.FixedStep <= throughStep);
    }

'''
    return replace(text, "    public PhysicsCaptureV2EngineSnapshot CreateFinalizedSnapshot()", method + "    public PhysicsCaptureV2EngineSnapshot CreateFinalizedSnapshot()")


def native_runtime(text):
    text = replace(text, "    private PhysicsCaptureV2FixedStepRecorder v2Recorder;",
                   "    private CanonicalNativeTrace nativeTrace;\n    private long nativeFirstStep;\n"
                   "    private long nativeLastRenderedStep = -1;\n    private PhysicsCaptureV2FixedStepRecorder v2Recorder;")
    text = replace(text, "        registry.ResetLevel();", '''        if (nativeTrace != null && !nativeTrace.Closed && v2Recorder != null)
            nativeTrace.Finish(v2Recorder, "level_reset_before_terminal");
        nativeTrace = null;
        registry.ResetLevel();''')
    text = replace(text, "        v2Recorder.BeginPreInterventionFromUnity(Clock.FixedStep, V2CausalObjects());", '''        if (configuredStride != CanonicalNativeTrace.ObservationStride)
            throw new InvalidOperationException("native streaming requires observation stride 50");
        v2Recorder.BeginPreInterventionFromUnity(Clock.FixedStep, V2CausalObjects());
        nativeFirstStep = Clock.FixedStep;
        nativeLastRenderedStep = -1;
        string root = Environment.GetEnvironmentVariable(PhysicsCaptureV2AlignedObservationRecorder.RootEnvironmentVariable);
        if (!string.IsNullOrEmpty(root))
            nativeTrace = new CanonicalNativeTrace(System.IO.Path.Combine(root, v2Recorder.CaptureId), v2Recorder.CaptureId, nativeFirstStep);''')
    text = replace(text, "        if (v2ObservationRecorder != null) v2ObservationRecorder.Capture(this);",
                   "        CaptureNativeObservation();")
    text = replace(text, "        Clock.ObserveFixedStep(Time.fixedTime);", '''        if (Time.fixedDeltaTime != CanonicalNativeTrace.FixedDeltaSeconds)
            throw new InvalidOperationException("native streaming requires the original 0.0004-second timestep");
        Clock.ObserveFixedStep(Time.fixedTime);''')
    start = text.index("    private void CaptureV2PostPhysicsStep()")
    end = text.index("    private void ObserveV2Stability()", start)
    text = text[:start] + '''    private void CaptureNativeObservation()
    {
        if (v2ObservationRecorder != null && nativeLastRenderedStep != Clock.FixedStep)
        {
            v2ObservationRecorder.Capture(this);
            nativeLastRenderedStep = Clock.FixedStep;
        }
    }

    private void CaptureV2PostPhysicsStep()
    {
        if (v2Recorder == null || v2Recorder.IsFinalized || (nativeTrace != null && nativeTrace.Closed)) return;
        if (v2Recorder.Failure == null) v2Recorder.RecordUnityFixedStep(Clock.FixedStep, V2CausalObjects());
        if (v2Recorder.Failure != null)
        {
            if (nativeTrace != null) nativeTrace.Finish(v2Recorder, v2Recorder.Failure.Message);
            v2Recorder.Deactivate();
            return;
        }
        if ((Clock.FixedStep - nativeFirstStep) % CanonicalNativeTrace.ObservationStride == 0)
            CaptureNativeObservation();
        ObserveV2Stability();
        if (nativeTrace != null && !nativeTrace.Closed)
        {
            if (Clock.FixedStep - nativeFirstStep >= CanonicalNativeTrace.MaximumShotSteps)
            {
                nativeTrace.Finish(v2Recorder, "native_time_window_limit");
                v2Recorder.Deactivate();
            }
            else nativeTrace.Observe(v2Recorder);
        }
    }

''' + text[end:]
    text = replace(text, "        if (v2StabilityCandidateSteps == 2", "        if (v2StabilityCandidateSteps == 100")
    text = replace(text, "            v2Recorder.FinalizeTerminal(Clock.FixedStep, reason);", '''        {
            v2Recorder.FinalizeTerminal(Clock.FixedStep, reason);
            CaptureNativeObservation();
            if (nativeTrace != null) nativeTrace.Finish(v2Recorder);
        }''')
    return text


def prepare():
    if WORK.exists():
        raise ValueError("native work already exists; preserve its version")
    changes = {}
    observers = PARENT / "project/Assets/Scripts/CanonicalCapture"
    for name, transform in (("PhysicsCaptureV2EngineProtocol.cs", native_protocol),
                            ("PhysicsCaptureV2FixedStepRecorder.cs", native_recorder),
                            ("PhysicalSnapshotRuntime.cs", native_runtime)):
        before = (observers / name).read_text()
        changes[name] = {"before": before, "after": transform(before)}
    WORK.mkdir(parents=True)
    print("[native-player] copying isolated source/cache; C2 project and player remain untouched", flush=True)
    shutil.copytree(PARENT / "project", WORK / "project",
                    ignore=lambda directory, names: [n for n in ("Temp", "Logs") if n in names])
    for name in ("preparation.json", "instrumentation.json", "shader-restoration.json"):
        shutil.copy2(PARENT / name, WORK / name)
    destination = WORK / "project/Assets/Scripts/CanonicalCapture"
    for name, values in changes.items():
        (destination / name).write_text(values["after"])
    shutil.copy2(ROOT / "tasks/issue_76_canonical/CanonicalNativeTrace.cs", destination)
    shutil.copy2(ROOT / "tasks/issue_76_canonical/CanonicalPigNativeTraceTests.cs", WORK / "project/Assets/Editor")
    (WORK / "native-port.json").write_text(json.dumps({"schema": "issue_76_native_streaming_port_v1",
        "parent_project": str(PARENT), "changes": changes, "fixed_delta_seconds": .0004,
        "observation_stride": 50, "max_steps": 30000, "chunk_samples": 250,
        "physics_timestep_changed": False, "fresh_access": False}, indent=2) + "\n")
    print(f"[native-player] prepared {WORK}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    for mode in ("dry-run", "prepare", "test", "build"):
        modes.add_argument("--" + mode, action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        observers = PARENT / "project/Assets/Scripts/CanonicalCapture"
        for name, transform in (("PhysicsCaptureV2EngineProtocol.cs", native_protocol),
                                ("PhysicsCaptureV2FixedStepRecorder.cs", native_recorder),
                                ("PhysicalSnapshotRuntime.cs", native_runtime)):
            transform((observers / name).read_text())
        print("[native-player] no-write native observer transform passed: .0004s physics, stride50 RGB, <=250-sample chunks", flush=True)
    elif args.prepare:
        prepare()
    else:
        # Only this unfrozen native working project is updated; C2 stays fixed.
        shutil.copy2(ROOT / "tasks/issue_76_canonical/CanonicalPigNativeTraceTests.cs", WORK / "project/Assets/Editor")
        shutil.copy2(ROOT / "tasks/issue_76_canonical/CanonicalNativeTrace.cs", WORK / "project/Assets/Scripts/CanonicalCapture")
        base.run_editor("test" if args.test else "build", work=WORK)
        if args.test:
            import xml.etree.ElementTree as ET
            result = ET.parse(max(WORK.glob("tests-*.xml"), key=lambda p: p.stat().st_mtime)).getroot()
            if int(result.attrib["passed"]) != 10 or int(result.attrib["failed"]) != 0:
                raise ValueError("the native player requires all ten fixtures, not only the seven C2 fixtures")


if __name__ == "__main__":
    main()
