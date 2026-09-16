using System;
using System.Globalization;
using System.IO;
using UnityEngine;

// Shared transport synchronization, not a policy input or a model prediction.
// The development player enables this before launch; the canonical parent is untouched.
public static class NativeDecisionBarrier
{
    public const string TargetEnvironmentVariable = "NOVPHY_NATIVE_DECISION_STEP";
    public const int FrameCount = 3;
    private static PhysicalSnapshotRuntime owner;
    private static PhysicsCaptureV2AlignedObservationRecorder recorder;
    private static long target;
    private static int captured;
    private static bool released;
    private static string directory;
    public static bool Paused { get; private set; }

    [Serializable]
    private class ReadyManifest
    {
        public string schema = "issue_76_native_decision_history_v1";
        public long target_fixed_step;
        public int frame_count = FrameCount;
        public int stride_native_steps = CanonicalNativeTrace.ObservationStride;
        public float native_step_seconds = CanonicalNativeTrace.FixedDeltaSeconds;
        public string capture_id;
    }

    public static void Cancel()
    {
        if (Paused) Time.timeScale = 1f;
        Paused = false;
        owner = null;
        recorder = null;
    }

    public static void Reset(PhysicalSnapshotRuntime runtime)
    {
        Cancel();
        string configured = Environment.GetEnvironmentVariable(TargetEnvironmentVariable);
        if (string.IsNullOrEmpty(configured)) return;
        if (!long.TryParse(configured, NumberStyles.None, CultureInfo.InvariantCulture, out target)
            || target <= (FrameCount - 1) * CanonicalNativeTrace.ObservationStride)
            throw new InvalidOperationException("native decision target must allow three prior RGB frames");
        string root = Environment.GetEnvironmentVariable(PhysicsCaptureV2AlignedObservationRecorder.RootEnvironmentVariable);
        if (string.IsNullOrEmpty(root))
            throw new InvalidOperationException("native decision history requires an aligned RGB root");
        owner = runtime;
        captured = 0;
        released = false;
        directory = Path.Combine(root, "decision-history-" + runtime.CaptureId);
    }

    public static void AfterBookkeeping(PhysicalSnapshotRuntime runtime)
    {
        if (owner != runtime || released || Paused) return;
        long next = target - (FrameCount - 1 - captured) * CanonicalNativeTrace.ObservationStride;
        if (runtime.Clock.FixedStep < next) return;
        if (runtime.Clock.FixedStep != next)
            throw new InvalidOperationException("native decision history missed its fixed-step capture");
        if (recorder == null) recorder = PhysicsCaptureV2AlignedObservationRecorder.Create(Path.GetFileName(directory));
        recorder.Capture(runtime);
        captured++;
        if (captured != FrameCount) return;
        Paused = true;
        Time.timeScale = 0f;
        string path = Path.Combine(directory, "ready.json");
        File.WriteAllText(path + ".tmp", JsonUtility.ToJson(new ReadyManifest
        {
            target_fixed_step = target,
            capture_id = Path.GetFileName(directory)
        }));
        File.Move(path + ".tmp", path);
    }

    public static void RequireReady(PhysicalSnapshotRuntime runtime)
    {
        if (owner != runtime || released) return;
        if (!Paused || captured != FrameCount || runtime.Clock.FixedStep != target)
            throw new InvalidOperationException("chosen native shot must start at the sealed decision barrier");
    }

    // Called only AFTER the shot recorder has captured the same pre-intervention state.
    public static void ReleaseAfterShotSnapshot(PhysicalSnapshotRuntime runtime)
    {
        if (owner != runtime || released) return;
        RequireReady(runtime);
        released = true;
        Paused = false;
        Time.timeScale = 1f;
    }
}
