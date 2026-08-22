using System;
using System.Reflection;
using System.Text;
using System.Linq;
using NUnit.Framework;
using UnityEngine;

public sealed class PhysicsCaptureV2RuntimeIntegrationTests
{
    private string previousStride;
    private GameObject host;
    private GameObject causal;

    [SetUp]
    public void SetUp()
    {
        previousStride = Environment.GetEnvironmentVariable(
            PhysicsCaptureV2EngineProtocol.StrideEnvironmentVariable,
            EnvironmentVariableTarget.Process);
        host = new GameObject("physical-runtime");
        causal = new GameObject("causal-bird");
        ScenarioObjectIdentity.Assign(causal, "bird:0001");
        causal.AddComponent<Rigidbody2D>();
        causal.AddComponent<CircleCollider2D>();
    }

    [TearDown]
    public void TearDown()
    {
        UnityEngine.Object.DestroyImmediate(causal);
        UnityEngine.Object.DestroyImmediate(host);
        Environment.SetEnvironmentVariable(
            PhysicsCaptureV2EngineProtocol.StrideEnvironmentVariable, previousStride,
            EnvironmentVariableTarget.Process);
    }

    [Test]
    public void RuntimeBeginsBeforeInterventionRecordsFixedStepsAndFinalizesTerminalEvents()
    {
        Environment.SetEnvironmentVariable(
            PhysicsCaptureV2EngineProtocol.StrideEnvironmentVariable, "1",
            EnvironmentVariableTarget.Process);
        PhysicalSnapshotRuntime runtime = PhysicalSnapshotRuntime.Attach(host);

        runtime.BeginShot(32, 64 * 1024, 10f);
        Assert.IsNotNull(PhysicsCaptureV2FixedStepRecorder.Active);
        Assert.AreEqual(1, PhysicsCaptureV2EngineProtocol.BuildCaptureEnvelope()[9],
            "request 71 must remain pending before terminal finalization");
        InvokeFixedUpdate(runtime);
        string entityId = runtime.EntityIdFor(causal);
        runtime.RecordLaunch(entityId, Vector2.right);
        runtime.RecordLevelClear(123);

        byte[] envelope = PhysicsCaptureV2EngineProtocol.BuildCaptureEnvelope();
        Assert.AreEqual(0, envelope[9]);
        string json = Payload(envelope);
        StringAssert.Contains("\"event_type\":\"bird_launched\"", json);
        StringAssert.Contains("\"event_type\":\"level_clear\"", json);
        StringAssert.Contains("\"reason\":\"level_clear\"", json);
        Assert.AreEqual(2, json.Split(new[] { "\"payload\":" },
            StringSplitOptions.None).Length - 1,
            "each of the two events must serialize exactly one payload member");
        PhysicsCaptureV2EngineSnapshot snapshot =
            PhysicsCaptureV2FixedStepRecorder.Active.CreateFinalizedSnapshot();
        Assert.AreEqual(snapshot.TerminalFixedStep,
            snapshot.FrameRecords[snapshot.FrameRecords.Count - 1].FixedStep);
    }

    [Test]
    public void MissingStrideLeavesTheV2RuntimeOptedOutWithoutChangingV1()
    {
        Environment.SetEnvironmentVariable(
            PhysicsCaptureV2EngineProtocol.StrideEnvironmentVariable, null,
            EnvironmentVariableTarget.Process);
        PhysicalSnapshotRuntime runtime = PhysicalSnapshotRuntime.Attach(host);

        runtime.BeginShot(32, 64 * 1024, 10f);

        Assert.IsNotNull(runtime.ShotRecorder, "v1 shot behavior changed during v2 opt-out");
        Assert.IsTrue(PhysicsCaptureV2FixedStepRecorder.Active == null);
    }

    [Test]
    public void NormalRequestShotBeginsOnlyTheOptInV2Recorder()
    {
        Environment.SetEnvironmentVariable(
            PhysicsCaptureV2EngineProtocol.StrideEnvironmentVariable, "1",
            EnvironmentVariableTarget.Process);
        PhysicalSnapshotRuntime runtime = PhysicalSnapshotRuntime.Attach(host);

        runtime.BeginV2Shot();

        Assert.IsNull(runtime.ShotRecorder,
            "normal request-13/14 shooting must not create or change a request-70 recorder");
        Assert.IsNotNull(PhysicsCaptureV2FixedStepRecorder.Active);
        Assert.AreEqual(runtime.Clock.FixedStep,
            PhysicsCaptureV2FixedStepRecorder.Active.LastFixedStep);
    }

    [Test]
    public void RuntimeRetainsInactiveActivatedAndDestroyedSpawnLifecycleRows()
    {
        Environment.SetEnvironmentVariable(
            PhysicsCaptureV2EngineProtocol.StrideEnvironmentVariable, "1",
            EnvironmentVariableTarget.Process);
        GameObject spawn = new GameObject("spawned-child");
        ScenarioObjectIdentity.AssignSpawn(spawn,
            causal.GetComponent<ScenarioObjectIdentity>(), "fragment", 2);
        spawn.AddComponent<Rigidbody2D>();
        spawn.AddComponent<CircleCollider2D>();
        spawn.SetActive(false);
        try
        {
            PhysicalSnapshotRuntime runtime = PhysicalSnapshotRuntime.Attach(host);
            runtime.BeginShot(32, 64 * 1024, 10f);
            spawn.SetActive(true);
            InvokeFixedUpdate(runtime);
            UnityEngine.Object.DestroyImmediate(spawn);
            spawn = null;
            InvokeFixedUpdate(runtime);
            runtime.RecordLevelFail("test-terminal");
            PhysicsCaptureV2EngineSnapshot snapshot =
                PhysicsCaptureV2FixedStepRecorder.Active.CreateFinalizedSnapshot();
            const string spawnedId = "runtime:bird:0001/spawn:fragment:0002";

            Assert.AreEqual("inactive", snapshot.FixedStepSamples[0].Entities
                .Single(entity => entity.EntityId == spawnedId).Lifecycle);
            Assert.AreEqual("active", snapshot.FixedStepSamples[1].Entities
                .Single(entity => entity.EntityId == spawnedId).Lifecycle);
            PhysicsCaptureV2EntitySnapshot destroyed = snapshot.FixedStepSamples[2].Entities
                .Single(entity => entity.EntityId == spawnedId);
            Assert.AreEqual("destroyed", destroyed.Lifecycle);
            Assert.IsNull(destroyed.Body);
        }
        finally
        {
            if (spawn != null) UnityEngine.Object.DestroyImmediate(spawn);
        }
    }

    [Test]
    public void FixedUpdateDoesNotFreezeV2BeforeUnityAdvancesPhysics()
    {
        Environment.SetEnvironmentVariable(
            PhysicsCaptureV2EngineProtocol.StrideEnvironmentVariable, "1",
            EnvironmentVariableTarget.Process);
        PhysicalSnapshotRuntime runtime = PhysicalSnapshotRuntime.Attach(host);
        runtime.BeginShot(32, 64 * 1024, 10f);

        typeof(PhysicalSnapshotRuntime).GetMethod(
            "FixedUpdate", BindingFlags.Instance | BindingFlags.NonPublic).Invoke(runtime, null);
        Assert.AreEqual(0, PhysicsCaptureV2FixedStepRecorder.Active.LastFixedStep,
            "FixedUpdate must not label pre-simulation state as the completed fixed step");
        runtime.RecordLevelClear(1);

        PhysicsCaptureV2EngineSnapshot snapshot =
            PhysicsCaptureV2FixedStepRecorder.Active.CreateFinalizedSnapshot();
        Assert.AreEqual(2, snapshot.FixedStepSamples.Count,
            "terminal finalization must still freeze the completed post-physics step");
    }

    [Test]
    public void BatchedFixedUpdatesRetainCoverageBeforeLevelClearCallback()
    {
        Environment.SetEnvironmentVariable(
            PhysicsCaptureV2EngineProtocol.StrideEnvironmentVariable, "1",
            EnvironmentVariableTarget.Process);
        PhysicalSnapshotRuntime runtime = PhysicalSnapshotRuntime.Attach(host);
        runtime.BeginShot(32, 64 * 1024, 10f);

        InvokeRuntimeFixedUpdate(runtime);
        InvokeRuntimeFixedUpdate(runtime);
        runtime.RecordLevelClear(123);

        PhysicsCaptureV2FixedStepRecorder recorder = PhysicsCaptureV2FixedStepRecorder.Active;
        Assert.IsNull(recorder.Failure,
            "batched fixed steps must not leave a level-clear event outside recorded coverage");
        PhysicsCaptureV2EngineSnapshot snapshot = recorder.CreateFinalizedSnapshot();
        Assert.IsNotNull(snapshot);
        Assert.AreEqual(3, snapshot.FixedStepSamples.Count);
        Assert.AreEqual("level_clear", snapshot.TerminalReason);
    }

    [Test]
    public void StableTerminalRequiresTheInterventionLaunchToHaveOccurred()
    {
        Environment.SetEnvironmentVariable(
            PhysicsCaptureV2EngineProtocol.StrideEnvironmentVariable, "1",
            EnvironmentVariableTarget.Process);
        PhysicalSnapshotRuntime runtime = PhysicalSnapshotRuntime.Attach(host);
        causal.GetComponent<Rigidbody2D>().velocity = Vector2.zero;
        runtime.BeginShot(32, 64 * 1024, 10f);

        InvokeFixedUpdate(runtime);
        InvokeFixedUpdate(runtime);
        Assert.IsFalse(PhysicsCaptureV2FixedStepRecorder.Active.IsFinalized,
            "pre-release stability must not terminate the capture");

        runtime.RecordLaunch(runtime.EntityIdFor(causal), Vector2.zero);
        InvokeFixedUpdate(runtime);
        InvokeFixedUpdate(runtime);
        Assert.IsTrue(PhysicsCaptureV2FixedStepRecorder.Active.IsFinalized);
        Assert.AreEqual("stable_entered",
            PhysicsCaptureV2FixedStepRecorder.Active.CreateFinalizedSnapshot().TerminalReason);
    }

    private static void InvokeFixedUpdate(PhysicalSnapshotRuntime runtime)
    {
        InvokeRuntimeFixedUpdate(runtime);
        typeof(PhysicalSnapshotRuntime).GetMethod(
            "CaptureV2PostPhysicsStep", BindingFlags.Instance | BindingFlags.NonPublic)
            .Invoke(runtime, null);
    }

    private static void InvokeRuntimeFixedUpdate(PhysicalSnapshotRuntime runtime)
    {
        typeof(PhysicalSnapshotRuntime).GetMethod(
            "FixedUpdate", BindingFlags.Instance | BindingFlags.NonPublic).Invoke(runtime, null);
    }

    private static string Payload(byte[] envelope)
    {
        int length = envelope[12] << 24 | envelope[13] << 16 | envelope[14] << 8 | envelope[15];
        return Encoding.UTF8.GetString(envelope, 16, length);
    }
}
