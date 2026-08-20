using System;
using System.Linq;
using System.Text;
using NUnit.Framework;
using UnityEngine;

public class PhysicsCaptureV2RecorderTests
{
    private string previousStride;
    private GameObject host;

    [SetUp]
    public void SetUp()
    {
        previousStride = Environment.GetEnvironmentVariable(
            PhysicsCaptureV2EngineProtocol.StrideEnvironmentVariable,
            EnvironmentVariableTarget.Process);
        Environment.SetEnvironmentVariable(
            PhysicsCaptureV2EngineProtocol.StrideEnvironmentVariable, "2",
            EnvironmentVariableTarget.Process);
        host = new GameObject("physics-capture-v2-recorder");
    }

    [TearDown]
    public void TearDown()
    {
        UnityEngine.Object.DestroyImmediate(host);
        Environment.SetEnvironmentVariable(
            PhysicsCaptureV2EngineProtocol.StrideEnvironmentVariable, previousStride,
            EnvironmentVariableTarget.Process);
    }

    [Test]
    public void RecordsEveryConsecutiveFixedStepFromPreInterventionThroughTerminal()
    {
        PhysicsCaptureV2FixedStepRecorder recorder =
            host.AddComponent<PhysicsCaptureV2FixedStepRecorder>();

        recorder.BeginPreIntervention(100);
        recorder.RecordFixedStep(101);
        recorder.RecordFixedStep(102);
        recorder.FinalizeTerminal(102);
        PhysicsCaptureV2EngineSnapshot snapshot = recorder.CreateFinalizedSnapshot();

        CollectionAssert.AreEqual(new long[] { 100, 101, 102 },
            snapshot.FixedStepSamples.Select(sample => sample.FixedStep).ToArray());
        Assert.AreEqual(100, snapshot.PreInterventionFixedStep);
        Assert.AreEqual(102, snapshot.TerminalFixedStep);
    }

    [Test]
    public void RetainsScheduledStateRecordsAtTheConfiguredStrideThroughOnGridTerminal()
    {
        PhysicsCaptureV2FixedStepRecorder recorder =
            host.AddComponent<PhysicsCaptureV2FixedStepRecorder>();

        recorder.BeginPreIntervention(100);
        for (long fixedStep = 101; fixedStep <= 104; fixedStep++)
            recorder.RecordFixedStep(fixedStep);
        recorder.FinalizeTerminal(104);
        PhysicsCaptureV2EngineSnapshot snapshot = recorder.CreateFinalizedSnapshot();

        CollectionAssert.AreEqual(new long[] { 100, 102, 104 },
            snapshot.FrameRecords.Select(record => record.FixedStep).ToArray());
        Assert.IsTrue(snapshot.FrameRecords.All(record => !record.ForcedTerminal));
        Assert.AreEqual(snapshot.TerminalFixedStep, snapshot.FrameRecords.Last().FixedStep);
    }

    [Test]
    public void AppendsExactlyOneForcedTerminalStateRecordWhenTerminalIsOffGrid()
    {
        PhysicsCaptureV2FixedStepRecorder recorder =
            host.AddComponent<PhysicsCaptureV2FixedStepRecorder>();

        recorder.BeginPreIntervention(100);
        recorder.RecordFixedStep(101);
        recorder.RecordFixedStep(102);
        recorder.RecordFixedStep(103);
        recorder.FinalizeTerminal(103);
        PhysicsCaptureV2EngineSnapshot snapshot = recorder.CreateFinalizedSnapshot();

        CollectionAssert.AreEqual(new long[] { 100, 102, 103 },
            snapshot.FrameRecords.Select(record => record.FixedStep).ToArray());
        Assert.AreEqual(1, snapshot.FrameRecords.Count(record => record.ForcedTerminal));
        Assert.IsTrue(snapshot.FrameRecords.Last().ForcedTerminal);
        Assert.AreEqual(snapshot.TerminalFixedStep, snapshot.FrameRecords.Last().FixedStep);
    }

    [Test]
    public void Request71SerializesTheFrozenFixedStepAndStateRecordCoverage()
    {
        PhysicsCaptureV2FixedStepRecorder recorder =
            host.AddComponent<PhysicsCaptureV2FixedStepRecorder>();
        recorder.BeginPreIntervention(100);
        recorder.RecordFixedStep(101);
        recorder.RecordFixedStep(102);
        recorder.RecordFixedStep(103);
        recorder.FinalizeTerminal(103);
        PhysicsCaptureV2EngineSnapshot snapshot = recorder.CreateFinalizedSnapshot();

        byte[] envelope = PhysicsCaptureV2EngineProtocol.BuildCaptureEnvelope(snapshot);
        int payloadLength = ReadUInt32(envelope, 12);
        string json = Encoding.UTF8.GetString(envelope, 16, payloadLength);

        StringAssert.Contains("\"schema_version\":\"physics_capture_v2_engine_v1\"", json);
        StringAssert.Contains("\"configured_fixed_step_capture_stride\":2", json);
        StringAssert.Contains("\"pre_intervention_fixed_step\":100", json);
        StringAssert.Contains("\"fixed_step_samples\":[{\"fixed_step\":100", json);
        StringAssert.Contains("{\"fixed_step\":101", json);
        StringAssert.Contains("{\"fixed_step\":102", json);
        StringAssert.Contains("{\"fixed_step\":103", json);
        StringAssert.Contains("\"frame_records\":[{\"fixed_step\":100,\"state_id\":\"state:100\",\"forced_terminal\":false}", json);
        StringAssert.Contains("{\"fixed_step\":102,\"state_id\":\"state:102\",\"forced_terminal\":false}", json);
        StringAssert.Contains("{\"fixed_step\":103,\"state_id\":\"state:103\",\"forced_terminal\":true}", json);
        StringAssert.Contains("\"terminal_evidence\":{\"reason\":\"terminal\",\"fixed_step\":103", json);
    }

    [Test]
    public void RecorderStartsHaveDistinctFrozenCaptureAndShotIdentities()
    {
        PhysicsCaptureV2FixedStepRecorder first =
            host.AddComponent<PhysicsCaptureV2FixedStepRecorder>();
        first.BeginPreIntervention(0);
        first.FinalizeTerminal(0);
        PhysicsCaptureV2EngineSnapshot firstSnapshot = first.CreateFinalizedSnapshot();
        byte[] firstRead = PhysicsCaptureV2EngineProtocol.BuildCaptureEnvelope(firstSnapshot);
        byte[] repeatedRead = PhysicsCaptureV2EngineProtocol.BuildCaptureEnvelope(firstSnapshot);

        PhysicsCaptureV2FixedStepRecorder second =
            host.AddComponent<PhysicsCaptureV2FixedStepRecorder>();
        second.BeginPreIntervention(0);
        second.FinalizeTerminal(0);
        PhysicsCaptureV2EngineSnapshot secondSnapshot = second.CreateFinalizedSnapshot();

        Assert.IsNotEmpty(firstSnapshot.CaptureId);
        Assert.IsNotEmpty(firstSnapshot.ShotId);
        Assert.AreNotEqual(firstSnapshot.CaptureId, secondSnapshot.CaptureId);
        Assert.AreNotEqual(firstSnapshot.ShotId, secondSnapshot.ShotId);
        CollectionAssert.AreEqual(firstRead, repeatedRead);
    }

    private static int ReadUInt32(byte[] bytes, int offset)
    {
        return bytes[offset] << 24 | bytes[offset + 1] << 16
            | bytes[offset + 2] << 8 | bytes[offset + 3];
    }
}
