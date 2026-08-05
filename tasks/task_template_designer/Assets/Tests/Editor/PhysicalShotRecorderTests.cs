using System.Linq;
using NUnit.Framework;
using UnityEngine;

public class PhysicalShotRecorderTests
{
    [Test]
    public void RecordContacts_CanonicalizesReverseInputAndExcludesTriggers()
    {
        PhysicalShotRecorder recorder = new PhysicalShotRecorder(new PhysicalCaptureLimits(32, 16384, 10f));

        recorder.RecordContacts(1, 0.02f, new[]
        {
            new PhysicalContactInput("20:0", 20, new Vector2(1f, 0f), new Vector2(0f, -1f),
                0.01f, new Vector2(2f, 0f), 0f, "10:0", 10, new Vector2(0f, 0f), false),
            new PhysicalContactInput("10:0", 10, new Vector2(1f, 0f), new Vector2(0f, 1f),
                0.01f, new Vector2(2f, 0f), 0f, "20:0", 20, new Vector2(0f, 1f), false),
            new PhysicalContactInput("10:0", 10, Vector2.zero, Vector2.up, 0f, Vector2.zero, 0f,
                "30:0", 30, Vector2.up, true)
        });

        Assert.AreEqual(1, recorder.RawContacts.Count);
        Assert.AreEqual(10, recorder.RawContacts[0].ColliderIdA);
        Assert.AreEqual(20, recorder.RawContacts[0].ColliderIdB);
        Assert.AreEqual(new Vector2(0f, 1f), recorder.RawContacts[0].Normal);
        Assert.IsFalse(recorder.RawContacts[0].IsTrigger);
    }

    [Test]
    public void SupportRequiresTwoConsecutiveFixedStepsAndIsRemovedWhenContactStops()
    {
        PhysicalShotRecorder recorder = new PhysicalShotRecorder(new PhysicalCaptureLimits(32, 16384, 10f));
        PhysicalContactInput contact = SupportContact();

        recorder.RecordContacts(1, 0.02f, new[] { contact });
        Assert.AreEqual(0, recorder.SupportEdges.Count);

        recorder.RecordContacts(2, 0.04f, new[] { contact });
        Assert.AreEqual(1, recorder.SupportEdges.Count);
        Assert.AreEqual("10:0", recorder.SupportEdges[0].SupporterEntityId);
        Assert.AreEqual("20:0", recorder.SupportEdges[0].SupportedEntityId);

        recorder.RecordContacts(3, 0.06f, new PhysicalContactInput[0]);
        Assert.AreEqual(0, recorder.SupportEdges.Count);
    }

    [Test]
    public void EventsAreExactlyOnceExceptCollisionWhichIsOncePerPairPerFixedStep()
    {
        PhysicalShotRecorder recorder = new PhysicalShotRecorder(new PhysicalCaptureLimits(32, 16384, 10f));

        recorder.RecordEvent(1, 0.02f, PhysicalMacroEventKind.Launch, "bird:1");
        recorder.RecordEvent(1, 0.02f, PhysicalMacroEventKind.Launch, "bird:1");
        recorder.RecordCollision(1, 0.02f, "20:0", "10:0");
        recorder.RecordCollision(1, 0.02f, "10:0", "20:0");
        recorder.RecordCollision(2, 0.04f, "10:0", "20:0");
        recorder.RecordEvent(2, 0.04f, PhysicalMacroEventKind.LevelClear, "level");
        recorder.RecordEvent(2, 0.04f, PhysicalMacroEventKind.LevelClear, "level");

        Assert.AreEqual(4, recorder.Events.Count);
        Assert.AreEqual(PhysicalMacroEventKind.Launch, recorder.Events[0].Kind);
        Assert.AreEqual(PhysicalMacroEventKind.Collision, recorder.Events[1].Kind);
        Assert.AreEqual(PhysicalMacroEventKind.Collision, recorder.Events[2].Kind);
        Assert.AreEqual(PhysicalMacroEventKind.LevelClear, recorder.Events[3].Kind);
        Assert.IsTrue(recorder.Events.SequenceEqual(recorder.Events.OrderBy(e => e.Sequence)));
    }

    [Test]
    public void StabilityEventsEmitOnlyOnTransitions()
    {
        PhysicalShotRecorder recorder = new PhysicalShotRecorder(new PhysicalCaptureLimits(32, 16384, 10f));
        recorder.RecordStability(1, true);
        recorder.RecordStability(1, true);
        recorder.RecordStability(2, false);
        recorder.RecordStability(2, false);
        recorder.RecordStability(3, true);

        Assert.AreEqual(3, recorder.Events.Count);
        Assert.AreEqual(PhysicalMacroEventKind.StabilityEnter, recorder.Events[0].Kind);
        Assert.AreEqual(PhysicalMacroEventKind.StabilityExit, recorder.Events[1].Kind);
        Assert.AreEqual(PhysicalMacroEventKind.StabilityEnter, recorder.Events[2].Kind);
    }

    [Test]
    public void OverflowTimeoutAndTruncatedFinalizationAreTypedFailures()
    {
        PhysicalShotRecorder overflow = new PhysicalShotRecorder(new PhysicalCaptureLimits(1, 16384, 10f));
        overflow.RecordContacts(1, 0.02f, new[] { SupportContact(), Contact("30:0", 30, "40:0", 40) });
        Assert.AreEqual(PhysicalCaptureFailureCode.RecordLimitExceeded, overflow.Failure.Code);

        PhysicalShotRecorder timeout = new PhysicalShotRecorder(new PhysicalCaptureLimits(32, 16384, 0.1f));
        timeout.RecordContacts(1, 0.02f, new PhysicalContactInput[0]);
        timeout.RecordContacts(2, 0.2f, new PhysicalContactInput[0]);
        Assert.AreEqual(PhysicalCaptureFailureCode.CaptureTimeout, timeout.Failure.Code);

        PhysicalShotRecorder truncated = new PhysicalShotRecorder(new PhysicalCaptureLimits(32, 16384, 10f));
        PhysicalCaptureResult result = truncated.FinalizeShot(false);
        Assert.AreEqual(PhysicalCaptureFailureCode.TruncatedFinalization, result.Failure.Code);
    }

    private static PhysicalContactInput SupportContact()
    {
        return new PhysicalContactInput("10:0", 10, Vector2.zero, Vector2.up, 0.01f, Vector2.zero, 1f,
            "20:0", 20, Vector2.up, false);
    }

    private static PhysicalContactInput Contact(string entityA, int colliderA, string entityB, int colliderB)
    {
        return new PhysicalContactInput(entityA, colliderA, Vector2.zero, Vector2.up, 0f, Vector2.zero, 0f,
            entityB, colliderB, Vector2.up, false);
    }
}
