using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.TestTools;

public class PhysicsShotRecorderTests
{
    // The exact strings the recorder logs when it refuses a collision. LogAssert
    // matches a string argument exactly, so keeping them here means a wording
    // change in the product fails these fixtures instead of silently passing.
    private const string CollisionEvidenceRejection =
        "physics_capture_v1: refusing a collision event without contact evidence; no event emitted.";
    private const string CollisionSpeedRejection =
        "physics_capture_v1: refusing a collision event whose relative speed is not finite and non-negative; no event emitted.";

    [Test]
    public void FixedStepContacts_ExcludeTriggersAndSortCanonicalPairs()
    {
        PhysicsShotRecorder recorder = new PhysicsShotRecorder(32, 64 * 1024);
        GameObject low = new GameObject("low");
        GameObject high = new GameObject("high");
        GameObject trigger = new GameObject("trigger");
        BoxCollider2D lowCollider = low.AddComponent<BoxCollider2D>();
        BoxCollider2D highCollider = high.AddComponent<BoxCollider2D>();
        BoxCollider2D triggerCollider = trigger.AddComponent<BoxCollider2D>();
        triggerCollider.isTrigger = true;

        try
        {
            recorder.RecordContacts(3, new[]
            {
                new PhysicsContactInput(highCollider, lowCollider, new Vector2(2f, 1f), new Vector2(-1f, 0f), 0.2f, new Vector2(3f, 0f), 4f),
                new PhysicsContactInput(triggerCollider, lowCollider, Vector2.zero, Vector2.up, 0f, Vector2.zero, 0f),
                new PhysicsContactInput(lowCollider, highCollider, new Vector2(1f, 1f), Vector2.right, 0.1f, Vector2.zero, 2f)
            });

            Assert.IsNull(recorder.Failure);
            Assert.AreEqual(2, recorder.RawContacts.Count);
            Assert.LessOrEqual(string.CompareOrdinal(recorder.RawContacts[0].EntityIdA, recorder.RawContacts[0].EntityIdB), 0);
            Assert.LessOrEqual(string.CompareOrdinal(recorder.RawContacts[1].EntityIdA, recorder.RawContacts[1].EntityIdB), 0);
            Assert.LessOrEqual(recorder.RawContacts[0].Point.x, recorder.RawContacts[1].Point.x);
        }
        finally
        {
            UnityEngine.Object.DestroyImmediate(low);
            UnityEngine.Object.DestroyImmediate(high);
            UnityEngine.Object.DestroyImmediate(trigger);
        }
    }

    [Test]
    public void SupportV1_RequiresTwoConsecutiveFixedStepsAndRemovesAfterGap()
    {
        PhysicsShotRecorder recorder = new PhysicsShotRecorder(32, 64 * 1024);
        GameObject upper = new GameObject("upper");
        GameObject lower = new GameObject("lower");
        BoxCollider2D upperCollider = upper.AddComponent<BoxCollider2D>();
        BoxCollider2D lowerCollider = lower.AddComponent<BoxCollider2D>();
        upper.transform.position = new Vector2(0f, 1f);
        lower.transform.position = Vector2.zero;

        try
        {
            PhysicsContactInput contact = new PhysicsContactInput(upperCollider, lowerCollider, Vector2.zero, Vector2.up, 0f, Vector2.zero, 0f);
            recorder.RecordContacts(10, new[] { contact });
            Assert.AreEqual(0, recorder.SupportEdges.Count);
            recorder.RecordContacts(11, new[] { contact });
            Assert.AreEqual(1, recorder.SupportEdges.Count);
            Assert.AreEqual(lowerCollider.GetInstanceID().ToString(), recorder.SupportEdges[0].SupporterEntityId);
            Assert.AreEqual(upperCollider.GetInstanceID().ToString(), recorder.SupportEdges[0].SupportedEntityId);
            recorder.RecordContacts(12, new PhysicsContactInput[0]);
            Assert.AreEqual(0, recorder.SupportEdges.Count);
        }
        finally
        {
            UnityEngine.Object.DestroyImmediate(upper);
            UnityEngine.Object.DestroyImmediate(lower);
        }
    }

    [Test]
    public void Events_AreExactlyOnceCollisionIsPerPairPerStepAndTerminalIsExclusive()
    {
        PhysicsShotRecorder recorder = new PhysicsShotRecorder(32, 64 * 1024);
        recorder.RecordLaunch("bird:0", 1);
        recorder.RecordLaunch("bird:0", 1);
        recorder.RecordCollision(2, 0.04f, "a", "b", new[] { "contact:2:a|b:0" }, 1f);
        recorder.RecordCollision(2, 0.04f, "b", "a", new[] { "contact:2:a|b:0" }, 1f);
        recorder.RecordCollision(3, 0.06f, "a", "b", new[] { "contact:3:a|b:0" }, 1f);
        recorder.RecordDestroyed("block:0", 4);
        recorder.RecordDeath("block:0", 4);
        recorder.RecordDestroyed("block:0", 4);
        recorder.RecordPigRemoved("pig:0", 5);
        recorder.RecordPigRemoved("pig:0", 5);
        recorder.RecordTntExplosion("tnt:0", 6);
        recorder.RecordTntExplosion("tnt:0", 6);
        recorder.RecordBirdExhaustion(7);
        recorder.RecordBirdExhaustion(7);
        recorder.RecordLevelClear(8);
        recorder.RecordLevelClear(8);
        recorder.RecordLevelFail(9);
        recorder.RecordLevelFail(9);

        Assert.IsNull(recorder.Failure);
        Assert.AreEqual(8, recorder.Events.Count);
        Assert.AreEqual(2, recorder.Events.Count(e => e.Taxonomy == "collision"));
        Assert.AreEqual(1, recorder.Events.Count(e => e.Taxonomy == "bird_launched"));
        Assert.AreEqual(1, recorder.Events.Count(e => e.Taxonomy == "entity_destroyed"));
        Assert.AreEqual(1, recorder.Events.Count(e => e.Taxonomy == "pig_removed"));
        Assert.AreEqual(1, recorder.Events.Count(e => e.Taxonomy == "explosion"));
        Assert.AreEqual(1, recorder.Events.Count(e => e.Taxonomy == "bird_exhausted"));
        Assert.AreEqual(1, recorder.Events.Count(e => e.Taxonomy == "level_cleared"));
        Assert.AreEqual(0, recorder.Events.Count(e => e.Taxonomy == "level_failed"));

        PhysicsShotRecorder failed = new PhysicsShotRecorder(4, 1024);
        failed.RecordLevelFail(1);
        failed.RecordLevelClear(2);
        Assert.AreEqual(1, failed.Events.Count);
        Assert.AreEqual("level_failed", failed.Events[0].Taxonomy);
    }

    [Test]
    public void CollisionPayloadRejectsMissingOrInvalidEvidence()
    {
        // The two evidence-bearing overloads are reachable from a Unity physics
        // callback, so they refuse by logging and returning: throwing there would
        // abort the rest of the caller's OnCollisionEnter2D. The refusal is still
        // fail-closed at the wire — no event is emitted either way, so
        // physics_capture_v1 never sees a collision without contact evidence.
        PhysicsShotRecorder recorder = new PhysicsShotRecorder(16, 64 * 1024);

        LogAssert.Expect(LogType.Error, CollisionEvidenceRejection);
        recorder.RecordCollision(2, 0.04f, "a:0", "b:0", new string[0], 0f);
        Assert.AreEqual(0, recorder.Events.Count);

        LogAssert.Expect(LogType.Error, CollisionSpeedRejection);
        recorder.RecordCollision(2, 0.04f, "a:0", "b:0", new[] { "contact:2" }, -1f);
        Assert.AreEqual(0, recorder.Events.Count);

        LogAssert.Expect(LogType.Error, CollisionSpeedRejection);
        recorder.RecordCollision(2, 0.04f, "a:0", "b:0", new[] { "contact:2" }, float.NaN);
        Assert.AreEqual(0, recorder.Events.Count);

        // The evidence-free overload has no product caller and is unreachable from
        // any callback, so its throw stays: it is the API guard that makes
        // "record a collision without evidence" unusable rather than merely noisy.
        Assert.Throws<ArgumentException>(delegate
        {
            recorder.RecordCollision(2, 0.04f, "a:0", "b:0");
        });
        Assert.Throws<ArgumentException>(delegate
        {
            recorder.RecordCollision("a:0", "b:0", 2);
        });
        Assert.AreEqual(0, recorder.Events.Count);
    }

    [Test]
    public void CollisionContactSamplesCreateDeterministicPayloadEvidence()
    {
        PhysicsShotRecorder recorder = new PhysicsShotRecorder(16, 64 * 1024);
        PhysicalContactInput[] contacts =
        {
            new PhysicalContactInput("z:0", 2, Vector2.right, Vector2.up, -0.1f,
                Vector2.left, 1f, "a:0", 1, Vector2.zero, Vector2.one, false),
            new PhysicalContactInput("z:0", 2, Vector2.zero, Vector2.up, -0.1f,
                Vector2.left, 1f, "a:0", 1, Vector2.zero, Vector2.one, false)
        };

        recorder.RecordCollision(2, 0.04f, "z:0", "a:0", contacts, 3.5f);
        recorder.RecordCollision(2, 0.04f, "a:0", "z:0", contacts.Reverse().ToArray(), 3.5f);

        Assert.AreEqual(2, recorder.RawContacts.Count);
        Assert.AreEqual(1, recorder.Events.Count);
        CollectionAssert.AreEqual(
            recorder.RawContacts.Select(contact => contact.ContactId).OrderBy(id => id).ToArray(),
            recorder.Events[0].Payload.ContactIds);
        Assert.AreEqual(3.5f, recorder.Events[0].Payload.RelativeSpeed.Value);
    }

    [Test]
    public void CollisionContactSamplesRejectBeforeRecorderMutation()
    {
        // Covers the two guards on the PhysicalContactInput[] overload — a
        // non-finite relative speed, and evidence that names some other pair. Both
        // now refuse by logging instead of throwing, and a logged refusal must
        // leave exactly as little behind as the thrown one did: no raw contacts,
        // no events, nothing half-written. It does not cover the other overloads
        // or the other rejection reasons; CollisionPayloadRejectsMissingOrInvalidEvidence
        // does that.
        PhysicsShotRecorder invalidSpeed = new PhysicsShotRecorder(16, 64 * 1024);
        PhysicalContactInput matching = new PhysicalContactInput("a:0", 1, Vector2.zero, Vector2.up, -0.1f,
            Vector2.left, 1f, "b:0", 2, Vector2.zero, Vector2.one, false);

        LogAssert.Expect(LogType.Error, CollisionSpeedRejection);
        invalidSpeed.RecordCollision(2, 0.04f, "a:0", "b:0", new[] { matching }, float.NaN);
        Assert.AreEqual(0, invalidSpeed.RawContacts.Count);
        Assert.AreEqual(0, invalidSpeed.Events.Count);

        PhysicsShotRecorder unrelatedPair = new PhysicsShotRecorder(16, 64 * 1024);
        PhysicalContactInput unrelated = new PhysicalContactInput("a:0", 1, Vector2.zero, Vector2.up, -0.1f,
            Vector2.left, 1f, "c:0", 3, Vector2.zero, Vector2.one, false);

        LogAssert.Expect(LogType.Error, CollisionEvidenceRejection);
        unrelatedPair.RecordCollision(2, 0.04f, "a:0", "b:0", new[] { unrelated }, 1f);
        Assert.AreEqual(0, unrelatedPair.RawContacts.Count);
        Assert.AreEqual(0, unrelatedPair.Events.Count);
    }

    [Test]
    public void BoundedRecorder_ReportsTypedOverflowTimeoutAndTruncatedFinalization()
    {
        PhysicsShotRecorder overflow = new PhysicsShotRecorder(1, 64 * 1024);
        GameObject first = new GameObject("first");
        GameObject second = new GameObject("second");
        try
        {
            PhysicsContactInput input = new PhysicsContactInput(
                first.AddComponent<BoxCollider2D>(), second.AddComponent<BoxCollider2D>(),
                Vector2.zero, Vector2.up, 0f, Vector2.zero, 0f);
            overflow.RecordContacts(1, new[] { input });
            overflow.RecordContacts(2, new[] { input });
            Assert.AreEqual("record_limit_exceeded", overflow.Failure.CodeName);
        }
        finally
        {
            UnityEngine.Object.DestroyImmediate(first);
            UnityEngine.Object.DestroyImmediate(second);
        }

        PhysicsShotRecorder timeout = new PhysicsShotRecorder(4, 64 * 1024);
        timeout.FailTimeout("shot timeout");
        Assert.AreEqual("capture_timeout", timeout.Failure.CodeName);

        PhysicsShotRecorder truncated = new PhysicsShotRecorder(4, 64 * 1024);
        Assert.IsFalse(truncated.TryFinalize(false));
        Assert.AreEqual(PhysicsCaptureFailureCode.TruncatedFinalization, truncated.Failure.Code);
        Assert.AreEqual("truncated_finalization", truncated.Failure.CodeName);

        PhysicsShotRecorder bytes = new PhysicsShotRecorder(4, 1);
        bytes.RecordLaunch("bird:0", 1);
        Assert.AreEqual("byte_limit_exceeded", bytes.Failure.CodeName);
    }

    [Test]
    public void Contacts_CanonicalizeByLifetimeIdAndNegateDirectionalValues()
    {
        PhysicsShotRecorder recorder = new PhysicsShotRecorder(8, 64 * 1024);
        recorder.RecordContacts(1, new[] { new PhysicalContactInput(
            "z:0", 2, Vector2.zero, Vector2.right, 0f, new Vector2(3f, 4f), 1f,
            "a:0", 1, Vector2.zero, Vector2.zero, false) });

        Assert.AreEqual("a:0", recorder.RawContacts[0].EntityIdA);
        Assert.AreEqual("z:0", recorder.RawContacts[0].EntityIdB);
        Assert.AreEqual(Vector2.left, recorder.RawContacts[0].Normal);
        Assert.AreEqual(new Vector2(-3f, -4f), recorder.RawContacts[0].RelativeVelocity);
    }

    [Test]
    public void Contacts_PreserveTangentImpulseAndAssignUniquePointIndices()
    {
        PhysicsShotRecorder recorder = new PhysicsShotRecorder(8, 64 * 1024);
        recorder.RecordContacts(3, new[]
        {
            new PhysicalContactInput("a:0", 1, new Vector2(0f, 1f), Vector2.up, -0.1f,
                Vector2.zero, 2f, 0.25f, "b:0", 2, Vector2.zero, Vector2.up, false),
            new PhysicalContactInput("a:0", 1, new Vector2(1f, 1f), Vector2.up, -0.1f,
                Vector2.zero, 3f, 0.5f, "b:0", 2, Vector2.zero, Vector2.up, false)
        });

        Assert.AreEqual(0.25f, recorder.RawContacts[0].TangentImpulse);
        Assert.AreEqual(0.5f, recorder.RawContacts[1].TangentImpulse);
        Assert.AreEqual(0, recorder.RawContacts[0].PointIndex);
        Assert.AreEqual(1, recorder.RawContacts[1].PointIndex);
        Assert.AreNotEqual(recorder.RawContacts[0].ContactId, recorder.RawContacts[1].ContactId);
        Assert.IsTrue(recorder.RawContacts[0].ContactId.EndsWith(":0"));
        Assert.IsTrue(recorder.RawContacts[1].ContactId.EndsWith(":1"));
    }

    [Test]
    public void Support_CitesBothConsecutiveContactSamplesAndRejectsOneStepHistory()
    {
        PhysicsShotRecorder recorder = new PhysicsShotRecorder(16, 64 * 1024);
        PhysicalContactInput sample = new PhysicalContactInput(
            "a:0", 2, Vector2.zero, Vector2.up, 0f, Vector2.zero, 0f,
            "z:0", 1, new Vector2(0f, 1f), false);
        recorder.RecordContacts(4, new[] { sample });
        Assert.AreEqual(0, recorder.SupportEdges.Count);
        recorder.RecordContacts(5, new[] { sample });
        Assert.AreEqual(1, recorder.SupportEdges.Count);
        Assert.AreEqual(4L, recorder.SupportEdges[0].FixedStepA);
        Assert.AreEqual(5L, recorder.SupportEdges[0].FixedStepB);
        Assert.IsTrue(recorder.SupportEdges[0].ContactIdA.Contains("contact:4:"));
        recorder.RecordContacts(7, new[] { sample });
        Assert.AreEqual(0, recorder.SupportEdges.Count);
    }

    [Test]
    public void Support_RejectsVerticalOrderingReversedBetweenSamples()
    {
        PhysicsShotRecorder recorder = new PhysicsShotRecorder(16, 64 * 1024);
        recorder.RecordContacts(1, new[] { ContactWithCenters(0f, 1f) });
        recorder.RecordContacts(2, new[] { ContactWithCenters(1f, 0f) });

        Assert.AreEqual(0, recorder.SupportEdges.Count);
    }

    [Test]
    public void CollisionAtAStepDoesNotErasePriorSupportEdges()
    {
        // The collision path ingests one pair's contacts, but UpdateSupport treats
        // its argument as the complete contact set for the step and prunes every
        // edge it does not see. A collision therefore used to wipe the support
        // graph of the whole tower at exactly the steps where something happens.
        // Support derivation belongs solely to the full-set FixedUpdate sampler.
        PhysicsShotRecorder recorder = new PhysicsShotRecorder(64, 64 * 1024);
        recorder.RecordContacts(1, new[] { ContactWithCenters(0f, 1f) });
        recorder.RecordContacts(2, new[] { ContactWithCenters(0f, 1f) });
        Assert.AreEqual(1, recorder.SupportEdges.Count, "the resting pair must have produced a support edge");
        string survivingPair = recorder.SupportEdges[0].PairKey;

        // A different pair collides on the same fixed step the sampler just covered.
        PhysicalContactInput colliding = new PhysicalContactInput(
            "c:0", 3, Vector2.zero, Vector2.up, -0.05f, Vector2.left, 1f,
            "d:0", 4, Vector2.zero, Vector2.one, false);
        recorder.RecordCollision(2, 0.04f, "c:0", "d:0", new[] { colliding }, 2.5f);

        Assert.AreEqual(1, recorder.Events.Count, "the collision itself must still be recorded");
        Assert.AreEqual(1, recorder.SupportEdges.Count,
            "a collision on one pair must not prune the support edges of every other pair");
        Assert.AreEqual(survivingPair, recorder.SupportEdges[0].PairKey);
    }

    [Test]
    public void Events_UseFrozenTaxonomyAndBoundsIncludeEvents()
    {
        PhysicsShotRecorder recorder = new PhysicsShotRecorder(2, 1024);
        recorder.RecordLevelFail(2);
        recorder.RecordLaunch("bird:0", 1);
        Assert.IsNull(recorder.Failure);
        Assert.AreEqual("bird_launched", recorder.Events[0].Taxonomy);
        Assert.AreEqual(1L, recorder.Events[0].Sequence);
        Assert.AreEqual(2L, recorder.Events[1].Sequence);
        recorder.RecordDestroyed("block:0", 3);
        Assert.AreEqual("record_limit_exceeded", recorder.Failure.CodeName);
    }

    [Test]
    public void Events_SortByFixedStepThenTaxonomyAndParticipantBeforeSequence()
    {
        PhysicsShotRecorder recorder = new PhysicsShotRecorder(8, 64 * 1024);
        recorder.RecordEvent(2, 0.05f, PhysicalMacroEventKind.Destroy, "b:0");
        recorder.RecordEvent(2, 0.05f, PhysicalMacroEventKind.Launch, "a:0");
        recorder.RecordEvent(2, 0.04f, PhysicalMacroEventKind.Destroy, "z:0");
        recorder.RecordEvent(2, 0.05f, PhysicalMacroEventKind.Destroy, "a:0");

        CollectionAssert.AreEqual(
            new[] { "bird_launched", "entity_destroyed", "entity_destroyed", "entity_destroyed" },
            recorder.Events.Select(item => item.Taxonomy).ToArray());
        CollectionAssert.AreEqual(
            new[] { "a:0", "a:0", "b:0", "z:0" },
            recorder.Events.Select(item => item.Subject).ToArray());
        CollectionAssert.AreEqual(
            new long[] { 1, 2, 3, 4 },
            recorder.Events.Select(item => item.Sequence).ToArray());
    }

    [Test]
    public void Events_ExposeSortedParticipantsAndFrozenPayloadFields()
    {
        PhysicsShotRecorder recorder = new PhysicsShotRecorder(16, 64 * 1024);
        recorder.RecordLaunch("bird:0", 1, new Vector2(2f, 1f));
        recorder.RecordCollision(2, 0.04f, "z:0", "a:0",
            new[] { "contact:2:a:0|1:z:0|2:0" }, 3.5f);
        recorder.RecordTntExplosion("tnt:0", 3, 2.5f);
        recorder.RecordDestroyed("block:0", 4, "damage");
        recorder.RecordPigRemoved("pig:0", 5, "destroyed");
        recorder.RecordBirdExhaustion(6);
        recorder.RecordStability(7, true);
        recorder.RecordLevelClear(8, 12345);

        CollectionAssert.AreEqual(new[] { "bird:0" }, recorder.Events[0].Participants);
        Assert.AreEqual(new Vector2(2f, 1f), recorder.Events[0].Payload.LaunchVelocity.Value);
        CollectionAssert.AreEqual(new[] { "a:0", "z:0" }, recorder.Events[1].Participants);
        CollectionAssert.AreEqual(new[] { "contact:2:a:0|1:z:0|2:0" }, recorder.Events[1].Payload.ContactIds);
        Assert.AreEqual(3.5f, recorder.Events[1].Payload.RelativeSpeed.Value);
        Assert.AreEqual(2.5f, recorder.Events[2].Payload.RadiusUnityUnits.Value);
        Assert.AreEqual("damage", recorder.Events[3].Payload.Reason);
        Assert.AreEqual("destroyed", recorder.Events[4].Payload.Reason);
        Assert.AreEqual(0, recorder.Events[5].Payload.BirdsRemaining.Value);
        Assert.AreEqual(2, recorder.Events[6].Payload.DebounceFixedSteps.Value);
        Assert.AreEqual(12345, recorder.Events[7].Payload.Score.Value);

        PhysicsShotRecorder failed = new PhysicsShotRecorder(4, 1024);
        failed.RecordLevelFail(1, "no_playable_birds");
        Assert.AreEqual("no_playable_birds", failed.Events[0].Payload.Reason);
        Assert.AreEqual(0, failed.Events[0].Participants.Count);
    }

    [Test]
    public void RuntimeContacts_UseStaticWorldIdsAndLifetimeRegistryIds()
    {
        GameObject ground = new GameObject("ground");
        GameObject bodyObject = new GameObject("body");
        BoxCollider2D groundCollider = ground.AddComponent<BoxCollider2D>();
        BoxCollider2D bodyCollider = bodyObject.AddComponent<BoxCollider2D>();
        Rigidbody2D body = bodyObject.AddComponent<Rigidbody2D>();
        body.gravityScale = 0f;
        body.constraints = RigidbodyConstraints2D.FreezeAll;
        groundCollider.size = new Vector2(4f, 1f);
        bodyObject.transform.position = new Vector2(0f, 0.9f);
        PhysicsShotRecorder recorder = new PhysicsShotRecorder(16, 64 * 1024);
        PhysicalEntityRegistry registry = new PhysicalEntityRegistry();
        bool autoSimulation = Physics2D.autoSimulation;

        try
        {
            Physics2D.autoSimulation = false;
            Physics2D.SyncTransforms();
            Physics2D.Simulate(0.02f);
            Physics2D.Simulate(0.02f);
            recorder.RecordUnityContacts(1, 0.02f, new Collider2D[] { groundCollider, bodyCollider }, registry);

            string staticId = PhysicsShotRecorder.RuntimeEntityId(groundCollider);
            string dynamicId = registry.RegisterCollider(bodyCollider);
            Assert.IsTrue(staticId.StartsWith("world:static:"));
            Assert.IsFalse(dynamicId.StartsWith("world:static:"));
            recorder.RecordContacts(2, new[] { new PhysicalContactInput(
                dynamicId, bodyCollider.GetInstanceID(), Vector2.zero, Vector2.up, 0f, Vector2.zero, 0f,
                staticId, groundCollider.GetInstanceID(), ground.transform.position, bodyObject.transform.position, false) });
            Assert.IsTrue(recorder.RawContacts.Any(contact =>
                contact.EntityIdA.StartsWith("world:static:") || contact.EntityIdB.StartsWith("world:static:")));
        }
        finally
        {
            Physics2D.autoSimulation = autoSimulation;
            UnityEngine.Object.DestroyImmediate(bodyObject);
            UnityEngine.Object.DestroyImmediate(ground);
        }
    }

    [Test]
    public void LifetimeReuse_ProducesDistinctContactAndEventParticipants()
    {
        PhysicalEntityRegistry registry = new PhysicalEntityRegistry();
        GameObject firstLifetime = new GameObject("first-lifetime");
        GameObject secondLifetime = new GameObject("second-lifetime");
        PhysicsShotRecorder recorder = new PhysicsShotRecorder(16, 64 * 1024);

        try
        {
            string firstId = registry.Register(41, firstLifetime);
            string secondId = registry.Register(41, secondLifetime);
            recorder.RecordContacts(1, new[] { new PhysicalContactInput(
                firstId, 1, Vector2.zero, Vector2.up, 0f, Vector2.zero, 0f,
                "world:static:7", 7, Vector2.zero, Vector2.up, false) });
            recorder.RecordContacts(2, new[] { new PhysicalContactInput(
                secondId, 2, Vector2.zero, Vector2.up, 0f, Vector2.zero, 0f,
                "world:static:7", 7, Vector2.zero, Vector2.up, false) });
            recorder.RecordDestroyed(firstId, 1);
            recorder.RecordDestroyed(secondId, 2);

            Assert.AreEqual("41:0", firstId);
            Assert.AreEqual("41:1", secondId);
            Assert.AreNotEqual(recorder.RawContacts[0].PairKey, recorder.RawContacts[1].PairKey);
            Assert.AreEqual(2, recorder.Events.Count(e => e.Taxonomy == "entity_destroyed"));
        }
        finally
        {
            UnityEngine.Object.DestroyImmediate(secondLifetime);
            UnityEngine.Object.DestroyImmediate(firstLifetime);
        }
    }

    [Test]
    public void RuntimeStability_RequiresTwoAuthoritativeFixedStepSamplesPerTransition()
    {
        GameObject host = new GameObject("runtime");
        GameObject moving = new GameObject("moving");
        PhysicalSnapshotRuntime runtime = PhysicalSnapshotRuntime.Attach(host);
        Rigidbody2D body = moving.AddComponent<Rigidbody2D>();
        body.gravityScale = 0f;
        MethodInfo fixedUpdate = typeof(PhysicalSnapshotRuntime).GetMethod(
            "FixedUpdate", BindingFlags.Instance | BindingFlags.NonPublic);

        try
        {
            runtime.BeginShot(16, 64 * 1024, 10f);
            body.velocity = Vector2.right;
            fixedUpdate.Invoke(runtime, null);
            Assert.AreEqual(0, runtime.ShotRecorder.Events.Count);
            fixedUpdate.Invoke(runtime, null);
            Assert.AreEqual(1, runtime.ShotRecorder.Events.Count(e => e.Taxonomy == "stable_exited"));

            body.velocity = Vector2.zero;
            fixedUpdate.Invoke(runtime, null);
            Assert.AreEqual(0, runtime.ShotRecorder.Events.Count(e => e.Taxonomy == "stable_entered"));
            fixedUpdate.Invoke(runtime, null);
            Assert.AreEqual(1, runtime.ShotRecorder.Events.Count(e => e.Taxonomy == "stable_entered"));
        }
        finally
        {
            UnityEngine.Object.DestroyImmediate(moving);
            UnityEngine.Object.DestroyImmediate(host);
        }
    }

    [Test]
    public void TimeoutUsesElapsedFixedTimeFromFirstAuthoritativeSample()
    {
        PhysicsShotRecorder recorder = new PhysicsShotRecorder(
            new PhysicalCaptureLimits(16, 64 * 1024, 0.1f));

        recorder.RecordContacts(1000, 50f, new PhysicalContactInput[0]);
        Assert.IsNull(recorder.Failure, "a late first sample must establish the shot clock");
        recorder.RecordContacts(1001, 49f, new PhysicalContactInput[0]);
        Assert.IsNull(recorder.Failure, "elapsed shot time must not become negative");
        recorder.RecordContacts(1002, 50.1f, new PhysicalContactInput[0]);
        Assert.IsNull(recorder.Failure, "the timeout boundary is inclusive");
        recorder.RecordContacts(1003, 50.101f, new PhysicalContactInput[0]);

        Assert.IsNotNull(recorder.Failure);
        Assert.AreEqual(PhysicalCaptureFailureCode.CaptureTimeout, recorder.Failure.Code);
    }

    [Test]
    public void FinalizedRecorderRejectsEveryPublicMutationPath()
    {
        AssertFinalizedMutationIsNoOp(delegate(PhysicsShotRecorder recorder)
        {
            recorder.RecordContacts(2, new[] { ContactWithCenters(0f, 1f) });
        }, "contacts array");
        AssertFinalizedMutationIsNoOp(delegate(PhysicsShotRecorder recorder)
        {
            recorder.RecordContacts(2, (System.Collections.Generic.IEnumerable<PhysicalContactInput>)new[] { ContactWithCenters(0f, 1f) });
        }, "contacts enumerable");
        AssertFinalizedMutationIsNoOp(delegate(PhysicsShotRecorder recorder)
        {
            recorder.RecordContacts(2, 0.04f, new[] { ContactWithCenters(0f, 1f) });
        }, "contacts with fixed time");
        AssertFinalizedUnityContactsReturnBeforeRegistryMutation();
        AssertFinalizedMutationIsNoOp(delegate(PhysicsShotRecorder recorder)
        {
            recorder.RecordEvent(2, 0.04f, PhysicalMacroEventKind.Destroy, "block:2");
        }, "generic event");
        AssertFinalizedMutationIsNoOp(delegate(PhysicsShotRecorder recorder)
        {
            recorder.RecordCollision(2, 0.04f, "a:0", "b:0");
        }, "collision");
        AssertFinalizedMutationIsNoOp(delegate(PhysicsShotRecorder recorder)
        {
            recorder.RecordCollision(2, 0.04f, "a:0", "b:0", new[] { "contact:2" }, 3f);
        }, "collision payload");
        AssertFinalizedMutationIsNoOp(delegate(PhysicsShotRecorder recorder)
        {
            recorder.RecordCollision("a:0", "b:0", 2);
        }, "collision compatibility overload");
        AssertFinalizedMutationIsNoOp(delegate(PhysicsShotRecorder recorder) { recorder.RecordLaunch("bird:2", 2); }, "launch");
        AssertFinalizedMutationIsNoOp(delegate(PhysicsShotRecorder recorder) { recorder.RecordLaunch("bird:2", 2, Vector2.right); }, "launch payload");
        AssertFinalizedMutationIsNoOp(delegate(PhysicsShotRecorder recorder) { recorder.RecordDestroyed("block:2", 2); }, "destroy");
        AssertFinalizedMutationIsNoOp(delegate(PhysicsShotRecorder recorder) { recorder.RecordDestroyed("block:2", 2, "damage"); }, "destroy payload");
        AssertFinalizedMutationIsNoOp(delegate(PhysicsShotRecorder recorder) { recorder.RecordDeath("block:2", 2); }, "death");
        AssertFinalizedMutationIsNoOp(delegate(PhysicsShotRecorder recorder) { recorder.RecordPigRemoved("pig:2", 2); }, "pig removal");
        AssertFinalizedMutationIsNoOp(delegate(PhysicsShotRecorder recorder) { recorder.RecordPigRemoved("pig:2", 2, "damage"); }, "pig removal payload");
        AssertFinalizedMutationIsNoOp(delegate(PhysicsShotRecorder recorder) { recorder.RecordTntExplosion("tnt:2", 2); }, "TNT explosion");
        AssertFinalizedMutationIsNoOp(delegate(PhysicsShotRecorder recorder) { recorder.RecordTntExplosion("tnt:2", 2, 2f); }, "TNT explosion payload");
        AssertFinalizedMutationIsNoOp(delegate(PhysicsShotRecorder recorder) { recorder.RecordBirdExhaustion(2); }, "bird exhaustion");
        AssertFinalizedMutationIsNoOp(delegate(PhysicsShotRecorder recorder) { recorder.RecordStability(2, true); }, "stability");
        AssertFinalizedMutationIsNoOp(delegate(PhysicsShotRecorder recorder) { recorder.RecordLevelClear(2); }, "level clear");
        AssertFinalizedMutationIsNoOp(delegate(PhysicsShotRecorder recorder) { recorder.RecordLevelClear(2, 10); }, "level clear payload");
        AssertFinalizedMutationIsNoOp(delegate(PhysicsShotRecorder recorder) { recorder.RecordLevelFail(2); }, "level fail");
        AssertFinalizedMutationIsNoOp(delegate(PhysicsShotRecorder recorder) { recorder.RecordLevelFail(2, "reason"); }, "level fail payload");
        AssertFinalizedMutationIsNoOp(delegate(PhysicsShotRecorder recorder) { recorder.FailTimeout("late timeout"); }, "timeout failure");
    }

    [Test]
    public void TruncatedFinalizationSetsFailureBeforeBecomingTerminal()
    {
        PhysicsShotRecorder recorder = new PhysicsShotRecorder(8, 64 * 1024);

        PhysicalCaptureResult result = recorder.FinalizeShot(false);

        Assert.IsFalse(result.IsValid);
        Assert.AreEqual(PhysicalCaptureFailureCode.TruncatedFinalization, result.Failure.Code);
        recorder.FailTimeout("must not replace finalized failure");
        Assert.AreEqual(PhysicalCaptureFailureCode.TruncatedFinalization, recorder.Failure.Code);
    }

    [Test]
    public void Retention_KeepsLastTwoFixedStepsAndFinalizedContactsAreGloballyOrdered()
    {
        // F1 + F7 together: after retention the finalized raw-contact stream holds
        // exactly the last two full fixed steps, and it is globally ordered by the
        // parser's key (CompareContacts extended with the contact_id ordinal), not
        // merely sorted within each step. The step-major concatenation puts
        // pair-low(step 2) before pair-high(step 2) before pair-low(step 3), which
        // the global sort must repair by moving pair-low(step 3) ahead of
        // pair-high(step 2).
        PhysicsShotRecorder recorder = new PhysicsShotRecorder(64, 64 * 1024);
        PhysicalContactInput pairLow = new PhysicalContactInput(
            "100:0", 11, new Vector2(0.5f, 0.25f), Vector2.up, -0.01f,
            Vector2.right, 1f, "200:0", 21, Vector2.zero, Vector2.one, false);
        PhysicalContactInput pairHigh = new PhysicalContactInput(
            "300:0", 31, new Vector2(0.75f, 0.25f), Vector2.up, -0.01f,
            Vector2.right, 1f, "400:0", 41, Vector2.zero, Vector2.one, false);

        for (long step = 1; step <= 3; step++)
            recorder.RecordContacts(step, new[] { pairLow, pairHigh });

        Assert.IsNull(recorder.Failure);
        Assert.AreEqual(4, recorder.RawContacts.Count, "retention must keep exactly the last two full steps");
        CollectionAssert.AreEquivalent(new long[] { 2, 2, 3, 3 },
            recorder.RawContacts.Select(contact => contact.FixedStep).ToArray());

        recorder.FinalizeShot(true);
        PhysicalShotRecorderSnapshot snapshot = recorder.CreateFinalizedSnapshot();
        Assert.IsNotNull(snapshot);

        List<PhysicalRawContact> expected = snapshot.RawContacts
            .OrderBy(c => c.EntityIdA, StringComparer.Ordinal)
            .ThenBy(c => c.EntityIdB, StringComparer.Ordinal)
            .ThenBy(c => c.ColliderIdA)
            .ThenBy(c => c.ColliderIdB)
            .ThenBy(c => c.Point.x)
            .ThenBy(c => c.Point.y)
            .ThenBy(c => c.ContactId, StringComparer.Ordinal)
            .ToList();
        CollectionAssert.AreEqual(expected, snapshot.RawContacts,
            "finalized raw_contacts must be globally ordered by the parser key including contact_id");
    }

    [Test]
    public void SupportEdges_FinalizedOrderFollowsSupporterSupportedSupportId()
    {
        // F2: UpdateSupport appends edges in contact-pair order, but the parser
        // sorts by supporter_id. The geometry below makes the append order differ:
        // a low-numbered pair whose upper member is entity A (supporter is the
        // high id) is sampled before a high-numbered pair whose lower member is
        // entity A (supporter is the low id).
        PhysicsShotRecorder recorder = new PhysicsShotRecorder(64, 64 * 1024);
        PhysicalContactInput upperMemberLowPair = new PhysicalContactInput(
            "100:0", 11, Vector2.zero, Vector2.up, 0f, Vector2.zero, 0f,
            "900:0", 91, new Vector2(0f, 1f), new Vector2(0f, 0f), false);
        PhysicalContactInput lowerMemberHighPair = new PhysicalContactInput(
            "200:0", 21, Vector2.zero, Vector2.up, 0f, Vector2.zero, 0f,
            "300:0", 31, new Vector2(0f, 0f), new Vector2(0f, 1f), false);
        recorder.RecordContacts(1, new[] { upperMemberLowPair, lowerMemberHighPair });
        recorder.RecordContacts(2, new[] { upperMemberLowPair, lowerMemberHighPair });
        Assert.AreEqual(2, recorder.SupportEdges.Count);
        Assert.AreEqual("900:0", recorder.SupportEdges[0].SupporterEntityId,
            "the fixture must append the high-id supporter first, or it is not the F2 geometry");

        recorder.FinalizeShot(true);
        PhysicalShotRecorderSnapshot snapshot = recorder.CreateFinalizedSnapshot();
        Assert.IsNotNull(snapshot);
        Assert.AreEqual(2, snapshot.SupportEdges.Count);
        Assert.AreEqual("200:0", snapshot.SupportEdges[0].SupporterEntityId);
        Assert.AreEqual("300:0", snapshot.SupportEdges[0].SupportedEntityId);
        Assert.AreEqual("support:200:0->300:0", SupportIdOf(snapshot.SupportEdges[0]));
        Assert.AreEqual("900:0", snapshot.SupportEdges[1].SupporterEntityId);
        Assert.AreEqual("100:0", snapshot.SupportEdges[1].SupportedEntityId);
        Assert.AreEqual("support:900:0->100:0", SupportIdOf(snapshot.SupportEdges[1]));
    }

    [Test]
    public void CollisionEvidence_CitesOnlyTheExactColliderPairNotSiblingPairs()
    {
        // F3: entity a:0 has two colliders; both pairs with b:0 were sampled by
        // the full-step sampler. A collision callback for only one collider pair
        // must cite that pair's contact ids, not its sibling's.
        PhysicsShotRecorder recorder = new PhysicsShotRecorder(32, 64 * 1024);
        recorder.RecordContacts(5, new[]
        {
            new PhysicalContactInput("a:0", 1, new Vector2(0f, 0f), Vector2.up, -0.1f,
                Vector2.left, 1f, "b:0", 2, Vector2.zero, Vector2.one, false),
            new PhysicalContactInput("a:0", 3, new Vector2(1f, 1f), Vector2.up, -0.1f,
                Vector2.left, 1f, "b:0", 2, Vector2.zero, Vector2.one, false)
        });
        Assert.AreEqual(2, recorder.RawContacts.Count);
        Assert.IsTrue(recorder.RawContacts.Any(c => c.PairKey == "a:0:1|b:0:2"));
        Assert.IsTrue(recorder.RawContacts.Any(c => c.PairKey == "a:0:3|b:0:2"));

        recorder.RecordCollision(5, 0.1f, "a:0", "b:0", new[]
        {
            new PhysicalContactInput("a:0", 1, new Vector2(0f, 0f), Vector2.up, -0.1f,
                Vector2.left, 1f, "b:0", 2, Vector2.zero, Vector2.one, false)
        }, 2f);

        Assert.AreEqual(1, recorder.Events.Count);
        CollectionAssert.AreEqual(
            new[] { "contact:5:a:0:1|b:0:2:0" },
            recorder.Events[0].Payload.ContactIds,
            "the event must cite only the collider pair the callback named");
    }

    [Test]
    public void BoundedRetention_AllowsLongSequencesAndPreservesCollisionCitedRows()
    {
        // F7: with retention, a long sequence of full samples stays under a low
        // record budget because only the last two fixed steps are kept; support
        // evidence still resolves (it cites the previous step); and a contact row
        // a collision event cited at step 1 survives every later prune. The 12
        // record budget is chosen so the sequence would trip RecordLimitExceeded
        // at step 10 without retention (raw contacts accumulate one row per step).
        PhysicsShotRecorder recorder = new PhysicsShotRecorder(12, 64 * 1024);
        PhysicalContactInput pair = new PhysicalContactInput(
            "a:0", 1, Vector2.zero, Vector2.up, 0f, Vector2.zero, 0f,
            "b:0", 2, new Vector2(0f, 0f), new Vector2(0f, 1f), false);

        recorder.RecordContacts(1, new[] { pair });
        string stepOneContactId = recorder.RawContacts[0].ContactId;
        recorder.RecordCollision(1, 0.02f, "a:0", "b:0", new[] { stepOneContactId }, 1f);
        Assert.AreEqual(1, recorder.Events.Count);

        for (long step = 2; step <= 20; step++)
            recorder.RecordContacts(step, new[] { pair });

        Assert.IsNull(recorder.Failure,
            "retention must keep a 20-step sequence under a 12-record budget");
        Assert.AreEqual(3, recorder.RawContacts.Count,
            "retention must keep the last two full steps plus the collision-cited row");
        Assert.IsTrue(recorder.RawContacts.Any(c => c.ContactId == stepOneContactId),
            "the collision-cited step-1 contact must survive pruning");
        Assert.IsTrue(recorder.RawContacts.Any(c => c.FixedStep >= 19),
            "the last two full steps must be retained");
        Assert.GreaterOrEqual(recorder.SupportEdges.Count, 1,
            "support evidence must still resolve after retention");

        recorder.FinalizeShot(true);
        PhysicalShotRecorderSnapshot snapshot = recorder.CreateFinalizedSnapshot();
        Assert.IsNotNull(snapshot);
        Assert.IsTrue(snapshot.RawContacts.Any(c => c.ContactId == stepOneContactId),
            "the cited row must reach the finalized snapshot");
        Assert.IsTrue(snapshot.SupportEdges.Any(edge => edge.PairKey == "a:0:1|b:0:2"),
            "the support edge citing the retained steps must reach the finalized snapshot");
    }

    [Test]
    public void ViolationEvidence_MinimumSurvivesOrdinaryContactPruning()
    {
        PhysicsShotRecorder recorder = new PhysicsShotRecorder(32, 64 * 1024);
        PhysicalContactInput deep = new PhysicalContactInput(
            "a:0", 1, Vector2.zero, Vector2.up, -2f, Vector2.zero, 0f,
            "b:0", 2, Vector2.zero, Vector2.one, false);
        PhysicalContactInput shallow = new PhysicalContactInput(
            "a:0", 1, Vector2.zero, Vector2.up, -0.1f, Vector2.zero, 0f,
            "b:0", 2, Vector2.zero, Vector2.one, false);

        recorder.RecordContacts(1, new[] { deep });
        string deepId = recorder.RawContacts[0].ContactId;
        recorder.RecordContacts(2, new[] { shallow });
        recorder.RecordContacts(3, new[] { shallow });
        Assert.IsFalse(recorder.RawContacts.Any(contact => contact.ContactId == deepId),
            "fixture did not prune the deep step-one contact");

        recorder.FinalizeShot(true);
        PhysicalViolationEngineEvidenceSnapshot evidence = recorder.CreateFinalizedEvidenceSnapshot();
        Assert.IsTrue(evidence.MinimumObserved);
        Assert.AreEqual(-2f, evidence.MinimumSeparation.Value);
        Assert.AreEqual(deepId, evidence.MinimumContactId);
        Assert.AreEqual(1L, evidence.MinimumFixedStep.Value);
    }

    [Test]
    public void ViolationEvidence_EntityOverflowIsExplicitlyIncomplete()
    {
        PhysicsShotRecorder recorder = new PhysicsShotRecorder(512, 1024 * 1024);
        PhysicalEntityRegistry registry = new PhysicalEntityRegistry();
        List<GameObject> objects = new List<GameObject>();
        try
        {
            for (int i = 0; i <= PhysicalViolationEngineEvidenceSnapshot.MaxEntitiesPerStep; i++)
            {
                GameObject item = new GameObject("evidence-body-" + i);
                item.AddComponent<Rigidbody2D>().gravityScale = 0f;
                objects.Add(item);
            }
            recorder.RecordUnityContacts(1, 0.02f, new Collider2D[0],
                objects.Select(item => item.GetComponent<Rigidbody2D>()).ToArray(), registry);
            recorder.FinalizeShot(true);

            PhysicalViolationEngineEvidenceSnapshot evidence = recorder.CreateFinalizedEvidenceSnapshot();
            Assert.IsFalse(evidence.Complete);
            Assert.AreEqual("entity_sample_overflow", evidence.IncompleteReason);
            Assert.AreEqual(PhysicalViolationEngineEvidenceSnapshot.MaxEntitiesPerStep,
                evidence.Trace[0].Entities.Count);
        }
        finally
        {
            foreach (GameObject item in objects) UnityEngine.Object.DestroyImmediate(item);
        }
    }

    [Test]
    public void ViolationEvidence_TerminalTraceIsBoundedAndConsecutive()
    {
        PhysicsShotRecorder recorder = new PhysicsShotRecorder(64, 64 * 1024);
        PhysicalEntityRegistry registry = new PhysicalEntityRegistry();
        GameObject item = new GameObject("trace-body");
        Rigidbody2D body = item.AddComponent<Rigidbody2D>();
        body.gravityScale = 0.25f;
        try
        {
            for (long step = 1; step <= 10; step++)
                recorder.RecordUnityContacts(step, step * 0.02f, new Collider2D[0],
                    new[] { body }, registry);
            recorder.FinalizeShot(true);

            PhysicalViolationEngineEvidenceSnapshot evidence = recorder.CreateFinalizedEvidenceSnapshot();
            Assert.IsTrue(evidence.TraceTruncated);
            Assert.AreEqual(8, evidence.Trace.Count);
            CollectionAssert.AreEqual(Enumerable.Range(3, 8).Select(value => (long)value).ToArray(),
                evidence.Trace.Select(sample => sample.FixedStep).ToArray());
            Assert.AreEqual("dynamic", evidence.Trace[7].Entities[0].BodyType);
            Assert.AreEqual(0.25f, evidence.Trace[7].Entities[0].GravityScale.Value);
        }
        finally
        {
            UnityEngine.Object.DestroyImmediate(item);
        }
    }

    [Test]
    public void ViolationEvidence_NoSamplesAndFixedStepGapsAreExplicitlyIncomplete()
    {
        PhysicsShotRecorder empty = new PhysicsShotRecorder(16, 64 * 1024);
        empty.FinalizeShot(true);
        Assert.AreEqual("no_fixed_step_samples",
            empty.CreateFinalizedEvidenceSnapshot().IncompleteReason);

        PhysicsShotRecorder gapped = new PhysicsShotRecorder(16, 64 * 1024);
        PhysicalEntityRegistry registry = new PhysicalEntityRegistry();
        GameObject item = new GameObject("gapped-trace-body");
        Rigidbody2D body = item.AddComponent<Rigidbody2D>();
        try
        {
            gapped.RecordUnityContacts(1, 0.02f, new Collider2D[0], new[] { body }, registry);
            gapped.RecordUnityContacts(3, 0.06f, new Collider2D[0], new[] { body }, registry);
            gapped.FinalizeShot(true);
            PhysicalViolationEngineEvidenceSnapshot evidence = gapped.CreateFinalizedEvidenceSnapshot();
            Assert.IsFalse(evidence.Complete);
            Assert.AreEqual("fixed_step_gap", evidence.IncompleteReason);
            Assert.AreEqual(2, evidence.SampleCount);
            CollectionAssert.AreEqual(new long[] { 3 },
                evidence.Trace.Select(sample => sample.FixedStep).ToArray());
            Assert.AreEqual(1, evidence.Trace[0].Entities.Count);
        }
        finally
        {
            UnityEngine.Object.DestroyImmediate(item);
        }
    }

    [Test]
    public void ViolationEvidence_PartialCollisionDoesNotAuthorMinimumWitness()
    {
        PhysicsShotRecorder recorder = new PhysicsShotRecorder(32, 64 * 1024);
        PhysicalContactInput partial = new PhysicalContactInput(
            "a:0", 1, Vector2.zero, Vector2.up, -5f, Vector2.zero, 0f,
            "b:0", 2, Vector2.zero, Vector2.one, false);
        PhysicalContactInput full = new PhysicalContactInput(
            "a:0", 1, Vector2.zero, Vector2.up, -0.25f, Vector2.zero, 0f,
            "b:0", 2, Vector2.zero, Vector2.one, false);

        recorder.RecordCollision(99, 1.98f, "a:0", "b:0", new[] { partial }, 1f);
        recorder.RecordContacts(1, 0.02f, new[] { full });
        string fullContactId = recorder.RawContacts.Single(contact => contact.FixedStep == 1).ContactId;
        recorder.FinalizeShot(true);

        PhysicalViolationEngineEvidenceSnapshot evidence = recorder.CreateFinalizedEvidenceSnapshot();
        Assert.AreEqual(1L, evidence.FirstFixedStep.Value);
        Assert.AreEqual(1L, evidence.LastFixedStep.Value);
        Assert.AreEqual(-0.25f, evidence.MinimumSeparation.Value);
        Assert.AreEqual(1L, evidence.MinimumFixedStep.Value);
        Assert.AreEqual(fullContactId, evidence.MinimumContactId);
    }

    [Test]
    public void ViolationEvidence_TraceCopiesAuthoritativeSupportFacts()
    {
        PhysicsShotRecorder recorder = new PhysicsShotRecorder(32, 64 * 1024);
        PhysicalEntityRegistry registry = new PhysicalEntityRegistry();
        GameObject bodyObject = new GameObject("supported-evidence-body");
        Rigidbody2D body = bodyObject.AddComponent<Rigidbody2D>();
        string supportedId = registry.RegisterObject(bodyObject);
        string supporterId = "world:static:900";
        PhysicalContactInput firstSupport = new PhysicalContactInput(
            supporterId, 900, Vector2.zero, Vector2.up, 0f, Vector2.zero, 0f,
            supportedId, 1000, Vector2.zero, Vector2.up, false);
        PhysicalContactInput secondSupport = new PhysicalContactInput(
            supporterId, 901, Vector2.right, Vector2.up, 0f, Vector2.zero, 0f,
            supportedId, 1001, Vector2.zero, Vector2.up, false);
        try
        {
            recorder.RecordContacts(1, new[] { secondSupport, firstSupport });
            recorder.RecordContacts(2, new[] { secondSupport, firstSupport });
            Assert.AreEqual(2, recorder.SupportEdges.Count,
                "fixture must exercise compound-collider support witnesses");
            PhysicalSupportEdge canonical = recorder.SupportEdges
                .OrderBy(edge => edge.PairKey, StringComparer.Ordinal)
                .ThenBy(edge => edge.ContactIdA, StringComparer.Ordinal).First();
            typeof(PhysicalShotRecorder).GetMethod("RecordEvidenceTrace",
                BindingFlags.Instance | BindingFlags.NonPublic).Invoke(
                    recorder, new object[] { 2L, new[] { body }, registry, Physics2D.gravity });
            recorder.FinalizeShot(true);

            PhysicalEvidenceEntity entity = recorder.CreateFinalizedEvidenceSnapshot()
                .Trace[0].Entities.Single(item => item.EntityId == supportedId);
            Assert.AreEqual(1, entity.Supports.Count);
            Assert.AreEqual("support:" + supporterId + "->" + supportedId,
                entity.Supports[0].SupportId);
            CollectionAssert.AreEqual(new long[] { 1, 2 }, new[] {
                entity.Supports[0].FixedStepA, entity.Supports[0].FixedStepB });
            Assert.AreEqual(canonical.ContactIdA, entity.Supports[0].ContactIdA);
            Assert.AreEqual(canonical.ContactIdB, entity.Supports[0].ContactIdB);
        }
        finally
        {
            UnityEngine.Object.DestroyImmediate(bodyObject);
        }
    }

    private static string SupportIdOf(PhysicalSupportEdge edge)
    {
        return "support:" + edge.SupporterEntityId + "->" + edge.SupportedEntityId;
    }

    private static void AssertFinalizedMutationIsNoOp(Action<PhysicsShotRecorder> mutation, string mutationName)
    {
        PhysicsShotRecorder recorder = new PhysicsShotRecorder(16, 64 * 1024);
        recorder.RecordLaunch("bird:1", 1, new Vector2(2f, 1f));
        recorder.FinalizeShot(true);
        PhysicalShotRecorderSnapshot before = recorder.CreateFinalizedSnapshot();

        mutation(recorder);

        AssertFinalizedStateUnchanged(recorder, before, mutationName);
    }

    private static void AssertFinalizedUnityContactsReturnBeforeRegistryMutation()
    {
        GameObject ground = new GameObject("finalized-ground");
        GameObject bodyObject = new GameObject("finalized-body");
        BoxCollider2D groundCollider = ground.AddComponent<BoxCollider2D>();
        BoxCollider2D bodyCollider = bodyObject.AddComponent<BoxCollider2D>();
        Rigidbody2D body = bodyObject.AddComponent<Rigidbody2D>();
        PhysicalEntityRegistry registry = new PhysicalEntityRegistry();
        PhysicsShotRecorder recorder = new PhysicsShotRecorder(16, 64 * 1024);
        bool autoSimulation = Physics2D.autoSimulation;

        try
        {
            groundCollider.size = new Vector2(4f, 1f);
            bodyObject.transform.position = new Vector2(0f, 0.9f);
            body.gravityScale = 0f;
            body.constraints = RigidbodyConstraints2D.FreezeAll;
            Physics2D.autoSimulation = false;
            Physics2D.SyncTransforms();
            Physics2D.Simulate(0.02f);
            Physics2D.Simulate(0.02f);
            Assert.Greater(bodyCollider.GetContacts(new ContactPoint2D[4]), 0,
                "finalized Unity-contact fixture did not produce an authoritative contact");
            recorder.RecordLaunch("bird:1", 1, Vector2.right);
            recorder.FinalizeShot(true);
            PhysicalShotRecorderSnapshot before = recorder.CreateFinalizedSnapshot();

            recorder.RecordUnityContacts(
                2, 0.04f, new Collider2D[] { groundCollider, bodyCollider }, registry);

            System.Collections.IDictionary lifetimes = (System.Collections.IDictionary)typeof(PhysicalEntityRegistry)
                .GetField("currentLifetimes", BindingFlags.Instance | BindingFlags.NonPublic).GetValue(registry);
            Assert.AreEqual(0, lifetimes.Count,
                "finalized Unity contacts traversed contacts and mutated the entity registry");
            AssertFinalizedStateUnchanged(recorder, before, "Unity contacts");
        }
        finally
        {
            Physics2D.autoSimulation = autoSimulation;
            UnityEngine.Object.DestroyImmediate(bodyObject);
            UnityEngine.Object.DestroyImmediate(ground);
        }
    }

    private static void AssertFinalizedStateUnchanged(
        PhysicsShotRecorder recorder, PhysicalShotRecorderSnapshot before, string mutationName)
    {
        PhysicalShotRecorderSnapshot after = recorder.CreateFinalizedSnapshot();
        Assert.IsNull(recorder.Failure, mutationName + " changed finalized failure state");
        Assert.IsNotNull(after, mutationName + " invalidated the finalized snapshot");
        Assert.AreEqual(before.RawContacts.Count, after.RawContacts.Count, mutationName + " changed contacts");
        Assert.AreEqual(before.SupportEdges.Count, after.SupportEdges.Count, mutationName + " changed supports");
        Assert.AreEqual(before.Events.Count, after.Events.Count, mutationName + " changed events");
        Assert.AreEqual(before.Events[0].Subject, after.Events[0].Subject, mutationName + " changed snapshot content");
        Assert.AreEqual(before.Events[0].Payload.LaunchVelocity, after.Events[0].Payload.LaunchVelocity,
            mutationName + " changed snapshot payload");
    }

    private static PhysicalContactInput ContactWithCenters(float centerAY, float centerBY)
    {
        return new PhysicalContactInput(
            "a:0", 1, Vector2.zero, Vector2.up, 0f, Vector2.zero, 0f,
            "b:0", 2, new Vector2(0f, centerAY), new Vector2(0f, centerBY), false);
    }
}
