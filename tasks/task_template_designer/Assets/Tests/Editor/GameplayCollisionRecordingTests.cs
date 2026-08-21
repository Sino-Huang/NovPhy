using System;
using System.Linq;
using System.Reflection;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.TestTools;

/// <summary>
/// Covers the gameplay-callback -> recorder seam that no other fixture reaches.
///
/// PhysicsCaptureProtocolTests covers the recorder -> wire seam and says so
/// explicitly; it feeds PhysicalContactInput[] in directly, so it never exercises
/// the Collision2D -> PhysicalContactInput[] conversion in PhysicalSnapshotRuntime
/// nor the isTrigger filter there. On the level the smoke plays, that conversion
/// is never invoked at all, because the only object types a bird can reach —
/// BirdBlack itself and the wood blocks — did not call the recorder. These
/// fixtures pin that they now do.
///
/// Collision2D and ContactPoint2D expose no constructor a test can populate, so
/// they are synthesized by reflection over the engine's internal fields. The
/// field names were read out of this editor's own UnityEngine.Physics2DModule.dll
/// before this file was written, and every lookup below asserts non-null, so a
/// Unity version bump fails loudly here instead of silently handing the product
/// code an unpopulated collision.
/// </summary>
public class GameplayCollisionRecordingTests
{
    // Both damage formulas subtract _defense, so a defense between their two
    // outputs is what makes them tell apart. At relativeSpeed 4.25 with unit
    // masses and the wood multiplier of 1.0:
    //   ABBlock:      4.25 * 1 * 1.0 * 11.22        - 47 =  0.685 damage
    //   ABGameObject: 4.25 * 1 * 11.22 * 0.5        - 47 = clamped to 0
    // 47 also keeps damage under the 10-point score threshold, so the score path
    // and its ScoreHud singleton — absent in EditMode — are never entered.
    private const float BlockDefense = 47f;
    private const float BirdOnBlockRelativeSpeed = 4.25f;
    private const float BirdOnBlockLifeAfterBlockFormula = 10000f - 0.685f;

    [Test]
    public void V2OnlyRuntimeRecordsCollisionWhileV1ShotRecorderIsNull()
    {
        string previousStride = V2Stride();
        GameObject first = CausalCollisionObject("v2-only-first");
        GameObject second = CausalCollisionObject("v2-only-second");
        GameObject runtimeObject = null;
        try
        {
            SetV2Stride("1");
            runtimeObject = NewV2OnlyRuntime();
            PhysicalSnapshotRuntime runtime =
                runtimeObject.GetComponent<PhysicalSnapshotRuntime>();
            Collision2D collision = NewCollision(first.GetComponent<BoxCollider2D>(),
                second.GetComponent<BoxCollider2D>(), new Vector2(2f, 0f), 2);

            PhysicalSnapshotRuntime.RecordCollisionCallback(collision);
            runtime.FinalizeTerminal();

            Assert.IsNull(runtime.ShotRecorder);
            PhysicsCaptureV2EngineSnapshot snapshot =
                PhysicsCaptureV2FixedStepRecorder.Active.CreateFinalizedSnapshot();
            Assert.AreEqual(1, snapshot.Events.Count(item => item.EventType == "collision"));
            PhysicsCaptureV2EventSnapshot collisionEvent =
                snapshot.Events.Single(item => item.EventType == "collision");
            PhysicsCaptureV2FixedStepSample sample = snapshot.FixedStepSamples.Single(
                item => item.FixedStep == collisionEvent.FixedStep);
            Assert.AreEqual(2, sample.Contacts.Count);
            CollectionAssert.AreEquivalent(collisionEvent.Participants,
                new[] { sample.Contacts[0].EntityAId, sample.Contacts[0].EntityBId });
        }
        finally
        {
            if (runtimeObject != null) UnityEngine.Object.DestroyImmediate(runtimeObject);
            UnityEngine.Object.DestroyImmediate(second);
            UnityEngine.Object.DestroyImmediate(first);
            SetV2Stride(previousStride);
        }
    }

    [Test]
    public void LethalPigCallbackFreezesSameStepContactBeforeColliderDisablement()
    {
        string previousStride = V2Stride();
        GameObject recorderHost = new GameObject("lethal-pig-v2-recorder");
        GameObject bird = CausalCollisionObject("bird:lethal");
        GameObject pig = CausalCollisionObject("pig:lethal");
        try
        {
            SetV2Stride("1");
            PhysicsCaptureV2FixedStepRecorder recorder =
                recorderHost.AddComponent<PhysicsCaptureV2FixedStepRecorder>();
            GameObject[] causalObjects = { bird, pig };
            recorder.BeginPreIntervention(10, causalObjects);
            Collision2D collision = NewCollision(bird.GetComponent<BoxCollider2D>(),
                pig.GetComponent<BoxCollider2D>(), new Vector2(6f, 0f), 2);

            ((IPhysicsCaptureV2UnityCollisionRecorder)recorder).RecordUnityCollision(
                11, collision, causalObjects);
            pig.GetComponent<BoxCollider2D>().enabled = false;
            recorder.RecordFixedStep(11, causalObjects,
                new PhysicsCaptureV2ContactInput[0], true);
            recorder.FinalizeTerminal(11);

            PhysicsCaptureV2EngineSnapshot snapshot = recorder.CreateFinalizedSnapshot();
            PhysicsCaptureV2EventSnapshot collisionEvent =
                snapshot.Events.Single(item => item.EventType == "collision");
            Assert.AreEqual(11, collisionEvent.FixedStep);
            PhysicsCaptureV2FixedStepSample collisionSample =
                snapshot.FixedStepSamples.Single(item => item.FixedStep == 11);
            Assert.AreEqual(2, collisionSample.Contacts.Count,
                "the callback contacts must survive the lethal collider disablement");
            Assert.IsFalse(collisionSample.Colliders.Single(
                item => item.EntityId == "runtime:pig:lethal").Enabled);
        }
        finally
        {
            UnityEngine.Object.DestroyImmediate(pig);
            UnityEngine.Object.DestroyImmediate(bird);
            UnityEngine.Object.DestroyImmediate(recorderHost);
            SetV2Stride(previousStride);
        }
    }

    [Test]
    public void V2DuplicateBodyCallbacksProduceOneEventAndUniqueContactPoints()
    {
        string previousStride = V2Stride();
        GameObject recorderHost = new GameObject("duplicate-callback-v2-recorder");
        GameObject first = CausalCollisionObject("duplicate:first");
        GameObject second = CausalCollisionObject("duplicate:second");
        try
        {
            SetV2Stride("1");
            PhysicsCaptureV2FixedStepRecorder recorder =
                recorderHost.AddComponent<PhysicsCaptureV2FixedStepRecorder>();
            GameObject[] causalObjects = { first, second };
            recorder.BeginPreIntervention(10, causalObjects);
            Collision2D collision = NewCollision(first.GetComponent<BoxCollider2D>(),
                second.GetComponent<BoxCollider2D>(), new Vector2(2f, 0f), 2);
            IPhysicsCaptureV2UnityCollisionRecorder callbackRecorder = recorder;

            callbackRecorder.RecordUnityCollision(11, collision, causalObjects);
            callbackRecorder.RecordUnityCollision(11, collision, causalObjects);
            recorder.RecordFixedStep(11, causalObjects,
                new PhysicsCaptureV2ContactInput[0], true);
            recorder.FinalizeTerminal(11);

            PhysicsCaptureV2EngineSnapshot snapshot = recorder.CreateFinalizedSnapshot();
            Assert.AreEqual(1, snapshot.Events.Count(item => item.EventType == "collision"));
            Assert.AreEqual(2, snapshot.FixedStepSamples.Single(
                item => item.FixedStep == 11).Contacts.Count);
        }
        finally
        {
            UnityEngine.Object.DestroyImmediate(second);
            UnityEngine.Object.DestroyImmediate(first);
            UnityEngine.Object.DestroyImmediate(recorderHost);
            SetV2Stride(previousStride);
        }
    }

    [Test]
    public void V2CollisionCallbacksFailClosedForTriggersUnresolvedIdentitiesAndLimits()
    {
        string previousStride = V2Stride();
        GameObject first = CausalCollisionObject("callback-limit:first");
        GameObject second = CausalCollisionObject("callback-limit:second");
        GameObject outsider = new GameObject("callback-unresolved");
        GameObject triggerHost = new GameObject("callback-trigger-recorder");
        GameObject unresolvedHost = new GameObject("callback-unresolved-recorder");
        GameObject limitHost = new GameObject("callback-limit-recorder");
        try
        {
            SetV2Stride("1");
            outsider.AddComponent<BoxCollider2D>();
            GameObject[] causalObjects = { first, second };

            PhysicsCaptureV2FixedStepRecorder trigger =
                triggerHost.AddComponent<PhysicsCaptureV2FixedStepRecorder>();
            trigger.BeginPreIntervention(10, causalObjects);
            first.GetComponent<BoxCollider2D>().isTrigger = true;
            ((IPhysicsCaptureV2UnityCollisionRecorder)trigger).RecordUnityCollision(10,
                NewCollision(first.GetComponent<BoxCollider2D>(),
                    second.GetComponent<BoxCollider2D>(), Vector2.right, 1), causalObjects);
            trigger.FinalizeTerminal(10);
            Assert.AreEqual(0, trigger.CreateFinalizedSnapshot().Events.Count(
                item => item.EventType == "collision"));
            first.GetComponent<BoxCollider2D>().isTrigger = false;

            PhysicsCaptureV2FixedStepRecorder unresolved =
                unresolvedHost.AddComponent<PhysicsCaptureV2FixedStepRecorder>();
            unresolved.BeginPreIntervention(10, causalObjects);
            ((IPhysicsCaptureV2UnityCollisionRecorder)unresolved).RecordUnityCollision(10,
                NewCollision(outsider.GetComponent<BoxCollider2D>(),
                    second.GetComponent<BoxCollider2D>(), Vector2.right, 1), causalObjects);
            Assert.AreEqual(PhysicsCaptureV2EngineFailureCode.UnresolvedContactIdentity,
                unresolved.Failure.Code);

            PhysicsCaptureV2FixedStepRecorder limited =
                limitHost.AddComponent<PhysicsCaptureV2FixedStepRecorder>();
            limited.BeginPreIntervention(10, causalObjects,
                new PhysicsCaptureV2ContactInput[0], true,
                new PhysicsCaptureV2CaptureLimits(10, 10, 1));
            ((IPhysicsCaptureV2UnityCollisionRecorder)limited).RecordUnityCollision(10,
                NewCollision(first.GetComponent<BoxCollider2D>(),
                    second.GetComponent<BoxCollider2D>(), Vector2.right, 2), causalObjects);
            Assert.AreEqual(PhysicsCaptureV2EngineFailureCode.ContactLimitExceeded,
                limited.Failure.Code);
            Assert.IsNull(limited.CreateFinalizedSnapshot(),
                "a callback limit failure must not yield a finalized wire record");
        }
        finally
        {
            UnityEngine.Object.DestroyImmediate(limitHost);
            UnityEngine.Object.DestroyImmediate(unresolvedHost);
            UnityEngine.Object.DestroyImmediate(triggerHost);
            UnityEngine.Object.DestroyImmediate(outsider);
            UnityEngine.Object.DestroyImmediate(second);
            UnityEngine.Object.DestroyImmediate(first);
            SetV2Stride(previousStride);
        }
    }

    [Test]
    public void BirdBlackImpactRecordsACollisionEventWithContractGradeEvidence()
    {
        GameObject runtimeObject = NewRuntime();
        PhysicalSnapshotRuntime runtime = runtimeObject.GetComponent<PhysicalSnapshotRuntime>();
        GameObject birdObject = new GameObject("birdblack-impact-bird");
        GameObject blockObject = new GameObject("birdblack-impact-target");

        try
        {
            BoxCollider2D blockCollider = blockObject.AddComponent<BoxCollider2D>();
            ABBirdBlack bird = NewBirdBlack(birdObject);
            BoxCollider2D birdCollider = birdObject.GetComponent<BoxCollider2D>();

            // Unity hands this object's own collider as `otherCollider` and the
            // incoming one as `collider`, so the bird's own collider is `other`.
            Collision2D collision = NewCollision(blockCollider, birdCollider, new Vector2(3.5f, -1.5f), 2);

            bird.OnCollisionEnter2D(collision);

            AssertOneContractGradeCollision(runtime, new Vector2(3.5f, -1.5f).magnitude);
        }
        finally
        {
            UnityEngine.Object.DestroyImmediate(blockObject);
            UnityEngine.Object.DestroyImmediate(birdObject);
            UnityEngine.Object.DestroyImmediate(runtimeObject);
        }
    }

    [Test]
    public void BirdOnBlockImpactRecordsACollisionEventWithContractGradeEvidence()
    {
        GameObject runtimeObject = NewRuntime();
        PhysicalSnapshotRuntime runtime = runtimeObject.GetComponent<PhysicalSnapshotRuntime>();
        GameObject birdObject = new GameObject("block-impact-bird");
        GameObject blockObject = new GameObject("block-impact-block");

        try
        {
            ABBirdBlack bird = NewBirdBlack(birdObject);
            BoxCollider2D birdCollider = birdObject.GetComponent<BoxCollider2D>();
            // The bird branch of ABBlock.OnCollisionEnter2D is selected by tag.
            birdObject.tag = "Bird";
            ABBlock block = NewBlock(blockObject);
            BoxCollider2D blockCollider = blockObject.GetComponent<BoxCollider2D>();

            Collision2D collision = NewCollision(birdCollider, blockCollider, new Vector2(BirdOnBlockRelativeSpeed, 0f), 2);

            block.OnCollisionEnter2D(collision);

            AssertOneContractGradeCollision(runtime, BirdOnBlockRelativeSpeed);
            // The damage model must still run and must still be the block's own,
            // not ABGameObject's: hoisting `base` would have changed gameplay.
            // The two formulas land on different lives here by construction, so
            // this assertion fails if the hoist ever becomes a `base` call.
            Assert.AreEqual(BirdOnBlockLifeAfterBlockFormula, block.getCurrentLife(), 1e-2f,
                "the bird branch must still deal ABBlock's own damage; ABGameObject's formula "
                + "clamps to zero here and would leave the block at 10000");
            Assert.IsNotNull(bird, "the bird component is the branch selector and must survive the call");
        }
        finally
        {
            UnityEngine.Object.DestroyImmediate(blockObject);
            UnityEngine.Object.DestroyImmediate(birdObject);
            UnityEngine.Object.DestroyImmediate(runtimeObject);
        }
    }

    [Test]
    public void NonBirdImpactOnBlockRecordsExactlyOneCollisionEvent()
    {
        // The else branch still calls base.OnCollisionEnter2D, which records too.
        // The recorder dedupes on fixedStep:first:second, so the hoisted call must
        // not double-count. This is the regression the hoist could have introduced.
        GameObject runtimeObject = NewRuntime();
        PhysicalSnapshotRuntime runtime = runtimeObject.GetComponent<PhysicalSnapshotRuntime>();
        GameObject otherObject = new GameObject("nonbird-impact-other");
        GameObject blockObject = new GameObject("nonbird-impact-block");

        try
        {
            otherObject.AddComponent<Rigidbody2D>();
            BoxCollider2D otherCollider = otherObject.AddComponent<BoxCollider2D>();
            ABBlock block = NewBlock(blockObject);
            BoxCollider2D blockCollider = blockObject.GetComponent<BoxCollider2D>();

            Collision2D collision = NewCollision(otherCollider, blockCollider, new Vector2(1.5f, 0f), 2);

            block.OnCollisionEnter2D(collision);

            AssertOneContractGradeCollision(runtime, 1.5f);
            // Dedupe must suppress the second call whole, not just its event: the
            // collision path also ingests the contacts it was handed, so a dedupe
            // that ran after that ingestion would double the raw contact stream
            // while still reporting one event.
            Assert.AreEqual(2, runtime.ShotRecorder.RawContacts.Count,
                "the deduped second call must not re-ingest the collision's two contacts");
        }
        finally
        {
            UnityEngine.Object.DestroyImmediate(blockObject);
            UnityEngine.Object.DestroyImmediate(otherObject);
            UnityEngine.Object.DestroyImmediate(runtimeObject);
        }
    }

    [Test]
    public void BirdBlackImpactWithoutUsableContactEvidenceCompletesTheHandler()
    {
        // Every contact on this collision sits on a trigger collider, so
        // PhysicalSnapshotRuntime filters the array empty and the recorder is
        // handed a collision it must refuse. Refusing by throwing would abort the
        // rest of OnCollisionEnter2D inside a Unity physics callback: the
        // BirdBlack explosion would never play, no terminal event would be
        // reached, and the smoke's 30 s finalize deadline would expire with the
        // single non-retryable run consumed. The refusal must be fail-closed at
        // the wire — no event emitted — without destroying the frame.
        GameObject runtimeObject = NewRuntime();
        PhysicalSnapshotRuntime runtime = runtimeObject.GetComponent<PhysicalSnapshotRuntime>();
        GameObject birdObject = new GameObject("empty-evidence-bird");
        GameObject blockObject = new GameObject("empty-evidence-target");

        try
        {
            BoxCollider2D blockCollider = blockObject.AddComponent<BoxCollider2D>();
            blockCollider.isTrigger = true;
            ABBirdBlack bird = NewBirdBlack(birdObject);
            BoxCollider2D birdCollider = birdObject.GetComponent<BoxCollider2D>();
            ABParticleSystem trail = TrailParticles(bird);
            trail._shootParticles = true;

            Collision2D collision = NewCollision(blockCollider, birdCollider, new Vector2(3.5f, 0f), 2);

            LogAssert.Expect(LogType.Error, CollisionEvidenceRejectionPattern());
            Assert.DoesNotThrow(delegate { bird.OnCollisionEnter2D(collision); },
                "an empty-evidence refusal must not abort the physics handler");

            Assert.AreEqual(0, runtime.ShotRecorder.Events.Count(e => e.Taxonomy == "collision"),
                "the refusal must stay fail-closed at the wire: no collision event may be emitted");
            Assert.AreEqual(0, runtime.ShotRecorder.RawContacts.Count,
                "a refused collision must not half-mutate the recorder");
            Assert.IsFalse(trail._shootParticles,
                "the statement after the recorder call must still have run");
        }
        finally
        {
            UnityEngine.Object.DestroyImmediate(blockObject);
            UnityEngine.Object.DestroyImmediate(birdObject);
            UnityEngine.Object.DestroyImmediate(runtimeObject);
        }
    }

    [Test]
    public void CallbackBoundary_ContainsARecorderPathExceptionAndLogsTheStableError()
    {
        // F4: the static physics callback seam must never let a recorder-path
        // failure escape into Unity. The recorder path dereferences the entity
        // registry; a broken runtime state is exactly the unforeseen defect the
        // no-throw boundary exists to contain. It must be logged with the stable
        // physics_capture_v1 refusal prefix and swallowed at the boundary, and
        // nothing may be half-recorded.
        GameObject runtimeObject = NewRuntime();
        PhysicalSnapshotRuntime runtime = runtimeObject.GetComponent<PhysicalSnapshotRuntime>();
        GameObject a = new GameObject("callback-boundary-a");
        GameObject b = new GameObject("callback-boundary-b");

        try
        {
            BoxCollider2D aCollider = a.AddComponent<BoxCollider2D>();
            BoxCollider2D bCollider = b.AddComponent<BoxCollider2D>();

            // Sabotage the recorder path the same way an unforeseen defect would:
            // drop the registry the callback dereferences, so the collision
            // conversion throws inside Active.RecordCollision.
            FieldInfo registryField = typeof(PhysicalSnapshotRuntime).GetField(
                "registry", BindingFlags.Instance | BindingFlags.NonPublic);
            Assert.IsNotNull(registryField, "PhysicalSnapshotRuntime.registry gates the recorder path");
            registryField.SetValue(runtime, null);

            Collision2D collision = NewCollision(aCollider, bCollider, new Vector2(1f, 0f), 1);

            LogAssert.Expect(LogType.Error, CallbackBoundaryErrorPattern());
            Assert.DoesNotThrow(delegate { PhysicalSnapshotRuntime.RecordCollisionCallback(collision); },
                "a recorder-path exception must not escape the static physics callback boundary");

            Assert.AreEqual(0, runtime.ShotRecorder.Events.Count,
                "a contained failure must stay fail-closed at the wire: no event may be emitted");
            Assert.AreEqual(0, runtime.ShotRecorder.RawContacts.Count,
                "a contained failure must not half-mutate the recorder");
        }
        finally
        {
            UnityEngine.Object.DestroyImmediate(b);
            UnityEngine.Object.DestroyImmediate(a);
            UnityEngine.Object.DestroyImmediate(runtimeObject);
        }
    }

    [Test]
    public void EggImpactRecordsACollisionEventAndCompletesTheHandler()
    {
        // ABEgg previously reached the recorder on no path at all: its
        // OnCollisionEnter2D overrides the base without calling it. The callback
        // must now be the first statement so gameplay order is preserved --
        // explosion and death still run after the recorder call, which is what
        // the disabled-collider assertion below proves.
        GameObject runtimeObject = NewRuntime();
        PhysicalSnapshotRuntime runtime = runtimeObject.GetComponent<PhysicalSnapshotRuntime>();
        GameObject worldObject = new GameObject("egg-world");
        ABGameWorld world = worldObject.AddComponent<ABGameWorld>();
        world._isSimulation = true;
        GameObject eggObject = new GameObject("egg-impact-egg");
        GameObject targetObject = new GameObject("egg-impact-target");

        try
        {
            Rigidbody2D eggBody = eggObject.AddComponent<Rigidbody2D>();
            eggBody.gravityScale = 0f;
            eggObject.AddComponent<BoxCollider2D>();
            eggObject.AddComponent<SpriteRenderer>();
            ABParticleSystem particles = eggObject.AddComponent<ABParticleSystem>();
            ABEgg egg = eggObject.AddComponent<ABEgg>();
            egg._explosionArea = 1f;
            AwakeComponent(egg);
            Assert.IsNotNull(particles, "the egg's destroy effect must be bound by ABGameObject.Awake");

            Rigidbody2D targetBody = targetObject.AddComponent<Rigidbody2D>();
            targetBody.gravityScale = 0f;
            BoxCollider2D targetCollider = targetObject.AddComponent<BoxCollider2D>();
            targetObject.transform.position = eggObject.transform.position;
            BoxCollider2D eggCollider = eggObject.GetComponent<BoxCollider2D>();
            Assert.IsNotNull(eggCollider, "the egg fixture must carry the collider ABEgg.Awake binds");

            Collision2D collision = NewCollision(eggCollider, targetCollider, new Vector2(2f, 0f), 2);

            Assert.DoesNotThrow(delegate { egg.OnCollisionEnter2D(collision); },
                "the egg's collision handler must complete after recording");

            AssertOneContractGradeCollision(runtime, 2f);
            Assert.AreEqual(1, runtime.ShotRecorder.Events.Count(e => e.Taxonomy == "entity_destroyed"),
                "the egg's death must still be recorded after the collision");
            Assert.IsFalse(eggCollider.enabled,
                "the statements after the recorder call — explosion and Die — must still have run");
        }
        finally
        {
            UnityEngine.Object.DestroyImmediate(targetObject);
            UnityEngine.Object.DestroyImmediate(eggObject);
            UnityEngine.Object.DestroyImmediate(worldObject);
            UnityEngine.Object.DestroyImmediate(runtimeObject);
        }
    }

    // ---- assertions -------------------------------------------------------

    private static void AssertOneContractGradeCollision(PhysicalSnapshotRuntime runtime, float relativeSpeed)
    {
        PhysicalMacroEvent[] collisions = runtime.ShotRecorder.Events
            .Where(e => e.Taxonomy == "collision").ToArray();
        Assert.AreEqual(1, collisions.Length, "expected exactly one recorded collision event");

        System.Collections.Generic.IList<string> contactIds = collisions[0].Payload.ContactIds;
        Assert.IsNotNull(contactIds, "collision payload carries no contact_ids");
        Assert.GreaterOrEqual(contactIds.Count, 1, "collision payload carries no contact evidence");
        for (int i = 0; i < contactIds.Count; i++)
            Assert.IsNotEmpty(contactIds[i], "collision payload carries an empty contact id");
        for (int i = 1; i < contactIds.Count; i++)
            Assert.Less(string.CompareOrdinal(contactIds[i - 1], contactIds[i]), 0,
                "collision contact ids are not sorted unique by ordinal");

        Assert.IsTrue(collisions[0].Payload.RelativeSpeed.HasValue, "collision payload carries no relative_speed");
        float speed = collisions[0].Payload.RelativeSpeed.Value;
        Assert.IsFalse(float.IsNaN(speed) || float.IsInfinity(speed), "relative_speed must be finite");
        Assert.GreaterOrEqual(speed, 0f, "relative_speed must be non-negative");
        Assert.AreEqual(relativeSpeed, speed, 1e-4f, "relative_speed must be the collision's own magnitude");
    }

    /// <summary>
    /// Matched against Debug.LogError output, which UnityEngine.TestTools compares
    /// as a regex when given one. Kept in one place so the product message and the
    /// fixture cannot drift apart silently.
    /// </summary>
    private static System.Text.RegularExpressions.Regex CollisionEvidenceRejectionPattern()
    {
        return new System.Text.RegularExpressions.Regex("physics_capture_v1.*contact evidence");
    }

    /// <summary>
    /// The F4 callback-boundary refusal, matched the same way. The prefix is the
    /// same stable string the smoke's engine-log scan matches, so the product
    /// message and this fixture cannot drift apart silently.
    /// </summary>
    private static System.Text.RegularExpressions.Regex CallbackBoundaryErrorPattern()
    {
        return new System.Text.RegularExpressions.Regex("physics_capture_v1: refusing a recorder exception inside a physics callback");
    }

    // ---- scene construction ----------------------------------------------

    private static GameObject NewRuntime()
    {
        GameObject runtimeObject = new GameObject("gameplay-collision-runtime");
        PhysicalSnapshotRuntime runtime = runtimeObject.AddComponent<PhysicalSnapshotRuntime>();
        // PhysicalSnapshotRuntime.Active is bound in Awake and the static record
        // callbacks no-op while it is null, so the runtime needs its lifecycle
        // before BeginShot. EditMode never raises Awake for AddComponent.
        AwakeComponent(runtime);
        runtime.BeginShot(32, 64 * 1024, 30f);
        return runtimeObject;
    }

    private static GameObject NewV2OnlyRuntime()
    {
        GameObject runtimeObject = new GameObject("gameplay-collision-v2-only-runtime");
        PhysicalSnapshotRuntime runtime = runtimeObject.AddComponent<PhysicalSnapshotRuntime>();
        AwakeComponent(runtime);
        runtime.BeginV2Shot();
        return runtimeObject;
    }

    private static GameObject CausalCollisionObject(string scenarioObjectId)
    {
        GameObject value = new GameObject(scenarioObjectId);
        ScenarioObjectIdentity.Assign(value, scenarioObjectId);
        value.AddComponent<Rigidbody2D>().bodyType = RigidbodyType2D.Static;
        value.AddComponent<BoxCollider2D>();
        return value;
    }

    private static string V2Stride()
    {
        return Environment.GetEnvironmentVariable(
            PhysicsCaptureV2EngineProtocol.StrideEnvironmentVariable,
            EnvironmentVariableTarget.Process);
    }

    private static void SetV2Stride(string value)
    {
        Environment.SetEnvironmentVariable(
            PhysicsCaptureV2EngineProtocol.StrideEnvironmentVariable,
            value, EnvironmentVariableTarget.Process);
    }

    private static ABBirdBlack NewBirdBlack(GameObject host)
    {
        host.AddComponent<Rigidbody2D>();
        host.AddComponent<BoxCollider2D>();
        ABBirdBlack bird = host.AddComponent<ABBirdBlack>();
        // Runs the real initialisation chain — ABBirdBlack declares no Awake, so
        // this resolves to ABGameObject's, which binds the collider and body.
        AwakeComponent(bird);
        // ABBird.Start, not Awake, is what creates _trailParticles; Start also
        // does Resources.Load and schedules an Invoke, neither of which works in
        // EditMode. Only the one field OnCollisionEnter2D dereferences is set
        // here, exactly as Start sets it.
        FieldInfo trailField = TrailField();
        trailField.SetValue(bird, host.AddComponent<ABParticleSystem>());
        return bird;
    }

    private static ABBlock NewBlock(GameObject host)
    {
        host.AddComponent<Rigidbody2D>();
        host.AddComponent<BoxCollider2D>();
        ABBlock block = host.AddComponent<ABBlock>();
        // ABBlock.Awake calls SetMaterial, which reaches ABWorldAssets and the
        // sprite table; neither is loadable in EditMode. Awake is also virtual, so
        // invoking ABGameObject's declaration by reflection still dispatches to
        // ABBlock's — the base half cannot be run on its own. The one field that
        // half would set and this test needs is bound directly instead.
        //
        // _rigidBody is not optional here: it is the gate on ABGameObject's damage
        // model. Leave it null and the base handler computes nothing, which would
        // make the "the hoist did not swap in base" assertion pass no matter which
        // formula ran.
        FieldInfo bodyField = typeof(ABGameObject).GetField(
            "_rigidBody", BindingFlags.Instance | BindingFlags.NonPublic);
        Assert.IsNotNull(bodyField, "ABGameObject._rigidBody gates the base damage model");
        bodyField.SetValue(block, host.GetComponent<Rigidbody2D>());
        //   _sprites  — DealDamage dereferences _sprites.Length unconditionally.
        //   _defense  — chosen to sit between the two formulas' outputs, see
        //               BlockDefense.
        block._sprites = new Sprite[1];
        block._defense = BlockDefense;
        return block;
    }

    private static FieldInfo TrailField()
    {
        FieldInfo field = typeof(ABBird).GetField(
            "_trailParticles", BindingFlags.Instance | BindingFlags.NonPublic);
        Assert.IsNotNull(field, "ABBird._trailParticles is the observable that proves the handler continued");
        return field;
    }

    private static ABParticleSystem TrailParticles(ABBird bird)
    {
        ABParticleSystem trail = (ABParticleSystem)TrailField().GetValue(bird);
        Assert.IsNotNull(trail, "the fixture must have supplied the trail particle system");
        return trail;
    }

    private static void AwakeComponent(MonoBehaviour component)
    {
        for (Type type = component.GetType(); type != null; type = type.BaseType)
        {
            MethodInfo awake = type.GetMethod(
                "Awake", BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.DeclaredOnly);
            if (awake == null) continue;
            awake.Invoke(component, null);
            return;
        }
        Assert.Fail("no Awake found on " + component.GetType().Name + " or its base types");
    }

    // ---- Collision2D synthesis -------------------------------------------

    private static Collision2D NewCollision(Collider2D incoming, Collider2D self, Vector2 relativeVelocity, int contactCount)
    {
        Collision2D collision = (Collision2D)Activator.CreateInstance(typeof(Collision2D), true);
        ContactPoint2D[] contacts = new ContactPoint2D[contactCount];
        for (int i = 0; i < contactCount; i++)
            contacts[i] = NewContact(incoming, self, i, relativeVelocity);

        SetInternal(collision, "m_Collider", incoming.GetInstanceID());
        SetInternal(collision, "m_OtherCollider", self.GetInstanceID());
        SetInternal(collision, "m_Rigidbody", InstanceIdOfBody(incoming));
        SetInternal(collision, "m_OtherRigidbody", InstanceIdOfBody(self));
        SetInternal(collision, "m_RelativeVelocity", relativeVelocity);
        SetInternal(collision, "m_Enabled", 1);
        SetInternal(collision, "m_ContactCount", contactCount);
        SetInternal(collision, "m_ReusedContacts", contacts);
        // `contacts` lazily copies from m_ReusedContacts only while this is null;
        // setting it makes the array the property returns exactly the one built here.
        SetInternal(collision, "m_LegacyContacts", contacts);

        Assert.AreSame(incoming, collision.collider, "synthesized collision did not resolve its incoming collider");
        Assert.AreSame(self, collision.otherCollider, "synthesized collision did not resolve its own collider");
        Assert.AreEqual(contactCount, collision.contacts.Length, "synthesized collision did not expose its contacts");
        return collision;
    }

    private static ContactPoint2D NewContact(Collider2D incoming, Collider2D self, int index, Vector2 relativeVelocity)
    {
        object boxed = new ContactPoint2D();
        SetInternal(boxed, "m_Point", new Vector2(0.5f + 0.25f * index, 0.25f));
        SetInternal(boxed, "m_Normal", Vector2.up);
        SetInternal(boxed, "m_RelativeVelocity", relativeVelocity);
        SetInternal(boxed, "m_Separation", -0.01f);
        SetInternal(boxed, "m_NormalImpulse", 1f);
        SetInternal(boxed, "m_TangentImpulse", 0.5f);
        SetInternal(boxed, "m_Collider", incoming.GetInstanceID());
        SetInternal(boxed, "m_OtherCollider", self.GetInstanceID());
        SetInternal(boxed, "m_Rigidbody", InstanceIdOfBody(incoming));
        SetInternal(boxed, "m_OtherRigidbody", InstanceIdOfBody(self));
        SetInternal(boxed, "m_Enabled", 1);
        return (ContactPoint2D)boxed;
    }

    private static int InstanceIdOfBody(Collider2D collider)
    {
        Rigidbody2D body = collider.GetComponent<Rigidbody2D>();
        return body == null ? 0 : body.GetInstanceID();
    }

    private static void SetInternal(object target, string fieldName, object value)
    {
        FieldInfo field = target.GetType().GetField(
            fieldName, BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.Public);
        Assert.IsNotNull(field,
            target.GetType().Name + "." + fieldName + " is gone; this Unity version changed the "
            + "collision layout and this fixture must be rewritten rather than skipped");
        field.SetValue(target, value);
    }
}
