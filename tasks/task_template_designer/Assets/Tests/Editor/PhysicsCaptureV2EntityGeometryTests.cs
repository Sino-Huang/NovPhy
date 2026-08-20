using System;
using System.Collections.Generic;
using System.Text;
using NUnit.Framework;
using SimpleJSON;
using UnityEngine;

public sealed class PhysicsCaptureV2EntityGeometryTests
{
    private string previousStride;
    private GameObject recorderHost;

    [SetUp]
    public void SetUp()
    {
        previousStride = Environment.GetEnvironmentVariable(
            PhysicsCaptureV2EngineProtocol.StrideEnvironmentVariable,
            EnvironmentVariableTarget.Process);
        Environment.SetEnvironmentVariable(
            PhysicsCaptureV2EngineProtocol.StrideEnvironmentVariable, "1",
            EnvironmentVariableTarget.Process);
        recorderHost = new GameObject("physics-capture-v2-recorder");
    }

    [TearDown]
    public void TearDown()
    {
        UnityEngine.Object.DestroyImmediate(recorderHost);
        Environment.SetEnvironmentVariable(
            PhysicsCaptureV2EngineProtocol.StrideEnvironmentVariable, previousStride,
            EnvironmentVariableTarget.Process);
    }

    [Test]
    public void Request71RetainsAuthoredAndRuntimeIdentityBodyGravityMotionAndBoxGeometry()
    {
        GameObject causal = new GameObject("authored-block");
        try
        {
            ScenarioObjectIdentity.Assign(causal, "block:0001");
            causal.transform.position = new Vector3(3f, 4f, 0f);
            causal.transform.rotation = Quaternion.Euler(0f, 0f, 30f);
            Rigidbody2D body = causal.AddComponent<Rigidbody2D>();
            body.bodyType = RigidbodyType2D.Dynamic;
            body.simulated = true;
            body.gravityScale = 0.75f;
            body.velocity = new Vector2(1.5f, -2f);
            body.angularVelocity = 12f;
            BoxCollider2D box = causal.AddComponent<BoxCollider2D>();
            box.offset = new Vector2(0.5f, -0.25f);
            box.size = new Vector2(2f, 1f);
            Vector2 expectedVelocity = body.velocity;
            float expectedAngularVelocity = body.angularVelocity;
            PhysicsCaptureV2FixedStepRecorder recorder =
                recorderHost.AddComponent<PhysicsCaptureV2FixedStepRecorder>();

            recorder.BeginPreIntervention(5, new[] { causal });
            recorder.FinalizeTerminal(5);
            JSONNode capture = CaptureJson(recorder.CreateFinalizedSnapshot());
            JSONNode entity = capture["fixed_step_samples"][0]["entities"][0];
            JSONNode collider = capture["fixed_step_samples"][0]["colliders"][0];

            Assert.AreEqual("block:0001", entity["scenario_object_id"].Value);
            Assert.AreEqual("runtime:block:0001", entity["entity_id"].Value);
            Assert.AreEqual("active", entity["lifecycle"].Value);
            Assert.AreEqual(3f, entity["body"]["position"][0].AsFloat, 1e-5f);
            Assert.AreEqual(4f, entity["body"]["position"][1].AsFloat, 1e-5f);
            Assert.AreEqual(30f, entity["body"]["rotation_degrees"].AsFloat, 1e-5f);
            Assert.AreEqual("dynamic", entity["body"]["body_type"].Value);
            Assert.IsTrue(entity["body"]["simulated"].AsBool);
            Assert.AreEqual(0.75f, entity["body"]["gravity_scale"].AsFloat, 1e-5f);
            Assert.IsTrue(entity["body"]["gravity_applicable"].AsBool);
            Assert.AreEqual(expectedVelocity.x,
                entity["body"]["velocity"][0].AsFloat, 1e-5f);
            Assert.AreEqual(expectedVelocity.y,
                entity["body"]["velocity"][1].AsFloat, 1e-5f);
            Assert.AreEqual(expectedAngularVelocity,
                entity["body"]["angular_velocity_degrees_per_second"].AsFloat, 1e-5f);
            Assert.AreEqual("runtime:block:0001:collider:0000", collider["collider_id"].Value);
            Assert.AreEqual("unity_collider_2d", collider["geometry_source"].Value);
            Assert.AreEqual("box", collider["shape"]["kind"].Value);
            Assert.AreEqual(2f, collider["shape"]["size"][0].AsFloat, 1e-5f);
            Assert.AreEqual(1f, collider["shape"]["size"][1].AsFloat, 1e-5f);
            Assert.AreEqual(30f, collider["shape"]["angle_degrees"].AsFloat, 1e-5f);
        }
        finally
        {
            UnityEngine.Object.DestroyImmediate(causal);
        }
    }

    [Test]
    public void Request71ExportsCirclePolygonEdgeAndCapsuleWorldGeometry()
    {
        GameObject circleObject = Authored("circle", "circle:0001", new Vector2(1f, 2f));
        GameObject polygonObject = Authored("polygon", "polygon:0001", new Vector2(3f, 4f));
        GameObject edgeObject = Authored("edge", "edge:0001", new Vector2(5f, 6f));
        GameObject capsuleObject = Authored("capsule", "capsule:0001", new Vector2(7f, 8f));
        try
        {
            CircleCollider2D circle = circleObject.AddComponent<CircleCollider2D>();
            circle.radius = 0.75f;
            PolygonCollider2D polygon = polygonObject.AddComponent<PolygonCollider2D>();
            polygon.points = new[] { new Vector2(-1f, 0f), new Vector2(1f, 0f), new Vector2(0f, 2f) };
            EdgeCollider2D edge = edgeObject.AddComponent<EdgeCollider2D>();
            edge.points = new[] { new Vector2(-1f, 0f), new Vector2(1f, 0.5f) };
            CapsuleCollider2D capsule = capsuleObject.AddComponent<CapsuleCollider2D>();
            capsule.size = new Vector2(2f, 4f);
            capsule.direction = CapsuleDirection2D.Vertical;
            PhysicsCaptureV2FixedStepRecorder recorder =
                recorderHost.AddComponent<PhysicsCaptureV2FixedStepRecorder>();

            recorder.BeginPreIntervention(5,
                new[] { circleObject, polygonObject, edgeObject, capsuleObject });
            recorder.FinalizeTerminal(5);
            JSONNode colliders = CaptureJson(recorder.CreateFinalizedSnapshot())
                ["fixed_step_samples"][0]["colliders"];

            Assert.AreEqual("capsule", colliders[0]["shape"]["kind"].Value);
            Assert.AreEqual("vertical", colliders[0]["shape"]["direction"].Value);
            Assert.AreEqual(2f, colliders[0]["shape"]["size"][0].AsFloat, 1e-5f);
            Assert.AreEqual(4f, colliders[0]["shape"]["size"][1].AsFloat, 1e-5f);
            Assert.AreEqual("circle", colliders[1]["shape"]["kind"].Value);
            Assert.AreEqual(0.75f, colliders[1]["shape"]["radius"].AsFloat, 1e-5f);
            Assert.AreEqual("edge", colliders[2]["shape"]["kind"].Value);
            Assert.AreEqual(2, colliders[2]["shape"]["points"].Count);
            Assert.AreEqual(1, colliders[3]["shape"]["paths"].Count);
            Assert.AreEqual(3, colliders[3]["shape"]["paths"][0].Count);
        }
        finally
        {
            UnityEngine.Object.DestroyImmediate(circleObject);
            UnityEngine.Object.DestroyImmediate(polygonObject);
            UnityEngine.Object.DestroyImmediate(edgeObject);
            UnityEngine.Object.DestroyImmediate(capsuleObject);
        }
    }

    [Test]
    public void Request71MarksBodyAbsenceExplicitly()
    {
        GameObject causal = new GameObject("bodyless-causal-object");
        try
        {
            ScenarioObjectIdentity.Assign(causal, "platform:0001");
            PhysicsCaptureV2FixedStepRecorder recorder =
                recorderHost.AddComponent<PhysicsCaptureV2FixedStepRecorder>();
            recorder.BeginPreIntervention(5, new[] { causal });
            recorder.FinalizeTerminal(5);

            string json = CaptureText(recorder.CreateFinalizedSnapshot());

            StringAssert.Contains("\"scenario_object_id\":\"platform:0001\"", json);
            StringAssert.Contains("\"body\":null", json);
        }
        finally
        {
            UnityEngine.Object.DestroyImmediate(causal);
        }
    }

    [Test]
    public void Request71FailsTheWholeCaptureForIdentityGeometryFiniteAndBoundViolations()
    {
        GameObject missingIdentity = new GameObject("missing-identity");
        GameObject unsupported = Authored("unsupported", "unsupported:0001", Vector2.zero);
        GameObject incomplete = Authored("incomplete", "incomplete:0001", Vector2.zero);
        GameObject first = Authored("first", "first:0001", Vector2.zero);
        GameObject second = Authored("second", "second:0001", Vector2.zero);
        GameObject tooManyColliders = Authored("colliders", "colliders:0001", Vector2.zero);
        try
        {
            unsupported.AddComponent<CompositeCollider2D>();
            incomplete.transform.localScale = Vector3.zero;
            incomplete.AddComponent<BoxCollider2D>();
            tooManyColliders.AddComponent<BoxCollider2D>();
            tooManyColliders.AddComponent<BoxCollider2D>();

            Assert.AreEqual(5, FailureCode(new[] { missingIdentity },
                PhysicsCaptureV2CaptureLimits.Default));
            Assert.AreEqual(6, FailureCode(new[] { unsupported },
                PhysicsCaptureV2CaptureLimits.Default));
            Assert.AreEqual(7, FailureCode(new[] { incomplete },
                PhysicsCaptureV2CaptureLimits.Default));
            Assert.AreEqual(8, NonFiniteSnapshotFailureCode());
            Assert.AreEqual(9, FailureCode(new[] { first, second },
                new PhysicsCaptureV2CaptureLimits(1, 10)));
            Assert.AreEqual(10, FailureCode(new[] { tooManyColliders },
                new PhysicsCaptureV2CaptureLimits(10, 1)));
        }
        finally
        {
            UnityEngine.Object.DestroyImmediate(missingIdentity);
            UnityEngine.Object.DestroyImmediate(unsupported);
            UnityEngine.Object.DestroyImmediate(incomplete);
            UnityEngine.Object.DestroyImmediate(first);
            UnityEngine.Object.DestroyImmediate(second);
            UnityEngine.Object.DestroyImmediate(tooManyColliders);
        }
    }

    private static GameObject Authored(string name, string scenarioObjectId, Vector2 position)
    {
        GameObject value = new GameObject(name);
        value.transform.position = position;
        ScenarioObjectIdentity.Assign(value, scenarioObjectId);
        value.AddComponent<Rigidbody2D>().bodyType = RigidbodyType2D.Static;
        return value;
    }

    private static JSONNode CaptureJson(PhysicsCaptureV2EngineSnapshot snapshot)
    {
        return JSONNode.Parse(CaptureText(snapshot).Replace(":null", ":\"null\""));
    }

    private static string CaptureText(PhysicsCaptureV2EngineSnapshot snapshot)
    {
        byte[] envelope = PhysicsCaptureV2EngineProtocol.BuildCaptureEnvelope(snapshot);
        int payloadLength = ReadUInt32(envelope, 12);
        return Encoding.UTF8.GetString(envelope, 16, payloadLength);
    }

    private int FailureCode(GameObject[] causalObjects, PhysicsCaptureV2CaptureLimits limits)
    {
        PhysicsCaptureV2FixedStepRecorder recorder =
            recorderHost.AddComponent<PhysicsCaptureV2FixedStepRecorder>();
        recorder.BeginPreIntervention(5, causalObjects, limits);
        byte[] envelope = PhysicsCaptureV2EngineProtocol.BuildCaptureEnvelope();
        Assert.AreEqual(1, envelope[9], "invalid capture was not rejected as one v2 envelope");
        return envelope[10] << 8 | envelope[11];
    }

    private static int NonFiniteSnapshotFailureCode()
    {
        PhysicsCaptureV2EntitySnapshot entity = new PhysicsCaptureV2EntitySnapshot(
            "runtime:bad:0001", "bad:0001", "active", new Vector2(float.NaN, 0f),
            0f, null);
        PhysicsCaptureV2FixedStepSample sample = new PhysicsCaptureV2FixedStepSample(5,
            new List<PhysicsCaptureV2EntitySnapshot> { entity },
            new List<PhysicsCaptureV2ColliderSnapshot>());
        PhysicsCaptureV2EngineSnapshot snapshot = new PhysicsCaptureV2EngineSnapshot(
            1, 5, 5, new List<PhysicsCaptureV2FixedStepSample> { sample },
            new List<PhysicsCaptureV2FrameRecord> { new PhysicsCaptureV2FrameRecord(5, false) });

        byte[] envelope = PhysicsCaptureV2EngineProtocol.BuildCaptureEnvelope(snapshot);
        Assert.AreEqual(1, envelope[9]);
        return envelope[10] << 8 | envelope[11];
    }

    private static int ReadUInt32(byte[] bytes, int offset)
    {
        return bytes[offset] << 24 | bytes[offset + 1] << 16
            | bytes[offset + 2] << 8 | bytes[offset + 3];
    }
}
