using System;
using System.Collections.Generic;
using System.Reflection;
using System.Text;
using NUnit.Framework;
using SimpleJSON;
using UnityEngine;

public sealed class PhysicsCaptureV2ContactTests
{
    private string previousStride;
    private GameObject recorderHost;
    private GameObject lower;
    private GameObject upper;

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
        lower = Causal("lower:0001", Vector2.zero);
        upper = Causal("upper:0001", Vector2.up);
    }

    [TearDown]
    public void TearDown()
    {
        UnityEngine.Object.DestroyImmediate(lower);
        UnityEngine.Object.DestroyImmediate(upper);
        UnityEngine.Object.DestroyImmediate(recorderHost);
        Environment.SetEnvironmentVariable(
            PhysicsCaptureV2EngineProtocol.StrideEnvironmentVariable, previousStride,
            EnvironmentVariableTarget.Process);
    }

    [Test]
    public void Request71SerializesAnExplicitlyCompleteEmptyContactStepAndUnobservedMinimum()
    {
        PhysicsCaptureV2FixedStepRecorder recorder =
            recorderHost.AddComponent<PhysicsCaptureV2FixedStepRecorder>();
        BoxCollider2D lowerCollider = lower.GetComponent<BoxCollider2D>();
        lowerCollider.isTrigger = true;
        PhysicsCaptureV2ContactInput trigger = new PhysicsCaptureV2ContactInput(
            lowerCollider, upper.GetComponent<BoxCollider2D>(), Vector2.zero, Vector2.up, 0f);

        recorder.BeginPreIntervention(10, new[] { lower, upper },
            new[] { trigger }, true);
        recorder.FinalizeTerminal(10);
        string json = CaptureText(recorder.CreateFinalizedSnapshot());

        StringAssert.Contains("\"causal_entities\":[\"runtime:lower:0001\",\"runtime:upper:0001\"]", json);
        StringAssert.Contains("\"contacts\":[]", json);
        StringAssert.Contains("\"supports\":[]", json);
        StringAssert.Contains("\"minimum_contact_separation\":{\"observed\":false,"
            + "\"separation\":null,\"contact_id\":null,\"fixed_step\":null}", json);
    }

    [Test]
    public void Request71OrdersContactsRecomputesMinimumAndTracksPersistentSupportChange()
    {
        BoxCollider2D lowerCollider = lower.GetComponent<BoxCollider2D>();
        BoxCollider2D upperCollider = upper.GetComponent<BoxCollider2D>();
        PhysicsCaptureV2ContactInput left = new PhysicsCaptureV2ContactInput(
            lowerCollider, upperCollider, new Vector2(-0.5f, 0.5f), Vector2.up, -0.05f);
        PhysicsCaptureV2ContactInput right = new PhysicsCaptureV2ContactInput(
            lowerCollider, upperCollider, new Vector2(0.5f, 0.5f), Vector2.up, -0.02f);
        PhysicsCaptureV2FixedStepRecorder recorder =
            recorderHost.AddComponent<PhysicsCaptureV2FixedStepRecorder>();

        recorder.BeginPreIntervention(10, new[] { lower, upper }, new[] { right, left }, true);
        recorder.RecordFixedStep(11, new[] { lower, upper }, new[] { right }, true);
        recorder.RecordFixedStep(12, new[] { lower, upper }, new PhysicsCaptureV2ContactInput[0], true);
        recorder.FinalizeTerminal(12);
        JSONNode capture = JSONNode.Parse(CaptureText(recorder.CreateFinalizedSnapshot()));

        AssertHasFields(capture, "schema_version", "capture_id", "shot_id",
            "configured_fixed_step_capture_stride", "pre_intervention_fixed_step",
            "coordinate_convention", "causal_entities", "colliders", "fixed_step_samples",
            "minimum_contact_separation", "frame_records", "events", "terminal_evidence");
        AssertHasFields(capture["colliders"][0],
            "collider_id", "entity_id", "geometry_source");
        AssertHasFields(capture["fixed_step_samples"][0], "fixed_step",
            "complete_raw_non_trigger_contacts", "world", "entities", "colliders",
            "contacts", "supports");
        AssertHasFields(capture["fixed_step_samples"][0]["entities"][0], "entity_id",
            "scenario_object_id", "lifecycle", "body_present", "body", "contact_ids",
            "supported_by_entity_ids", "supports_entity_ids");
        Assert.AreEqual("contact:10:0000",
            capture["fixed_step_samples"][0]["contacts"][0]["contact_id"].Value);
        Assert.AreEqual(-0.5f,
            capture["fixed_step_samples"][0]["contacts"][0]["point"][0].AsFloat, 1e-5f);
        Assert.AreEqual("runtime:lower:0001",
            capture["fixed_step_samples"][0]["supports"][0]["supporter_entity_id"].Value);
        Assert.AreEqual(2,
            capture["fixed_step_samples"][0]["supports"][0]["contact_ids"].Count);
        Assert.AreEqual(2,
            capture["fixed_step_samples"][0]["entities"][0]["contact_ids"].Count);
        Assert.AreEqual("runtime:upper:0001",
            capture["fixed_step_samples"][0]["entities"][0]["supports_entity_ids"][0].Value);
        Assert.AreEqual("runtime:lower:0001",
            capture["fixed_step_samples"][0]["entities"][1]["supported_by_entity_ids"][0].Value);
        Assert.AreEqual(1, capture["fixed_step_samples"][1]["supports"].Count);
        Assert.AreEqual(0, capture["fixed_step_samples"][2]["supports"].Count);
        Assert.AreEqual(0,
            capture["fixed_step_samples"][2]["entities"][0]["supports_entity_ids"].Count);
        Assert.IsTrue(capture["minimum_contact_separation"]["observed"].AsBool);
        Assert.AreEqual(-0.05f, capture["minimum_contact_separation"]["separation"].AsFloat, 1e-5f);
        Assert.AreEqual("contact:10:0000", capture["minimum_contact_separation"]["contact_id"].Value);
        Assert.AreEqual(10, capture["minimum_contact_separation"]["fixed_step"].AsInt);
    }

    [Test]
    public void Request71ReturnsTypedFailuresForOverflowGapIncompleteAndUnresolvedContacts()
    {
        BoxCollider2D lowerCollider = lower.GetComponent<BoxCollider2D>();
        BoxCollider2D upperCollider = upper.GetComponent<BoxCollider2D>();
        PhysicsCaptureV2ContactInput first = new PhysicsCaptureV2ContactInput(
            lowerCollider, upperCollider, Vector2.zero, Vector2.up, -0.01f);
        PhysicsCaptureV2ContactInput second = new PhysicsCaptureV2ContactInput(
            lowerCollider, upperCollider, Vector2.right, Vector2.up, -0.02f);

        PhysicsCaptureV2FixedStepRecorder incomplete = NewRecorder();
        incomplete.BeginPreIntervention(10, new[] { lower, upper },
            new PhysicsCaptureV2ContactInput[0], false);
        Assert.AreEqual(11, ActiveFailureCode());

        PhysicsCaptureV2FixedStepRecorder overflow = NewRecorder();
        overflow.BeginPreIntervention(10, new[] { lower, upper }, new[] { first, second }, true,
            new PhysicsCaptureV2CaptureLimits(10, 10, 1));
        Assert.AreEqual(12, ActiveFailureCode());

        GameObject outsider = Causal("outsider:0001", Vector2.right);
        try
        {
            PhysicsCaptureV2ContactInput unresolved = new PhysicsCaptureV2ContactInput(
                lowerCollider, outsider.GetComponent<BoxCollider2D>(), Vector2.zero, Vector2.up, 0f);
            PhysicsCaptureV2FixedStepRecorder unresolvedRecorder = NewRecorder();
            unresolvedRecorder.BeginPreIntervention(10, new[] { lower, upper },
                new[] { unresolved }, true);
            Assert.AreEqual(13, ActiveFailureCode());
            StringAssert.Contains("collider_a=lower:0001",
                unresolvedRecorder.Failure.Message);
            StringAssert.Contains("collider_b=outsider:0001",
                unresolvedRecorder.Failure.Message);
            StringAssert.Contains("resolved_a=true",
                unresolvedRecorder.Failure.Message);
            StringAssert.Contains("resolved_b=false",
                unresolvedRecorder.Failure.Message);
        }
        finally
        {
            UnityEngine.Object.DestroyImmediate(outsider);
        }

        PhysicsCaptureV2FixedStepRecorder gap = NewRecorder();
        gap.BeginPreIntervention(10, new[] { lower, upper });
        gap.RecordFixedStep(12, new[] { lower, upper });
        Assert.AreEqual(14, ActiveFailureCode());
    }

    [Test]
    public void Request71RetainsTheWorldGravityFromEachRecordedFixedStep()
    {
        Vector2 previousGravity = Physics2D.gravity;
        try
        {
            PhysicsCaptureV2FixedStepRecorder recorder = NewRecorder();
            Physics2D.gravity = new Vector2(0f, -9f);
            recorder.BeginPreIntervention(10, new[] { lower, upper });
            Physics2D.gravity = new Vector2(1f, -3f);
            recorder.RecordFixedStep(11, new[] { lower, upper });
            recorder.FinalizeTerminal(11);

            string json = CaptureText(recorder.CreateFinalizedSnapshot());

            StringAssert.Contains("\"fixed_step\":10,\"complete_raw_non_trigger_contacts\":true,"
                + "\"world\":{\"world_id\":\"unity-physics2d\",\"gravity_vector\":[0,-9]}", json);
            StringAssert.Contains("\"fixed_step\":11,\"complete_raw_non_trigger_contacts\":true,"
                + "\"world\":{\"world_id\":\"unity-physics2d\",\"gravity_vector\":[1,-3]}", json);
        }
        finally
        {
            Physics2D.gravity = previousGravity;
        }
    }

    [Test]
    public void UnityEnumerationIncludesUnboundSceneCollidersForFailClosedResolution()
    {
        GameObject outsider = new GameObject("unbound-world-collider");
        GameObject causal = null;
        try
        {
            outsider.AddComponent<Rigidbody2D>().bodyType = RigidbodyType2D.Static;
            BoxCollider2D outsiderCollider = outsider.AddComponent<BoxCollider2D>();
            causal = Causal("causal:0001", new Vector2(0.25f, 0f));
            MethodInfo method = typeof(PhysicsCaptureV2FixedStepRecorder).GetMethod(
                "ActiveSceneColliders", BindingFlags.Static | BindingFlags.NonPublic);
            Assert.IsNotNull(method);
            Collider2D[] candidates = (Collider2D[])method.Invoke(null, null);

            CollectionAssert.Contains(candidates, outsiderCollider,
                "unbound colliders must reach contact resolution so a live contact fails closed");
            CollectionAssert.Contains(candidates, causal.GetComponent<BoxCollider2D>());
        }
        finally
        {
            if (causal != null) UnityEngine.Object.DestroyImmediate(causal);
            UnityEngine.Object.DestroyImmediate(outsider);
        }
    }

    [Test]
    public void UnityFixedStepEnumerationUsesTheAuthoritativeContactPointColliderPair()
    {
        bool previousAutoSimulation = Physics2D.autoSimulation;
        BoxCollider2D lowerCollider = lower.GetComponent<BoxCollider2D>();
        Rigidbody2D upperBody = upper.GetComponent<Rigidbody2D>();
        try
        {
            lowerCollider.size = new Vector2(4f, 1f);
            upper.transform.position = new Vector2(0f, 0.9f);
            upperBody.bodyType = RigidbodyType2D.Dynamic;
            upperBody.gravityScale = 0f;
            upperBody.constraints = RigidbodyConstraints2D.FreezeAll;
            Physics2D.autoSimulation = false;
            Physics2D.SyncTransforms();
            Physics2D.Simulate(0.02f);
            Physics2D.Simulate(0.02f);
            Assert.Greater(upper.GetComponent<BoxCollider2D>().GetContacts(
                new ContactPoint2D[4]), 0,
                "the Unity fixture did not produce an authoritative resting contact");
            PhysicsCaptureV2FixedStepRecorder recorder = NewRecorder();

            recorder.BeginPreInterventionFromUnity(10, new[] { lower, upper });
            recorder.FinalizeTerminal(10);

            PhysicsCaptureV2FixedStepSample sample =
                recorder.CreateFinalizedSnapshot().FixedStepSamples[0];
            Assert.GreaterOrEqual(sample.Contacts.Count, 1);
            Assert.AreEqual(1, sample.Supports.Count);
            Assert.AreEqual("runtime:lower:0001", sample.Supports[0].SupporterEntityId);
            Assert.AreEqual("runtime:upper:0001", sample.Supports[0].SupportedEntityId);
        }
        finally
        {
            Physics2D.autoSimulation = previousAutoSimulation;
        }
    }

    private PhysicsCaptureV2FixedStepRecorder NewRecorder()
    {
        return recorderHost.AddComponent<PhysicsCaptureV2FixedStepRecorder>();
    }

    private static int ActiveFailureCode()
    {
        byte[] envelope = PhysicsCaptureV2EngineProtocol.BuildCaptureEnvelope();
        Assert.AreEqual(1, envelope[9]);
        return envelope[10] << 8 | envelope[11];
    }

    private static void AssertHasFields(JSONNode value, params string[] expected)
    {
        HashSet<string> actual = new HashSet<string>();
        foreach (KeyValuePair<string, JSONNode> field in value.AsObject) actual.Add(field.Key);
        CollectionAssert.AreEquivalent(expected, actual);
    }

    private static GameObject Causal(string scenarioObjectId, Vector2 position)
    {
        GameObject value = new GameObject(scenarioObjectId);
        value.transform.position = position;
        ScenarioObjectIdentity.Assign(value, scenarioObjectId);
        value.AddComponent<Rigidbody2D>().bodyType = RigidbodyType2D.Static;
        value.AddComponent<BoxCollider2D>();
        return value;
    }

    private static string CaptureText(PhysicsCaptureV2EngineSnapshot snapshot)
    {
        byte[] envelope = PhysicsCaptureV2EngineProtocol.BuildCaptureEnvelope(snapshot);
        int payloadLength = envelope[12] << 24 | envelope[13] << 16 | envelope[14] << 8 | envelope[15];
        return Encoding.UTF8.GetString(envelope, 16, payloadLength);
    }
}
