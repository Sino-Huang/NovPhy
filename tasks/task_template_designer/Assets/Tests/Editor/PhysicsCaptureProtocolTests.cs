using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Diagnostics;
using System.Globalization;
using System.Net;
using System.Net.Sockets;
using System.Reflection;
using System.Text.RegularExpressions;
using System.Threading;
using System.Threading.Tasks;
using NUnit.Framework;
using SimpleJSON;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;

public class PhysicsCaptureProtocolTests
{
    private Texture2D texture;
    private Sprite sprite;

    [SetUp]
    public void SetUp()
    {
        EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
        GameObject schema = new GameObject("LoadLevelSchema");
        schema.AddComponent<LoadLevelSchema>();
        GameObject cameraObject = new GameObject("Main Camera");
        cameraObject.tag = "MainCamera";
        cameraObject.AddComponent<Camera>();
        texture = new Texture2D(2, 2);
        sprite = Sprite.Create(texture, new Rect(0f, 0f, 2f, 2f), new Vector2(0.5f, 0.5f), 1f);
        GameObject slingshot = new GameObject("slingshot_back");
        slingshot.tag = "Slingshot";
        slingshot.AddComponent<SpriteRenderer>().sprite = sprite;
    }

    [TearDown]
    public void TearDown()
    {
        UnityEngine.Object.DestroyImmediate(sprite);
        UnityEngine.Object.DestroyImmediate(texture);
        EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
    }

    [Test]
    public void Request70EnvelopeContainsActualPngAndSameRenderFrameBatch()
    {
        Texture2D image = new Texture2D(1, 1, TextureFormat.RGB24, false);
        image.SetPixel(0, 0, Color.red);
        image.Apply();
        byte[] png = image.EncodeToPNG();
        UnityEngine.Object.DestroyImmediate(image);

        GameObject runtimeObject = new GameObject("snapshot-runtime");
        PhysicalSnapshotRuntime runtime = runtimeObject.AddComponent<PhysicalSnapshotRuntime>();
        PhysicalSceneSnapshot snapshot = runtime.CaptureCurrent(new SymbolicGameState(false), 73, 1.25f);
        byte[] envelope = PhysicsCaptureV1Protocol.BuildCaptureEnvelope(png, snapshot, null);

        Assert.AreEqual(70, PhysicsCaptureV1Protocol.RequestCode);
        Assert.AreEqual((byte)'S', envelope[4]);
        Assert.AreEqual(2, envelope[9], "current request-70 producer must set the fourth-component flag");
        Assert.AreEqual(0x89, envelope[32]);
        Assert.AreEqual(0x50, envelope[33]);
        string envelopeText = System.Text.Encoding.UTF8.GetString(envelope);
        StringAssert.Contains("\"render_frame\":73", envelopeText);
        byte[] batchEnvelope = PhysicsCaptureV1Protocol.BuildSuccessEnvelope(
            png, "{\"render_frame\":73}", "[{\"render_frame\":73}]");
        StringAssert.Contains("[{\"render_frame\":73}]", System.Text.Encoding.UTF8.GetString(batchEnvelope));
    }

    [Test]
    public void CaptureAtEndOfRenderFrameUsesWaitForEndOfFrameBeforeBatch()
    {
        PhysicalSnapshotRuntime runtime = new GameObject("snapshot-runtime").AddComponent<PhysicalSnapshotRuntime>();
        PhysicalSceneSnapshot captured = null;
        IEnumerator coroutine = runtime.CaptureAtEndOfRenderFrame(
            new SymbolicGameState(false), delegate(PhysicalSceneSnapshot value) { captured = value; });

        Assert.IsTrue(coroutine.MoveNext());
        Assert.IsInstanceOf<WaitForEndOfFrame>(coroutine.Current);
        Assert.IsNull(captured);
        Assert.IsFalse(coroutine.MoveNext());
        Assert.IsNotNull(captured);
    }

    [Test]
    public void Request70RejectsRecorderThatWasNotFinalized()
    {
        PhysicalShotRecorder recorder = new PhysicalShotRecorder(new PhysicalCaptureLimits(32, 16384, 10f));
        byte[] envelope = PhysicsCaptureV1Protocol.BuildCaptureEnvelope(
            new byte[] { 0x01 }, new PhysicalSceneSnapshot(1, 0f, 0, 0f, new PhysicalNodeSnapshot[0]), recorder);

        Assert.AreEqual(1, envelope[9]);
    }

    [Test]
    public void Request70SerializesFrozenRecorderBatch()
    {
        PhysicalShotRecorder recorder = new PhysicalShotRecorder(new PhysicalCaptureLimits(32, 16384, 10f));
        recorder.RecordLaunch("bird:1", 1);
        recorder.FinalizeShot(true);
        PhysicalSceneSnapshot snapshot = new PhysicalSceneSnapshot(1, 0f, 1, 0.02f, new PhysicalNodeSnapshot[0]);

        byte[] before = PhysicsCaptureV1Protocol.BuildCaptureEnvelope(
            new byte[] { 0x01 }, snapshot, recorder);
        recorder.RecordDestroyed("bird:2", 2);
        byte[] after = PhysicsCaptureV1Protocol.BuildCaptureEnvelope(
            new byte[] { 0x01 }, snapshot, recorder);

        CollectionAssert.AreEqual(before, after);
    }

    [Test]
    public void Request70RecordsMatchFrozenPhysicsCaptureV1ProducerContract()
    {
        PhysicalShotRecorder recorder = new PhysicalShotRecorder(new PhysicalCaptureLimits(32, 16384, 10f));
        PhysicalContactInput contact = new PhysicalContactInput(
            "10:0", 10, Vector2.zero, Vector2.up, 0.01f, Vector2.zero, 1f,
            "20:0", 20, Vector2.up, false);
        recorder.RecordContacts(1, 0.02f, new[] { contact });
        recorder.RecordContacts(2, 0.04f, new[] { contact });
        recorder.RecordLaunch("10:0", 2, new Vector2(2f, 1f));
        recorder.FinalizeShot(true);
        PhysicalSceneSnapshot snapshot = new PhysicalSceneSnapshot(73, 1.25f, 2, 0.04f, new PhysicalNodeSnapshot[0]);

        byte[] envelope = PhysicsCaptureV1Protocol.BuildCaptureEnvelope(new byte[] { 0x01 }, snapshot, recorder);
        string schemaJson = File.ReadAllText(Path.GetFullPath(Path.Combine(
            Application.dataPath, "../../../docs/data_contracts/physics_capture_v1.schema.json")));
        JSONNode state = JSONNode.Parse(EnvelopeJson(envelope, false));
        JSONNode events = JSONNode.Parse(EnvelopeJson(envelope, true));
        string evidenceJson = EnvelopeEvidenceJson(envelope);

        StringAssert.Contains("\"physics_violation_engine_evidence_v1\"", schemaJson);
        AssertHasFields(state, "schema_version", "capture_id", "sequence", "render_frame",
            "render_time", "fixed_step", "fixed_time", "coordinates", "nodes", "raw_contacts",
            "support_edges");
        AssertHasFields(state["rgb_frame"], "render_frame", "source");
        AssertHasFields(state["raw_contacts"][0], "contact_id", "entity_a_id", "entity_b_id",
            "collider_a_id", "collider_b_id", "point", "normal_a_to_b", "separation",
            "relative_velocity_a_to_b", "normal_impulse", "tangent_impulse", "is_trigger");
        AssertHasFields(state["support_edges"][0], "support_id", "rule_version", "supporter_id",
            "supported_id", "evidence_contact_ids", "evidence_fixed_steps");
        Assert.Greater(events.Count, 0);
        AssertHasFields(events[0], "schema_version", "capture_id", "sequence", "render_frame",
            "render_time", "fixed_step", "fixed_time", "coordinates", "event_id", "event_type",
            "participants", "payload");
        foreach (string field in new[] { "schema_version", "capture_id", "shot_id", "sequence",
            "fixed_step_coverage", "minimum_contact_separation", "terminal_trace" })
            StringAssert.Contains("\"" + field + "\":", evidenceJson);
        StringAssert.Contains("\"schema_version\":\"physics_violation_engine_evidence_v1\"", evidenceJson);
        Assert.AreEqual("physics_capture_v1", state["schema_version"].Value);
        Assert.IsNotEmpty(state["capture_id"].Value);
        Assert.GreaterOrEqual(state["sequence"].AsInt, 1);
        Assert.AreEqual("synchronized_endpoint", state["rgb_frame"]["source"].Value);
        Assert.AreEqual(state["render_frame"].AsInt, state["rgb_frame"]["render_frame"].AsInt);
        Assert.AreEqual(0, events[0]["sequence"].AsInt);
        Assert.AreEqual("event:00000000", events[0]["event_id"].Value);
        Assert.AreEqual("bird_launched", events[0]["event_type"].Value);
        Assert.AreEqual(2f, events[0]["payload"]["launch_velocity"]["x"].AsFloat);
    }

    [Test]
    public void Request70CollisionPayloadCarriesTheContactEvidenceTheContractRequires()
    {
        // The 2026-08-06 staged player serialized every collision payload as the
        // empty object {}, which matches no branch of the frozen event_payload
        // oneOf, so the Python consumer rejected every shot containing a
        // collision. The recorder-level fixtures did not catch it because the
        // defect lived at the recorder -> wire seam: the emitter handed the
        // serializer an empty ContactIds list and the serializer, correctly,
        // wrote nothing. This asserts the serialized bytes, not the recorder API.
        //
        // Scope, stated honestly: this covers the recorder -> wire seam only. It
        // feeds PhysicalContactInput[] in directly, so it does NOT cover the
        // Collision2D.contacts -> PhysicalContactInput[] conversion in
        // PhysicalSnapshotRuntime, which is where the original defect actually
        // lived, nor the isTrigger filter there that can empty the array.
        PhysicalShotRecorder recorder = new PhysicalShotRecorder(new PhysicalCaptureLimits(32, 16384, 10f));
        PhysicalContactInput near = new PhysicalContactInput(
            "10:0", 10, new Vector2(0.5f, 0.25f), Vector2.up, 0.01f, new Vector2(3.5f, 0f), 1f,
            "20:0", 20, Vector2.zero, Vector2.up, false);
        PhysicalContactInput far = new PhysicalContactInput(
            "10:0", 10, new Vector2(0.9f, 0.25f), Vector2.up, 0.02f, new Vector2(3.5f, 0f), 1f,
            "20:0", 20, Vector2.zero, Vector2.up, false);
        recorder.RecordCollision(2, 0.04f, "20:0", "10:0", new[] { far, near }, 3.5f);
        recorder.FinalizeShot(true);

        byte[] envelope = PhysicsCaptureV1Protocol.BuildCaptureEnvelope(
            new byte[] { 0x01 },
            new PhysicalSceneSnapshot(73, 1.25f, 2, 0.04f, new PhysicalNodeSnapshot[0]), recorder);
        string eventsJson = EnvelopeJson(envelope, true);
        JSONNode events = JSONNode.Parse(eventsJson);

        JSONNode collision = null;
        for (int i = 0; i < events.Count; i++)
        {
            if (events[i]["event_type"].Value == "collision") collision = events[i];
        }
        Assert.IsNotNull(collision, "request-70 emitted no collision event");

        // The literal below governs the assertion; checking it against the raw
        // schema keeps this a drift detector without asking the legacy SimpleJSON
        // test dependency to parse the schema's legal null values.
        string schemaJson = File.ReadAllText(Path.GetFullPath(Path.Combine(
            Application.dataPath, "../../../docs/data_contracts/physics_capture_v1.schema.json")));
        StringAssert.Contains("\"required\": [\"contact_ids\", \"relative_speed\"]", schemaJson,
            "the frozen collision payload branch is not the one this fixture pins");
        string[] required = { "contact_ids", "relative_speed" };

        List<string> keys = new List<string>();
        foreach (KeyValuePair<string, JSONNode> field in collision["payload"].AsObject) keys.Add(field.Key);
        CollectionAssert.AreEquivalent(required, keys,
            "serialized collision payload does not match the frozen contract branch: " + collision["payload"].ToString());

        JSONArray contactIds = collision["payload"]["contact_ids"].AsArray;
        Assert.GreaterOrEqual(contactIds.Count, 1, "collision payload carries no contact evidence");
        for (int i = 0; i < contactIds.Count; i++)
        {
            Assert.IsNotEmpty(contactIds[i].Value, "collision payload carries an empty contact id");
        }
        for (int i = 1; i < contactIds.Count; i++)
        {
            Assert.Less(string.CompareOrdinal(contactIds[i - 1].Value, contactIds[i].Value), 0,
                "collision contact ids are not sorted unique by ordinal");
        }

        // Matched against the bytes the envelope actually carries, so a
        // string-typed speed cannot pass: AsFloat would coerce "3.5" and the
        // schema requires a JSON number.
        //
        // It must be the raw string and not `collision["payload"].ToString()`.
        // This SimpleJSON build stores every scalar as text and re-quotes it on
        // ToString, so a round-tripped payload reads "relative_speed":"3.5"
        // whatever the wire said — the parsed node cannot tell a JSON number
        // from a JSON string, and asserting on it measures the parser.
        //
        // The pattern is anchored on contact_ids, which only the collision
        // branch carries, so it stays scoped to a collision payload rather than
        // to any event that happens to serialize a speed; the count assertion
        // keeps it pinned to this fixture's single collision.
        //
        // Success alone already rejects every non-finite form: AppendFloat's "R"
        // format writes NaN and Infinity as bare words, which the leading -?[0-9]
        // cannot match. A negative or drifted magnitude fails the equality below.
        MatchCollection speeds = Regex.Matches(
            eventsJson, "\"contact_ids\":\\[[^\\]]*\\],\"relative_speed\":(-?[0-9][0-9.eE+-]*)");
        Assert.AreEqual(1, speeds.Count,
            "expected exactly one collision payload carrying a JSON-number relative_speed: " + eventsJson);
        float relativeSpeed = float.Parse(speeds[0].Groups[1].Value, CultureInfo.InvariantCulture);
        Assert.AreEqual(3.5f, relativeSpeed, 1e-6f, "relative_speed does not carry the recorded magnitude");
    }

    [Test]
    public void Request70RuntimeCaptureIdentityIsStableAndStateSequenceIncreases()
    {
        PhysicalSnapshotRuntime runtime = new GameObject("snapshot-runtime").AddComponent<PhysicalSnapshotRuntime>();

        string captureId = runtime.CaptureId;
        long first = runtime.NextCaptureSequence;
        long second = runtime.NextCaptureSequence;

        Assert.IsNotEmpty(captureId);
        Assert.AreEqual(captureId, runtime.CaptureId);
        Assert.AreEqual(first + 1, second);
        Assert.GreaterOrEqual(first, 1);
    }

    [Test]
    public void Request70SerializesProducerAuthoredGravityAndSupportEvidence()
    {
        GameObject bodyObject = new GameObject("evidence-body");
        Rigidbody2D body = bodyObject.AddComponent<Rigidbody2D>();
        body.gravityScale = 0.75f;
        Vector2 previousGravity = Physics2D.gravity;
        try
        {
            Physics2D.gravity = new Vector2(1.25f, -8f);
            PhysicalShotRecorder recorder = new PhysicalShotRecorder(new PhysicalCaptureLimits(32, 64 * 1024, 10f));
            recorder.RecordUnityContacts(7, 0.14f, new Collider2D[0], new[] { body }, new PhysicalEntityRegistry());
            recorder.FinalizeShot(true);
            byte[] envelope = PhysicsCaptureV1Protocol.BuildCaptureEnvelope(
                new byte[] { 0x01 },
                new PhysicalSceneSnapshot(73, 1.25f, 7, 0.14f, new PhysicalNodeSnapshot[0]),
                recorder, "capture-evidence", 4);

            string evidence = EnvelopeEvidenceJson(envelope);
            StringAssert.Contains("\"schema_version\":\"physics_violation_engine_evidence_v1\"", evidence);
            StringAssert.Contains("\"capture_id\":\"capture-evidence\"", evidence);
            StringAssert.Contains("\"sequence\":4", evidence);
            StringAssert.Contains("\"physics2d_gravity\":{\"x\":1.25,\"y\":-8}", evidence);
            StringAssert.Contains("\"gravity_scale\":0.75", evidence);
            StringAssert.Contains("\"support_v1\":{\"present\":false,\"edges\":[]}", evidence);
        }
        finally
        {
            Physics2D.gravity = previousGravity;
            UnityEngine.Object.DestroyImmediate(bodyObject);
        }
    }

    [Test]
    public void Request70EvidenceBytesMatchThePythonGolden()
    {
        Vector2 previousGravity = Physics2D.gravity;
        try
        {
            Physics2D.gravity = new Vector2(1.25f, -8f);
            PhysicalShotRecorder recorder = new PhysicalShotRecorder(
                new PhysicalCaptureLimits(32, 64 * 1024, 10f));
            typeof(PhysicalShotRecorder).GetField("evidenceShotId",
                BindingFlags.Instance | BindingFlags.NonPublic).SetValue(recorder, "engine-shot-golden");
            recorder.RecordUnityContacts(10, 0.2f, new Collider2D[0],
                new Rigidbody2D[0], new PhysicalEntityRegistry());
            recorder.FinalizeShot(true);
            byte[] envelope = PhysicsCaptureV1Protocol.BuildCaptureEnvelope(
                new byte[] { 0x01 },
                new PhysicalSceneSnapshot(101, 1.683333333f, 10, 0.2f, new PhysicalNodeSnapshot[0]),
                recorder, "capture-golden-001", 2);
            byte[] actual = System.Text.Encoding.UTF8.GetBytes(EnvelopeEvidenceJson(envelope) + "\n");
            byte[] expected = File.ReadAllBytes(Path.GetFullPath(Path.Combine(
                Application.dataPath,
                "../../../tests/fixtures/physics_capture_v1/physics_violation_engine_evidence_v1.csharp.jsonl")));

            CollectionAssert.AreEqual(expected, actual,
                "the Python golden must be the exact JSONL-framed evidence bytes emitted by C#");
        }
        finally
        {
            Physics2D.gravity = previousGravity;
        }
    }

    [Test]
    public void Request70ReturnsTypedFailureForEveryUnityEnvelopeBound()
    {
        string json = new string('x', PhysicsCaptureV1Protocol.MaxJsonBytes + 1);
        string failure = new string('x', PhysicsCaptureV1Protocol.MaxFailureMessageBytes + 1);
        byte[] png = new byte[PhysicsCaptureV1Protocol.MaxPngBytes + 1];

        Assert.AreEqual(1, PhysicsCaptureV1Protocol.BuildSuccessEnvelope(png, "{}", "[]")[9]);
        Assert.AreEqual(1, PhysicsCaptureV1Protocol.BuildSuccessEnvelope(new byte[0], json, "[]")[9]);
        Assert.AreEqual(1, PhysicsCaptureV1Protocol.BuildSuccessEnvelope(new byte[0], "{}", json)[9]);
        Assert.AreEqual(1, PhysicsCaptureV1Protocol.BuildFailureEnvelope(
            new PhysicalCaptureFailure(PhysicalCaptureFailureCode.CaptureTimeout, failure))[9]);
        Assert.AreEqual(1, PhysicsCaptureV1Protocol.BuildSuccessEnvelope(
            new byte[PhysicsCaptureV1Protocol.MaxPngBytes],
            new string('s', PhysicsCaptureV1Protocol.MaxJsonBytes),
            new string('e', PhysicsCaptureV1Protocol.MaxJsonBytes))[9]);
    }

    [Test]
    public void Request70PreflightsUtf8AndCombinedEnvelopeBeforeAllocation()
    {
        string state = new string('\u20ac', 5592405);
        string events = new string('\u20ac', 5592405);

        Assert.LessOrEqual(state.Length, PhysicsCaptureV1Protocol.MaxJsonBytes);
        Assert.LessOrEqual(System.Text.Encoding.UTF8.GetByteCount(state), PhysicsCaptureV1Protocol.MaxJsonBytes);
        Assert.Greater(
            16L + 12L + PhysicsCaptureV1Protocol.MaxPngBytes
                + System.Text.Encoding.UTF8.GetByteCount(state)
                + System.Text.Encoding.UTF8.GetByteCount(events),
            PhysicsCaptureV1Protocol.MaxEnvelopeBytes);
        Assert.AreEqual(1, PhysicsCaptureV1Protocol.BuildSuccessEnvelope(
            new byte[PhysicsCaptureV1Protocol.MaxPngBytes], state, events)[9]);
    }

    [Test]
    public void Request70ResponseTransmissionYieldsAndExpiresWhenClientStopsReading()
    {
        const int testDeadlineMilliseconds = 75;
        TcpListener listener = new TcpListener(IPAddress.Loopback, 0);
        TcpClient client = new TcpClient();
        TcpClient server = null;

        try
        {
            listener.Start();
            client.ReceiveBufferSize = 1024;
            client.Connect(IPAddress.Loopback, ((IPEndPoint)listener.LocalEndpoint).Port);
            server = listener.AcceptTcpClient();
            server.SendBufferSize = 1024;
            client.GetStream().Write(new byte[] { PhysicsCaptureV1Protocol.RequestCode }, 0, 1);
            Assert.AreEqual(PhysicsCaptureV1Protocol.RequestCode, server.GetStream().ReadByte());

            MethodInfo transmit = typeof(PhysicsCaptureDirectSocket).GetMethod(
                "TransmitResponse", BindingFlags.Static | BindingFlags.NonPublic, null,
                new[] { typeof(TcpClient), typeof(byte[]), typeof(int) }, null);
            FieldInfo timeout = typeof(PhysicsCaptureDirectSocket).GetField(
                "ResponseWriteTimeoutMilliseconds", BindingFlags.Static | BindingFlags.NonPublic);
            Assert.IsNotNull(transmit, "request-70 must use a bounded asynchronous response transmitter");
            Assert.IsNotNull(timeout, "request-70 response transmission must have a fixed deadline");
            int timeoutMilliseconds = (int)timeout.GetRawConstantValue();
            Assert.Greater(timeoutMilliseconds, 0);
            Assert.LessOrEqual(timeoutMilliseconds, 1000);

            byte[] boundedResponse = new byte[4 * 1024 * 1024];
            IEnumerator transmission = (IEnumerator)transmit.Invoke(
                null, new object[] { server, boundedResponse, testDeadlineMilliseconds });
            Stopwatch firstAdvance = Stopwatch.StartNew();
            Assert.IsTrue(transmission.MoveNext(),
                "a non-reading client must make the request-70 transmitter yield instead of blocking");
            firstAdvance.Stop();
            Assert.Less(firstAdvance.ElapsedMilliseconds, 250,
                "request-70 response transmission blocked the Unity-facing coroutine");

            Stopwatch deadline = Stopwatch.StartNew();
            bool running = true;
            while (running && deadline.ElapsedMilliseconds <= testDeadlineMilliseconds + 250)
            {
                Thread.Sleep(1);
                running = transmission.MoveNext();
            }

            Assert.IsFalse(running, "request-70 response transmission exceeded its fixed deadline");
            Assert.LessOrEqual(deadline.ElapsedMilliseconds, testDeadlineMilliseconds + 250);
            Assert.IsTrue(IsLocallyClosed(server), "the expired request-70 connection was not closed");
        }
        finally
        {
            if (server != null) server.Close();
            client.Close();
            listener.Stop();
        }
    }

    [Test]
    public void Request70JsonEscapesEveryControlCharacterAndRoundTrips()
    {
        char[] controls = new char[32];
        for (int i = 0; i < controls.Length; i++) controls[i] = (char)i;
        string value = new string(controls);
        PhysicalSceneSnapshot snapshot = new PhysicalSceneSnapshot(
            1, 0f, 1, 0.02f, new PhysicalNodeSnapshot[0]);

        byte[] envelope = PhysicsCaptureV1Protocol.BuildCaptureEnvelope(
            new byte[0], snapshot, null, value, 1);
        string json = EnvelopeJson(envelope, false);
        JSONNode parsed = JSONNode.Parse(json);

        Assert.AreEqual(value, parsed["capture_id"].Value);
        for (int i = 0; i < controls.Length; i++)
        {
            Assert.IsFalse(json.Contains(controls[i].ToString()),
                "JSON contains an unescaped U+" + i.ToString("X4") + " control character");
            string expected;
            switch (i)
            {
                case 8: expected = "\\b"; break;
                case 9: expected = "\\t"; break;
                case 10: expected = "\\n"; break;
                case 12: expected = "\\f"; break;
                case 13: expected = "\\r"; break;
                default: expected = "\\u" + i.ToString("x4"); break;
            }
            StringAssert.Contains(expected, json,
                "JSON did not use the deterministic escape for U+" + i.ToString("X4"));
        }
    }

    [Test]
    public void SilentLoopbackClientDoesNotBlockTheRequestCoroutineBeforeItsFirstYield()
    {
        GameObject host = new GameObject("physics-capture-socket");
        PhysicsCaptureDirectSocket socket = PhysicsCaptureDirectSocket.Attach(host, ReserveLoopbackPort());
        TcpClient client = new TcpClient();
        TcpClient accepted = null;
        Task<bool> firstStep = null;

        try
        {
            client.Connect(IPAddress.Loopback, ListeningPort(socket));
            Assert.IsTrue(WaitUntil(delegate { return PendingClients(socket).Count == 1; }),
                "silent loopback client was not accepted");
            Queue<TcpClient> pending = PendingClients(socket);
            lock (pending) accepted = pending.Dequeue();
            IEnumerator coroutine = (IEnumerator)typeof(PhysicsCaptureDirectSocket)
                .GetMethod("Serve", BindingFlags.Instance | BindingFlags.NonPublic)
                .Invoke(socket, new object[] { accepted });

            Stopwatch elapsed = Stopwatch.StartNew();
            firstStep = Task.Run(delegate { return coroutine.MoveNext(); });
            Assert.IsTrue(firstStep.Wait(250),
                "a connected client that sends no byte blocked the request coroutine");
            elapsed.Stop();
            Assert.IsTrue(firstStep.Result, "silent client should yield while awaiting its request byte");
            Assert.Less(elapsed.ElapsedMilliseconds, 250,
                "request-byte polling exceeded the Unity main-thread safety budget");

            elapsed.Restart();
            Assert.IsTrue(coroutine.MoveNext(), "silent client should remain timeout-governed");
            elapsed.Stop();
            Assert.Less(elapsed.ElapsedMilliseconds, 50,
                "advancing the silent-client coroutine on the main thread must not block");
        }
        finally
        {
            client.Close();
            if (accepted != null) accepted.Close();
            if (firstStep != null && !firstStep.IsCompleted)
            {
                try { firstStep.Wait(1000); }
                catch (AggregateException) { }
            }
            UnityEngine.Object.DestroyImmediate(host);
        }
    }

    [Test]
    public void LoopbackConnectionsBeyondPendingCapacityAreRejectedAndClosed()
    {
        const int capacity = PhysicsCaptureDirectSocket.MaxPendingClients;
        GameObject host = new GameObject("physics-capture-socket");
        PhysicsCaptureDirectSocket socket = PhysicsCaptureDirectSocket.Attach(host, ReserveLoopbackPort());
        List<TcpClient> clients = new List<TcpClient>();

        try
        {
            int port = ListeningPort(socket);
            for (int i = 0; i < capacity * 2; i++)
            {
                TcpClient client = new TcpClient();
                client.Connect(IPAddress.Loopback, port);
                clients.Add(client);
            }

            Assert.IsTrue(WaitUntil(delegate
            {
                return PendingClients(socket).Count > capacity
                    || ClosedClientCount(clients) >= capacity;
            }), "loopback burst was not fully accepted or rejected");
            Assert.LessOrEqual(PendingClients(socket).Count, capacity,
                "pending loopback clients grew beyond the fixed capacity");
            Assert.GreaterOrEqual(ClosedClientCount(clients), capacity,
                "connections beyond the fixed pending capacity were not closed");
        }
        finally
        {
            for (int i = 0; i < clients.Count; i++) clients[i].Close();
            UnityEngine.Object.DestroyImmediate(host);
        }
    }

    [Test]
    public void SilentInFlightClientsHoldCapacityUntilTimeoutThenReleaseIt()
    {
        const int capacity = PhysicsCaptureDirectSocket.MaxPendingClients;
        GameObject host = new GameObject("physics-capture-socket");
        PhysicsCaptureDirectSocket socket = PhysicsCaptureDirectSocket.Attach(host, ReserveLoopbackPort());
        List<TcpClient> silentClients = new List<TcpClient>();
        List<TcpClient> acceptedClients = new List<TcpClient>();
        List<IEnumerator> requests = new List<IEnumerator>();
        TcpClient overflow = null;
        TcpClient subsequent = null;

        try
        {
            int port = ListeningPort(socket);
            for (int i = 0; i < capacity; i++)
            {
                TcpClient client = new TcpClient();
                client.Connect(IPAddress.Loopback, port);
                silentClients.Add(client);
            }
            Assert.IsTrue(WaitUntil(delegate { return PendingClients(socket).Count == capacity; }));
            Queue<TcpClient> pending = PendingClients(socket);
            lock (pending)
            {
                while (pending.Count > 0) acceptedClients.Add(pending.Dequeue());
            }
            for (int i = 0; i < acceptedClients.Count; i++)
            {
                IEnumerator request = (IEnumerator)typeof(PhysicsCaptureDirectSocket)
                    .GetMethod("Serve", BindingFlags.Instance | BindingFlags.NonPublic)
                    .Invoke(socket, new object[] { acceptedClients[i] });
                Assert.IsTrue(request.MoveNext(), "silent request should remain in flight");
                requests.Add(request);
            }

            overflow = new TcpClient();
            overflow.Connect(IPAddress.Loopback, port);
            Assert.IsTrue(WaitUntil(delegate { return ClosedClientCount(new List<TcpClient> { overflow }) == 1; }),
                "queued plus in-flight silent clients did not hold the global capacity");

            Thread.Sleep(1100);
            for (int i = 0; i < requests.Count; i++) Assert.IsFalse(requests[i].MoveNext());

            subsequent = new TcpClient();
            subsequent.Connect(IPAddress.Loopback, port);
            Assert.IsTrue(WaitUntil(delegate { return PendingClients(socket).Count == 1; }),
                "timeout cleanup did not release capacity for a normal subsequent request");
            subsequent.GetStream().Write(new byte[] { PhysicsCaptureV1Protocol.RequestCode }, 0, 1);
            TcpClient acceptedSubsequent;
            lock (pending) acceptedSubsequent = pending.Dequeue();
            acceptedClients.Add(acceptedSubsequent);
            Assert.IsTrue(WaitUntil(delegate { return acceptedSubsequent.Available == 1; }));
            IEnumerator normalRequest = (IEnumerator)typeof(PhysicsCaptureDirectSocket)
                .GetMethod("Serve", BindingFlags.Instance | BindingFlags.NonPublic)
                .Invoke(socket, new object[] { acceptedSubsequent });
            Assert.IsTrue(normalRequest.MoveNext());
            Assert.IsInstanceOf<WaitForEndOfFrame>(normalRequest.Current,
                "normal request 70 did not reach the synchronized capture boundary after timeout cleanup");
        }
        finally
        {
            for (int i = 0; i < silentClients.Count; i++) silentClients[i].Close();
            for (int i = 0; i < acceptedClients.Count; i++) acceptedClients[i].Close();
            if (overflow != null) overflow.Close();
            if (subsequent != null) subsequent.Close();
            UnityEngine.Object.DestroyImmediate(host);
        }
    }

    private static void AssertHasFields(JSONNode actual, params string[] required)
    {
        HashSet<string> keys = new HashSet<string>();
        foreach (KeyValuePair<string, JSONNode> item in actual.AsObject)
            keys.Add(item.Key);
        foreach (string field in required)
            Assert.IsTrue(keys.Contains(field), "request-70 producer is missing required field: " + field);
    }

    private static string EnvelopeJson(byte[] envelope, bool events)
    {
        int pngLength = ReadUInt32(envelope, 16);
        int stateLength = ReadUInt32(envelope, 20);
        int eventsLength = ReadUInt32(envelope, 24);
        int componentHeaderLength = envelope[9] == 2 ? 16 : 12;
        int offset = 16 + componentHeaderLength + pngLength + (events ? stateLength : 0);
        return System.Text.Encoding.UTF8.GetString(envelope, offset, events ? eventsLength : stateLength);
    }

    private static string EnvelopeEvidenceJson(byte[] envelope)
    {
        Assert.AreEqual(2, envelope[9], "envelope has no evidence component");
        int pngLength = ReadUInt32(envelope, 16);
        int stateLength = ReadUInt32(envelope, 20);
        int eventsLength = ReadUInt32(envelope, 24);
        int evidenceLength = ReadUInt32(envelope, 28);
        return System.Text.Encoding.UTF8.GetString(
            envelope, 32 + pngLength + stateLength + eventsLength, evidenceLength);
    }

    private static int ReadUInt32(byte[] bytes, int offset)
    {
        return bytes[offset] << 24 | bytes[offset + 1] << 16 | bytes[offset + 2] << 8 | bytes[offset + 3];
    }

    private static int ReserveLoopbackPort()
    {
        TcpListener probe = new TcpListener(IPAddress.Loopback, 0);
        probe.Start();
        int port = ((IPEndPoint)probe.LocalEndpoint).Port;
        probe.Stop();
        return port;
    }

    private static int ListeningPort(PhysicsCaptureDirectSocket socket)
    {
        TcpListener listener = (TcpListener)typeof(PhysicsCaptureDirectSocket)
            .GetField("listener", BindingFlags.Instance | BindingFlags.NonPublic).GetValue(socket);
        return ((IPEndPoint)listener.LocalEndpoint).Port;
    }

    private static Queue<TcpClient> PendingClients(PhysicsCaptureDirectSocket socket)
    {
        return (Queue<TcpClient>)typeof(PhysicsCaptureDirectSocket)
            .GetField("clients", BindingFlags.Instance | BindingFlags.NonPublic).GetValue(socket);
    }

    private static int ClosedClientCount(List<TcpClient> clients)
    {
        int closed = 0;
        for (int i = 0; i < clients.Count; i++)
        {
            try
            {
                Socket socket = clients[i].Client;
                if (socket.Poll(1000, SelectMode.SelectRead) && socket.Available == 0) closed++;
            }
            catch (ObjectDisposedException)
            {
                closed++;
            }
            catch (SocketException)
            {
                closed++;
            }
        }
        return closed;
    }

    private static bool WaitUntil(Func<bool> predicate)
    {
        Stopwatch elapsed = Stopwatch.StartNew();
        while (elapsed.ElapsedMilliseconds < 2000)
        {
            if (predicate()) return true;
            Thread.Sleep(10);
        }
        return predicate();
    }

    private static bool IsLocallyClosed(TcpClient client)
    {
        try
        {
            client.GetStream();
            return false;
        }
        catch (ObjectDisposedException)
        {
            return true;
        }
        catch (InvalidOperationException)
        {
            return true;
        }
    }
}
