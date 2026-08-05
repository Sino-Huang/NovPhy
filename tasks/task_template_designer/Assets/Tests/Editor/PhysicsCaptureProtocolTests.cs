using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Diagnostics;
using System.Net;
using System.Net.Sockets;
using System.Reflection;
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
        Assert.AreEqual(0x89, envelope[28]);
        Assert.AreEqual(0x50, envelope[29]);
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
        JSONNode schema = JSONNode.Parse(File.ReadAllText(Path.GetFullPath(Path.Combine(
            Application.dataPath, "../../../docs/data_contracts/physics_capture_v1.schema.json"))));
        JSONNode state = JSONNode.Parse(EnvelopeJson(envelope, false));
        JSONNode events = JSONNode.Parse(EnvelopeJson(envelope, true));

        AssertRequiredExcept(state, schema["$defs"]["record_clock"]["required"].AsArray, "shot_id");
        AssertRequiredExcept(state, schema["$defs"]["state"]["allOf"][1]["required"].AsArray,
            "record_type", "rgb_frame");
        AssertRequiredExcept(state["rgb_frame"], schema["$defs"]["rgb_frame"]["required"].AsArray,
            "relative_path", "width_pixels", "height_pixels");
        AssertRequiredExcept(state["raw_contacts"][0], schema["$defs"]["raw_contact"]["required"].AsArray);
        AssertRequiredExcept(state["support_edges"][0], schema["$defs"]["support_edge"]["required"].AsArray);
        Assert.Greater(events.Count, 0);
        AssertRequiredExcept(events[0], schema["$defs"]["record_clock"]["required"].AsArray, "shot_id");
        AssertRequiredExcept(events[0], schema["$defs"]["event"]["allOf"][1]["required"].AsArray, "record_type");
        Assert.AreEqual("physics_capture_v1", state["schema_version"].Value);
        Assert.AreEqual("synchronized_endpoint", state["rgb_frame"]["source"].Value);
        Assert.AreEqual(state["render_frame"].AsInt, state["rgb_frame"]["render_frame"].AsInt);
        Assert.AreEqual(0, events[0]["sequence"].AsInt);
        Assert.AreEqual("event:00000000", events[0]["event_id"].Value);
        Assert.AreEqual("bird_launched", events[0]["event_type"].Value);
        Assert.AreEqual(2f, events[0]["payload"]["launch_velocity"]["x"].AsFloat);
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

    private static void AssertRequiredExcept(JSONNode actual, JSONArray required, params string[] suppliedByConsumer)
    {
        HashSet<string> keys = new HashSet<string>();
        foreach (KeyValuePair<string, JSONNode> item in actual.AsObject)
            keys.Add(item.Key);
        HashSet<string> consumerFields = new HashSet<string>(suppliedByConsumer);
        foreach (JSONNode field in required)
        {
            if (!consumerFields.Contains(field.Value))
                Assert.IsTrue(keys.Contains(field.Value), "request-70 producer is missing required field: " + field.Value);
        }
    }

    private static string EnvelopeJson(byte[] envelope, bool events)
    {
        int pngLength = ReadUInt32(envelope, 16);
        int stateLength = ReadUInt32(envelope, 20);
        int eventsLength = ReadUInt32(envelope, 24);
        int offset = 28 + pngLength + (events ? stateLength : 0);
        return System.Text.Encoding.UTF8.GetString(envelope, offset, events ? eventsLength : stateLength);
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
}
