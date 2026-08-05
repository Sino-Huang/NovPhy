using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.Net;
using System.Net.Sockets;
using System.Text;
using UnityEngine;

public static class PhysicsCaptureV1Protocol
{
    public const byte RequestCode = 70;
    public const byte Version = 1;
    public const int MaxEnvelopeBytes = 64 * 1024 * 1024;
    public const int MaxPngBytes = 32 * 1024 * 1024;
    public const int MaxJsonBytes = 16 * 1024 * 1024;
    public const int MaxFailureMessageBytes = 16 * 1024 * 1024;
    private static readonly byte[] Magic = { (byte)'S', (byte)'B', (byte)'P', (byte)'V' };

    public static byte[] BuildSuccessEnvelope(byte[] png, string stateJson, string eventsJson)
    {
        if (png != null && png.Length > MaxPngBytes)
            return BuildLimitFailure("PNG exceeds request-70 bounds");

        string stateText = stateJson ?? "{}";
        string eventsText = eventsJson ?? "[]";
        int stateByteCount = Encoding.UTF8.GetByteCount(stateText);
        int eventsByteCount = Encoding.UTF8.GetByteCount(eventsText);
        long payloadLength;
        long envelopeLength;
        try
        {
            payloadLength = checked(12L + (png == null ? 0 : png.Length)
                + stateByteCount + eventsByteCount);
            envelopeLength = checked(16L + payloadLength);
        }
        catch (OverflowException)
        {
            return BuildLimitFailure("request-70 envelope exceeds bounds");
        }
        if (stateByteCount > MaxJsonBytes || eventsByteCount > MaxJsonBytes
            || envelopeLength > MaxEnvelopeBytes)
            return BuildLimitFailure("request-70 envelope exceeds bounds");

        byte[] image = Copy(png);
        byte[] state = Encoding.UTF8.GetBytes(stateText);
        byte[] events = Encoding.UTF8.GetBytes(eventsText);
        byte[] payload = new byte[(int)payloadLength];
        WriteUInt32(payload, 0, (uint)image.Length);
        WriteUInt32(payload, 4, (uint)state.Length);
        WriteUInt32(payload, 8, (uint)events.Length);
        Buffer.BlockCopy(image, 0, payload, 12, image.Length);
        Buffer.BlockCopy(state, 0, payload, 12 + image.Length, state.Length);
        Buffer.BlockCopy(events, 0, payload, 12 + image.Length + state.Length, events.Length);
        return BuildEnvelope(0, 0, payload);
    }

    public static byte[] BuildFailureEnvelope(PhysicalCaptureFailure failure)
    {
        int code = failure == null ? 255 : (int)failure.Code + 1;
        string text = failure == null ? "capture unavailable" : (failure.Message ?? "capture failed");
        int messageByteCount = Encoding.UTF8.GetByteCount(text);
        long envelopeLength;
        try
        {
            envelopeLength = checked(16L + 4L + messageByteCount);
        }
        catch (OverflowException)
        {
            return BuildLimitFailure("capture failure message exceeds bounds");
        }
        if (messageByteCount > MaxFailureMessageBytes || envelopeLength > MaxEnvelopeBytes)
            return BuildLimitFailure("capture failure message exceeds bounds");
        byte[] message = Encoding.UTF8.GetBytes(text);
        byte[] payload = new byte[4 + message.Length];
        WriteUInt32(payload, 0, (uint)message.Length);
        Buffer.BlockCopy(message, 0, payload, 4, message.Length);
        return BuildEnvelope(1, code, payload);
    }

    public static byte[] BuildCaptureEnvelope(byte[] png, PhysicalSceneSnapshot snapshot, PhysicalShotRecorder recorder)
    {
        return BuildCaptureEnvelope(png, snapshot, recorder, "capture-standalone", 1);
    }

    public static byte[] BuildCaptureEnvelope(byte[] png, PhysicalSceneSnapshot snapshot,
        PhysicalShotRecorder recorder, string captureId, long sequence)
    {
        if (snapshot == null)
            return BuildFailureEnvelope(new PhysicalCaptureFailure(PhysicalCaptureFailureCode.TruncatedFinalization, "snapshot unavailable"));
        if (recorder != null && recorder.Failure != null)
            return BuildFailureEnvelope(recorder.Failure);
        PhysicalShotRecorderSnapshot frozen = recorder == null ? null : recorder.CreateFinalizedSnapshot();
        if (recorder != null && frozen == null)
            return BuildFailureEnvelope(new PhysicalCaptureFailure(
                PhysicalCaptureFailureCode.TruncatedFinalization, "no finalized recorder batch"));
        string state = BuildStateJson(snapshot, frozen, captureId, sequence);
        string events = BuildEventsJson(snapshot, captureId, frozen == null ? null : frozen.Events);
        return BuildSuccessEnvelope(png, state, events);
    }

    private static byte[] BuildLimitFailure(string message)
    {
        byte[] bytes = Encoding.UTF8.GetBytes(message);
        byte[] payload = new byte[4 + bytes.Length];
        WriteUInt32(payload, 0, (uint)bytes.Length);
        Buffer.BlockCopy(bytes, 0, payload, 4, bytes.Length);
        return BuildEnvelope(1, (int)PhysicalCaptureFailureCode.EnvelopeLimitExceeded + 1, payload);
    }

    private static byte[] BuildEnvelope(byte flags, int failureCode, byte[] payload)
    {
        int bodyLength = 12 + payload.Length;
        if (bodyLength + 4L > MaxEnvelopeBytes)
            return BuildLimitFailure("request-70 envelope exceeds bounds");
        byte[] result = new byte[4 + bodyLength];
        WriteUInt32(result, 0, (uint)bodyLength);
        Buffer.BlockCopy(Magic, 0, result, 4, Magic.Length);
        result[8] = Version;
        result[9] = flags;
        WriteUInt16(result, 10, (ushort)failureCode);
        WriteUInt32(result, 12, (uint)payload.Length);
        Buffer.BlockCopy(payload, 0, result, 16, payload.Length);
        return result;
    }

    private static string BuildStateJson(PhysicalSceneSnapshot snapshot, PhysicalShotRecorderSnapshot recorder,
        string captureId, long sequence)
    {
        StringBuilder json = new StringBuilder(snapshot.ToJson());
        int insert = json.Length - 1;
        StringBuilder fields = new StringBuilder();
        fields.Append(",\"capture_id\":"); AppendString(fields, captureId);
        fields.Append(",\"sequence\":").Append(sequence.ToString(CultureInfo.InvariantCulture));
        fields.Append(",\"rgb_frame\":{\"render_frame\":")
            .Append(snapshot.RenderFrame.ToString(CultureInfo.InvariantCulture))
            .Append(",\"source\":\"synchronized_endpoint\"}");
        fields.Append(",\"raw_contacts\":[").Append(BuildContactsJson(recorder == null ? null : recorder.RawContacts));
        fields.Append("],\"support_edges\":[").Append(BuildSupportJson(recorder == null ? null : recorder.SupportEdges));
        fields.Append(']');
        json.Insert(insert, fields.ToString());
        return json.ToString();
    }

    private static string BuildContactsJson(IList<PhysicalRawContact> contacts)
    {
        StringBuilder json = new StringBuilder();
        for (int i = 0; contacts != null && i < contacts.Count; i++)
        {
            if (i > 0) json.Append(',');
            PhysicalRawContact c = contacts[i];
            json.Append("{\"contact_id\":"); AppendString(json, c.ContactId);
            json.Append(",\"entity_a_id\":"); AppendString(json, c.EntityIdA);
            json.Append(",\"entity_b_id\":"); AppendString(json, c.EntityIdB);
            json.Append(",\"collider_a_id\":").Append(c.ColliderIdA.ToString(CultureInfo.InvariantCulture));
            json.Append(",\"collider_b_id\":").Append(c.ColliderIdB.ToString(CultureInfo.InvariantCulture));
            json.Append(",\"point\":"); AppendVector(json, c.Point);
            json.Append(",\"normal_a_to_b\":"); AppendVector(json, c.Normal);
            json.Append(",\"separation\":"); AppendFloat(json, c.Separation);
            json.Append(",\"relative_velocity_a_to_b\":"); AppendVector(json, c.RelativeVelocity);
            json.Append(",\"normal_impulse\":"); AppendFloat(json, c.NormalImpulse);
            json.Append(",\"tangent_impulse\":"); AppendFloat(json, c.TangentImpulse);
            json.Append(",\"is_trigger\":").Append(c.IsTrigger ? "true" : "false");
            json.Append('}');
        }
        return json.ToString();
    }

    private static string BuildSupportJson(IList<PhysicalSupportEdge> edges)
    {
        StringBuilder json = new StringBuilder();
        for (int i = 0; edges != null && i < edges.Count; i++)
        {
            if (i > 0) json.Append(',');
            PhysicalSupportEdge edge = edges[i];
            json.Append("{\"support_id\":");
            AppendString(json, "support:" + edge.SupporterEntityId + "->" + edge.SupportedEntityId);
            json.Append(",\"rule_version\":\"support_v1\",\"supporter_id\":"); AppendString(json, edge.SupporterEntityId);
            json.Append(",\"supported_id\":"); AppendString(json, edge.SupportedEntityId);
            json.Append(",\"evidence_contact_ids\":["); AppendString(json, edge.ContactIdA);
            json.Append(','); AppendString(json, edge.ContactIdB);
            json.Append("],\"evidence_fixed_steps\":[").Append(edge.FixedStepA.ToString(CultureInfo.InvariantCulture));
            json.Append(',').Append(edge.FixedStepB.ToString(CultureInfo.InvariantCulture)).Append("]}");
        }
        return json.ToString();
    }

    private static string BuildEventsJson(PhysicalSceneSnapshot snapshot, string captureId,
        IList<PhysicalMacroEvent> events)
    {
        StringBuilder json = new StringBuilder("[");
        for (int i = 0; events != null && i < events.Count; i++)
        {
            if (i > 0) json.Append(',');
            PhysicalMacroEvent item = events[i];
            json.Append("{\"schema_version\":\"physics_capture_v1\",\"capture_id\":"); AppendString(json, captureId);
            json.Append(",\"sequence\":").Append(i.ToString(CultureInfo.InvariantCulture));
            json.Append(",\"render_frame\":").Append(snapshot.RenderFrame.ToString(CultureInfo.InvariantCulture));
            json.Append(",\"render_time\":"); AppendFloat(json, snapshot.RenderTime);
            json.Append(",\"fixed_step\":").Append(item.FixedStep.ToString(CultureInfo.InvariantCulture));
            json.Append(",\"fixed_time\":"); AppendFloat(json, item.FixedTime);
            AppendCoordinates(json);
            json.Append(",\"event_id\":"); AppendString(json, "event:" + i.ToString("D8", CultureInfo.InvariantCulture));
            json.Append(",\"event_type\":"); AppendString(json, item.Taxonomy);
            json.Append(",\"participants\":[");
            for (int p = 0; p < item.Participants.Count; p++)
            {
                if (p > 0) json.Append(',');
                AppendString(json, item.Participants[p]);
            }
            json.Append("],\"payload\":"); AppendPayload(json, item.Payload);
            json.Append('}');
        }
        return json.Append(']').ToString();
    }

    private static void AppendPayload(StringBuilder json, PhysicalMacroEventPayload payload)
    {
        payload = payload ?? new PhysicalMacroEventPayload();
        json.Append('{');
        if (payload.LaunchVelocity.HasValue) { json.Append("\"launch_velocity\":"); AppendVector(json, payload.LaunchVelocity.Value); }
        else if (payload.ContactIds.Count > 0) { json.Append("\"contact_ids\":["); for (int i = 0; i < payload.ContactIds.Count; i++) { if (i > 0) json.Append(','); AppendString(json, payload.ContactIds[i]); } json.Append("],\"relative_speed\":"); AppendFloat(json, payload.RelativeSpeed ?? 0f); }
        else if (payload.RadiusUnityUnits.HasValue) { json.Append("\"radius_unity_units\":"); AppendFloat(json, payload.RadiusUnityUnits.Value); }
        else if (payload.Reason != null) { json.Append("\"reason\":"); AppendString(json, payload.Reason); }
        else if (payload.BirdsRemaining.HasValue) json.Append("\"birds_remaining\":").Append(payload.BirdsRemaining.Value.ToString(CultureInfo.InvariantCulture));
        else if (payload.DebounceFixedSteps.HasValue) json.Append("\"debounce_fixed_steps\":").Append(payload.DebounceFixedSteps.Value.ToString(CultureInfo.InvariantCulture));
        else if (payload.Score.HasValue) json.Append("\"score\":").Append(payload.Score.Value.ToString(CultureInfo.InvariantCulture));
        json.Append('}');
    }

    private static void AppendCoordinates(StringBuilder json)
    {
        json.Append(",\"coordinates\":{\"world_space\":\"unity_world_2d\",\"world_origin\":\"scene_defined\",\"world_x_axis\":\"right\",\"world_y_axis\":\"up\",\"world_length_unit\":\"unity_unit\",\"screen_space\":\"rgb_pixel_2d\",\"screen_origin\":\"top_left\",\"screen_x_axis\":\"right\",\"screen_y_axis\":\"down\",\"screen_length_unit\":\"pixel\",\"time_unit\":\"second\",\"angle_unit\":\"degree\",\"mass_unit\":\"unity_mass_unit\",\"velocity_unit\":\"unity_unit/second\",\"angular_velocity_unit\":\"degree/second\",\"kinetic_energy_unit\":\"unity_mass_unit*unity_unit^2/second^2\",\"impulse_unit\":\"unity_mass_unit*unity_unit/second\"}");
    }

    private static void AppendVector(StringBuilder json, Vector2 value)
    {
        json.Append("{\"x\":"); AppendFloat(json, value.x);
        json.Append(",\"y\":"); AppendFloat(json, value.y); json.Append('}');
    }

    private static void AppendFloat(StringBuilder json, float value)
    {
        json.Append(value.ToString("R", CultureInfo.InvariantCulture));
    }

    private static void AppendString(StringBuilder json, string value)
    {
        json.Append('"');
        string text = value ?? string.Empty;
        for (int i = 0; i < text.Length; i++)
        {
            char c = text[i];
            if (c == '"' || c == '\\') json.Append('\\').Append(c);
            else if (c == '\n') json.Append("\\n");
            else if (c == '\r') json.Append("\\r");
            else if (c == '\t') json.Append("\\t");
            else json.Append(c);
        }
        json.Append('"');
    }

    private static byte[] Copy(byte[] source)
    {
        byte[] copy = source == null ? new byte[0] : new byte[source.Length];
        if (source != null) Buffer.BlockCopy(source, 0, copy, 0, source.Length);
        return copy;
    }

    private static void WriteUInt16(byte[] buffer, int offset, ushort value)
    {
        buffer[offset] = (byte)(value >> 8);
        buffer[offset + 1] = (byte)value;
    }

    private static void WriteUInt32(byte[] buffer, int offset, uint value)
    {
        buffer[offset] = (byte)(value >> 24);
        buffer[offset + 1] = (byte)(value >> 16);
        buffer[offset + 2] = (byte)(value >> 8);
        buffer[offset + 3] = (byte)value;
    }
}

public sealed class PhysicsCaptureDirectSocket : MonoBehaviour
{
    public const int MaxPendingClients = 4;
    private const int RequestReadTimeoutMilliseconds = 1000;
    private readonly Queue<TcpClient> clients = new Queue<TcpClient>();
    private TcpListener listener;
    private int reservedClientCount;
    private volatile bool stopping;

    public static PhysicsCaptureDirectSocket Attach(GameObject host, int port)
    {
        PhysicsCaptureDirectSocket socket = host.GetComponent<PhysicsCaptureDirectSocket>();
        if (socket == null) socket = host.AddComponent<PhysicsCaptureDirectSocket>();
        socket.StartListening(port);
        return socket;
    }

    public void StartListening(int port)
    {
        if (listener != null) return;
        listener = new TcpListener(IPAddress.Loopback, port);
        listener.Start();
        listener.BeginAcceptTcpClient(AcceptClient, null);
    }

    private void AcceptClient(IAsyncResult result)
    {
        TcpClient client;
        try
        {
            client = listener.EndAcceptTcpClient(result);
        }
        catch (ObjectDisposedException) { return; }
        catch (SocketException) { return; }
        bool reject;
        lock (clients)
        {
            reject = stopping || reservedClientCount >= MaxPendingClients;
            if (!reject)
            {
                clients.Enqueue(client);
                reservedClientCount++;
            }
        }
        if (reject) client.Close();
        if (!stopping) listener.BeginAcceptTcpClient(AcceptClient, null);
    }

    private void Update()
    {
        TcpClient client = null;
        lock (clients)
        {
            if (clients.Count > 0) client = clients.Dequeue();
        }
        if (client != null) StartCoroutine(Serve(client));
    }

    private IEnumerator Serve(TcpClient client)
    {
        try
        {
            NetworkStream stream = client.GetStream();
            Stopwatch requestWait = Stopwatch.StartNew();
            while (client.Available == 0 && requestWait.ElapsedMilliseconds < RequestReadTimeoutMilliseconds)
                yield return null;
            if (client.Available == 0) yield break;
            byte[] request = new byte[1];
            int read = stream.Read(request, 0, 1);
            if (read != 1 || request[0] != PhysicsCaptureV1Protocol.RequestCode)
            {
                byte[] failure = PhysicsCaptureV1Protocol.BuildFailureEnvelope(null);
                stream.Write(failure, 0, failure.Length);
                yield break;
            }
            yield return new WaitForEndOfFrame();
            PhysicalSnapshotRuntime runtime = PhysicalSnapshotRuntime.Active;
            if (runtime == null) yield break;
            PhysicalSceneSnapshot snapshot = runtime.CaptureCurrent(new SymbolicGameState(false), Time.frameCount, Time.time);
            Texture2D texture = ScreenCapture.CaptureScreenshotAsTexture();
            byte[] png = texture == null ? new byte[0] : texture.EncodeToPNG();
            if (texture != null) Destroy(texture);
            byte[] response = PhysicsCaptureV1Protocol.BuildCaptureEnvelope(
                png, snapshot, runtime.ShotRecorder, runtime.CaptureId, runtime.NextCaptureSequence);
            stream.Write(response, 0, response.Length);
        }
        finally
        {
            client.Close();
            lock (clients) reservedClientCount--;
        }
    }

    private void OnDestroy()
    {
        stopping = true;
        if (listener != null) listener.Stop();
        lock (clients)
        {
            while (clients.Count > 0)
            {
                clients.Dequeue().Close();
                reservedClientCount--;
            }
        }
    }
}
