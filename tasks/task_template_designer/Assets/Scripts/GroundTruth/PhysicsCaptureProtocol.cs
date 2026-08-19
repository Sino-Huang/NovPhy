using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
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
        return BuildSuccessEnvelopeInternal(png, stateJson, eventsJson, null, false);
    }

    public static byte[] BuildSuccessEnvelope(byte[] png, string stateJson, string eventsJson, string evidenceJson)
    {
        return BuildSuccessEnvelopeInternal(png, stateJson, eventsJson, evidenceJson ?? "null", true);
    }

    private static byte[] BuildSuccessEnvelopeInternal(byte[] png, string stateJson, string eventsJson,
        string evidenceJson, bool carriesEvidenceComponent)
    {
        if (png != null && png.Length > MaxPngBytes)
            return BuildLimitFailure("PNG exceeds request-70 bounds");

        string stateText = stateJson ?? "{}";
        string eventsText = eventsJson ?? "[]";
        int stateByteCount = Encoding.UTF8.GetByteCount(stateText);
        int eventsByteCount = Encoding.UTF8.GetByteCount(eventsText);
        int evidenceByteCount = carriesEvidenceComponent ? Encoding.UTF8.GetByteCount(evidenceJson) : 0;
        long payloadLength;
        long envelopeLength;
        try
        {
            payloadLength = checked((carriesEvidenceComponent ? 16L : 12L)
                + (png == null ? 0 : png.Length) + stateByteCount + eventsByteCount + evidenceByteCount);
            envelopeLength = checked(16L + payloadLength);
        }
        catch (OverflowException)
        {
            return BuildLimitFailure("request-70 envelope exceeds bounds");
        }
        if (stateByteCount > MaxJsonBytes || eventsByteCount > MaxJsonBytes
            || evidenceByteCount > MaxJsonBytes
            || envelopeLength > MaxEnvelopeBytes)
            return BuildLimitFailure("request-70 envelope exceeds bounds");

        byte[] image = Copy(png);
        byte[] state = Encoding.UTF8.GetBytes(stateText);
        byte[] events = Encoding.UTF8.GetBytes(eventsText);
        byte[] evidence = carriesEvidenceComponent ? Encoding.UTF8.GetBytes(evidenceJson) : new byte[0];
        byte[] payload = new byte[(int)payloadLength];
        WriteUInt32(payload, 0, (uint)image.Length);
        WriteUInt32(payload, 4, (uint)state.Length);
        WriteUInt32(payload, 8, (uint)events.Length);
        int headerLength = carriesEvidenceComponent ? 16 : 12;
        if (carriesEvidenceComponent) WriteUInt32(payload, 12, (uint)evidence.Length);
        Buffer.BlockCopy(image, 0, payload, headerLength, image.Length);
        Buffer.BlockCopy(state, 0, payload, headerLength + image.Length, state.Length);
        Buffer.BlockCopy(events, 0, payload, headerLength + image.Length + state.Length, events.Length);
        if (carriesEvidenceComponent)
            Buffer.BlockCopy(evidence, 0, payload,
                headerLength + image.Length + state.Length + events.Length, evidence.Length);
        return BuildEnvelope(carriesEvidenceComponent ? (byte)2 : (byte)0, 0, payload);
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
        PhysicalViolationEngineEvidenceSnapshot evidence = recorder == null
            ? null : recorder.CreateFinalizedEvidenceSnapshot();
        string evidenceJson = evidence == null ? "null" : BuildEvidenceJson(evidence, captureId, sequence);
        return BuildSuccessEnvelope(png, state, events, evidenceJson);
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

    private static string BuildEvidenceJson(PhysicalViolationEngineEvidenceSnapshot evidence,
        string captureId, long sequence)
    {
        StringBuilder json = new StringBuilder("{\"schema_version\":\"")
            .Append(PhysicalViolationEngineEvidenceSnapshot.SchemaVersion).Append("\",\"capture_id\":");
        AppendString(json, captureId);
        json.Append(",\"shot_id\":"); AppendString(json, evidence.ShotId);
        json.Append(",\"sequence\":").Append(sequence.ToString(CultureInfo.InvariantCulture));
        json.Append(",\"fixed_step_coverage\":{\"first_fixed_step\":");
        AppendNullableLong(json, evidence.FirstFixedStep);
        json.Append(",\"last_fixed_step\":"); AppendNullableLong(json, evidence.LastFixedStep);
        json.Append(",\"sample_count\":").Append(evidence.SampleCount.ToString(CultureInfo.InvariantCulture));
        json.Append(",\"complete\":").Append(evidence.Complete ? "true" : "false");
        json.Append(",\"incomplete_reason\":"); AppendNullableString(json, evidence.IncompleteReason);
        json.Append("},\"minimum_contact_separation\":{\"observed\":")
            .Append(evidence.MinimumObserved ? "true" : "false");
        json.Append(",\"separation\":");
        if (evidence.MinimumSeparation.HasValue) AppendFloat(json, evidence.MinimumSeparation.Value); else json.Append("null");
        json.Append(",\"contact_id\":"); AppendNullableString(json, evidence.MinimumContactId);
        json.Append(",\"fixed_step\":"); AppendNullableLong(json, evidence.MinimumFixedStep);
        json.Append("},\"terminal_trace\":{\"max_fixed_steps\":")
            .Append(PhysicalViolationEngineEvidenceSnapshot.MaxTraceFixedSteps)
            .Append(",\"max_entities_per_step\":")
            .Append(PhysicalViolationEngineEvidenceSnapshot.MaxEntitiesPerStep)
            .Append(",\"first_fixed_step\":");
        AppendNullableLong(json, evidence.Trace.Count == 0 ? (long?)null : evidence.Trace[0].FixedStep);
        json.Append(",\"last_fixed_step\":");
        AppendNullableLong(json, evidence.Trace.Count == 0 ? (long?)null : evidence.Trace[evidence.Trace.Count - 1].FixedStep);
        json.Append(",\"truncated\":").Append(evidence.TraceTruncated ? "true" : "false");
        json.Append(",\"truncation_reason\":");
        AppendNullableString(json, evidence.TraceTruncated ? "terminal_trace_bound" : null);
        json.Append(",\"failure_reason\":"); AppendNullableString(json, evidence.IncompleteReason);
        json.Append(",\"samples\":[");
        for (int i = 0; i < evidence.Trace.Count; i++)
        {
            if (i > 0) json.Append(',');
            PhysicalEvidenceTraceSample sample = evidence.Trace[i];
            json.Append("{\"fixed_step\":").Append(sample.FixedStep.ToString(CultureInfo.InvariantCulture));
            json.Append(",\"physics2d_gravity\":"); AppendVector(json, sample.Physics2DGravity);
            json.Append(",\"entities\":[");
            for (int e = 0; e < sample.Entities.Count; e++)
            {
                if (e > 0) json.Append(',');
                AppendEvidenceEntity(json, sample.Entities[e]);
            }
            json.Append("]}");
        }
        return json.Append("]}}").ToString();
    }

    private static void AppendEvidenceEntity(StringBuilder json, PhysicalEvidenceEntity entity)
    {
        json.Append("{\"entity_id\":"); AppendString(json, entity.EntityId);
        json.Append(",\"observed\":true,\"present\":").Append(entity.Present ? "true" : "false");
        json.Append(",\"world_position\":");
        if (entity.WorldPosition.HasValue) AppendVector(json, entity.WorldPosition.Value); else json.Append("null");
        json.Append(",\"body_type\":"); AppendNullableString(json, entity.BodyType);
        json.Append(",\"simulated\":").Append(entity.Simulated.HasValue ? (entity.Simulated.Value ? "true" : "false") : "null");
        json.Append(",\"gravity_scale\":");
        if (entity.GravityScale.HasValue) AppendFloat(json, entity.GravityScale.Value); else json.Append("null");
        json.Append(",\"support_v1\":{\"present\":").Append(entity.Supports.Count > 0 ? "true" : "false");
        json.Append(",\"edges\":[");
        for (int i = 0; i < entity.Supports.Count; i++)
        {
            if (i > 0) json.Append(',');
            PhysicalEvidenceSupport support = entity.Supports[i];
            json.Append("{\"support_id\":"); AppendString(json, support.SupportId);
            json.Append(",\"supporter_id\":"); AppendString(json, support.SupporterId);
            json.Append(",\"evidence_contact_ids\":["); AppendString(json, support.ContactIdA);
            json.Append(','); AppendString(json, support.ContactIdB);
            json.Append("],\"evidence_fixed_steps\":[")
                .Append(support.FixedStepA.ToString(CultureInfo.InvariantCulture)).Append(',')
                .Append(support.FixedStepB.ToString(CultureInfo.InvariantCulture)).Append("]}");
        }
        json.Append("]}}");
    }

    private static void AppendNullableLong(StringBuilder json, long? value)
    {
        if (value.HasValue) json.Append(value.Value.ToString(CultureInfo.InvariantCulture));
        else json.Append("null");
    }

    private static void AppendNullableString(StringBuilder json, string value)
    {
        if (value == null) json.Append("null"); else AppendString(json, value);
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
            switch (c)
            {
                case '"': json.Append("\\\""); break;
                case '\\': json.Append("\\\\"); break;
                case '\b': json.Append("\\b"); break;
                case '\f': json.Append("\\f"); break;
                case '\n': json.Append("\\n"); break;
                case '\r': json.Append("\\r"); break;
                case '\t': json.Append("\\t"); break;
                default:
                    if (c < 32)
                        json.Append("\\u").Append(((int)c).ToString("x4", CultureInfo.InvariantCulture));
                    else
                        json.Append(c);
                    break;
            }
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
    private const int ResponseWriteTimeoutMilliseconds = 1000;
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
                IEnumerator failureTransmission = TransmitResponse(
                    client, failure, ResponseWriteTimeoutMilliseconds);
                while (failureTransmission.MoveNext())
                    yield return failureTransmission.Current;
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
            IEnumerator transmission = TransmitResponse(
                client, response, ResponseWriteTimeoutMilliseconds);
            while (transmission.MoveNext())
                yield return transmission.Current;
        }
        finally
        {
            client.Close();
            lock (clients) reservedClientCount--;
        }
    }

    private sealed class ResponseWriteCompletion
    {
        public readonly NetworkStream Stream;
        public volatile bool EndWriteFinished;

        public ResponseWriteCompletion(NetworkStream stream)
        {
            Stream = stream;
        }
    }

    private static IEnumerator TransmitResponse(TcpClient client, byte[] response, int timeoutMilliseconds)
    {
        if (response == null || response.Length > PhysicsCaptureV1Protocol.MaxEnvelopeBytes)
        {
            client.Close();
            yield break;
        }

        NetworkStream stream = client.GetStream();
        ResponseWriteCompletion completion = new ResponseWriteCompletion(stream);
        try
        {
            stream.BeginWrite(response, 0, response.Length, QueueEndWrite, completion);
        }
        catch (IOException)
        {
            client.Close();
            yield break;
        }
        catch (ObjectDisposedException)
        {
            yield break;
        }

        Stopwatch deadline = Stopwatch.StartNew();
        while (!completion.EndWriteFinished && deadline.ElapsedMilliseconds < timeoutMilliseconds)
            yield return null;
        if (!completion.EndWriteFinished)
        {
            client.Close();
        }
    }

    private static void QueueEndWrite(IAsyncResult write)
    {
        ThreadPool.QueueUserWorkItem(delegate
        {
            ResponseWriteCompletion completion = (ResponseWriteCompletion)write.AsyncState;
            try
            {
                completion.Stream.EndWrite(write);
            }
            catch (IOException) { }
            catch (ObjectDisposedException) { }
            finally
            {
                completion.EndWriteFinished = true;
            }
        });
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
