using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;

public enum PhysicsCaptureV2EngineFailureCode
{
    MissingCaptureStride = 1,
    InvalidCaptureStride = 2,
    CaptureUnavailable = 3,
    EnvelopeLimitExceeded = 4,
    MissingScenarioObjectIdentity = 5,
    UnsupportedColliderGeometry = 6,
    IncompleteColliderGeometry = 7,
    NonFiniteValue = 8,
    EntityLimitExceeded = 9,
    ColliderLimitExceeded = 10,
    IncompleteContactEnumeration = 11,
    ContactLimitExceeded = 12,
    UnresolvedContactIdentity = 13,
    FixedStepGap = 14
}

public static class PhysicsCaptureV2EngineProtocol
{
    public const byte RequestCode = 71;
    public const byte Version = 1;
    public const int MaxEnvelopeBytes = 64 * 1024 * 1024;
    public const string StrideEnvironmentVariable = "NOVPHY_PHYSICS_CAPTURE_V2_STRIDE";
    private static readonly byte[] Magic = { (byte)'S', (byte)'B', (byte)'P', (byte)'2' };

    public static byte[] BuildCaptureEnvelope()
    {
        PhysicsCaptureV2FixedStepRecorder recorder = PhysicsCaptureV2FixedStepRecorder.Active;
        if (recorder != null && recorder.Failure != null)
            return BuildFailureEnvelope(recorder.Failure.Code, recorder.Failure.Message);
        return BuildCaptureEnvelope(recorder == null ? null : recorder.CreateFinalizedSnapshot());
    }

    public static byte[] BuildCaptureEnvelope(PhysicsCaptureV2EngineSnapshot snapshot)
    {
        string configuredStride = Environment.GetEnvironmentVariable(
            StrideEnvironmentVariable, EnvironmentVariableTarget.Process);
        if (configuredStride == null)
            return BuildFailureEnvelope(PhysicsCaptureV2EngineFailureCode.MissingCaptureStride,
                "NOVPHY_PHYSICS_CAPTURE_V2_STRIDE is missing");
        int stride;
        if (!int.TryParse(configuredStride, NumberStyles.None, CultureInfo.InvariantCulture,
            out stride) || stride <= 0)
            return BuildFailureEnvelope(PhysicsCaptureV2EngineFailureCode.InvalidCaptureStride,
                "NOVPHY_PHYSICS_CAPTURE_V2_STRIDE must be a positive integer");
        if (snapshot == null || snapshot.ConfiguredFixedStepCaptureStride != stride)
            return BuildFailureEnvelope(PhysicsCaptureV2EngineFailureCode.CaptureUnavailable,
                "no finalized physics capture v2 record for the configured stride");
        PhysicsCaptureV2RecorderFailure snapshotFailure;
        if (!PhysicsCaptureV2EntityGeometryExporter.TryValidateFinite(snapshot, out snapshotFailure))
            return BuildFailureEnvelope(snapshotFailure.Code, snapshotFailure.Message);
        string json = BuildCaptureJson(snapshot);
        if (Encoding.UTF8.GetByteCount(json) > MaxEnvelopeBytes - 16)
            return BuildFailureEnvelope(PhysicsCaptureV2EngineFailureCode.EnvelopeLimitExceeded,
                "physics capture v2 envelope exceeds bounds");
        return BuildEnvelope(0, 0, Encoding.UTF8.GetBytes(json));
    }

    private static string BuildCaptureJson(PhysicsCaptureV2EngineSnapshot snapshot)
    {
        SortedDictionary<string, PhysicsCaptureV2EntitySnapshot> entities =
            new SortedDictionary<string, PhysicsCaptureV2EntitySnapshot>(StringComparer.Ordinal);
        SortedDictionary<string, PhysicsCaptureV2ColliderSnapshot> colliders =
            new SortedDictionary<string, PhysicsCaptureV2ColliderSnapshot>(StringComparer.Ordinal);
        PhysicsCaptureV2ContactSnapshot minimum = null;
        long minimumStep = 0;
        for (int sampleIndex = 0; sampleIndex < snapshot.FixedStepSamples.Count; sampleIndex++)
        {
            PhysicsCaptureV2FixedStepSample sample = snapshot.FixedStepSamples[sampleIndex];
            for (int i = 0; i < sample.Entities.Count; i++)
                if (!entities.ContainsKey(sample.Entities[i].EntityId))
                    entities.Add(sample.Entities[i].EntityId, sample.Entities[i]);
            for (int i = 0; i < sample.Colliders.Count; i++)
                if (!colliders.ContainsKey(sample.Colliders[i].ColliderId))
                    colliders.Add(sample.Colliders[i].ColliderId, sample.Colliders[i]);
            for (int i = 0; i < sample.Contacts.Count; i++)
                if (minimum == null || sample.Contacts[i].Separation < minimum.Separation
                    || (sample.Contacts[i].Separation == minimum.Separation
                        && (sample.FixedStep < minimumStep || (sample.FixedStep == minimumStep
                            && string.CompareOrdinal(sample.Contacts[i].ContactId,
                                minimum.ContactId) < 0))))
                {
                    minimum = sample.Contacts[i]; minimumStep = sample.FixedStep;
                }
        }
        StringBuilder json = new StringBuilder("{\"schema_version\":\"physics_capture_v2_engine_v1\","
            + "\"capture_id\":");
        AppendString(json, snapshot.CaptureId);
        json.Append(",\"shot_id\":"); AppendString(json, snapshot.ShotId);
        json.Append(",\"configured_fixed_step_capture_stride\":")
            .Append(snapshot.ConfiguredFixedStepCaptureStride.ToString(CultureInfo.InvariantCulture))
            .Append(",\"pre_intervention_fixed_step\":")
            .Append(snapshot.PreInterventionFixedStep.ToString(CultureInfo.InvariantCulture))
            .Append(",\"coordinate_convention\":{\"world_space\":\"unity_world_2d\","
                + "\"world_x_axis\":\"right\",\"world_y_axis\":\"up\","
                + "\"world_length_unit\":\"unity_unit\"},\"causal_entities\":[");
        int rootIndex = 0;
        foreach (string entityId in entities.Keys)
        {
            if (rootIndex++ > 0) json.Append(','); AppendString(json, entityId);
        }
        json.Append("],\"colliders\":["); rootIndex = 0;
        foreach (PhysicsCaptureV2ColliderSnapshot collider in colliders.Values)
        {
            if (rootIndex++ > 0) json.Append(','); AppendColliderCatalog(json, collider);
        }
        json.Append("],\"fixed_step_samples\":[");
        for (int i = 0; i < snapshot.FixedStepSamples.Count; i++)
        {
            if (i > 0) json.Append(',');
            PhysicsCaptureV2FixedStepSample sample = snapshot.FixedStepSamples[i];
            json.Append("{\"fixed_step\":")
                .Append(sample.FixedStep.ToString(CultureInfo.InvariantCulture))
                .Append(",\"complete_raw_non_trigger_contacts\":true,\"world\":{"
                    + "\"world_id\":\"unity-physics2d\",\"gravity_vector\":");
            AppendVector(json, sample.WorldGravity);
            json.Append("},\"entities\":[");
            for (int entityIndex = 0; entityIndex < sample.Entities.Count; entityIndex++)
            {
                if (entityIndex > 0) json.Append(',');
                AppendEntity(json, sample.Entities[entityIndex]);
            }
            json.Append("],\"colliders\":[");
            for (int colliderIndex = 0; colliderIndex < sample.Colliders.Count; colliderIndex++)
            {
                if (colliderIndex > 0) json.Append(',');
                AppendCollider(json, sample.Colliders[colliderIndex]);
            }
            json.Append("],\"contacts\":[");
            for (int contactIndex = 0; contactIndex < sample.Contacts.Count; contactIndex++)
            {
                if (contactIndex > 0) json.Append(',');
                AppendContact(json, sample.Contacts[contactIndex]);
            }
            json.Append("],\"supports\":[");
            for (int supportIndex = 0; supportIndex < sample.Supports.Count; supportIndex++)
            {
                if (supportIndex > 0) json.Append(',');
                AppendSupport(json, sample.Supports[supportIndex]);
            }
            json.Append("]}");
        }
        json.Append("],\"minimum_contact_separation\":{\"observed\":")
            .Append(minimum == null ? "false" : "true").Append(",\"separation\":");
        if (minimum == null) json.Append("null"); else AppendFloat(json, minimum.Separation);
        json.Append(",\"contact_id\":");
        if (minimum == null) json.Append("null"); else AppendString(json, minimum.ContactId);
        json.Append(",\"fixed_step\":");
        if (minimum == null) json.Append("null");
        else json.Append(minimumStep.ToString(CultureInfo.InvariantCulture));
        json.Append("},\"frame_records\":[");
        for (int i = 0; i < snapshot.FrameRecords.Count; i++)
        {
            if (i > 0) json.Append(',');
            PhysicsCaptureV2FrameRecord record = snapshot.FrameRecords[i];
            string fixedStep = record.FixedStep.ToString(CultureInfo.InvariantCulture);
            json.Append("{\"fixed_step\":").Append(fixedStep)
                .Append(",\"state_id\":\"state:").Append(fixedStep)
                .Append("\",\"forced_terminal\":")
                .Append(record.ForcedTerminal ? "true" : "false").Append('}');
        }
        string terminalEventId = snapshot.TerminalEventId;
        string terminalReason = string.IsNullOrEmpty(snapshot.TerminalReason)
            ? "terminal" : snapshot.TerminalReason;
        json.Append("],\"events\":[");
        for (int eventIndex = 0; eventIndex < snapshot.Events.Count; eventIndex++)
        {
            if (eventIndex > 0) json.Append(',');
            PhysicsCaptureV2EventSnapshot item = snapshot.Events[eventIndex];
            json.Append("{\"event_id\":"); AppendString(json, item.EventId);
            json.Append(",\"event_type\":"); AppendString(json, item.EventType);
            json.Append(",\"fixed_step\":").Append(item.FixedStep.ToString(CultureInfo.InvariantCulture));
            json.Append(",\"participants\":"); AppendStrings(json, item.Participants);
            json.Append(",\"payload\":").Append(item.PayloadJson).Append('}');
        }
        if (snapshot.Events.Count == 0)
        {
            terminalEventId = "terminal:" + snapshot.TerminalFixedStep.ToString(CultureInfo.InvariantCulture);
            json.Append("{\"event_id\":"); AppendString(json, terminalEventId);
            json.Append(",\"event_type\":"); AppendString(json, terminalReason);
            json.Append(",\"fixed_step\":")
                .Append(snapshot.TerminalFixedStep.ToString(CultureInfo.InvariantCulture))
                .Append(",\"participants\":[],\"payload\":{}}");
        }
        json.Append("],\"terminal_evidence\":{\"reason\":"); AppendString(json, terminalReason);
        json.Append(",\"fixed_step\":")
            .Append(snapshot.TerminalFixedStep.ToString(CultureInfo.InvariantCulture))
            .Append(",\"event_id\":"); AppendString(json, terminalEventId);
        return json.Append("}}").ToString();
    }

    private static void AppendEntity(StringBuilder json, PhysicsCaptureV2EntitySnapshot entity)
    {
        json.Append("{\"entity_id\":"); AppendString(json, entity.EntityId);
        json.Append(",\"scenario_object_id\":"); AppendString(json, entity.ScenarioObjectId);
        json.Append(",\"lifecycle\":"); AppendString(json, entity.Lifecycle);
        json.Append(",\"body_present\":").Append(entity.Body == null ? "false" : "true");
        json.Append(",\"body\":");
        if (entity.Body == null)
        {
            json.Append("null");
        }
        else
        {
            json.Append("{\"body_type\":"); AppendString(json, entity.Body.BodyType);
            json.Append(",\"simulated\":").Append(entity.Body.Simulated ? "true" : "false");
            json.Append(",\"gravity_scale\":"); AppendFloat(json, entity.Body.GravityScale);
            json.Append(",\"gravity_applicable\":")
                .Append(entity.Body.GravityApplicable ? "true" : "false");
            json.Append(",\"position\":"); AppendVector(json, entity.Position);
            json.Append(",\"rotation_degrees\":"); AppendFloat(json, entity.RotationDegrees);
            json.Append(",\"velocity\":"); AppendVector(json, entity.Body.LinearVelocity);
            json.Append(",\"angular_velocity_degrees_per_second\":");
            AppendFloat(json, entity.Body.AngularVelocity);
            json.Append('}');
        }
        json.Append(",\"contact_ids\":"); AppendStrings(json, entity.ContactIds);
        json.Append(",\"supported_by_entity_ids\":"); AppendStrings(json, entity.SupportedByEntityIds);
        json.Append(",\"supports_entity_ids\":"); AppendStrings(json, entity.SupportsEntityIds);
        json.Append('}');
    }

    private static void AppendContact(StringBuilder json, PhysicsCaptureV2ContactSnapshot contact)
    {
        json.Append("{\"contact_id\":"); AppendString(json, contact.ContactId);
        json.Append(",\"entity_a_id\":"); AppendString(json, contact.EntityAId);
        json.Append(",\"entity_b_id\":"); AppendString(json, contact.EntityBId);
        json.Append(",\"collider_a_id\":"); AppendString(json, contact.ColliderAId);
        json.Append(",\"collider_b_id\":"); AppendString(json, contact.ColliderBId);
        json.Append(",\"point\":"); AppendVector(json, contact.Point);
        json.Append(",\"normal_a_to_b\":"); AppendVector(json, contact.NormalAToB);
        json.Append(",\"separation\":"); AppendFloat(json, contact.Separation);
        json.Append('}');
    }

    private static void AppendSupport(StringBuilder json, PhysicsCaptureV2SupportSnapshot support)
    {
        json.Append("{\"supporter_entity_id\":"); AppendString(json, support.SupporterEntityId);
        json.Append(",\"supported_entity_id\":"); AppendString(json, support.SupportedEntityId);
        json.Append(",\"contact_ids\":"); AppendStrings(json, support.ContactIds);
        json.Append('}');
    }

    private static void AppendStrings(StringBuilder json, IList<string> values)
    {
        json.Append('[');
        for (int i = 0; i < values.Count; i++)
        {
            if (i > 0) json.Append(','); AppendString(json, values[i]);
        }
        json.Append(']');
    }

    private static void AppendCollider(StringBuilder json, PhysicsCaptureV2ColliderSnapshot collider)
    {
        json.Append("{\"collider_id\":"); AppendString(json, collider.ColliderId);
        json.Append(",\"entity_id\":"); AppendString(json, collider.EntityId);
        json.Append(",\"geometry_source\":\"unity_collider_2d\",\"enabled\":")
            .Append(collider.Enabled ? "true" : "false");
        json.Append(",\"is_trigger\":").Append(collider.IsTrigger ? "true" : "false");
        json.Append(",\"shape\":{\"kind\":"); AppendString(json, collider.Kind);
        if (collider.Kind == "box")
        {
            json.Append(",\"center\":"); AppendVector(json, collider.Center);
            json.Append(",\"size\":"); AppendVector(json, collider.Size);
            json.Append(",\"angle_degrees\":"); AppendFloat(json, collider.AngleDegrees);
        }
        else if (collider.Kind == "circle")
        {
            json.Append(",\"center\":"); AppendVector(json, collider.Center);
            json.Append(",\"radius\":"); AppendFloat(json, collider.Radius);
        }
        else if (collider.Kind == "polygon")
        {
            json.Append(",\"paths\":[");
            for (int pathIndex = 0; pathIndex < collider.Paths.Count; pathIndex++)
            {
                if (pathIndex > 0) json.Append(',');
                AppendPoints(json, collider.Paths[pathIndex]);
            }
            json.Append(']');
        }
        else if (collider.Kind == "edge")
        {
            json.Append(",\"points\":"); AppendPoints(json, collider.Points);
        }
        else if (collider.Kind == "capsule")
        {
            json.Append(",\"center\":"); AppendVector(json, collider.Center);
            json.Append(",\"size\":"); AppendVector(json, collider.Size);
            json.Append(",\"direction\":"); AppendString(json, collider.Direction);
            json.Append(",\"angle_degrees\":"); AppendFloat(json, collider.AngleDegrees);
        }
        json.Append("}}");
    }

    private static void AppendColliderCatalog(StringBuilder json,
        PhysicsCaptureV2ColliderSnapshot collider)
    {
        json.Append("{\"collider_id\":"); AppendString(json, collider.ColliderId);
        json.Append(",\"entity_id\":"); AppendString(json, collider.EntityId);
        json.Append(",\"geometry_source\":\"unity_collider_2d\"}");
    }

    private static void AppendPoints(StringBuilder json,
        System.Collections.Generic.IList<UnityEngine.Vector2> points)
    {
        json.Append('[');
        for (int i = 0; i < points.Count; i++)
        {
            if (i > 0) json.Append(',');
            AppendVector(json, points[i]);
        }
        json.Append(']');
    }

    private static void AppendVector(StringBuilder json, UnityEngine.Vector2 value)
    {
        json.Append('['); AppendFloat(json, value.x); json.Append(',');
        AppendFloat(json, value.y); json.Append(']');
    }

    private static void AppendFloat(StringBuilder json, float value)
    {
        json.Append(value.ToString("R", CultureInfo.InvariantCulture));
    }

    private static void AppendString(StringBuilder json, string value)
    {
        json.Append('\"').Append(value.Replace("\\", "\\\\").Replace("\"", "\\\"")).Append('\"');
    }

    public static byte[] BuildFailureEnvelope(PhysicsCaptureV2EngineFailureCode code, string message)
    {
        byte[] text = Encoding.UTF8.GetBytes(message);
        byte[] payload = new byte[4 + text.Length];
        WriteUInt32(payload, 0, (uint)text.Length);
        Buffer.BlockCopy(text, 0, payload, 4, text.Length);
        return BuildEnvelope(1, (int)code, payload);
    }

    private static byte[] BuildEnvelope(byte flags, int failureCode, byte[] payload)
    {
        int bodyLength = 12 + payload.Length;
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
