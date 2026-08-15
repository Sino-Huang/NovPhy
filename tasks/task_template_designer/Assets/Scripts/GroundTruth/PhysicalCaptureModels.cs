using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

// Data-only capture model types, split out of PhysicsShotRecorder.cs so that no
// file in this directory exceeds the 800-line ceiling. Nothing here owns
// recorder state or behavior: the recorder and the serializer both consume
// these shapes unchanged.

public enum PhysicalCaptureFailureCode
{
    RecordLimitExceeded,
    ByteLimitExceeded,
    CaptureTimeout,
    TruncatedFinalization,
    EnvelopeLimitExceeded
}

public static class PhysicsCaptureFailureCode
{
    public const PhysicalCaptureFailureCode RecordLimitExceeded = PhysicalCaptureFailureCode.RecordLimitExceeded;
    public const PhysicalCaptureFailureCode ByteLimitExceeded = PhysicalCaptureFailureCode.ByteLimitExceeded;
    public const PhysicalCaptureFailureCode CaptureTimeout = PhysicalCaptureFailureCode.CaptureTimeout;
    public const PhysicalCaptureFailureCode TruncatedFinalization = PhysicalCaptureFailureCode.TruncatedFinalization;
}

public sealed class PhysicalCaptureFailure
{
    public PhysicalCaptureFailureCode Code { get; private set; }
    public string CodeName
    {
        get
        {
            switch (Code)
            {
                case PhysicalCaptureFailureCode.RecordLimitExceeded: return "record_limit_exceeded";
                case PhysicalCaptureFailureCode.ByteLimitExceeded: return "byte_limit_exceeded";
                case PhysicalCaptureFailureCode.CaptureTimeout: return "capture_timeout";
                case PhysicalCaptureFailureCode.EnvelopeLimitExceeded: return "envelope_limit_exceeded";
                default: return "truncated_finalization";
            }
        }
    }
    public string Message { get; private set; }

    public PhysicalCaptureFailure(PhysicalCaptureFailureCode code, string message)
    {
        Code = code;
        Message = message;
    }
}

public sealed class PhysicalCaptureLimits
{
    public int MaxRecords { get; private set; }
    public int MaxBytes { get; private set; }
    public float TimeoutSeconds { get; private set; }

    public PhysicalCaptureLimits(int maxRecords, int maxBytes, float timeoutSeconds)
    {
        MaxRecords = maxRecords;
        MaxBytes = maxBytes;
        TimeoutSeconds = timeoutSeconds;
    }
}

public class PhysicalContactInput
{
    public string EntityIdA { get; private set; }
    public int ColliderIdA { get; private set; }
    public Vector2 Point { get; private set; }
    public Vector2 Normal { get; private set; }
    public float Separation { get; private set; }
    public Vector2 RelativeVelocity { get; private set; }
    public float NormalImpulse { get; private set; }
    public float TangentImpulse { get; private set; }
    public string EntityIdB { get; private set; }
    public int ColliderIdB { get; private set; }
    public Vector2 CenterA { get; private set; }
    public Vector2 CenterB { get; private set; }
    public bool IsTrigger { get; private set; }

    public PhysicalContactInput(
        string entityIdA, int colliderIdA, Vector2 point, Vector2 normal, float separation,
        Vector2 relativeVelocity, float normalImpulse, string entityIdB, int colliderIdB,
        Vector2 centerB, bool isTrigger)
        : this(entityIdA, colliderIdA, point, normal, separation, relativeVelocity, normalImpulse,
            0f, entityIdB, colliderIdB, Vector2.zero, centerB, isTrigger)
    {
    }

    public PhysicalContactInput(
        string entityIdA, int colliderIdA, Vector2 point, Vector2 normal, float separation,
        Vector2 relativeVelocity, float normalImpulse, string entityIdB, int colliderIdB,
        Vector2 centerA, Vector2 centerB, bool isTrigger)
        : this(entityIdA, colliderIdA, point, normal, separation, relativeVelocity, normalImpulse,
            0f, entityIdB, colliderIdB, centerA, centerB, isTrigger)
    {
    }

    public PhysicalContactInput(
        string entityIdA, int colliderIdA, Vector2 point, Vector2 normal, float separation,
        Vector2 relativeVelocity, float normalImpulse, float tangentImpulse, string entityIdB, int colliderIdB,
        Vector2 centerA, Vector2 centerB, bool isTrigger)
    {
        EntityIdA = entityIdA;
        ColliderIdA = colliderIdA;
        Point = point;
        Normal = normal;
        Separation = separation;
        RelativeVelocity = relativeVelocity;
        NormalImpulse = normalImpulse;
        TangentImpulse = tangentImpulse;
        EntityIdB = entityIdB;
        ColliderIdB = colliderIdB;
        CenterA = centerA;
        CenterB = centerB;
        IsTrigger = isTrigger;
    }

    public PhysicalContactInput(
        Collider2D colliderA, Collider2D colliderB, Vector2 point, Vector2 normal, float separation,
        Vector2 relativeVelocity, float normalImpulse)
        : this(colliderA.GetInstanceID().ToString(), colliderA.GetInstanceID(), point, normal, separation,
            relativeVelocity, normalImpulse, colliderB.GetInstanceID().ToString(), colliderB.GetInstanceID(),
            colliderA.transform.position, colliderB.transform.position,
            colliderA.isTrigger || colliderB.isTrigger)
    {
    }

}

public sealed class PhysicsContactInput : PhysicalContactInput
{
    public PhysicsContactInput(
        Collider2D colliderA, Collider2D colliderB, Vector2 point, Vector2 normal,
        float separation, Vector2 relativeVelocity, float normalImpulse)
        : base(colliderA, colliderB, point, normal, separation, relativeVelocity, normalImpulse)
    {
    }
}

public sealed class PhysicalRawContact
{
    public string EntityIdA { get; private set; }
    public string EntityIdB { get; private set; }
    public int ColliderIdA { get; private set; }
    public int ColliderIdB { get; private set; }
    public Vector2 Point { get; private set; }
    public Vector2 Normal { get; private set; }
    public float Separation { get; private set; }
    public Vector2 RelativeVelocity { get; private set; }
    public float NormalImpulse { get; private set; }
    public float TangentImpulse { get; private set; }
    public int PointIndex { get; internal set; }
    public Vector2 CenterA { get; private set; }
    public Vector2 CenterB { get; private set; }
    public bool IsTrigger { get; private set; }
    public long FixedStep { get; private set; }
    public float FixedTime { get; private set; }

    public PhysicalRawContact(PhysicalContactInput input, long fixedStep, float fixedTime)
    {
        bool swap = string.CompareOrdinal(input.EntityIdA, input.EntityIdB) > 0
            || (string.Equals(input.EntityIdA, input.EntityIdB, StringComparison.Ordinal)
                && input.ColliderIdA > input.ColliderIdB);
        EntityIdA = swap ? input.EntityIdB : input.EntityIdA;
        EntityIdB = swap ? input.EntityIdA : input.EntityIdB;
        ColliderIdA = swap ? input.ColliderIdB : input.ColliderIdA;
        ColliderIdB = swap ? input.ColliderIdA : input.ColliderIdB;
        Point = input.Point;
        Normal = swap ? -input.Normal : input.Normal;
        Separation = input.Separation;
        RelativeVelocity = swap ? -input.RelativeVelocity : input.RelativeVelocity;
        NormalImpulse = input.NormalImpulse;
        TangentImpulse = input.TangentImpulse;
        CenterA = swap ? input.CenterB : input.CenterA;
        CenterB = swap ? input.CenterA : input.CenterB;
        IsTrigger = input.IsTrigger;
        FixedStep = fixedStep;
        FixedTime = fixedTime;
    }

    public string PairKey
    {
        get { return EntityIdA + ":" + ColliderIdA + "|" + EntityIdB + ":" + ColliderIdB; }
    }

    public string ContactId
    {
        get { return "contact:" + FixedStep + ":" + PairKey + ":" + PointIndex; }
    }
}

public sealed class PhysicalSupportEdge
{
    public string SupporterEntityId { get; private set; }
    public string SupportedEntityId { get; private set; }
    public string PairKey { get; private set; }
    public string ContactIdA { get; private set; }
    public string ContactIdB { get; private set; }
    public long FixedStepA { get; private set; }
    public long FixedStepB { get; private set; }

    public PhysicalSupportEdge(string supporterEntityId, string supportedEntityId, string pairKey,
        string contactIdA, long fixedStepA, string contactIdB, long fixedStepB)
    {
        SupporterEntityId = supporterEntityId;
        SupportedEntityId = supportedEntityId;
        PairKey = pairKey;
        ContactIdA = contactIdA;
        FixedStepA = fixedStepA;
        ContactIdB = contactIdB;
        FixedStepB = fixedStepB;
    }
}

public enum PhysicalMacroEventKind
{
    Launch,
    Collision,
    Death,
    Destroy,
    PigRemoved,
    TntExplosion,
    BirdExhaustion,
    LevelClear,
    LevelFail,
    StabilityEnter,
    StabilityExit
}

public sealed class PhysicalMacroEvent
{
    public long Sequence { get; internal set; }
    public long FixedStep { get; private set; }
    public float FixedTime { get; private set; }
    public PhysicalMacroEventKind Kind { get; private set; }
    public string Taxonomy
    {
        get
        {
            switch (Kind)
            {
                case PhysicalMacroEventKind.Launch: return "bird_launched";
                case PhysicalMacroEventKind.Death: return "entity_destroyed";
                case PhysicalMacroEventKind.Destroy: return "entity_destroyed";
                case PhysicalMacroEventKind.TntExplosion: return "explosion";
                case PhysicalMacroEventKind.PigRemoved: return "pig_removed";
                case PhysicalMacroEventKind.BirdExhaustion: return "bird_exhausted";
                case PhysicalMacroEventKind.LevelClear: return "level_cleared";
                case PhysicalMacroEventKind.LevelFail: return "level_failed";
                case PhysicalMacroEventKind.StabilityEnter: return "stable_entered";
                case PhysicalMacroEventKind.StabilityExit: return "stable_exited";
                default: return Kind.ToString().ToLowerInvariant();
            }
        }
    }
    public string Subject { get; private set; }
    public IList<string> Participants { get; private set; }
    public PhysicalMacroEventPayload Payload { get; private set; }

    public PhysicalMacroEvent(long sequence, long fixedStep, float fixedTime, PhysicalMacroEventKind kind,
        string subject, IEnumerable<string> participants, PhysicalMacroEventPayload payload)
    {
        Sequence = sequence;
        FixedStep = fixedStep;
        FixedTime = fixedTime;
        Kind = kind;
        Subject = subject;
        Participants = (participants ?? new string[0]).Where(item => !string.IsNullOrEmpty(item))
            .Distinct().OrderBy(item => item, StringComparer.Ordinal).ToList().AsReadOnly();
        Payload = payload;
    }
}

public sealed class PhysicalMacroEventPayload
{
    public Vector2? LaunchVelocity { get; private set; }
    public IList<string> ContactIds { get; private set; }
    public float? RelativeSpeed { get; private set; }
    public float? RadiusUnityUnits { get; private set; }
    public string Reason { get; private set; }
    public int? BirdsRemaining { get; private set; }
    public int? DebounceFixedSteps { get; private set; }
    public int? Score { get; private set; }

    public PhysicalMacroEventPayload(Vector2? launchVelocity = null, IEnumerable<string> contactIds = null,
        float? relativeSpeed = null, float? radiusUnityUnits = null, string reason = null,
        int? birdsRemaining = null, int? debounceFixedSteps = null, int? score = null)
    {
        LaunchVelocity = launchVelocity;
        ContactIds = (contactIds ?? new string[0]).Distinct().OrderBy(item => item, StringComparer.Ordinal).ToList().AsReadOnly();
        RelativeSpeed = relativeSpeed;
        RadiusUnityUnits = radiusUnityUnits;
        Reason = reason;
        BirdsRemaining = birdsRemaining;
        DebounceFixedSteps = debounceFixedSteps;
        Score = score;
    }
}

public sealed class PhysicalCaptureResult
{
    public PhysicalCaptureFailure Failure { get; private set; }
    public bool IsValid { get { return Failure == null; } }

    public PhysicalCaptureResult(PhysicalCaptureFailure failure)
    {
        Failure = failure;
    }
}

public sealed class PhysicalShotRecorderSnapshot
{
    public IList<PhysicalRawContact> RawContacts { get; private set; }
    public IList<PhysicalSupportEdge> SupportEdges { get; private set; }
    public IList<PhysicalMacroEvent> Events { get; private set; }

    internal PhysicalShotRecorderSnapshot(IList<PhysicalRawContact> rawContacts,
        IList<PhysicalSupportEdge> supportEdges, IList<PhysicalMacroEvent> events)
    {
        RawContacts = new List<PhysicalRawContact>(rawContacts).AsReadOnly();
        SupportEdges = new List<PhysicalSupportEdge>(supportEdges).AsReadOnly();
        Events = new List<PhysicalMacroEvent>(events).AsReadOnly();
    }
}
