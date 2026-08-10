using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

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

public class PhysicalShotRecorder
{
    private readonly PhysicalCaptureLimits limits;
    private readonly Dictionary<string, int> persistence = new Dictionary<string, int>();
    private readonly Dictionary<string, long> lastSeenStep = new Dictionary<string, long>();
    private readonly Dictionary<string, PhysicalRawContact> previousContacts = new Dictionary<string, PhysicalRawContact>();
    private readonly HashSet<string> eventKeys = new HashSet<string>();
    private readonly HashSet<string> collisionKeys = new HashSet<string>();
    private readonly List<PhysicalRawContact> rawContacts = new List<PhysicalRawContact>();
    private readonly List<PhysicalSupportEdge> supportEdges = new List<PhysicalSupportEdge>();
    private readonly List<PhysicalMacroEvent> events = new List<PhysicalMacroEvent>();
    private int estimatedBytes;
    private long currentStep;
    private float currentTime;
    private float shotStartFixedTime;
    private bool hasShotStartFixedTime;
    private bool stabilityInitialized;
    private bool stable;
    private bool terminalRecorded;
    private bool finalized;

    public PhysicalCaptureFailure Failure { get; private set; }
    public IList<PhysicalRawContact> RawContacts { get { return rawContacts.AsReadOnly(); } }
    public IList<PhysicalSupportEdge> SupportEdges { get { return supportEdges.AsReadOnly(); } }
    public IList<PhysicalMacroEvent> Events { get { return events.AsReadOnly(); } }

    public PhysicalShotRecorder(int maxRecords, int maxBytes)
        : this(new PhysicalCaptureLimits(maxRecords, maxBytes, float.PositiveInfinity))
    {
    }

    public PhysicalShotRecorder(PhysicalCaptureLimits limits)
    {
        if (limits == null || limits.MaxRecords <= 0 || limits.MaxBytes <= 0)
        {
            throw new ArgumentException("Capture limits must be positive.");
        }
        this.limits = limits;
    }

    public void RecordContacts(long fixedStep, PhysicalContactInput[] contacts)
    {
        RecordContacts(fixedStep, fixedStep * 0.02f, contacts);
    }

    public void RecordContacts(long fixedStep, IEnumerable<PhysicalContactInput> contacts)
    {
        RecordContacts(fixedStep, fixedStep * 0.02f, contacts == null ? null : contacts.ToArray());
    }

    public void RecordContacts(long fixedStep, float fixedTime, PhysicalContactInput[] contacts)
    {
        if (Failure != null || finalized)
        {
            return;
        }
        currentStep = fixedStep;
        currentTime = fixedTime;
        if (!hasShotStartFixedTime)
        {
            shotStartFixedTime = fixedTime;
            hasShotStartFixedTime = true;
        }
        float elapsedFixedTime = Mathf.Max(0f, fixedTime - shotStartFixedTime);
        if (elapsedFixedTime > limits.TimeoutSeconds)
        {
            Fail(PhysicalCaptureFailureCode.CaptureTimeout, "elapsed fixed-time capture timeout");
            return;
        }

        List<PhysicalRawContact> stepContacts = new List<PhysicalRawContact>();
        foreach (PhysicalContactInput input in contacts ?? new PhysicalContactInput[0])
        {
            if (input == null || input.IsTrigger)
            {
                continue;
            }
            PhysicalRawContact contact = new PhysicalRawContact(input, fixedStep, fixedTime);
            string exactKey = contact.PairKey + ":" + contact.Point.x.ToString("R") + ":" + contact.Point.y.ToString("R");
            if (stepContacts.Any(existing => existing.PairKey + ":" + existing.Point.x.ToString("R") + ":" + existing.Point.y.ToString("R") == exactKey))
            {
                continue;
            }
            stepContacts.Add(contact);
        }
        stepContacts.Sort(CompareContacts);
        for (int index = 0; index < stepContacts.Count; index++)
            stepContacts[index].PointIndex = index;
        if (rawContacts.Count + supportEdges.Count + events.Count + stepContacts.Count > limits.MaxRecords)
        {
            Fail(PhysicalCaptureFailureCode.RecordLimitExceeded, "raw contact record limit exceeded");
            return;
        }
        estimatedBytes += stepContacts.Count * 192;
        if (estimatedBytes > limits.MaxBytes)
        {
            Fail(PhysicalCaptureFailureCode.ByteLimitExceeded, "capture byte limit exceeded");
            return;
        }
        rawContacts.AddRange(stepContacts);
        UpdateSupport(stepContacts, fixedStep);
    }

    public void RecordUnityContacts(long fixedStep, float fixedTime, Collider2D[] colliders, PhysicalEntityRegistry registry = null)
    {
        if (Failure != null || finalized)
            return;
        List<PhysicalContactInput> inputs = new List<PhysicalContactInput>();
        foreach (Collider2D collider in (colliders ?? new Collider2D[0]).OrderBy(c => c == null ? int.MaxValue : c.GetInstanceID()))
        {
            if (collider == null || collider.isTrigger)
            {
                continue;
            }
            ContactPoint2D[] points = new ContactPoint2D[64];
            int count = collider.GetContacts(points);
            for (int index = 0; index < count; index++)
            {
                ContactPoint2D point = points[index];
                Collider2D other = point.otherCollider;
                if (other == null || other.isTrigger || other.GetInstanceID() == collider.GetInstanceID())
                {
                    continue;
                }
                inputs.Add(new PhysicalContactInput(
                    registry == null ? RuntimeEntityId(collider) : registry.RegisterCollider(collider), collider.GetInstanceID(), point.point, point.normal,
                    point.separation, point.relativeVelocity, point.normalImpulse, point.tangentImpulse,
                    registry == null ? RuntimeEntityId(other) : registry.RegisterCollider(other), other.GetInstanceID(), collider.transform.position,
                    other.transform.position, false));
            }
        }
        RecordContacts(fixedStep, fixedTime, inputs.ToArray());
    }

    public static string RuntimeEntityId(Collider2D collider)
    {
        if (collider.attachedRigidbody == null || collider.attachedRigidbody.bodyType == RigidbodyType2D.Static)
            return "world:static:" + collider.GetInstanceID();
        return collider.attachedRigidbody.gameObject.GetInstanceID().ToString();
    }

    public void RecordEvent(long fixedStep, float fixedTime, PhysicalMacroEventKind kind, string subject)
    {
        if (Failure != null || finalized)
        {
            return;
        }
        if (kind == PhysicalMacroEventKind.Death)
            kind = PhysicalMacroEventKind.Destroy;
        if (kind == PhysicalMacroEventKind.LevelClear || kind == PhysicalMacroEventKind.LevelFail)
        {
            if (terminalRecorded)
                return;
            terminalRecorded = true;
        }
        string key = kind + ":" + subject;
        if (!eventKeys.Add(key))
        {
            return;
        }
        AddEvent(fixedStep, fixedTime, kind, subject, ParticipantsFor(kind, subject), DefaultPayload(kind));
    }

    public void RecordCollision(long fixedStep, float fixedTime, string entityA, string entityB)
    {
        if (Failure != null || finalized)
            return;
        throw new ArgumentException("Collision events require contact evidence.");
    }

    public void RecordCollision(long fixedStep, float fixedTime, string entityA, string entityB,
        IEnumerable<string> contactIds, float relativeSpeed)
    {
        if (Failure != null || finalized)
            return;
        string[] evidence = (contactIds ?? new string[0])
            .Where(contactId => !string.IsNullOrEmpty(contactId))
            .Distinct().OrderBy(contactId => contactId, StringComparer.Ordinal).ToArray();
        if (evidence.Length == 0)
            throw new ArgumentException("Collision events require contact evidence.");
        if (float.IsNaN(relativeSpeed) || float.IsInfinity(relativeSpeed) || relativeSpeed < 0f)
            throw new ArgumentException("Collision relative speed must be finite and non-negative.");
        string first = string.CompareOrdinal(entityA, entityB) <= 0 ? entityA : entityB;
        string second = first == entityA ? entityB : entityA;
        string key = fixedStep + ":" + first + ":" + second;
        if (collisionKeys.Add(key))
            AddEvent(fixedStep, fixedTime, PhysicalMacroEventKind.Collision, first + "|" + second,
                new[] { first, second }, new PhysicalMacroEventPayload(contactIds: evidence, relativeSpeed: relativeSpeed));
    }

    public void RecordCollision(long fixedStep, float fixedTime, string entityA, string entityB,
        PhysicalContactInput[] contacts, float relativeSpeed)
    {
        if (Failure != null || finalized)
            return;
        if (float.IsNaN(relativeSpeed) || float.IsInfinity(relativeSpeed) || relativeSpeed < 0f)
            throw new ArgumentException("Collision relative speed must be finite and non-negative.");
        string first = string.CompareOrdinal(entityA, entityB) <= 0 ? entityA : entityB;
        string second = first == entityA ? entityB : entityA;
        PhysicalContactInput[] evidence = (contacts ?? new PhysicalContactInput[0])
            .Where(contact => contact != null
                && (string.Equals(contact.EntityIdA, first, StringComparison.Ordinal)
                    && string.Equals(contact.EntityIdB, second, StringComparison.Ordinal)
                    || string.Equals(contact.EntityIdA, second, StringComparison.Ordinal)
                    && string.Equals(contact.EntityIdB, first, StringComparison.Ordinal)))
            .ToArray();
        if (evidence.Length == 0)
            throw new ArgumentException("Collision events require contact evidence.");
        string key = fixedStep + ":" + first + ":" + second;
        if (collisionKeys.Contains(key))
            return;
        string[] contactIds = rawContacts
            .Where(contact => contact.FixedStep == fixedStep
                && contact.EntityIdA == first && contact.EntityIdB == second)
            .Select(contact => contact.ContactId).Distinct().OrderBy(contactId => contactId, StringComparer.Ordinal).ToArray();
        if (contactIds.Length == 0)
        {
            RecordContacts(fixedStep, fixedTime, evidence);
            contactIds = rawContacts
                .Where(contact => contact.FixedStep == fixedStep
                    && contact.EntityIdA == first && contact.EntityIdB == second)
                .Select(contact => contact.ContactId).Distinct().OrderBy(contactId => contactId, StringComparer.Ordinal).ToArray();
        }
        RecordCollision(fixedStep, fixedTime, first, second, contactIds, relativeSpeed);
    }

    public void RecordCollision(string entityA, string entityB, long fixedStep)
    {
        RecordCollision(fixedStep, fixedStep * 0.02f, entityA, entityB);
    }

    public void RecordLaunch(string subject, long fixedStep) { RecordEvent(fixedStep, fixedStep * 0.02f, PhysicalMacroEventKind.Launch, subject); }
    public void RecordLaunch(string subject, long fixedStep, Vector2 launchVelocity) { AddOneShotEvent(fixedStep, PhysicalMacroEventKind.Launch, subject, new PhysicalMacroEventPayload(launchVelocity: launchVelocity)); }
    public void RecordDestroyed(string subject, long fixedStep) { RecordEvent(fixedStep, fixedStep * 0.02f, PhysicalMacroEventKind.Destroy, subject); }
    public void RecordDestroyed(string subject, long fixedStep, string reason) { AddOneShotEvent(fixedStep, PhysicalMacroEventKind.Destroy, subject, new PhysicalMacroEventPayload(reason: reason)); }
    public void RecordDeath(string subject, long fixedStep) { RecordEvent(fixedStep, fixedStep * 0.02f, PhysicalMacroEventKind.Death, subject); }
    public void RecordPigRemoved(string subject, long fixedStep) { RecordEvent(fixedStep, fixedStep * 0.02f, PhysicalMacroEventKind.PigRemoved, subject); }
    public void RecordPigRemoved(string subject, long fixedStep, string reason) { AddOneShotEvent(fixedStep, PhysicalMacroEventKind.PigRemoved, subject, new PhysicalMacroEventPayload(reason: reason)); }
    public void RecordTntExplosion(string subject, long fixedStep) { RecordEvent(fixedStep, fixedStep * 0.02f, PhysicalMacroEventKind.TntExplosion, subject); }
    public void RecordTntExplosion(string subject, long fixedStep, float radiusUnityUnits) { AddOneShotEvent(fixedStep, PhysicalMacroEventKind.TntExplosion, subject, new PhysicalMacroEventPayload(radiusUnityUnits: radiusUnityUnits)); }
    public void RecordBirdExhaustion(long fixedStep) { RecordEvent(fixedStep, fixedStep * 0.02f, PhysicalMacroEventKind.BirdExhaustion, "level"); }
    public void RecordLevelClear(long fixedStep) { RecordEvent(fixedStep, fixedStep * 0.02f, PhysicalMacroEventKind.LevelClear, "level"); }
    public void RecordLevelClear(long fixedStep, int score) { AddTerminalEvent(fixedStep, PhysicalMacroEventKind.LevelClear, new PhysicalMacroEventPayload(score: score)); }
    public void RecordLevelFail(long fixedStep) { RecordEvent(fixedStep, fixedStep * 0.02f, PhysicalMacroEventKind.LevelFail, "level"); }
    public void RecordLevelFail(long fixedStep, string reason) { AddTerminalEvent(fixedStep, PhysicalMacroEventKind.LevelFail, new PhysicalMacroEventPayload(reason: reason)); }

    public void RecordStability(long fixedStep, bool stable)
    {
        if (Failure != null || finalized)
            return;
        if (stabilityInitialized && this.stable == stable)
            return;
        stabilityInitialized = true;
        this.stable = stable;
        PhysicalMacroEventKind kind = stable ? PhysicalMacroEventKind.StabilityEnter : PhysicalMacroEventKind.StabilityExit;
        if (Failure == null)
            AddEvent(fixedStep, fixedStep * 0.02f, kind, "level", new string[0],
                new PhysicalMacroEventPayload(debounceFixedSteps: 2));
    }

    public void FailTimeout(string message) { Fail(PhysicalCaptureFailureCode.CaptureTimeout, message); }

    public PhysicalCaptureResult FinalizeShot(bool terminal)
    {
        if (finalized)
            return new PhysicalCaptureResult(Failure);
        if (Failure == null && !terminal)
        {
            Fail(PhysicalCaptureFailureCode.TruncatedFinalization, "shot finalized before terminal event");
        }
        finalized = true;
        return new PhysicalCaptureResult(Failure);
    }

    public PhysicalShotRecorderSnapshot CreateFinalizedSnapshot()
    {
        return finalized && Failure == null
            ? new PhysicalShotRecorderSnapshot(rawContacts, supportEdges, events)
            : null;
    }

    public bool TryFinalize(bool terminal)
    {
        return FinalizeShot(terminal).Failure == null;
    }

    private void UpdateSupport(List<PhysicalRawContact> stepContacts, long fixedStep)
    {
        HashSet<string> seen = new HashSet<string>();
        foreach (PhysicalRawContact contact in stepContacts)
        {
            seen.Add(contact.PairKey);
            int count = lastSeenStep.ContainsKey(contact.PairKey) && lastSeenStep[contact.PairKey] == fixedStep - 1
                ? persistence[contact.PairKey] + 1 : 1;
            persistence[contact.PairKey] = count;
            lastSeenStep[contact.PairKey] = fixedStep;
            supportEdges.RemoveAll(edge => edge.PairKey == contact.PairKey);
            PhysicalRawContact prior;
            bool consecutive = previousContacts.TryGetValue(contact.PairKey, out prior)
                && prior.FixedStep == fixedStep - 1;
            if (consecutive && Mathf.Abs(contact.Normal.y) >= 0.5f
                && Mathf.Abs(prior.Normal.y) >= 0.5f
                && (contact.CenterB.y - contact.CenterA.y >= 0.0001f
                    && prior.CenterB.y - prior.CenterA.y >= 0.0001f
                    || contact.CenterB.y - contact.CenterA.y <= -0.0001f
                    && prior.CenterB.y - prior.CenterA.y <= -0.0001f))
            {
                if (!CanAddRecord(128))
                    return;
                bool aBelowB = contact.CenterB.y > contact.CenterA.y;
                string supporter = aBelowB ? contact.EntityIdA : contact.EntityIdB;
                string supported = aBelowB ? contact.EntityIdB : contact.EntityIdA;
                supportEdges.Add(new PhysicalSupportEdge(
                    supporter, supported, contact.PairKey,
                    prior.ContactId, prior.FixedStep, contact.ContactId, contact.FixedStep));
                estimatedBytes += 128;
            }
            previousContacts[contact.PairKey] = contact;
        }
        supportEdges.RemoveAll(edge => !seen.Contains(edge.PairKey));
    }

    private void AddEvent(long fixedStep, float fixedTime, PhysicalMacroEventKind kind, string subject,
        IEnumerable<string> participants, PhysicalMacroEventPayload payload)
    {
        if (finalized)
            return;
        if (!CanAddRecord(128))
            return;
        events.Add(new PhysicalMacroEvent(0, fixedStep, fixedTime, kind, subject, participants, payload));
        estimatedBytes += 128;
        events.Sort(CompareEvents);
        for (int index = 0; index < events.Count; index++)
            events[index].Sequence = index + 1L;
    }

    private void AddOneShotEvent(long fixedStep, PhysicalMacroEventKind kind, string subject, PhysicalMacroEventPayload payload)
    {
        if (finalized)
            return;
        string key = kind + ":" + subject;
        if (Failure == null && eventKeys.Add(key))
            AddEvent(fixedStep, fixedStep * 0.02f, kind, subject, ParticipantsFor(kind, subject), payload);
    }

    private void AddTerminalEvent(long fixedStep, PhysicalMacroEventKind kind, PhysicalMacroEventPayload payload)
    {
        if (Failure != null || finalized || terminalRecorded)
            return;
        terminalRecorded = true;
        eventKeys.Add(kind + ":level");
        AddEvent(fixedStep, fixedStep * 0.02f, kind, "level", new string[0], payload);
    }

    private static IEnumerable<string> ParticipantsFor(PhysicalMacroEventKind kind, string subject)
    {
        if (kind == PhysicalMacroEventKind.BirdExhaustion || kind == PhysicalMacroEventKind.StabilityEnter
            || kind == PhysicalMacroEventKind.StabilityExit || kind == PhysicalMacroEventKind.LevelClear
            || kind == PhysicalMacroEventKind.LevelFail)
            return new string[0];
        return (subject ?? string.Empty).Split(new[] { '|' }, StringSplitOptions.RemoveEmptyEntries);
    }

    private static PhysicalMacroEventPayload DefaultPayload(PhysicalMacroEventKind kind)
    {
        switch (kind)
        {
            case PhysicalMacroEventKind.Launch: return new PhysicalMacroEventPayload(launchVelocity: Vector2.zero);
            case PhysicalMacroEventKind.Destroy: return new PhysicalMacroEventPayload(reason: "destroyed");
            case PhysicalMacroEventKind.PigRemoved: return new PhysicalMacroEventPayload(reason: "destroyed");
            case PhysicalMacroEventKind.TntExplosion: return new PhysicalMacroEventPayload(radiusUnityUnits: 0f);
            case PhysicalMacroEventKind.BirdExhaustion: return new PhysicalMacroEventPayload(birdsRemaining: 0);
            case PhysicalMacroEventKind.LevelClear: return new PhysicalMacroEventPayload(score: 0);
            case PhysicalMacroEventKind.LevelFail: return new PhysicalMacroEventPayload(reason: "failed");
            default: return new PhysicalMacroEventPayload();
        }
    }

    private void Fail(PhysicalCaptureFailureCode code, string message)
    {
        if (!finalized && Failure == null)
        {
            Failure = new PhysicalCaptureFailure(code, message);
        }
    }

    private static int CompareContacts(PhysicalRawContact left, PhysicalRawContact right)
    {
        int result = string.CompareOrdinal(left.EntityIdA, right.EntityIdA);
        if (result != 0) return result;
        result = string.CompareOrdinal(left.EntityIdB, right.EntityIdB);
        if (result != 0) return result;
        result = left.ColliderIdA.CompareTo(right.ColliderIdA);
        if (result != 0) return result;
        result = left.ColliderIdB.CompareTo(right.ColliderIdB);
        if (result != 0) return result;
        result = left.Point.x.CompareTo(right.Point.x);
        return result != 0 ? result : left.Point.y.CompareTo(right.Point.y);
    }

    private bool CanAddRecord(int bytes)
    {
        if (rawContacts.Count + supportEdges.Count + events.Count >= limits.MaxRecords)
        {
            Fail(PhysicalCaptureFailureCode.RecordLimitExceeded, "record limit exceeded");
            return false;
        }
        if (estimatedBytes + bytes > limits.MaxBytes)
        {
            Fail(PhysicalCaptureFailureCode.ByteLimitExceeded, "byte limit exceeded");
            return false;
        }
        return true;
    }

    private static int EventRank(PhysicalMacroEventKind kind)
    {
        switch (kind)
        {
            case PhysicalMacroEventKind.Launch: return 0;
            case PhysicalMacroEventKind.Collision: return 1;
            case PhysicalMacroEventKind.TntExplosion: return 2;
            case PhysicalMacroEventKind.Destroy: return 3;
            case PhysicalMacroEventKind.PigRemoved: return 4;
            case PhysicalMacroEventKind.BirdExhaustion: return 5;
            case PhysicalMacroEventKind.StabilityEnter: return 6;
            case PhysicalMacroEventKind.StabilityExit: return 7;
            case PhysicalMacroEventKind.LevelClear: return 8;
            case PhysicalMacroEventKind.LevelFail: return 9;
            default: return 99;
        }
    }

    private static int CompareEvents(PhysicalMacroEvent left, PhysicalMacroEvent right)
    {
        int result = left.FixedStep.CompareTo(right.FixedStep);
        if (result != 0) return result;
        result = left.FixedTime.CompareTo(right.FixedTime);
        if (result != 0) return result;
        result = EventRank(left.Kind).CompareTo(EventRank(right.Kind));
        if (result != 0) return result;
        return string.CompareOrdinal(left.Subject, right.Subject);
    }
}

public sealed class PhysicsShotRecorder : PhysicalShotRecorder
{
    public PhysicsShotRecorder(int maxRecords, int maxBytes) : base(maxRecords, maxBytes) { }
    public PhysicsShotRecorder(PhysicalCaptureLimits limits) : base(limits) { }
}
