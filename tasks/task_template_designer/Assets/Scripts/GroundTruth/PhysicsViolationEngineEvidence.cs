using System.Collections.Generic;
using UnityEngine;

// Engine-owned facts used by the violation-label boundary.  These types are
// intentionally data-only: callers cannot declare completeness or gravity.
public sealed class PhysicalEvidenceSupport
{
    public string SupportId { get; private set; }
    public string SupporterId { get; private set; }
    public string ContactIdA { get; private set; }
    public string ContactIdB { get; private set; }
    public long FixedStepA { get; private set; }
    public long FixedStepB { get; private set; }

    public PhysicalEvidenceSupport(PhysicalSupportEdge edge)
    {
        SupportId = "support:" + edge.SupporterEntityId + "->" + edge.SupportedEntityId;
        SupporterId = edge.SupporterEntityId;
        ContactIdA = edge.ContactIdA;
        ContactIdB = edge.ContactIdB;
        FixedStepA = edge.FixedStepA;
        FixedStepB = edge.FixedStepB;
    }
}

public sealed class PhysicalEvidenceEntity
{
    public string EntityId { get; private set; }
    public bool Present { get; private set; }
    public Vector2? WorldPosition { get; private set; }
    public string BodyType { get; private set; }
    public bool? Simulated { get; private set; }
    public float? GravityScale { get; private set; }
    public IList<PhysicalEvidenceSupport> Supports { get; private set; }

    public PhysicalEvidenceEntity(string entityId, Rigidbody2D body, IList<PhysicalEvidenceSupport> supports)
    {
        EntityId = entityId;
        Present = body != null;
        WorldPosition = body == null ? (Vector2?)null : body.position;
        BodyType = body == null ? null : BodyTypeName(body.bodyType);
        Simulated = body == null ? (bool?)null : body.simulated;
        GravityScale = body == null ? (float?)null : body.gravityScale;
        Supports = new List<PhysicalEvidenceSupport>(supports ?? new PhysicalEvidenceSupport[0]).AsReadOnly();
    }

    private static string BodyTypeName(RigidbodyType2D value)
    {
        switch (value)
        {
            case RigidbodyType2D.Dynamic: return "dynamic";
            case RigidbodyType2D.Kinematic: return "kinematic";
            default: return "static";
        }
    }
}

public sealed class PhysicalEvidenceTraceSample
{
    public long FixedStep { get; private set; }
    public Vector2 Physics2DGravity { get; private set; }
    public IList<PhysicalEvidenceEntity> Entities { get; private set; }

    public PhysicalEvidenceTraceSample(long fixedStep, Vector2 gravity, IList<PhysicalEvidenceEntity> entities)
    {
        FixedStep = fixedStep;
        Physics2DGravity = gravity;
        Entities = new List<PhysicalEvidenceEntity>(entities).AsReadOnly();
    }
}

public sealed class PhysicalViolationEngineEvidenceSnapshot
{
    public const string SchemaVersion = "physics_violation_engine_evidence_v1";
    public const int MaxTraceFixedSteps = 8;
    public const int MaxEntitiesPerStep = 128;

    public string ShotId { get; private set; }
    public long? FirstFixedStep { get; private set; }
    public long? LastFixedStep { get; private set; }
    public int SampleCount { get; private set; }
    public bool Complete { get; private set; }
    public string IncompleteReason { get; private set; }
    public bool MinimumObserved { get; private set; }
    public float? MinimumSeparation { get; private set; }
    public string MinimumContactId { get; private set; }
    public long? MinimumFixedStep { get; private set; }
    public bool TraceTruncated { get; private set; }
    public IList<PhysicalEvidenceTraceSample> Trace { get; private set; }

    public PhysicalViolationEngineEvidenceSnapshot(
        string shotId, long? firstFixedStep, long? lastFixedStep, int sampleCount,
        string incompleteReason, bool minimumObserved, float? minimumSeparation,
        string minimumContactId, long? minimumFixedStep, bool traceTruncated,
        IList<PhysicalEvidenceTraceSample> trace)
    {
        ShotId = shotId;
        FirstFixedStep = firstFixedStep;
        LastFixedStep = lastFixedStep;
        SampleCount = sampleCount;
        IncompleteReason = sampleCount == 0 && incompleteReason == null
            ? "no_fixed_step_samples" : incompleteReason;
        Complete = IncompleteReason == null;
        MinimumObserved = minimumObserved;
        MinimumSeparation = minimumSeparation;
        MinimumContactId = minimumContactId;
        MinimumFixedStep = minimumFixedStep;
        TraceTruncated = traceTruncated;
        Trace = new List<PhysicalEvidenceTraceSample>(trace).AsReadOnly();
    }
}
