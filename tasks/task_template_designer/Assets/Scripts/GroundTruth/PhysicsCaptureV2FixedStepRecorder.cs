using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Globalization;
using UnityEngine;

public sealed class PhysicsCaptureV2FixedStepSample
{
    public long FixedStep { get; private set; }
    public ReadOnlyCollection<PhysicsCaptureV2EntitySnapshot> Entities { get; private set; }
    public ReadOnlyCollection<PhysicsCaptureV2ColliderSnapshot> Colliders { get; private set; }
    public ReadOnlyCollection<PhysicsCaptureV2ContactSnapshot> Contacts { get; private set; }
    public ReadOnlyCollection<PhysicsCaptureV2SupportSnapshot> Supports { get; private set; }
    public Vector2 WorldGravity { get; private set; }

    public PhysicsCaptureV2FixedStepSample(long fixedStep,
        IList<PhysicsCaptureV2EntitySnapshot> entities,
        IList<PhysicsCaptureV2ColliderSnapshot> colliders)
        : this(fixedStep, entities, colliders, new PhysicsCaptureV2ContactSnapshot[0],
            new PhysicsCaptureV2SupportSnapshot[0])
    {
    }

    public PhysicsCaptureV2FixedStepSample(long fixedStep,
        IList<PhysicsCaptureV2EntitySnapshot> entities,
        IList<PhysicsCaptureV2ColliderSnapshot> colliders,
        IList<PhysicsCaptureV2ContactSnapshot> contacts,
        IList<PhysicsCaptureV2SupportSnapshot> supports)
    {
        FixedStep = fixedStep;
        WorldGravity = Physics2D.gravity;
        Entities = new List<PhysicsCaptureV2EntitySnapshot>(entities).AsReadOnly();
        Colliders = new List<PhysicsCaptureV2ColliderSnapshot>(colliders).AsReadOnly();
        Contacts = new List<PhysicsCaptureV2ContactSnapshot>(contacts).AsReadOnly();
        Supports = new List<PhysicsCaptureV2SupportSnapshot>(supports).AsReadOnly();
    }
}

public sealed class PhysicsCaptureV2FrameRecord
{
    public long FixedStep { get; private set; }
    public bool ForcedTerminal { get; private set; }

    public PhysicsCaptureV2FrameRecord(long fixedStep, bool forcedTerminal)
    {
        FixedStep = fixedStep;
        ForcedTerminal = forcedTerminal;
    }
}

public sealed class PhysicsCaptureV2EngineSnapshot
{
    public string CaptureId { get; private set; }
    public string ShotId { get; private set; }
    public int ConfiguredFixedStepCaptureStride { get; private set; }
    public long PreInterventionFixedStep { get; private set; }
    public long TerminalFixedStep { get; private set; }
    public ReadOnlyCollection<PhysicsCaptureV2FixedStepSample> FixedStepSamples { get; private set; }
    public ReadOnlyCollection<PhysicsCaptureV2FrameRecord> FrameRecords { get; private set; }
    public ReadOnlyCollection<PhysicsCaptureV2EventSnapshot> Events { get; private set; }
    public string TerminalReason { get; private set; }
    public string TerminalEventId { get; private set; }

    public PhysicsCaptureV2EngineSnapshot(int stride, long preInterventionFixedStep,
        long terminalFixedStep, IList<PhysicsCaptureV2FixedStepSample> fixedStepSamples,
        IList<PhysicsCaptureV2FrameRecord> frameRecords)
        : this("capture-v2-test", "shot-v2-test", stride, preInterventionFixedStep,
            terminalFixedStep, fixedStepSamples, frameRecords,
            new PhysicsCaptureV2EventSnapshot[0], "terminal", null)
    {
    }

    public PhysicsCaptureV2EngineSnapshot(string captureId, string shotId, int stride,
        long preInterventionFixedStep, long terminalFixedStep,
        IList<PhysicsCaptureV2FixedStepSample> fixedStepSamples,
        IList<PhysicsCaptureV2FrameRecord> frameRecords,
        IList<PhysicsCaptureV2EventSnapshot> events, string terminalReason, string terminalEventId)
    {
        CaptureId = captureId;
        ShotId = shotId;
        ConfiguredFixedStepCaptureStride = stride;
        PreInterventionFixedStep = preInterventionFixedStep;
        TerminalFixedStep = terminalFixedStep;
        FixedStepSamples = new List<PhysicsCaptureV2FixedStepSample>(fixedStepSamples).AsReadOnly();
        FrameRecords = new List<PhysicsCaptureV2FrameRecord>(frameRecords).AsReadOnly();
        Events = new List<PhysicsCaptureV2EventSnapshot>(events).AsReadOnly();
        TerminalReason = terminalReason;
        TerminalEventId = terminalEventId;
    }
}

public sealed class PhysicsCaptureV2FixedStepRecorder : MonoBehaviour
{
    private readonly List<PhysicsCaptureV2FixedStepSample> fixedStepSamples =
        new List<PhysicsCaptureV2FixedStepSample>();
    private readonly List<PhysicsCaptureV2FrameRecord> frameRecords =
        new List<PhysicsCaptureV2FrameRecord>();
    private readonly List<PhysicsCaptureV2EventSnapshot> events =
        new List<PhysicsCaptureV2EventSnapshot>();
    private readonly Dictionary<string, PhysicsCaptureV2EntitySnapshot> retainedEntities =
        new Dictionary<string, PhysicsCaptureV2EntitySnapshot>(StringComparer.Ordinal);
    private readonly Dictionary<string, PhysicsCaptureV2ColliderSnapshot> retainedColliders =
        new Dictionary<string, PhysicsCaptureV2ColliderSnapshot>(StringComparer.Ordinal);
    private int stride;
    private long preInterventionFixedStep;
    private long lastFixedStep;
    private long terminalFixedStep;
    private bool recording;
    private bool finalized;
    private string captureId;
    private string shotId;
    private string terminalReason;
    private string terminalEventId;
    private PhysicsCaptureV2CaptureLimits limits = PhysicsCaptureV2CaptureLimits.Default;

    public static PhysicsCaptureV2FixedStepRecorder Active { get; private set; }
    public PhysicsCaptureV2RecorderFailure Failure { get; private set; }
    public bool IsFinalized { get { return finalized; } }
    public long LastFixedStep { get { return lastFixedStep; } }

    private void Awake()
    {
        Active = this;
    }

    public void BeginPreIntervention(long fixedStep)
    {
        BeginPreIntervention(fixedStep, new GameObject[0]);
    }

    public void BeginPreIntervention(long fixedStep, GameObject[] causalObjects)
    {
        BeginPreIntervention(fixedStep, causalObjects, PhysicsCaptureV2CaptureLimits.Default);
    }

    public void BeginPreIntervention(long fixedStep, GameObject[] causalObjects,
        PhysicsCaptureV2ContactInput[] contacts, bool completeContactEnumeration)
    {
        BeginPreIntervention(fixedStep, causalObjects, contacts, completeContactEnumeration,
            PhysicsCaptureV2CaptureLimits.Default);
    }

    public void BeginPreInterventionFromUnity(long fixedStep, GameObject[] causalObjects)
    {
        bool complete;
        PhysicsCaptureV2ContactInput[] contacts = CaptureUnityContacts(causalObjects,
            PhysicsCaptureV2CaptureLimits.Default, out complete);
        BeginPreIntervention(fixedStep, causalObjects, contacts, complete,
            PhysicsCaptureV2CaptureLimits.Default);
    }

    public void BeginPreIntervention(long fixedStep, GameObject[] causalObjects,
        PhysicsCaptureV2CaptureLimits captureLimits)
    {
        BeginPreIntervention(fixedStep, causalObjects, new PhysicsCaptureV2ContactInput[0],
            true, captureLimits);
    }

    public void BeginPreIntervention(long fixedStep, GameObject[] causalObjects,
        PhysicsCaptureV2ContactInput[] contacts, bool completeContactEnumeration,
        PhysicsCaptureV2CaptureLimits captureLimits)
    {
        Active = this;
        int configuredStride;
        string value = Environment.GetEnvironmentVariable(
            PhysicsCaptureV2EngineProtocol.StrideEnvironmentVariable,
            EnvironmentVariableTarget.Process);
        if (!int.TryParse(value, NumberStyles.None, CultureInfo.InvariantCulture,
            out configuredStride) || configuredStride <= 0)
            throw new InvalidOperationException("physics capture v2 stride is not configured");
        stride = configuredStride;
        string identity = Guid.NewGuid().ToString("N");
        captureId = "capture-v2:" + identity;
        shotId = "shot-v2:" + identity;
        limits = captureLimits;
        Failure = null;
        events.Clear();
        retainedEntities.Clear();
        retainedColliders.Clear();
        terminalReason = null;
        terminalEventId = null;
        preInterventionFixedStep = fixedStep;
        lastFixedStep = fixedStep;
        fixedStepSamples.Clear();
        AddSample(fixedStep, causalObjects, contacts, completeContactEnumeration);
        frameRecords.Clear();
        frameRecords.Add(new PhysicsCaptureV2FrameRecord(fixedStep, false));
        recording = true;
        finalized = false;
    }

    public void RecordFixedStep(long fixedStep)
    {
        RecordFixedStep(fixedStep, new GameObject[0]);
    }

    public void RecordFixedStep(long fixedStep, GameObject[] causalObjects)
    {
        RecordFixedStep(fixedStep, causalObjects, new PhysicsCaptureV2ContactInput[0], true);
    }

    public void RecordFixedStep(long fixedStep, GameObject[] causalObjects,
        PhysicsCaptureV2ContactInput[] contacts, bool completeContactEnumeration)
    {
        if (!recording || finalized)
            throw new InvalidOperationException("physics capture v2 recorder is not active");
        if (fixedStep != lastFixedStep + 1)
        {
            Failure = new PhysicsCaptureV2RecorderFailure(
                PhysicsCaptureV2EngineFailureCode.FixedStepGap,
                "physics capture v2 fixed-step contact coverage has a gap");
            return;
        }
        AddSample(fixedStep, causalObjects, contacts, completeContactEnumeration);
        if ((fixedStep - preInterventionFixedStep) % stride == 0)
            frameRecords.Add(new PhysicsCaptureV2FrameRecord(fixedStep, false));
        lastFixedStep = fixedStep;
    }

    public void RecordUnityFixedStep(long fixedStep, GameObject[] causalObjects)
    {
        bool complete;
        PhysicsCaptureV2ContactInput[] contacts = CaptureUnityContacts(
            causalObjects, limits, out complete);
        RecordFixedStep(fixedStep, causalObjects, contacts, complete);
    }

    public void FinalizeTerminal(long fixedStep)
    {
        FinalizeTerminal(fixedStep, "terminal");
    }

    public void FinalizeTerminal(long fixedStep, string reason)
    {
        if (!recording || finalized || fixedStep != lastFixedStep)
            throw new InvalidOperationException("physics capture v2 terminal must cover the final fixed step");
        terminalFixedStep = fixedStep;
        if (frameRecords[frameRecords.Count - 1].FixedStep != fixedStep)
            frameRecords.Add(new PhysicsCaptureV2FrameRecord(fixedStep, true));
        terminalReason = string.IsNullOrEmpty(reason) ? "terminal" : reason;
        for (int i = events.Count - 1; i >= 0; i--)
            if (events[i].FixedStep == fixedStep && events[i].EventType == terminalReason)
            {
                terminalEventId = events[i].EventId; break;
            }
        if (terminalEventId == null)
            terminalEventId = RecordMacroEvent(terminalReason, new string[0], "{}");
        finalized = true;
    }

    public string RecordMacroEvent(string eventType, string[] participants, string payloadJson)
    {
        if (!recording || finalized) return null;
        int ordinal = events.Count;
        string eventId = "event:" + lastFixedStep + ":" + eventType + ":" + ordinal.ToString("D4");
        events.Add(new PhysicsCaptureV2EventSnapshot(eventId, eventType, lastFixedStep,
            participants, payloadJson));
        return eventId;
    }

    public void Deactivate()
    {
        if (Active == this) Active = null;
    }

    public PhysicsCaptureV2EngineSnapshot CreateFinalizedSnapshot()
    {
        if (!finalized) return null;
        return new PhysicsCaptureV2EngineSnapshot(captureId, shotId, stride, preInterventionFixedStep,
            terminalFixedStep, fixedStepSamples, frameRecords, events, terminalReason, terminalEventId);
    }

    private void AddSample(long fixedStep, GameObject[] causalObjects,
        PhysicsCaptureV2ContactInput[] contactInputs, bool completeContactEnumeration)
    {
        List<PhysicsCaptureV2EntitySnapshot> entities;
        List<PhysicsCaptureV2ColliderSnapshot> colliders;
        PhysicsCaptureV2RecorderFailure failure;
        if (!PhysicsCaptureV2EntityGeometryExporter.TryCapture(causalObjects, limits,
            out entities, out colliders, out failure))
        {
            Failure = failure;
            return;
        }
        HashSet<string> currentEntityIds = new HashSet<string>(StringComparer.Ordinal);
        for (int i = 0; i < entities.Count; i++)
        {
            currentEntityIds.Add(entities[i].EntityId);
            retainedEntities[entities[i].EntityId] = entities[i];
        }
        foreach (KeyValuePair<string, PhysicsCaptureV2EntitySnapshot> retained in retainedEntities)
            if (!currentEntityIds.Contains(retained.Key))
                entities.Add(retained.Value.WithLifecycle("destroyed"));
        HashSet<string> currentColliderIds = new HashSet<string>(StringComparer.Ordinal);
        for (int i = 0; i < colliders.Count; i++)
        {
            currentColliderIds.Add(colliders[i].ColliderId);
            retainedColliders[colliders[i].ColliderId] = colliders[i];
        }
        foreach (KeyValuePair<string, PhysicsCaptureV2ColliderSnapshot> retained in retainedColliders)
            if (!currentColliderIds.Contains(retained.Key))
                colliders.Add(retained.Value.WithEnabled(false));
        entities.Sort(delegate(PhysicsCaptureV2EntitySnapshot left,
            PhysicsCaptureV2EntitySnapshot right) { return string.CompareOrdinal(left.EntityId, right.EntityId); });
        colliders.Sort(delegate(PhysicsCaptureV2ColliderSnapshot left,
            PhysicsCaptureV2ColliderSnapshot right) { return string.CompareOrdinal(left.ColliderId, right.ColliderId); });
        List<PhysicsCaptureV2ContactSnapshot> contacts;
        List<PhysicsCaptureV2SupportSnapshot> supports;
        List<PhysicsCaptureV2EntitySnapshot> contextualEntities;
        if (!PhysicsCaptureV2ContactExporter.TryCapture(fixedStep, causalObjects, contactInputs,
            completeContactEnumeration, limits, entities, colliders, out contacts, out supports,
            out contextualEntities, out failure))
        {
            Failure = failure;
            return;
        }
        fixedStepSamples.Add(new PhysicsCaptureV2FixedStepSample(fixedStep, contextualEntities,
            colliders, contacts, supports));
    }

    private static PhysicsCaptureV2ContactInput[] CaptureUnityContacts(
        GameObject[] causalObjects, PhysicsCaptureV2CaptureLimits captureLimits,
        out bool complete)
    {
        complete = true;
        List<PhysicsCaptureV2ContactInput> inputs = new List<PhysicsCaptureV2ContactInput>();
        Collider2D[] colliders = ActiveSceneColliders();
        int bufferLength = Math.Max(1, captureLimits.MaxContactsPerStep + 1);
        ContactPoint2D[] points = new ContactPoint2D[bufferLength];
        for (int colliderIndex = 0; colliderIndex < colliders.Length; colliderIndex++)
        {
            Collider2D collider = colliders[colliderIndex];
            if (collider == null || collider.isTrigger || !collider.enabled
                || !collider.gameObject.activeInHierarchy)
                continue;
            int count = collider.GetContacts(points);
            if (count == points.Length) complete = false;
            for (int pointIndex = 0; pointIndex < count; pointIndex++)
            {
                ContactPoint2D point = points[pointIndex];
                Collider2D other = point.otherCollider;
                if (other == null || other.isTrigger || !other.enabled
                    || !other.gameObject.activeInHierarchy
                    || collider.GetInstanceID() >= other.GetInstanceID())
                    continue;
                inputs.Add(new PhysicsCaptureV2ContactInput(collider, other, point.point,
                    -point.normal, point.separation));
            }
        }
        return inputs.ToArray();
    }

    private static Collider2D[] ActiveSceneColliders()
    {
        List<Collider2D> colliders = new List<Collider2D>();
        Collider2D[] found = Resources.FindObjectsOfTypeAll<Collider2D>();
        for (int index = 0; index < found.Length; index++)
        {
            Collider2D collider = found[index];
            if (collider != null && collider.gameObject.scene.IsValid()) colliders.Add(collider);
        }
        colliders.Sort(delegate(Collider2D left, Collider2D right)
        {
            return left.GetInstanceID().CompareTo(right.GetInstanceID());
        });
        return colliders.ToArray();
    }

    private void OnDestroy()
    {
        if (Active == this) Active = null;
    }
}
