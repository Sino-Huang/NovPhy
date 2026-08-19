using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

public class PhysicalShotRecorder
{
    private readonly PhysicalCaptureLimits limits;
    private readonly Dictionary<string, int> persistence = new Dictionary<string, int>();
    private readonly Dictionary<string, long> lastSeenStep = new Dictionary<string, long>();
    private readonly Dictionary<string, PhysicalRawContact> previousContacts = new Dictionary<string, PhysicalRawContact>();
    private readonly HashSet<string> eventKeys = new HashSet<string>();
    private readonly HashSet<string> collisionKeys = new HashSet<string>();
    private readonly HashSet<string> collisionCitedContactIds = new HashSet<string>();
    private readonly List<PhysicalRawContact> rawContacts = new List<PhysicalRawContact>();
    private readonly List<PhysicalSupportEdge> supportEdges = new List<PhysicalSupportEdge>();
    private readonly List<PhysicalMacroEvent> events = new List<PhysicalMacroEvent>();
    private readonly List<PhysicalEvidenceTraceSample> evidenceTrace = new List<PhysicalEvidenceTraceSample>();
    private readonly HashSet<string> evidenceEntityIds = new HashSet<string>();
    private readonly string evidenceShotId = "engine-shot-" + Guid.NewGuid().ToString("N");
    private int estimatedBytes;
    private float shotStartFixedTime;
    private bool hasShotStartFixedTime;
    private bool stabilityInitialized;
    private bool stable;
    private bool terminalRecorded;
    private bool finalized;
    private long? evidenceFirstFixedStep;
    private long? evidenceLastFixedStep;
    private int evidenceSampleCount;
    private string evidenceIncompleteReason;
    private bool evidenceTraceTruncated;
    private bool minimumSeparationObserved;
    private float minimumSeparation;
    private string minimumSeparationContactId;
    private long? minimumSeparationFixedStep;

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
        RecordContacts(fixedStep, fixedTime, contacts, true);
    }

    // isFullStepSample says whether `contacts` is the complete contact set for this
    // fixed step. UpdateSupport prunes every edge whose pair is absent from what it
    // is handed, so it may only ever see a full sample. The collision path ingests
    // one pair's contacts to mint contact_ids, and passes false: the FixedUpdate
    // sampler remains the sole owner of support derivation and picks the new pair up
    // on its next full sample. Retention pruning is likewise full-sample-only.
    private void RecordContacts(long fixedStep, float fixedTime, PhysicalContactInput[] contacts, bool isFullStepSample)
    {
        if (Failure != null || finalized)
        {
            return;
        }
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
        if (isFullStepSample)
            ObserveEvidenceCoverage(fixedStep);

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
        // Partial collision callbacks mint retained contact IDs but are not full
        // fixed-step coverage.  Only the full sampler can author the capture-wide
        // minimum witness used as negative/positive violation evidence.
        if (isFullStepSample)
        {
            foreach (PhysicalRawContact contact in stepContacts)
            {
                if (!minimumSeparationObserved || contact.Separation < minimumSeparation
                    || contact.Separation == minimumSeparation
                        && string.CompareOrdinal(contact.ContactId, minimumSeparationContactId) < 0)
                {
                    minimumSeparationObserved = true;
                    minimumSeparation = contact.Separation;
                    minimumSeparationContactId = contact.ContactId;
                    minimumSeparationFixedStep = contact.FixedStep;
                }
            }
        }
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
        if (isFullStepSample)
        {
            UpdateSupport(stepContacts, fixedStep);
            PruneRawContacts(fixedStep);
        }
    }

    public void RecordUnityContacts(long fixedStep, float fixedTime, Collider2D[] colliders, PhysicalEntityRegistry registry = null)
    {
        Rigidbody2D[] bodies = (colliders ?? new Collider2D[0])
            .Where(collider => collider != null && collider.attachedRigidbody != null)
            .Select(collider => collider.attachedRigidbody).Distinct().ToArray();
        RecordUnityContacts(fixedStep, fixedTime, colliders, bodies, registry);
    }

    public void RecordUnityContacts(long fixedStep, float fixedTime, Collider2D[] colliders,
        Rigidbody2D[] bodies, PhysicalEntityRegistry registry)
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
            if (count == points.Length)
                MarkEvidenceIncomplete("contact_sample_overflow");
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
        try
        {
            // Only the evidence-only entity/gravity snapshot is degradable.  The
            // established contact/support recorder above retains its original
            // exception and mutation behavior.
            RecordEvidenceTrace(fixedStep, bodies, registry, Physics2D.gravity);
        }
        catch (Exception)
        {
            MarkEvidenceIncomplete("sampling_failure");
        }
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

    // Refusal messages for the two evidence-bearing overloads below. Those two are
    // the only ones a Unity physics callback can reach, and a callback that throws
    // abandons the rest of its handler: the caller's post-collision gameplay never
    // runs, no terminal event is ever reached, and a capture run is lost to the
    // finalize timeout. So they refuse by logging and returning instead. This is
    // still fail-closed at the wire — the refusal path emits no event, so a
    // physics_capture_v1 artifact can never carry a collision without contact
    // evidence or with a non-finite relative speed, exactly as before.
    private const string CollisionEvidenceRejection =
        "physics_capture_v1: refusing a collision event without contact evidence; no event emitted.";
    private const string CollisionSpeedRejection =
        "physics_capture_v1: refusing a collision event whose relative speed is not finite and non-negative; no event emitted.";

    // No product caller and unreachable from any callback: this overload exists
    // only to make "record a collision without evidence" an unusable API, so it
    // keeps throwing.
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
        {
            Debug.LogError(CollisionEvidenceRejection);
            return;
        }
        if (float.IsNaN(relativeSpeed) || float.IsInfinity(relativeSpeed) || relativeSpeed < 0f)
        {
            Debug.LogError(CollisionSpeedRejection);
            return;
        }
        string first = string.CompareOrdinal(entityA, entityB) <= 0 ? entityA : entityB;
        string second = first == entityA ? entityB : entityA;
        string key = fixedStep + ":" + first + ":" + second;
        if (collisionKeys.Add(key))
        {
            // F7: the ids a collision event cites are the rows the event's
            // evidence must survive finalization with, so they are exempt from
            // step-window retention pruning.
            foreach (string contactId in evidence)
                collisionCitedContactIds.Add(contactId);
            AddEvent(fixedStep, fixedTime, PhysicalMacroEventKind.Collision, first + "|" + second,
                new[] { first, second }, new PhysicalMacroEventPayload(contactIds: evidence, relativeSpeed: relativeSpeed));
        }
    }

    public void RecordCollision(long fixedStep, float fixedTime, string entityA, string entityB,
        PhysicalContactInput[] contacts, float relativeSpeed)
    {
        if (Failure != null || finalized)
            return;
        if (float.IsNaN(relativeSpeed) || float.IsInfinity(relativeSpeed) || relativeSpeed < 0f)
        {
            Debug.LogError(CollisionSpeedRejection);
            return;
        }
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
        {
            Debug.LogError(CollisionEvidenceRejection);
            return;
        }
        string key = fixedStep + ":" + first + ":" + second;
        if (collisionKeys.Contains(key))
            return;
        // F3: the evidence is resolved by exact canonical collider pair — the same
        // key the raw rows carry — not by entity pair alone, so a sibling collider
        // on the same entity can no longer leak its contact ids into this event.
        HashSet<string> evidencePairKeys = new HashSet<string>();
        foreach (PhysicalContactInput input in evidence)
            evidencePairKeys.Add(new PhysicalRawContact(input, fixedStep, fixedTime).PairKey);
        string[] contactIds = RawContactIdsFor(fixedStep, evidencePairKeys);
        if (contactIds.Length == 0)
        {
            RecordContacts(fixedStep, fixedTime, evidence, false);
            contactIds = RawContactIdsFor(fixedStep, evidencePairKeys);
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
        if (finalized && Failure == null)
        {
            // F1/F2: the wire carries these lists in list order, so the global
            // ordering the frozen parser contract demands is satisfied exactly
            // once, here, by the recorder that owns the arrays — not by the
            // serializer. Per-step ordering and PointIndex assignment are
            // untouched, so contact ids never change. The comparator is a total
            // order even across duplicated rows: identical rows (same step, pair,
            // point and PointIndex) serialize identically, so their relative order
            // cannot be observed on the wire.
            rawContacts.Sort(CompareContactsWithContactId);
            supportEdges.Sort(CompareSupportEdges);
            return new PhysicalShotRecorderSnapshot(rawContacts, supportEdges, events);
        }
        return null;
    }

    public PhysicalViolationEngineEvidenceSnapshot CreateFinalizedEvidenceSnapshot()
    {
        if (!finalized || Failure != null)
            return null;
        return new PhysicalViolationEngineEvidenceSnapshot(
            evidenceShotId, evidenceFirstFixedStep, evidenceLastFixedStep, evidenceSampleCount,
            evidenceIncompleteReason, minimumSeparationObserved,
            minimumSeparationObserved ? (float?)minimumSeparation : null,
            minimumSeparationContactId, minimumSeparationFixedStep,
            evidenceTraceTruncated, evidenceTrace);
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

    private void PruneRawContacts(long fixedStep)
    {
        // F7 bounded retention: a support edge can only cite the current and the
        // previous fixed step, so after a full sample anything older than that
        // window is dropped — except rows a collision event cited, which are the
        // evidence a published artifact must carry to finalization. The byte
        // estimate is decremented by the same per-contact charge the append path
        // made, so the byte limit keeps measuring what is actually retained.
        int removed = rawContacts.RemoveAll(contact =>
            contact.FixedStep < fixedStep - 1 && !collisionCitedContactIds.Contains(contact.ContactId));
        estimatedBytes -= removed * 192;
    }

    private void ObserveEvidenceCoverage(long fixedStep)
    {
        if (!evidenceFirstFixedStep.HasValue)
            evidenceFirstFixedStep = fixedStep;
        if (evidenceLastFixedStep.HasValue && fixedStep != evidenceLastFixedStep.Value + 1)
        {
            MarkEvidenceIncomplete("fixed_step_gap");
            evidenceTrace.Clear();
            evidenceTraceTruncated = false;
        }
        evidenceLastFixedStep = fixedStep;
        evidenceSampleCount++;
    }

    private void MarkEvidenceIncomplete(string reason)
    {
        if (evidenceIncompleteReason == null)
            evidenceIncompleteReason = reason;
    }

    private void RecordEvidenceTrace(long fixedStep, Rigidbody2D[] bodies,
        PhysicalEntityRegistry registry, Vector2 globalGravity)
    {
        Dictionary<string, Rigidbody2D> current = new Dictionary<string, Rigidbody2D>();
        foreach (Rigidbody2D body in (bodies ?? new Rigidbody2D[0])
            .Where(item => item != null).OrderBy(item => item.GetInstanceID()))
        {
            string entityId = registry == null
                ? body.gameObject.GetInstanceID().ToString() + ":0"
                : registry.RegisterObject(body.gameObject);
            if (body.bodyType == RigidbodyType2D.Dynamic || evidenceEntityIds.Contains(entityId))
            {
                current[entityId] = body;
                evidenceEntityIds.Add(entityId);
            }
        }
        if (evidenceEntityIds.Count > PhysicalViolationEngineEvidenceSnapshot.MaxEntitiesPerStep)
            MarkEvidenceIncomplete("entity_sample_overflow");

        List<PhysicalEvidenceEntity> entities = new List<PhysicalEvidenceEntity>();
        foreach (string entityId in evidenceEntityIds.OrderBy(item => item, StringComparer.Ordinal)
            .Take(PhysicalViolationEngineEvidenceSnapshot.MaxEntitiesPerStep))
        {
            Rigidbody2D body;
            current.TryGetValue(entityId, out body);
            List<PhysicalEvidenceSupport> supports = supportEdges
                .Where(edge => edge.SupportedEntityId == entityId)
                .GroupBy(edge => SupportIdFor(edge))
                .OrderBy(group => group.Key, StringComparer.Ordinal)
                .Select(group => group
                    .OrderBy(edge => edge.PairKey, StringComparer.Ordinal)
                    .ThenBy(edge => edge.ContactIdA, StringComparer.Ordinal)
                    .ThenBy(edge => edge.ContactIdB, StringComparer.Ordinal)
                    .First())
                .Select(edge => new PhysicalEvidenceSupport(edge)).ToList();
            entities.Add(new PhysicalEvidenceEntity(entityId, body, supports));
        }
        if (evidenceTrace.Count > 0 && evidenceTrace[evidenceTrace.Count - 1].FixedStep == fixedStep)
            evidenceTrace.RemoveAt(evidenceTrace.Count - 1);
        evidenceTrace.Add(new PhysicalEvidenceTraceSample(fixedStep, globalGravity, entities));
        if (evidenceTrace.Count > PhysicalViolationEngineEvidenceSnapshot.MaxTraceFixedSteps)
        {
            evidenceTrace.RemoveAt(0);
            evidenceTraceTruncated = true;
        }
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

    private string[] RawContactIdsFor(long fixedStep, HashSet<string> pairKeys)
    {
        return rawContacts
            .Where(contact => contact.FixedStep == fixedStep && pairKeys.Contains(contact.PairKey))
            .Select(contact => contact.ContactId).Distinct().OrderBy(contactId => contactId, StringComparer.Ordinal).ToArray();
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

    private static int CompareContactsWithContactId(PhysicalRawContact left, PhysicalRawContact right)
    {
        int result = CompareContacts(left, right);
        if (result != 0) return result;
        return string.CompareOrdinal(left.ContactId, right.ContactId);
    }

    private static string SupportIdFor(PhysicalSupportEdge edge)
    {
        return "support:" + edge.SupporterEntityId + "->" + edge.SupportedEntityId;
    }

    private static int CompareSupportEdges(PhysicalSupportEdge left, PhysicalSupportEdge right)
    {
        int result = string.CompareOrdinal(left.SupporterEntityId, right.SupporterEntityId);
        if (result != 0) return result;
        result = string.CompareOrdinal(left.SupportedEntityId, right.SupportedEntityId);
        if (result != 0) return result;
        return string.CompareOrdinal(SupportIdFor(left), SupportIdFor(right));
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
