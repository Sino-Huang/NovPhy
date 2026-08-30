using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using UnityEngine;

public sealed class PhysicalSnapshotRuntime : MonoBehaviour
{
    public static PhysicalSnapshotRuntime Active { get; private set; }
    private PhysicalEntityRegistry registry;
    private PhysicalSnapshotClock clock;
    private PhysicalSnapshotExporter exporter;
    private PhysicalShotRecorder shotRecorder;
    private string captureId;
    private long captureSequence;
    private bool stabilityCandidate;
    private int stabilityCandidateSteps;
    private bool v2StabilityCandidate;
    private int v2StabilityCandidateSteps;
    private bool? v2StableState;
    private bool v2InterventionObserved;
    private PhysicsCaptureV2FixedStepRecorder v2Recorder;
    private PhysicsCaptureV2AlignedObservationRecorder v2ObservationRecorder;
    private Coroutine v2PostPhysicsLoop;
    private readonly Dictionary<string, string> v2EntityIds = new Dictionary<string, string>();

    public PhysicalShotRecorder ShotRecorder { get { return shotRecorder; } }
    public string CaptureId { get { Initialize(); return captureId; } }
    public long NextCaptureSequence { get { Initialize(); return captureSequence++; } }

    public PhysicalEntityRegistry Registry
    {
        get
        {
            Initialize();
            return registry;
        }
    }

    public PhysicalSnapshotClock Clock
    {
        get
        {
            Initialize();
            return clock;
        }
    }

    public static PhysicalSnapshotRuntime Attach(GameObject host)
    {
        PhysicalSnapshotRuntime runtime = host.GetComponent<PhysicalSnapshotRuntime>();
        return runtime == null ? host.AddComponent<PhysicalSnapshotRuntime>() : runtime;
    }

    public void ResetLevel()
    {
        Initialize();
        if (v2PostPhysicsLoop != null) StopCoroutine(v2PostPhysicsLoop);
        v2PostPhysicsLoop = null;
        registry.ResetLevel();
        clock.ResetLevel();
        shotRecorder = null;
        if (v2Recorder != null) v2Recorder.Deactivate();
        v2Recorder = null;
        v2ObservationRecorder = null;
        v2EntityIds.Clear();
        captureId = "capture-" + Guid.NewGuid().ToString("N");
        captureSequence = 1;
        stabilityCandidateSteps = 0;
        v2StabilityCandidateSteps = 0;
        v2StableState = null;
        v2InterventionObserved = false;
    }

    public void BeginShot(int maxRecords, int maxBytes, float timeoutSeconds)
    {
        shotRecorder = new PhysicalShotRecorder(new PhysicalCaptureLimits(maxRecords, maxBytes, timeoutSeconds));
        BeginV2Shot();
    }

    public void BeginV2Shot()
    {
        int configuredStride;
        string stride = Environment.GetEnvironmentVariable(
            PhysicsCaptureV2EngineProtocol.StrideEnvironmentVariable,
            EnvironmentVariableTarget.Process);
        if (!int.TryParse(stride, NumberStyles.None, CultureInfo.InvariantCulture,
            out configuredStride) || configuredStride <= 0)
        {
            if (v2Recorder != null) v2Recorder.Deactivate();
            v2Recorder = null;
            return;
        }
        v2EntityIds.Clear();
        v2StabilityCandidateSteps = 0;
        v2StableState = null;
        v2InterventionObserved = false;
        v2Recorder = GetComponent<PhysicsCaptureV2FixedStepRecorder>();
        if (v2Recorder == null) v2Recorder = gameObject.AddComponent<PhysicsCaptureV2FixedStepRecorder>();
        v2Recorder.BeginPreInterventionFromUnity(Clock.FixedStep, V2CausalObjects());
        v2ObservationRecorder = PhysicsCaptureV2AlignedObservationRecorder.Create(
            v2Recorder.CaptureId);
        if (v2ObservationRecorder != null) v2ObservationRecorder.Capture(this);
        v2PostPhysicsLoop = StartCoroutine(CaptureV2PostPhysicsSteps());
    }

    public PhysicalCaptureResult FinalizeShot(bool terminal)
    {
        PhysicalCaptureResult result = shotRecorder == null
            ? new PhysicalCaptureResult(null)
            : shotRecorder.FinalizeShot(terminal);
        if (terminal) FinalizeV2("terminal");
        return result;
    }

    public void FinalizeTerminal()
    {
        if (shotRecorder != null)
            shotRecorder.FinalizeShot(true);
        FinalizeV2("terminal");
    }

    public PhysicalSceneSnapshot CaptureCurrent(
        SymbolicGameState symbolicState,
        int renderFrame,
        float renderTime)
    {
        Initialize();
        return exporter.Capture(symbolicState, registry, clock, renderFrame, renderTime);
    }

    public IEnumerator CaptureAtEndOfRenderFrame(
        SymbolicGameState symbolicState,
        Action<PhysicalSceneSnapshot> completed)
    {
        yield return new WaitForEndOfFrame();
        completed(CaptureCurrent(symbolicState, Time.frameCount, Time.time));
    }

    private void Awake()
    {
        Active = this;
        ResetLevel();
    }

    private void OnDestroy()
    {
        if (ReferenceEquals(Active, this))
            Active = null;
    }

    private void FixedUpdate()
    {
        if (v2Recorder != null && !v2Recorder.IsFinalized && v2Recorder.Failure == null
            && v2Recorder.LastFixedStep < Clock.FixedStep)
            CaptureV2PostPhysicsStep();
        Clock.ObserveFixedStep(Time.fixedTime);
        if (shotRecorder != null)
        {
            shotRecorder.RecordUnityContacts(
                Clock.FixedStep, Time.fixedTime, FindObjectsOfType<Collider2D>(),
                FindObjectsOfType<Rigidbody2D>(), Registry);
            ObserveStability();
        }
    }

    public void RecordCollision(Collision2D collision)
    {
        if (collision == null || collision.collider == null || collision.otherCollider == null)
            return;
        if (v2Recorder != null && !v2Recorder.IsFinalized)
            ((IPhysicsCaptureV2UnityCollisionRecorder)v2Recorder).RecordUnityCollision(
                Clock.FixedStep, collision, V2CausalObjects());
        if (shotRecorder == null)
            return;
        string first = registry.RegisterCollider(collision.collider);
        string second = registry.RegisterCollider(collision.otherCollider);
        PhysicalContactInput[] contacts = collision.contacts
            .Where(point => point.collider != null && point.otherCollider != null
                && !point.collider.isTrigger && !point.otherCollider.isTrigger)
            .Select(point => new PhysicalContactInput(
                registry.RegisterCollider(point.collider), point.collider.GetInstanceID(), point.point, point.normal,
                point.separation, point.relativeVelocity, point.normalImpulse, point.tangentImpulse,
                registry.RegisterCollider(point.otherCollider), point.otherCollider.GetInstanceID(),
                point.collider.transform.position, point.otherCollider.transform.position, false))
            .ToArray();
        shotRecorder.RecordCollision(Clock.FixedStep, Time.fixedTime, first, second,
            contacts, collision.relativeVelocity.magnitude);
    }

    public string EntityIdFor(GameObject gameObject)
    {
        Initialize();
        string entityId = registry.RegisterObject(gameObject);
        ScenarioObjectIdentity identity = gameObject == null
            ? null : gameObject.GetComponent<ScenarioObjectIdentity>();
        if (identity != null && !string.IsNullOrEmpty(identity.ScenarioObjectId))
            v2EntityIds[entityId] = "runtime:" + identity.ScenarioObjectId;
        return entityId;
    }

    public static string EntityIdForCallback(GameObject gameObject)
    {
        return Active == null ? null : Active.EntityIdFor(gameObject);
    }

    public void RecordLaunch(string entityId, Vector2 launchVelocity)
    {
        if (shotRecorder != null) shotRecorder.RecordLaunch(entityId, Clock.FixedStep, launchVelocity);
        v2InterventionObserved = true;
        v2StabilityCandidateSteps = 0;
        RecordV2("bird_launched", entityId,
            "{\"launch_velocity\":[" + F(launchVelocity.x) + "," + F(launchVelocity.y) + "]}");
    }
    public void RecordDeath(string entityId) { if (shotRecorder != null) shotRecorder.RecordDeath(entityId, Clock.FixedStep); RecordV2("entity_death", entityId, "{}"); }
    public void RecordDestroyed(string entityId) { if (shotRecorder != null) shotRecorder.RecordDestroyed(entityId, Clock.FixedStep); RecordV2("entity_destroyed", entityId, "{}"); }
    public void RecordPigRemoved(string entityId) { if (shotRecorder != null) shotRecorder.RecordPigRemoved(entityId, Clock.FixedStep); RecordV2("pig_removed", entityId, "{}"); }
    public void RecordTntExplosion(string entityId, float radiusUnityUnits) { if (shotRecorder != null) shotRecorder.RecordTntExplosion(entityId, Clock.FixedStep, radiusUnityUnits); RecordV2("tnt_explosion", entityId, "{\"radius_unity_units\":" + F(radiusUnityUnits) + "}"); }
    public void RecordBirdExhaustion() { if (shotRecorder != null) shotRecorder.RecordBirdExhaustion(Clock.FixedStep); RecordV2("bird_exhaustion", null, "{}"); }
    public void RecordLevelClear(int score) { if (shotRecorder != null) shotRecorder.RecordLevelClear(Clock.FixedStep, score); RecordV2("level_clear", null, "{\"score\":" + score + "}"); FinalizeV2("level_clear"); }
    public void RecordLevelFail(string reason) { if (shotRecorder != null) shotRecorder.RecordLevelFail(Clock.FixedStep, reason); RecordV2("level_fail", null, "{\"reason\":\"" + Escape(reason) + "\"}"); FinalizeV2("level_fail"); }
    public void RecordStability(bool stable) { if (shotRecorder != null) shotRecorder.RecordStability(Clock.FixedStep, stable); RecordV2(stable ? "stable_entered" : "stable_exited", null, "{}"); if (stable) FinalizeV2("stable_entered"); }

    public static void RecordCollisionCallback(Collision2D collision)
    {
        if (Active == null)
            return;
        // A Unity physics callback must never throw: an exception here abandons
        // the rest of the caller's OnCollisionEnter2D for the frame. The recorder
        // path is already fail-closed at the wire, but this is the narrow actual
        // boundary that keeps any unforeseen defect inside the recorder path from
        // escaping into the engine. The error carries the stable physics_capture_v1
        // refusal prefix so the smoke's log scan and any LogAssert fixture can
        // match it, and nothing outside this call is swallowed.
        try
        {
            Active.RecordCollision(collision);
        }
        catch (Exception)
        {
            Debug.LogError("physics_capture_v1: refusing a recorder exception inside a physics callback; no event emitted.");
        }
    }
    public static void RecordLaunchCallback(string entityId, Vector2 launchVelocity) { if (Active != null) Active.RecordLaunch(entityId, launchVelocity); }
    public static void RecordDeathCallback(string entityId) { if (Active != null) Active.RecordDeath(entityId); }
    public static void RecordDestroyedCallback(string entityId) { if (Active != null) Active.RecordDestroyed(entityId); }
    public static void RecordPigRemovedCallback(string entityId) { if (Active != null) Active.RecordPigRemoved(entityId); }
    public static void RecordTntExplosionCallback(string entityId, float radiusUnityUnits) { if (Active != null) Active.RecordTntExplosion(entityId, radiusUnityUnits); }
    public static void RecordBirdExhaustionCallback() { if (Active != null) Active.RecordBirdExhaustion(); }
    public static void RecordLevelClearCallback(int score) { if (Active != null) Active.RecordLevelClear(score); }
    public static void RecordLevelFailCallback(string reason) { if (Active != null) Active.RecordLevelFail(reason); }
    public static void RecordStabilityCallback(bool stable) { if (Active != null) Active.RecordStability(stable); }
    public static void BeginShotCallback(int maxRecords, int maxBytes, float timeoutSeconds)
    {
        if (Active != null)
            Active.BeginShot(maxRecords, maxBytes, timeoutSeconds);
    }
    public static void BeginV2ShotCallback() { if (Active != null) Active.BeginV2Shot(); }
    public static void FinalizeTerminalCallback() { if (Active != null) Active.FinalizeTerminal(); }

    private void Initialize()
    {
        if (registry != null)
        {
            EnsureCaptureContext();
            return;
        }

        registry = new PhysicalEntityRegistry();
        clock = new PhysicalSnapshotClock();
        exporter = new PhysicalSnapshotExporter();
        EnsureCaptureContext();
    }

    private void EnsureCaptureContext()
    {
        if (string.IsNullOrEmpty(captureId))
        {
            captureId = "capture-" + Guid.NewGuid().ToString("N");
            captureSequence = 1;
        }
    }

    private void ObserveStability()
    {
        bool candidate = true;
        foreach (Rigidbody2D body in FindObjectsOfType<Rigidbody2D>())
        {
            if (body.bodyType == RigidbodyType2D.Dynamic
                && (body.velocity.sqrMagnitude > 0.0001f || Mathf.Abs(body.angularVelocity) > 0.01f))
            {
                candidate = false;
                break;
            }
        }
        if (stabilityCandidateSteps == 0 || stabilityCandidate != candidate)
        {
            stabilityCandidate = candidate;
            stabilityCandidateSteps = 1;
        }
        else
        {
            stabilityCandidateSteps++;
        }
        if (stabilityCandidateSteps >= 2)
            shotRecorder.RecordStability(Clock.FixedStep, stabilityCandidate);
    }

    private IEnumerator CaptureV2PostPhysicsSteps()
    {
        WaitForFixedUpdate endOfPhysicsStep = new WaitForFixedUpdate();
        while (v2Recorder != null && !v2Recorder.IsFinalized && v2Recorder.Failure == null)
        {
            yield return endOfPhysicsStep;
            CaptureV2PostPhysicsStep();
        }
        v2PostPhysicsLoop = null;
    }

    private void CaptureV2PostPhysicsStep()
    {
        if (v2Recorder == null || v2Recorder.IsFinalized || v2Recorder.Failure != null) return;
        v2Recorder.RecordUnityFixedStep(Clock.FixedStep, V2CausalObjects());
        if (v2Recorder.Failure == null && v2ObservationRecorder != null)
            v2ObservationRecorder.Capture(this);
        if (v2Recorder.Failure == null) ObserveV2Stability();
    }

    private void ObserveV2Stability()
    {
        if (!v2InterventionObserved) return;
        bool candidate = true;
        foreach (Rigidbody2D body in FindObjectsOfType<Rigidbody2D>())
        {
            if (body.bodyType == RigidbodyType2D.Dynamic
                && (body.velocity.sqrMagnitude > 0.0001f || Mathf.Abs(body.angularVelocity) > 0.01f))
            {
                candidate = false;
                break;
            }
        }
        if (v2StabilityCandidateSteps == 0 || v2StabilityCandidate != candidate)
        {
            v2StabilityCandidate = candidate;
            v2StabilityCandidateSteps = 1;
        }
        else
        {
            v2StabilityCandidateSteps++;
        }
        if (v2StabilityCandidateSteps == 2
            && (!v2StableState.HasValue || v2StableState.Value != v2StabilityCandidate))
        {
            v2StableState = v2StabilityCandidate;
            RecordV2(v2StabilityCandidate ? "stable_entered" : "stable_exited", null, "{}");
            bool levelClearPending = Resources.FindObjectsOfTypeAll<ABGameWorld>()
                .Any(world => world != null && world.gameObject.scene.IsValid()
                    && world.IsLevelClearPending());
            if (v2StabilityCandidate && !levelClearPending)
                FinalizeV2("stable_entered");
        }
    }

    private GameObject[] V2CausalObjects()
    {
        return Resources.FindObjectsOfTypeAll<ScenarioObjectIdentity>()
            .Where(identity => identity != null && identity.gameObject.scene.IsValid())
            .OrderBy(identity => identity.ScenarioObjectId, StringComparer.Ordinal)
            .Select(identity => identity.gameObject).ToArray();
    }

    private void RecordV2(string eventType, string v1EntityId, string payload)
    {
        if (v2Recorder == null || v2Recorder.IsFinalized) return;
        string participant;
        string[] participants = v1EntityId != null && v2EntityIds.TryGetValue(v1EntityId, out participant)
            ? new[] { participant } : new string[0];
        v2Recorder.RecordMacroEvent(Clock.FixedStep, eventType, participants, payload);
    }

    private static string V2Id(Collider2D collider)
    {
        ScenarioObjectIdentity identity = collider == null
            ? null : collider.GetComponentInParent<ScenarioObjectIdentity>();
        return identity == null || string.IsNullOrEmpty(identity.ScenarioObjectId)
            ? null : "runtime:" + identity.ScenarioObjectId;
    }

    private void FinalizeV2(string reason)
    {
        if (v2Recorder != null && !v2Recorder.IsFinalized && v2Recorder.Failure == null
            && v2Recorder.LastFixedStep < Clock.FixedStep)
            CaptureV2PostPhysicsStep();
        if (v2Recorder != null && !v2Recorder.IsFinalized && v2Recorder.Failure == null)
            v2Recorder.FinalizeTerminal(Clock.FixedStep, reason);
    }

    private static string F(float value) { return value.ToString("R", CultureInfo.InvariantCulture); }
    private static string Escape(string value) { return (value ?? "").Replace("\\", "\\\\").Replace("\"", "\\\""); }
}
