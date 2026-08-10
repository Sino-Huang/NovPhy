using System;
using System.Collections;
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
        registry.ResetLevel();
        clock.ResetLevel();
        shotRecorder = null;
        captureId = "capture-" + Guid.NewGuid().ToString("N");
        captureSequence = 1;
        stabilityCandidateSteps = 0;
    }

    public void BeginShot(int maxRecords, int maxBytes, float timeoutSeconds)
    {
        shotRecorder = new PhysicalShotRecorder(new PhysicalCaptureLimits(maxRecords, maxBytes, timeoutSeconds));
    }

    public PhysicalCaptureResult FinalizeShot(bool terminal)
    {
        return shotRecorder == null
            ? new PhysicalCaptureResult(null)
            : shotRecorder.FinalizeShot(terminal);
    }

    public void FinalizeTerminal()
    {
        if (shotRecorder != null)
            shotRecorder.FinalizeShot(true);
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
        Clock.ObserveFixedStep(Time.fixedTime);
        if (shotRecorder != null)
        {
            shotRecorder.RecordUnityContacts(Clock.FixedStep, Time.fixedTime, FindObjectsOfType<Collider2D>(), Registry);
            ObserveStability();
        }
    }

    public void RecordCollision(Collision2D collision)
    {
        if (shotRecorder == null || collision == null || collision.collider == null || collision.otherCollider == null)
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
        return registry.RegisterObject(gameObject);
    }

    public static string EntityIdForCallback(GameObject gameObject)
    {
        return Active == null ? null : Active.EntityIdFor(gameObject);
    }

    public void RecordLaunch(string entityId, Vector2 launchVelocity) { if (shotRecorder != null) shotRecorder.RecordLaunch(entityId, Clock.FixedStep, launchVelocity); }
    public void RecordDeath(string entityId) { if (shotRecorder != null) shotRecorder.RecordDeath(entityId, Clock.FixedStep); }
    public void RecordDestroyed(string entityId) { if (shotRecorder != null) shotRecorder.RecordDestroyed(entityId, Clock.FixedStep); }
    public void RecordPigRemoved(string entityId) { if (shotRecorder != null) shotRecorder.RecordPigRemoved(entityId, Clock.FixedStep); }
    public void RecordTntExplosion(string entityId, float radiusUnityUnits) { if (shotRecorder != null) shotRecorder.RecordTntExplosion(entityId, Clock.FixedStep, radiusUnityUnits); }
    public void RecordBirdExhaustion() { if (shotRecorder != null) shotRecorder.RecordBirdExhaustion(Clock.FixedStep); }
    public void RecordLevelClear(int score) { if (shotRecorder != null) shotRecorder.RecordLevelClear(Clock.FixedStep, score); }
    public void RecordLevelFail(string reason) { if (shotRecorder != null) shotRecorder.RecordLevelFail(Clock.FixedStep, reason); }
    public void RecordStability(bool stable) { if (shotRecorder != null) shotRecorder.RecordStability(Clock.FixedStep, stable); }

    public static void RecordCollisionCallback(Collision2D collision) { if (Active != null) Active.RecordCollision(collision); }
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
            RecordStability(stabilityCandidate);
    }
}
