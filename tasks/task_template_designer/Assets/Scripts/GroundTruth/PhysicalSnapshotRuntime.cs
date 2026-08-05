using System;
using System.Collections;
using UnityEngine;

public sealed class PhysicalSnapshotRuntime : MonoBehaviour
{
    public static PhysicalSnapshotRuntime Active { get; private set; }
    private PhysicalEntityRegistry registry;
    private PhysicalSnapshotClock clock;
    private PhysicalSnapshotExporter exporter;
    private string captureId;
    private long captureSequence;

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
        captureId = "capture-" + Guid.NewGuid().ToString("N");
        captureSequence = 1;
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
}
