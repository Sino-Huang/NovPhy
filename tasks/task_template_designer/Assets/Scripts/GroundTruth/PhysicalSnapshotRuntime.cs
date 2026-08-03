using System;
using System.Collections;
using UnityEngine;

public sealed class PhysicalSnapshotRuntime : MonoBehaviour
{
    private PhysicalEntityRegistry registry;
    private PhysicalSnapshotClock clock;
    private PhysicalSnapshotExporter exporter;

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
        ResetLevel();
    }

    private void FixedUpdate()
    {
        Clock.ObserveFixedStep(Time.fixedTime);
    }

    private void Initialize()
    {
        if (registry != null)
        {
            return;
        }

        registry = new PhysicalEntityRegistry();
        clock = new PhysicalSnapshotClock();
        exporter = new PhysicalSnapshotExporter();
    }
}
