using System.Reflection;
using NUnit.Framework;
using UnityEngine;

public class ABBirdLaunchTests
{
    [Test]
    public void BirdLaunchRecordsTheExactVelocityAfterItIsAssigned()
    {
        GameObject runtimeObject = new GameObject("bird-launch-runtime");
        PhysicalSnapshotRuntime runtime = runtimeObject.AddComponent<PhysicalSnapshotRuntime>();
        // PhysicalSnapshotRuntime.Active is bound in Awake, and the static record callbacks
        // no-op while it is null, so the runtime needs its lifecycle before BeginShot.
        AwakeComponent(runtime);
        runtime.BeginShot(8, 64 * 1024, 10f);
        GameObject birdObject = new GameObject("bird-launch-test");
        Rigidbody2D body = birdObject.AddComponent<Rigidbody2D>();
        birdObject.AddComponent<BoxCollider2D>();
        ABBird bird = birdObject.AddComponent<ABBird>();
        Vector2 staleVelocity = new Vector2(-3f, 4f);
        Vector2 launchVelocity = new Vector2(12.5f, 7.25f);

        try
        {
            // EditMode never raises Awake for AddComponent, so _rigidBody would stay null.
            // Run the real initialisation chain rather than assigning the field directly.
            AwakeComponent(bird);
            body.velocity = staleVelocity;
            MethodInfo assignAndRecord = typeof(ABBird).GetMethod(
                "AssignLaunchVelocityAndRecord", BindingFlags.Instance | BindingFlags.NonPublic);
            Assert.IsNotNull(assignAndRecord, "ABBird must expose the narrow launch assignment test seam");

            assignAndRecord.Invoke(bird, new object[] { launchVelocity });

            Assert.AreEqual(launchVelocity, body.velocity);
            Assert.AreEqual(1, runtime.ShotRecorder.Events.Count);
            Assert.AreEqual("bird_launched", runtime.ShotRecorder.Events[0].Taxonomy);
            Assert.AreEqual(launchVelocity, runtime.ShotRecorder.Events[0].Payload.LaunchVelocity.Value);
            Assert.AreNotEqual(staleVelocity, runtime.ShotRecorder.Events[0].Payload.LaunchVelocity.Value);
        }
        finally
        {
            Object.DestroyImmediate(birdObject);
            Object.DestroyImmediate(runtimeObject);
        }
    }

    private static void AwakeComponent(MonoBehaviour component)
    {
        for (System.Type type = component.GetType(); type != null; type = type.BaseType)
        {
            MethodInfo awake = type.GetMethod(
                "Awake", BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.DeclaredOnly);
            if (awake == null) continue;
            awake.Invoke(component, null);
            return;
        }
        Assert.Fail("no Awake found on " + component.GetType().Name + " or its base types");
    }
}
