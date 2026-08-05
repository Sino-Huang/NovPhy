using NUnit.Framework;
using UnityEngine;

public class PhysicalEntityRegistryTests
{
    [Test]
    public void Register_ReusedUnityInstanceId_AllocatesDistinctLifetimeOrdinal()
    {
        PhysicalEntityRegistry registry = new PhysicalEntityRegistry();
        GameObject firstLifetime = new GameObject("first-lifetime");
        GameObject secondLifetime = new GameObject("second-lifetime");

        try
        {
            Assert.AreEqual("41:0", registry.Register(41, firstLifetime));
            Assert.AreEqual("41:0", registry.Register(41, firstLifetime));

            UnityEngine.Object.DestroyImmediate(firstLifetime);
            Assert.AreEqual("41:1", registry.Register(41, secondLifetime));
        }
        finally
        {
            UnityEngine.Object.DestroyImmediate(secondLifetime);
        }
    }

    [Test]
    public void ResetLevel_ClearsLifetimeOrdinalsAndFixedClock()
    {
        PhysicalEntityRegistry registry = new PhysicalEntityRegistry();
        PhysicalSnapshotClock clock = new PhysicalSnapshotClock();
        GameObject lifetime = new GameObject("lifetime");

        try
        {
            Assert.AreEqual("73:0", registry.Register(73, lifetime));
            clock.ObserveFixedStep(0.02f);
            clock.ObserveFixedStep(0.04f);
            Assert.AreEqual(2L, clock.FixedStep);
            Assert.AreEqual(0.04f, clock.FixedTime);

            registry.ResetLevel();
            clock.ResetLevel();

            Assert.AreEqual("73:0", registry.Register(73, lifetime));
            Assert.AreEqual(0L, clock.FixedStep);
            Assert.AreEqual(0f, clock.FixedTime);
        }
        finally
        {
            UnityEngine.Object.DestroyImmediate(lifetime);
        }
    }
}
