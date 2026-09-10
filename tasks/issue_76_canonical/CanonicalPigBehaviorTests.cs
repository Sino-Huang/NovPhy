using NUnit.Framework;
using System;
using System.Collections.Generic;
using System.Reflection;
using UnityEngine;

public class CanonicalPigBehaviorTests
{
    [Test]
    public void ManualObserverClockHasNoSecondAutomaticFixedUpdate()
    {
        var host = new GameObject("manual-clock-fixture");
        float speed = ABGameWorld.SimulationSpeed;
        try
        {
            ABGameWorld.SimulationSpeed = 1;
            var runtime = PhysicalSnapshotRuntime.Attach(host);
            Assert.IsNull(typeof(PhysicalSnapshotRuntime).GetMethod("FixedUpdate", BindingFlags.Instance | BindingFlags.NonPublic));
            Assert.AreEqual(0, runtime.Clock.FixedStep);
            runtime.BeforeManualSimulation();
            runtime.AfterManualSimulation();
            Assert.AreEqual(1, runtime.Clock.FixedStep);
            runtime.BeforeManualSimulation();
            runtime.AfterManualSimulation();
            Assert.AreEqual(2, runtime.Clock.FixedStep);
            ABGameWorld.SimulationSpeed = 2;
            Assert.Throws<InvalidOperationException>(() => runtime.BeforeManualSimulation());
            Assert.AreEqual(2, runtime.Clock.FixedStep);
        }
        finally
        {
            ABGameWorld.SimulationSpeed = speed;
            UnityEngine.Object.DestroyImmediate(host);
        }
    }

    [Test]
    public void EnvironmentVariableReallySeedsUnityRandom()
    {
        var previousState = UnityEngine.Random.state;
        string previous = Environment.GetEnvironmentVariable("NOVPHY_ENVIRONMENT_SEED");
        try
        {
            UnityEngine.Random.InitState(760720001);
            float expected = UnityEngine.Random.value;
            UnityEngine.Random.InitState(5);
            Environment.SetEnvironmentVariable("NOVPHY_ENVIRONMENT_SEED", "760720001");
            typeof(CanonicalCaptureSeed).GetMethod("BindEnvironmentSeed", BindingFlags.Static | BindingFlags.NonPublic).Invoke(null, null);
            Assert.AreEqual(expected, UnityEngine.Random.value);
        }
        finally
        {
            UnityEngine.Random.state = previousState;
            Environment.SetEnvironmentVariable("NOVPHY_ENVIRONMENT_SEED", previous);
        }
    }

    [Test]
    public void XmlScenarioIdentitiesSurviveCanonicalLevelLoading()
    {
        string xml = "<Level width='1'><Camera x='0' y='0' minWidth='25' maxWidth='35'/>"
            + "<Score highScore='0'/><Birds>\n<Bird type='BirdRed' scenarioObjectId='bird:0000'/>\n</Birds>"
            + "<Slingshot x='-8' y='-2' scenarioObjectId='world:slingshot'/>"
            + "<assetBundle path='absent'/><GameObjects>\n"
            + "<Pig type='PinkBigPig' x='0' y='0' rotation='0' scenarioObjectId='pig:0000'/>\n"
            + "</GameObjects></Level>";
        ABLevel level = LevelLoader.LoadXmlLevel(xml);
        Assert.AreEqual("bird:0000", level.birds[0].scenarioObjectId);
        Assert.AreEqual("pig:0000", level.pigs[0].scenarioObjectId);
        Assert.AreEqual("world:slingshot", level.slingshot.scenarioObjectId);
    }

    [Test]
    public void RepeatedBirdAndNonBirdContactDoNotAddDistinctHits()
    {
        GameObject first = UnityEngine.Object.Instantiate(ABWorldAssets.BIRDS["BirdRed"]);
        GameObject second = UnityEngine.Object.Instantiate(ABWorldAssets.BIRDS["BirdRed"]);
        GameObject pigObject = UnityEngine.Object.Instantiate(ABWorldAssets.NOVELTIES["PinkBigPig"]);
        GameObject block = new GameObject("non-bird-contact");
        block.AddComponent<BoxCollider2D>();
        try
        {
            var pig = pigObject.GetComponent<DieOnBirdHitCountPig>();
            // EditMode does not dispatch Awake for these gameplay behaviours.
            typeof(ABGameObject).GetMethod("Awake", BindingFlags.Instance | BindingFlags.NonPublic).Invoke(pig, null);
            Assert.AreEqual(10000f, pig.getCurrentLife());
            typeof(DieOnBirdHitCountPig).GetMethod("Start", BindingFlags.Instance | BindingFlags.NonPublic).Invoke(pig, null);
            pig.OnCollisionEnter2D(Collision(first, pigObject));
            pig.OnCollisionEnter2D(Collision(first, pigObject));
            pig.OnCollisionEnter2D(Collision(block, pigObject));
            pig.OnCollisionEnter2D(Collision(second, pigObject));
            var counted = (List<int>)typeof(DieOnBirdHitCountPig).GetField("collidedBirdIDs",
                BindingFlags.Instance | BindingFlags.NonPublic).GetValue(pig);
            Assert.AreEqual(2, counted.Count);
            Assert.IsTrue(pigObject.GetComponent<PolygonCollider2D>().enabled);
        }
        finally
        {
            UnityEngine.Object.DestroyImmediate(block);
            UnityEngine.Object.DestroyImmediate(pigObject);
            UnityEngine.Object.DestroyImmediate(second);
            UnityEngine.Object.DestroyImmediate(first);
        }
    }

    // Same Collision2D field seam used by the existing gameplay-recorder tests.
    private static Collision2D Collision(GameObject incoming, GameObject self)
    {
        var collision = (Collision2D)Activator.CreateInstance(typeof(Collision2D), true);
        Set(collision, "m_Collider", incoming.GetComponent<Collider2D>().GetInstanceID());
        Set(collision, "m_OtherCollider", self.GetComponent<Collider2D>().GetInstanceID());
        var body = incoming.GetComponent<Rigidbody2D>();
        Set(collision, "m_Rigidbody", body == null ? 0 : body.GetInstanceID());
        Set(collision, "m_OtherRigidbody", self.GetComponent<Rigidbody2D>().GetInstanceID());
        Set(collision, "m_RelativeVelocity", Vector2.zero);
        Assert.AreSame(incoming, collision.gameObject);
        return collision;
    }

    private static void Set(object target, string name, object value)
    {
        var field = target.GetType().GetField(name, BindingFlags.Instance | BindingFlags.NonPublic);
        Assert.IsNotNull(field);
        field.SetValue(target, value);
    }

}
