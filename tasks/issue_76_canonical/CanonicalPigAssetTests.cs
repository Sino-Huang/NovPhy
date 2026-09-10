using NUnit.Framework;
using UnityEditor;
using UnityEngine;

// Reference-bundle values, not inferred from the failed C1 rollout.
public class CanonicalPigAssetTests
{
    [TestCase("BasicBig", 100000000f, CollisionDetectionMode2D.Continuous)]
    [TestCase("PinkBigPig", 10000f, CollisionDetectionMode2D.Discrete)]
    public void RegisteredPrefabRetainsReferencePhysicsAndHitThreshold(
        string name, float life, CollisionDetectionMode2D detection)
    {
        GameObject prefab = name == "BasicBig" ? ABWorldAssets.PIGS[name] : ABWorldAssets.NOVELTIES[name];
        Assert.IsNotNull(prefab);
        var pig = prefab.GetComponent<DieOnBirdHitCountPig>();
        Assert.IsNotNull(pig, "a plain ABPig is not the reference behavior");
        Assert.AreEqual(life, pig._life);
        Assert.AreEqual(3, new SerializedObject(pig).FindProperty("numberOfBirdShotsToDie").intValue);
        var body = prefab.GetComponent<Rigidbody2D>();
        Assert.IsFalse(body.useAutoMass);
        Assert.AreEqual(1.5f, body.mass);
        Assert.AreEqual(0.5f, body.gravityScale);
        Assert.AreEqual(0.05f, body.angularDrag);
        Assert.AreEqual(detection, body.collisionDetectionMode);
        Assert.IsNotNull(prefab.GetComponent<SpriteRenderer>().sprite);
    }

    [Test]
    public void NormalAndNovelRetainIdenticalAuthoredCollisionPolygon()
    {
        var normal = ABWorldAssets.PIGS["BasicBig"].GetComponent<PolygonCollider2D>();
        var novel = ABWorldAssets.NOVELTIES["PinkBigPig"].GetComponent<PolygonCollider2D>();
        Assert.IsTrue(normal.enabled);
        Assert.IsTrue(novel.enabled);
        Assert.AreEqual(15, normal.points.Length);
        CollectionAssert.AreEqual(normal.points, novel.points);
    }
}
