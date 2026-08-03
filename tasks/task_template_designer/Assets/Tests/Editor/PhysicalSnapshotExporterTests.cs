using System;
using System.Collections;
using NUnit.Framework;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;

public class PhysicalSnapshotExporterTests
{
    private Texture2D texture;
    private Sprite dynamicSprite;
    private Sprite staticSprite;
    private Sprite slingshotSprite;
    private int originalNoveltyType;

    [SetUp]
    public void SetUp()
    {
        originalNoveltyType = ABGameWorld.noveltyTypeForNovelty3;
        EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
        ABGameWorld.noveltyTypeForNovelty3 = -1;

        GameObject schemaObject = new GameObject("LoadLevelSchema");
        LoadLevelSchema schema = schemaObject.AddComponent<LoadLevelSchema>();
        typeof(LoadLevelSchema)
            .GetProperty("devMode")
            .GetSetMethod(true)
            .Invoke(schema, new object[] { true });

        GameObject cameraObject = new GameObject("Main Camera");
        cameraObject.tag = "MainCamera";
        Camera camera = cameraObject.AddComponent<Camera>();
        camera.orthographic = true;
        cameraObject.transform.position = new Vector3(0f, 0f, -10f);

        texture = new Texture2D(4, 4);
        dynamicSprite = CreateSprite("DynamicWood");
        staticSprite = CreateSprite("StaticPlatform");
        slingshotSprite = CreateSprite("SlingshotBack");
        CreateSlingshot();
        CreateGround();
    }

    [TearDown]
    public void TearDown()
    {
        ABGameWorld.noveltyTypeForNovelty3 = originalNoveltyType;
        UnityEngine.Object.DestroyImmediate(dynamicSprite);
        UnityEngine.Object.DestroyImmediate(staticSprite);
        UnityEngine.Object.DestroyImmediate(slingshotSprite);
        UnityEngine.Object.DestroyImmediate(texture);
        EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
    }

    [Test]
    public void CaptureCurrent_ExportsCurrentPhysicsStaticNullsClocksAndDeterministicOrder()
    {
        GameObject dynamicObject = CreateRectObject("dynamic", "Rect", dynamicSprite);
        Rigidbody2D body = dynamicObject.AddComponent<Rigidbody2D>();
        body.mass = 2f;
        body.velocity = new Vector2(3f, 4f);
        body.angularVelocity = 12f;
        ABGameObject gameObjectState = dynamicObject.AddComponent<ABGameObject>();
        gameObjectState.lastVelocity = new Vector3(99f, 99f, 0f);

        GameObject staticObject = CreateRectObject("static-body", "Platform", staticSprite);
        BoxCollider2D staticCollider = staticObject.GetComponent<BoxCollider2D>();
        Rigidbody2D staticBody = staticObject.AddComponent<Rigidbody2D>();
        staticBody.bodyType = RigidbodyType2D.Static;

        GameObject absentBodyObject = CreateRectObject("absent-body", "Platform", staticSprite);
        BoxCollider2D absentBodyCollider = absentBodyObject.GetComponent<BoxCollider2D>();
        BoxCollider2D groundCollider = GameObject.Find("Ground").GetComponent<BoxCollider2D>();

        PhysicalSnapshotRuntime runtime = new GameObject("snapshot-runtime").AddComponent<PhysicalSnapshotRuntime>();
        runtime.Clock.ObserveFixedStep(0.02f);
        runtime.Clock.ObserveFixedStep(0.04f);

        PhysicalSceneSnapshot snapshot = runtime.CaptureCurrent(new SymbolicGameState(false), 17, 0.5f);
        PhysicalNodeSnapshot dynamicNode = FindNode(snapshot, dynamicObject.GetInstanceID());
        PhysicalNodeSnapshot staticNode = FindNode(snapshot, staticCollider.GetInstanceID());
        PhysicalNodeSnapshot absentBodyNode = FindNode(snapshot, absentBodyCollider.GetInstanceID());
        PhysicalNodeSnapshot groundNode = FindNode(snapshot, groundCollider.GetInstanceID());

        Assert.AreEqual(17, snapshot.RenderFrame);
        Assert.AreEqual(0.5f, snapshot.RenderTime);
        Assert.AreEqual(2L, snapshot.FixedStep);
        Assert.AreEqual(0.04f, snapshot.FixedTime);
        Assert.AreEqual("Rect", dynamicNode.ObjectClass);
        Assert.AreEqual("DynamicWood", dynamicNode.ObjectType);
        Assert.AreEqual(10000f, dynamicNode.Life);
        Assert.IsTrue(dynamicNode.Body.Present);
        Assert.AreEqual(new Vector2(3f, 4f), dynamicNode.Body.Velocity.Value);
        Assert.AreEqual(12f, dynamicNode.Body.AngularVelocityDegreesPerSecond.Value);
        Assert.AreEqual(2f, dynamicNode.Body.MassUnityUnits.Value);
        Assert.AreEqual(25f, dynamicNode.Body.KineticEnergyUnityUnits.Value);
        Assert.AreEqual(dynamicObject.transform.position.x, dynamicNode.WorldPosition.x);
        Assert.AreEqual(dynamicObject.transform.position.y, dynamicNode.WorldPosition.y);
        Assert.GreaterOrEqual(dynamicNode.ScreenPolygon.Length, 3);

        Assert.AreEqual("world:static:" + staticCollider.GetInstanceID(), staticNode.EntityId);
        Assert.IsNull(staticNode.Life);
        Assert.IsFalse(staticNode.Body.Present);
        Assert.IsFalse(staticNode.Body.Velocity.HasValue);
        Assert.IsFalse(staticNode.Body.AngularVelocityDegreesPerSecond.HasValue);
        Assert.IsFalse(staticNode.Body.MassUnityUnits.HasValue);
        Assert.IsFalse(staticNode.Body.KineticEnergyUnityUnits.HasValue);
        Assert.IsFalse(absentBodyNode.Body.Present);
        Assert.IsFalse(absentBodyNode.Body.Velocity.HasValue);
        Assert.IsFalse(absentBodyNode.Body.MassUnityUnits.HasValue);
        Assert.AreEqual("world", groundNode.ObjectClass);
        Assert.AreEqual("Ground", groundNode.ObjectType);
        Assert.AreEqual(4, groundNode.ScreenPolygon.Length);
        Assert.AreEqual(5, snapshot.Nodes.Length);

        for (int index = 1; index < snapshot.Nodes.Length; index++)
        {
            Assert.LessOrEqual(
                string.CompareOrdinal(snapshot.Nodes[index - 1].EntityId, snapshot.Nodes[index].EntityId),
                0);
        }

        string json = snapshot.ToJson();
        StringAssert.Contains("\"velocity\":null", json);
        StringAssert.Contains("\"mass_unity_units\":null", json);
        StringAssert.DoesNotContain("rgb_frame", json);
        StringAssert.DoesNotContain("raw_contacts", json);
        StringAssert.DoesNotContain("support_edges", json);
        StringAssert.DoesNotContain("event", json);

        PhysicalSceneSnapshot repeated = runtime.CaptureCurrent(new SymbolicGameState(false), 17, 0.5f);
        Assert.AreEqual(json, repeated.ToJson());
        CollectionAssert.AreEqual(EntityIds(snapshot), EntityIds(repeated));
    }

    [Test]
    public void CaptureAtEndOfRenderFrame_YieldsBeforeReadingSnapshot()
    {
        PhysicalSnapshotRuntime runtime = new GameObject("snapshot-runtime").AddComponent<PhysicalSnapshotRuntime>();
        PhysicalSceneSnapshot captured = null;
        IEnumerator capture = runtime.CaptureAtEndOfRenderFrame(
            new SymbolicGameState(false),
            delegate(PhysicalSceneSnapshot snapshot) { captured = snapshot; });

        Assert.IsTrue(capture.MoveNext());
        Assert.IsInstanceOf<WaitForEndOfFrame>(capture.Current);
        Assert.IsNull(captured);
        Assert.IsFalse(capture.MoveNext());
        Assert.IsNotNull(captured);
    }

    private Sprite CreateSprite(string name)
    {
        Sprite created = Sprite.Create(texture, new Rect(0f, 0f, 4f, 4f), new Vector2(0.5f, 0.5f), 1f);
        created.name = name;
        return created;
    }

    private void CreateSlingshot()
    {
        GameObject slingshot = new GameObject("slingshot_back");
        slingshot.tag = "Slingshot";
        slingshot.AddComponent<SpriteRenderer>().sprite = slingshotSprite;
        slingshot.AddComponent<BoxCollider2D>();
    }

    private static void CreateGround()
    {
        GameObject ground = new GameObject("Ground");
        ground.transform.position = new Vector3(0f, -3f, 0f);
        ground.AddComponent<BoxCollider2D>();
    }

    private static GameObject CreateRectObject(string name, string tag, Sprite sprite)
    {
        GameObject created = new GameObject(name, typeof(RectTransform));
        created.tag = tag;
        created.AddComponent<SpriteRenderer>().sprite = sprite;
        created.AddComponent<BoxCollider2D>();
        return created;
    }

    private static PhysicalNodeSnapshot FindNode(PhysicalSceneSnapshot snapshot, int unityInstanceId)
    {
        return Array.Find(snapshot.Nodes, delegate(PhysicalNodeSnapshot node)
        {
            return node.UnityInstanceId == unityInstanceId;
        });
    }

    private static string[] EntityIds(PhysicalSceneSnapshot snapshot)
    {
        return Array.ConvertAll(snapshot.Nodes, delegate(PhysicalNodeSnapshot node)
        {
            return node.EntityId;
        });
    }
}
