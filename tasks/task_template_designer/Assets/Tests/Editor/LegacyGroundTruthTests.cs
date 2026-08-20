using NUnit.Framework;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;

public class LegacyGroundTruthTests
{
    private int originalNoveltyType;
    private Texture2D texture;
    private Sprite sprite;

    [SetUp]
    public void SetUp()
    {
        originalNoveltyType = ABGameWorld.noveltyTypeForNovelty3;
        EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
        ABGameWorld.noveltyTypeForNovelty3 = -1;

        GameObject schemaObject = new GameObject("LoadLevelSchema");
        schemaObject.AddComponent<LoadLevelSchema>();

        GameObject cameraObject = new GameObject("Main Camera");
        cameraObject.tag = "MainCamera";
        Camera camera = cameraObject.AddComponent<Camera>();
        camera.orthographic = true;
        cameraObject.transform.position = new Vector3(0f, 0f, -10f);

        texture = new Texture2D(2, 2);
        sprite = Sprite.Create(texture, new Rect(0f, 0f, 2f, 2f), new Vector2(0.5f, 0.5f), 1f);

        GameObject slingshotBack = new GameObject("slingshot_back");
        slingshotBack.tag = "Slingshot";
        slingshotBack.AddComponent<SpriteRenderer>().sprite = sprite;
    }

    [TearDown]
    public void TearDown()
    {
        ABGameWorld.noveltyTypeForNovelty3 = originalNoveltyType;
        UnityEngine.Object.DestroyImmediate(sprite);
        UnityEngine.Object.DestroyImmediate(texture);
        EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
    }

    [Test]
    public void GetGTJson_MinimalFixture_MatchesExactLegacyBytes()
    {
        GameObject slingshotBack = GameObject.Find("slingshot_back");
        Camera camera = Camera.main;
        Bounds bounds = slingshotBack.GetComponent<Renderer>().bounds;
        float xmax = Mathf.Round(camera.WorldToScreenPoint(bounds.max).x);
        float ymin = Mathf.Round(camera.WorldToScreenPoint(bounds.min).y);
        float lowerY = Screen.height - ymin;

        string expected = "[{\"type\": \"FeatureCollection\",\"features\": [{\"type\": \"Feature\",\"geometry\":{"
            + "\"type\":\"Polygon\",\"coordinates\":[[[0," + lowerY + "],[0," + Screen.height + "],[" + xmax + ","
            + Screen.height + "],[" + xmax + "," + lowerY + "]]]},\"properties\":{\"id\":\""
            + slingshotBack.GetInstanceID() + "\",\"label\":\"Slingshot\",\"colormap\":[]}}]}]";

        string actual = new SymbolicGameState(false).GetGTJson(false);

        Assert.AreEqual(expected, actual);
        Debug.Log("LEGACY_GT_LENGTH=" + actual.Length + ";PREVIEW="
            + actual.Substring(0, Mathf.Min(actual.Length, 80)));
    }

    [Test]
    public void GetGTJson_PigWithoutPolygonCollider_PreservesResponse()
    {
        GameObject pig = new GameObject("BasicSmall(Clone)");
        pig.tag = "PigSmall";
        pig.AddComponent<SpriteRenderer>().sprite = sprite;
        pig.AddComponent<CircleCollider2D>();
        pig.AddComponent<Rigidbody2D>();
        pig.AddComponent<ABGameObject>();

        string json = new SymbolicGameState(false).GetGTJson(false);

        StringAssert.Contains("\"label\":\"Slingshot\"", json);
        StringAssert.DoesNotContain(pig.GetInstanceID().ToString(), json);
    }

    [Test]
    public void GetGTJson_BirdWithoutPolygonCollider_PreservesResponse()
    {
        GameObject bird = new GameObject("BirdBlack(Clone)");
        bird.tag = "Bird";
        bird.AddComponent<SpriteRenderer>().sprite = sprite;
        bird.AddComponent<CircleCollider2D>();
        bird.AddComponent<Rigidbody2D>();
        bird.AddComponent<ABGameObject>();

        string json = new SymbolicGameState(false).GetGTJson(false);

        StringAssert.Contains("\"label\":\"Slingshot\"", json);
        StringAssert.DoesNotContain(bird.GetInstanceID().ToString(), json);
    }
}
