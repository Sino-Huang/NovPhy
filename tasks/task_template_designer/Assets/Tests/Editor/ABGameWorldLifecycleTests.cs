using System.Reflection;
using NUnit.Framework;
using UnityEditor.SceneManagement;
using UnityEngine;

public class ABGameWorldLifecycleTests
{
    [SetUp]
    public void SetUp()
    {
        EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
    }

    [TearDown]
    public void TearDown()
    {
        EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
    }

    [Test]
    public void Start_PrePopulatedScene_InitializesSlingshotBaseTransform()
    {
        GameObject blocks = new GameObject("Blocks");
        new GameObject("PrePopulatedBlock").transform.parent = blocks.transform;
        new GameObject("Birds");
        new GameObject("Platforms");
        new GameObject("slingshot_base");
        new GameObject("LevelFailedBanner");
        new GameObject("LevelClearedBanner");

        GameObject cameraObject = new GameObject("Camera");
        cameraObject.AddComponent<Camera>();
        cameraObject.AddComponent<ABGameplayCamera>();

        GameObject worldObject = new GameObject("GameWorld");
        worldObject.SetActive(false);
        worldObject.AddComponent<AudioSource>();
        ABGameWorld world = worldObject.AddComponent<ABGameWorld>();
        world._isSimulation = true;
        worldObject.SetActive(true);

        typeof(ABGameWorld).GetMethod("Awake", BindingFlags.Instance | BindingFlags.NonPublic)
            .Invoke(world, null);
        typeof(ABGameWorld).GetMethod("Start", BindingFlags.Instance | BindingFlags.NonPublic)
            .Invoke(world, null);

        Assert.IsNotNull(world.slingshotBaseTransform);
        Assert.AreEqual("slingshot_base", world.slingshotBaseTransform.name);
    }
}
