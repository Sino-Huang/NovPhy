using NUnit.Framework;
using UnityEngine;

public sealed class ScenarioObjectIdentityTests
{
    [Test]
    public void LevelLoaderAndRuntimeObjectRetainAuthoredScenarioObjectIds()
    {
        const string xml =
            "<Level width=\"1\">\n<Camera x=\"0\" y=\"0\" minWidth=\"20\" maxWidth=\"30\"/>\n" +
            "<Score highScore=\"0\"/>\n<Birds>\n<Bird type=\"BirdRed\" scenarioObjectId=\"bird:0000\"/>\n</Birds>\n" +
            "<Slingshot x=\"-8\" y=\"-2\" scenarioObjectId=\"slingshot:0000\"/>\n" +
            "<GameObjects>\n" +
            "<Pig type=\"BasicSmall\" x=\"1\" y=\"-3\" scenarioObjectId=\"pig:0000\"/>\n" +
            "<Block type=\"SquareSmall\" material=\"wood\" x=\"2\" y=\"-3\" scenarioObjectId=\"block:0000\"/>\n" +
            "<Platform type=\"Platform\" x=\"3\" y=\"-3\" scenarioObjectId=\"platform:0000\"/>\n" +
            "<TNT type=\"TNT\" x=\"4\" y=\"-3\" scenarioObjectId=\"tnt:0000\"/>\n" +
            "</GameObjects>\n</Level>";

        ABLevel level = LevelLoader.LoadXmlLevel(xml);

        Assert.AreEqual("slingshot:0000", level.slingshot.scenarioObjectId);
        Assert.AreEqual("bird:0000", level.birds[0].scenarioObjectId);
        Assert.AreEqual("pig:0000", level.pigs[0].scenarioObjectId);
        Assert.AreEqual("block:0000", level.blocks[0].scenarioObjectId);
        Assert.AreEqual("platform:0000", level.platforms[0].scenarioObjectId);
        Assert.AreEqual("tnt:0000", level.tnts[0].scenarioObjectId);

        GameObject instantiated = new GameObject("authored-object");
        GameObject spawned = new GameObject("spawned-object");
        try
        {
            ScenarioObjectIdentity.Assign(instantiated, level.pigs[0].scenarioObjectId);
            Assert.AreEqual(
                "pig:0000",
                instantiated.GetComponent<ScenarioObjectIdentity>().ScenarioObjectId);
            ScenarioObjectIdentity.AssignSpawn(
                spawned,
                instantiated.GetComponent<ScenarioObjectIdentity>(),
                "fragment",
                2);
            Assert.AreEqual(
                "pig:0000/spawn:fragment:0002",
                spawned.GetComponent<ScenarioObjectIdentity>().ScenarioObjectId);
        }
        finally
        {
            Object.DestroyImmediate(instantiated);
            Object.DestroyImmediate(spawned);
        }
    }
}
