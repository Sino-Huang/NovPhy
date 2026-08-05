using System.Collections.Generic;
using System.Globalization;
using SimpleJSON;
using UnityEngine;

public sealed class PhysicalSnapshotExporter
{
    public PhysicalSceneSnapshot Capture(
        SymbolicGameState symbolicState,
        PhysicalEntityRegistry registry,
        PhysicalSnapshotClock clock,
        int renderFrame,
        float renderTime)
    {
        Dictionary<int, GameObject> liveObjects = IndexLiveObjects();
        List<PhysicalNodeSnapshot> nodes = new List<PhysicalNodeSnapshot>();
        JSONNode root = JSONNode.Parse(symbolicState.GetGTJson(false));

        foreach (JSONNode collection in root.Children)
        {
            foreach (JSONNode feature in collection["features"].Children)
            {
                PhysicalNodeSnapshot node = BuildNode(feature, liveObjects, registry);
                if (node != null)
                {
                    nodes.Add(node);
                }
            }
        }

        nodes.Sort(delegate(PhysicalNodeSnapshot left, PhysicalNodeSnapshot right)
        {
            return string.CompareOrdinal(left.EntityId, right.EntityId);
        });

        return new PhysicalSceneSnapshot(
            renderFrame,
            renderTime,
            clock.FixedStep,
            clock.FixedTime,
            nodes.ToArray());
    }

    private static PhysicalNodeSnapshot BuildNode(
        JSONNode feature,
        Dictionary<int, GameObject> liveObjects,
        PhysicalEntityRegistry registry)
    {
        string objectType = feature["properties"]["label"].Value;
        bool isGround = objectType == "Ground";
        if (!isGround && feature["geometry"]["type"].Value != "Polygon")
        {
            return null;
        }

        int legacyInstanceId;
        if (!int.TryParse(
            feature["properties"]["id"].Value,
            NumberStyles.Integer,
            CultureInfo.InvariantCulture,
            out legacyInstanceId))
        {
            return null;
        }

        GameObject liveObject;
        if (!liveObjects.TryGetValue(legacyInstanceId, out liveObject))
        {
            return null;
        }

        Vector2[] polygon = isGround
            ? ReadGroundPolygon(feature["properties"]["yindex"].AsInt)
            : ReadFirstPolygon(feature["geometry"]["coordinates"]);
        if (polygon.Length < 3)
        {
            return null;
        }

        Rigidbody2D rigidBody = liveObject.GetComponent<Rigidbody2D>();
        Collider2D collider = liveObject.GetComponent<Collider2D>();
        bool isStatic = rigidBody == null || rigidBody.bodyType == RigidbodyType2D.Static;
        if (isStatic && collider == null)
        {
            return null;
        }

        int unityInstanceId = isStatic ? collider.GetInstanceID() : liveObject.GetInstanceID();
        string entityId = isStatic
            ? "world:static:" + unityInstanceId.ToString(CultureInfo.InvariantCulture)
            : registry.Register(unityInstanceId, liveObject);

        ABGameObject gameObjectState = liveObject.GetComponent<ABGameObject>();
        float? life = gameObjectState == null ? (float?)null : gameObjectState.getCurrentLife();
        PhysicalBodySnapshot body = rigidBody != null && rigidBody.bodyType == RigidbodyType2D.Dynamic
            ? PhysicalBodySnapshot.Dynamic(rigidBody)
            : PhysicalBodySnapshot.Absent();

        return new PhysicalNodeSnapshot(
            entityId,
            unityInstanceId,
            isGround ? "world" : liveObject.tag,
            objectType,
            polygon,
            new Vector2(liveObject.transform.position.x, liveObject.transform.position.y),
            liveObject.transform.eulerAngles.z,
            life,
            body);
    }

    private static Dictionary<int, GameObject> IndexLiveObjects()
    {
        GameObject[] objects = UnityEngine.Object.FindObjectsOfType<GameObject>();
        Dictionary<int, GameObject> indexed = new Dictionary<int, GameObject>(objects.Length);
        foreach (GameObject gameObject in objects)
        {
            indexed[gameObject.GetInstanceID()] = gameObject;
        }
        return indexed;
    }

    private static Vector2[] ReadFirstPolygon(JSONNode coordinates)
    {
        List<Vector2> points = new List<Vector2>();
        JSONNode firstPath = coordinates[0];
        foreach (JSONNode point in firstPath.Children)
        {
            points.Add(new Vector2(point[0].AsFloat, point[1].AsFloat));
        }
        return points.ToArray();
    }

    private static Vector2[] ReadGroundPolygon(int yindex)
    {
        return new[]
        {
            new Vector2(0f, yindex),
            new Vector2(0f, Screen.height),
            new Vector2(Screen.width, Screen.height),
            new Vector2(Screen.width, yindex)
        };
    }
}
