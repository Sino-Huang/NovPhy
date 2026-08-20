using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using UnityEngine;

public sealed class PhysicsCaptureV2CaptureLimits
{
    public static readonly PhysicsCaptureV2CaptureLimits Default =
        new PhysicsCaptureV2CaptureLimits(1024, 4096, 32768);
    public int MaxEntitiesPerStep { get; private set; }
    public int MaxCollidersPerStep { get; private set; }
    public int MaxContactsPerStep { get; private set; }

    public PhysicsCaptureV2CaptureLimits(int maxEntitiesPerStep, int maxCollidersPerStep)
        : this(maxEntitiesPerStep, maxCollidersPerStep, 32768)
    {
    }

    public PhysicsCaptureV2CaptureLimits(int maxEntitiesPerStep, int maxCollidersPerStep,
        int maxContactsPerStep)
    {
        MaxEntitiesPerStep = maxEntitiesPerStep;
        MaxCollidersPerStep = maxCollidersPerStep;
        MaxContactsPerStep = maxContactsPerStep;
    }
}

public sealed class PhysicsCaptureV2RecorderFailure
{
    public PhysicsCaptureV2EngineFailureCode Code { get; private set; }
    public string Message { get; private set; }

    public PhysicsCaptureV2RecorderFailure(PhysicsCaptureV2EngineFailureCode code, string message)
    {
        Code = code;
        Message = message;
    }
}

public sealed class PhysicsCaptureV2BodySnapshot
{
    public string BodyType { get; private set; }
    public bool Simulated { get; private set; }
    public float GravityScale { get; private set; }
    public bool GravityApplicable { get; private set; }
    public Vector2 LinearVelocity { get; private set; }
    public float AngularVelocity { get; private set; }

    public PhysicsCaptureV2BodySnapshot(string bodyType, bool simulated, float gravityScale,
        bool gravityApplicable, Vector2 linearVelocity, float angularVelocity)
    {
        BodyType = bodyType;
        Simulated = simulated;
        GravityScale = gravityScale;
        GravityApplicable = gravityApplicable;
        LinearVelocity = linearVelocity;
        AngularVelocity = angularVelocity;
    }
}

public sealed class PhysicsCaptureV2EntitySnapshot
{
    public string EntityId { get; private set; }
    public string ScenarioObjectId { get; private set; }
    public string Lifecycle { get; private set; }
    public Vector2 Position { get; private set; }
    public float RotationDegrees { get; private set; }
    public PhysicsCaptureV2BodySnapshot Body { get; private set; }
    public ReadOnlyCollection<string> ContactIds { get; private set; }
    public ReadOnlyCollection<string> SupportedByEntityIds { get; private set; }
    public ReadOnlyCollection<string> SupportsEntityIds { get; private set; }

    public PhysicsCaptureV2EntitySnapshot(string entityId, string scenarioObjectId,
        string lifecycle, Vector2 position, float rotationDegrees, PhysicsCaptureV2BodySnapshot body)
    {
        EntityId = entityId;
        ScenarioObjectId = scenarioObjectId;
        Lifecycle = lifecycle;
        Position = position;
        RotationDegrees = rotationDegrees;
        Body = body;
        ContactIds = new List<string>().AsReadOnly();
        SupportedByEntityIds = new List<string>().AsReadOnly();
        SupportsEntityIds = new List<string>().AsReadOnly();
    }

    public PhysicsCaptureV2EntitySnapshot WithContext(IList<string> contactIds,
        IList<string> supportedByEntityIds, IList<string> supportsEntityIds)
    {
        PhysicsCaptureV2EntitySnapshot value = new PhysicsCaptureV2EntitySnapshot(
            EntityId, ScenarioObjectId, Lifecycle, Position, RotationDegrees, Body);
        value.ContactIds = new List<string>(contactIds).AsReadOnly();
        value.SupportedByEntityIds = new List<string>(supportedByEntityIds).AsReadOnly();
        value.SupportsEntityIds = new List<string>(supportsEntityIds).AsReadOnly();
        return value;
    }

    public PhysicsCaptureV2EntitySnapshot WithLifecycle(string lifecycle)
    {
        return new PhysicsCaptureV2EntitySnapshot(EntityId, ScenarioObjectId, lifecycle,
            Position, RotationDegrees, lifecycle == "destroyed" ? null : Body);
    }
}

public sealed class PhysicsCaptureV2ColliderSnapshot
{
    public string ColliderId { get; private set; }
    public string EntityId { get; private set; }
    public bool Enabled { get; private set; }
    public bool IsTrigger { get; private set; }
    public string Kind { get; private set; }
    public Vector2 Center { get; private set; }
    public Vector2 Size { get; private set; }
    public float Radius { get; private set; }
    public string Direction { get; private set; }
    public float AngleDegrees { get; private set; }
    public ReadOnlyCollection<Vector2> Points { get; private set; }
    public ReadOnlyCollection<ReadOnlyCollection<Vector2>> Paths { get; private set; }

    public PhysicsCaptureV2ColliderSnapshot(string colliderId, string entityId, bool enabled,
        bool isTrigger, string kind, Vector2 center, Vector2 size, float radius, string direction,
        float angleDegrees, IList<Vector2> points, IList<IList<Vector2>> paths)
    {
        ColliderId = colliderId;
        EntityId = entityId;
        Enabled = enabled;
        IsTrigger = isTrigger;
        Kind = kind;
        Center = center;
        Size = size;
        Radius = radius;
        Direction = direction;
        AngleDegrees = angleDegrees;
        Points = new List<Vector2>(points ?? new Vector2[0]).AsReadOnly();
        List<ReadOnlyCollection<Vector2>> frozenPaths = new List<ReadOnlyCollection<Vector2>>();
        if (paths != null)
            for (int i = 0; i < paths.Count; i++)
                frozenPaths.Add(new List<Vector2>(paths[i]).AsReadOnly());
        Paths = frozenPaths.AsReadOnly();
    }

    public PhysicsCaptureV2ColliderSnapshot WithEnabled(bool enabled)
    {
        List<IList<Vector2>> paths = new List<IList<Vector2>>();
        for (int i = 0; i < Paths.Count; i++) paths.Add(Paths[i]);
        return new PhysicsCaptureV2ColliderSnapshot(ColliderId, EntityId, enabled, IsTrigger,
            Kind, Center, Size, Radius, Direction, AngleDegrees, Points, paths);
    }
}

public static class PhysicsCaptureV2EntityGeometryExporter
{
    public static bool TryValidateFinite(PhysicsCaptureV2EngineSnapshot snapshot,
        out PhysicsCaptureV2RecorderFailure failure)
    {
        failure = null;
        for (int sampleIndex = 0; sampleIndex < snapshot.FixedStepSamples.Count; sampleIndex++)
        {
            PhysicsCaptureV2FixedStepSample sample = snapshot.FixedStepSamples[sampleIndex];
            for (int entityIndex = 0; entityIndex < sample.Entities.Count; entityIndex++)
            {
                PhysicsCaptureV2EntitySnapshot entity = sample.Entities[entityIndex];
                if (!Finite(entity.Position) || !Finite(entity.RotationDegrees)
                    || (entity.Body != null && (!Finite(entity.Body.GravityScale)
                        || !Finite(entity.Body.LinearVelocity)
                        || !Finite(entity.Body.AngularVelocity))))
                    return Fail(PhysicsCaptureV2EngineFailureCode.NonFiniteValue,
                        "physics capture v2 snapshot contains a non-finite entity value", out failure);
            }
            for (int colliderIndex = 0; colliderIndex < sample.Colliders.Count; colliderIndex++)
            {
                PhysicsCaptureV2ColliderSnapshot collider = sample.Colliders[colliderIndex];
                if (!Finite(collider.Center) || !Finite(collider.Size) || !Finite(collider.Radius)
                    || !Finite(collider.AngleDegrees))
                    return Fail(PhysicsCaptureV2EngineFailureCode.NonFiniteValue,
                        "physics capture v2 snapshot contains a non-finite collider value", out failure);
                for (int point = 0; point < collider.Points.Count; point++)
                    if (!Finite(collider.Points[point]))
                        return Fail(PhysicsCaptureV2EngineFailureCode.NonFiniteValue,
                            "physics capture v2 snapshot contains a non-finite collider point", out failure);
                for (int path = 0; path < collider.Paths.Count; path++)
                    for (int point = 0; point < collider.Paths[path].Count; point++)
                        if (!Finite(collider.Paths[path][point]))
                            return Fail(PhysicsCaptureV2EngineFailureCode.NonFiniteValue,
                                "physics capture v2 snapshot contains a non-finite collider path", out failure);
            }
        }
        return true;
    }

    public static bool TryCapture(GameObject[] causalObjects, PhysicsCaptureV2CaptureLimits limits,
        out List<PhysicsCaptureV2EntitySnapshot> entities,
        out List<PhysicsCaptureV2ColliderSnapshot> colliders,
        out PhysicsCaptureV2RecorderFailure failure)
    {
        entities = new List<PhysicsCaptureV2EntitySnapshot>();
        colliders = new List<PhysicsCaptureV2ColliderSnapshot>();
        failure = null;
        causalObjects = causalObjects ?? new GameObject[0];
        if (causalObjects.Length > limits.MaxEntitiesPerStep)
            return Fail(PhysicsCaptureV2EngineFailureCode.EntityLimitExceeded,
                "physics capture v2 entity bound exceeded", out failure);
        HashSet<string> scenarioIds = new HashSet<string>(StringComparer.Ordinal);
        for (int i = 0; i < causalObjects.Length; i++)
        {
            GameObject causalObject = causalObjects[i];
            ScenarioObjectIdentity identity = causalObject == null
                ? null : causalObject.GetComponent<ScenarioObjectIdentity>();
            if (identity == null || string.IsNullOrEmpty(identity.ScenarioObjectId)
                || !scenarioIds.Add(identity.ScenarioObjectId))
                return Fail(PhysicsCaptureV2EngineFailureCode.MissingScenarioObjectIdentity,
                    "causal object has no unique authored scenario object identity", out failure);
            string entityId = "runtime:" + identity.ScenarioObjectId;
            Vector2 position = causalObject.transform.position;
            float rotation = causalObject.transform.eulerAngles.z;
            Rigidbody2D body = causalObject.GetComponent<Rigidbody2D>();
            PhysicsCaptureV2BodySnapshot bodySnapshot = body == null ? null
                : new PhysicsCaptureV2BodySnapshot(BodyType(body.bodyType), body.simulated,
                    body.gravityScale, body.bodyType == RigidbodyType2D.Dynamic && body.simulated
                        && body.gravityScale != 0f,
                    body.velocity, body.angularVelocity);
            if (!Finite(position) || !Finite(rotation)
                || (bodySnapshot != null && (!Finite(bodySnapshot.GravityScale)
                    || !Finite(bodySnapshot.LinearVelocity) || !Finite(bodySnapshot.AngularVelocity))))
                return Fail(PhysicsCaptureV2EngineFailureCode.NonFiniteValue,
                    "physics capture v2 entity contains a non-finite value", out failure);
            entities.Add(new PhysicsCaptureV2EntitySnapshot(entityId, identity.ScenarioObjectId,
                causalObject.activeInHierarchy ? "active" : "inactive", position, rotation, bodySnapshot));
            Collider2D[] objectColliders = causalObject.GetComponentsInChildren<Collider2D>(true);
            if (colliders.Count + objectColliders.Length > limits.MaxCollidersPerStep)
                return Fail(PhysicsCaptureV2EngineFailureCode.ColliderLimitExceeded,
                    "physics capture v2 collider bound exceeded", out failure);
            for (int colliderIndex = 0; colliderIndex < objectColliders.Length; colliderIndex++)
            {
                Collider2D collider = objectColliders[colliderIndex];
                BoxCollider2D box = collider as BoxCollider2D;
                string colliderId = entityId + ":collider:" + colliderIndex.ToString("D4");
                if (box != null)
                {
                    Vector2 scale = Abs(box.transform.lossyScale);
                    Vector2 size = Vector2.Scale(box.size, scale);
                    Vector2 center = box.transform.TransformPoint(box.offset);
                    float angle = box.transform.eulerAngles.z;
                    if (!PositiveFinite(size) || !Finite(center) || !Finite(angle))
                        return Fail(PhysicsCaptureV2EngineFailureCode.IncompleteColliderGeometry,
                            "physics capture v2 box geometry is incomplete", out failure);
                    colliders.Add(new PhysicsCaptureV2ColliderSnapshot(colliderId, entityId,
                        collider.enabled, collider.isTrigger, "box", center, size, 0f, null,
                        angle, null, null));
                    continue;
                }
                CircleCollider2D circle = collider as CircleCollider2D;
                if (circle != null)
                {
                    Vector2 scale = Abs(circle.transform.lossyScale);
                    Vector2 center = circle.transform.TransformPoint(circle.offset);
                    float radius = circle.radius * scale.x;
                    if (!Mathf.Approximately(scale.x, scale.y) || !Finite(center)
                        || !Finite(radius) || radius <= 0f)
                        return Fail(PhysicsCaptureV2EngineFailureCode.IncompleteColliderGeometry,
                            "physics capture v2 circle geometry is incomplete", out failure);
                    colliders.Add(new PhysicsCaptureV2ColliderSnapshot(colliderId, entityId,
                        collider.enabled, collider.isTrigger, "circle", center, Vector2.zero,
                        radius, null, 0f, null, null));
                    continue;
                }
                PolygonCollider2D polygon = collider as PolygonCollider2D;
                if (polygon != null)
                {
                    List<IList<Vector2>> paths = new List<IList<Vector2>>();
                    for (int pathIndex = 0; pathIndex < polygon.pathCount; pathIndex++)
                    {
                        Vector2[] localPath = polygon.GetPath(pathIndex);
                        if (localPath.Length < 3)
                            return Fail(PhysicsCaptureV2EngineFailureCode.IncompleteColliderGeometry,
                                "physics capture v2 polygon geometry is incomplete", out failure);
                        List<Vector2> worldPath = TransformPoints(polygon.transform, localPath);
                        if (worldPath == null)
                            return Fail(PhysicsCaptureV2EngineFailureCode.NonFiniteValue,
                                "physics capture v2 polygon contains a non-finite value", out failure);
                        paths.Add(worldPath);
                    }
                    if (paths.Count == 0)
                        return Fail(PhysicsCaptureV2EngineFailureCode.IncompleteColliderGeometry,
                            "physics capture v2 polygon geometry is incomplete", out failure);
                    colliders.Add(new PhysicsCaptureV2ColliderSnapshot(colliderId, entityId,
                        collider.enabled, collider.isTrigger, "polygon", Vector2.zero,
                        Vector2.zero, 0f, null, 0f, null, paths));
                    continue;
                }
                EdgeCollider2D edge = collider as EdgeCollider2D;
                if (edge != null)
                {
                    List<Vector2> points = TransformPoints(edge.transform, edge.points);
                    if (points == null)
                        return Fail(PhysicsCaptureV2EngineFailureCode.NonFiniteValue,
                            "physics capture v2 edge contains a non-finite value", out failure);
                    if (points.Count < 2)
                        return Fail(PhysicsCaptureV2EngineFailureCode.IncompleteColliderGeometry,
                            "physics capture v2 edge geometry is incomplete", out failure);
                    colliders.Add(new PhysicsCaptureV2ColliderSnapshot(colliderId, entityId,
                        collider.enabled, collider.isTrigger, "edge", Vector2.zero, Vector2.zero,
                        0f, null, 0f, points, null));
                    continue;
                }
                CapsuleCollider2D capsule = collider as CapsuleCollider2D;
                if (capsule != null)
                {
                    Vector2 size = Vector2.Scale(capsule.size, Abs(capsule.transform.lossyScale));
                    Vector2 center = capsule.transform.TransformPoint(capsule.offset);
                    float angle = capsule.transform.eulerAngles.z;
                    if (!PositiveFinite(size) || !Finite(center) || !Finite(angle))
                        return Fail(PhysicsCaptureV2EngineFailureCode.IncompleteColliderGeometry,
                            "physics capture v2 capsule geometry is incomplete", out failure);
                    string direction = capsule.direction == CapsuleDirection2D.Vertical
                        ? "vertical" : "horizontal";
                    colliders.Add(new PhysicsCaptureV2ColliderSnapshot(colliderId, entityId,
                        collider.enabled, collider.isTrigger, "capsule", center, size, 0f,
                        direction, angle, null, null));
                    continue;
                }
                return Fail(PhysicsCaptureV2EngineFailureCode.UnsupportedColliderGeometry,
                    "physics capture v2 collider type is unsupported: " + collider.GetType().Name,
                    out failure);
            }
        }
        entities.Sort(delegate(PhysicsCaptureV2EntitySnapshot left,
            PhysicsCaptureV2EntitySnapshot right)
        {
            return string.CompareOrdinal(left.EntityId, right.EntityId);
        });
        colliders.Sort(delegate(PhysicsCaptureV2ColliderSnapshot left,
            PhysicsCaptureV2ColliderSnapshot right)
        {
            return string.CompareOrdinal(left.ColliderId, right.ColliderId);
        });
        return true;
    }

    private static string BodyType(RigidbodyType2D bodyType)
    {
        if (bodyType == RigidbodyType2D.Dynamic) return "dynamic";
        if (bodyType == RigidbodyType2D.Kinematic) return "kinematic";
        return "static";
    }

    private static bool Fail(PhysicsCaptureV2EngineFailureCode code, string message,
        out PhysicsCaptureV2RecorderFailure failure)
    {
        failure = new PhysicsCaptureV2RecorderFailure(code, message);
        return false;
    }

    private static Vector2 Abs(Vector3 value)
    {
        return new Vector2(Mathf.Abs(value.x), Mathf.Abs(value.y));
    }

    private static List<Vector2> TransformPoints(Transform transform, Vector2[] localPoints)
    {
        List<Vector2> world = new List<Vector2>();
        for (int i = 0; i < localPoints.Length; i++)
        {
            Vector2 point = transform.TransformPoint(localPoints[i]);
            if (!Finite(point)) return null;
            world.Add(point);
        }
        return world;
    }

    private static bool PositiveFinite(Vector2 value)
    {
        return value.x > 0f && value.y > 0f && Finite(value);
    }

    private static bool Finite(Vector2 value)
    {
        return Finite(value.x) && Finite(value.y);
    }

    private static bool Finite(float value)
    {
        return !float.IsNaN(value) && !float.IsInfinity(value);
    }
}
