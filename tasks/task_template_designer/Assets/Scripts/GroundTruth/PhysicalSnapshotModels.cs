using UnityEngine;

public sealed class PhysicalSnapshotClock
{
    public long FixedStep { get; private set; }
    public float FixedTime { get; private set; }

    public void ObserveFixedStep(float fixedTime)
    {
        FixedStep++;
        FixedTime = fixedTime;
    }

    public void ResetLevel()
    {
        FixedStep = 0;
        FixedTime = 0f;
    }
}

public sealed class PhysicalBodySnapshot
{
    public bool Present { get; private set; }
    public Vector2? Velocity { get; private set; }
    public float? AngularVelocityDegreesPerSecond { get; private set; }
    public float? MassUnityUnits { get; private set; }
    public float? KineticEnergyUnityUnits { get; private set; }

    public static PhysicalBodySnapshot Absent()
    {
        return new PhysicalBodySnapshot();
    }

    public static PhysicalBodySnapshot Dynamic(Rigidbody2D body)
    {
        Vector2 velocity = body.velocity;
        float mass = body.mass;
        return new PhysicalBodySnapshot
        {
            Present = true,
            Velocity = velocity,
            AngularVelocityDegreesPerSecond = body.angularVelocity,
            MassUnityUnits = mass,
            KineticEnergyUnityUnits = 0.5f * mass * (velocity.x * velocity.x + velocity.y * velocity.y)
        };
    }
}

public sealed class PhysicalNodeSnapshot
{
    public string EntityId { get; private set; }
    public int UnityInstanceId { get; private set; }
    public string ObjectClass { get; private set; }
    public string ObjectType { get; private set; }
    public Vector2[] ScreenPolygon { get; private set; }
    public Vector2 WorldPosition { get; private set; }
    public float RotationDegrees { get; private set; }
    public float? Life { get; private set; }
    public PhysicalBodySnapshot Body { get; private set; }

    public PhysicalNodeSnapshot(
        string entityId,
        int unityInstanceId,
        string objectClass,
        string objectType,
        Vector2[] screenPolygon,
        Vector2 worldPosition,
        float rotationDegrees,
        float? life,
        PhysicalBodySnapshot body)
    {
        EntityId = entityId;
        UnityInstanceId = unityInstanceId;
        ObjectClass = objectClass;
        ObjectType = objectType;
        ScreenPolygon = screenPolygon;
        WorldPosition = worldPosition;
        RotationDegrees = rotationDegrees;
        Life = life;
        Body = body;
    }
}

public sealed class PhysicalSceneSnapshot
{
    public const string SchemaVersion = "physics_capture_v1";

    public int RenderFrame { get; private set; }
    public float RenderTime { get; private set; }
    public long FixedStep { get; private set; }
    public float FixedTime { get; private set; }
    public PhysicalNodeSnapshot[] Nodes { get; private set; }

    public PhysicalSceneSnapshot(
        int renderFrame,
        float renderTime,
        long fixedStep,
        float fixedTime,
        PhysicalNodeSnapshot[] nodes)
    {
        RenderFrame = renderFrame;
        RenderTime = renderTime;
        FixedStep = fixedStep;
        FixedTime = fixedTime;
        Nodes = nodes;
    }

    public string ToJson()
    {
        return PhysicalSnapshotJson.Serialize(this);
    }
}
