using System.Collections.Generic;
using System.Globalization;
using UnityEngine;

public sealed class PhysicalEntityRegistry
{
    private sealed class LifetimeEntry
    {
        public readonly Object Lifetime;
        public readonly int Ordinal;

        public LifetimeEntry(Object lifetime, int ordinal)
        {
            Lifetime = lifetime;
            Ordinal = ordinal;
        }
    }

    private readonly Dictionary<int, LifetimeEntry> currentLifetimes = new Dictionary<int, LifetimeEntry>();
    private readonly Dictionary<int, int> nextOrdinals = new Dictionary<int, int>();

    public string Register(int unityInstanceId, Object lifetime)
    {
        LifetimeEntry current;
        if (currentLifetimes.TryGetValue(unityInstanceId, out current)
            && ReferenceEquals(current.Lifetime, lifetime))
        {
            return FormatEntityId(unityInstanceId, current.Ordinal);
        }

        int ordinal;
        if (!nextOrdinals.TryGetValue(unityInstanceId, out ordinal))
        {
            ordinal = 0;
        }

        currentLifetimes[unityInstanceId] = new LifetimeEntry(lifetime, ordinal);
        nextOrdinals[unityInstanceId] = ordinal + 1;
        return FormatEntityId(unityInstanceId, ordinal);
    }

    public string RegisterObject(GameObject gameObject)
    {
        if (gameObject == null)
            return null;
        Rigidbody2D body = gameObject.GetComponent<Rigidbody2D>();
        if (body == null || body.bodyType == RigidbodyType2D.Static)
        {
            Collider2D collider = gameObject.GetComponent<Collider2D>();
            return collider == null ? null : "world:static:" + collider.GetInstanceID().ToString(CultureInfo.InvariantCulture);
        }
        return Register(gameObject.GetInstanceID(), gameObject);
    }

    public string RegisterCollider(Collider2D collider)
    {
        if (collider == null)
            return null;
        Rigidbody2D body = collider.attachedRigidbody;
        if (body == null || body.bodyType == RigidbodyType2D.Static)
            return "world:static:" + collider.GetInstanceID().ToString(CultureInfo.InvariantCulture);
        return Register(body.gameObject.GetInstanceID(), body.gameObject);
    }

    public void ResetLevel()
    {
        currentLifetimes.Clear();
        nextOrdinals.Clear();
    }

    private static string FormatEntityId(int unityInstanceId, int ordinal)
    {
        return unityInstanceId.ToString(CultureInfo.InvariantCulture)
            + ":"
            + ordinal.ToString(CultureInfo.InvariantCulture);
    }
}
