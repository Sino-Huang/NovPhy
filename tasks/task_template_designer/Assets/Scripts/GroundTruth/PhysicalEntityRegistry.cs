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
