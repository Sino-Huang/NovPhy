using System;
using UnityEngine;

public static class CanonicalCaptureSeed
{
    [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.BeforeSceneLoad)]
    private static void BindEnvironmentSeed()
    {
        string value = Environment.GetEnvironmentVariable("NOVPHY_ENVIRONMENT_SEED");
        if (string.IsNullOrEmpty(value)) return;
        int seed = int.Parse(value, System.Globalization.CultureInfo.InvariantCulture);
        UnityEngine.Random.InitState(seed);
        Debug.Log("[canonical-capture] Unity RNG seed=" + seed);
    }
}
