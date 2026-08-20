using System;
using System.Globalization;
using UnityEngine;

public sealed class ScenarioObjectIdentity : MonoBehaviour
{
    [SerializeField]
    private string scenarioObjectId;

    public string ScenarioObjectId { get { return scenarioObjectId; } }

    public static void Assign(GameObject target, string value)
    {
        if (target == null || string.IsNullOrEmpty(value))
            return;
        ScenarioObjectIdentity identity = target.GetComponent<ScenarioObjectIdentity>();
        if (identity == null)
            identity = target.AddComponent<ScenarioObjectIdentity>();
        identity.scenarioObjectId = value;
    }

    public static void AssignSpawn(
        GameObject target,
        ScenarioObjectIdentity parent,
        string kind,
        int ordinal)
    {
        if (parent == null || string.IsNullOrEmpty(parent.ScenarioObjectId)
            || string.IsNullOrEmpty(kind) || ordinal < 0)
            throw new ArgumentException("Causal spawn identity requires parent, kind, and nonnegative ordinal.");
        Assign(
            target,
            parent.ScenarioObjectId + "/spawn:" + kind + ":"
                + ordinal.ToString("D4", CultureInfo.InvariantCulture));
    }
}
