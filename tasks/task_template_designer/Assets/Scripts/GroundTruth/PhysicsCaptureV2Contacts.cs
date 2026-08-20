using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using UnityEngine;

public sealed class PhysicsCaptureV2ContactInput
{
    public Collider2D ColliderA { get; private set; }
    public Collider2D ColliderB { get; private set; }
    public Vector2 Point { get; private set; }
    public Vector2 NormalAToB { get; private set; }
    public float Separation { get; private set; }

    public PhysicsCaptureV2ContactInput(Collider2D colliderA, Collider2D colliderB,
        Vector2 point, Vector2 normalAToB, float separation)
    {
        ColliderA = colliderA;
        ColliderB = colliderB;
        Point = point;
        NormalAToB = normalAToB;
        Separation = separation;
    }
}

public sealed class PhysicsCaptureV2ContactSnapshot
{
    public string ContactId { get; private set; }
    public string EntityAId { get; private set; }
    public string EntityBId { get; private set; }
    public string ColliderAId { get; private set; }
    public string ColliderBId { get; private set; }
    public Vector2 Point { get; private set; }
    public Vector2 NormalAToB { get; private set; }
    public float Separation { get; private set; }

    public PhysicsCaptureV2ContactSnapshot(string contactId, string entityAId, string entityBId,
        string colliderAId, string colliderBId, Vector2 point, Vector2 normalAToB, float separation)
    {
        ContactId = contactId;
        EntityAId = entityAId;
        EntityBId = entityBId;
        ColliderAId = colliderAId;
        ColliderBId = colliderBId;
        Point = point;
        NormalAToB = normalAToB;
        Separation = separation;
    }
}

public sealed class PhysicsCaptureV2SupportSnapshot
{
    public string SupporterEntityId { get; private set; }
    public string SupportedEntityId { get; private set; }
    public ReadOnlyCollection<string> ContactIds { get; private set; }

    public PhysicsCaptureV2SupportSnapshot(string supporterEntityId, string supportedEntityId,
        IList<string> contactIds)
    {
        SupporterEntityId = supporterEntityId;
        SupportedEntityId = supportedEntityId;
        ContactIds = new List<string>(contactIds).AsReadOnly();
    }
}

public static class PhysicsCaptureV2ContactExporter
{
    public static bool TryCapture(long fixedStep, GameObject[] causalObjects,
        PhysicsCaptureV2ContactInput[] inputs, bool complete,
        PhysicsCaptureV2CaptureLimits limits, List<PhysicsCaptureV2EntitySnapshot> entities,
        List<PhysicsCaptureV2ColliderSnapshot> colliders,
        out List<PhysicsCaptureV2ContactSnapshot> contacts,
        out List<PhysicsCaptureV2SupportSnapshot> supports,
        out List<PhysicsCaptureV2EntitySnapshot> contextualEntities,
        out PhysicsCaptureV2RecorderFailure failure)
    {
        contacts = new List<PhysicsCaptureV2ContactSnapshot>();
        supports = new List<PhysicsCaptureV2SupportSnapshot>();
        contextualEntities = entities;
        failure = null;
        if (!complete)
            return Fail(PhysicsCaptureV2EngineFailureCode.IncompleteContactEnumeration,
                "physics capture v2 contact enumeration is incomplete", out failure);
        inputs = inputs ?? new PhysicsCaptureV2ContactInput[0];
        List<PhysicsCaptureV2ContactSnapshot> unordered = new List<PhysicsCaptureV2ContactSnapshot>();
        for (int i = 0; i < inputs.Length; i++)
        {
            PhysicsCaptureV2ContactInput input = inputs[i];
            if (input == null || input.ColliderA == null || input.ColliderB == null)
                return Fail(PhysicsCaptureV2EngineFailureCode.UnresolvedContactIdentity,
                    "physics capture v2 contact has unresolved colliders", out failure);
            if (input.ColliderA.isTrigger || input.ColliderB.isTrigger
                || !input.ColliderA.enabled || !input.ColliderB.enabled
                || !input.ColliderA.gameObject.activeInHierarchy
                || !input.ColliderB.gameObject.activeInHierarchy)
                continue;
            string entityA; string colliderA; string entityB; string colliderB;
            if (!Resolve(input.ColliderA, causalObjects, colliders, out entityA, out colliderA)
                || !Resolve(input.ColliderB, causalObjects, colliders, out entityB, out colliderB))
                return Fail(PhysicsCaptureV2EngineFailureCode.UnresolvedContactIdentity,
                    "physics capture v2 contact does not resolve to retained identity and geometry",
                    out failure);
            if (!Finite(input.Point) || !Finite(input.NormalAToB) || !Finite(input.Separation))
                return Fail(PhysicsCaptureV2EngineFailureCode.NonFiniteValue,
                    "physics capture v2 contact contains a non-finite value", out failure);
            Vector2 normal = input.NormalAToB;
            if (string.CompareOrdinal(colliderA, colliderB) > 0)
            {
                Swap(ref entityA, ref entityB); Swap(ref colliderA, ref colliderB); normal = -normal;
            }
            unordered.Add(new PhysicsCaptureV2ContactSnapshot(null, entityA, entityB,
                colliderA, colliderB, input.Point, normal, input.Separation));
        }
        if (unordered.Count > limits.MaxContactsPerStep)
            return Fail(PhysicsCaptureV2EngineFailureCode.ContactLimitExceeded,
                "physics capture v2 per-step contact bound exceeded", out failure);
        unordered.Sort(CompareContacts);
        for (int i = 0; i < unordered.Count; i++)
        {
            PhysicsCaptureV2ContactSnapshot contact = unordered[i];
            contacts.Add(new PhysicsCaptureV2ContactSnapshot(
                "contact:" + fixedStep + ":" + i.ToString("D4"),
                contact.EntityAId, contact.EntityBId, contact.ColliderAId, contact.ColliderBId,
                contact.Point, contact.NormalAToB, contact.Separation));
        }
        BuildSupportAndContext(contacts, entities, out supports, out contextualEntities);
        return true;
    }

    private static void BuildSupportAndContext(List<PhysicsCaptureV2ContactSnapshot> contacts,
        List<PhysicsCaptureV2EntitySnapshot> entities,
        out List<PhysicsCaptureV2SupportSnapshot> supports,
        out List<PhysicsCaptureV2EntitySnapshot> contextualEntities)
    {
        SortedDictionary<string, List<string>> supportContacts =
            new SortedDictionary<string, List<string>>(StringComparer.Ordinal);
        Dictionary<string, SortedSet<string>> contactIds = NewSets(entities);
        Dictionary<string, SortedSet<string>> supportedBy = NewSets(entities);
        Dictionary<string, SortedSet<string>> supportsIds = NewSets(entities);
        for (int i = 0; i < contacts.Count; i++)
        {
            PhysicsCaptureV2ContactSnapshot contact = contacts[i];
            contactIds[contact.EntityAId].Add(contact.ContactId);
            contactIds[contact.EntityBId].Add(contact.ContactId);
            string supporter = null; string supported = null;
            if (contact.NormalAToB.y > 0.5f) { supporter = contact.EntityAId; supported = contact.EntityBId; }
            else if (contact.NormalAToB.y < -0.5f) { supporter = contact.EntityBId; supported = contact.EntityAId; }
            if (supporter == null || supporter == supported) continue;
            string key = supporter + "\n" + supported;
            List<string> ids;
            if (!supportContacts.TryGetValue(key, out ids))
            {
                ids = new List<string>(); supportContacts.Add(key, ids);
            }
            ids.Add(contact.ContactId);
            supportedBy[supported].Add(supporter);
            supportsIds[supporter].Add(supported);
        }
        supports = new List<PhysicsCaptureV2SupportSnapshot>();
        foreach (KeyValuePair<string, List<string>> item in supportContacts)
        {
            string[] pair = item.Key.Split('\n');
            supports.Add(new PhysicsCaptureV2SupportSnapshot(pair[0], pair[1], item.Value));
        }
        contextualEntities = new List<PhysicsCaptureV2EntitySnapshot>();
        for (int i = 0; i < entities.Count; i++)
        {
            PhysicsCaptureV2EntitySnapshot entity = entities[i];
            contextualEntities.Add(entity.WithContext(new List<string>(contactIds[entity.EntityId]),
                new List<string>(supportedBy[entity.EntityId]),
                new List<string>(supportsIds[entity.EntityId])));
        }
    }

    private static Dictionary<string, SortedSet<string>> NewSets(
        List<PhysicsCaptureV2EntitySnapshot> entities)
    {
        Dictionary<string, SortedSet<string>> values =
            new Dictionary<string, SortedSet<string>>(StringComparer.Ordinal);
        for (int i = 0; i < entities.Count; i++)
            values.Add(entities[i].EntityId, new SortedSet<string>(StringComparer.Ordinal));
        return values;
    }

    private static bool Resolve(Collider2D collider, GameObject[] causalObjects,
        List<PhysicsCaptureV2ColliderSnapshot> colliders, out string entityId, out string colliderId)
    {
        entityId = null; colliderId = null;
        for (int i = 0; i < causalObjects.Length; i++)
        {
            GameObject causal = causalObjects[i];
            if (causal == null || !collider.transform.IsChildOf(causal.transform)) continue;
            ScenarioObjectIdentity identity = causal.GetComponent<ScenarioObjectIdentity>();
            if (identity == null) return false;
            entityId = "runtime:" + identity.ScenarioObjectId;
            Collider2D[] candidates = causal.GetComponentsInChildren<Collider2D>(true);
            for (int index = 0; index < candidates.Length; index++)
                if (candidates[index] == collider)
                    colliderId = entityId + ":collider:" + index.ToString("D4");
            break;
        }
        if (colliderId == null) return false;
        for (int i = 0; i < colliders.Count; i++)
            if (colliders[i].ColliderId == colliderId) return true;
        return false;
    }

    private static int CompareContacts(PhysicsCaptureV2ContactSnapshot left,
        PhysicsCaptureV2ContactSnapshot right)
    {
        int value = string.CompareOrdinal(left.ColliderAId, right.ColliderAId);
        if (value != 0) return value;
        value = string.CompareOrdinal(left.ColliderBId, right.ColliderBId);
        if (value != 0) return value;
        value = left.Point.x.CompareTo(right.Point.x); if (value != 0) return value;
        value = left.Point.y.CompareTo(right.Point.y); if (value != 0) return value;
        return left.Separation.CompareTo(right.Separation);
    }

    private static void Swap(ref string left, ref string right)
    {
        string value = left; left = right; right = value;
    }

    private static bool Fail(PhysicsCaptureV2EngineFailureCode code, string message,
        out PhysicsCaptureV2RecorderFailure failure)
    {
        failure = new PhysicsCaptureV2RecorderFailure(code, message); return false;
    }

    private static bool Finite(Vector2 value) { return Finite(value.x) && Finite(value.y); }
    private static bool Finite(float value) { return !float.IsNaN(value) && !float.IsInfinity(value); }
}
