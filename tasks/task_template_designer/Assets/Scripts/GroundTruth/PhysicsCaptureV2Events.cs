using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;

public sealed class PhysicsCaptureV2EventSnapshot
{
    public string EventId { get; private set; }
    public string EventType { get; private set; }
    public long FixedStep { get; private set; }
    public ReadOnlyCollection<string> Participants { get; private set; }
    public string PayloadJson { get; private set; }

    public PhysicsCaptureV2EventSnapshot(string eventId, string eventType, long fixedStep,
        IList<string> participants, string payloadJson)
    {
        EventId = eventId;
        EventType = eventType;
        FixedStep = fixedStep;
        List<string> sorted = new List<string>(participants ?? new string[0]);
        sorted.Sort(StringComparer.Ordinal);
        Participants = sorted.AsReadOnly();
        PayloadJson = string.IsNullOrEmpty(payloadJson) ? "{}" : payloadJson;
    }
}
