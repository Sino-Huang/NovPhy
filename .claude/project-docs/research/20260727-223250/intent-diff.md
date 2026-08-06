# Intent vs Reality

| intent_id | Expected truth | Observed reality | Diff | Intent source | Status | Linked claims |
| --- | --- | --- | --- | --- | --- | --- |
| I1 | Per-frame scene graphs can be gathered with reliable object identity and geometry. | Unknown whether the active collector can query a runtime state API; a Unity task-template source project contains a GroundTruth component. | Need distinguish source-template capability from shipped-build capability. | User request | unknown | C1, C2 |
| I2 | Contacts/support can be collected without image inference. | Unity 2D physics normally exposes collisions; active project integration remains unverified. | Need source and protocol evidence. | User request | unknown | C3 |
| I3 | Velocity/kinetic-energy data can be emitted per frame. | Unity Rigidbody2D normally exposes velocity, but runtime access path is unknown. | Need verify local scripts and collector protocol. | User request | unknown | C4 |
| I4 | Macro events can be labeled reproducibly. | Collector already records outcomes/shot metadata but not event labels. | Need identify direct callbacks/state transitions and define limits. | User request | unknown | C5 |
