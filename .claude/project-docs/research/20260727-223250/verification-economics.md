# Verification Economics

| claim | risk | verification path | outcome | residual risk |
| --- | --- | --- | --- | --- |
| C1-C5 | high | Source/protocol inspection plus primary Unity documentation; execute only if a local callable state surface exists. | pending | Template source may diverge from deployed game. |
| C1 | high | Compare source names/handlers with deployed Assembly-CSharp metadata. | supported | Exact wire payload remains unexecuted. |
| C2 | high | Inspect exporter and active collector/bridge separately. | supported | Ground-truth noise mode must be smoke-tested. |
| C3-C4 | high | Inspect existing Rigidbody/collision integration and Unity 2019.4 primary APIs. | supported | Requires a new exporter schema. |
| C5 | high | Trace launch/collision/death/clear/fail/stability anchors. | supported | Event taxonomy and timestamp domain still need a design decision. |
