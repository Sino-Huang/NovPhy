# G-port sensitivity check (2026-10-24, CPU, not part of the gate)

`compare_arm` on `reference-novphy-carrier` with one deliberate defect injected into the port at a time
(monkeypatched in-process; the port file was not changed). Every defect fails G-port:

| injected defect | arm pass | failing module forwards | step-1 loss rel. | step-1 update rel. L2 | params within 1e-7 |
| --- | --- | --- | --- | --- | --- |
| torch-style Adam (epsilon inside the bias correction) | False | none | 2.36e-07 | 1.14 | 0.126 |
| GateL0RD gates without the straight-through `ceil` | False | gatel0rd gates (both), observe post/prior gates | 0.375 | 1.01 | 0.505 |
| KL balance flipped (0.8 -> 0.2) | False | none | 2.36e-07 | 0.292 | 0.375 |

The unmodified port passes (`gport.json`): worst step update relative L2 1.6e-05, worst gradient tensor relative L2 1.6e-05.
