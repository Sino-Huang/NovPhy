## Decision P-engine (2026-10-29): A3 Relational-Image native-task fidelity — funded

**Trigger.** Rule of the 2026-10-25 / 2026-10-28 P-engine decisions: Relational-Image follows only if the remainder after Relational-Symbolic's realised cost still covers its projection with the 25 % margin.

**Numbers (fidelity/a3/, ledger/engine-a3-fidelity.json).** Relational-Symbolic realised 2.285 worker-h (560 episodes, 170 game sessions, status complete); A3 engine total so far 3.227 worker-h; remainder 33.116 worker-h. Relational-Image projection = Relational-Symbolic's realised run cost (same protocol: 970 episodes, 170 sessions) = 2.285 worker-h; x 1.25 = 2.857.

**Choice.** Relational-Image's run starts now (daytime engine window), with its own 50-episode smoke gate against the remainder.

**Reason.** The frozen rule applied mechanically; no A3 NovPhy result enters it (Relational-Symbolic's band outcome is not used).
