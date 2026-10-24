# Agent-level G-port sensitivity check (2026-10-24, CPU, not part of the gate)

`agent_compare_arm` with one defect injected into the port at a time (monkeypatched in-process; files unchanged):

| injected defect | arm | arm pass | step-1 worst metric rel. | optimizer calls failing |
| --- | --- | --- | --- | --- |
| λ-return λ 0.95 → 0.9 | dreamerv2 | False | 1.73e-01 | actor, critic |
| THICK Dreamer ψ 0.9 → 0.1 | thick-dreamer | False | 7.53e-02 | critic |
| Plan2Explore disagreement with the unbiased std | plan2explore | False | 6.00e-02 | exploration actor, exploration critic |

The unmodified port passes (`gport.json`).
