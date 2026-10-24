# #127 program-level decisions

## Decision P-budget (2026-10-25): how the 400 GPU-h program budget is counted

**Trigger.** Rule P-budget allocates "the 400 GPU-h"; the first night ran several #127 jobs side by side on the same GPU (light jobs of 1–4 GB beside each other), as the scheduler's `shares_gpu` allows. The #123 ledger convention counts each process's wall time on its GPU, so two jobs sharing one GPU for an hour count as two GPU-hours. Fixed now, before any budget-dependent decision (no Tier-C or coverage arm has been funded or considered).

**Decision.**
- **Program budget = device time.** For each GPU, the union of the time intervals in which at least one #127 scheduler run (job ids `128-*` … `132-*` and later #127 children) was running on it, summed over the four GPUs; computed from the scheduler's run records (`launched_at` / `ended_at` in `.local-artifacts/gpu-window/queue.json`). This is the resource the owner's budget controls. P-budget, P-cap and P-coverage funding checks use it.
- **Child ledgers are unchanged:** each child keeps the #123 per-process GPU-seconds (each arm-seed's training counted once from its `report.json`), and its frozen per-child cap is checked against that per-process number, which can only be larger than the child's device time. Where an arm-seed shared its GPU, its recorded training GPU-h overstates exclusive compute; the program summary marks shared runs and also reports each arm's exclusive-throughput estimate (updates × the smoke-measured seconds per update alone) beside the recorded number, for the AUC-vs-compute frontier.
- Both numbers are published in the program summary; the budget statement uses device time.

**Not changed:** child caps, the 48 h-on-4-GPUs projection rule, every arm's native budget and seeds.
