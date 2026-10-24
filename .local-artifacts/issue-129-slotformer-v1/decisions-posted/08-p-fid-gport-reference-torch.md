## Decision P-fid (2026-10-24): follow-up to the G-port correction — the frozen G-port reference runs on torch 2.13

**Trigger.** The orchestrator decided that B1's G-port must not rest on a loosened criterion when the cause can be
removed. The cause was the torch version: the official reference ran on 2.7.1 and the adapter runs on 2.13. A1, A2
and A4 got Δ = 0 because their official sources ran on the adapter's torch.

**Change.** This removes the version difference; it does not change any tolerance.
- **Official side:** `scripts/b127_slotformer_port_reference.py reference` now runs the official SlotFormer and
  nerv sources from the pinned clones (`SlotFormer@10a6d9b`, `nerv@e83ac66`, on `PYTHONPATH`) on **torch
  2.13.0+cu130, CPU**, the same build as the adapter. It runs in venv `b127-slotformer-ref213`, a
  `--system-site-packages` venv over the novphy env; its package list is in `envs/b127-slotformer-ref213.txt`.
- **Gate:** `G_port` now gates on **#123's exact rule**: share of |Δparam| ≤ 1e-7 ≥ 0.999 and max |Δ| ≤ 2·lr_max
  + 1e-7. `gport.json` records `G_port_rule: issue_123_exact` and the reference torch and Python.
- **Smoke:** `--smoke` requires this rule.
- **Diagnostics, recorded but not gating:**
  - the isclose share, as `G_port_isclose_variant`;
  - the torch-2.7.1 comparison, kept as `gport/torch271/gport.json`. It was referenced from `diagnostic_torch_2_7_1`
    with sha256 `f1e9f999…ded6`, `G_port_123_exact` = false.

**Result: G-port PASS under #123's exact rule, bit-identical.**
- All six arms pass: the official PHYRE config, native stride 15, matched stride 15, native stride 5, the official
  readout and the #129 head.
- In both steps of every arm, loss, gradient and Adam-moment deltas are exactly 0, the absolute parameter share is
  1.000000, and max |Δparam| = 0.
- Predictions are identical, and so is the step-2-from-official-state diagnostic.
