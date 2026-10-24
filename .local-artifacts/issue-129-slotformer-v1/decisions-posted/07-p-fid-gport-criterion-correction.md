## Decision P-fid (2026-10-24): correction — the B1 G-port parameter criterion is a declared variant of #123's rule

**Trigger.** The pre-freeze review (Review129, P2) found an error in my earlier comments on #129: the adapter
decision (issuecomment-6382359850) and `decisions.md` said that the G-port parameter tolerance "follows #123". It
does not:
- **#123's rule** (`run_hjepa_comparison.GPORT_TOLERANCE`): the share of elements with |Δparam| ≤ 1e-7 (absolute)
  must be ≥ 0.999, and max |Δ| ≤ 2·lr_max + 1e-7.
- **The B1 compare** gated on the share of elements with `isclose(rtol 1e-5, atol 1e-6)`.

**What changed.** `gport/gport.json` now records both, for every arm and step:
- `param_close_share_abs`, `pass_123_exact` and `G_port_123_exact`: #123's exact rule;
- `param_close_share_isclose` and `G_port`: the variant.

Both use #123's max-|Δ| bound, with lr_max taken after the step as in #123. Every other check (loss, gradient
tensors, grad norm, Adam moments, LRs, optimizer, scheduler, parameter counts, batches, predictions) is the same
under both rules.

**Result.**
- **#123's exact rule fails** for one arm-step: `b1-native-s5`, step 2 (the positive-LR step at lr 2e-4), with
  an absolute share of 0.99202 (< 0.999). Its isclose share is 0.99984, and max |Δ| is 3.1e-4 ≤ 2·lr = 4e-4.
- The other 11 arm-steps pass #123's exact rule: shares 0.99922–0.99999 (official PHYRE config, native and
  matched stride 15, both heads).
- **The variant passes** for all six arms.

So **`G_port` = PASS rests on a declared variant**: a share-based version of #123's rule, loosened per element from
|Δ| ≤ 1e-7 to isclose(rtol 1e-5, atol 1e-6). The loosening is needed because Adam normalises every element, which
amplifies round-off-level gradient differences between torch 2.7.1 (official side) and torch 2.13 (adapter).
#123's positive-LR step ran at 5e-5; ours runs at the official 2e-4.

**Evidence that this is float noise, not a port defect.**
- **Losses** are bit-equal in all 12 arm-steps; the one exception is head-b1 step 2, with a relative delta of 8e-8.
- **Gradients:** tensor relative L2 ≤ 6.6e-5 everywhere except `b1-native-s5` step 2, at 4.3e-4. That run has 45
  autoregressive steps, and it starts step 2 from parameters that already differ by the amplified step-1 round-off.
- **Step 2 rerun from the official step-1 state** (non-gating diagnostic `step2_from_official_step1_state`; the
  official step-1 parameters and Adam state are loaded, so step-1 differences cannot propagate):
  - `b1-native-s5` gives an absolute share of 0.99960 and passes #123's rule;
  - `b1-native-s15`, which passes in the chained run, gives 0.99759 here.

  The absolute 1e-7 share at lr 2e-4 therefore moves on either side of 0.999 with the run's float history. It does
  not track a systematic difference.

The orchestrator decides whether the variant is acceptable. Until then this is recorded as a disclosed variant,
not as #123's rule.
