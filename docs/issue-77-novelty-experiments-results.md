# Issue #77 — novelty-level impact experiments: results record

Date: 2026-09-20. Executed end-to-end per
[issue-77-novelty-experiments-plan.md](issue-77-novelty-experiments-plan.md).
Claim boundary: descriptive development evidence under the small-project protocol;
zero-shot and few-shot are separate claims and never mixed; this record does not
modify #15/#74/#75/#76 dispositions and does not unblock #64/#65.

Published artifacts (machine-readable tables beside the modules that produced them):

| Stage | Output root | Report files |
| --- | --- | --- |
| N1 campaign | `.local-artifacts/issue-77-n1-v1/` | `coverage.json` |
| N1 training | `.local-artifacts/issue-77-n1-dynamics-v1/` | `plan.json`, `seed-*/{continuous,hybrid}/` |
| N1 diagnostic | `.local-artifacts/issue-77-n1-diagnostic-v1/` | `summary.json`, `comparisons.csv`, `findings.md` |
| N2 campaigns | `.local-artifacts/issue-77-n2-appearance-v1/`, `…/issue-77-n2n-v1/` | `coverage.json` |
| N2 evaluation | `.local-artifacts/issue-77-n2-eval-v1/` | `summary.json`, `comparisons.csv`, `findings.md` |

## N0 — representation readiness (decided before outcomes)

- Ran with the frozen 18-slot / 236-carrier contract; no amendments were needed for
  the executed cells, so no symmetric-repair cycle was opened.
- Ready cells (executed): normal rolling `type010103`, normal sliding `type010105`
  (N1 breadth); appearance level-1 pairs for `type010101` and `type010102` (N2).
- Unsupported, recorded: `type010104` normal falling (8 platforms > 6-platform slot
  vocabulary); novelty levels 2,3,4,6,8 (unrepresented forces/agents); level 5
  (right-side slingshot vs left-pull action contract); level 7 (task-objective
  change needs redeclared evaluation semantics).

## N1 — normal-mechanics breadth (rolling + sliding)

Campaign: 208/208 branches attempted (2 families × 8 lineages × 13 fixed actions),
178 admissible, 30 typed failures, coverage complete. Training: 3 seeds
(20260908/09/10) × 2 arms at identical 9,000-step budgets, 504 windows from 12
training-role lineages, wall 2126 s shard build + ~51 min training. Diagnostic:
4 held-out states, 52 candidates, 12 systems × 3 seeds = 1,872 fixed records;
published and exact-validated.

Headroom: all four states informative; mean ordinal09-prior regret 0.492, mean
uniform-random expected regret 0.461.

Paired contrasts (mean regret difference, positive favors tested; descriptive 95%
intervals over 4 states):

| Tested | Reference | Kind | Mean | Interval |
| --- | --- | --- | --- | --- |
| hybrid_continuous_h1 | continuous_h1 | training_effect | +0.0887 | [+0.0155, +0.2118] |
| hybrid_micro_h1 | hybrid_continuous_h1 | symbolic_execution | +0.0348 | [−0.0102, +0.1147] |
| hybrid_macro_h1 | hybrid_continuous_h1 | symbolic_execution | +0.0239 | [−0.0199, +0.0938] |
| hybrid_continuous_h5 | continuous_h5 | training_effect | −0.0028 | [−0.1333, +0.1708] |
| hybrid_micro_h5 | hybrid_continuous_h5 | symbolic_execution | −0.1443 | [−0.3414, +0.0024] |
| hybrid_macro_h5 | hybrid_continuous_h5 | symbolic_execution | −0.0677 | [−0.1480, +0.0074] |
| hybrid_continuous_h15 | continuous_h15 | training_effect | −0.0051 | [−0.1625, +0.1373] |
| hybrid_micro_h15 | hybrid_continuous_h15 | symbolic_execution | −0.0038 | [−0.1011, +0.0897] |
| hybrid_macro_h15 | hybrid_continuous_h15 | symbolic_execution | +0.0944 | [+0.0231, +0.1878] |

Reading: the h=1 hybrid-arm advantage seen on #74's single-force/multiple-forces
tasks reproduces on rolling+sliding (training effect interval excludes zero);
h=5/h=15 training effects are null; the h=15 macro-mode symbolic-execution gain is
the only other interval excluding zero. Carrier-MSE curves agree at short times and
diverge in magnitude at t≥150 (long-rollout instability, both arms); t=600 is an
end-of-window cost on mostly right-censored branches, not a settled cost. About half
of admissible N1 branches terminate before frame 601 (early level clear/fail); their
t>terminal targets use recorded stable-terminal absorption.

## N2 — matched normal/novel appearance pairs

Captures: novel side 208/208 attempted, 186 admissible, 22 typed failures
(`issue-77-n2-appearance-v1`). Normal side for `type010102` (`issue-77-n2n-v1`,
normal `type010101` side already exists in R3): 104/104 attempted, **34 admissible,
70 typed failures**. The n2n campaign ran in parallel with the N2 campaign's tail;
its failure kinds (native-manifest deadlines, paired-view mismatches) match N2's
but at ~6× the rate, indicating contention-induced rather than family-systematic
failure. Retained as typed failures per protocol; the type010102 normal side
therefore contributes 5 of 8 lineages with 4–11 candidates each (31 dropped
candidates reported in the headroom table, never worst-cased).

Evaluation: 17 states (8 R3 normal + 5 n2n normal + 4 held-out novel), 189
candidates, 12 systems × 3 seeds × 2 conditions = 13,608 fixed records; 6 adapted
predictors (3 seeds × 2 arms, 2,000 steps, batch 64, lr 1e-4, identical budgets,
adaptation pool = 8 novel predictor lineages, `issue-77-n2-009` dropped on zero
admissible branches). Controllers not adapted; fixed systems only. Published and
exact-validated.

### Zero-shot (frozen N1 checkpoints; no novelty fitting)

Novel-side headroom is asymmetric: type010101 novel states show low prior regret
(0.095/0.215 vs 0.26–0.96 on R3 normal), so ranking differences there carry less
information; type010102 novel states retain headroom (0.71/0.81 prior regret).

Cross-side change in the hybrid-minus-continuous effect (negative = hybrid advantage
shrinks on the novelty; bootstrap over each side's independent states):

| Pair | Tested | Reference | Novel effect | Normal effect | Change [95%] |
| --- | --- | --- | --- | --- | --- |
| type010101 | hybrid_continuous_h1 | continuous_h1 | +0.1065 | +0.0404 | +0.0662 [−0.1008, +0.2288] |
| type010101 | hybrid_continuous_h15 | continuous_h15 | −0.1824 | +0.0104 | −0.1928 [−0.4525, +0.0845] |
| type010101 | hybrid_macro_h15 | hybrid_continuous_h15 | +0.0075 | +0.1263 | −0.1189 [−0.2569, +0.0199] |
| type010102 | hybrid_continuous_h1 | continuous_h1 | +0.0384 | −0.0560 | +0.0943 [−0.0486, +0.2373] |
| type010102 | hybrid_continuous_h15 | continuous_h15 | −0.0098 | +0.2595 | −0.2693 [−0.5412, −0.0050] |
| type010102 | hybrid_macro_h15 | hybrid_continuous_h15 | +0.2664 | −0.1578 | +0.4242 [+0.0915, +0.7657] |

### Few-shot (2,000-step adaptation on novel predictor lineages)

| Pair | Tested | Reference | Novel effect | Normal effect | Change [95%] |
| --- | --- | --- | --- | --- | --- |
| type010101 | hybrid_continuous_h1 | continuous_h1 | +0.0348 | +0.0824 | −0.0476 [−0.1489, +0.0384] |
| type010101 | hybrid_continuous_h15 | continuous_h15 | +0.0839 | +0.0125 | +0.0714 [−0.1083, +0.2579] |
| type010101 | hybrid_macro_h15 | hybrid_continuous_h15 | −0.1286 | −0.0438 | −0.0848 [−0.2938, +0.1262] |
| type010102 | hybrid_continuous_h1 | continuous_h1 | −0.1830 | −0.1494 | −0.0337 [−0.2021, +0.1503] |
| type010102 | hybrid_continuous_h15 | continuous_h15 | +0.1787 | −0.0166 | +0.1953 [+0.0583, +0.3363] |
| type010102 | hybrid_macro_h15 | hybrid_continuous_h15 | +0.0254 | +0.2147 | −0.1892 [−0.2720, −0.0945] |

Adaptation change on the novel side (few-shot minus zero-shot, negative favors
adaptation): type010102 `continuous_h1` −0.1970 [−0.2688, −0.1253] improves, while
`continuous_h5` +0.1850 [+0.1012, +0.2688] and `continuous_h15` +0.0672
[+0.0432, +0.0912] degrade; hybrid systems mostly straddle zero. type010101 shows
the same direction pattern at weaker magnitude (`continuous_h5` +0.1544 [+0.0062,
+0.3025], `continuous_h15` +0.1080 [+0.0339, +0.1820]).

### N2 reading

- Appearance novelty moves the comparison: two zero-shot type010102 intervals
  exclude zero in opposite directions per system (trained-hybrid h15 advantage
  shrinks; fixed macro-mode h15 advantage grows). No single "novelty helps/hurts
  hybrid" statement is supportable — effects are system- and horizon-specific.
- Few-shot adaptation does not uniformly recover normal-side behavior: it helps
  short-horizon continuous ranking and hurts longer-horizon continuous ranking,
  with hybrid systems largely unchanged.
- As the perception control, the appearance cell shows the shared CNN parser
  transfers well enough that ranking headroom survives on type010102 novel states
  (prior regret 0.71/0.81); the type010101 novel side's low headroom is a property
  of those layouts/cost surfaces, not a parser failure signature.

## Limitations (carried from the published tables)

- Shared semantically supervised CNN, not an end-to-end symbol-free comparison.
- t=600 is end-of-window cost on mostly right-censored branches, not settled cost;
  early-terminal branches use recorded stable-terminal absorption.
- At most four independent held-out states per side per family (two per family on
  the novel side); bootstrap intervals are descriptive, not multiplicity-adjusted.
- n2n normal side is 5/8 lineages with reduced candidate sets (contention-era typed
  failures retained, reported, never worst-cased).
- R3 normal-side lineages retain #75 development model-selection scoring optimism.
- Controllers not adapted in N2; fixed systems only. Linear MACs, not full FLOPs.

## Reproduction

```bash
# N1 campaign (captures already retained; --validate re-derives coverage)
python scripts/run_issue_77_n1.py --validate
# N1 training + diagnostic
python -m scripts.run_issue_77_n1_train --train --device cuda
python -m scripts.run_issue_77_n1_diagnostic --run-diagnostic --device cuda
python -m scripts.run_issue_77_n1_diagnostic --publish --device cuda
python -m scripts.run_issue_77_n1_diagnostic --validate
# N2 campaigns + evaluation
python scripts/run_issue_77_n2.py --validate
python scripts/run_issue_77_n2n.py --validate
python -m scripts.run_issue_77_n2_eval --build-shards --device cuda
python -m scripts.run_issue_77_n2_eval --adapt --device cuda
python -m scripts.run_issue_77_n2_eval --run-evaluation --device cuda
python -m scripts.run_issue_77_n2_eval --publish --device cuda
python -m scripts.run_issue_77_n2_eval --validate
```

## Readiness record delta

Scenario×novelty cells with runtime evidence move from **0/45 to 4/45**: normal
rolling, normal sliding (N1), appearance×single-force, appearance×multiple-forces
(N2, zero-shot and few-shot). All other cells stay `unsupported` with the named
blockers in [issue-76-novelty-level-evaluation-design.md](issue-76-novelty-level-evaluation-design.md).
