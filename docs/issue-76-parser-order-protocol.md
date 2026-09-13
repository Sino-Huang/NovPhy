# Prospective parser-order diagnosis

This single candidate tests whether the native v2 parser's family-blocked
training schedule contributed to its near-constant predictions. It follows the
published negative v2 refit. It is not a new optimizer sweep, a fresh experiment,
or an advancement protocol. Freeze this file and the executable before fitting.

## Exact intervention and exposure

Use the original 250 assigned training episodes, including the two unusable
records, in five groups of 50. The original schedule visits all 50 entries of
one family before the next. The candidate visits the first entry of each sorted
family, then the second of each family, and so on. Repeat for six passes.

The permutation carries the **original update number** with each entry; its
original `generator(seed, original_update)` selects the same 32 image/target
rows. Thus all 1,500 scheduled batches and all 12 unusable skips are retained,
and only their order changes. No frame resampling, new examples, replacement
episodes, or outcome-based filtering. Test permutation and frame equivalence
before execution.

Run all three original seeds 760930001/2/3 from the same original initialization.
Reuse NativeVisualParser, perception_loss, AdamW at .001, weight decay .0001,
gradient norm cap 1, image scaling, and 1,500 scheduled updates. Compare against
the completed ordered v2 parser for the same seed. No history, predictor,
controller, alternative architecture, learning-rate search, or new rendering.
New checkpoints are parser-only and cannot be deployed as common checkpoints.

## Frozen measurement and decision

Evaluate all 250 training and 50 model-selection assignments. Calibration is
not used for this recipe decision. On each usable episode, select 32 evenly
spaced actual frame indices by integer arithmetic, including first and last;
use all frames if fewer than 32 exist. The same rows serve both parsers. Retain
unusable assignments and report their coverage. Only training enters optimizers.

Record per-episode block-count MAE, pig-count MAE, task-slot presence Brier error,
and center MAE on present task slots (seven block slots plus one pig slot).
Record first-frame predicted/true block counts and pig class/error counts.
No missing pig-death examples are fabricated; absent class support limits any
perception-readiness claim. Average episode metrics within each role/family,
then weight the five families equally for the model-selection block-count MAE.

Support for this ordering hypothesis requires **all three seeds** to meet both:

- model-selection block-count MAE at most 50% of the paired ordered baseline;
- a predicted-count span of at least 3 across the first usable training episode
  of each of the five families (true counts previously observed as 1/5/6/7/4).

These are mechanistic development criteria chosen after v2 diagnosis but before
the candidate fit. They do not replace any #76 advancement margin, physical
validity, useful-mode, strong-baseline, gameplay, or matched-compute requirement.
Report every seed/cell regardless of the result. Do not choose a favorable seed,
extend training after inspection, force symbolic modes, or start fresh access.
A favorable result still requires a separately frozen symmetric full refit and
gameplay-readiness assessment; a negative result remains negative evidence.

## Cost, persistence, and source binding

Separate root `.local-artifacts/issue-76-parser-order-v1`; no v2 source,
checkpoint, cache, result, or budget changes. Reuse existing prepared shards
without new hashes, derivations, or full-corpus integrity scans. Bind the parent
plan, assigned entries, exact update permutation, exact evaluation frame indices,
original common budgets, executable, and this protocol in plan.json before fit.

Allowance: 900 active seconds per seed for new fitting plus paired scoring,
2,700 seconds total; 12 GiB process-tree CPU RSS, 8 GiB allocated GPU memory,
and 1 GiB of new artifacts. Expected duration is under ten minutes total on the
current GPU, but the allowance is not an invitation to add updates. Original
ordered-baseline training costs remain separately available in the bound parent
common budgets. Persist progress and resource use using the existing FitBudget;
retain failures/stops and require an explicit audit of an unclean interruption.
No concurrent GPU job. Metadata/unit-test CPU work is not a new research fit.

With the `novphy` environment activated and `env.sh` sourced:

```bash
python -m unittest -q tests.test_issue_76_order_probe
python -m scripts.run_issue_76_order_probe --dry-run
python -m scripts.run_issue_76_order_probe --prepare
python -u -m scripts.run_issue_76_order_probe --run
python -m scripts.run_issue_76_order_probe --publish
python -m scripts.run_issue_76_order_probe --validate
```
