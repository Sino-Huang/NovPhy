# The existing parser can fit a tiny mixed batch

The prospectively frozen, training-only five-image check completed all 1,000
updates in 26.0287 active seconds. Its published report validates at
`data/issue-76-parser-learnability/report.json`.

Final block-count MAE was .0007626 (required <=.1), and task-slot presence Brier
error was .0000009218 (required <=.01). Predicted block counts were
`[1.00036, 5.00058, 5.99948, 3.99943, 6.99822]` for true counts `[1,5,6,4,7]`,
in the plan's sorted-family order. Perception loss fell from 11.20763 to .001998;
present task-slot center MAE fell from .115997 to .016325. There were no skipped
updates. Peak allocated GPU memory was 32.24 MiB and CPU RSS 2034.98 MiB.

The unchanged 64×96-input architecture therefore can fit these five known
images when trained jointly. This does not prove that resolution is adequate
for every scenario, that mixed batches alone explain the production failure,
or that generalization is good. All five images have a present pig, so this
check says nothing about recognizing pig removal. Its weights are not deployable.

The next controlled comparison will mix the existing image/label samples across
episodes within each batch, preserving total exposure, optimizer settings, update
counts, failed assignments, and the original three seeds. Its outcome must be
measured under a new prospective protocol; no advancement requirement changes.
