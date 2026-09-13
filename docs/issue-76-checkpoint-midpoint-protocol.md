# Fixed adjacent-checkpoint midpoint diagnosis

The completed endpoint continuation loses several previously attained local
or endpoint accuracies. A minimized actual-checkpoint replay,
`python -m scripts.diagnose_issue_76_trajectory_retention`, returned failure
twice with identical ratios: hybrid seed 1 local 1.197268006; pure seed 1
local 1.197970162 and endpoint 1.209204326; pure seed 3 endpoint 1.221902004.
All exceed 1.10. In-process runtimes were 0.953 and 0.921 seconds. It uses
the original five assigned lineages and only one or 15 recursive horizon-750
steps as required by each symptom, matching validated means exactly.

## Ranked hypotheses, before midpoint outcomes

1. Adjacent checkpoints contain complementary training variation: a fixed
   parameter midpoint improves both local and endpoint accuracy across seeds.
2. Errors share persistent bias: even their post-hoc prediction midpoint
   cannot beat the stronger parent's error on the failed comparison.
3. Nonlinear parameter interpolation prevents averaging gains: prediction
   midpoint improves error but parameter midpoint does not.

These tests distinguish outcomes but do not establish a unique cause of the
whole training trajectory. No favorable single-step result licenses a long fit.

## Frozen diagnostic

Before outcomes, fix one rule: elementwise 50/50 parameter averaging of the
completed targeted-local checkpoint and its endpoint-continuation child.
Apply the identical rule to all three seeds and both independently trained
arms. Never average hybrid with pure, choose weights by seed or outcome,
search alternative weights, or write a saved predictor checkpoint.

Use the existing five assigned training lineages, first-shot boundary states,
all nine hybrid pairs and three pure horizons. Recompute all 30 records and
180 policy curves for the parameter midpoint using the unchanged diagnostic.
Retain all parent curves and every qualification requirement, including an
additional initial/endpoint retention check against the endpoint-continuation
checkpoint so its stronger results cannot be discarded.

Separately recurse each parent independently and record error MSE, error
cross-product and the error of their 50/50 output average at elapsed 750 and
11,250 native steps. This output average is a post-hoc diagnostic at common
times, not a recurrent ensemble policy or a deployable candidate. It does not
use truth resets; only actually observed targets are scored. A parameter
midpoint has the original model architecture and inference work, but all
parent training and diagnostic work remains chargeable.

Archive this source and protocol before `--run`. Initialize novphy and source
env.sh. Use a cumulative 180-active-second CPU budget, 12 GiB RSS, and no GPU
or optimization. Retain earlier replay costs and verify source model tensors
remain unchanged. No new capture, held-out/fresh/final access, controller
training or advancement claim. Any adoption requires independent validation
and full development/gameplay readiness; these training curves alone do not
pass #76's advancement gate.
