# Joint local/endpoint retention diagnosis

The targeted-local stage is complete and independently validated, but pure
seed 760930001 fails endpoint retention and pure seed 760930002 fails local
retention. The minimal actual-checkpoint replay is:

```sh
python -m scripts.diagnose_issue_76_joint_retention
```

After initializing novphy and sourcing env.sh, two executions returned exit 1
with identical measurements: seed 1 endpoint 0.0038189456332474946 / reference
0.003193629835732281 = 1.1958009630667896; seed 2 initial error
0.00412751967087388 / reference 0.0036573459627106788 = 1.1285559837535106.
The maximum permitted ratio is 1.10. In-process times were 0.501 and 0.515
seconds. The replay omits unrelated pairs and later steps for the local check,
but retains all five assigned lineages and the full 15-step recursion for
the endpoint check. Its means exactly match the independently validated report.
No optimization, new observations or held-out access occurs.

## Ranked hypotheses before new probe outcomes

1. Endpoint error is underweighted by averaging recursive losses across time.
   Adding one unit-weight endpoint MSE term to the unchanged current objective
   should improve finite-step endpoint change relative to control.
2. Horizon-750 local weighting trades off with endpoint accuracy at the current
   parameters. Reducing its weight from 10 to 1 should improve endpoint change
   but may worsen local change; measure both rather than calling either a fix.
3. Mixed-start sampling dilutes boundary accuracy at the newer checkpoint.
   Boundary-only sampling with the current objective should improve both
   remaining anchors relative to the same mixed control batch.

## Frozen one-factor probes

Assign all three completed targeted-local seeds and all 36 hybrid/pure
policy/seed cells before outcomes. Reuse the exact five assigned training
lineages, first-shot boundary rows and seeded uniform rows from the earlier
paired probe: five boundary starts plus one uniform frame per lineage, CPU
generator seeded by model seed. These starts are identical across arms/pairs;
this controlled ten-start batch is not a replay of production batch size 32.

For each cell, compute four temporary clipped AdamW deltas: current weighted
objective control; add endpoint MSE with coefficient 1; local weight 1; and
boundary-only sampling. Keep the saved learning rate and optimizer history.
The local-weight-1 variant equals control at horizons 50/250 by construction.
Endpoint loss recursively predicts the full 11,250-native-step target and
supervises only rows with an actually observed endpoint. Missing endpoints
are excluded, not infilled. Retain current local and symbolic terms otherwise.

Measure unchanged current mixed objective, horizon-750 local anchor and
endpoint error, and the updated policy's boundary endpoint error, before and
after each variant. Retain every measurement and restore exact parameters
after each perturbation. Compare variants against both control and original
weights. No saved checkpoint writes, candidate selection or fitting occur.

Archive this protocol and source before `--run`. Use a cumulative 600-active-
second CUDA budget with 12 GiB RSS and 8 GiB allocated-CUDA limits; retain the
earlier replay cost. No new capture, fresh/final access, controller fitting or
advancement claim. A subsequent training repair must be separately specified;
one-step counterfactuals are not proof of full-fit or gameplay improvement.
