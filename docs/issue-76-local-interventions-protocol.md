# Paired residual-local-accuracy intervention probes

Before any probe outcomes, assign all three lower-rate seeds, all nine hybrid
pairs and all three pure pairs. Use the existing five assigned training
lineages. For each seed, construct five first-shot boundary starts plus one
uniformly sampled frame per lineage using a CPU generator seeded by that seed.
These ten starts are identical across both arms and all pairs. They are a
controlled diagnostic batch, not a replay of the production 32-start schedule.
Missing future observations retain their original masks.

For each of the 36 cells, compare four temporary one-step parameter changes:

1. Control: current full-duration objective and retained AdamW state.
2. Local weight 10: multiply only the local objective by ten.
3. Boundary only: use only the five boundary rows, with coefficients unchanged.
4. Reset optimizer history: zero first/second moments and step counters in a
   temporary copy, retaining learning rate, weight decay and all other settings.

Use the existing tested clipped AdamW delta calculation with learning rate
0.00003. Restore exact parameters after each variant; never write a checkpoint
or mutate saved optimizer tensors. Report all variants, not just improvements.
Measure the unchanged mixed-batch objective, horizon-750 anchor first-step MSE,
anchor endpoint MSE, and the updated policy's endpoint MSE on the five boundary
rows. Endpoints remain 11,250 native steps. A positive result is evidence about
one finite step, not proof of a full-fit cause or a qualified repair.

The ranked predictions are recorded in the
[diagnosis](issue-76-local-retention-diagnosis.md). Compare each variant to
control, with special attention to both failing pure seeds; retain other arms
and policies as paired checks. No outcome-dependent data or seed selection,
threshold changes, controller fits, capture or fresh/final access is allowed.

Record a cumulative 600-active-second CUDA budget, at most 12 GiB RSS and
8 GiB allocated CUDA. Include earlier replay costs. Archive this protocol and
source before running `python -m scripts.diagnose_issue_76_local_interventions --run`
in the initialized novphy environment with env.sh sourced. A subsequent
training recipe must be separately specified before fitting.
