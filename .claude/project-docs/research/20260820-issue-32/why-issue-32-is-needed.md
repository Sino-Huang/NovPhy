# Why issue #32 is needed

## Short answer

[Issue #32, “Prove fail-closed downstream ingestion”](https://github.com/Sino-Huang/NovPhy/issues/32), is the safety check between **publishing research data** and **using that data in an experiment**.

Issue #31 produced a frozen, named cohort release. Issue #32 must prove that every required training, scoring, and evaluation reader can read that exact release without changing its meaning—and that it stops when evidence is missing, damaged, or belongs to a different release. In simple words: **we need to test the data at the point where the experiments actually use it, not only at the point where we create it.**

## What “fail closed” means

“Fail closed” means **stop with an error instead of guessing**. A downstream reader must refuse to continue when, for example:

- a required file or capability is missing;
- a label file is malformed or was made for another cohort release;
- an event is joined to the wrong simulation step;
- the final observation of a rollout is absent;
- object, rollout, or frame identities do not match; or
- a label whose value is `unavailable` is silently treated as `false`.

The project specification explicitly requires downstream smoke ingestion to preserve identity, timing, availability, and exposure restrictions ([data specification, lines 321–331](../../../../docs/data_generation_and_collection_spec.md#L321-L331)). It also says a successful command or smoke test alone is not enough to complete the research-data work ([lines 333–333](../../../../docs/data_generation_and_collection_spec.md#L333)).

## Why issue #31 was not enough

Issue #31 published the immutable input that #32 can now test. The accepted release contains four real Unity rollouts across two scenario lineages, and it gives stable identities to the cohort, its derivations, and its partition manifest ([release README, lines 1–17](../../../evidence/representative-cohort-release-20260820/README.md#L1-L17)). It also names 15 capabilities or labels that are unavailable or excluded, rather than inventing values for them ([lines 17–28](../../../evidence/representative-cohort-release-20260820/README.md#L17-L28)).

But publication only proves that the package was assembled. The current publication verifier checks the top-level publication, cohort-release, and derivation identities and file digests ([`verify_cohort_publication`, lines 241–255](../../../../scripts/cohort_release.py#L241-L255)). The existing world-model adapter reads rollout directories and their sidecars ([dataset adapter, lines 208–221](../../../../world_model/data/dataset.py#L208-L221)); it does not, by that fact alone, prove end-to-end ingestion from the new publication entry point. Issue #32 exists to close this last gap with the real released artifacts and deliberate failure cases. This is an inference from the current public interfaces and the issue acceptance criteria.

## How this connects to the planned research experiments

The experiments depend on several meanings surviving ingestion exactly:

1. **Stage 1: oracle-symbol upper bound.** We train and score every time-horizon/description-level pair using engine truth, then use those scores as labels for the later controller ([research proposal, lines 280–286](../../../../docs/research_proposal.md#L280-L286)). If frames, identities, or macro events are joined incorrectly, the “best pair” teacher labels are wrong.

2. **Stage 2: learned symbolic state.** The visual parser is measured against engine labels on held-out levels ([lines 288–296](../../../../docs/research_proposal.md#L288-L296)). If a held-out lineage leaks into training, or `unavailable` becomes `false`, predicate F1 and generalization results can look better or worse for the wrong reason.

3. **Stage 3: the central controller ablation.** The paper compares joint, factorized, single-axis, and fixed controllers using prediction, physical-plausibility, compute, and out-of-distribution results ([lines 298–307](../../../../docs/research_proposal.md#L298-L307)). Using a sidecar from the wrong release or shifting events between fixed steps makes those comparisons scientifically unfair even if the program still runs.

4. **Final evaluation.** Training, calibration, model selection, and final evaluation have different exposure permissions ([data specification, lines 98–123](../../../../docs/data_generation_and_collection_spec.md#L98-L123)). The readers must preserve those permissions so the final test data cannot quietly influence model choices.

Two details are especially important:

- **Fixed-step timing:** the simulator’s fixed step, not the rendered video frame, is the authority for physical events ([data specification, lines 200–215](../../../../docs/data_generation_and_collection_spec.md#L200-L215)). The current physics reader deliberately brackets events by fixed step ([physics reader, lines 139–160](../../../../world_model/data/supervision.py#L139-L160)); #32 must prove this survives release ingestion.
- **Three-state labels:** support and other evidence-dependent labels can be `true`, `false`, or `unavailable`; unavailable must never be converted to false ([data specification, lines 242–248](../../../../docs/data_generation_and_collection_spec.md#L242-L248)). This matters immediately because the lightweight release intentionally leaves many capabilities unavailable.

## The practical research reason

Without issue #32, an experiment could finish, produce clean-looking graphs, and still be based on the wrong data meaning. The likely failures are subtle: a missing label becomes a negative example, a terminal state disappears, an event moves to the wrong time, a training lineage leaks into evaluation, or labels from one cohort are paired with traces from another.

So this ticket is not “extra data plumbing.” It is the evidence that lets us say:

> These model results came from this exact cohort release and these exact derivation versions, with timing, identities, missing information, and train/evaluation boundaries preserved.

That statement is required before the enriched cohort can safely support the Stage 1–4 experiments and the paper’s main claim.

## Primary sources

- [GitHub issue #32](https://github.com/Sino-Huang/NovPhy/issues/32)
- [GitHub issue #31](https://github.com/Sino-Huang/NovPhy/issues/31)
- [GitHub issue #18](https://github.com/Sino-Huang/NovPhy/issues/18)
- [Data generation and collection research specification](../../../../docs/data_generation_and_collection_spec.md)
- [Research proposal](../../../../docs/research_proposal.md)
- [Published representative cohort release README](../../../evidence/representative-cohort-release-20260820/README.md)
- [Cohort release verifier](../../../../scripts/cohort_release.py)
- [World-model physics supervision reader](../../../../world_model/data/supervision.py)
