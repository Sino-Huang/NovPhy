# Manuscript Research Evidence Ledger

## Scope and Research Target

**NovPhy benchmark.** NovPhy is the published physical-reasoning benchmark for open-world AI systems. The published paper evaluates novelty detection and adaptation across five physical scenarios and eight novelty levels. The canonical citation is V. Pinto, C. Gamage, C. Xue, P. Zhang, E. Nikonova, M. Stephenson, and J. Renz, “NovPhy: A physical reasoning benchmark for open-world AI systems,” *Artificial Intelligence*, vol. 336, p. 104198, 2024. https://doi.org/10.1016/j.artint.2024.104198. It is distinct from the BG-NS-JEPA work described here.

**BG-NS-JEPA program.** BG-NS-JEPA is an in-progress world-model program for action-sparse persistent-effect environments. Its controller selects a requested horizon and description mode, while a continuous latent carries the rollout. The central claim remains unresolved: `[TODO: result]`.

**Central research question.** At a predeclared common compute budget, does a joint horizon-description controller improve a common endpoint over fixed and factorized policies? This is a specified test, not a result.

## Evidence Status

### Verified, bounded implementation evidence

| Evidence | Bounded finding | Claim boundary and source |
|---|---|---|
| Cohort-v2 pair-evaluation surface | The central evaluator declares a 3 by 3 grid of horizons `(1, 5, 15)` and modes `continuous`, `micro`, and `macro`. It enumerates eligible nonterminal states and retains requested horizons under terminal clamping. | This establishes the pair-evaluation scope, not model quality or a completed cross-policy result. [pair-evaluation contract](../docs/data_contracts/cohort_v2_exhaustive_pair_evaluation_v1.md#L6-L20) [issue-3 summary](../data/runtime_evidence/issue-3/cohort-v2-exhaustive-pair-evaluation-summary.json#L1) |
| Micro and macro transition supervision | Micro and macro transition supervision are implemented. The issue-7 pair-measurement surface contains 378 macro-mode available records. | Macro availability is limited. It does not establish macro generalization or a controller advantage. [issue-5 summary](../data/runtime_evidence/issue-5/cohort-v2-micro-experiment-summary.json#L1-L64) [issue-6 summary](../data/runtime_evidence/issue-6/cohort-v2-macro-experiment-summary.json#L1-L85) [issue-7 pair-measurement summary](../data/runtime_evidence/issue-7/cohort-v2-pair-measurement-summary.json#L1) |
| Endpoint violation measurement | Endpoint measurement covers only `excess_penetration` and `unsupported_stationary_or_floating_body`. | This is endpoint measurement, not dense-path plausibility. `illegal_contact` is not emitted. [physical-violation contract](../docs/data_contracts/cohort_v2_physical_violations_v1.md#L1-L11) [issue-7 summary](../data/runtime_evidence/issue-7/cohort-v2-pair-measurement-summary.json#L1) |
| Trajectory-optimal labels | Trajectory-optimal labels are implemented for the bounded cohort-v2 controller workflow. | They are labels for model selection. They do not establish terminal-outcome accuracy or policy effectiveness. [issue-8 summary](../data/runtime_evidence/issue-8/cohort-v2-trajectory-label-summary.json#L1) |
| Policy baselines | Four controller-free policy baselines are implemented. Issue 9 evaluates six states for each policy, exposure-role cell. | These baselines are not compute-matched to issue 10. Do not use issue-9 evidence for a matched-compute comparison. [issue-9 summary](../data/runtime_evidence/issue-9/cohort-v2-policy-baseline-summary.json#L1) |
| Controller comparison | A distilled joint-pair controller and a parameter-matched two-head controller are implemented. Their bounded model-selection comparison uses 1,600 states, agent-observable inputs, and excludes oracle engine state. The controllers have equal scores, with no observed advantage for the joint controller. The recorded endpoint violation rate is `0.0002777777777777778`. | This is model-selection evidence only. It is not final evaluation, terminal-outcome accuracy, or a central joint-versus-factorized advantage. [issue-10 summary](../data/runtime_evidence/issue-10/cohort-v2-controller-summary.json#L1) |
| Issue-11 aggregation | One aggregation round covers six rollouts and 109 controller decisions using aligned ground-truth-expert carrier continuation. The source cohort is not mutated, and the aggregation reports zero deltas against the oracle-state baseline. | This is neither a model closed-loop rollout nor evidence of terminal-outcome accuracy or effectiveness. [issue-11 summary](../data/runtime_evidence/issue-11/cohort-v2-controller-aggregation-summary.json#L1) |

### Blocked evidence

- **Final evaluation is Blocked.** Six final-evaluation rollouts have been collected and sealed. No final-evaluation metric has been derived or consumed because final-evaluation authorization is pending. A complete cohort release does not authorize final scoring or manuscript claims. [release record](../data/runtime_evidence/issue-53-mixed-termination-v5/cohort-v2-release.json#L541-L550) [partition and access contract](../docs/data_contracts/cohort_v2_partition_exposure_v1.md#L50-L66)
- **Central controller result is Blocked.** The matched-compute joint-versus-factorized advantage remains `[TODO: result]`. The issue-10 equal model-selection score is not evidence for or against the planned final comparison.
- **Terminal-outcome result is Blocked.** Terminal-outcome accuracy is specified as a common endpoint, but no final-evaluation metric is available for reporting while authorization is pending.

### Specified or unavailable evidence

- A common final-state readout and shared coordinate decoder are **Specified** but not available as a completed common evaluator. ADE, FDE, and event F1 are **Unavailable**.
- OOD, template-held-out, and cross-domain claims are **Unavailable**. The central contract provides instance-held-out exposure roles and explicitly creates no template-held-out score. [partition and exposure contract](../docs/data_contracts/cohort_v2_partition_exposure_v1.md#L24-L43) [capability declaration](../docs/data_contracts/cohort_v2_capabilities_v1.json#L15-L24)
- Dense-path plausibility is **Unavailable**. The two accepted violation checks are endpoint measurements only. [physical-violation contract](../docs/data_contracts/cohort_v2_physical_violations_v1.md#L1-L11) [accepted violation labels](../docs/data_contracts/cohort_v2_capabilities_v1.json#L81-L84) `illegal_contact` is **Unavailable** because it is excluded from the central capability declaration. [illegal-contact exclusion](../docs/data_contracts/cohort_v2_capabilities_v1.json#L23)
- The learned reliability gate, learned predicate parser, and SPSG are **Specified** components. Their benefit is **Unavailable** because no bounded source here evaluates it.
- Legacy backbone, smoke, and continuous-only temporal assertions formerly linked through retired evidence paths are **Unavailable** in this ledger. They must not be cited as current implementation evidence without a valid `data/runtime_evidence/issue-*` or `docs/data_contracts/*` source.

## Source Hierarchy

1. **Published benchmark record.** Use Pinto et al., *Artificial Intelligence* 336:104198 (2024), https://doi.org/10.1016/j.artint.2024.104198, only for published NovPhy claims.
2. **Contracts.** Use `docs/data_contracts/*` for declared scope, label semantics, capability limits, exposure roles, and authorization conditions.
3. **Runtime evidence.** Use `data/runtime_evidence/issue-*` only for the bounded implementation facts recorded by that artifact.
4. **Proposal material.** Treat any design outside the sources above as Specified, not empirical evidence.

## Manuscript Claim Boundaries

- Attribute NovPhy benchmark findings to the published NovPhy paper. Do not present them as BG-NS-JEPA results.
- Report the 3 by 3 pair surface, implemented supervision, endpoint measurements, trajectory labels, baselines, and controller comparison only at their recorded scope.
- Do not claim a joint-controller advantage. The central result remains `[TODO: result]`.
- Do not claim final-evaluation metrics, terminal-outcome accuracy, common final-state readout performance, shared coordinate decoding, ADE, FDE, event F1, dense-path plausibility, illegal-contact handling, OOD performance, template-held-out performance, cross-domain performance, learned-gate benefit, parser performance, or SPSG benefit.
- Do not describe issue 9 as controller-based or compute-matched to issue 10.
- Do not describe issue 11 as a model closed-loop rollout or terminal-outcome evidence.
- Do not treat cohort-release completion as final-evaluation authorization or manuscript authorization.

## Evidence-to-Provisional Six-Section Outline

| Section | Evidence-supported role | Provisional boundary |
|---|---|---|
| 1. Introduction | State the mechanism question and distinguish published NovPhy from BG-NS-JEPA. | The central advantage remains `[TODO: result]`. |
| 2. Related Work | Position the published benchmark and the specified evaluation problem. | Verify citations independently before drafting. |
| 3. Method | Describe the 3 by 3 pair scope and mark unevaluated components by status. | Do not turn specification into efficacy. |
| 4. Experiments | Report the bounded model-selection artifacts and the blocked final-evaluation boundary. | No final metric or controller-advantage table. |
| 5. Discussion | Explain what the bounded controller comparison does and does not show. | Do not infer effectiveness from equal scores or zero deltas. |
| 6. Conclusion | State the specified test and missing authorized final evidence. | No efficacy conclusion. |

## Unresolved Writing Decisions

- Retain `[TODO: result]` until a final authorized comparison supports or rejects the central claim.
- Decide whether a paper limited to bounded model-selection evidence is appropriate for the intended venue.
- Define the common final-state readout and coordinate-decoder protocol before adding terminal-outcome, ADE, FDE, or event-F1 claims.
