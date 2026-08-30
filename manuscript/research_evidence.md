# Manuscript Research Evidence Ledger

## Scope and Research Target

**NovPhy benchmark.** NovPhy is the published physical-reasoning benchmark for open-world AI systems. The published paper evaluates novelty detection and adaptation across five physical scenarios and eight novelty levels. The canonical citation is V. Pinto, C. Gamage, C. Xue, P. Zhang, E. Nikonova, M. Stephenson, and J. Renz, “NovPhy: A physical reasoning benchmark for open-world AI systems,” *Artificial Intelligence*, vol. 336, p. 104198, 2024. https://doi.org/10.1016/j.artint.2024.104198. It is distinct from the BG-NS-JEPA work described here.

**BG-NS-JEPA program.** BG-NS-JEPA is an in-progress world-model program for action-sparse persistent-effect environments. Its controller selects a requested horizon and description mode, while a continuous latent carries the rollout. The central claim remains unresolved: `[TODO: result]`.

**Central research question.** Does the joint-pair controller improve a common endpoint against `fixed-pair`, `temporal-only`, `description-only`, `independent-axes`, and the parameter-matched two-head controller? Only after final-evaluation authorization may one predeclared common-compute protocol evaluate that question. The result remains `[TODO: result]`.

## Evidence Status

### Verified, bounded implementation evidence

| Evidence | Bounded finding | Claim boundary and source |
|---|---|---|
| Cohort-v2 pair-evaluation surface | The central evaluator declares a 3 by 3 grid of horizons `(1, 5, 15)` and modes `continuous`, `micro`, and `macro`. It enumerates eligible nonterminal states and retains requested horizons under terminal clamping. | This establishes the pair-evaluation scope, not model quality or a completed cross-policy result. [pair-evaluation contract](../docs/data_contracts/cohort_v2_exhaustive_pair_evaluation_v1.md#L6-L20) [issue-3 summary](../data/runtime_evidence/issue-3/cohort-v2-exhaustive-pair-evaluation-summary.json#L1) |
| Micro and macro transition supervision | Micro and macro transition supervision are implemented. The issue-7 pair-measurement surface contains 378 macro-mode available records. | Macro availability is limited. It does not establish macro generalization or a controller advantage. [issue-5 summary](../data/runtime_evidence/issue-5/cohort-v2-micro-experiment-summary.json#L1-L64) [issue-6 summary](../data/runtime_evidence/issue-6/cohort-v2-macro-experiment-summary.json#L1-L85) [issue-7 pair-measurement summary](../data/runtime_evidence/issue-7/cohort-v2-pair-measurement-summary.json#L1) |
| Endpoint violation measurement | Endpoint measurement covers only `excess_penetration` and `unsupported_stationary_or_floating_body`. | This is endpoint measurement, not dense-path plausibility. `illegal_contact` is not emitted. [physical-violation contract](../docs/data_contracts/cohort_v2_physical_violations_v1.md#L1-L11) [issue-7 summary](../data/runtime_evidence/issue-7/cohort-v2-pair-measurement-summary.json#L1) |
| Trajectory-optimal labels | Trajectory-optimal labels are implemented for the bounded cohort-v2 controller workflow. | They are labels for model selection. They do not establish terminal-outcome accuracy or policy effectiveness. [issue-8 summary](../data/runtime_evidence/issue-8/cohort-v2-trajectory-label-summary.json#L1) |
| Policy baselines | The controller-free `fixed-pair`, `temporal-only`, `description-only`, and `independent-axes` policies are implemented. Issue 9 evaluates six states for each policy, exposure-role cell. | This is bounded controller-free evidence, not a matched-compute suite with issue 10. Recomputing these policies under the authorized final protocol remains `[TODO: result]`. [issue-9 summary](../data/runtime_evidence/issue-9/cohort-v2-policy-baseline-summary.json#L1) |
| Controller comparison | A distilled joint-pair controller and a parameter-matched two-head controller are implemented. Their bounded model-selection comparison uses 1,600 states, agent-observable inputs, and excludes oracle engine state. The controllers have equal scores, with no observed advantage for the joint controller. The recorded endpoint violation rate is `0.0002777777777777778`. | This is model-selection evidence only. It is not final evaluation, terminal-outcome accuracy, or the authorized prospective common-compute comparison. [issue-10 summary](../data/runtime_evidence/issue-10/cohort-v2-controller-summary.json#L1) |
| Issue-11 aggregation | One aggregation round covers six rollouts and 109 controller decisions using aligned ground-truth-expert carrier continuation. The source cohort is not mutated, and the aggregation reports zero deltas against the oracle-state baseline. | This is neither a model closed-loop rollout nor evidence of terminal-outcome accuracy or effectiveness. [issue-11 summary](../data/runtime_evidence/issue-11/cohort-v2-controller-aggregation-summary.json#L1) |
| Issue-57 held-out gameplay matrix | Five systems ran on five held-out levels with three seeds each, for 75 trials. Each system recorded `0/15` successes, yielding a complete zero-success floor and the disposition `not_supported_by_this_experiment`. | This is verified, bounded negative gameplay evidence for the frozen stack, packaged instance-held-out levels, seeds, limits, and decision rule. It does not establish equivalence, impossibility, causal training-data insufficiency, controller efficacy or inefficacy, or the manuscript's central claim. [issue-57 summary](../data/runtime_evidence/issue-57/cohort-v2-gameplay-success-summary-v2.json#L13-L15) [system results](../data/runtime_evidence/issue-57/cohort-v2-gameplay-success-summary-v2.json#L133-L198) [trial matrix](../data/runtime_evidence/issue-57/cohort-v2-gameplay-success-summary-v2.json#L200-L225) |
| Issue-57 adaptive-granularity use | Adaptive granularity was not materially exercised. Adaptive CEM/MPC requested `continuous-h15` for all 44 recorded decisions. | The gameplay floor cannot test variation across requested horizon-description pairs. [issue-57 interpretation and usage](../data/runtime_evidence/issue-57/cohort-v2-gameplay-success-summary-v2.json#L42-L56) |
| Issue-60 deployment temporal carrier | Issue 60, included in merge `6119b6c`, implements method infrastructure for a deployment-aligned temporal carrier. It uses aligned prior context for motion, an explicit motion-availability mask when that context is absent, complete trajectory atomicity, and one exposure role per scenario lineage. Oracle or canonical engine state is excluded from model and planner input except for declared supervision or alignment diagnosis. | This is implemented infrastructure, not an empirical result or a controller-effectiveness finding. [carrier input and motion contract](../world_model/data/deployment_temporal.py#L93-L141) [carrier construction](../world_model/data/deployment_temporal.py#L630-L705) [trajectory and role-isolation contract](../world_model/data/deployment_temporal.py#L311-L515) [input-isolation and contract tests](../tests/test_deployment_temporal_carrier.py#L196-L326) |

### Blocked evidence

- **Final evaluation is Blocked.** Six final-evaluation rollouts have been collected and sealed. No final-evaluation metric has been derived or consumed because final-evaluation authorization is pending. A complete cohort release does not authorize final scoring or manuscript claims. [release record](../data/runtime_evidence/issue-53-mixed-termination-v5/cohort-v2-release.json#L541-L550) [partition and access contract](../docs/data_contracts/cohort_v2_partition_exposure_v1.md#L50-L66)
- **Central controller result is Blocked.** Only after final-evaluation authorization may one predeclared common-compute protocol evaluate the joint-pair controller against `fixed-pair`, `temporal-only`, `description-only`, `independent-axes`, and the parameter-matched two-head controller. Recomputing every comparator under that protocol remains `[TODO: result]`. The issue-9 and issue-10 artifacts are not a matched-compute suite.
- **Terminal-outcome result is Blocked.** Terminal-outcome accuracy is specified as a common endpoint, but no final-evaluation metric is available for reporting while authorization is pending.

### Specified or unavailable evidence

- A common final-state readout and shared coordinate decoder are **Specified** but not available as a completed common evaluator. ADE, FDE, and event F1 are **Unavailable**.
- OOD, template-held-out, and cross-domain claims are **Unavailable**. The central contract provides instance-held-out exposure roles and explicitly creates no template-held-out score. [partition and exposure contract](../docs/data_contracts/cohort_v2_partition_exposure_v1.md#L24-L43) [capability declaration](../docs/data_contracts/cohort_v2_capabilities_v1.json#L15-L24)
- Dense-path plausibility is **Unavailable**. The two accepted violation checks are endpoint measurements only. [physical-violation contract](../docs/data_contracts/cohort_v2_physical_violations_v1.md#L1-L11) [accepted violation labels](../docs/data_contracts/cohort_v2_capabilities_v1.json#L81-L84) `illegal_contact` is **Unavailable** because it is excluded from the central capability declaration. [illegal-contact exclusion](../docs/data_contracts/cohort_v2_capabilities_v1.json#L23)
- The learned reliability gate, learned predicate parser, and SPSG are **Specified** components. Their benefit is **Unavailable** because no bounded source here evaluates it.
- Issues #61 through #65 are **Specified/Open** future work in this ledger. Their results remain `[TODO: result]`. Do not report an implementation or outcome for any of them.
- Legacy backbone, smoke, and continuous-only temporal assertions formerly linked through retired evidence paths are **Unavailable** in this ledger. They must not be cited as current implementation evidence without a valid `data/runtime_evidence/issue-*` or `docs/data_contracts/*` source.

## Source Hierarchy

1. **Published benchmark record.** Use Pinto et al., *Artificial Intelligence* 336:104198 (2024), https://doi.org/10.1016/j.artint.2024.104198, only for published NovPhy claims.
2. **Contracts.** Use `docs/data_contracts/*` for declared scope, label semantics, capability limits, exposure roles, and authorization conditions.
3. **Runtime evidence.** Use `data/runtime_evidence/issue-*` only for the bounded implementation facts recorded by that artifact.
4. **Proposal material.** Treat any design outside the sources above as Specified, not empirical evidence.

## Manuscript Claim Boundaries

- Attribute NovPhy benchmark findings to the published NovPhy paper. Do not present them as BG-NS-JEPA results.
- Report the 3 by 3 pair surface, implemented supervision, endpoint measurements, trajectory labels, baselines, and controller comparison only at their recorded scope.
- Do not claim a joint-controller advantage. The authorized prospective comparison against the five named comparators remains `[TODO: result]`.
- Do not claim final-evaluation metrics, terminal-outcome accuracy, common final-state readout performance, shared coordinate decoding, ADE, FDE, event F1, dense-path plausibility, illegal-contact handling, OOD performance, template-held-out performance, cross-domain performance, learned-gate benefit, parser performance, or SPSG benefit.
- Do not describe issue 9 as controller-based or compute-matched to issue 10.
- Do not describe issue 11 as a model closed-loop rollout or terminal-outcome evidence.
- Describe issue 57 only as the bounded zero-success gameplay matrix. It is not evidence of controller equivalence, impossibility, training-data causality, controller efficacy or inefficacy, or the central joint-pair comparison.
- Describe issue 60 as implemented deployment-temporal method infrastructure. Do not report it as an empirical result.
- Keep issues #61 through #65 as Specified/Open work with `[TODO: result]` outcomes.
- Do not treat cohort-release completion as final-evaluation authorization or manuscript authorization.

## Evidence-to-Provisional Six-Section Outline

| Section | Evidence-supported role | Provisional boundary |
|---|---|---|
| 1. Introduction | State the mechanism question and distinguish published NovPhy from BG-NS-JEPA. | The central advantage remains `[TODO: result]`. |
| 2. Related Work | Position the published benchmark and the specified evaluation problem. | Verify citations independently before drafting. |
| 3. Method | Describe the 3 by 3 pair scope and the implemented deployment-temporal carrier contract. Mark unevaluated components by status. | Do not turn implementation or specification into efficacy. |
| 4. Experiments | Report the bounded model-selection artifacts, issue-57 zero-success floor, and blocked central final-evaluation boundary. | No final metric or controller-advantage table. |
| 5. Discussion | Explain what the bounded controller comparison does and does not show. | Do not infer effectiveness from equal scores or zero deltas. |
| 6. Conclusion | State the specified test and missing authorized final evidence. | No efficacy conclusion. |

## Unresolved Writing Decisions

- Retain `[TODO: result]` until a final authorized comparison supports or rejects the central claim.
- Decide whether a paper limited to bounded model-selection evidence is appropriate for the intended venue.
- Define the common final-state readout and coordinate-decoder protocol before adding terminal-outcome, ADE, FDE, or event-F1 claims.
- Keep issues #61 through #65 as future-work items until evidence records an implementation and a result.
