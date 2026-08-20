# Why issue #33 is needed

## Short answer

[Issue #33, “Assemble the issue #2 completion evidence”](https://github.com/Sino-Huang/NovPhy/issues/33), is the final audit before anyone can say that the enriched research-data milestone is complete.

Issues #31 and #32 answered two narrower questions: “Did we publish a real, frozen cohort?” and “Can the experiment code read it safely?” Issue #33 asks the larger question: **Does all of the evidence, taken together, satisfy every requirement for closing issue #2?** It must point to evidence for every requirement and say `pass` or `fail`; it must not hide missing capabilities behind a successful command, fixture, or small smoke run ([issue #33](https://github.com/Sino-Huang/NovPhy/issues/33); [data specification, lines 321–333](../../../../docs/data_generation_and_collection_spec.md#L321-L333)).

In simple words: #31 made the package, #32 proved the package can be opened safely, and #33 checks whether the package is scientifically complete enough for the experiments we want to claim.

## Why the earlier tickets are not enough

The release from #31 is real and useful, but deliberately small: four accepted Unity rollouts from two scenario lineages. It has collision, destruction, stability-transition, and level-fail evidence, and it records 15 capabilities or labels as unavailable or excluded ([release README, lines 3–19](../../evidence/representative-cohort-release-20260820/README.md#L3-L19)). The #31 completion comment says explicitly that this lightweight release does **not** claim the broader issue #18 definition of done ([issue #31 completion comment](https://github.com/Sino-Huang/NovPhy/issues/31#issuecomment-5350236251)).

Issue #32 then proved that the public readers ingest this exact release and stop on missing, malformed, or mismatched evidence. Its proof preserves the fixed-step events, terminal observations, identities, and `unavailable` labels ([release README, lines 28–34](../../evidence/representative-cohort-release-20260820/README.md#L28-L34); [issue #32 completion comment](https://github.com/Sino-Huang/NovPhy/issues/32#issuecomment-5351051979)). That is necessary, but it only satisfies the downstream-ingestion part of the larger checklist.

The larger checklist also asks whether the release is representative, whether each rollout is fully valid, whether coverage and failures are completely reported, whether training and final-evaluation data are isolated, whether every required derived label is accepted and bound to this release, and whether systematic exporter defects are gone ([data specification, lines 321–331](../../../../docs/data_generation_and_collection_spec.md#L321-L331)). Issue #33 is needed to evaluate all seven conditions in one reviewable place.

## Why this matters to the research experiments

The paper’s main idea is that the model should jointly choose how far ahead to predict and whether to use a continuous, micro-relational, or macro-event description. The research data must therefore contain trustworthy engine state, relations, events, labels, timing, and train/evaluation boundaries—not just images ([research proposal, lines 278–307](../../../../docs/research_proposal.md#L278-L307)).

The connection to each experiment stage is direct:

1. **Stage 1: oracle-symbol upper bound.** We score every time-horizon/description-level pair using engine truth and use the best pair as teacher data for the controller ([research proposal, lines 280–286](../../../../docs/research_proposal.md#L280-L286)). If macro labels, contact facts, or timing are unavailable or not accepted, the full pair grid cannot be trained or scored honestly.

2. **Stage 2: learned symbols.** The visual parser is compared with engine labels on held-out levels, and the reliability gate is calibrated against oracle regimes ([research proposal, lines 288–296](../../../../docs/research_proposal.md#L288-L296)). Without proven lineage splits and accepted labels, this comparison can leak training examples into evaluation or treat “unknown” as “false.”

3. **Stage 3: the central paper experiment.** The joint controller is compared with fixed, single-axis, and factorized controllers using endpoint correctness, physical plausibility, compute, and out-of-distribution generalization ([research proposal, lines 298–307](../../../../docs/research_proposal.md#L298-L307)). Missing physical-violation labels or unsupported macro predicates mean some planned metrics or controller choices are not yet available.

4. **Stage 4 and final evaluation.** The full model is evaluated on standard and novelty scenarios ([research proposal, lines 309–316](../../../../docs/research_proposal.md#L309-L316)). The data rules require frozen evaluation inputs and access controls so final-evaluation evidence cannot influence training or model selection ([data specification, lines 293–311](../../../../docs/data_generation_and_collection_spec.md#L293-L311)).

The execution plan states the consequence plainly: no enriched cohort means symbolic and micro/macro evidence is blocked, and accepted data/oracle dependencies must come before the later research claims ([execution plan, lines 104–118](../../../../docs/high_level_plans/bg_ns_jepa_research_execution.md#L104-L118); [lines 232–250](../../../../docs/high_level_plans/bg_ns_jepa_research_execution.md#L232-L250)).

## What #33 protects us from

Without this ticket, the team could see “release published” and “reader test passed” and close issue #2 too early. That would make it easy to run experiments that produce numbers but do not support the paper’s claims—for example, because template-held-out evaluation is unavailable, a physical-violation metric has no accepted labels, or a macro predicate was rejected on representative evidence.

The current release summary already says that material, damage, physical-violation, physical-regime, illegal-contact, template-held-out, replay, access-separated observation, and several macro capabilities are unavailable or excluded ([release README, lines 17–18](../../evidence/representative-cohort-release-20260820/README.md#L17-L18)). The specification says unavailable capabilities must stay unavailable, and that fixture success, smoke success, command success, or merely reaching a rollout count cannot close issue #2 ([data specification, lines 313–333](../../../../docs/data_generation_and_collection_spec.md#L313-L333)).

Therefore #33 may legitimately conclude that some conditions fail and issue #2 must remain open. That is not a failed ticket; it is the scientifically correct result of the audit. Its value is a clear boundary between:

- experiments the current lightweight cohort really supports;
- experiments or metrics that must remain unavailable; and
- further data work required before the full BG-NS-JEPA claims can be tested.

## Practical definition

Issue #33 should produce one evidence map that lets a reviewer answer:

> For every issue #2 completion requirement, where is the proof, does it pass, and which planned experiments are allowed by that result?

That evidence map is what makes later model results traceable to an exact cohort, exact labels, exact splits, and exact known limitations. It is the bridge from “the pipeline ran” to “this experiment is scientifically defensible.”

## Primary sources

- [GitHub issue #33](https://github.com/Sino-Huang/NovPhy/issues/33) (no comments as of 2026-08-20)
- [GitHub issue #2](https://github.com/Sino-Huang/NovPhy/issues/2)
- [GitHub issue #18](https://github.com/Sino-Huang/NovPhy/issues/18)
- [GitHub issue #31 and completion comment](https://github.com/Sino-Huang/NovPhy/issues/31#issuecomment-5350236251)
- [GitHub issue #32 and completion comment](https://github.com/Sino-Huang/NovPhy/issues/32#issuecomment-5351051979)
- [Domain context](../../../../CONTEXT.md)
- [Data generation and collection research specification](../../../../docs/data_generation_and_collection_spec.md)
- [Research proposal](../../../../docs/research_proposal.md)
- [High-level research execution plan](../../../../docs/high_level_plans/bg_ns_jepa_research_execution.md)
- [Representative cohort release README](../../evidence/representative-cohort-release-20260820/README.md)
