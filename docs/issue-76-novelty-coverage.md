# #76: scenario and novelty coverage assessment

Date: 2026-09-10. Source inspection only; no new captures, fitting, model scoring, or final-evaluation access. This note motivates a prospective compatibility audit, not a claim that changing the population will make hybrid dynamics win.

## What the current experiments cover

The #62 generator fixes `novelty_level_0` and two families: `type010101` and `type010102`. They are the normal single-force and multiple-forces scenarios, respectively. #68 imports the same family contract; #74 reuses the #71 training shards and #72 development plan. More generated lineages therefore increased within-family variation, not coverage of novel conditions or the dedicated rolling/falling/sliding scenario families. Sources: [family and action contract](../world_model/data/successor_cohort.py), [materialization](../scripts/run_issue_62_successor_cohort.py), [#68 plan](../world_model/data/corrective_ranking_cohort.py), [#74 reuse](../scripts/run_issue_74_matched_dynamics.py).

The upstream benchmark separates five physical tasks—single force, multiple forces, rolling, falling, sliding—from eight novelty categories: changed object appearance, a fan agent, stronger turbulence action, magnetic interactions, a right-side slingshot, reversed gravity, changed turbulence goal, and a storm triggered by bird death. It supplies 40 normal/novel template pairs. `novelty_level_0` denotes normal templates; suffixes `01` through `05` identify the physical scenario, not novelty severity. Source: [upstream README, scenarios, novelties and directory naming](https://github.com/phy-q/NovPhy/blob/main/README.md#1-physical-scenarios-in-novphy).

Thus coverage is narrow, but **normal-only coverage is not an established cause of the negative hybrid result**. Existing tasks already contain contact and support transitions. A broader population could help, hurt, or leave the comparison unchanged. The #73 controller coverage problem and #74/#75 negative outcomes must remain recorded: [#73 findings](issue-73-findings.md), [#74](https://github.com/Sino-Huang/NovPhy/issues/74), [#75](https://github.com/Sino-Huang/NovPhy/issues/75).

## Concrete template candidates and compatibility limits

These are candidates for technical inspection, not approved experimental membership. Normal/novel pairs must use the same `type` family under different novelty directories, rather than pairing arbitrary layouts after seeing outcomes.

| Purpose | Local template reference | Inspection result / concern |
| --- | --- | --- |
| Current normal multiple-force reference | `novelty_level_0/type010102/Levels/00001_0_1_010102_0_2.xml` | Three RedBirds, one BasicBig pig, three platforms. |
| Matched appearance novelty | `novelty_level_1/type010102/Levels/00001_0_1_010102_1_2.xml` | Same listed layout, PinkBigPig replaces BasicBig. Tests visual shift, not solely dynamics. |
| Additional normal rolling | `novelty_level_0/type010103/Levels/00001_0_1_010103_0_3.xml` | One stone circle and four platforms; slot count alone does not prove perception generalization. |
| Additional normal falling | `novelty_level_0/type010104/Levels/00001_0_1_010104_0_4.xml` | Three circular blocks and **eight platforms**: exceeds the frozen six-platform vocabulary. |
| Additional normal sliding | `novelty_level_0/type010105/Levels/00001_0_1_010105_0_5.xml` | One stone RectFat block and two platforms; inspect geometry/perception and action reachability. |
| Prospective reversed-gravity pair | `novelty_level_0/type010601/Levels/00001_0_1_010601_0_1.xml` and `novelty_level_6/type010601/Levels/00001_0_1_010601_6_1.xml` | Requires explicit novelty-object handling and observable dynamics audit before any evaluation. |

References in the table are relative to [tasks/task_templates](../tasks/task_templates). Templates were inspected as local source, not executed. Several XML files declare UTF-16 despite containing ASCII bytes; use the existing materialization reader rather than assuming all files have identical encoding.

The frozen #71/#74 contract has 18 authored-ID slots: three birds, five blocks, one pig, six platforms, one slingshot, and two landscape slots. Its learned carrier is 236 values. Exact source: `.local-artifacts/issue-71-hybrid-readiness-v1/plan.json` → `contract.vocabulary`; [checkpoint rejection and fixed dimensions](../world_model/training/cnn_hybrid.py). **Even some normal tasks are therefore not drop-in inputs.** Do not silently drop excess objects, remap IDs, increase the vocabulary or retrain only one comparison arm.

The deployed hybrid predicates are contact/support plus steady-state/structure-unstable. No storm-event or explicit novel-force state head is instantiated. Runtime inputs are carrier, action and remaining steps, not a privileged novelty ID. A novel force can affect visible motion without its hidden cause being identified; whether the available observation history is sufficient must be checked rather than assumed. Source: [hybrid runtime](../world_model/training/cnn_hybrid.py), [label names and temporal-window construction](../scripts/run_issue_71_hybrid_readiness.py).

Current drag-x bounds are negative (`[-160,-10]`), with drag-y `[-80,80]` and release duration 600 ms. The current collector is also explicitly normal-condition materialization. Right-side slingshot tasks therefore require a versioned action/interface contract, not simply additional paths. Novel appearance may break shared perception; fan/magnet/gravity/storm conditions may require new represented entities, observation history or event semantics. These are compatibility hypotheses to audit, not already reproduced runtime failures. Sources: [action bounds](../world_model/data/successor_cohort.py), [collector condition](../scripts/run_issue_62_successor_cohort.py), [upstream novelty definitions](https://github.com/phy-q/NovPhy/blob/main/README.md#2-novelties-in-novphy).

## Lightweight recommendation for #76

1. First use existing checkpoints/data to separate fixed-horizon dynamics comparisons from controller comparisons. Report recursive error at common physical times and action-ranking regret separately. This does not require expanded collection.
2. Add a metadata-only feasibility matrix over the five scenarios and eight novelty categories: required object IDs/counts, image/geometry range, legal action direction, visible versus hidden force/event information, capture labels, and source/role restrictions. Record unsupported cells explicitly.
3. Before any live smoke, freeze a small metadata-selected list and hard time/frame/disk limits. Start with additional normal rolling/sliding compatibility and one prospectively chosen matched normal/novel pair only if it fits the existing interface. Preserve all attempted cells and technical failures. Membership must not depend on hybrid score, pig removal or perceived solvability.
4. Inspect RGB video, parser correctness, represented objects, action reachability, physical timing and label availability. This is an engineering smoke—not statistical evidence of superiority. A representation/action failure stops that cell; substantial repair needs a separately costed, symmetric, versioned change and explicit approval.
5. If feasible and worthwhile before the deadline, separately freeze fresh non-final comparison membership, effect size/sample-size reasoning, budgets, baselines and stop rule. Keep original negative results and normal/novel strata distinct. Fixed-model zero-shot transfer and training/adaptation on novel examples are different claims; choose one prospectively. If the required study is too expensive, publish limited coverage rather than open an unbounded novelty sweep.

Do not add all eight novelties now, select the most favorable novelty after inspecting comparative scores, or use this diagnostic expansion to authorize #64. Any improvement on new tasks must be established by a fair new experiment, not inferred from benchmark breadth.
