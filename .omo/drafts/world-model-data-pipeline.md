---
slug: world-model-data-pipeline
status: reviewing
intent: unclear
review_required: true
plan_path: .omo/plans/world-model-data-pipeline.md
plan_sha256: f6b49cf1861b2d170aea641773b6f047116488631848d434d9021763f78d35ec
review_round_id: world-model-data-pipeline-20260727-r1
round_status: inconclusive
pending-action: review .omo/plans/world-model-data-pipeline.md
review:
  momus:
    status: approved
    workspace_root: /mnt/array/sukaih/Project/NovPhy
    runtime_home: null
    target: .omo/plans/world-model-data-pipeline.md
    round_id: world-model-data-pipeline-20260727-r1
    plan_sha256: f6b49cf1861b2d170aea641773b6f047116488631848d434d9021763f78d35ec
    launch_id: momus-20260727-r1
    session: native-momus-agent:/root/momus_plan_review
    result: "OKAY; descriptor-relative read verified the exact plan SHA-256 f6b49cf1861b2d170aea641773b6f047116488631848d434d9021763f78d35ec"
  independent:
    status: inconclusive
    workspace_root: /tmp/novphy_plan_review_zrogiW
    runtime_home: /tmp/novphy_codex_home_mcsltC
    target: .omo/plans/world-model-data-pipeline.md
    round_id: world-model-data-pipeline-20260727-r1
    plan_sha256: f6b49cf1861b2d170aea641773b6f047116488631848d434d9021763f78d35ec
    launch_id: codex-20260727-r1
    session: codex-cli:019fa387-de0b-7df0-bf3c-5341197a19af
    result: "inconclusive: isolated Codex CLI runtime returned HTTP 401 before descriptor validation or plan review"
approach: Build a standalone, typed PyTorch world-model data package around immutable canonical episode indexes, lazy deterministic temporal-window sampling, and declarative curriculum/ablation policies. Reuse the collector's canonical-completeness contract without mutating the active collection. Scope this plan to image/action temporal prediction and temporal-granularity experiments only. Add test-split collection support as a separate, explicitly tested capability because the current root has train/dev only.
---

# Draft: world-model-data-pipeline

## Components (topology ledger)
<!-- Lock the SHAPE before depth. One row per top-level component that can succeed or fail independently. -->
<!-- id | outcome (one line) | status: active|deferred | evidence path -->
1 | Canonical episode catalog admits only complete, safe artifacts while collection continues | active | scripts/prepare_rollout_dataset.py:326-438
2 | Lazy trajectory-window dataset exposes image/action/metadata samples for configurable temporal horizons | active | scripts/collect_rollouts.py:1158-1190, 1219-1360
3 | Declarative split, temporal curriculum, and continuous-world-model ablation policy selects samples without data duplication or leakage | active | docs/research_proposal.md:237-246
4 | Deterministic sampler and collator support fixed-stride and temporal-only studies at matched sample and compute budgets | active | docs/training_mechanism_and_architecture_specs.md:242-249,381-398
5 | Inspection/reporting surface proves catalog health, split coverage, temporal-window availability, and curriculum composition | active | data/novphy_rollouts_dataset_20260708_171531
6 | Test split can be collected and consumed without changing the train/dev contract | active | scripts/prepare_rollout_dataset.py:508-528; scripts/collect_full_rollout_training_dataset.sh:232-251

## Open assumptions (announced defaults)
<!-- Intent is UNCLEAR: research resolves ambiguity, defaults are adopted (not asked), and each is surfaced in the plan's human TL;DR for veto. -->
<!-- assumption | adopted default | rationale | reversible? -->
Framework | PyTorch Dataset/DataLoader with no Lightning dependency | Existing repository training code uses torch.utils.data; no current world-model training package exists | yes
Read consistency | Build an immutable in-memory catalog only from canonically complete episodes; rescan only on explicit refresh | Collector reserves and writes episode directories non-atomically, so live filesystem iteration is unsafe | yes
Sample unit | One accepted shot plus a configurable temporal window drawn from its contiguous raw frames | Metadata provides exact frame lists and action/protocol context; this supports fixed-stride and variable-horizon JEPA targets | yes
Split policy | Consume collector train/dev directories as-is; add a separate test-collection mode before reporting held-out results | Current data root has no test episodes; train/dev leakage must not be repaired by re-splitting samples | yes
Curriculum policy | Schedule novelty level, scenario/type, stride, horizon, and episode progress only | These fields exist in the current episode layout or can be derived from ordered frame paths; this directly supports temporal-granularity experiments | yes
Image handling | Decode RGB frames lazily in workers; make resize/normalization an explicit configuration, not a hidden default | Raw frames are PNG and future JEPA encoder resolution is not yet fixed | yes

## Findings (cited - path:lines)
- `data/novphy_rollouts_dataset_20260708_171531` currently has 4,914 contract-complete train episodes, 405 dev episodes, 18 train partial/other episodes, and no test episodes; each complete episode exposes 12 accepted shots.
- `scripts/prepare_rollout_dataset.py:326-438` provides the existing canonical-completeness contract: bounded non-symlink paths, readable manifest/action logs, matching capture contract, no retry exhaustion, canonical accepted attempts, and all expected raw frame artifacts.
- `scripts/prepare_rollout_dataset.py:468-528` partitions and schedules train/dev only; `scripts/collect_full_rollout_training_dataset.sh:232-251` launches those two splits.
- `scripts/collect_rollouts.py:828-920` validates a shot's metadata and gameplay motion; `scripts/collect_rollouts.py:1158-1190` loads action logs; `scripts/collect_rollouts.py:1219-1360` writes raw frame and metadata artifacts.
- `scripts/collect_rollouts.py:1582-1598,1744-1775` writes manifests and action logs directly, so collection exposes incomplete directories and transient JSON/frame states to a concurrent reader.
- `docs/research_proposal.md:235-274` requires fixed, temporal-only, abstraction-only, factorized, and joint-controller comparisons at matched compute.
- `docs/training_mechanism_and_architecture_specs.md:242-249,363-378` specifies joint-controller ablations and effective prediction-step/controller-cost metrics.
- Existing PyTorch usage is in legacy agent training (`sciencebirdsagents/TrainLearningAgent.py:11-14`); no current world-model data package exists.

## Decisions (with rationale)
- Treat the canonical collector predicate as the source contract, but move a reusable non-private equivalent into the new data package rather than importing a planning-script private helper.
- Keep collection, indexing, sampling, curriculum, and training separate. The loader must never create, repair, delete, or resplit raw collection artifacts.
- Treat `novelty_level_*` and `type010*` as episode metadata for filtering and stratification; never infer a held-out test set from individual frames or shots.
- Make curriculum progression training-state input, not hidden sampler state, so runs are resumable and ablations are reproducible.
- Do not infer symbolic, physical, or event-regime labels from image/action metadata. Those requirements belong to a separate future engine-instrumentation plan.

## Scope IN
- World-model dataset/index, typed image/action temporal-sample contract, lazy frame decoding, deterministic sampler/collator, temporal curriculum and stride/horizon ablation policies, inspection/reporting CLI, unit and integration tests, and test-split collection plumbing.

## Scope OUT (Must NOT have)
- No JEPA backbone, controller implementation, symbolic/reliability supervision, scene-graph perception model, engine instrumentation, trainer, distributed training stack, retroactive mutation of active raw episodes, fabricated oracle labels, or new external training framework.

## Open questions
- None. The defaults above are reversible; the plan will make encoder image transforms and curriculum schedules explicit configuration surfaces.

## Approval gate
status: review-inconclusive
The approved decision-complete plan has been written. Native Momus approved the bound artifact; the independent isolated Codex CLI lane was inconclusive because its supplied runtime credential returned HTTP 401 before it could validate or read the plan. No implementation has started.
<!-- When exploration is exhausted and unknowns are answered, set status: awaiting-approval. -->
<!-- That durable record is the loop guard: on a later turn read it and resume at the gate instead of re-running exploration. -->
