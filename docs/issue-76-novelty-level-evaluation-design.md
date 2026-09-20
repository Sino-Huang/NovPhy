# Novelty-level evaluation: experiment design and readiness record

**Status:** experiment design + readiness record for
[issue #77](https://github.com/Sino-Huang/NovPhy/issues/77) (Sino-Huang/NovPhy).
Reframed 2026-09-20 as a **lightweight small-project program**: no authorization
gates, freeze ceremonies, or audit cycles — the stages below run in order as ordinary
experiments. Kept, because it is cheap and keeps results interpretable: decide cell
membership, templates, seeds and metrics before looking at outcomes; report every
attempted cell including failures; never report only the novelty on which a system
wins.

**Framing:** this program *tests whether and where* hybrid dynamics change under
novelty. It is not designed to "show advantage" on any novelty cell.

Executable plan: [issue-77-novelty-experiments-plan.md](issue-77-novelty-experiments-plan.md).
Consolidates under #35's exploratory stratum alongside
[issue-35-hybrid-profile-findings.md](issue-35-hybrid-profile-findings.md).

## 0. Direct answer: do novelty levels change performance?

**Partially tested (2026-09-20).** Four of 45 scenario×novelty cells now have
runtime evidence — normal rolling, normal sliding (stage N1 dynamics breadth), and
appearance×single-force / appearance×multiple-forces (stage N2, zero-shot and
few-shot). Appearance novelty moves the hybrid-minus-continuous comparison in
system- and horizon-specific directions; no uniform "novelty helps/hurts hybrid"
statement is supportable. All other cells remain untested with the blockers below.
Full record: [issue-77-novelty-experiments-results.md](issue-77-novelty-experiments-results.md).
Every trained checkpoint, diagnostic, and experiment in this program — including the
confirmatory #15, stress tests #16/#17, gameplay #57, the #73–#75 corrective line, the
A0.1 dynamics diagnostic, the R3 campaign, and the hybrid profile probes — uses
`novelty_level_0` only (families `type010101`/`type010102`; R3 added held-out
`type010105` attempts that never dispatched). Sources:
[`GENERATOR_FAMILIES`](../world_model/data/successor_cohort.py#L59),
[issue-76 novelty coverage assessment](issue-76-novelty-coverage.md).

Normal-only coverage is **not** an established cause of the negative hybrid results,
and a broader population could help, hurt, or leave every comparison unchanged. Nothing
in this document infers performance from benchmark breadth.

## 1. Readiness results (recorded now; metadata only)

Source: A0.2 inventory
[`.local-artifacts/issue-76-dynamics-diagnostic-v1/novelty-inventory.json`](../.local-artifacts/issue-76-dynamics-diagnostic-v1/novelty-inventory.json)
(`metadata_only: true`; `fresh_evaluation_opened: false`), and the
[coverage assessment](issue-76-novelty-coverage.md) (source inspection, no execution).

| Quantity | Recorded value |
| --- | --- |
| Scenario × novelty cells (5 scenarios × 9 categories incl. normal) | 45 |
| Templates / normal–novel pairs (upstream) | 80 / 40 |
| Cells with any runtime or performance test | **4 / 45** (2026-09-20: normal rolling, normal sliding, appearance×010101, appearance×010102) |
| Templates passing static slot+action fit vs the frozen 18-slot contract | 27 / 80 |
| Templates failing static fit (missing slots) | 48 / 80 |

Named per-category blockers (compatibility hypotheses from source inspection, not
reproduced runtime failures):

- **Normal falling** (`type010104`): 8 platforms vs the frozen 6-platform slot
  vocabulary — even normal coverage is not drop-in.
- **Appearance novelty** (level 1): same-layout PinkBigPig substitution; risks the
  shared CNN perception rather than dynamics; useful as a perception control.
- **Right-side slingshot** (level 5): current action contract is left-pull only
  (drag-x ∈ [−160,−10]); requires an action-contract extension.
- **Fan / stronger turbulence / magnetic / reversed gravity / storm** (levels 2,3,4,6,8):
  novel forces or event agents with no represented entity, no observation-history
  support, and no predicate semantics — deployed predicates are contact/support plus
  steady-state/structure-unstable only. A hidden force can affect visible motion
  without being identifiable from the permitted observation history; sufficiency must
  be checked, not assumed.
- **Changed turbulence goal** (level 7): task-objective change; evaluation semantics
  must be redeclared before any comparison.

## 2. Staged experiment design

Stages run in order (N0 → N1 → N2). Membership, templates, seeds and metrics are
decided before looking at model outcomes on any novelty cell.

### Stage N0 — Representation readiness (prerequisite, per cell)

For every unsupported cell, a minimal, symmetric, versioned amendment: slot/vocabulary
changes, action-contract extension, observation-history or event-semantics additions,
and the retraining both comparison arms require under identical budgets. No silent
one-arm refits, dropped entities, or resized frozen checkpoints. A cell we do not fund
is recorded `unsupported`, not dropped.

### Stage N1 — Normal-mechanics expansion (rolling, falling, sliding)

- Membership: metadata-selected normal templates from `type010103/04/05`, decided
  before any rendered smoke; count and exact identities pinned up front, disjoint from
  all previously exposed sets and future pools.
- Estimands: the established paired fixed-system contrasts (independent continuous vs
  hybrid-trained continuous at h ∈ {1,5,15}; recursive carrier error at common physical
  endpoints; ranked-action regret; ordinal09/no-model prior retained).
- Claim scope: normal-mechanics breadth only. No novelty claim from this stage.

### Stage N2 — Matched normal/novel pairs

- Membership: normal/novel pairs of the **same `type` family** under different novelty
  directories (upstream's 40-pair mapping; never filename-similarity pairing). Subset
  size decided after the metadata audit, before any rendered smoke; a small subset
  supports only a small-subset claim, with deferred cells explicitly untested.
- Two separate claims, never mixed:
  - **Zero-shot novelty generalization** — frozen models, no fitting on the target
    novelty; report normal vs novel performance separately plus the paired change in
    the hybrid-minus-continuous effect with uncertainty.
  - **Few-shot adaptation** — identical declared training data, update budgets, and
    seed pairing for both arms; held-out novel lineages for evaluation. Broadening
    training and evaluation together is not a zero-shot test.
- Controls retained: independent pure-continuous, same-hybrid-checkpoint fixed modes,
  ordinal09 prior; appearance-novelty cell as the perception control.
- Multiplicity/family weighting declared per stage; never report only the novelty on
  which a system wins.

### Resource floor (from R3 actuals)

R3 cost ≈563 s/branch mean (late-ordinal mean 774 s), 96 worker-h for 616 branches.
Any N1/N2 cell requires the same per-branch physics capture **after** its N0
representation work. Even a minimal N1 (3 scenario families × 8 lineages × 13 branches
= 312 branches) is ≈49 worker-h of capture alone at the early-ordinal rate (≈6–7 h
wall at 8 workers) — before representation amendments, training, and evaluation.

## 3. Boundaries

- Performance evidence exists only for the four executed cells and is recorded in
  [issue-77-novelty-experiments-results.md](issue-77-novelty-experiments-results.md);
  every other cell remains metadata/design only.
- Does not modify #15/#74/#75/#76 dispositions; does not unblock #64/#65.
- If the v2 bounded-transfer campaign
  ([proposal](issue-76-b-plus-v2-campaign-proposal.md)) is ever run, keep this
  program's lineages/seeds disjoint from its inventory as ordinary coordination.

## 4. Manuscript usage

Limitations/future-work only, with wording equivalent to: *"Experiments use
normal-condition templates (`novelty_level_0`) plus four runtime-tested
scenario×novelty cells (normal rolling, normal sliding, appearance×single-force,
appearance×multiple-forces; 4/45 cells). The remaining novelty categories are
untested, and 48/80 upstream templates are statically incompatible with the
frozen perception contract, so novelty-level generalization remains an explicitly
open question with a staged evaluation design."* — counts updated 2026-09-20 as
cells were tested; see the results record.
