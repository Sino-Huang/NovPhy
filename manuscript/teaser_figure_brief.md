# First-Page Teaser Figure Brief: BG-NS-JEPA

## Audience and Purpose

**Audience:** ICLR readers who need the falsifiable mechanism before architecture detail.

**Purpose:** show the granularity mismatch in an action-sparse persistent-effect rollout, then present BG-NS-JEPA as the NovPhy instantiation. A controller selects one coupled `(requested horizon, description mode)` pair while a continuous latent remains the rollout carrier. The controller has bounded model-selection evidence but no observed advantage. This figure is a mechanism and protocol figure, not final-evaluation evidence.

**Selected direction:** use the balanced three-part composition of persistent cascade, coupled decision and continuous carrier, and a neutral prospective comparison strip.

## Recommended Composition

Use a full-width horizontal figure (`\textwidth`, about `2.3:1`), read left to right and then down.

### A. Mechanism Problem: Persistent Cascade

Draw one stylized physical scene as four small fixed-step vignettes: intervention, collision onset, active cascade, settled endpoint. Use simple blocks, a projectile arc, and contact marks. Do not use a game screenshot, score, or success icon. Join the vignettes with a thin gray time arrow labelled `fixed steps`. Put a sparse intervention marker only above the first vignette. Beneath all four, draw one unbroken teal ribbon labelled `continuous carrier $z$`.

Overlay neutral short, adaptive, and long brackets. The visual claim is only that one recorded intervention can produce an autonomous heterogeneous cascade. Add `NovPhy instantiation` in small gray type at the panel foot.

### B. Coupled Decision and Carrier

Place `BG-NS-JEPA (bounded model selection)` between the timeline and a compact loop. A state-summary chip, `$h_t$: uncertainty, event signals, relation reliability`, feeds the controller. One prominent orange arrow goes to a `joint pair $(\Delta,\alpha)$` matrix with rows `1`, `5`, and `15` requested fixed steps and columns `continuous`, `micro`, `macro`. Highlight exactly one cell with an orange outline and the label `select one pair`. Every other cell stays neutral. The cell is schematic and does not indicate a preferred policy.

One blue arrow leads from the matrix to `$F^{\Delta,\alpha} \rightarrow \hat z_{t+\Delta}$`. Return `\hat z_{t+\Delta}` to the next `$h$` state with a teal loop arrow. Attach dashed side readouts, `micro readout` and `macro readout`, ending at `constraints / readouts`. Place `$z$ carries every rollout step` under the loop. Description mode conditions the prediction task and readout. It never replaces the continuous carrier.

### C. Prospective Comparison and Reserved Result Field

Divide the quieter lower band with a thin rule and title it `Prospective authorized common-compute protocol`. The strip does not report a current matched-compute suite. Only after final-evaluation authorization may the protocol evaluate the joint-pair controller against `fixed-pair`, `temporal-only`, `description-only`, `independent-axes`, and the parameter-matched two-head controller. Four equal, unfilled controller-free baseline labels and two neutral controller labels feed one common outlined box:

`fixed-pair` | `temporal-only` | `description-only` | `independent-axes` | `two-head` | `joint pair`

`independent-axes` is the fourth controller-free policy baseline. `two-head` is the separate parameter-matched controller. The current issue-9 and issue-10 artifacts are bounded evidence only. Recomputing all five comparators under the authorized protocol remains `[TODO: result]`.

Label the box `common final interface [TODO: result]`. After it, show neutral tokens `terminal outcome [TODO: result]`, `two endpoint checks`, and `compute [TODO: result]`. Put `final authorization pending` beneath the box. At the far right, reserve a small empty, thin-gray outlined rectangle labelled only `Matched-compute result [TODO: result]`.

The rectangle must remain blank white. It has no visual connection to a comparator, endpoint, or model. Do not use axes, bars, ranks, curves, checkmarks, numerical values, arrows, color, or emphasis in the reservation. The figure must not imply that issue-9 baseline compute matches issue-10 controller compute.

After final authorization and predeclared scoring, replace only the reservation with the factual common-interface result. Retain its boundary, size, position, and neutral surrounding comparison strip. Do not change the cascade or controller panels to imply a favorable result.

### Visual Hierarchy, Arrows, and State Encoding

1. Primary elements are the cascade and orange coupled-pair matrix.
2. The continuous carrier loop is secondary and resolves the representation role.
3. The prospective comparison strip is tertiary.
4. Solid arrows mean proposed inference or rollout flow. Dashed purple arrows mean readout or reliability-gated constraint flow. Do not draw separate horizon and mode decision arrows.
5. Use `$z_t$` and `$\hat z_{t+\Delta}$` only for continuous carrier states. Do not show hard predicates as recurrent state.

### Color and Style

Use the existing light Okabe-Ito palette: blue `#0072B2` for continuous prediction, teal `#009E73` for the carrier, orange `#E69F00` for the coupled decision, purple `#CC79A7` only for readout or gating accents, and gray `#5A5A5A` for neutral structure. Use a white background, thin dark-gray outlines, 10--18% tints, sans-serif text, and no gradients. Color encodes role, never quality, validity, or superiority.

## Alternatives

1. **Cascade-timeline dominant:** use the upper two-thirds for a four-regime timeline with schematic pair badges and the comparison strip below. Retain it only if each badge says `schematic`.
2. **Pair-grid dominant:** center the cross-product matrix, with a small cascade left, carrier loop right, and comparison strip below. This distinguishes coupled from independent-axes choice but can obscure the physical mechanism.

## Exact On-Figure Text

Use only the following strings with manuscript math formatting:

```text
Action-sparse persistent-effect cascade
one intervention
fixed steps
continuous carrier z
BG-NS-JEPA (bounded model selection)
h_t: uncertainty, event signals, relation reliability
joint pair (Delta, alpha)
1   5   15
continuous   micro   macro
select one pair
F^(Delta, alpha) -> z-hat_(t+Delta)
z carries every rollout step
micro readout
macro readout
constraints / readouts
Prospective authorized common-compute protocol
fixed-pair   temporal-only   description-only   independent-axes   two-head   joint pair
common final interface [TODO: result]
terminal outcome [TODO: result]   two endpoint checks   compute [TODO: result]
final authorization pending
Matched-compute result [TODO: result]
NovPhy instantiation
```

Do not add a slogan, performance adjective, dataset size, result number, physics analogy text, or a visual statement that the joint controller has an advantage.

## Symbols and Legend

| Item | Meaning |
|---|---|
| `$z_t$, \hat z_{t+\Delta}$` | Continuous predictive latent and sole rollout carrier. |
| `$h_t$` | Controller summary state. |
| `$\Delta` | Requested horizon of 1, 5, or 15 fixed steps. It is distinct from a terminal-clamped effective horizon. |
| `$\alpha` | Requested mode: continuous, micro, or macro. |
| Orange outlined cell | One schematic coupled choice, not an empirical selection. |
| Solid arrow | Proposed rollout or inference flow. |
| Dashed purple arrow | Proposed readout or reliability-gated constraint, not recurrent state flow. |
| Colors | Prediction, carrier, decision, and readout roles only. |

## Explicit Prohibited Visual Claims

- Do not use arrows, ordering, checkmarks, stars, curves, or color intensity to suggest that the joint controller wins, is Pareto-optimal, or has a demonstrated advantage.
- Do not describe issue-9 baselines as compute-matched to issue 10.
- Do not imply a final-evaluation metric, terminal-outcome accuracy, or controller effectiveness. Final evaluation is sealed and authorization is pending.
- Do not show a macro predicate as accepted supervision beyond the neutral `macro readout` label.
- Do not show a penetration, floating, or contact illustration that implies dense-path plausibility, `illegal_contact` measurement, or improved physical plausibility. The `two endpoint checks` token is neutral.
- Do not show a planner, agent, shot selection, game score, task-success label, or causal-outcome icon.
- Do not show symbolic state in the rollout loop or an arrow from a readout back to `$z$` or the next `$h$`.
- Do not show a result table, confidence interval, ranking, benchmark score, empirical regime frequency, or representative screenshot.
- The reserved result field must contain only `Matched-compute result [TODO: result]`. It must have no axes, numbers, bars, curves, rankings, arrows, checkmarks, colors, or favorable wording.

## Implementation Guidance

- Build editable vector artwork in TikZ or a vector editor. Group it as `cascade`, `controller`, `pair_grid`, `carrier_loop`, `readouts`, and `protocol`.
- Target a `12 x 5.2 cm` artboard at two-column width. Keep text at least 6.5--7 pt at final scale.
- Make vignettes geometric abstractions rather than copied assets. Keep them readable in grayscale.
- Make the pair grid a true equal-cell 3 by 3 table. Use an outline and pointer rather than a winner-like fill.
- Keep the teal carrier continuous through all selections. Route dashed arrows outward to labels, never back to the loop.
- Keep the result reservation white with a 0.4--0.5 pt gray outline, no icon, no arrow, and no tinted background.
- Before use, review the figure against the prohibited claims and inspect a 300-dpi final-size raster.

## Caption Draft

**Figure 1: Mechanism and prospective comparison protocol for BG-NS-JEPA in the NovPhy instantiation.** A single intervention can induce an action-sparse persistent-effect cascade whose prediction demands vary across fixed steps. The bounded model-selection controller selects a coupled requested horizon and description mode, while the continuous latent remains the rollout carrier and symbolic outputs serve only as readouts or reliability-gated constraints. Only after final-evaluation authorization may the predeclared common-compute protocol evaluate joint selection against `fixed-pair`, `temporal-only`, `description-only`, `independent-axes`, and the parameter-matched two-head controller. Current issue-9 and issue-10 artifacts are not a matched-compute suite. The figure reports no controller advantage or final-evaluation metric. Six final rollouts are sealed, and final authorization is pending. The reserved result field remains intentionally empty until authorized predeclared scoring is complete.

## Source Paths

- `manuscript/content_brief.md`: paper-status boundary and Method and Experiments specification.
- `manuscript/CONTEXT.md`: canonical controller, final-evaluation, and claim-status vocabulary.
- `manuscript/research_evidence.md`: bounded controller, baseline, supervision, endpoint, and final-evaluation evidence.
- `manuscript/reviewer_expectations_2026.md`: common-interface and completeness requirements.
- `docs/data_contracts/cohort_v2_exhaustive_pair_evaluation_v1.md`: 3 by 3 pair surface.
- `docs/data_contracts/cohort_v2_physical_violations_v1.md`: the two endpoint checks and `illegal_contact` exclusion.
- `data/runtime_evidence/issue-9/cohort-v2-policy-baseline-summary.json`: controller-free baseline boundary.
- `data/runtime_evidence/issue-10/cohort-v2-controller-summary.json`: bounded model-selection controller evidence.
- `data/runtime_evidence/issue-11/cohort-v2-controller-aggregation-summary.json`: aggregation boundary.

## Validation

Validation owner: parent. Do not compile or treat this brief as TeX prose. Review that the rendered figure has no favorable performance implication, no final metric, and no statement that the issue-9 and issue-10 compute are matched.
