# Manuscript Content Brief

Status note: The working title is *Granularity Is a Decision: Joint Horizon-Description Selection in Latent World Models*. This repository contains Method and Experiments specifications, not drafted TeX prose. Every paper section remains `NOT STARTED`. The central authorized comparison remains `[TODO: result]`. Cohort-v2 supports bounded model-selection artifacts, while six final-evaluation rollouts are sealed and final metric derivation and consumption are **Blocked** pending authorization. A completed cohort release is not final-evaluation or manuscript authorization. Use [research_evidence.md](research_evidence.md), [CONTEXT.md](CONTEXT.md), and [reviewer_expectations_2026.md](reviewer_expectations_2026.md) as the current claim boundary.

## Progress

| # | Section | Status | Notes |
|---|---|---|---|
| 1 | Introduction | NOT STARTED | No TeX prose drafted. State the falsifiable mechanism question and retain `[TODO: result]`. |
| 2 | Related Work | NOT STARTED | No TeX prose drafted. Use independently verified citations only. |
| 3 | Method | NOT STARTED | No TeX prose drafted. Specify the 3 by 3 pair scope, continuous carrier, supervision, and status boundaries. |
| 4 | Experiments | NOT STARTED | No TeX prose drafted. Report bounded model-selection facts and the blocked final-evaluation boundary. |
| 5 | Discussion | NOT STARTED | No TeX prose drafted. Separate equal model-selection scores from any efficacy claim. |
| 6 | Conclusion | NOT STARTED | No TeX prose drafted. State the specified final test and its missing authorized result. |

## Section Summary

- **Introduction**: NOT STARTED. Define the granularity-mismatch problem and distinguish published NovPhy from BG-NS-JEPA. The joint-controller advantage remains `[TODO: result]`.
- **Related Work**: NOT STARTED. Use [related_work_citation_map.md](related_work_citation_map.md) for citation planning. Do not turn related work into evidence for the controller.
- **Method**: NOT STARTED. Describe the cohort-v2 3 by 3 horizon-mode surface. Record micro and macro transition supervision. The issue-7 pair-measurement surface contains 378 macro-mode available records. [issue-7 pair-measurement summary](../data/runtime_evidence/issue-7/cohort-v2-pair-measurement-summary.json#L1) Define the continuous carrier, trajectory-optimal labels, the distilled joint-pair controller, and the parameter-matched two-head controller. Mark the reliability gate, parser, SPSG, common final-state readout, and shared coordinate decoder by their current status.
- **Experiments**: NOT STARTED. Report the controller-free issue-9 `fixed-pair`, `temporal-only`, `description-only`, and `independent-axes` policies as bounded evidence only, with six states per policy and exposure-role cell. Do not call their compute matched to issue 10. Recomputing all four policies and the parameter-matched two-head controller against the joint-pair controller under the authorized predeclared common-compute protocol remains `[TODO: result]`. Report issue-10 model-selection scope only: 1,600 states, agent-observable inputs, oracle engine state excluded, equal controller scores, no observed joint-controller advantage, and endpoint violation rate `0.0002777777777777778`. Report issue-11 only as one aggregation round over six rollouts and 109 decisions using aligned ground-truth-expert carrier continuation, with zero deltas against the oracle-state baseline and no source-cohort mutation. State that it is not a model closed-loop rollout or terminal-outcome evidence.
- **Discussion**: NOT STARTED. State that only endpoint measurements of excess penetration and unsupported stationary or floating bodies are available. Do not claim dense-path plausibility or `illegal_contact` handling.
- **Conclusion**: NOT STARTED. State the specified final test and retain `[TODO: result]`. Do not claim final metrics or controller efficacy.

## Open Items

- **Central empirical claim**: Only after final-evaluation authorization may the predeclared common-compute protocol evaluate the learned joint-pair controller against `fixed-pair`, `temporal-only`, `description-only`, `independent-axes`, and the parameter-matched two-head controller. It remains `[TODO: result]`.
- **Final evaluation**: Six rollouts are collected and sealed. No final-evaluation metric is derived or consumed because authorization is pending. This item is **Blocked**, not Unavailable or unrun.
- **Primary endpoint**: Terminal-outcome accuracy through a common final-state readout is Specified and remains `[TODO: result]`. The shared coordinate decoder is not completed. ADE, FDE, and event F1 are Unavailable.
- **Physical plausibility**: Endpoint measurement is limited to excess penetration and unsupported stationary or floating bodies. Dense-path plausibility and `illegal_contact` are Unavailable.
- **Generalization**: Instance-held-out exposure is the central setting. Template-held-out, OOD, and cross-domain claims are Unavailable.
- **Specified components**: The learned reliability gate, learned parser, and SPSG have no recorded benefit. Do not write efficacy claims for them.
- **Submission gate**: Cohort-release completion does not authorize final scoring or manuscript claims. A main-track submission with the central result still `[TODO: result]` requires an honest scope change or the missing authorized evidence.
- **Compile gate**: The local TeX Live 2026 installation at `texlive/` built `iclr2026_conference.tex` successfully with `latexmk` 4.88. The generated seven-page PDF has no undefined citations, references, or LaTeX/natbib warnings.
- **Critic review**: [2026-08-25 critic 1](critics/2026-08-25-critic-1.md) found no CRITICAL issue. Its two HIGH corrections now specify one authorized prospective comparator set and requested horizons 1, 5, and 15 fixed steps. The MEDIUM teaser endpoint and parameter-matching labels and the LOW null-rate context remain out of scope. Gate 3 also deferred two LOW clarity edits: use one `common-compute` result label and mark Panel A duration brackets as schematic.
- **Phase 3 critic**: [2026-08-25 critic 2](critics/2026-08-25-critic-2.md) confirms both HIGH risks are resolved with no regression. It retains a BORDERLINE verdict because the teaser still conflates the specified primary endpoint with ancillary diagnostics and hides parameter matching in the scannable two-head label; the rate-context and protocol-label issues remain LOW clarity work.

## Next

- Obtain final-evaluation authorization before deriving or consuming a final metric.
- Define and complete the common final-state readout and shared coordinate decoder before adding terminal-outcome, ADE, FDE, or event-F1 results.
- Obtain final-evaluation authorization, then run the predeclared common-compute comparison of the joint-pair controller against `fixed-pair`, `temporal-only`, `description-only`, `independent-axes`, and the parameter-matched two-head controller. Keep its result `[TODO: result]` until then.
- Keep every paper section `NOT STARTED` until TeX prose is drafted.
- Preserve the local TeX toolchain and rerun the mandatory build after subsequent TeX changes.
- Preserve the implemented HIGH corrections. The MEDIUM teaser-label and LOW null-rate clarity issues remain for a later lane.
- Resolve the two Gate 3 LOW teaser clarity edits before treating the figure brief as submission-ready.
- Recommended next session: resolve the two MEDIUM teaser issues before drafting paper sections; then decide whether to clean the remaining LOW clarity issues or draft the Method section.
