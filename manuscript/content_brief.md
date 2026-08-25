# Manuscript Content Brief

Status note: The working title is *Granularity Is a Decision: Joint Horizon-Description Selection in Latent World Models*. This repository contains Method and Experiments specifications, not drafted TeX prose. Every paper section remains `NOT STARTED`. The central joint-versus-factorized advantage remains `[TODO: result]`. Cohort-v2 supports bounded model-selection artifacts, while six final-evaluation rollouts are sealed and final metric derivation and consumption are **Blocked** pending authorization. A completed cohort release is not final-evaluation or manuscript authorization. Use [research_evidence.md](research_evidence.md), [CONTEXT.md](CONTEXT.md), and [reviewer_expectations_2026.md](reviewer_expectations_2026.md) as the current claim boundary.

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
- **Experiments**: NOT STARTED. Report four controller-free issue-9 baselines, each evaluated on six states per policy and exposure-role cell, without calling their compute matched to issue 10. Report issue-10 model-selection scope only: 1,600 states, agent-observable inputs, oracle engine state excluded, equal controller scores, no observed joint-controller advantage, and endpoint violation rate `0.0002777777777777778`. Report issue-11 only as one aggregation round over six rollouts and 109 decisions using aligned ground-truth-expert carrier continuation, with zero deltas against the oracle-state baseline and no source-cohort mutation. State that it is not a model closed-loop rollout or terminal-outcome evidence.
- **Discussion**: NOT STARTED. State that only endpoint measurements of excess penetration and unsupported stationary or floating bodies are available. Do not claim dense-path plausibility or `illegal_contact` handling.
- **Conclusion**: NOT STARTED. State the specified final test and retain `[TODO: result]`. Do not claim final metrics or controller efficacy.

## Open Items

- **Central empirical claim**: At predeclared common compute, the learned joint-pair controller must improve a shared endpoint over the four controller-free baselines and the parameter-matched two-head controller. It remains `[TODO: result]`.
- **Final evaluation**: Six rollouts are collected and sealed. No final-evaluation metric is derived or consumed because authorization is pending. This item is **Blocked**, not Unavailable or unrun.
- **Primary endpoint**: Terminal-outcome accuracy through a common final-state readout is Specified and remains `[TODO: result]`. The shared coordinate decoder is not completed. ADE, FDE, and event F1 are Unavailable.
- **Physical plausibility**: Endpoint measurement is limited to excess penetration and unsupported stationary or floating bodies. Dense-path plausibility and `illegal_contact` are Unavailable.
- **Generalization**: Instance-held-out exposure is the central setting. Template-held-out, OOD, and cross-domain claims are Unavailable.
- **Specified components**: The learned reliability gate, learned parser, and SPSG have no recorded benefit. Do not write efficacy claims for them.
- **Submission gate**: Cohort-release completion does not authorize final scoring or manuscript claims. A main-track submission with the central result still `[TODO: result]` requires an honest scope change or the missing authorized evidence.
- **Compile gate**: The local TeX Live 2026 installation at `texlive/` built `iclr2026_conference.tex` successfully with `latexmk` 4.88. The generated seven-page PDF has no undefined citations, references, or LaTeX/natbib warnings.
- **Critic review**: [2026-08-25 critic 1](critics/2026-08-25-critic-1.md) found no CRITICAL issue, but two HIGH drafting risks remain: stabilize one prospective common-compute comparator set and map `short`/`medium`/`long` to requested horizons 1/5/15 fixed steps. Resolve the MEDIUM teaser endpoint and parameter-matching labels and the LOW null-rate context before using the figure brief for a central-efficacy draft.

## Next

- Obtain final-evaluation authorization before deriving or consuming a final metric.
- Define and complete the common final-state readout and shared coordinate decoder before adding terminal-outcome, ADE, FDE, or event-F1 results.
- Freeze the final matched-compute comparison and its claim boundary before drafting TeX prose.
- Keep every paper section `NOT STARTED` until TeX prose is drafted.
- Preserve the local TeX toolchain and rerun the mandatory build after subsequent TeX changes.
- Address the critic's HIGH issues before drafting paper sections; then resolve the teaser-label and null-rate clarity issues.
