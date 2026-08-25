# Reviewer Expectations for a Prospective BG-NS-JEPA Paper

**Access date for all external sources:** 2026-08-17.

## Purpose and Source Use

This record applies the cited ICLR and NeurIPS guidance to an ICLR, NeurIPS, or ICML review standard. The cited research provides context for evaluation design. It does not establish venue policy or any BG-NS-JEPA result.

## Review-Relevant Conclusions

### 1. Frame the paper around a falsifiable mechanism

The paper should lead with the granularity-mismatch problem in action-sparse persistent-effect environments. NovPhy is the test setting. The kinetic or Knudsen analogy is bounded intuition for state-dependent horizon and description selection. It is not a derivation.

The central empirical claim remains `[TODO: result]`. Only after final-evaluation authorization may one predeclared common-compute protocol evaluate the joint-pair controller against `fixed-pair`, `temporal-only`, `description-only`, `independent-axes`, and the parameter-matched two-head controller.

### 2. The central controller claim needs one final interface

A controller-advantage claim requires a common final-evaluation interface across every compared policy. It must define compute accounting, role-separated partitions, a common endpoint, and a result corresponding to the claim. Only after final-evaluation authorization may the protocol evaluate the joint-pair controller against `fixed-pair`, `temporal-only`, `description-only`, `independent-axes`, and the parameter-matched two-head controller. This is a prospective protocol, not a current matched-compute suite.

Relevant technical context includes [Physion](https://arxiv.org/abs/2106.08261), [V-JEPA](https://arxiv.org/abs/2404.08471), [DreamerV3](https://arxiv.org/abs/2301.04104), [ACT](https://arxiv.org/abs/1603.08983), and [Mixture-of-Depths](https://arxiv.org/abs/2404.02258). These papers do not prove a venue requirement or a BG-NS-JEPA result.

### 3. Current controller evidence is model-selection evidence only

The repository records a distilled joint-pair controller and a parameter-matched two-head controller. Their model-selection comparison uses 1,600 states and agent-observable inputs, excludes oracle engine state, gives equal scores, and records no observed joint-controller advantage. Its endpoint violation rate is `0.0002777777777777778`. This does not establish terminal-outcome accuracy, controller effectiveness, or the central comparison. [research_evidence.md](research_evidence.md)

Issue 9 supplies the current bounded controller-free `fixed-pair`, `temporal-only`, `description-only`, and `independent-axes` evidence, with six states per policy and exposure-role cell. Its compute is not matched to issue 10. Recomputing those policies under the authorized final protocol remains `[TODO: result]`. Issue 11 aggregates one round of six rollouts and 109 decisions using aligned ground-truth-expert carrier continuation. It neither rolls out the model closed loop nor measures terminal outcomes. [research_evidence.md](research_evidence.md)

### 4. Final evaluation and endpoint design remain incomplete

Six final-evaluation rollouts have been collected and sealed. Final-evaluation authorization is pending, so no final-evaluation metric has been derived or consumed. This is **Blocked**, not Unavailable or unrun. Cohort-release completion does not authorize final evaluation or manuscript claims.

Terminal-outcome accuracy through a common final-state readout remains the specified primary endpoint. The common readout and shared coordinate decoder are not completed. ADE, FDE, and event F1 are unavailable. FDE@H cannot serve as the primary endpoint without a shared coordinate decoder and a fixed absolute-horizon protocol.

### 5. Remaining evidence boundaries

Only endpoint measurements of excess penetration and unsupported stationary or floating bodies are available. Dense-path plausibility and `illegal_contact` are unavailable. OOD, template-held-out, and cross-domain claims are unavailable. The learned reliability gate, parser, and SPSG remain specified without evidence of benefit.

### 6. Completeness risk and publication options

The ICLR 2027 materials emphasize complete, well-supported work. A main-track submission whose central controller experiment remains `[TODO: result]` is not presently defensible. The options are:

1. Complete the authorized final controller study with the common evaluator and controls.
2. Rescope to a contribution supported by the bounded evidence.
3. Choose a venue suited to a design or partial-evidence contribution.

This is a manuscript-planning judgement based on the ICLR materials. It is not venue policy set by the cited technical papers.

## External Source Record

| Source | URL | Accessed |
|---|---|---|
| ICLR 2027 Reviewer Guidelines | https://iclr.cc/Conferences/2027/ReviewerGuidelines | 2026-08-17 |
| ICLR 2027 Call for Papers | https://iclr.cc/Conferences/2027/CallForPapers | 2026-08-17 |
| ICLR 2027 Author Guidelines | https://iclr.cc/Conferences/2027/AuthorGuidelines | 2026-08-17 |
| NeurIPS 2026 Reviewer Guidelines | https://neurips.cc/Conferences/2026/ReviewerGuidelines | 2026-08-17 |
| NeurIPS 2026 Call for Papers | https://neurips.cc/Conferences/2026/CallForPapers | 2026-08-17 |
| Physion | https://arxiv.org/abs/2106.08261 | 2026-08-17 |
| V-JEPA | https://arxiv.org/abs/2404.08471 | 2026-08-17 |
| DreamerV3 | https://arxiv.org/abs/2301.04104 | 2026-08-17 |
| ACT | https://arxiv.org/abs/1603.08983 | 2026-08-17 |
| Mixture-of-Depths | https://arxiv.org/abs/2404.02258 | 2026-08-17 |
