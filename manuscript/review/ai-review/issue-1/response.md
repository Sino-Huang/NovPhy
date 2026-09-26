 The reviewer is right about the missing state if the claim is optimality for autoregressive rollouts. But your implemented DP is mathematically valid for a fixed-context surrogate. The manuscript needs to
 distinguish those two objectives explicitly.

 There is also an important difference between the feedback and your current manuscript: Equation 6 currently uses z̃_(kⱼ), not ẑ_(kⱼ). That permits a defensible interpretation—but changing the symbol alone does not
 clearly establish it.

 1. What the code actually optimizes

 The N1 trainer imports and calls matched_dynamics.controller_labels. That labeler:

 1. Takes a precomputed array of context carriers, one per frame.
 2. Evaluates every feasible request from that frame’s stored context.
 3. Computes its error against the observed future carrier.
 4. Runs backward DP over frame indices.

 Sources: run_issue_77_n1_train.py:59–63,480–498 (scripts/run_issue_77_n1_train.py#L480), matched_dynamics.py:158–184 (world_model/training/matched_dynamics.py#L158).

 Writing the fixed context array as C=(c₀,…,c_(L-1)), the implemented objective is

 G   ⎛ r   ∣ C ⎞
  ctx⎝  1:J    ⎠
 =
 ∑
 j
 [
  Δ(rⱼ)
 ───────
    d
 ║F ⎛ c  ,a,rⱼ ⎞-z        ║²
 ║ θ⎝  kⱼ      ⎠  kⱼ+Δ(rⱼ)║
 +λ MAC(rⱼ)
 ].

 Crucially, C is fixed before optimizing the schedule. Choosing a different request does not change the context used by later edges.

 For this objective,

 V (k)=min⎡g (k,r)+V (k+Δ(r))⎤
  C     r ⎣ C       C        ⎦

 is an exact shortest-path recurrence. The DP itself is not broken.

 2. Why the reviewer’s objection nevertheless matters

 Deployment is genuinely autoregressive:

 ```python
z = model.carrier(z, action, pair)
 ```

 The next request is selected from that updated carrier. See matched_dynamics.py:132–154 (world_model/training/matched_dynamics.py#L132).

 If the objective evaluates each request from the carrier produced by that same schedule’s prefix, two prefixes reaching frame k can produce different carriers. Their continuation costs need not agree.

 The correct Bellman state would then include the carrier:

 V   (k,x)
  rec
 =
 min
  r
 [
  Δ(r)
 ──────
   d
 ║F (x,a,r)-z      ║²
 ║ θ         k+Δ(r)║
 +λ MAC(r)
 +
 V   ⎛k+Δ(r),F (x,a,r)⎞
  rec⎝        θ       ⎠
 ].

 Your labeler does not compute this quantity.

 Calling the error “one-step prediction error” does not resolve this issue: even a one-step error depends on the schedule prefix when its input carrier is recursively generated.

 3. Round 2 does not restore autoregressive optimality

 The implementation is more specific than the manuscript’s description:

 ┌───────┬──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
 │ Round │ Context array used by DP                                                                                                                         │
 ├───────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
 │ 1     │ Observed carriers at every frame.                                                                                                                │
 ├───────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
 │ 2     │ Start with observed carriers; overwrite only frames reached by the round-1 policy with its predicted carriers. Unvisited frames remain observed. │
 └───────┴──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

 This comes directly from:

 ```python
contexts = z.clone()
...
current = model.carrier(current, a, pair)
t += pair.delta
contexts[:, t] = current
 ```

 The code then computes labels and retains training examples at all nonterminal frames, not just policy-visited frames. See run_issue_77_n1_train.py:480–499 (scripts/run_issue_77_n1_train.py#L480).

 For a counterfactual request r at frame k:

 - Its immediate prediction is evaluated from the stored cₖ.
 - Its continuation uses the stored context at k+Δ(r).
 - That continuation context is not generally F_θ(cₖ,a,r).

 Thus round 2 is a valid fixed-context surrogate with policy-generated contexts, not exact counterfactual rollout optimization. “DAgger-style” describes the data aggregation; it does not supply an optimality
 guarantee for the recursive objective.

 4. Is the current manuscript wrong?

 Under-specified and easy to overread, rather than an inherently invalid DP derivation.

 In the current method section (/home/sukaih/p/Project/NovPhy-manuscript/manuscript/iclr2026/iclr2026_conference_method.tex#L129):

 - Lines 129–136 define an autoregressive rollout and its carrier-dependent state.
 - Equation 6 then introduces z̃ as “the context carrier.”
 - Lines 141 and 297 say DP solves that objective.
 - Line 304 describes the two rounds, but omits the observed-context fallback at unvisited frames.

 If z̃ explicitly means a context table fixed independently of the candidate schedule, the exactness claim is correct for that table.

 If it means the candidate schedule’s own predicted carrier, the reviewer is correct and the claim is false.

 I would not rebut this with “our DP already optimizes the autoregressive objective.” The implementation does not support that defense.

 5. Is exact enumeration feasible?

 Not exhaustively for the full 60-frame, nine-request setting.

 The code uses three description levels and horizons {1,5,15}, with 60-frame training windows. Sources: cnn_hybrid.py:18 (world_model/training/cnn_hybrid.py#L18), run_issue_77_n1_train.py:14–18
 (scripts/run_issue_77_n1_train.py#L14).

 The number of request sequences satisfies

 N(L)=3N(L-1)+3N(L-5)+3N(L-15),

 with N(0)=1 and N(L)=0 for negative L.

 Just the sequences using horizon 1 throughout give

 N(60)≥ 3⁶⁰≈ 4.2×10²⁸.

 So “nine requests” does not imply a small sequence space. Exact enumeration could be a useful diagnostic on drastically shortened windows, but not a practical exhaustive replacement at the stated window length
 without additional exploitable structure.

 6. Recommended correction

 Keep the implementation, but make its actual optimization target explicit:

 1. Name Equation 6 the fixed-context scheduling surrogate, preferably G_(ctx)(r_(1:J)∣ C).
 2. State that C is held fixed while optimizing the schedule.
 3. Define both rounds’ context tables, including observed carriers at unvisited frames.
 4. Explain that counterfactual requests do not propagate their predicted carriers into the DP continuation.
 5. Limit exactness to the surrogate; do not imply that either the DP or the distilled policy is globally optimal for autoregressive rollout cost.

 Suggested wording:

 ▏ With the context array held fixed, backward dynamic programming exactly minimizes the resulting additive scheduling surrogate. In the second round, contexts at frames visited by the first-round policy are replaced
 ▏ by its predicted carriers, while unvisited frames retain observed carriers. Counterfactual requests are evaluated from these fixed contexts; their predicted carriers are not propagated into subsequent DP states.
 ▏ Consequently, the labels do not constitute an exact solution to the autoregressive schedule-optimization problem.

 Bottom line: your method can be defended as DP-based surrogate supervision with rollout-context aggregation. The reviewer’s requested distinction is technically justified. This concern alone does not invalidate the
 measured results, but it does rule out interpreting them as evidence that the policy was distilled from globally optimal autoregressive schedules.

 No manuscript or source files were changed. The verdict above rests on source inspection and mathematical reasoning, not a numerical comparison.


 You can defend the method, but not the stronger claim that the DP finds an optimal autoregressive schedule. I would classify this as a substantive but potentially repairable weakness, not automatically a fatal flaw.

 ### How serious is it?

 It depends on what the paper needs this claim to establish:

 ┌───────────────────────────────────────────────────────────────────────────────────────────────┬─────────────────────────────────────────────────────────────────────────────────┐
 │ Central claim                                                                                 │ Impact                                                                          │
 ├───────────────────────────────────────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────┤
 │ “We exactly optimize autoregressive prediction error versus compute.”                         │ Hard weakness. The implemented algorithm does not establish this.               │
 ├───────────────────────────────────────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────┤
 │ “We learn request policies from DP labels that optimize an explicit fixed-context surrogate.” │ Defensible. The recurrence and implementation support this.                     │
 ├───────────────────────────────────────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────┤
 │ “The learned policy provides useful empirical error–compute trade-offs.”                      │ Not settled by this objection. That must stand on the experimental comparisons. │
 └───────────────────────────────────────────────────────────────────────────────────────────────┴─────────────────────────────────────────────────────────────────────────────────┘

 The distinction matters because this is not merely a missing symbol. Clarifying the surrogate preserves the algorithm, but narrows what its optimality means.

 The strongest rebuttal strategy

 Concede the precise limitation, defend the valid optimization, and delimit the empirical claims.

 1. Agree with their mathematical point.
    A carrier-dependent recursive objective requires a carrier-dependent Bellman state.
 2. Explain what the implementation actually does.
    It fixes a context table before labeling. Consequently, its recurrence is an exact shortest-path computation for that table—not an erroneous approximation to a Bellman recurrence that secretly propagates
    carriers.
 3. Acknowledge that the manuscript failed to make the distinction sufficiently explicit.
    The current z̃ notation helps, but “you misread our notation” would be a weak response given the surrounding autoregressive formulation.
 4. Preserve only the supported contribution.
    The contribution can remain joint request selection, shared-carrier prediction, and surrogate-supervised policy learning. Do not rely on global autoregressive optimality.
 5. Address enumeration concretely.
    The full setting has at least 3⁶⁰≈4.2×10²⁸ schedules. That answers feasibility, but computational difficulty does not itself justify the surrogate’s quality.

 Suggested rebuttal

 ▏ We agree that exact optimization of a schedule’s own autoregressive rollout cost would require a Bellman state that includes the predicted carrier. Our implemented labeler instead optimizes a fixed-context
 ▏ scheduling surrogate: the context array is constructed before dynamic programming and held fixed across candidate schedules. The frame-indexed recurrence is exact for this surrogate, but does not provide an
 ▏ optimality guarantee for the autoregressive rollout objective. Our manuscript did not distinguish these objectives sufficiently clearly.
 ▏
 ▏ We will explicitly condition the objective on the context array and clarify that, in round 2, policy-visited frames use predicted carriers while unvisited frames retain observed carriers. Counterfactual requests
 ▏ do not propagate their outputs into the DP continuation. We will correspondingly restrict the optimality claim to the surrogate. Exhaustive enumeration over the full 60-frame setting is impractical: the horizon-1
 ▏ schedules alone contain 3⁶⁰ possible mode sequences.

 Then cite existing results that directly support the learned policy’s usefulness, if those comparisons are strong. Do not substitute unrelated world-model accuracy results for evidence about the policy.

 What remains vulnerable after that response?

 The reviewer can reasonably ask:

 ▏ “Why should this surrogate produce good autoregressive request decisions?”

 The code alone cannot answer that. Neither exact surrogate optimization nor DAgger-style aggregation guarantees improvement on recursive cost. A short-window exact comparison or a relevant aggregation ablation would
 strengthen the answer, but without those, you should leave the approximation quality as an acknowledged limitation—not imply it has been validated.

 My recommendation: defend the algorithm and its empirically demonstrated utility; concede the overbroad interpretation of optimality. If exact autoregressive optimization is the paper’s main novelty, this is a major
 problem. If it is an overstated justification for an otherwise well-supported method, a precise correction can preserve the contribution.


Yes—I read the relevant formulation, request-policy section, and introduction, and checked them against the implementation. I did not read the entire paper end to end. You are right that the current manuscript
 supports the surrogate interpretation more strongly than my previous response conveyed.

 My earlier framing of this as a potentially “hard weakness” was too broad for your current text.

 What your paper actually says

 There are three important pieces:

 1. The introduction describes surrogate supervision, not an optimal deployed policy:
    “a policy distilled from dynamic-programming labels that trade prediction error against compute”
    That is supported by the implementation.
 2. Equation 6 deliberately uses z̃_(kⱼ), whereas the autoregressive rollout uses ẑₖ. The accompanying text says:
    “The error term is the duration-weighted error of one prediction from its context.”
 3. The request-policy section specifies where those contexts come from:
    “In round 1 the contexts $\tilde z_k$ are the observed carriers. Round 2 is DAgger-style: it takes as contexts the carriers that the round-1 policy visits in its own rollout…”
    Sources: formulation, lines 129–141 (/home/sukaih/p/Project/NovPhy-manuscript/manuscript/iclr2026/iclr2026_conference_method.tex#L129), request policy, lines 297–304
    (/home/sukaih/p/Project/NovPhy-manuscript/manuscript/iclr2026/iclr2026_conference_method.tex#L297).

 Read together, these support your interpretation: Equation 6 concerns externally supplied contexts, not necessarily the candidate schedule’s own recursively generated carriers. Under that interpretation, Equation 15
 does solve Equation 6.

 What is still insufficiently explicit?

 The manuscript does not directly state the decisive condition:

 ▏ The context array is constructed before dynamic programming and held fixed across all candidate schedules.

 The reviewer instead identifies z̃ₖ with the candidate schedule’s ẑₖ. Their mathematical objection follows from that identification—but that is not what your implementation does, and the current notation
 distinguishes the two.

 The surrounding transition from autoregressive rollout to “a schedule … costs” makes their reading understandable. Nevertheless, that is an ambiguity to clarify, not evidence that your implemented DP is
 mathematically incorrect.

 Revised rebuttal position

 Defend that the recurrence correctly solves the stated fixed-context objective; acknowledge that the fixed-context condition deserves a more explicit statement. You need not concede that Equation 15 fails to solve
 Equation 6 under your intended definition.

 A better response would be:

 ▏ The concern would apply if the contexts in Equation 6 were generated recursively by each candidate schedule. However, Equation 6 uses z̃ₖ, rather than the autoregressive carrier ẑₖ: these contexts are constructed
 ▏ before labeling, as described in the request-policy section, and held fixed during dynamic programming. Thus gₖ(r) depends only on the frame and request, and Equation 15 exactly minimizes this fixed-context
 ▏ objective. We will make this distinction explicit and clarify the treatment of unvisited frames in round 2. We do not claim exact optimization of each candidate schedule’s own autoregressive rollout cost.

 So yes: your intended defense is supported by the current paper and code. The remaining issue is making “fixed across candidate schedules” unambiguous—not replacing an invalid algorithm.


Yes—but precisely: the reviewer appears to have interpreted z̃ₖ as the carrier generated by each candidate schedule, ẑₖ. That identification is not correct for your intended objective or implementation.

 The mistake is not merely confusing two symbols. It is assuming that the contexts in Equation 6 depend on the schedule being optimized. Your labeler constructs those contexts beforehand and holds them fixed.
 Therefore:

 - The reviewer’s mathematical argument is correct for an autoregressive objective.
 - Their conclusion that Equation 15 cannot solve Equation 6 does not follow under your fixed-context definition.

 However, the current manuscript leaves that fixed-context condition implicit enough that I would not write “the reviewer misread the notation.” That sounds dismissive and does not establish the mathematical
 distinction.

 ### Recommended rebuttal

 ▏ We believe this concern arises from identifying the context carrier z̃ₖ in Equation 6 with the autoregressive carrier ẑₖ in Equation 5. These quantities have different roles. For label generation, the context array
 ▏ is constructed before schedule optimization and held fixed across candidate schedules: round 1 uses observed carriers, while round 2 substitutes the first-round policy’s predicted carriers at visited frames,
 ▏ retaining observed carriers elsewhere. Consequently, gₖ(r) depends only on k and r, and Equation 15 is an exact shortest-path recurrence for Equation 6. The carrier-dependent Bellman state proposed by the reviewer
 ▏ would be necessary for optimizing each candidate schedule’s own autoregressive rollout cost, which is not the objective optimized by our labeler. We will explicitly state the fixed-context condition to eliminate
 ▏ this ambiguity.

 That defends your actual claim without conceding a nonexistent DP error—and acknowledges the genuine opportunity to clarify the presentation.