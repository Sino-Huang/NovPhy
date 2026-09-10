## Category and purpose
Prospective fresh non-final experiment and explicit advancement gate. Parent #1. Successor to the CLOSED #72 development stop, not a rerun that changes its conclusion.

Required inputs: #73 diagnostic; #74 genuine pure-continuous/hybrid readiness; #75 validated bounded-pilot disposition (including explicit not_needed if no intervention is justified). If these do not yield a viable candidate and feasible test budget, publish a readiness/precision stop rather than opening fresh data.

## Scientific questions — keep separate
1. Hybrid system versus an independently trained PURE-CONTINUOUS DYNAMICS baseline under the shared learned-perception contract.
2. Adaptive switching versus a fixed continuous-mode control using the SAME hybrid-trained checkpoint.
3. Strongest no-model / fixed-policy comparison, so beating a weak learned control cannot hide failure to beat a simple action prior.

Question 1 is the user's intended hybrid-versus-continuous test. Question 2 isolates switching and cannot substitute for question 1. A hybrid-trained checkpoint locked to continuous is not a pure-continuous training baseline.

The shared CNN/object-centric carrier already has semantic supervision. Do not claim an end-to-end symbol-free baseline. A uniquely joint/nonseparable mechanism additionally needs the prospectively required axis/factorization controls; without them limit the claim to the tested hybrid system. Preserve #15's central oracle-symbol result and all #70/#71/#72 outcomes.

## Amendment 2026-09-10: isolate dynamics and audit NovPhy coverage

This amendment records the user's request to reassess task/novelty coverage and separate dynamics quality from adaptive-control performance. It does NOT replace the negative #72/#74/#75 evidence, lower their gates, or open fresh data. #75 publication currently reports not_supported_by_this_pilot; its final validation/disposition must still be recorded. Phase A may perform a bounded feasibility diagnosis, but neither novelty expansion nor this amendment automatically makes the failed candidate eligible for fresh evaluation.

### Verified coverage and interpretation

Upstream documents five physical scenarios (single force, multiple forces, rolling, falling, sliding) and eight novelty categories, with paired normal and novel templates: https://github.com/phy-q/NovPhy/blob/main/README.md#2-novelties-in-novphy and https://github.com/phy-q/NovPhy/blob/main/README.md#42-tasks-generated-for-baseline-analysis .

Our current collection uses only novelty_level_0 and families type010101/type010102: see world_model/data/successor_cohort.py:GENERATOR_FAMILIES and scripts/run_issue_62_successor_cohort.py:_materialize_slot. These are normal single-force and multiple-forces scenarios, not comprehensive NovPhy coverage. Incidental object movement in them is not coverage of the dedicated rolling/falling/sliding tasks, nor a novel-physics test. Source inspection and candidate template inventory are recorded in docs/issue-76-novelty-coverage.md.

Consequences:
- Current results remain valid for their narrow sampled population. They neither establish nor rule out a hybrid advantage on other scenarios/novelties.
- Normal-level selection is a possible limitation, NOT an established cause of hybrid failure. Harder/novel levels may worsen both models.
- Adding more instances of the same two families does not repair missing task/novelty coverage. A broad NovPhy/open-world claim needs appropriate scenario and novelty strata or an explicitly narrower claim.
- Never choose novelty families because a hybrid happens to win there. Preserve all scheduled conditions, failures and unsupported capability findings.

### Phase A0.1: bounded existing-data dynamics diagnostic, before any new cohort

Separate these contrasts and estimands:
1. **Training effect:** independently trained continuous dynamics versus the hybrid-trained model executing continuous transitions, with the SAME fixed h in {1,5,15}. This isolates hybrid training effects under the declared capacity/exposure differences.
2. **Symbolic execution effect:** within the SAME hybrid checkpoint, fixed continuous versus fixed micro versus fixed macro at each SAME h. This distinguishes actual symbolic conditioning from a controller's choices.
3. **Adaptation effect:** hybrid adaptive versus its fixed-pair controls; continuous adaptive versus its fixed-horizon controls. Do not infer poor dynamics solely from a poor adaptive controller.

Use the three paired #74 seeds/checkpoints, unchanged CNN/carrier/action encoding/objective and the existing 12-action design. Reuse unchanged cached controls where possible. For a deadline-bounded diagnostic, prospectively freeze 24 existing calibration states (12 per current generator family, evenly spaced in sorted state-identity order; exact identities frozen before new diagnostic scores). No outcome/solvability-based selection. These are already-opened development data, NOT fresh confirmation; no checkpoint fitting or new encoder search.

For every compared fixed pair, measure recursive carrier/count error at common elapsed physical endpoints {15,30,60,150,225}. These times are divisible by all three fixed horizons. Predictions must feed the next prediction with no truth resets. Also report explicitly separate observed-context local-step errors to distinguish local accuracy from recursive drift; local diagnostics may use observed contexts, deployment counterfactual scoring may not. Do not infer error-growth shape from a single final endpoint or call h5's action regret a local prediction error.

Use the existing source-bound carrier target construction, predecessor-frame motion semantics and disclosed stable-terminal absorbing convention for all arms. Record actual launch/settlement timing and target availability. Do not silently compare different physical times, substitute an earlier observed frame as a requested-horizon target, or initialize a candidate from an observed post-launch frame. No interpolation of adaptive traces may be presented as observed model predictions at skipped checkpoints; fixed-pair curves avoid that ambiguity.

Report 225-step predicted-task ranking versus REAL settled replay cost separately from recursive carrier/count MSE, including regret/top-k, ties, absolute task progress, all seeds and paired uncertainty. Predeclare fieldwise presence, position, motion and time-feature errors with availability masks alongside aggregate carrier MSE; a large carrier error is not automatically a physical violation or proof that every field diverged. Do not choose scaling/fields after seeing which favors a model. A225-step prediction may precede settlement; this limitation is not fixed by this diagnostic. No inference from these curves alone to complete-shot or multi-shot gameplay competence.

Before execution freeze source/target manifests, exact paired membership, plot/metric definitions and work counters. Diagnostic allowance: no new captures or model fitting; at most one GPU-hour active work and 2GiB new derived artifacts, with bounded-memory batches, measured progress and an explicit stop if exceeded. This is an explanatory feasibility diagnostic, not an additional optimizer/repair search or an advancement gate. Reusing old outcomes must be labeled as such.

### Phase A0.2: metadata-first scenario/novelty compatibility inventory

Before collecting wider data, produce a matrix covering all five physical scenarios and upstream novelty levels0..8. Record available normal/novel counterpart template paths, generator constraints, object/slot inventory, physics modification, action legality, timing/event requirements, prior exposure roles and runtime observability. Inventory metadata, not held-out outcomes. Match counterparts by the documented template mapping, not similar filenames or unrelated layouts.

Check the existing deployed interface explicitly:
- The frozen CNN has18 authored slots, including only six platform slots. The normal falling template type010104 contains eight Platform entries; do NOT silently truncate it into the existing representation.
- Novel object appearances/classes, rolling/sliding shapes, new forces/agents and event history may be outside parser/carrier/training support. Classify perception failure, insufficient state observability and dynamics error separately.
- Right-side slingshot variants require appropriate legal action support; the current left-pull action design cannot be assumed valid unchanged.
- A world model cannot identify an unseen hidden force or event phase from identical insufficient initial observations merely because it has symbolic adapters. Audit what the permitted agent observation/history can reveal. Engine novelty IDs, true forces or oracle symbols may be evaluation metadata, never privileged runtime hints to one arm.
- Existing contact/support and two macro-state predicates are not a universal representation of magnetic forces, inverted gravity or triggered storms. Verify the actual required semantics/capabilities.

Label unsupported cells honestly. If new slots, perception/history inputs, legal action handling or training are required, produce a costed representation/training amendment applying shared changes symmetrically. Do NOT silently refit one arm, drop entities, resize frozen checkpoints or treat infrastructure failures as proof against either dynamics approach. If this cannot fit the deadline, stop with limited coverage/insufficient readiness rather than claiming full NovPhy evaluation.

### Phase A0.3: wider-test recommendation and separate protocols

Recommend broader coverage for any broad NovPhy claim, but stage it:
- Expand normal-mechanics coverage to dedicated rolling/falling/sliding alongside the current two scenarios, subject to representation readiness.
- Pair novel conditions with their corresponding normal templates. Prioritize metadata review of appearance-only novelty as a perception control and genuine physical changes (for example magnetic interactions or altered gravity) as dynamics stressors. These are prospective candidates, NOT prevalidated compatible choices or guaranteed favorable tests.
- Freeze a small, stratified subset and its claim scope after the metadata audit but BEFORE rendered performance smoke or outcomes. Metadata/capability-based exclusions must be documented prospectively. A small subset cannot support an all-eight-novelties claim; deferred cells remain explicitly untested.
- Keep **zero-shot novelty generalization** (no fitting on the target novelty) separate from **few-shot adaptation** (identical declared training data, update budgets, seed pairing and held-out novel lineages for both models). Broadening training and evaluation together is not a zero-shot test.
- Any rendered compatibility/power smoke uses a declared development role and is excluded from fresh evaluation. Bind normal/novel variants of the same base scenario specification to the same partition and paired analysis cluster; seed/frame/action repeats are not independent lineages.
- Report normal and novel performance separately, plus the paired change in the hybrid-minus-continuous effect under novelty with uncertainty. Control multiplicity/family weighting; do not report only the novelty on which the hybrid wins.

No expanded collection or retraining is launched by this ticket amendment. Phase A must turn any feasible expansion into a numeric, costed, source-bound protocol with exact templates, per-cell counts, seeds, endpoints, action/compute budgets and failure rules before execution. If new engineering materially exceeds this deadline-bounded task, request explicit direction and split that implementation work only then.

### Claim and advancement boundaries

Maintain separate decisions for dynamics-training effects, symbolic execution, adaptation and gameplay. A dynamics diagnostic may be informative even when the adaptive gameplay candidate is not ready. A favorable carrier-error curve, novelty subgroup or fixed-mode contrast cannot substitute for the existing primary adaptive-system/strong-baseline/compute gates or retroactively rescue #75. The overall advancement dispositions and #64/#65 authorization rules below are unchanged. #15's central experiment remains separate.

## Phase A: protocol and feasibility BEFORE fresh access
Freeze an immutable protocol with actual numerical values, not deferred placeholders:
- population/generator families and parameters, independent paired unit, all model and environment seeds, role/provenance/disjointness rules;
- #74/#75 parser/predictor/controller/checkpoint identities, trained capabilities and strongest eligible pure-continuous policy selected using permitted development data;
- no-model prior selected using development only, with selection optimism disclosed; never select or retune it on the new set;
- identical action generator/search, legal bounds and release/tap semantics across model arms; default is the existing 12-action grid unless #75 prospectively justifies another shared design;
- common physical prediction endpoint, stable-terminal handling, state re-observation after each executed shot, maximum shots, timeouts, technical retries and failure penalties;
- primary/secondary estimands, positive practical-effect margins, violation/validity margins where measurable, confidence level, multiplicity/decision hierarchy, nondegenerate useful-mode criteria;
- sample-size/power or precision justification using development information, accounting for paired binary success, rare events, model seeds and clustered repeated units;
- complete training and deployment compute accounting, per-shot/state/trajectory budgets and matched-compute comparison policy. Equal candidates alone is not matched compute; charge symbolic decoding, adapters, controller, perception and any infilling. Linear MACs must not be labeled full FLOPs;
- CPU/GPU memory, storage, collection/runtime estimates and an operator-agreed deadline/resource cap.

Default to the #74 paired training seeds. Do not treat candidate actions, frames, repeated shots or seed repeats on the same lineage as independent scenario replicates. A reduced single-checkpoint study must be explicitly scoped before access and cannot silently replace a seed-robust primary claim.

If necessary precision or nonzero event prevalence is infeasible within the agreed budget, publish readiness_or_precision_insufficient BEFORE fresh collection. Do not reduce margins or inspect successive candidate cohorts until a favorable test becomes affordable.

## Phase B: once-only fresh non-final evaluation
Generate the entire outcome-independent level/seed inventory only AFTER the above freeze. New lineages/seeds must be disjoint from all model/parser/controller fitting data, calibration, opened #69/#70/#71/#72 development evidence, #73/#75 diagnostics/pilots, and prior final sets. Names alone do not prove disjointness; bind generator inputs and scenario lineage identity.

Execute every declared system on every assigned paired unit, including unsolved levels and failures. No solvability screening, replacement levels, extra favorable seeds, checkpoint switching, reward changes or gate changes after access. No live engine oracle may choose deployment actions; simulator outcomes are evaluation targets only.

Measure separately:
- actual gameplay success and frozen penalized/censored shots-to-success;
- candidate ranking regret/top-k, best-observed headroom and available count errors;
- equal-physical-endpoint prediction/recursive error and genuinely available physical-validity diagnostics;
- executed modes/horizons, controller decisions, failure counts and full cost.

Do not fabricate physical violations for latent carriers, equate support movement with collapse, equate replay top-1 with winning, or confuse settled task outcomes with offset-225 labels. Missing capabilities remain unavailable and constrain claims.

## Required advancement decision
Numerically freeze all required contrasts before access. To authorize #64, require:
- the primary adaptive hybrid versus strongest eligible independently trained pure-continuous contrast to exceed the practical margin with the specified uncertainty rule;
- the adaptive-versus-SAME-hybrid-checkpoint fixed-mode contrast to pass, if advancing an adaptive-hybrid claim;
- the required strongest fixed/no-model comparisons to pass (retain #72's simple-prior concern, do not replace it with a weaker baseline);
- nonzero useful gameplay, meaningful nondegenerate description-mode/horizon use, absolute validity/failure and matched-compute constraints to pass.

A favorable training-recipe contrast, lower carrier MSE alone, or arbitrary mode variation cannot rescue a failed primary. Adaptive mean improvement without the required uncertainty is not support.

Output exactly one validated disposition:
- supported_for_issue_64_non_final_pilot, with the frozen complete system matrix and limited authorization;
- not_supported_by_this_experiment; or
- readiness_or_precision_insufficient.

Distinguish a pre-access feasibility stop from an executed fresh negative result. Both may complete this ticket honestly but neither unlocks #64/#65. No automatic corrective loop or subsequent alternative-optimizer sweep.

## Acceptance / reproducibility
- Fully frozen protocol, recorded access ordering, complete paired membership/results and individual claim decisions with effect sizes, uncertainty and compute.
- Machine-readable compact publication and complete source-linked artifacts; exact validation of inventories, metrics, repairs and decisions.
- Save reviewable frames/WebMs plus gallery/manifest under data/, including failed/incomplete attempts and timing disclosures. Bound real rendered smoke before a long capture run; no -nographics capture.
- Isolated-process parallel collection only with separate ports/work directories and bounded RAM. Progress/ETA through generation, capture, merge, scoring, publication and validation; deterministic resume without outcome-conditioned replacement.
- No new hashes or full-corpus integrity passes. No fresh/final outcome reads in dry runs. Long commands left to operator after no-write dry-run and bounded real smoke.
- Freeze and archive code/checkpoint assets before fresh evidence access, including the uncommitted #70–#72 prerequisites. Record any approved technical correction explicitly; no false claim that a local snapshot is a pushed archived release. Obtain commit/push authority where needed.

## Next / stop branch
Only a validated supported disposition unblocks #64's separate non-final power/design pilot and eventual sealed benchmark, followed by #65. A negative/infeasible result goes to evidence consolidation (#35–#38) with unexecuted benchmark work explicitly blocked/not run, not fabricated as completed.

## Linked solving order
#73 -> #74 -> #75 (pilot or explicit skip/stop) -> #76 -> [supported, exactly validated advancement gate] -> #64 -> #65 -> #35 -> #36 -> #37 -> #38.
Existing #71/#72 stay closed with their recorded limitations. #35 may prepare/consolidate existing evidence independently; a negative branch does not fabricate #64/#65 completion.
