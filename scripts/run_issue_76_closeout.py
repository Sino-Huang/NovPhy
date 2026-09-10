"""Complete #76 reporting from preserved development evidence; never run an experiment."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import io
from pathlib import Path
import subprocess
import sys

import numpy as np

from scripts import run_issue_76_dynamics_diagnostic as diagnostic


ROOT = diagnostic.ROOT
OUTPUT = ROOT/"data/issue-76-closeout"
SOURCE = "scripts/run_issue_76_closeout.py"
SYSTEMS = (*diagnostic.SYSTEMS, *diagnostic.ADAPTIVE)
BASELINES = {"prior_ordinal09":"prior_ordinal09_regret", "uniform_expected":"uniform_expected_regret"}
SCOPE = "reporting completion after diagnostic inspection; descriptive reused-calibration, not fresh confirmation or an advancement test"


def log(message):
    print(f"[issue-76-closeout] {message}", flush=True)


def direction(interval):
    bounds = interval.get("descriptive_95_percent_interval")
    if bounds is None: return "unavailable"
    low, high = bounds
    if low > 0: return "descriptive_lower_error_or_regret"
    if high < 0: return "descriptive_higher_error_or_regret"
    return "interval_includes_zero_not_established"


def continuous_reference(previous):
    """Repeat #75's selection, never optimize on the #76 diagnostic subset."""
    policies = (*diagnostic.base.POLICIES[:4], "covered_continuous")
    summaries = previous["per_seed"]
    means = {p:float(np.mean([s[p]["mean_regret"] for s in summaries.values()])) for p in policies}
    eligible = [p for p in policies if not any(s[p]["prediction_failure_states"] for s in summaries.values())]
    selected = min(eligible, key=lambda p:(means[p], policies.index(p))) if eligible else None
    if selected is None or selected != previous["selected_continuous_policy"]:
        raise ValueError("#75 independent continuous selection does not reproduce")
    if any(means[p] != previous["means"][p] for p in policies):
        raise ValueError("#75 policy means do not reproduce")
    return selected


def paired_comparison(report, tested, reference):
    headroom = {r["state"]:r for r in report["headroom"]}
    ids = sorted(headroom)
    per_seed, differences = {}, []
    for seed in diagnostic.SEEDS:
        rows = report["per_state"][str(seed)]
        if set(rows[tested]) != set(ids) or (reference not in BASELINES and set(rows[reference]) != set(ids)):
            raise ValueError("paired state membership differs")
        values = []
        for state in ids:
            baseline = (headroom[state][BASELINES[reference]] if reference in BASELINES
                        else rows[reference][state]["task"]["regret"])
            values.append(baseline-rows[tested][state]["task"]["regret"])
        per_seed[str(seed)] = {"mean":float(np.mean(values)), "paired_state_differences":dict(zip(ids,values,strict=True))}
    for state in ids:
        differences.append(float(np.mean([per_seed[str(s)]["paired_state_differences"][state] for s in diagnostic.SEEDS])))
    interval = diagnostic.base.grid.paired_interval(differences)
    families = {}
    for family in sorted({r["family"] for r in headroom.values()}):
        values = [v for state,v in zip(ids,differences,strict=True) if headroom[state]["family"] == family]
        families[family] = {"paired_states":len(values), **diagnostic.base.grid.paired_interval(values)}
    return {"tested":tested,"reference":reference,"metric":"normalized_settled_count_cost_regret",
            "positive_is_improvement":True,"paired_states":len(ids),"regret":interval,
            "per_seed":per_seed,"per_family":families,
            "paired_state_mean_differences":dict(zip(ids,differences,strict=True)),
            "descriptive_decision":direction(interval),"scope":SCOPE}


def prior_outcomes(plan):
    rows = []
    for sample in plan["samples"]:
        state = sample["state"]; source = sample["sources"][8]
        path = Path(plan["source_release"])/"candidate-results"/f"cal-s{state['role_ordinal']+1:04d}-c09.json"
        record = diagnostic.read(path)
        if record["candidate_identity"] != source["identity"] or record["state_identity"] != state["identity"]:
            raise ValueError("ordinal09 source membership differs")
        rows.append({"state":state["identity"],"candidate_identity":source["identity"],"source":str(path),
                     "status":record["status"],"realized_count_cost":record["goal_count_cost"] if record["status"] == "accepted" else 1e9,
                     "outcomes":diagnostic.outcome_flags(record)})
    return rows


def build_report(plan, report, previous, prior):
    if not report["diagnostics_complete"] or report["disposition"] != "readiness_or_precision_insufficient":
        raise ValueError("requires the completed #76 pre-access diagnostic stop")
    if any(report[k] for k in ("fresh_evaluation_opened","final_evaluation_opened","issue_64_authorized")):
        raise ValueError("supplement is restricted to the existing pre-access stop")
    if (previous["plan_identity"] != plan["source75_plan_identity"]
            or previous["disposition"] != plan["source75_disposition"]
            or previous["checks"] != plan["source75_checks"]):
        raise ValueError("#75 disposition binding differs")
    reference = continuous_reference(previous)
    comparisons = [paired_comparison(report,system,baseline) for baseline in BASELINES for system in SYSTEMS]
    comparisons += [paired_comparison(report,system,reference) for system in ("hybrid_adaptive","covered_hybrid")]
    claim_reviews = [{**c,"regret_descriptive_decision":direction(c["regret"]),
                      "carrier_descriptive_decisions":{t:direction(v) for t,v in c["carrier_mse"].items()},
                      "advancement_authorized":False} for c in report["contrasts"]]
    baseline_summary = {name:{"mean_regret":float(np.mean([r[field] for r in report["headroom"]])),
                            "interpretation":"expected regret over all 12 actions, not a sampled random-policy trial"
                                if name == "uniform_expected" else "ordinal09 frozen before #76; development selection optimism retained"}
                        for name,field in BASELINES.items()}
    baseline_summary["prior_ordinal09"].update({"independent_states":len(prior),
        "mean_realized_count_cost":float(np.mean([r["realized_count_cost"] for r in prior])),
        "selected_outcome_counts":{k:sum(r["outcomes"][k] for r in prior) for k in prior[0]["outcomes"]}})
    return {"schema":"issue_76_reporting_closeout_v1","scope":SCOPE,"plan_identity":plan["identity"],
            "source75_plan_identity":previous["plan_identity"],"source75_disposition":previous["disposition"],
            "continuous_reference":reference,"continuous_selection_scope":"#75 complete 200-state development publication; not reselected on #76's 24 states",
            "continuous_selection_means":previous["means"],"independent_states":len(report["headroom"]),
            "seeds":list(diagnostic.SEEDS),"bootstrap":plan["definitions"]["bootstrap"],
            "uncertainty_scope":"95% descriptive paired-state bootstrap; average paired seed differences within each state before resampling; no multiplicity-adjusted confirmation",
            "baseline_summary":baseline_summary,"prior_selected_replays":prior,"comparisons":comparisons,
            "original_claim_reviews":claim_reviews,"original_per_seed":report["per_seed"],
            "headroom":report["headroom"],"timing":report["timing"],"budget":report["budget"],
            "source_training_cost":report["source_training_cost"],"target_parsing_seconds":report["target_parsing_seconds"],
            "coverage":report["novelty_inventory_summary"],"wider_recommendation":report["wider_recommendation"],
            "requirement_dispositions":report["requirement_ledger"],
            "disposition":report["disposition"],"disposition_scope":report["disposition_scope"],
            "preaccess_reasons":report["preaccess_reasons"],
            "scientific_claims":{"general_hybrid_superiority":"not_established",
                                 "uniquely_joint_mechanism":"not_established",
                                 "adaptive_gameplay_advantage":"not_established",
                                 "broad_novelty_generalization":"not_tested"},
            "fresh_evaluation_opened":False,"final_evaluation_opened":False,"issue_64_authorized":False,
            "new_predictions":0,"new_fitting_updates":0,"new_captures":0,"original_artifacts_rewritten":False,
            "handoff":{"destination":"#35 analysis -> #36 archival -> #37 reporting -> #38 completion accounting",
                       "issue64_issue65":"blocked/not run; no supported advancement receipt",
                       "conditional_expansion":"requires explicit engineering/role/protocol authority before execution"}}


def comparison_csv(result):
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(("tested","reference","paired_states","mean_improvement","lower95","upper95","descriptive_decision"))
    for c in result["comparisons"]:
        value = c["regret"]
        writer.writerow((c["tested"],c["reference"],c["paired_states"],value["mean"],
                         *value["descriptive_95_percent_interval"],c["descriptive_decision"]))
    return stream.getvalue().encode("utf-8")


def interval_text(value):
    if "mean" not in value: return "unavailable"
    low,high = value["descriptive_95_percent_interval"]
    return f"{value['mean']:.6f} [{low:.6f}, {high:.6f}]"


def findings(result):
    lines = ["# #76 completed diagnostic: findings and pre-access stop", "",
             "Disposition: `readiness_or_precision_insufficient`, BEFORE fresh access. This is not an executed fresh negative experiment and does not authorize #64/#65.", "",
             "## Evidence and scope", "",
             "The source-bound diagnostic completed 24 calibration states × three paired seeds × 12 fixed systems × 12 actions (10,368 records). Original and covered adaptive controls are retained from #74/#75. No new predictions, fitting, captures or fresh/final access were performed for this reporting supplement.", "",
             "This supplement completes omitted comparator reporting after inspection of the diagnostic. It does not retroactively preregister new confirmation, select a favorable seed/subgroup, or replace the original publication. The shared CNN is semantically supervised; neither an end-to-end symbol-free baseline nor a uniquely joint mechanism is established.", "",
             "## Strongest comparator and no-model results", "",
             f"Independent continuous reference: `{result['continuous_reference']}`, reproduced from #75's full 200-state development selection, not chosen on these 24 states. The ordinal09 prior was also frozen before #76; its development selection optimism remains. Uniform expected regret averages the 12 existing action outcomes, not a sampled random-policy gameplay experiment.", "",
             "Improvement = reference regret minus tested regret; positive is better. Intervals are descriptive 95% paired-state bootstrap intervals (10,000 draws, RNG seed7201). Seed differences are averaged within state before resampling. The 24 states, not 72 seed repeats or 288 actions, are independent units. Family summaries in JSON retain 12 states per family. No multiplicity-adjusted confirmatory claim follows.", "",
             "| Tested | Reference | Mean improvement [95% interval] | Descriptive decision |",
             "| --- | --- | --- | --- |"]
    for c in result["comparisons"]:
        lines.append(f"| {c['tested']} | {c['reference']} | {interval_text(c['regret'])} | {c['descriptive_decision']} |")
    lines += ["", "## Training, symbolic execution and adaptation decisions", "",
              "The following are separate component/endpoint decisions, not a single hybrid-win label. A lower carrier error does not imply lower task regret; interval exclusion of zero on reused calibration data is not fresh support. Adaptive intermediate errors remain unavailable rather than interpolated.", "",
              "| Kind | Tested vs reference | Regret improvement [95% interval] | Offset225 carrier-MSE improvement [95% interval] |",
              "| --- | --- | --- | --- |"]
    for c in result["original_claim_reviews"]:
        lines.append(f"| {c['kind']} | {c['tested']} vs {c['reference']} | {interval_text(c['regret'])} | {interval_text(c['carrier_mse']['225'])} |")
    lines += ["", "All five original recursive endpoints, local-step/field errors, each seed, target masks and per-comparison descriptive direction decisions remain in the JSON and the original [diagnostic gallery](../issue-76-review/index.html). No favorable endpoint or field is selected to define a new success gate.", "",
              "## Absolute task progress, failures and headroom", "",
              "These are outcomes of selected existing single-shot candidate replays, not newly executed closed-loop or multi-shot gameplay. Counts are shown per seed; do not pool repeats as independent trials. Top1 ranking is not level success.", "",
              "| Seed | System | Mean regret | Top1 | Mean settled count cost | Pig-removing / level-clear states | Prediction-failure / all-tied states |",
              "| --- | --- | ---: | ---: | ---: | --- | --- |"]
    for seed,systems in result["original_per_seed"].items():
        for name,values in systems.items():
            outcomes = values["selected_outcome_counts"]
            lines.append(f"| {seed} | {name} | {values['mean_regret']:.6f} | {values['top1_fraction']:.6f} | {values['mean_realized_count_cost']:.6f} | {outcomes['pig_removed']} / {outcomes['level_clear']} | {values['prediction_failure_states']} / {values['predicted_all_tied_states']} |")
    prior = result["baseline_summary"]["prior_ordinal09"]
    opportunities = {k:sum(r["opportunities"][k] for r in result["headroom"]) for k in result["headroom"][0]["opportunities"]}
    lines += ["",f"Frozen prior: mean regret {prior['mean_regret']:.6f}, mean settled count cost {prior['mean_realized_count_cost']:.6f}, selected outcome counts {prior['selected_outcome_counts']} across {prior['independent_states']} states (not repeated as three new trials). Uniform expected regret: {result['baseline_summary']['uniform_expected']['mean_regret']:.6f}.", "",
              f"Informative states: {sum(r['informative'] for r in result['headroom'])}/24. States with at least one candidate opportunity: {opportunities}.", "",
              "## Timing, perception and compute limitations", ""]
    timings = [c for row in result["timing"] for c in row["candidates"]]
    launches = Counter(v for c in timings for v in c["events"]["bird_launched"])
    terminals = Counter(c["terminal_reason"] for c in timings)
    lines += [f"Actual launch-offset counts: {dict(launches)}. Terminal reasons: {dict(terminals)}. {sum(c['last_offset'] < 225 for c in timings)}/{len(timings)} trajectories end before225; target availability and stable-terminal absorption are explicitly recorded in the original targets. Early endpoints can precede launch. Offset225 prediction is not necessarily settlement.", "",
              "Targets are learned-perception carriers, not oracle physical state. Large aggregate error is not a physical violation or proof that every field diverges. Local observed-context predictions are evaluation-only and do not reset deployment rollouts. Cached adaptive wall times are not a paired latency experiment; unchanged candidate counts are not matched total work.", "",
              "| System | Total transition linear MACs | Local probe MACs | Mean instrumented model seconds/state across seeds |",
              "| --- | ---: | ---: | ---: |"]
    for name in SYSTEMS:
        rows = [s[name] for s in result["original_per_seed"].values()]
        lines.append(f"| {name} | {sum(r['transition_linear_macs'] for r in rows)} | {sum(r['local_probe_linear_macs'] for r in rows)} | {np.mean([r['mean_instrumented_model_seconds_per_state'] for r in rows]):.6f} |")
    budget = result["budget"]
    lines += ["",f"Diagnostic active work {budget['active_seconds']:.2f}s; derived artifacts {budget['artifact_bytes']:,} bytes; peak CPU RSS {budget['peak_cpu_rss_mib']:.2f}MiB, CUDA allocation {budget['peak_cuda_allocated_mib']:.2f}MiB; budget stopped={budget['stopped']}. Target parsing {result['target_parsing_seconds']:.2f}s is retained separately. Historical #74 and #75 training/label/scoring costs, including the failed correction, are retained in JSON rather than treated as free. Validation and this reporting supplement are separate from the original diagnostic allowance. MACs exclude nonlinear work and are not full FLOPs or a matched end-to-end compute claim.", "",
              "## Coverage, conditional work and stop-branch handoff", "",
              "The 80-template/40-pair/45-cell audit is metadata, not performance evidence on all five scenarios or eight novelties. The costed rolling/falling/sliding and normal/novel appearance recommendation is not an executable live protocol. New slots, perception/history/action handling, shared refitting, exact per-cell seeds/roles/budgets, rendered compatibility and lineage exposure audits require separate approval and freeze. Zero-shot and few-shot experiments remain separate and unexecuted.", "",
              "| Original requirement | Disposition | Reason |", "| --- | --- | --- |"]
    for item in result["requirement_dispositions"]:
        lines.append(f"| {item['requirement']} | {item['status']} | {item.get('reason','')} |")
    lines += ["", "The fresh population/power/protocol, once-only collection, actual gameplay success/shots and fresh physical-validity tests are not executed. Source/checkpoints remain local, not an archived release. No gated item is converted into a pass by this stop.", "",
              "Handoff: #35 consolidates existing evidence with separate provenance/claim labels; #36 handles archival with authority; #37 reports the limits and results; #38 accounts for conditional scope. #64/#65 remain blocked/not run. The validated #75 negative decision is unchanged. Finishing this readiness-stop branch does not finish downstream tickets or authorize another corrective search.", ""]
    return "\n".join(lines)


def validated_inputs(args):
    log("checking original diagnostic publication, review and exact source amendment")
    subprocess.run([sys.executable,"-u","-m","scripts.run_issue_76_dynamics_diagnostic",
                    "--validate","--device",args.device,"--output",str(args.diagnostic),
                    "--review",str(args.review),"--issue74",str(args.issue74),"--issue75",str(args.issue75)],check=True,cwd=ROOT)
    plan = diagnostic.read(args.diagnostic/"plan.json")
    report = diagnostic.read(args.diagnostic/"result.json")
    previous = diagnostic.read(args.issue75/"result.json")
    result = build_report(plan,report,previous,prior_outcomes(plan))
    result["sources"] = {"diagnostic":str(args.diagnostic.resolve()),"review":str(args.review.resolve()),
                         "issue75":str(args.issue75.resolve()),"source_text":{SOURCE:(ROOT/SOURCE).read_text()},
                         "validation_repair":diagnostic.read(args.diagnostic/diagnostic.VALIDATION_REPAIR)}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    for mode in ("dry-run","publish","validate"): modes.add_argument("--"+mode,action="store_true")
    parser.add_argument("--diagnostic",type=Path,default=diagnostic.OUTPUT)
    parser.add_argument("--review",type=Path,default=diagnostic.REVIEW)
    parser.add_argument("--issue74",type=Path,default=diagnostic.base.OUTPUT)
    parser.add_argument("--issue75",type=Path,default=diagnostic.covered.OUTPUT)
    parser.add_argument("--output",type=Path,default=OUTPUT)
    parser.add_argument("--device",choices=("cpu","cuda"),default="cpu")
    args = parser.parse_args()
    try:
        if args.dry_run:
            for path in (args.diagnostic/"plan.json",args.diagnostic/"result.json",args.issue75/"result.json"):
                if not path.is_file(): raise ValueError(f"missing source: {path}")
            log("no-write dry-run passed; 32 no-model comparisons plus 2 predecessor-selected continuous comparisons; no image scoring, fitting or capture")
            return 0
        result = validated_inputs(args)
        csv_bytes = comparison_csv(result); markdown = findings(result)
        if args.publish:
            diagnostic.write(args.output/"summary.json",result)
            (args.output/"comparisons.csv").write_bytes(csv_bytes)
            (args.output/"findings.md").write_text(markdown)
            log(f"published {len(result['comparisons'])} supplemental paired comparisons and {len(result['original_claim_reviews'])} original claim reviews to {args.output}")
        elif (result != diagnostic.read(args.output/"summary.json")
              or csv_bytes != (args.output/"comparisons.csv").read_bytes()
              or markdown != (args.output/"findings.md").read_text()):
            raise ValueError("supplemental report/CSV/findings differ from preserved evidence")
        else:
            log("exact supplemental validation passed; original artifacts preserved; readiness stop unchanged; #64/#65 unauthorized")
        return 0
    except (ValueError,OSError,subprocess.CalledProcessError) as error:
        log(f"error: {error}"); return 1


if __name__ == "__main__":
    raise SystemExit(main())
