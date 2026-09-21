"""Issue-81 ADD-EXP: pooled boundary-curve synthesis and external-comparison tables.

Pre-declared analysis pass over published, exact-validated upstream artifacts.
No re-scoring, no re-fitting, no new captures: every interval and every mean in
the published tables is either copied verbatim from a source artifact or is a
deterministic seed-mean of values published inside ONE source artifact. All
bootstrap resampling happened upstream; this runner is CPU-only and NEVER takes
the GPU lock (/tmp/novphy-addexp-gpu.lock) because no wall-time-measured GPU
phase exists here.

Inputs (READ-ONLY):
- .local-artifacts/issue-77-n1-diagnostic-v1  (NovPhy normal-mechanics family)
- .local-artifacts/issue-77-n2-eval-v1        (appearance-novelty family; the
  zero-shot and few-shot conditions are separate cells and never pooled)
- .local-artifacts/issue-74-matched-dynamics-v1/readiness.json contributing
  ONLY selected_continuous_policy, per_seed.policies.*.mean_regret,
  per_seed.policies.*.mean_perception_planning_seconds, and the compute ledger
- .local-artifacts/issue-78-external-temporal-baselines-v1 (external temporal
  adaptation baselines family)
- .local-artifacts/issue-79-clevrer-boundary-v1 (CLEVRER out-of-family
  replication; recursive state MSE estimand, no regret)
- .local-artifacts/issue-80-reactive-diagnostic-v1 (reactive-control
  diagnostic; TERMINAL typed disposition -> carried as an explicitly labelled
  UNEXECUTED cell, never dropped)

Pooling rule (frozen in plan.json before any synthesis number is computed):
paired contrasts within each cell first (intervals inherited from the source
artifacts' paired bootstraps over paired units), cells reported side by side;
NO cross-family meta-analytic pooling - the paper claim is the shape and its
boundary conditions, not an average effect. The only aggregation performed
here is a seed-mean over the three published per-seed values of one
(family, condition, side, arm) cell.

Claim registry: every candidate paper claim carries its disposition token
(supported / not_supported_by_this_experiment / readiness_or_precision_insufficient)
and its contributing artifact rows; the mandated stop branch records claims
scoped to the UNEXECUTED reactive cell as `untested` instead of omitting them.

Modes (mutually exclusive):
--dry-run    no-write inventory of inputs and frozen-cell counts
--prepare    freeze plan.json (inputs pinned by sha256, table/figure
             definitions, pooling rule, claim registry, decision rule)
--run        compute the synthesis report from inputs + frozen plan; write
             compute.json (deterministic resume point)
--publish    transform compute.json into summary.json, comparisons.csv,
             findings.md, claim-registry.csv, figures/advantage_vs_horizon.csv
--validate   recompute every table and figure from the input artifacts and
             byte-compare all published files

Exact validation command: python -u -m scripts.run_boundary_synthesis --validate
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / ".local-artifacts/issue-81-boundary-synthesis-v1"
SCHEMA = "issue_81_boundary_synthesis_plan_v1"
REPORT_SCHEMA = "issue_81_boundary_synthesis_report_v1"
IDENTITY = "issue-81-boundary-synthesis-v1"
REPORT_IDENTITY = "issue-81-boundary-synthesis-report-v1"

GPU_LOCK_PATH = "/tmp/novphy-addexp-gpu.lock"
DERIVED_BYTES_BUDGET = 1024 ** 3

REGRET_ESTIMAND = ("normalized ranking regret against the t=600 end-of-window replay cost "
                   "(right-censored; NOT a settled cost)")
CLEVRER_ESTIMAND = ("recursive state mean squared error at frozen CLEVRER annotation endpoints; "
                    "no planning/ranking-regret estimand and no candidate inventory on CLEVRER")
INTERVAL_LABEL = "descriptive_95_percent_interval"
DESCRIPTIVE_SCOPE = ("descriptive intervals inherited from the source artifacts' paired bootstraps "
                     "over paired units; no inferential superiority decision")

SELECTION_DISCLOSURE = ("continuous_h5 is the #74 selected comparator "
                        "(selected_continuous_policy=continuous_h5, selection_optimism=true, "
                        "selected on development evidence); disclosed wherever it is referenced")

WALL_DEFINITIONS = {
    "issue-77-instrumented": "issue-77 instrumented model wall seconds per state (transition wall; "
                             "initial perception wall reported separately)",
    "issue-78-locked": "issue-78 per-state decision wall measured under that ticket's exclusive GPU lock",
    "issue-78-reference": "wall from frozen issue-77 records measured 2026-09-20 (no GPU-lock regime "
                          "in force then; disclosed by issue-78)",
    "issue-79-rollout": "issue-79 mean rollout wall seconds (annotation-derived contexts)",
    "issue-74-perception-planning": "issue-74 mean perception+planning wall seconds per state",
}

FAMILIES = {
    "novphy_normal_mechanics": {
        "source_artifact": "issue-77-n1-diagnostic-v1",
        "condition": "adapted",
        "states": 4,
        "estimand": REGRET_ESTIMAND,
        "condition_semantics": "frozen issue-77 N1 dynamics checkpoints on held-out normal-mechanics states",
    },
    "appearance_novelty": {
        "source_artifact": "issue-77-n2-eval-v1",
        "conditions": ("zero-shot", "few-shot"),
        "states": 17,
        "estimand": REGRET_ESTIMAND,
        "condition_semantics": {"zero-shot": "frozen issue-77 N1 dynamics checkpoints; no novelty fitting",
                                "few-shot": "N1 checkpoints adapted on N2 novel predictor lineages"},
        "condition_separation": "zero-shot and few-shot are separate cells; never pooled",
    },
    "external_temporal_baselines": {
        "source_artifact": "issue-78-external-temporal-baselines-v1",
        "condition": "adapted (in-ticket retrained arms with in-ticket continuous reference)",
        "states": 21,
        "estimand": REGRET_ESTIMAND,
        "condition_semantics": "TAWM/VLWM ports retrained in-ticket; contrasts are within-ticket pairs",
    },
    "clevrer_replication": {
        "source_artifact": "issue-79-clevrer-boundary-v1",
        "condition": "out-of-family retrained",
        "states": 12,
        "estimand": CLEVRER_ESTIMAND,
        "condition_semantics": "arms retrained from scratch on CLEVRER annotation-derived carriers",
    },
    "reactive_diagnostic": {
        "source_artifact": "issue-80-reactive-diagnostic-v1",
        "condition": "UNEXECUTED",
        "states": 0,
        "estimand": REGRET_ESTIMAND,
        "status": "UNEXECUTED",
        "condition_semantics": "mandatory nonzero-prevalence pilot gate failed before the full run",
    },
}

# issue-78 contrasts over state-count subsets; the 4/17 split is the frozen
# state counts of the two upstream corpora, cross-checked against that
# artifact's own work_reported_comparison `corpus` rows.
CORPUS_BY_STATE_COUNT = {4: "issue-77-n1-diagnostic-v1 subset",
                         17: "issue-77-n2-eval-v1 subset",
                         21: "all frozen states"}

T1_COLUMNS = ("family", "condition", "side", "kind", "tested", "reference", "horizon", "endpoint",
              "scope_states", "scope", "positive_is_improvement", "positive_favors", "metric",
              "mean", "interval_low", "interval_high", "excludes_zero", "interval_label",
              "estimand", "source_artifact", "status")
T2_COLUMNS = ("family", "condition", "side", "arm", "states", "estimand", "quality_metric",
              "mean_regret", "wall_seconds_per_state", "wall_definition", "transition_linear_macs",
              "linear_macs_definition", "transition_calls", "parameters", "controller_parameters",
              "candidate_count", "candidate_count_definition", "source_artifact", "status")


def log(message):
    print(f"[issue-81-synthesis] {message}", flush=True)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text)
    temporary.replace(path)


def digest(path):
    data = Path(path).read_bytes()
    return {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def seed_mean(values):
    """The ONLY aggregation this runner performs: mean over the published
    per-seed values of one cell. Never called across families or conditions."""
    if len(values) != 3:
        raise ValueError(f"seed_mean expects the three frozen seeds, got {len(values)}")
    if any(value is None for value in values):
        return None
    return sum(float(v) for v in values) / 3.0


def excludes_zero(low, high):
    return bool(low > 0 or high < 0)


def horizon_of(name):
    suffix = name.rsplit("_h", 1)
    if len(suffix) == 2 and suffix[1].isdigit():
        return int(suffix[1])
    return None


def direction(mean_value, positive_is_improvement=True):
    if not positive_is_improvement:
        raise ValueError("this synthesis assumes the source convention positive_is_improvement=true")
    if mean_value > 0:
        return "tested_favoring"
    if mean_value < 0:
        return "reference_favoring"
    return "tied"


def sign(value):
    return (value > 0) - (value < 0)


# ---------------------------------------------------------------- inputs

INPUT_FILES = {
    "issue-77-n1-diagnostic-v1": ("summary.json", "NovPhy normal-mechanics family report"),
    "issue-77-n2-eval-v1": ("summary.json", "appearance-novelty family report (zero-shot + few-shot cells)"),
    "issue-74-matched-dynamics-v1": ("readiness.json", "compute-frontier rows; allowed fields only"),
    "issue-74-matched-dynamics-v1/plan.json": ("plan.json", "issue-74 cohort-size provenance; only "
                                               "development.states is read"),
    "issue-78-external-temporal-baselines-v1": ("summary.json", "external temporal-adaptation baselines report"),
    "issue-79-clevrer-boundary-v1": ("summary.json", "CLEVRER out-of-family replication report"),
    "issue-80-reactive-diagnostic-v1": ("summary.json", "reactive-control diagnostic terminal report"),
}



def input_path(key):
    name, _ = INPUT_FILES[key]
    directory = key.rsplit("/", 1)[0] if "/" in key else key
    return ROOT / ".local-artifacts" / directory / name


def load_inputs():
    """Read the six read-only inputs. issue-74 is reduced to its allowed fields here."""
    sources = {}
    for artifact, (name, role) in INPUT_FILES.items():
        path = input_path(artifact)
        if not path.exists():
            raise ValueError(f"missing input artifact file: {path}")
        sources[artifact] = {"path": path, "role": role, "document": read(path)}
    return sources


def issue74_allowed_fields(document, plan_document):
    """The #81 ticket allows ONLY these fields from the issue-74 readiness.json,
    plus the cohort-size scalar from that artifact's own frozen plan.json."""
    policies = {}
    for seed, entry in sorted(document["per_seed"].items()):
        for policy_name, policy in sorted(entry["policies"].items()):
            policies.setdefault(policy_name, {})[seed] = {
                "mean_regret": policy["mean_regret"],
                "mean_perception_planning_seconds": policy["mean_perception_planning_seconds"],
            }
    return {"selected_continuous_policy": document["selected_continuous_policy"],
            "compute": document["compute"],
            "policies": policies,
            "cohort_states": plan_document["development"]["states"]}


# ---------------------------------------------------------------- T1: advantage vs horizon

def t1_row(*, family, condition, side, kind, tested, reference, scope_states, scope, metric,
           metric_block, endpoint, estimand, source_artifact, positive_favors=None,
           positive_is_improvement=True, status="executed"):
    return {"family": family, "condition": condition, "side": side, "kind": kind,
            "tested": tested, "reference": reference,
            "horizon": horizon_of(tested), "endpoint": endpoint,
            "scope_states": scope_states, "scope": scope,
            "positive_is_improvement": positive_is_improvement,
            "positive_favors": positive_favors or tested,
            "metric": metric,
            "mean": metric_block["mean"],
            "interval_low": metric_block["descriptive_95_percent_interval"][0],
            "interval_high": metric_block["descriptive_95_percent_interval"][1],
            "excludes_zero": excludes_zero(metric_block["descriptive_95_percent_interval"][0],
                                           metric_block["descriptive_95_percent_interval"][1]),
            "interval_label": INTERVAL_LABEL,
            "estimand": estimand, "source_artifact": source_artifact, "status": status}


def t1_regret_family(family, condition, side, contrasts, artifact, scope_states):
    rows = []
    for contrast in contrasts:
        rows.append(t1_row(family=family, condition=condition, side=side,
                           kind=contrast["kind"], tested=contrast["tested"],
                           reference=contrast["reference"],
                           scope_states=scope_states,
                           scope=contrast["scope"], metric="normalized_ranking_regret",
                           metric_block=contrast["regret"], endpoint=None,
                           estimand=REGRET_ESTIMAND, source_artifact=artifact))
    return rows


def t1_external(external):
    rows = []
    for contrast in external["contrasts"]:
        scope = f"{contrast['scope']} ({CORPUS_BY_STATE_COUNT[contrast['paired_states']]})"
        rows.append(t1_row(family="external_temporal_baselines",
                           condition=FAMILIES["external_temporal_baselines"]["condition"],
                           side=None, kind=contrast["kind"], tested=contrast["tested"],
                           reference=contrast["reference"],
                           scope_states=contrast["paired_states"], scope=scope,
                           metric="normalized_ranking_regret", metric_block=contrast["regret"],
                           endpoint=None, estimand=REGRET_ESTIMAND,
                           source_artifact="issue-78-external-temporal-baselines-v1"))
    return rows


def t1_clevrer(clevrer):
    rows = []
    for contrast in clevrer["contrasts"]:
        for metric in ("carrier_mse", "position_mse", "velocity_mse"):
            if metric not in contrast:
                continue
            block = contrast[metric]
            rows.append(t1_row(family="clevrer_replication",
                               condition=FAMILIES["clevrer_replication"]["condition"], side=None,
                               kind=contrast["kind"], tested=contrast["tested"],
                               reference=contrast["reference"],
                               scope_states=block["paired_units"],
                               scope=f"{contrast['scope']} (paired scene-by-context units, "
                                     "not scenes)", metric=metric, metric_block=block,
                               endpoint=contrast["endpoint"], estimand=CLEVRER_ESTIMAND,
                               source_artifact="issue-79-clevrer-boundary-v1",
                               positive_favors=contrast["positive_favors"]))
    return rows


def t1_reactive(reactive):
    gate = reactive["pilot_gate"]
    return [{"family": "reactive_diagnostic", "condition": "UNEXECUTED", "side": None,
             "kind": None, "tested": None, "reference": None, "horizon": None, "endpoint": None,
             "scope_states": 0, "scope": "UNEXECUTED cell (stop branch)",
             "positive_is_improvement": None, "positive_favors": None, "metric": None,
             "mean": None, "interval_low": None, "interval_high": None, "excludes_zero": None,
             "interval_label": INTERVAL_LABEL, "estimand": REGRET_ESTIMAND,
             "source_artifact": "issue-80-reactive-diagnostic-v1", "status": "UNEXECUTED",
             "terminal_disposition": reactive["ticket_disposition"],
             "stop_explanation": reactive["stopExplanation"],
             "pilot_gate": {"prevalence": gate["prevalence"], "successes": gate["successes"],
                            "valid_executions": gate["valid_executions"], "floor": gate["floor"],
                            "passed": gate["passed"],
                            "typed_failures": gate["typed_failures"]}}]


def build_t1(sources):
    n1 = sources["issue-77-n1-diagnostic-v1"]["document"]
    n2 = sources["issue-77-n2-eval-v1"]["document"]
    external = sources["issue-78-external-temporal-baselines-v1"]["document"]
    clevrer = sources["issue-79-clevrer-boundary-v1"]["document"]
    reactive = sources["issue-80-reactive-diagnostic-v1"]["document"]
    rows = t1_regret_family("novphy_normal_mechanics", FAMILIES["novphy_normal_mechanics"]["condition"],
                            None, n1["contrasts"], "issue-77-n1-diagnostic-v1",
                            FAMILIES["novphy_normal_mechanics"]["states"])
    counts = n2["independent_state_counts"]
    for condition in FAMILIES["appearance_novelty"]["conditions"]:
        for side in sorted(n2["conditions"][condition]["per_side"]):
            side_contrasts = [contrast for contrast in n2["conditions"][condition]["contrasts"]
                              if contrast["side"] == side]
            rows += t1_regret_family("appearance_novelty", condition, side, side_contrasts,
                                     "issue-77-n2-eval-v1", counts[side])
    rows += t1_external(external)
    rows += t1_clevrer(clevrer)
    rows += t1_reactive(reactive)
    return rows


# ---------------------------------------------------------------- T2: work frontier

def t2_issue77_arm(family, condition, side, arm, seeds, wall_definition, candidate_definition,
                   artifact):
    published = [seeds[seed][arm] for seed in sorted(seeds)]
    states = published[0]["states"]
    for entry in published:
        if entry["states"] != states:
            raise ValueError(f"state count differs across seeds for {family}/{arm}")
    return {"family": family, "condition": condition, "side": side, "arm": arm, "states": states,
            "estimand": REGRET_ESTIMAND, "quality_metric": "normalized_ranking_regret",
            "mean_regret": seed_mean([entry["mean_regret"] for entry in published]),
            "wall_seconds_per_state": seed_mean([entry["mean_instrumented_model_seconds_per_state"]
                                                 for entry in published]),
            "wall_definition": wall_definition,
            "transition_linear_macs": seed_mean([entry["transition_linear_macs"]
                                                 for entry in published]),
            "linear_macs_definition": "total transition linear MACs over the state set "
                                      "(seed mean; linear MACs, not full FLOPs)",
            "transition_calls": seed_mean([entry["executed_calls"]["transition_calls"]
                                           for entry in published]),
            "parameters": None, "controller_parameters": None,
            "candidate_count": None,
            "candidate_count_definition": candidate_definition,
            "source_artifact": artifact, "status": "executed"}


def build_t2(sources):
    n1 = sources["issue-77-n1-diagnostic-v1"]["document"]
    n2 = sources["issue-77-n2-eval-v1"]["document"]
    external = sources["issue-78-external-temporal-baselines-v1"]["document"]
    clevrer = sources["issue-79-clevrer-boundary-v1"]["document"]
    readiness = sources["issue-74-matched-dynamics-v1"]["reduced"]
    rows = []
    for arm in sorted(n1["per_seed"][sorted(n1["per_seed"])[0]]):
        rows.append(t2_issue77_arm("novphy_normal_mechanics",
                                   FAMILIES["novphy_normal_mechanics"]["condition"], None, arm,
                                   n1["per_seed"], WALL_DEFINITIONS["issue-77-instrumented"],
                                   "13 fixed candidates per state reduced by typed branch failures "
                                   "(issue-77 N1 headroom)", "issue-77-n1-diagnostic-v1"))
    for condition in FAMILIES["appearance_novelty"]["conditions"]:
        per_side = n2["conditions"][condition]["per_side"]
        for side in sorted(per_side):
            for arm in sorted(per_side[side][sorted(per_side[side])[0]]):
                rows.append(t2_issue77_arm("appearance_novelty", condition, side, arm,
                                           per_side[side],
                                           WALL_DEFINITIONS["issue-77-instrumented"],
                                           "13 fixed candidates per state reduced by typed branch "
                                           "failures (issue-77 N2 headroom)",
                                           "issue-77-n2-eval-v1"))
    for arm in sorted(external["frontier"]):
        frontier_arm = external["frontier"][arm]
        seeds = sorted(frontier_arm["per_seed"])
        wall = seed_mean([frontier_arm["per_seed"][seed]["mean_per_state_wall_seconds"]
                          for seed in seeds])
        macs = seed_mean([frontier_arm["per_seed"][seed]["linear_macs"] for seed in seeds])
        calls = seed_mean([frontier_arm["per_seed"][seed]["transition_calls"] for seed in seeds])
        candidates = frontier_arm["per_seed"][seeds[0]]["candidates"]
        rows.append({"family": "external_temporal_baselines",
                     "condition": FAMILIES["external_temporal_baselines"]["condition"],
                     "side": None, "arm": arm,
                     "states": FAMILIES["external_temporal_baselines"]["states"],
                     "estimand": REGRET_ESTIMAND,
                     "quality_metric": "normalized_ranking_regret",
                     "mean_regret": seed_mean([external["per_seed"][seed][arm]["mean_regret"]
                                               for seed in sorted(external["per_seed"])]),
                     "wall_seconds_per_state": wall,
                     "wall_definition": WALL_DEFINITIONS["issue-78-locked"],
                     "transition_linear_macs": macs,
                     "linear_macs_definition": "total transition linear MACs over the state set "
                                               "(seed mean; linear MACs, not full FLOPs)",
                     "transition_calls": calls,
                     "parameters": frontier_arm["parameters"],
                     "controller_parameters": frontier_arm["controller_parameters"],
                     "candidate_count": candidates,
                     "candidate_count_definition": "frozen candidate decisions per seed "
                                                   "(issue-78 frontier)",
                     "source_artifact": "issue-78-external-temporal-baselines-v1",
                     "status": "executed"})
    reference = external["reference_frontier_continuous_h5"]
    seeds = sorted(reference["per_seed"])
    rows.append({"family": "external_temporal_baselines",
                 "condition": "reference continuous_h5 (frozen #77 records)", "side": None,
                 "arm": reference["system"], "states": 21, "estimand": REGRET_ESTIMAND,
                 "quality_metric": "normalized_ranking_regret",
                 "mean_regret": reference["mean_regret"],
                 "wall_seconds_per_state": seed_mean([reference["per_seed"][seed]
                                                      ["mean_per_state_wall_seconds"]
                                                      for seed in seeds]),
                 "wall_definition": WALL_DEFINITIONS["issue-78-reference"],
                 "transition_linear_macs": seed_mean([reference["per_seed"][seed]["linear_macs"]
                                                      for seed in seeds]),
                 "linear_macs_definition": "total transition linear MACs over the state set "
                                           "(seed mean; linear MACs, not full FLOPs)",
                 "transition_calls": seed_mean([reference["per_seed"][seed]["transition_calls"]
                                                for seed in seeds]),
                 "parameters": None, "controller_parameters": None,
                 "candidate_count": reference["per_seed"][seeds[0]]["candidates"],
                 "candidate_count_definition": "frozen candidate decisions per seed "
                                               "(issue-78 reference frontier)",
                 "source_artifact": "issue-78-external-temporal-baselines-v1",
                 "status": "executed"})
    for system in sorted(clevrer["per_seed"][sorted(clevrer["per_seed"])[0]]):
        published = [clevrer["per_seed"][seed][system] for seed in sorted(clevrer["per_seed"])]
        rows.append({"family": "clevrer_replication",
                     "condition": FAMILIES["clevrer_replication"]["condition"], "side": None,
                     "arm": system, "states": FAMILIES["clevrer_replication"]["states"],
                     "estimand": CLEVRER_ESTIMAND,
                     "quality_metric": "recursive_position_mse_e120",
                     "mean_regret": seed_mean([entry["curves"]["120"]["position_mse"]
                                               for entry in published]),
                     "wall_seconds_per_state": seed_mean([entry["mean_rollout_wall_seconds"]
                                                          for entry in published]),
                     "wall_definition": WALL_DEFINITIONS["issue-79-rollout"],
                     "transition_linear_macs": seed_mean([entry["linear_macs_per_step"]
                                                          for entry in published]),
                     "linear_macs_definition": "linear MACs per rollout step (issue-79; not a "
                                               "state-set total)",
                     "transition_calls": None,
                     "parameters": published[0]["active_parameters"],
                     "controller_parameters": None, "candidate_count": None,
                     "candidate_count_definition": "no candidate inventory on CLEVRER",
                     "source_artifact": "issue-79-clevrer-boundary-v1", "status": "executed"})
    for policy in sorted(readiness["policies"]):
        seeds = readiness["policies"][policy]
        rows.append({"family": "issue74_matched_dynamics_compute_reference",
                     "condition": "development cohort (selection on development evidence; "
                                  "training_compute_matched=false)",
                     "side": None, "arm": policy, "states": readiness["cohort_states"],
                     "estimand": REGRET_ESTIMAND,
                     "quality_metric": "normalized_ranking_regret",
                     "mean_regret": seed_mean([seeds[seed]["mean_regret"] for seed in sorted(seeds)]),
                     "wall_seconds_per_state": seed_mean(
                         [seeds[seed]["mean_perception_planning_seconds"] for seed in sorted(seeds)]),
                     "wall_definition": WALL_DEFINITIONS["issue-74-perception-planning"],
                     "transition_linear_macs": None, "linear_macs_definition": None,
                     "transition_calls": None,
                     "parameters": None, "controller_parameters": None, "candidate_count": None,
                     "candidate_count_definition": "not contributed by the allowed issue-74 fields",
                     "source_artifact": "issue-74-matched-dynamics-v1", "status": "executed"})
    rows.append({"family": "reactive_diagnostic", "condition": "UNEXECUTED", "side": None,
                 "arm": None, "states": 0, "estimand": REGRET_ESTIMAND, "quality_metric": None,
                 "mean_regret": None, "wall_seconds_per_state": None, "wall_definition": None,
                 "transition_linear_macs": None, "linear_macs_definition": None,
                 "transition_calls": None, "parameters": None,
                 "controller_parameters": None, "candidate_count": None,
                 "candidate_count_definition": None,
                 "source_artifact": "issue-80-reactive-diagnostic-v1", "status": "UNEXECUTED"})
    return rows


# ---------------------------------------------------------------- T3: mechanisms

def growth_row(family, condition, side, arm, per_seed, short, long_, metric, artifact):
    """per_seed: dict seed -> arm metrics with curves[metric] values (may be null)."""
    seeds = sorted(per_seed)
    short_value = seed_mean([per_seed[seed]["curves"][str(short)].get(metric) for seed in seeds])
    long_value = seed_mean([per_seed[seed]["curves"][str(long_)].get(metric) for seed in seeds])
    unavailable = short_value is None or long_value is None
    return {"mechanism": "recursive_error_growth", "family": family, "condition": condition,
            "side": side, "arm": arm, "metric": metric, "short_endpoint": short,
            "long_endpoint": long_, "mse_short": short_value, "mse_long": long_value,
            "growth_ratio": None if unavailable else long_value / short_value,
            "status": "curve_unavailable_in_source" if unavailable else "executed",
            "source_artifact": artifact}


def build_t3(sources, t1):
    n1 = sources["issue-77-n1-diagnostic-v1"]["document"]
    n2 = sources["issue-77-n2-eval-v1"]["document"]
    external = sources["issue-78-external-temporal-baselines-v1"]["document"]
    clevrer = sources["issue-79-clevrer-boundary-v1"]["document"]
    reactive = sources["issue-80-reactive-diagnostic-v1"]["document"]
    growth = []
    n1_seeds = sorted(n1["per_seed"])
    for arm in sorted(n1["per_seed"][n1_seeds[0]]):
        growth.append(growth_row("novphy_normal_mechanics",
                                 FAMILIES["novphy_normal_mechanics"]["condition"], None, arm,
                                 {seed: n1["per_seed"][seed][arm] for seed in n1_seeds},
                                 15, 600, "carrier_mse", "issue-77-n1-diagnostic-v1"))
    for condition in FAMILIES["appearance_novelty"]["conditions"]:
        per_side = n2["conditions"][condition]["per_side"]
        for side in sorted(per_side):
            seeds = sorted(per_side[side])
            for arm in sorted(per_side[side][seeds[0]]):
                growth.append(growth_row("appearance_novelty", condition, side, arm,
                                         {seed: per_side[side][seed][arm] for seed in seeds},
                                         15, 600, "carrier_mse", "issue-77-n2-eval-v1"))
    external_seeds = sorted(external["per_seed"])
    for arm in sorted(external["per_seed"][external_seeds[0]]):
        growth.append(growth_row("external_temporal_baselines",
                                 FAMILIES["external_temporal_baselines"]["condition"], None, arm,
                                 {seed: external["per_seed"][seed][arm] for seed in external_seeds},
                                 15, 600, "carrier_mse", "issue-78-external-temporal-baselines-v1"))
    clevrer_seeds = sorted(clevrer["per_seed"])
    for arm in sorted(clevrer["per_seed"][clevrer_seeds[0]]):
        growth.append(growth_row("clevrer_replication",
                                 FAMILIES["clevrer_replication"]["condition"], None, arm,
                                 {seed: clevrer["per_seed"][seed][arm] for seed in clevrer_seeds},
                                 15, 120, "position_mse", "issue-79-clevrer-boundary-v1"))
    symbolic = [row for row in t1 if row["kind"] == "symbolic_execution"]
    regime = [{"mechanism": "regime_dependence", "family": "clevrer_replication",
               "condition": FAMILIES["clevrer_replication"]["condition"], "horizon": entry["horizon"],
               "endpoint": entry["endpoint"], "still": entry["still"], "active": entry["active"],
               "difference_active_minus_still": entry["difference_active_minus_still"],
               "source_artifact": "issue-79-clevrer-boundary-v1"}
              for entry in clevrer["regime"]]
    for family in ("novphy_normal_mechanics", "appearance_novelty", "external_temporal_baselines",
                   "reactive_diagnostic"):
        regime.append({"mechanism": "regime_dependence", "family": family,
                       "condition": None, "horizon": None, "endpoint": None, "still": None,
                       "active": None, "difference_active_minus_still": None,
                       "status": "not_measured_in_family",
                       "source_artifact": FAMILIES[family]["source_artifact"]})
    n1_drops = sum(entry["dropped_candidates"] for entry in n1["headroom"])
    n2_drops = sum(entry["dropped_candidates"] for entry in n2["headroom"])
    reactive_gate = reactive["pilot_gate"]
    failures = [
        {"mechanism": "typed_failure_concentration", "family": "novphy_normal_mechanics",
         "total_dropped_candidates": n1_drops, "typed_failures_count": 0,
         "states_affected": sum(entry["dropped_candidates"] > 0 for entry in n1["headroom"]),
         "states": len(n1["headroom"]),
         "note": "dropped candidates are the typed branch failures (issue-77 N1 headroom); "
                 "excluded from candidates and reported, never worst-cased",
         "source_artifact": "issue-77-n1-diagnostic-v1"},
        {"mechanism": "typed_failure_concentration", "family": "appearance_novelty",
         "total_dropped_candidates": n2_drops, "typed_failures_count": 0,
         "states_affected": sum(entry["dropped_candidates"] > 0 for entry in n2["headroom"]),
         "states": len(n2["headroom"]),
         "note": "appearance-novelty families concentrate the typed branch-failure drops "
                 "(issue-77 N2 headroom)",
         "source_artifact": "issue-77-n2-eval-v1"},
        {"mechanism": "typed_failure_concentration", "family": "external_temporal_baselines",
         "total_dropped_candidates": 0, "typed_failures_count": len(external["typed_failures"]),
         "states_affected": 0, "states": 21,
         "note": "issue-78 published typed_failures list is empty" if not external["typed_failures"]
                 else "issue-78 published typed failures reproduced verbatim",
         "source_artifact": "issue-78-external-temporal-baselines-v1"},
        {"mechanism": "typed_failure_concentration", "family": "clevrer_replication",
         "total_dropped_candidates": 0, "typed_failures_count": 0,
         "states_affected": 0, "states": 12,
         "note": "no typed arm failures published by issue-79 (macro arm unsupported at Phase 0; "
                 "its estimand dropped by that ticket's frozen plan)",
         "source_artifact": "issue-79-clevrer-boundary-v1"},
        {"mechanism": "typed_failure_concentration", "family": "reactive_diagnostic",
         "total_dropped_candidates": 0,
         "typed_failures_count": len(reactive_gate["typed_failures"]),
         "states_affected": 0, "states": 0,
         "note": f"UNEXECUTED cell: pilot gate failed at prevalence "
                 f"{reactive_gate['prevalence']} < floor {reactive_gate['floor']} "
                 f"({reactive_gate['successes']}/{reactive_gate['valid_executions']} valid "
                 f"executions); {len(reactive_gate['typed_failures'])} typed decision failures "
                 f"(decision_failure: prior_candidate_absent) recorded; terminal disposition "
                 f"{reactive['ticket_disposition']}",
         "source_artifact": "issue-80-reactive-diagnostic-v1"},
    ]
    return {"recursive_error_growth": growth, "symbolic_execution_effects": symbolic,
            "regime_dependence": regime, "typed_failure_concentrations": failures}


# ---------------------------------------------------------------- claim registry

def row_key(row):
    tested = row.get("tested")
    parts = [row.get("family"), row.get("condition"), row.get("side"), row.get("kind"),
             tested if tested is not None else row.get("arm")]
    if row.get("horizon") is not None:
        parts.append(f"h{row['horizon']}")
    if row.get("endpoint") is not None:
        parts.append(f"e{row['endpoint']}")
    if row.get("metric") not in (None, "normalized_ranking_regret"):
        parts.append(str(row["metric"]))
    if row.get("scope_states") is not None:
        parts.append(f"n{row['scope_states']}")
    return ":".join("" if part is None else str(part) for part in parts)


def descriptor_key(descriptor):
    if "source_artifact" in descriptor and "family" not in descriptor:
        return f"{descriptor.get('source_artifact', '')}:{descriptor.get('field', '')}"
    tested = descriptor.get("tested")
    horizon = descriptor.get("horizon")
    if horizon is None and isinstance(tested, str):
        horizon = horizon_of(tested)
    return row_key({"family": descriptor.get("family"), "condition": descriptor.get("condition"),
                    "side": descriptor.get("side"), "kind": descriptor.get("kind"),
                    "tested": tested, "horizon": horizon,
                    "endpoint": descriptor.get("endpoint"), "metric": descriptor.get("metric"),
                    "scope_states": descriptor.get("scope_states")})


def resolve_rows(table_rows, descriptor):
    matched = []
    for row in table_rows:
        for field, value in descriptor.items():
            if row.get(field) != value:
                break
        else:
            matched.append(row)
    return matched


def resolve_one_each(table_rows, descriptors):
    """Resolve every descriptor to exactly one row. Returns (rows, None) on
    success or (None, descriptor) when any descriptor resolves != 1."""
    resolved = []
    for descriptor in descriptors:
        match = descriptor["match"] if "match" in descriptor else descriptor
        matched = resolve_rows(table_rows, match)
        if len(matched) != 1:
            return None, descriptor
        resolved.append(matched[0])
    return resolved, None


def check_expectation(expectation, rows, sources):
    """Frozen decision rule. Returns (token, checks)."""
    checks = []
    kind = expectation["type"]
    if kind == "all_rows":
        missing = False
        for descriptor in expectation["rows"]:
            matched = resolve_rows(rows, descriptor["match"])
            if len(matched) != 1:
                checks.append({"row": descriptor["match"], "observed": f"{len(matched)} rows",
                               "match": False})
                missing = True
                continue
            row = matched[0]
            observed_direction = direction(row["mean"]) if row["mean"] is not None else None
            ok = True
            expected_direction = descriptor.get("direction")
            if expected_direction is not None and observed_direction != expected_direction:
                ok = False
            expected_exclusion = descriptor.get("excludes_zero")
            if expected_exclusion is not None and bool(row["excludes_zero"]) != expected_exclusion:
                ok = False
            checks.append({"row": row_key(row), "expectation": descriptor,
                           "observed": {"direction": observed_direction,
                                        "excludes_zero": row["excludes_zero"],
                                        "mean": row["mean"]}, "match": ok})
        if missing:
            return "readiness_or_precision_insufficient", checks
        token = "supported" if checks and all(item["match"] for item in checks) \
            else "not_supported_by_this_experiment"
        return token, checks
    if kind == "side_sign_inconsistency":
        matched, bad = resolve_one_each(rows, expectation["rows"])
        if matched is None:
            return "readiness_or_precision_insufficient", [
                {"expectation": "exactly one row per contributing descriptor",
                 "observed": f"descriptor resolves != 1 row: {bad['match'] if 'match' in bad else bad}",
                 "match": False}]
        signs = {sign(row["mean"]) for row in matched}
        consistent = len(signs) == 1
        checks.append({"expectation": "signs are NOT all equal across sides",
                       "observed": sorted(f"{row['side']}:{row['mean']:+.4f}" for row in matched),
                       "match": not consistent})
        return ("supported" if not consistent else "not_supported_by_this_experiment"), checks
    if kind == "condition_sign_disagreement":
        matched, bad = resolve_one_each(rows, expectation["rows"])
        if matched is None:
            return "readiness_or_precision_insufficient", [
                {"expectation": "exactly one row per contributing descriptor",
                 "observed": f"descriptor resolves != 1 row: {bad['match'] if 'match' in bad else bad}",
                 "match": False}]
        by_cell = {}
        for row in matched:
            by_cell.setdefault(row["side"], {})[row["condition"]] = row["mean"]
        disagreements = sorted(side for side, cells in by_cell.items()
                               if len(cells) == len(expectation["conditions"])
                               and len({sign(value) for value in cells.values()}) > 1)
        checks.append({"expectation": "at least one matched (side, horizon) cell differs in sign "
                                      "between the conditions; conditions stay separate",
                       "observed": {side: cells for side, cells in sorted(by_cell.items())},
                       "disagreeing_sides": disagreements, "match": bool(disagreements)})
        return ("supported" if disagreements else "not_supported_by_this_experiment"), checks
    if kind == "upstream_token":
        artifact = expectation["artifact"]
        document = sources[artifact]["document"]
        value = document
        for field in expectation["field"].split("."):
            value = value[field]
        observed = value if isinstance(value, str) else value["disposition"]
        ok = observed == expectation["token"]
        checks.append({"expectation": f"{expectation['field']} == {expectation['token']}",
                       "observed": observed, "match": ok})
        return ("supported" if ok else "not_supported_by_this_experiment"), checks
    if kind == "field_values":
        document = sources[expectation["artifact"]]["reduced"]
        ok = True
        for field, expected in expectation["values"].items():
            observed = document
            for part in field.split("."):
                observed = observed[part]
            match = observed == expected
            ok = ok and match
            checks.append({"expectation": f"{field} == {expected!r}", "observed": observed,
                           "match": match})
        return ("supported" if ok else "not_supported_by_this_experiment"), checks
    if kind == "growth_margin":
        matched, bad = resolve_one_each(rows, expectation["rows"])
        if matched is None:
            return "readiness_or_precision_insufficient", [
                {"expectation": "exactly one growth row per contributing descriptor",
                 "observed": f"descriptor resolves != 1 row: {bad['match'] if 'match' in bad else bad}",
                 "match": False}]
        if any(row["growth_ratio"] is None for row in matched):
            return "readiness_or_precision_insufficient", [
                {"expectation": f"growth_ratio >= {expectation['margin']} for every arm",
                 "observed": "at least one arm has a null published curve aggregate",
                 "match": False}]
        failures = [row_key(row) for row in matched if row["growth_ratio"] < expectation["margin"]]
        checks.append({"expectation": f"growth_ratio >= {expectation['margin']} for every arm",
                       "observed_min": min(row["growth_ratio"] for row in matched),
                       "failing_rows": failures, "match": not failures})
        return ("supported" if not failures else "not_supported_by_this_experiment"), checks
    if kind == "boundary_shape":
        return check_boundary_shape(expectation, rows, sources)
    if kind == "untested_cell":
        matched = resolve_rows(rows, expectation["match"])
        ok = len(matched) == 1 and matched[0]["status"] == "UNEXECUTED"
        checks.append({"expectation": "exactly one UNEXECUTED reactive row present in T1",
                       "observed": f"{len(matched)} UNEXECUTED rows", "match": ok})
        return ("untested" if ok else "readiness_or_precision_insufficient"), checks
    raise ValueError(f"unknown expectation type {kind}")


def check_boundary_shape(expectation, rows, sources):
    checks = []
    contrast_rows, bad = resolve_one_each(rows, expectation["contrast_rows"])
    if contrast_rows is None:
        return "readiness_or_precision_insufficient", [
            {"expectation": "exactly one row per contributing contrast descriptor",
             "observed": f"descriptor resolves != 1 row: {bad['match'] if 'match' in bad else bad}",
             "match": False}]
    side_rows, bad = resolve_one_each(rows, expectation["side_rows"])
    if side_rows is None:
        return "readiness_or_precision_insufficient", [
            {"expectation": "exactly one row per contributing side descriptor",
             "observed": f"descriptor resolves != 1 row: {bad['match'] if 'match' in bad else bad}",
             "match": False}]
    unexecuted = resolve_rows(rows, expectation["unexecuted_row"])
    if len(unexecuted) != 1 or unexecuted[0]["status"] != "UNEXECUTED":
        return "readiness_or_precision_insufficient", [
            {"expectation": "exactly one UNEXECUTED reactive row present in T1",
             "observed": f"{len(unexecuted)} UNEXECUTED rows", "match": False}]
    ok = True
    for row, descriptor in zip(contrast_rows, expectation["contrast_rows"], strict=True):
        observed_direction = direction(row["mean"])
        match = observed_direction == descriptor["direction"]
        ok = ok and match
        checks.append({"row": row_key(row), "expectation": {"direction": descriptor["direction"]},
                       "observed": {"direction": observed_direction, "mean": row["mean"]},
                       "match": match})
    signs = {sign(row["mean"]) for row in side_rows}
    inconsistent = len(signs) > 1
    ok = ok and inconsistent
    checks.append({"expectation": "appearance-novelty few-shot h=1 training-effect signs are "
                                  "side-inconsistent (boundary condition, not reconciled)",
                   "observed": sorted(f"{row['side']}:{row['mean']:+.4f}" for row in side_rows),
                   "match": inconsistent})
    checks.append({"expectation": "reactive family present as an UNEXECUTED row (never dropped)",
                   "observed": "1 UNEXECUTED row", "match": True})
    return ("supported" if ok else "not_supported_by_this_experiment"), checks


def claim_definitions():
    nm = {"family": "novphy_normal_mechanics", "condition": "adapted", "side": None}
    an = {"family": "appearance_novelty"}
    et = {"family": "external_temporal_baselines"}
    return [
        {"id": "nm-training-effect-h1",
         "claim": "On the NovPhy normal-mechanics cell (adapted), the training effect "
                  "(hybrid_continuous vs continuous) has a positive paired mean regret difference "
                  "at h=1 and its descriptive interval excludes zero.",
         "families": ["novphy_normal_mechanics"], "conditions": ["adapted"],
         "expectation": {"type": "all_rows", "rows": [
             {"match": {**nm, "kind": "training_effect", "tested": "hybrid_continuous_h1",
                        "scope_states": 4}, "direction": "tested_favoring", "excludes_zero": True}]}},
        {"id": "nm-training-effect-h5-null",
         "claim": "On the NovPhy normal-mechanics cell, the training-effect paired mean regret "
                  "difference at h=5 does not separate from zero (descriptive interval includes zero).",
         "families": ["novphy_normal_mechanics"], "conditions": ["adapted"],
         "expectation": {"type": "all_rows", "rows": [
             {"match": {**nm, "kind": "training_effect", "tested": "hybrid_continuous_h5",
                        "scope_states": 4}, "direction": None, "excludes_zero": False}]}},
        {"id": "nm-training-effect-h15-null",
         "claim": "On the NovPhy normal-mechanics cell, the training-effect paired mean regret "
                  "difference at h=15 does not separate from zero (descriptive interval includes zero).",
         "families": ["novphy_normal_mechanics"], "conditions": ["adapted"],
         "expectation": {"type": "all_rows", "rows": [
             {"match": {**nm, "kind": "training_effect", "tested": "hybrid_continuous_h15",
                        "scope_states": 4}, "direction": None, "excludes_zero": False}]}},
        {"id": "nm-symbolic-execution-h1-null",
         "claim": "On the NovPhy normal-mechanics cell, the symbolic-execution contrasts "
                  "(hybrid_micro and hybrid_macro vs hybrid_continuous) do not separate from zero "
                  "at h=1.",
         "families": ["novphy_normal_mechanics"], "conditions": ["adapted"],
         "expectation": {"type": "all_rows", "rows": [
             {"match": {**nm, "kind": "symbolic_execution", "tested": "hybrid_micro_h1"}, "direction": None,
              "excludes_zero": False},
             {"match": {**nm, "kind": "symbolic_execution", "tested": "hybrid_macro_h1"}, "direction": None,
              "excludes_zero": False}]}},
        {"id": "nm-symbolic-execution-h15",
         "claim": "On the NovPhy normal-mechanics cell at h=15, the micro symbolic-execution "
                  "contrast does not separate from zero while the macro contrast separates with a "
                  "positive paired mean favoring macro symbolic execution.",
         "families": ["novphy_normal_mechanics"], "conditions": ["adapted"],
         "expectation": {"type": "all_rows", "rows": [
             {"match": {**nm, "kind": "symbolic_execution", "tested": "hybrid_micro_h15"},
              "direction": None, "excludes_zero": False},
             {"match": {**nm, "kind": "symbolic_execution", "tested": "hybrid_macro_h15"},
              "direction": "tested_favoring", "excludes_zero": True}]}},
        {"id": "an-few-shot-h1-side-inconsistent",
         "claim": "Under few-shot appearance-novelty adaptation, the h=1 training-effect paired "
                  "mean regret differences are NOT sign-consistent across the four appearance "
                  "sides; the h=1 direction does not carry over uniformly under novelty adaptation.",
         "families": ["appearance_novelty"], "conditions": ["few-shot"],
         "expectation": {"type": "side_sign_inconsistency", "rows": [
             {"match": {**an, "condition": "few-shot", "side": side, "kind": "training_effect",
                        "tested": "hybrid_continuous_h1"}} for side in
             ("type010101:normal", "type010101:novel", "type010102:normal", "type010102:novel")]}},
        {"id": "an-zero-shot-h1-side-inconsistent",
         "claim": "Under the zero-shot appearance-novelty condition, the h=1 training-effect "
                  "paired mean regret differences are NOT sign-consistent across the four sides.",
         "families": ["appearance_novelty"], "conditions": ["zero-shot"],
         "expectation": {"type": "side_sign_inconsistency", "rows": [
             {"match": {**an, "condition": "zero-shot", "side": side, "kind": "training_effect",
                        "tested": "hybrid_continuous_h1"}} for side in
             ("type010101:normal", "type010101:novel", "type010102:normal", "type010102:novel")]}},
        {"id": "an-condition-contradiction-h1",
         "claim": "At h=1 the zero-shot and few-shot appearance-novelty conditions disagree in "
                  "sign on at least one matched appearance side; the contradiction is reported as "
                  "a boundary condition and the two conditions are never pooled.",
         "families": ["appearance_novelty"], "conditions": ["zero-shot", "few-shot"],
         "expectation": {"type": "condition_sign_disagreement", "conditions": ["zero-shot", "few-shot"],
                         "rows": [{"match": {**an, "condition": condition, "side": side,
                                             "kind": "training_effect",
                                             "tested": "hybrid_continuous_h1"}}
                                  for condition in ("zero-shot", "few-shot")
                                  for side in ("type010101:normal", "type010101:novel",
                                               "type010102:normal", "type010102:novel")]}},
        {"id": "et-external-h1-direction",
         "claim": "On the external temporal-baselines cell (all 21 frozen states), both external "
                  "arms show positive paired mean regret differences over their in-ticket "
                  "continuous reference at h=1, with the TAWM interval excluding zero; this is a "
                  "within-ticket contrast, not a claim about the original papers' environments.",
         "families": ["external_temporal_baselines"], "conditions": ["adapted"],
         "expectation": {"type": "all_rows", "rows": [
             {"match": {**et, "kind": "external_vs_fixed", "tested": "tawm_h1", "scope_states": 21},
              "direction": "tested_favoring", "excludes_zero": True},
             {"match": {**et, "kind": "external_vs_fixed", "tested": "vlwm_h1", "scope_states": 21},
              "direction": "tested_favoring", "excludes_zero": None}]}},
        {"id": "et-q1-method-class",
         "claim": "As a method class, the TAWM/VLWM ports do not establish readiness or precision "
                  "for a method-class superiority decision (issue-78 q1 terminal disposition).",
         "families": ["external_temporal_baselines"], "conditions": ["adapted"],
         "expectation": {"type": "upstream_token", "artifact": "issue-78-external-temporal-baselines-v1",
                         "field": "dispositions.q1_method_class",
                         "token": "readiness_or_precision_insufficient"}},
        {"id": "et-q2-work-reported",
         "claim": "The external arms are reported on the full work-quality frontier (wall, MACs, "
                  "calls, parameters, candidate counts) with inference NOT equalized across arms "
                  "(issue-78 q2 disposition).",
         "families": ["external_temporal_baselines"], "conditions": ["adapted"],
         "expectation": {"type": "upstream_token", "artifact": "issue-78-external-temporal-baselines-v1",
                         "field": "dispositions.q2_work_reported", "token": "supported"}},
        {"id": "et-q3-training-effect",
         "claim": "The within-ticket training-effect analog contrasts are supported at the "
                  "descriptive level (issue-78 q3 disposition).",
         "families": ["external_temporal_baselines"], "conditions": ["adapted"],
         "expectation": {"type": "upstream_token", "artifact": "issue-78-external-temporal-baselines-v1",
                         "field": "dispositions.q3_training_effect", "token": "supported"}},
        {"id": "cv-horizon-decay-signature",
         "claim": "The NovPhy horizon-decay boundary signature does NOT replicate on CLEVRER "
                  "(issue-79 question-1 signature); reported as an out-of-family boundary "
                  "condition, not reconciled with the NovPhy families.",
         "families": ["clevrer_replication"], "conditions": ["out-of-family retrained"],
         "expectation": {"type": "upstream_token", "artifact": "issue-79-clevrer-boundary-v1",
                         "field": "dispositions.question_1_signature",
                         "token": "not_supported_by_this_experiment"}},
        {"id": "cv-regime-question",
         "claim": "The contact-activity regime question on CLEVRER is answered "
                  "not_supported_by_this_experiment at the pre-declared margins (issue-79 "
                  "question-2 regime disposition).",
         "families": ["clevrer_replication"], "conditions": ["out-of-family retrained"],
         "expectation": {"type": "upstream_token", "artifact": "issue-79-clevrer-boundary-v1",
                         "field": "dispositions.question_2_detail.disposition",
                         "token": "not_supported_by_this_experiment"}},
        {"id": "cv-training-effect-h1-e15",
         "claim": "On CLEVRER at endpoint e=15, the h=1 training-effect paired position-MSE "
                  "difference is positive (favoring the hybrid-continuous arm) with its "
                  "descriptive interval excluding zero.",
         "families": ["clevrer_replication"], "conditions": ["out-of-family retrained"],
         "expectation": {"type": "all_rows", "rows": [
             {"match": {"family": "clevrer_replication", "condition": "out-of-family retrained",
                        "kind": "training_effect", "tested": "hybrid_continuous_h1",
                        "endpoint": 15, "metric": "position_mse"},
              "direction": "tested_favoring", "excludes_zero": True}]}},
        {"id": "nm-recursive-growth",
         "claim": "On the NovPhy normal-mechanics cell, every system's seed-mean recursive "
                  "carrier MSE at t=600 is at least 10x its t=15 value (pre-declared margin 10).",
         "families": ["novphy_normal_mechanics"], "conditions": ["adapted"],
         "expectation": {"type": "growth_margin", "margin": 10, "rows": [
             {"match": {"mechanism": "recursive_error_growth", "family": "novphy_normal_mechanics",
                        "condition": "adapted", "arm": arm}} for arm in
             ("continuous_h1", "continuous_h5", "continuous_h15", "hybrid_continuous_h1",
              "hybrid_continuous_h5", "hybrid_continuous_h15", "hybrid_micro_h1",
              "hybrid_micro_h5", "hybrid_micro_h15", "hybrid_macro_h1", "hybrid_macro_h5",
              "hybrid_macro_h15")]}},
        {"id": "wf-comparator-disclosure",
         "claim": "The compute-frontier reference policy is the #74 selection "
                  "(selected_continuous_policy=continuous_h5, selection on development evidence) "
                  "and inference is NOT compute-matched across arms (training_compute_matched=false); "
                  "the frontier is reported, never equalized.",
         "families": ["issue74_matched_dynamics_compute_reference"], "conditions": ["development cohort"],
         "expectation": {"type": "field_values", "artifact": "issue-74-matched-dynamics-v1",
                         "values": {"selected_continuous_policy": "continuous_h5",
                                    "compute.training_compute_matched": False}}},
        {"id": "rc-reactive-first-shot",
         "claim": "Single-shot reactive control with the frozen carrier achieves nonzero "
                  "first-shot success prevalence on development states.",
         "families": ["reactive_diagnostic"], "conditions": ["UNEXECUTED"],
         "stop_branch": True,
         "note": "UNEXECUTED cell (mandatory stop branch): issue-80 terminated with "
                 "readiness_or_precision_insufficient (nonzero-prevalence pilot gate failed, "
                 "prevalence 0.0 < 0.10 floor, 0/45 valid executions, 3 typed decision failures) "
                 "before the full run; the claim is recorded untested, never dropped.",
         "expectation": {"type": "untested_cell",
                         "match": {"family": "reactive_diagnostic", "status": "UNEXECUTED"}}},
        {"id": "xb-h1-boundary-shape",
         "claim": "The h=1 shape is family-dependent: tested-favoring on the NovPhy "
                  "normal-mechanics training effect and on the external TAWM within-ticket "
                  "contrast, side-inconsistent under appearance-novelty few-shot adaptation, and "
                  "unexecuted for reactive control; cells stay side by side with no cross-family "
                  "pooling.",
         "families": ["novphy_normal_mechanics", "appearance_novelty",
                      "external_temporal_baselines", "reactive_diagnostic"],
         "conditions": ["adapted", "few-shot", "UNEXECUTED"],
         "expectation": {"type": "boundary_shape",
                         "contrast_rows": [
                             {"match": {**nm, "kind": "training_effect",
                                        "tested": "hybrid_continuous_h1", "scope_states": 4},
                              "direction": "tested_favoring"},
                             {"match": {**et, "kind": "external_vs_fixed", "tested": "tawm_h1",
                                        "scope_states": 21}, "direction": "tested_favoring"}],
                         "side_rows": [{"match": {**an, "condition": "few-shot", "side": side,
                                                  "kind": "training_effect",
                                                  "tested": "hybrid_continuous_h1"}} for side in
                                       ("type010101:normal", "type010101:novel",
                                        "type010102:normal", "type010102:novel")],
                         "unexecuted_row": {"family": "reactive_diagnostic",
                                            "status": "UNEXECUTED"}}},
    ]


def build_registry(sources, t1, t3):
    growth = t3["recursive_error_growth"]
    registry = []
    for claim in claim_definitions():
        table_rows = growth if claim["expectation"]["type"] == "growth_margin" else t1
        token, checks = check_expectation(claim["expectation"], table_rows, sources)
        expectation = claim["expectation"]
        if expectation["type"] == "boundary_shape":
            contributing = ([item["match"] for item in expectation["contrast_rows"]]
                            + [item["match"] for item in expectation["side_rows"]]
                            + [expectation["unexecuted_row"]])
        elif expectation["type"] == "upstream_token":
            contributing = [{"source_artifact": expectation["artifact"],
                             "field": expectation["field"]}]
        elif expectation["type"] == "field_values":
            contributing = [{"source_artifact": expectation["artifact"], "field": field}
                            for field in expectation["values"]]
        elif expectation["type"] == "untested_cell":
            contributing = [expectation["match"]]
        else:
            contributing = [descriptor.get("match", descriptor)
                            for descriptor in expectation.get("rows", [])]
        registry.append({"id": claim["id"], "claim": claim["claim"],
                         "families": claim["families"], "conditions": claim["conditions"],
                         "contributing_rows": contributing, "expectation": claim["expectation"],
                         "disposition": token, "checks": checks,
                         "note": claim.get("note"),
                         "stop_branch": bool(claim.get("stop_branch"))})
    if not all(entry["contributing_rows"] for entry in registry):
        raise ValueError("claim registry violated: no claim without contributing rows")
    return registry


# ---------------------------------------------------------------- report assembly

def build_report(sources):
    t1 = build_t1(sources)
    t2 = build_t2(sources)
    t3 = build_t3(sources, t1)
    registry = build_registry(sources, t1, t3)
    reactive = sources["issue-80-reactive-diagnostic-v1"]["document"]
    reactive_gate = reactive["pilot_gate"]
    families = json.loads(json.dumps(FAMILIES))
    families["reactive_diagnostic"]["terminal_disposition"] = reactive["ticket_disposition"]
    families["reactive_diagnostic"]["stop_explanation"] = reactive["stopExplanation"]
    families["reactive_diagnostic"]["pilot_gate"] = {
        "prevalence": reactive_gate["prevalence"], "successes": reactive_gate["successes"],
        "valid_executions": reactive_gate["valid_executions"], "floor": reactive_gate["floor"],
        "passed": reactive_gate["passed"], "cells": reactive_gate["cells"],
        "typed_failures": reactive_gate["typed_failures"]}
    inputs = []
    for artifact in sorted(sources):
        facts = digest(sources[artifact]["path"])
        inputs.append({"artifact": artifact, "path": str(sources[artifact]["path"].relative_to(ROOT)),
                       "role": sources[artifact]["role"], **facts})
    return {"schema": REPORT_SCHEMA, "identity": REPORT_IDENTITY,
            "plan_identity": None,  # filled by run()
            "feeds_ticket": 37,
            "descriptive_intervals": True,
            "pooling_rule": ("paired contrasts within each cell first (paired-bootstrap intervals "
                             "inherited verbatim from the source artifacts), cells reported side by "
                             "side; NO cross-family meta-analytic pooling; the only aggregation is "
                             "a seed-mean over the three published per-seed values of one cell"),
            "estimands": {"regret": REGRET_ESTIMAND, "clevrer": CLEVRER_ESTIMAND,
                          "interval_label": INTERVAL_LABEL, "scope": DESCRIPTIVE_SCOPE},
            "selection_disclosure": SELECTION_DISCLOSURE,
            "gpu_lock": f"never taken by this runner ({GPU_LOCK_PATH}); CPU-only synthesis with no "
                        "wall-time-measured GPU phase",
            "inputs": inputs, "families": families,
            "claim_boundary": ("synthesis of published development-stage diagnostics; no new "
                               "evidence, no reweighting, no upgraded prior dispositions (#15, #72, "
                               "#74, #75, #77, #78, #79, #80 unchanged); zero-shot and few-shot "
                               "never pooled; contradictions between families reported as boundary "
                               "conditions, not reconciled"),
            "diagnostics_complete": True,
            "tables": {"t1_advantage_vs_horizon": t1, "t2_work_frontier": t2,
                       "t3_mechanism": t3},
            "claim_registry": registry,
            "stop_branch": {"family": "reactive_diagnostic", "status": "UNEXECUTED",
                            "terminal_disposition": reactive["ticket_disposition"],
                            "stop_explanation": reactive["stopExplanation"],
                            "registry_tokens": ["untested"]},
            "limitations": [
                "at most four independent held-out N1 states and 17 appearance-novelty states; "
                "all intervals are descriptive",
                "regret contrasts reference the t=600 end-of-window replay cost: right-censored, "
                "not a settled cost",
                "wall-time definitions differ across families (instrumented model wall, locked "
                "decision wall, frozen-record reference wall, rollout wall, perception+planning "
                "wall) and are reported per row, never ranked against each other",
                "issue-74 rows contribute only the ticket-allowed fields; the continuous_h5 "
                "comparator retains selection optimism",
                "the reactive-control cell is UNEXECUTED (issue-80 terminal typed disposition) and "
                "contributes no outcome numbers",
                "linear MACs are not full FLOPs or matched total work"],
            "archived_release": False, "fresh_evaluation_opened": False,
            "final_evaluation_opened": False, "issue_64_authorized": False}


# ---------------------------------------------------------------- published emitters

def t1_csv_rows(t1):
    for row in t1:
        yield [row["family"], row["condition"] or "", row["side"] or "", row["kind"] or "",
               row["tested"] or "", row["reference"] or "",
               "" if row["horizon"] is None else row["horizon"],
               "" if row["endpoint"] is None else row["endpoint"], row["scope_states"],
               row["scope"] or "",
               "" if row["positive_is_improvement"] is None else row["positive_is_improvement"],
               row["positive_favors"] or "", row["metric"] or "",
               "" if row["mean"] is None else f"{row['mean']:.6f}",
               "" if row["interval_low"] is None else f"{row['interval_low']:.6f}",
               "" if row["interval_high"] is None else f"{row['interval_high']:.6f}",
               "" if row["excludes_zero"] is None else row["excludes_zero"],
               row["interval_label"], row["estimand"], row["source_artifact"], row["status"]]


def t2_csv_rows(t2):
    for row in t2:
        yield [row["family"], row["condition"] or "", row["side"] or "", row["arm"] or "",
               row["states"], row["estimand"] or "", row["quality_metric"] or "",
               "" if row["mean_regret"] is None else f"{row['mean_regret']:.6f}",
               "" if row["wall_seconds_per_state"] is None
               else f"{row['wall_seconds_per_state']:.6f}",
               row["wall_definition"] or "",
               "" if row["transition_linear_macs"] is None
               else f"{row['transition_linear_macs']:.2f}",
               row["linear_macs_definition"] or "",
               "" if row["transition_calls"] is None else f"{row['transition_calls']:.2f}",
               "" if row["parameters"] is None else row["parameters"],
               "" if row["controller_parameters"] is None else row["controller_parameters"],
               "" if row["candidate_count"] is None else row["candidate_count"],
               row["candidate_count_definition"] or "", row["source_artifact"], row["status"]]


def growth_csv_rows(growth):
    for row in growth:
        yield [row["family"], row["condition"], row["side"] or "", row["arm"], row["metric"],
               row["short_endpoint"], row["long_endpoint"],
               "n/a" if row["mse_short"] is None else f"{row['mse_short']:.6g}",
               "n/a" if row["mse_long"] is None else f"{row['mse_long']:.6g}",
               "n/a" if row["growth_ratio"] is None else f"{row['growth_ratio']:.6g}",
               row.get("status") or "executed", row["source_artifact"]]


def regime_csv_rows(regime):
    for row in regime:
        yield [row["family"], "" if row["horizon"] is None else row["horizon"],
               "" if row["endpoint"] is None else row["endpoint"],
               "n/a" if row["still"] is None else f"{row['still']['mean']:.6g}",
               "n/a" if row["active"] is None else f"{row['active']['mean']:.6g}",
               "n/a" if row["difference_active_minus_still"] is None
               else f"{row['difference_active_minus_still']:.6g}",
               row.get("status") or "executed", row["source_artifact"]]


def failure_csv_rows(failures):
    for row in failures:
        yield [row["family"], row["total_dropped_candidates"], row["typed_failures_count"],
               row["states_affected"], row["states"], row["note"], row["source_artifact"]]


def comparisons_csv(report):
    stream = io.StringIO()
    writer = csv.writer(stream)
    t1 = report["tables"]["t1_advantage_vs_horizon"]
    writer.writerow(("# section", "t1_advantage_vs_horizon (cells side by side; no cross-family "
                                  "pooling)",))
    writer.writerow(("family", "condition", "side", "kind", "tested", "reference", "horizon",
                     "endpoint", "scope_states", "scope", "positive_is_improvement",
                     "positive_favors", "metric", "mean", "interval_low",
                     "interval_high", "excludes_zero", "interval_label", "estimand",
                     "source_artifact", "status"))
    for values in t1_csv_rows(t1):
        writer.writerow(values)
    writer.writerow(())
    writer.writerow(("# section", "t2_work_frontier (per-row wall definitions; not equalized, "
                                  "not cross-family ranked)",))
    writer.writerow(("family", "condition", "side", "arm", "states", "estimand",
                     "quality_metric", "mean_regret", "wall_seconds_per_state", "wall_definition",
                     "transition_linear_macs", "linear_macs_definition", "transition_calls",
                     "parameters", "controller_parameters", "candidate_count",
                     "candidate_count_definition", "source_artifact", "status"))
    for values in t2_csv_rows(report["tables"]["t2_work_frontier"]):
        writer.writerow(values)
    writer.writerow(())
    t3 = report["tables"]["t3_mechanism"]
    writer.writerow(("# section", "t3_mechanism.recursive_error_growth",))
    writer.writerow(("family", "condition", "side", "arm", "metric", "short_endpoint",
                     "long_endpoint", "mse_short", "mse_long", "growth_ratio", "status",
                     "source_artifact"))
    for values in growth_csv_rows(t3["recursive_error_growth"]):
        writer.writerow(values)
    writer.writerow(())
    writer.writerow(("# section", "t3_mechanism.regime_dependence",))
    writer.writerow(("family", "horizon", "endpoint", "still_mean", "active_mean",
                     "difference_active_minus_still", "status", "source_artifact"))
    for values in regime_csv_rows(t3["regime_dependence"]):
        writer.writerow(values)
    writer.writerow(())
    writer.writerow(("# section", "t3_mechanism.typed_failure_concentrations",))
    writer.writerow(("family", "total_dropped_candidates", "typed_failures_count",
                     "states_affected", "states", "note", "source_artifact"))
    for values in failure_csv_rows(t3["typed_failure_concentrations"]):
        writer.writerow(values)
    return stream.getvalue()


def claim_registry_csv(registry):
    stream = io.StringIO()
    writer = csv.writer(stream)
    writer.writerow(("claim_id", "families", "conditions", "claim", "disposition",
                     "contributing_rows", "checks_passed", "checks_failed", "stop_branch", "note"))
    for entry in registry:
        keys = ";".join(descriptor_key(descriptor) for descriptor in entry["contributing_rows"])
        passed = sum(1 for check in entry["checks"] if check["match"])
        failed = sum(1 for check in entry["checks"] if not check["match"])
        writer.writerow((entry["id"], ";".join(entry["families"]), ";".join(entry["conditions"]),
                         entry["claim"], entry["disposition"], keys, passed, failed,
                         entry["stop_branch"], entry["note"] or ""))
    return stream.getvalue()


def advantage_csv(report):
    stream = io.StringIO()
    writer = csv.writer(stream)
    writer.writerow(("family", "condition", "side", "kind", "tested", "reference", "horizon",
                     "endpoint", "scope_states", "positive_is_improvement", "positive_favors",
                     "metric", "mean",
                     "interval_low", "interval_high", "excludes_zero", "interval_label",
                     "estimand", "source_artifact", "status"))
    for row in report["tables"]["t1_advantage_vs_horizon"]:
        writer.writerow([row["family"], row["condition"] or "", row["side"] or "",
                         row["kind"] or "", row["tested"] or "", row["reference"] or "",
                         "" if row["horizon"] is None else row["horizon"],
                         "" if row["endpoint"] is None else row["endpoint"],
                         row["scope_states"],
                         "" if row["positive_is_improvement"] is None else row["positive_is_improvement"],
                         row["positive_favors"] or "",
                         row["metric"] or "",
                         "" if row["mean"] is None else f"{row['mean']:.6f}",
                         "" if row["interval_low"] is None else f"{row['interval_low']:.6f}",
                         "" if row["interval_high"] is None else f"{row['interval_high']:.6f}",
                         "" if row["excludes_zero"] is None else row["excludes_zero"],
                         row["interval_label"], row["estimand"], row["source_artifact"],
                         row["status"]])
    return stream.getvalue()


def advantage_table_md(rows, title):
    lines = [f"### {title}", "",
             "| Family | Condition | Kind | Tested | Reference | h | States | Mean | Descriptive 95% interval | Excludes zero |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for row in rows:
        mean_text = "n/a" if row["mean"] is None else f"{row['mean']:+.4f}"
        interval = "n/a" if row["mean"] is None else \
            f"[{row['interval_low']:+.4f}, {row['interval_high']:+.4f}]"
        lines.append(f"| {row['family']} | {row['condition']} | {row['kind'] or '-'} | "
                     f"{row['tested'] or '-'} | {row['reference'] or '-'} | "
                     f"{row['horizon'] if row['horizon'] is not None else '-'} | "
                     f"{row['scope_states']} | {mean_text} | {interval} | "
                     f"{'' if row['excludes_zero'] is None else row['excludes_zero']} |")
    return lines


def findings_md(report):
    lines = ["# Issue-81 pooled boundary-curve synthesis - findings", ""]
    lines.append(f"Plan: {report['plan_identity']}. Feeds #37 (consolidated final report; not "
                 "executed here). Diagnostics complete: "
                 f"{report['diagnostics_complete']}.")
    lines.append("")
    lines.append(f"Pooling rule: {report['pooling_rule']}")
    lines.append("")
    lines.append(f"Regret estimand: {report['estimands']['regret']}")
    lines.append("")
    lines.append(f"Comparator disclosure: {report['selection_disclosure']}")
    lines.append("")
    lines.append(f"Interval semantics: {report['estimands']['scope']}")
    lines.append("")
    t1 = report["tables"]["t1_advantage_vs_horizon"]
    lines += advantage_table_md([row for row in t1 if row["status"] == "executed"
                                 and row["metric"] == "normalized_ranking_regret"],
                                "Advantage vs horizon - ranking-regret cells (side by side)")
    lines.append("")
    lines += advantage_table_md([row for row in t1 if row["status"] == "executed"
                                 and row["metric"] != "normalized_ranking_regret"],
                                "Advantage vs horizon - CLEVRER recursive-MSE contrasts "
                                "(out-of-family; no regret estimand)")
    lines.append("")
    lines.append("### Reactive-control diagnostic - UNEXECUTED cell (mandatory stop branch)")
    lines.append("")
    stop = report["stop_branch"]
    gate = report["families"]["reactive_diagnostic"]["pilot_gate"]
    lines.append(f"Terminal disposition: {stop['terminal_disposition']}. "
                 f"Stop explanation: {stop['stop_explanation']}.")
    lines.append("")
    lines.append(f"Pilot gate: prevalence {gate['prevalence']} "
                 f"({gate['successes']}/{gate['valid_executions']} valid executions) < floor "
                 f"{gate['floor']}; passed={gate['passed']}; typed failures: "
                 f"{len(gate['typed_failures'])}. No outcome numbers exist for this cell and none "
                 "are imputed; registry claims scoped to it are recorded `untested`.")
    lines.append("")
    lines.append("### Work frontier (regret vs per-state wall; per-row wall definitions; "
                 "not equalized)")
    lines.append("")
    lines.append("| Family | Condition | Arm | States | Mean regret / quality | Wall s/state | "
                 "Linear MACs | Parameters |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in report["tables"]["t2_work_frontier"]:
        quality = "n/a" if row["mean_regret"] is None else \
            (f"{row['mean_regret']:.4f}" if row["quality_metric"] == "normalized_ranking_regret"
             else f"{row['mean_regret']:.4g} ({row['quality_metric']})")
        wall = "n/a" if row["wall_seconds_per_state"] is None else \
            f"{row['wall_seconds_per_state']:.4f}"
        macs = "n/a" if row["transition_linear_macs"] is None else \
            f"{row['transition_linear_macs']:.3g}"
        params = "n/a" if row["parameters"] is None else row["parameters"]
        lines.append(f"| {row['family']} | {row['condition']} | {row['arm'] or '-'} | "
                     f"{row['states']} | {quality} | {wall} | {macs} | {params} |")
    lines.append("")
    t3 = report["tables"]["t3_mechanism"]
    lines.append("### Mechanism: recursive-error growth (seed-mean carrier/position MSE)")
    lines.append("")
    lines.append("| Family | Arm | Metric | t_short | t_long | MSE short | MSE long | Growth ratio |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in t3["recursive_error_growth"]:
        short = "n/a" if row["mse_short"] is None else f"{row['mse_short']:.4g}"
        long_ = "n/a" if row["mse_long"] is None else f"{row['mse_long']:.4g}"
        ratio = "n/a" if row["growth_ratio"] is None else f"{row['growth_ratio']:.4g}"
        lines.append(f"| {row['family']} | {row['arm']} | {row['metric']} | "
                     f"{row['short_endpoint']} | {row['long_endpoint']} | {short} | {long_} | "
                     f"{ratio} |")
    lines.append("")
    lines.append("### Mechanism: regime dependence (CLEVRER active vs still; other families "
                 "did not measure a regime split)")
    lines.append("")
    lines.append("| Family | h | Endpoint | Still mean | Active mean | Active-still difference |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for row in t3["regime_dependence"]:
        still = "n/a" if row["still"] is None else f"{row['still']['mean']:.4g}"
        active = "n/a" if row["active"] is None else f"{row['active']['mean']:.4g}"
        diff = "n/a" if row["difference_active_minus_still"] is None \
            else f"{row['difference_active_minus_still']:.4g}"
        lines.append(f"| {row['family']} | {row['horizon'] if row['horizon'] is not None else '-'} | "
                     f"{row['endpoint'] if row['endpoint'] is not None else '-'} | {still} | "
                     f"{active} | {diff} |")
    lines.append("")
    lines.append("### Mechanism: typed-failure concentrations")
    lines.append("")
    lines.append("| Family | Dropped candidates | Typed failures | States affected | Note |")
    lines.append("| --- | --- | --- | --- | --- |")
    for row in t3["typed_failure_concentrations"]:
        lines.append(f"| {row['family']} | {row['total_dropped_candidates']} | "
                     f"{row['typed_failures_count']} | {row['states_affected']} | {row['note']} |")
    lines.append("")
    lines.append("## Claim registry")
    lines.append("")
    lines.append("| Claim | Disposition | Contributing rows |")
    lines.append("| --- | --- | --- |")
    for entry in report["claim_registry"]:
        keys = "; ".join(descriptor_key(item) for item in entry["contributing_rows"])
        lines.append(f"| {entry['id']} | {entry['disposition']} | {keys} |")
    lines.append("")
    lines.append("## Boundary conditions (contradictions reported, not reconciled)")
    lines.append("")
    lines.append("- The h=1 training-effect direction is tested-favoring on the NovPhy "
                 "normal-mechanics cell and on the external TAWM within-ticket contrast, but is "
                 "NOT sign-consistent across appearance-novelty sides under few-shot adaptation; "
                 "the zero-shot and few-shot appearance conditions disagree in sign on at least "
                 "one matched side (see `an-condition-contradiction-h1`).")
    lines.append("- The horizon-decay signature does not replicate on CLEVRER "
                 "(`cv-horizon-decay-signature`) while the CLEVRER h=1 e=15 training-effect "
                 "position-MSE contrast separates tested-favoring (`cv-training-effect-h1-e15`); "
                 "both are reported without reconciliation.")
    lines.append("- Wall-time axes differ per source (see frontier row definitions) and are never "
                 "ranked against each other.")
    lines.append("")
    lines.append("## Limitations")
    lines.append("")
    for item in report["limitations"]:
        lines.append(f"- {item}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- modes

def load_plan(output):
    plan = read(output / "plan.json")
    if plan["schema"] != SCHEMA or plan["identity"] != IDENTITY:
        raise ValueError("frozen plan identity/schema mismatch")
    frozen_definitions = {"families": json.loads(json.dumps(FAMILIES)),
                          "pooling_rule": ("paired contrasts within each cell first "
                                           "(paired-bootstrap intervals inherited verbatim from "
                                           "the source artifacts), cells reported side by side; "
                                           "NO cross-family meta-analytic pooling; the only "
                                           "aggregation is a seed-mean over the three published "
                                           "per-seed values of one cell"),
                          "claim_registry": {"claims": json.loads(json.dumps(claim_definitions()))}}
    if plan["families"] != frozen_definitions["families"] or \
            plan["pooling_rule"] != frozen_definitions["pooling_rule"] or \
            plan["claim_registry"]["claims"] != frozen_definitions["claim_registry"]["claims"]:
        raise ValueError("frozen plan families/pooling-rule/claim-registry mismatch with code")
    if plan["tables"]["t1_advantage_vs_horizon"]["columns"] != list(T1_COLUMNS) or \
            plan["tables"]["t2_work_frontier"]["columns"] != list(T2_COLUMNS):
        raise ValueError("frozen plan table columns mismatch with code")
    for entry in plan["inputs"]:
        current = digest(ROOT / entry["path"])
        if current != {"sha256": entry["sha256"], "bytes": entry["bytes"]}:
            raise ValueError(f"frozen input changed after plan freeze: {entry['path']}")
    return plan


def make_plan(sources):
    inputs = []
    for artifact in sorted(sources):
        facts = digest(sources[artifact]["path"])
        inputs.append({"artifact": artifact,
                       "path": str(sources[artifact]["path"].relative_to(ROOT)),
                       "role": sources[artifact]["role"], **facts})
    return {"schema": SCHEMA, "identity": IDENTITY,
            "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "inputs": inputs,
            "families": FAMILIES,
            "pooling_rule": ("paired contrasts within each cell first (paired-bootstrap intervals "
                             "inherited verbatim from the source artifacts), cells reported side by "
                             "side; NO cross-family meta-analytic pooling; the only aggregation is "
                             "a seed-mean over the three published per-seed values of one cell"),
            "estimands": {"regret": REGRET_ESTIMAND, "clevrer": CLEVRER_ESTIMAND,
                          "interval_label": INTERVAL_LABEL, "scope": DESCRIPTIVE_SCOPE},
            "issue74_allowed_fields": ["selected_continuous_policy",
                                       "per_seed.policies.*.mean_regret",
                                       "per_seed.policies.*.mean_perception_planning_seconds",
                                       "compute",
                                       "plan.json:development.states (cohort-size provenance only)"],
            "tables": {
                "t1_advantage_vs_horizon": {
                    "columns": list(T1_COLUMNS),
                    "definition": "every published paired contrast of every executed family plus "
                                  "the reactive UNEXECUTED row; horizons 1/5/15 where the source "
                                  "system carries one; intervals copied verbatim with their "
                                  "descriptive label"},
                "t2_work_frontier": {
                    "columns": list(T2_COLUMNS),
                    "definition": "one row per (family, condition, side, arm): seed-mean regret or "
                                  "quality metric vs seed-mean per-state wall with the wall "
                                  "definition recorded per row, plus MACs, calls, parameters and "
                                  "candidate counts where the source publishes them; not equalized, "
                                  "not cross-family ranked"},
                "t3_mechanism": {
                    "recursive_error_growth": "seed-mean MSE at the short vs long endpoint per arm "
                                              "per family",
                    "symbolic_execution_effects": "the T1 symbolic_execution rows",
                    "regime_dependence": "issue-79 active-vs-still regime rows; other families "
                                         "recorded as not_measured_in_family",
                    "typed_failure_concentrations": "dropped candidates and published typed "
                                                    "failures per family, including the reactive "
                                                    "pilot-gate typed failures"}},
            "figure": {"advantage_vs_horizon": {
                "path": "figures/advantage_vs_horizon.csv",
                "definition": "machine-readable advantage-vs-horizon curve: one row per T1 "
                              "contrast (family, condition, side, kind, tested, reference, "
                              "horizon, endpoint, mean, descriptive interval, excludes_zero); "
                              "includes the reactive UNEXECUTED row"}},
            "claim_registry": {
                "decision_rule": "supported iff every contributing row is present and matches its "
                                 "frozen expectation; not_supported_by_this_experiment iff any "
                                 "row contradicts; readiness_or_precision_insufficient iff a row "
                                 "is missing or lacks its statistic; claims scoped to the "
                                 "UNEXECUTED reactive cell are recorded `untested` (stop branch)",
                "claims": claim_definitions()},
            "compute": {"device": "cpu", "gpu_lock": f"not taken ({GPU_LOCK_PATH}); no "
                                                     "wall-time-measured GPU phase exists in this "
                                                     "runner",
                        "new_inference": "none; published intervals are inherited verbatim",
                        "derived_artifact_budget_bytes": DERIVED_BYTES_BUDGET},
            "validation_command": "python -u -m scripts.run_boundary_synthesis --validate",
            "stop_branch": "the reactive-control family enters as an explicitly labelled "
                           "UNEXECUTED cell with its terminal disposition "
                           "readiness_or_precision_insufficient; never dropped, never imputed"}


def prepare(output):
    if (output / "plan.json").exists():
        plan = load_plan(output)
        log(f"plan already frozen at {plan['frozen_at']}; inputs verified against pinned hashes")
        return plan
    sources = load_sources_for_freeze()
    plan = make_plan(sources)
    write(output / "plan.json", plan)
    log(f"plan frozen: {len(plan['inputs'])} pinned inputs, {len(plan['claim_registry']['claims'])} "
        f"registered claims; no synthesis number computed yet")
    return plan


def load_sources_for_freeze():
    sources = load_inputs()
    sources["issue-74-matched-dynamics-v1"]["reduced"] = issue74_allowed_fields(
        sources["issue-74-matched-dynamics-v1"]["document"],
        sources["issue-74-matched-dynamics-v1/plan.json"]["document"])
    return sources


def dry_run():
    sources = load_sources_for_freeze()
    n1 = sources["issue-77-n1-diagnostic-v1"]["document"]
    n2 = sources["issue-77-n2-eval-v1"]["document"]
    external = sources["issue-78-external-temporal-baselines-v1"]["document"]
    clevrer = sources["issue-79-clevrer-boundary-v1"]["document"]
    reactive = sources["issue-80-reactive-diagnostic-v1"]["document"]
    readiness = sources["issue-74-matched-dynamics-v1"]["reduced"]
    claims = claim_definitions()
    log(f"no-write dry-run; CPU-only; GPU lock {GPU_LOCK_PATH} NOT taken")
    log(f"inputs: {len(sources)} artifacts verified readable")
    log(f"cells: novphy_normal_mechanics states={FAMILIES['novphy_normal_mechanics']['states']} "
        f"contrasts={len(n1['contrasts'])}; appearance_novelty zero-shot="
        f"{len(n2['conditions']['zero-shot']['contrasts'])} few-shot="
        f"{len(n2['conditions']['few-shot']['contrasts'])} states="
        f"{FAMILIES['appearance_novelty']['states']}; external_temporal_baselines "
        f"contrasts={len(external['contrasts'])} frontier_arms={len(external['frontier'])}; "
        f"clevrer_replication contrasts={len(clevrer['contrasts'])} regime_rows="
        f"{len(clevrer['regime'])}; reactive_diagnostic status="
        f"{reactive['ticket_disposition']} (UNEXECUTED cell)")
    log(f"issue-74 allowed-field policies: {sorted(readiness['policies'])}")
    log(f"claim registry: {len(claims)} claims; stop-branch claims="
        f"{sum(1 for claim in claims if claim.get('stop_branch'))}")
    return 0


def run(output, plan):
    sources = load_sources_for_freeze()
    report = build_report(sources)
    report["plan_identity"] = plan["identity"]
    began = time.monotonic()
    total = len(report["tables"]["t1_advantage_vs_horizon"]) + \
        len(report["tables"]["t2_work_frontier"])
    log(f"computing synthesis from frozen plan: t1={len(report['tables']['t1_advantage_vs_horizon'])} "
        f"t2={len(report['tables']['t2_work_frontier'])} t3 growth="
        f"{len(report['tables']['t3_mechanism']['recursive_error_growth'])} regime="
        f"{len(report['tables']['t3_mechanism']['regime_dependence'])} registry="
        f"{len(report['claim_registry'])} rows total={total} (CPU-only)")
    write(output / "compute.json", report)
    tokens = {}
    for entry in report["claim_registry"]:
        tokens[entry["disposition"]] = tokens.get(entry["disposition"], 0) + 1
    log(f"synthesis computed in {time.monotonic() - began:.2f}s; registry tokens: "
        f"{json.dumps(tokens, sort_keys=True)}")
    return report


def publish(output):
    plan = load_plan(output)
    if not (output / "compute.json").exists():
        raise ValueError("compute.json missing; run --run before --publish")
    report = read(output / "compute.json")
    if report.get("plan_identity") != plan["identity"] or \
            [(entry["artifact"], entry["sha256"]) for entry in report["inputs"]] != \
            [(entry["artifact"], entry["sha256"]) for entry in plan["inputs"]]:
        raise ValueError("compute.json was produced under a different frozen plan or pinned inputs")
    summary = report
    write(output / "summary.json", summary)
    write_text(output / "comparisons.csv", comparisons_csv(report))
    write_text(output / "findings.md", findings_md(report))
    write_text(output / "claim-registry.csv", claim_registry_csv(report["claim_registry"]))
    write_text(output / "figures" / "advantage_vs_horizon.csv", advantage_csv(report))
    derived = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
    if derived > DERIVED_BYTES_BUDGET:
        raise ValueError(f"derived artifacts exceed the {DERIVED_BYTES_BUDGET}-byte budget")
    log(f"published summary.json comparisons.csv findings.md claim-registry.csv "
        f"figures/advantage_vs_horizon.csv; derived bytes={derived}")
    return report


def validate(output):
    plan = load_plan(output)
    sources = load_sources_for_freeze()
    report = build_report(sources)
    report["plan_identity"] = plan["identity"]
    expected = {"summary.json": json.dumps(report, indent=2, sort_keys=True,
                                           allow_nan=False).encode() + b"\n",
                "comparisons.csv": comparisons_csv(report).encode(),
                "findings.md": findings_md(report).encode(),
                "claim-registry.csv": claim_registry_csv(report["claim_registry"]).encode(),
                "figures/advantage_vs_horizon.csv": advantage_csv(report).encode()}
    for name, content in expected.items():
        path = output / name
        if not path.exists():
            raise ValueError(f"published artifact missing: {name}")
        if path.read_bytes() != content:
            raise ValueError(f"published {name} differs from bound source evidence")
    log("exact saved-evidence validation passed: every table and figure recomputed from the "
        "input artifacts matches the published bytes")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    for mode in ("dry-run", "prepare", "run", "publish", "validate"):
        modes.add_argument("--" + mode, action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    try:
        if args.dry_run:
            return dry_run()
        output = args.output
        if args.prepare:
            prepare(output)
            return 0
        plan = load_plan(output)
        if args.run:
            run(output, plan)
        elif args.publish:
            publish(output)
        else:
            validate(output)
        return 0
    except (ValueError, OSError, KeyError) as error:
        log(f"error: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
