"""Issue #109 work items 2-3: rendered capture-reliability smoke on the fixed pipeline.

Re-captures all 208 frozen #77 N1 branches (16 development lineages x 13
candidate actions; exposed lineages, no sealed access) with
``scripts/capture_pipeline_v2.py`` and publishes:

* the typed-failure rate against the frozen < 5 % gate;
* the frozen timeout-outcome audit (#90 rerun on the new pipeline), plus the
  retrospective #87 / #77-N1 failure-outcome contrasts, measured with engine
  truth from this uniform pipeline;
* determinism against the #77 N1 / #87 / #89 references;
* the per-stage throughput profile used to budget the sealed cohort.

Modes: --dry-run / --prepare / --run / --scan / --publish / --validate.
Validation: ``python -u -m scripts.run_capture_reliability_smoke --validate``.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import sys
import time

from scripts import capture_pipeline_v2 as pipeline

ROOT = Path(__file__).resolve().parents[1]
IDENTITY = "issue-109-capture-smoke-v1"
OUTPUT = ROOT / ".local-artifacts" / IDENTITY
MEDIA = ROOT / "data" / "issue-109-capture-smoke"
N1 = ROOT / ".local-artifacts" / "issue-77-n1-v1"
EIGHTY_SEVEN = ROOT / ".local-artifacts" / "issue-87-closed-loop-oracle-v1"
EIGHTY_NINE = ROOT / ".local-artifacts" / "issue-89-oracle-completion-v1"
VALIDATION_COMMAND = "python -u -m scripts.run_capture_reliability_smoke --validate"
WORKERS = 6
SCAN_PROCESSES = 12
LIMITS = {
    "workers": WORKERS,
    "attempt_seconds": 1800,
    "worker_cpu_rss_mib": 4096,
    "aggregate_cpu_rss_mib": 32768,
    "minimum_free_bytes": 100 * 2**30,
    "artifact_bytes": 150 * 2**30,
    "decision_fixed_step": 30000,
    **pipeline.DEFAULT_LIMITS,
}
AUDIT = {
    "failure_rate_gate": 0.05,
    "bias_margin": 0.05,
    "contrast_margin": 0.10,
    "bootstrap": {"draws": 10000, "seed": 10901, "quantiles": [0.025, 0.975],
                  "unit": "branch", "cluster": "source member (lineage)",
                  "label": "DESCRIPTIVE"},
    "outcomes": {
        "pig_removed": "engine-truth #85 channel verdict over the retained shot window",
        "native_clear": "native terminal kind == level_clear",
        "right_censored": "shot window reached 30000 native steps without a terminal",
    },
    "precision_guards": {"min_target_slots": 20, "min_target_members": 8,
                         "min_usable_draws_fraction": 0.9},
    "part_a": {
        "question": ("On the fixed pipeline, can the typed capture failures bias any "
                     "branch-level outcome prevalence by more than the bias margin?"),
        "bound": ("B = typed failures / scheduled branches; for any [0,1] outcome, "
                  "|p_executed - p_scheduled| = B * |p_executed - p_failed| <= B"),
        "direct_contrast": ("where >= 5 smoke-failed branches carry a #77 N1 v1 reference "
                            "outcome: mean v1 reference outcome over smoke-failed minus over "
                            "smoke-executed branches (same-source outcomes), member-clustered"),
        "verdict": {
            "outcome_correlated": ("the direct contrast is computable and, for some outcome, "
                                   "|difference| >= contrast_margin with the interval excluding 0"),
            "bias_bounded_below_margin": ("not outcome_correlated and B <= bias_margin: the failure "
                                          "mechanism cannot move any outcome prevalence by more "
                                          "than the margin whatever its correlation"),
            "indeterminate": "otherwise",
        },
        "disposition": {"supported": "verdict bias_bounded_below_margin",
                        "not_supported_by_this_experiment": "verdict outcome_correlated",
                        "readiness_or_precision_insufficient": "verdict indeterminate"},
    },
    "part_b": {
        "question": ("Were the old-pipeline typed failures (#87 oracle timeouts; #77 N1 v1 capture "
                     "failures) correlated with engine-truth outcomes, measured for every branch on "
                     "this uniform pipeline?"),
        "strata": {
            "b1_issue87": ("#87 terminal oracle slots joined to the smoke branch with the same "
                           "branch identity: target = TimeoutError slots, reference = executed slots; "
                           "the 3 stability slots are a separate typed stratum, never pooled"),
            "b2_issue77_n1": ("#77 N1 v1 branches: target = typed failures, reference = complete"),
        },
        "statistic": ("target mean minus reference mean of the smoke outcome over slots whose smoke "
                      "branch executed; member-clustered percentile bootstrap"),
        "verdict": {
            "outcome_correlated": ("no precision guard tripped and some outcome shows "
                                   "|difference| >= contrast_margin with the interval excluding 0"),
            "outcome_blind": ("no precision guard tripped and every outcome has an interval "
                              "entirely inside (-contrast_margin, +contrast_margin)"),
            "indeterminate": "otherwise",
        },
        "disposition": {"supported": "verdict outcome_correlated (the failures were outcome-correlated)",
                        "not_supported_by_this_experiment": "verdict outcome_blind",
                        "readiness_or_precision_insufficient": "verdict indeterminate"},
    },
    "determinism": ("agreement of stop kind (vs #77 N1 v1 complete branches) and of engine-truth "
                    "pig_removed (vs #87 executed slots and #89 completion slots) for the same branch"),
    "throughput": ("per-stage wall seconds (capture_pipeline_v2.StageClock) over complete branches: "
                   "mean, p50, p90; per-branch wall; campaign wall; branches per hour at the frozen "
                   "worker count; publish-time engine-outcome scan CPU seconds"),
}


def log(message):
    print(f"[{IDENTITY}] {message}", flush=True)


def sha256_file(path):
    return "sha256:" + sha256(Path(path).read_bytes()).hexdigest()


def n1_plan():
    return json.loads((N1 / "plan.json").read_text())


def records():
    from scripts import run_issue_77_n1 as n1
    plan = n1_plan()
    members = {member["identity"]: member for member in plan["members"]}
    return [n1._dispatch_record(members[branch["source_member_identity"]], branch)
            for branch in plan["branches"]]


def load_plan():
    plan = json.loads((OUTPUT / "plan.json").read_text())
    for item in plan["inputs"]:
        if sha256_file(ROOT / item["artifact"]) != item["sha256"]:
            raise ValueError(f"frozen input changed: {item['artifact']}")
    return plan


def dry_run():
    items = records()
    log(f"membership: {len(items)} branches over {len({r['source_member_identity'] for r in items})} "
        f"members; workers {WORKERS}; pipeline {pipeline.PIPELINE_IDENTITY}")
    log(f"limits: {json.dumps(LIMITS, sort_keys=True)}")
    log("dry run computes no statistic and writes nothing")


def prepare():
    if (OUTPUT / "plan.json").exists():
        raise ValueError("plan.json is frozen; refusing to overwrite")
    items = records()
    inputs = [{"name": name, "artifact": str(path.relative_to(ROOT)), "sha256": sha256_file(path)}
              for name, path in (("issue77_n1_plan", N1 / "plan.json"),
                                 ("issue77_n1_ledger", N1 / "ledger.json"),
                                 ("issue77_n1_coverage", N1 / "coverage.json"),
                                 ("issue87_ledger", EIGHTY_SEVEN / "ledger.json"),
                                 ("issue87_plan", EIGHTY_SEVEN / "plan.json"),
                                 ("issue89_summary", EIGHTY_NINE / "summary.json"))]
    plan = {
        "schema": "issue_109_capture_smoke_plan_v1",
        "identity": IDENTITY,
        "version": 1,
        "role": "terminal",
        "frozen_at": pipeline.utc_now(),
        "frozen_before_any_smoke_outcome": True,
        "validation_command": VALIDATION_COMMAND,
        "pipeline": {"identity": pipeline.PIPELINE_IDENTITY,
                     "module_sha256": sha256_file(ROOT / "scripts/capture_pipeline_v2.py"),
                     "port_base": pipeline.PORT_BASE, "port_slot_stride": pipeline.PORT_SLOT_STRIDE,
                     "character_box_pad_pixels": pipeline.CHARACTER_BOX_PAD_PIXELS},
        "runner_sha256_at_freeze": sha256_file(Path(__file__)),
        "membership": {"source": "issue-77-n1-v1 plan branches, plan order",
                       "branches": [r["identity"] for r in items],
                       "members": sorted({r["source_member_identity"] for r in items}),
                       "exposure": "development/exposed #77 N1 lineages; no sealed or final access"},
        "limits": LIMITS,
        "retry_policy": "technical_retries 0; every scheduled branch executed once or retained as a typed failure",
        "audit": AUDIT,
        "inputs": inputs,
        "claim_boundary": ("rendered re-capture of 208 exposed development branches on the fixed "
                           "pipeline; reliability, determinism and throughput evidence only; no model "
                           "is trained or scored; prior dispositions (#77, #87, #89, #90) are "
                           "read-only inputs and are not reopened"),
        "development_evidence": ("pre-freeze engineering runs (EXPLORATORY, not scored): "
                                 ".local-artifacts/issue-109-capture-dev/dev1 = issue-77-n1-002-a01, "
                                 "-005-a02, -003-a04 on 3 workers, 3/3 complete"),
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    pipeline.write_json(OUTPUT / "plan.json", plan)
    if not (OUTPUT / "player").exists():
        shutil.copytree(N1 / "player", OUTPUT / "player", symlinks=True)
    log(f"plan frozen at {plan['frozen_at']}: {len(items)} branches")


def run():
    plan = load_plan()
    if sha256_file(ROOT / "scripts/capture_pipeline_v2.py") != plan["pipeline"]["module_sha256"]:
        raise ValueError("capture pipeline module differs from the frozen pipeline")
    items = {r["identity"]: r for r in records()}
    ordered = [items[identity] for identity in plan["membership"]["branches"]]
    wall = pipeline.run_campaign(OUTPUT, IDENTITY, ordered, plan["limits"], IDENTITY)
    log(f"campaign pass finished in {wall / 60:.1f} min")
    scan()


def _scan_one(result_path):
    result = json.loads(Path(result_path).read_text())
    started = time.process_time()
    outcome = pipeline.branch_outcome(result["segments"][0])
    return result["member_identity"], outcome, time.process_time() - started


def scan():
    """Engine-truth outcome scan of every retained complete shot (smoke + v1 references)."""
    load_plan()
    jobs = {"outcomes.json": sorted(p for p in (OUTPUT / "results").glob("*.json")
                                    if json.loads(p.read_text()).get("complete") is True),
            "reference-outcomes.json": sorted(p for p in (N1 / "results").glob("*.json")
                                              if json.loads(p.read_text()).get("complete") is True)}
    for name, paths in jobs.items():
        began = time.monotonic()
        with ProcessPoolExecutor(SCAN_PROCESSES) as pool:
            rows = list(pool.map(_scan_one, [str(p) for p in paths]))
        value = {"schema": "issue_109_engine_outcomes_v1",
                 "source": "smoke" if name == "outcomes.json" else "issue-77-n1-v1",
                 "outcomes": {identity: outcome for identity, outcome, _ in rows},
                 "scan_cpu_seconds": sum(cpu for _, _, cpu in rows),
                 "scan_wall_seconds": time.monotonic() - began}
        pipeline.write_json(OUTPUT / name, value)
        log(f"{name}: {len(rows)} shots scanned in {value['scan_wall_seconds']:.0f}s wall")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    for mode in ("dry-run", "prepare", "run", "scan", "publish", "validate"):
        modes.add_argument(f"--{mode}", action="store_true")
    args = parser.parse_args()
    try:
        if args.dry_run:
            dry_run()
        elif args.prepare:
            prepare()
        elif args.run:
            run()
        elif args.scan:
            scan()
        elif args.publish:
            from scripts import capture_smoke_report as report
            report.publish()
        else:
            from scripts import capture_smoke_report as report
            report.validate()
    except Exception as error:
        log(f"error: {type(error).__name__}: {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
