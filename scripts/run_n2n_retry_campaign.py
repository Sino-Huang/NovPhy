"""Issue #109 work item 5: the pending #77 n2n-v2 retry campaign on the fixed pipeline.

Membership (outcome-free, frozen before any capture): exactly the 70 branches
that the terminal ``issue-77-n2n-v1`` ledger retained as typed failures
(type010102 normal side; 8 held-out evaluation lineages x 13 actions). Each
is captured once with ``scripts/capture_pipeline_v2.py``; v1's 34 admissible
branches are not re-run. The published combined inventory is v1-admissible
plus v2-complete, per lineage, with the remaining typed failures retained.

Modes: --dry-run / --prepare / --run / --scan / --publish / --validate.
Validation: ``python -u -m scripts.run_n2n_retry_campaign --validate``.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import csv
import io
import json
from pathlib import Path
import shutil
import statistics
import sys
import time

from scripts import capture_pipeline_v2 as pipeline
from scripts.run_capture_reliability_smoke import sha256_file

ROOT = Path(__file__).resolve().parents[1]
IDENTITY = "issue-77-n2n-v2"
OUTPUT = ROOT / ".local-artifacts" / IDENTITY
V1 = ROOT / ".local-artifacts" / "issue-77-n2n-v1"
SMOKE = ROOT / ".local-artifacts" / "issue-109-capture-smoke-v1"
VALIDATION_COMMAND = "python -u -m scripts.run_n2n_retry_campaign --validate"
SCAN_PROCESSES = 12
LIMITS = {
    "workers": 6,
    "attempt_seconds": 1800,
    "worker_cpu_rss_mib": 4096,
    "aggregate_cpu_rss_mib": 32768,
    "minimum_free_bytes": 100 * 2**30,
    "artifact_bytes": 150 * 2**30,
    "decision_fixed_step": 30000,
    **pipeline.DEFAULT_LIMITS,
}


LEGACY_FAILURE_CLASSES = (
    ("paired_view_mismatch", "corrected single differs from the latest corrected history frame"),
    ("shot_manifest_deadline", "native shot manifest deadline"),
)


def v1_failure_class(failure):
    """Old-pipeline failure strings mapped onto named classes (then the v2 taxonomy)."""
    for name, needle in LEGACY_FAILURE_CLASSES:
        if needle in str(failure):
            return name
    return pipeline.failure_class(failure)


def log(message):
    print(f"[{IDENTITY}] {message}", flush=True)


def v1_plan():
    return json.loads((V1 / "plan.json").read_text())


def v1_failed():
    ledger = json.loads((V1 / "ledger.json").read_text())
    if ledger["status"] != "complete":
        raise ValueError("issue-77-n2n-v1 ledger is not terminal")
    return {identity: entry["failure"] for identity, entry in ledger["branches"].items()
            if entry["status"] == "failed"}


def records():
    from scripts import run_issue_77_n2n as n2n
    plan = v1_plan()
    failed = v1_failed()
    members = {member["identity"]: member for member in plan["members"]}
    return [n2n._dispatch_record(members[branch["source_member_identity"]], branch)
            for branch in plan["branches"] if branch["identity"] in failed]


def load_plan():
    plan = json.loads((OUTPUT / "plan.json").read_text())
    for item in plan["inputs"]:
        if sha256_file(ROOT / item["artifact"]) != item["sha256"]:
            raise ValueError(f"frozen input changed: {item['artifact']}")
    return plan


def dry_run():
    items = records()
    kinds = Counter(v1_failure_class(f) for f in v1_failed().values())
    log(f"membership: {len(items)} v1 typed-failure branches; v1 failure classes {dict(kinds)}")
    log("dry run computes no statistic and writes nothing")


def prepare():
    if (OUTPUT / "plan.json").exists():
        raise ValueError("plan.json is frozen; refusing to overwrite")
    smoke = json.loads((SMOKE / "summary.json").read_text())
    gate = smoke["dispositions"]["q_a1_failure_rate_gate"]
    if gate["token"] != "supported":
        raise ValueError("the #109 capture smoke did not clear its typed-failure gate")
    items = records()
    plan = {
        "schema": "issue_77_n2n_v2_plan_v1",
        "identity": IDENTITY,
        "version": 1,
        "role": "terminal",
        "frozen_at": pipeline.utc_now(),
        "frozen_before_any_capture": True,
        "validation_command": VALIDATION_COMMAND,
        "parent_issue": 109,
        "pipeline": {"identity": pipeline.PIPELINE_IDENTITY,
                     "module_sha256": sha256_file(ROOT / "scripts/capture_pipeline_v2.py")},
        "smoke_gate": {"identity": "issue-109-capture-smoke-v1", "token": gate["token"],
                       "typed_failure_rate": gate["typed_failure_rate"]},
        "runner_sha256_at_freeze": sha256_file(Path(__file__)),
        "membership": {"rule": ("exactly the branches retained as typed failures by the terminal "
                                "issue-77-n2n-v1 ledger (status failed); v1 admissible branches are "
                                "not re-run; selection uses v1 failure status only, never an outcome"),
                       "branches": [r["identity"] for r in items]},
        "limits": LIMITS,
        "retry_policy": ("one attempt per branch on the fixed pipeline; technical_retries 0; "
                         "a v2 failure is retained as a typed failure"),
        "inputs": [{"name": name, "artifact": str(path.relative_to(ROOT)), "sha256": sha256_file(path)}
                   for name, path in (("n2n_v1_plan", V1 / "plan.json"),
                                      ("n2n_v1_ledger", V1 / "ledger.json"),
                                      ("n2n_v1_coverage", V1 / "coverage.json"),
                                      ("capture_smoke_summary", SMOKE / "summary.json"))],
        "claim_boundary": ("capture-only retry of the n2n-v1 typed failures on the fixed pipeline; "
                           "no model is fit or scored; the #77 N2 evaluation disposition is not "
                           "reopened; merging the recovered branches into an evaluation is a later "
                           "ticket's frozen decision"),
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    pipeline.write_json(OUTPUT / "plan.json", plan)
    if not (OUTPUT / "player").exists():
        shutil.copytree(V1 / "player", OUTPUT / "player", symlinks=True)
    log(f"plan frozen at {plan['frozen_at']}: {len(items)} branches")


def run():
    plan = load_plan()
    if sha256_file(ROOT / "scripts/capture_pipeline_v2.py") != plan["pipeline"]["module_sha256"]:
        raise ValueError("capture pipeline module differs from the frozen pipeline")
    items = {r["identity"]: r for r in records()}
    wall = pipeline.run_campaign(OUTPUT, IDENTITY, [items[i] for i in plan["membership"]["branches"]],
                                 plan["limits"], IDENTITY)
    log(f"campaign pass finished in {wall / 60:.1f} min")
    scan()


def _scan_one(result_path):
    result = json.loads(Path(result_path).read_text())
    return result["member_identity"], pipeline.branch_outcome(result["segments"][0])


def scan():
    load_plan()
    paths = sorted(str(p) for p in (OUTPUT / "results").glob("*.json")
                   if json.loads(p.read_text()).get("complete") is True)
    began = time.monotonic()
    with ProcessPoolExecutor(SCAN_PROCESSES) as pool:
        rows = list(pool.map(_scan_one, paths))
    pipeline.write_json(OUTPUT / "outcomes.json",
                        {"schema": "issue_109_engine_outcomes_v1", "source": IDENTITY,
                         "outcomes": dict(rows), "scan_wall_seconds": time.monotonic() - began})


def _quantile(values, q):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * (len(ordered) - 1) + .5))] if ordered else None


def compute():
    plan = load_plan()
    v1_ledger = json.loads((V1 / "ledger.json").read_text())["branches"]
    v1_coverage = json.loads((V1 / "coverage.json").read_text())["branches"]
    outcomes = json.loads((OUTPUT / "outcomes.json").read_text())["outcomes"]
    rows = []
    for identity in sorted(v1_ledger):
        member = identity.rsplit("-a", 1)[0]
        if identity in plan["membership"]["branches"]:
            entry = pipeline.branch_entry(OUTPUT, identity)
            source = "v2"
            status = entry["status"]
            failure_class = entry.get("failure_class")
            stop = entry.get("stop_kind")
            wall = entry.get("wall_seconds")
        else:
            source = "v1"
            status = "complete" if v1_coverage[identity]["status"] == "admissible" else "failed"
            failure_class = None
            stop = v1_coverage[identity]["stop_kind"]
            wall = v1_ledger[identity]["wall_seconds"]
        rows.append({"branch": identity, "member": member, "source": source, "status": status,
                     "v1_failure_class": v1_failure_class(v1_ledger[identity]["failure"]),
                     "failure_class": failure_class, "stop_kind": stop,
                     "pig_removed": (outcomes.get(identity) or {}).get("pig_removed") if source == "v2" else None,
                     "wall_seconds": wall})
    v2 = [r for r in rows if r["source"] == "v2"]
    lineages = {}
    for row in rows:
        item = lineages.setdefault(row["member"], {"v1_admissible": 0, "v2_recovered": 0,
                                                   "still_failed": 0, "candidates": 13})
        if row["source"] == "v1" and row["status"] == "complete":
            item["v1_admissible"] += 1
        elif row["source"] == "v2" and row["status"] == "complete":
            item["v2_recovered"] += 1
        else:
            item["still_failed"] += 1
    for item in lineages.values():
        item["combined_admissible"] = item["v1_admissible"] + item["v2_recovered"]
        item["lineage_complete"] = item["combined_admissible"] == item["candidates"]
    run_logs = [json.loads(p.read_text()) for p in sorted((OUTPUT / "run-log").glob("*.json"))]
    stage_names = sorted({name for r in v2 if r["status"] == "complete"
                          for name in json.loads((OUTPUT / "results" / f"{r['branch']}.json").read_text())
                          .get("stage_seconds", {})})
    stages = {}
    for name in stage_names:
        values = [json.loads((OUTPUT / "results" / f"{r['branch']}.json").read_text())["stage_seconds"].get(name, 0.0)
                  for r in v2 if r["status"] == "complete"]
        stages[name] = {"mean": round(statistics.fmean(values), 3), "p50": round(_quantile(values, .5), 3),
                        "p90": round(_quantile(values, .9), 3)}
    recovered = sum(r["status"] == "complete" for r in v2)
    failed = len(v2) - recovered
    summary = {
        "schema": "issue_77_n2n_v2_report_v1",
        "identity": IDENTITY,
        "plan_frozen_at": plan["frozen_at"],
        "validation_command": VALIDATION_COMMAND,
        "scheduled": len(v2), "recovered": recovered, "typed_failures": failed,
        "typed_failure_rate": round(failed / len(v2), 6),
        "v2_failure_classes": dict(Counter(r["failure_class"] for r in v2 if r["status"] != "complete")),
        "v1_failure_classes_of_scheduled": dict(Counter(r["v1_failure_class"] for r in v2)),
        "recovery_by_v1_failure_class": {
            kind: {"scheduled": sum(r["v1_failure_class"] == kind for r in v2),
                   "recovered": sum(r["v1_failure_class"] == kind and r["status"] == "complete" for r in v2)}
            for kind in sorted({r["v1_failure_class"] for r in v2})},
        "combined": {"branches": len(rows), "admissible": sum(r["status"] == "complete" for r in rows),
                     "lineages_complete": sum(item["lineage_complete"] for item in lineages.values()),
                     "lineages": len(lineages)},
        "lineages": lineages,
        "v2_stop_kinds": dict(Counter(r["stop_kind"] for r in v2 if r["status"] == "complete")),
        "v2_pig_removed": sum(bool(r["pig_removed"]) for r in v2 if r["status"] == "complete"),
        "compute": {"campaign_wall_seconds": round(sum(item["wall_seconds"] for item in run_logs), 1),
                    "workers": plan["limits"]["workers"], "gpu_seconds": 0.0,
                    "per_branch_wall_mean": round(statistics.fmean(r["wall_seconds"] for r in v2), 1),
                    "stage_seconds": stages},
        "disposition": {
            "question": "Does the fixed pipeline recover the n2n-v1 contention-era typed failures?",
            "rule": ("supported iff the v2 typed-failure rate over the 70 scheduled branches is below "
                     "the #109 smoke gate 0.05; not_supported_by_this_experiment otherwise"),
            "token": "supported" if failed / len(v2) < 0.05 else "not_supported_by_this_experiment"},
    }
    return summary, rows


def render(summary, rows):
    stream = io.StringIO()
    writer = csv.writer(stream, lineterminator="\n")
    keys = ["branch", "member", "source", "status", "v1_failure_class", "failure_class",
            "stop_kind", "pig_removed"]
    writer.writerow(keys)
    for row in rows:
        writer.writerow([row[key] for key in keys])
    lines = [f"# {IDENTITY}: n2n retry campaign on the fixed capture pipeline (issue #109 item 5)", "",
             f"- plan frozen {summary['plan_frozen_at']} before any capture; validation `{VALIDATION_COMMAND}`",
             f"- scheduled {summary['scheduled']} (every n2n-v1 typed failure), recovered "
             f"{summary['recovered']}, typed failures {summary['typed_failures']} "
             f"(rate {summary['typed_failure_rate']:.4f})",
             f"- disposition: **{summary['disposition']['token']}** ({summary['disposition']['rule']})", "",
             "## Recovery by v1 failure class", "", "| v1 failure class | scheduled | recovered |",
             "|---|---|---|"]
    for kind, item in summary["recovery_by_v1_failure_class"].items():
        lines.append(f"| {kind} | {item['scheduled']} | {item['recovered']} |")
    lines += ["", "## Combined type010102 normal-side inventory (v1 admissible + v2 recovered)", "",
              "| lineage | v1 admissible | v2 recovered | still failed | combined / 13 |", "|---|---|---|---|---|"]
    for member, item in sorted(summary["lineages"].items()):
        lines.append(f"| {member} | {item['v1_admissible']} | {item['v2_recovered']} | "
                     f"{item['still_failed']} | {item['combined_admissible']} |")
    lines += ["", f"- combined admissible {summary['combined']['admissible']}/{summary['combined']['branches']}; "
              f"complete lineages {summary['combined']['lineages_complete']}/{summary['combined']['lineages']}",
              f"- v2 stop kinds {summary['v2_stop_kinds']}; v2 engine-truth pig removals {summary['v2_pig_removed']}",
              f"- v2 remaining failure classes {summary['v2_failure_classes']}", "",
              "## Compute", "",
              f"- campaign wall {summary['compute']['campaign_wall_seconds']} s at "
              f"{summary['compute']['workers']} workers; mean branch wall "
              f"{summary['compute']['per_branch_wall_mean']} s; GPU 0 s",
              "- stage seconds (complete branches): " + json.dumps(summary["compute"]["stage_seconds"], sort_keys=True),
              "", "## Claim boundary", "", load_plan()["claim_boundary"], ""]
    return {"summary.json": json.dumps(summary, indent=2, sort_keys=True) + "\n",
            "branches.csv": stream.getvalue(), "findings.md": "\n".join(lines)}


def publish():
    for name, text in render(*compute()).items():
        (OUTPUT / name).write_text(text)
    log("published summary.json, branches.csv, findings.md")


def validate():
    for name, text in render(*compute()).items():
        if (OUTPUT / name).read_text() != text:
            raise ValueError(f"{name} differs from its recomputation")
    sample = sorted(json.loads((OUTPUT / "outcomes.json").read_text())["outcomes"])[:4]
    saved = json.loads((OUTPUT / "outcomes.json").read_text())["outcomes"]
    for identity in sample:
        _, outcome = _scan_one(OUTPUT / "results" / f"{identity}.json")
        if outcome != saved[identity]:
            raise ValueError(f"engine outcome rescan differs for {identity}")
    log("validation passed: summary.json, branches.csv, findings.md byte-compared; 4 outcome rescans equal")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    for mode in ("dry-run", "prepare", "run", "scan", "publish", "validate"):
        modes.add_argument(f"--{mode}", action="store_true")
    args = parser.parse_args()
    try:
        {"dry_run": dry_run, "prepare": prepare, "run": run, "scan": scan,
         "publish": publish, "validate": validate}[next(k for k, v in vars(args).items() if v)]()
    except Exception as error:
        log(f"error: {type(error).__name__}: {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
