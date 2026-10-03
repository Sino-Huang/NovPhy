"""Issue #104 phase 2 report: capture accounting and mixed-verdict yield of the frozen campaign.

Reads the terminal campaign of ``scripts.run_power_cohort_campaign`` (``complete.json`` or a
finalized ``stop.json``); refuses a running campaign, so no outcome is read before the capture
ends. Nothing is fitted or scored.

* Capture accounting per stream: scheduled, complete, typed failures by class, undispatched
  (stop rule), renderer and #113 window-rule checks per complete branch, wall, bytes,
  throughput, and the GPU accounting samples.
* Yield (work item 2): per stream, level type and candidate inventory, the levels whose
  inventory is fully captured and how many of them are mixed-verdict (engine pig removal
  differs across the inventory's candidates; the #96 ``mixed`` rule). Level type is the
  generator family for #109 and the #110 cell (scenario | novelty level, normal and novel
  sides separately).

Modes: ``--publish`` (write ``summary.json``, ``branches.json``, ``yield.csv``,
``findings.md`` into the campaign directory), ``--validate`` (recompute and compare).
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import csv
from io import StringIO
import json
from pathlib import Path
import statistics
import sys

from scripts import capture_pipeline_v2 as pipeline
from scripts import prepare_power_cohort as cohort
from scripts import run_power_cohort_campaign as campaign
from scripts import run_power_cohort_smoke as smoke
from world_model.training import engine_outcome_reactive_diagnostic as engine

OUTPUT = campaign.OUTPUT
SCHEMA = "issue_104_power_cohort_campaign_report_v1"
VALIDATION_COMMAND = "python -u -m scripts.publish_power_cohort_campaign --validate"
PUBLISHED = ("summary.json", "branches.json", "yield.csv", "findings.md")
ANALYSIS_PROCESSES = 40          # runs after the capture, when the 48 threads are free


def log(message):
    print(f"[{campaign.IDENTITY}:publish] {message}", flush=True)


def terminal_state():
    if (OUTPUT / "complete.json").is_file():
        return {"state": "complete", **campaign.read(OUTPUT / "complete.json")}
    if campaign.STOP.is_file() and "undispatched" in campaign.read(campaign.STOP):
        stop = campaign.read(campaign.STOP)
        return {"state": "stopped", "rule": stop["rule"], "token": stop["token"],
                "evidence": stop["evidence"], "undispatched_branches": stop["undispatched_branches"]}
    raise SystemExit("the campaign is not terminal; no outcome is read before the capture ends")


# ------------------------------------------------------------- per branch

def branch_meta(plan):
    """Ordered per-branch metadata (no level files beyond the frozen dispatch records)."""
    scenario_of = {pair["family"]: pair["scenario"]
                   for pair in cohort.read(cohort.COHORT_110 / "plan.json")["capture_membership"]["pairs"]}
    rows = []
    for stream in plan["membership"]["order"]:
        for record in cohort.campaign_records(plan, streams={stream}):
            if stream.startswith("issue110"):
                side = "novel" if "-novel-" in record["source_member_identity"] else "normal"
                level_type = f"{scenario_of[record['generator_family']]}|novelty_level_{record['novelty_level']}"
            else:
                side, level_type = None, record["generator_family"]
            rows.append({"stream": stream, "identity": record["identity"],
                         "level": record["source_member_identity"], "family": record["generator_family"],
                         "side": side, "level_type": level_type,
                         "inventories": sorted({entry["inventory"] for entry in record["inventory_entries"]})})
    return rows


def analyze(meta, expected_renderer):
    """Capture checks and the engine verdict of one branch.

    One pass over the retained native chunks applies the frozen #85 channel functions that
    ``pipeline.branch_outcome`` applies (event leg ``pig_channel``, lifecycle leg, consistency rule),
    without ``NativeSegmentTrace``'s per-sample re-validation, which the capture already ran.
    """
    output = OUTPUT / meta["stream"]
    entry = pipeline.branch_entry(output, meta["identity"])
    row = {**meta, "status": entry["status"], "failure_class": entry.get("failure_class"),
           "wall_seconds": entry.get("wall_seconds")}
    if entry["status"] != "complete":
        return row
    result = campaign.read(output / "results" / f"{meta['identity']}.json")
    segment = result["segments"][0]
    root = Path(segment["native_root"])
    manifest = campaign.read(root / "native-manifest.json")
    events, destroyed = [], set()
    for chunk in smoke.chunks(root, manifest):
        events += chunk["events"]
        destroyed.update(engine.pig_lifecycle_destroyed(chunk["fixed_step_samples"]))
    channel = engine.pig_channel(events)
    consistent, explanation = smoke.window_check(
        manifest, [(event["fixed_step"], event["event_type"], tuple(event["participants"])) for event in events])
    row.update(renderer_ok=result.get("renderer") == expected_renderer, window_consistent=consistent,
               window_explanation=explanation, stop_kind=pipeline.stop_kind(segment),
               pig_removed=bool(channel["pig_removed"]),
               channel_consistency=engine.channel_consistency(channel["pig_removed"], sorted(destroyed)),
               shot_capture_seconds=result["stage_seconds"].get("shot_capture"),
               cold_start_seconds=(result["stage_seconds"].get("engine_boot_connect", 0)
                                   + result["stage_seconds"].get("menu_to_playing", 0)))
    return row


# ------------------------------------------------------------------ tables

def yield_table(rows):
    """Per (stream, level type, inventory): levels, complete inventories, mixed / all-removed / none-removed."""
    levels = defaultdict(lambda: defaultdict(list))
    for row in rows:
        for inventory in row["inventories"]:
            levels[(row["stream"], row["level_type"], inventory)][row["level"]].append(row)
    table = []
    for (stream, level_type, inventory), members in sorted(levels.items()):
        counts = Counter()
        for branches in members.values():
            counts["levels"] += 1
            if not all(branch["status"] == "complete" for branch in branches):
                counts["incomplete"] += 1
                continue
            verdicts = {branch["pig_removed"] for branch in branches}
            counts["complete"] += 1
            counts["mixed" if verdicts == {True, False} else "all_removed" if verdicts == {True} else "none_removed"] += 1
        table.append({"stream": stream, "level_type": level_type, "inventory": inventory,
                      "candidates": len(next(iter(members.values()))), "levels": counts["levels"],
                      "complete": counts["complete"], "incomplete": counts["incomplete"],
                      "mixed": counts["mixed"], "all_removed": counts["all_removed"],
                      "none_removed": counts["none_removed"],
                      "mixed_share": round(counts["mixed"] / counts["complete"], 4) if counts["complete"] else None})
    return table


def yield_totals(table):
    totals = defaultdict(Counter)
    for row in table:
        totals[(row["stream"], row["inventory"])].update(
            {key: row[key] for key in ("levels", "complete", "incomplete", "mixed", "all_removed", "none_removed")})
    return [{"stream": stream, "inventory": inventory, **dict(counts),
             "mixed_share": round(counts["mixed"] / counts["complete"], 4) if counts["complete"] else None}
            for (stream, inventory), counts in sorted(totals.items())]


def capture_accounting(plan, rows, state):
    streams = {}
    for stream in plan["membership"]["order"]:
        items = [row for row in rows if row["stream"] == stream]
        complete = [row for row in items if row["status"] == "complete"]
        runs = [campaign.read(path) for path in sorted((OUTPUT / stream / "run-log").glob("*.json"))]
        failed = [row for row in items if row["status"] == "failed"]
        walls = [row["wall_seconds"] for row in complete]
        streams[stream] = {
            "scheduled": len(items), "complete": len(complete), "failed": len(failed),
            "undispatched": sum(row["status"] == "unattempted" for row in items),
            "failure_classes": dict(Counter(row["failure_class"] for row in failed)),
            "typed_failure_share": round(len(failed) / max(1, len(complete) + len(failed)), 4),
            "renderer_mismatched": sum(not row["renderer_ok"] for row in complete),
            "window_violations": [row["identity"] for row in complete if not row["window_consistent"]],
            "window_explanations": dict(Counter(row["window_explanation"] for row in complete)),
            "stop_kinds": dict(Counter(row["stop_kind"] for row in complete)),
            "channel_consistency": dict(Counter(row["channel_consistency"] for row in complete)),
            "run_wall_hours": round(sum(run["wall_seconds"] for run in runs) / 3600, 2),
            "runs": len(runs), "interrupted_runs": sum(bool(run["interrupted"]) for run in runs),
            "artifact_bytes": int(sum(run["artifact_bytes_added"] for run in runs)),
            "branch_wall_mean": round(statistics.mean(walls), 1) if walls else None,
            "shot_capture_mean": round(statistics.mean(row["shot_capture_seconds"] for row in complete), 1)
            if complete else None,
            "placement": (campaign.read(campaign.MANIFEST)["placement"] or {}).get("branches", {}).get(stream)}
    total_wall = sum(value["run_wall_hours"] for value in streams.values()) * 3600
    finished = sum(value["complete"] + value["failed"] for value in streams.values())
    return {"state": state, "streams": streams,
            "branches": {key: sum(value[key] for value in streams.values())
                         for key in ("scheduled", "complete", "failed", "undispatched")},
            "campaign_wall_hours": round(total_wall / 3600, 2),
            "amortized_seconds_per_branch": round(total_wall / finished, 2) if finished else None,
            "artifact_tib": round(sum(value["artifact_bytes"] for value in streams.values()) / 2**40, 3),
            "frozen_projection": {"wall_hours": plan["budget"]["projected_wall_hours"],
                                  "artifact_tib": plan["budget"]["projected_artifact_tib"]},
            "smoke_reprojection": campaign.read(campaign.SMOKE_SUMMARY)["throughput"]["campaign_reprojection"]}


def gpu_accounting():
    samples = [json.loads(line) for line in (OUTPUT / "gpu-accounting.jsonl").read_text().splitlines() if line]
    readable = [sample for sample in samples if "gpus" in sample]
    return {"samples": len(samples), "unreadable_samples": len(samples) - len(readable),
            "first": samples[0]["at"] if samples else None, "last": samples[-1]["at"] if samples else None,
            "samples_with_compute_processes": sum(bool(sample["compute_apps"]) for sample in readable),
            "compute_processes_seen": sorted({app for sample in readable for app in sample["compute_apps"]}),
            "max_utilization_percent": max((gpu["utilization_percent"] for sample in readable
                                            for gpu in sample["gpus"]), default=None),
            "max_memory_used_mib": max((gpu["memory_used_mib"] for sample in readable
                                        for gpu in sample["gpus"]), default=None),
            "sampling_seconds": campaign.GPU_SAMPLE_SECONDS}


def build():
    plan = cohort.load_plan()
    state = terminal_state()
    metas = branch_meta(plan)
    expected = plan["limits"]["renderer"]
    with ProcessPoolExecutor(ANALYSIS_PROCESSES) as pool:
        rows = list(pool.map(analyze, metas, [expected] * len(metas), chunksize=16))
    table = yield_table(rows)
    summary = {
        "schema": SCHEMA, "identity": campaign.IDENTITY, "validation_command": VALIDATION_COMMAND,
        "cohort_plan": cohort.relative(cohort.OUTPUT / "plan.json"),
        "cohort_plan_sha256": cohort.file_sha256(cohort.OUTPUT / "plan.json"),
        "campaign_manifest_sha256": cohort.file_sha256(campaign.MANIFEST),
        "capture": capture_accounting(plan, rows, state),
        "gpu": gpu_accounting(),
        "yield": {"rule": "mixed = engine pig-removal verdicts over the inventory's candidates are {True, False} "
                          "(#96 rule); a level counts only if every candidate of the inventory is complete",
                  "by_level_type": table, "by_stream": yield_totals(table),
                  "planned_mixed_members_issue109_evaluation": plan["depth"].get("expected_mixed_members")},
        "claim_boundary": ("capture accounting and DESCRIPTIVE verdict yield of the frozen campaign; nothing is fitted "
                           "or scored; the #96 contrasts and Gate B are later stages"),
    }
    return summary, rows


# ------------------------------------------------------------------ outputs

def yield_csv(summary):
    buffer = StringIO()
    fields = ["stream", "level_type", "inventory", "candidates", "levels", "complete", "incomplete", "mixed",
              "all_removed", "none_removed", "mixed_share"]
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(summary["yield"]["by_level_type"])
    return buffer.getvalue()


def findings(summary):
    capture, gpu = summary["capture"], summary["gpu"]
    lines = ["# Issue #104 phase 2: campaign capture and mixed-verdict yield", "",
             f"State: **{capture['state']['state']}**"
             + (f" ({capture['state']['token']})" if capture["state"]["state"] == "stopped" else ""), "",
             "## Capture accounting", "",
             "| stream | placement | scheduled | complete | typed failures | undispatched | renderer mismatches | "
             "window violations | wall h | TiB |", "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for name, value in capture["streams"].items():
        lines.append(f"| {name} | {value['placement']} | {value['scheduled']} | {value['complete']} | "
                     f"{value['failed']} {value['failure_classes'] or ''} | {value['undispatched']} | "
                     f"{value['renderer_mismatched']} | {len(value['window_violations'])} | "
                     f"{value['run_wall_hours']} | {value['artifact_bytes'] / 2**40:.3f} |")
    lines += ["", f"- Campaign wall {capture['campaign_wall_hours']} h, {capture['amortized_seconds_per_branch']} s/branch "
                  f"amortized, {capture['artifact_tib']} TiB (frozen projection {capture['frozen_projection']}).",
              f"- GPU: {gpu['samples']} samples every {gpu['sampling_seconds']} s from {gpu['first']} to {gpu['last']}; "
              f"samples with a compute process {gpu['samples_with_compute_processes']}; max utilization "
              f"{gpu['max_utilization_percent']} %, max memory {gpu['max_memory_used_mib']} MiB.", "",
              "## Mixed-verdict yield by stream", "", summary["yield"]["rule"] + ".", "",
              "| stream | inventory | levels | complete | mixed | all removed | none removed | mixed share |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for row in summary["yield"]["by_stream"]:
        lines.append(f"| {row['stream']} | {row['inventory']} | {row['levels']} | {row['complete']} | {row['mixed']} | "
                     f"{row['all_removed']} | {row['none_removed']} | {row['mixed_share']} |")
    lines += ["", "## Mixed-verdict yield by level type", "",
              "| stream | level type | inventory | complete / levels | mixed | mixed share |",
              "| --- | --- | --- | --- | --- | --- |"]
    for row in summary["yield"]["by_level_type"]:
        lines.append(f"| {row['stream']} | {row['level_type']} | {row['inventory']} | {row['complete']} / "
                     f"{row['levels']} | {row['mixed']} | {row['mixed_share']} |")
    lines += ["", f"Claim boundary: {summary['claim_boundary']}.", ""]
    return "\n".join(lines)


def outputs(summary, rows):
    return {"summary.json": campaign.json_text(summary), "branches.json": campaign.json_text(rows),
            "yield.csv": yield_csv(summary), "findings.md": findings(summary)}


def publish():
    summary, rows = build()
    for name, text in outputs(summary, rows).items():
        (OUTPUT / name).write_text(text)
    log(f"published: {summary['capture']['branches']}")


def validate():
    summary, rows = build()
    problems = [name for name, text in outputs(summary, rows).items()
                if not (OUTPUT / name).is_file() or (OUTPUT / name).read_text() != text]
    print(campaign.json_text({"problems": problems, "validated": not problems}), end="")
    return not problems


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--publish", action="store_true")
    mode.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    if args.publish:
        publish()
    elif not validate():
        sys.exit(1)


if __name__ == "__main__":
    main()
