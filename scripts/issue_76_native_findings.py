"""Validate and summarize every finished native capture, including failed prefixes.

Read-only with respect to captures. The report never promotes a timeout to a
gameplay terminal or exposes this diagnostic engine evidence to an agent.
"""
import argparse
from pathlib import Path

from scripts import issue_76_expansion as files
from scripts import run_issue_76_native_continuation as run
from scripts.canonical_native_trace import NativeTrace


def moving_bodies(sample):
    result = []
    for entity in sample["entities"]:
        body = entity["body"]
        if entity["lifecycle"] != "active" or not body or body["body_type"] != "dynamic":
            continue
        if sum(v * v for v in body["velocity"]) > .0001 or abs(body["angular_velocity_degrees_per_second"]) > .01:
            result.append({"entity_id": entity["entity_id"], "velocity": body["velocity"],
                           "angular_velocity_degrees_per_second": body["angular_velocity_degrees_per_second"]})
    return result


def inspect_trace(root):
    trace = NativeTrace(root)
    summary = trace.validate()
    quiet_suffix = 0
    last = None
    for chunk in trace.chunks():
        for sample in chunk["fixed_step_samples"]:
            quiet_suffix = 0 if moving_bodies(sample) else quiet_suffix + 1
            last = sample
    manifest = trace.manifest
    if last["fixed_step"] != summary["last_fixed_step"] or summary["sample_count"] != summary["last_fixed_step"] - summary["first_fixed_step"] + 1:
        raise ValueError("native prefix does not cover its declared last step")
    return {"native_root": str(root), "capture_id": manifest["capture_id"], "shot_id": manifest["shot_id"],
            "summary": summary, "engine_seed": manifest["engine_seed"],
            "terminal_evidence": manifest["terminal_evidence"],
            "last_moving_recorded_bodies": moving_bodies(last),
            "quiet_recorded_body_suffix_samples": quiet_suffix,
            "quiet_recorded_body_suffix_seconds": max(quiet_suffix - 1, 0) * .0004,
            "interpretation": "Recorded-body quietness is diagnostic only: the runtime stability predicate scans all active Rigidbody2D objects, not just recorded causal identities. No missing terminal is inferred."}


def publication():
    plan = run.load_plan()
    entries = []
    for member in plan["members"]:
        result_path = run.c2.result_path(run.OUTPUT, member)
        if not result_path.exists():
            raise ValueError("finish the frozen assignments before publishing final findings")
        result = files.read(result_path)
        roots = list((run.OUTPUT / "attempts" / member["identity"] / "aligned").glob("*/native-manifest.json"))
        captures = sorted((inspect_trace(p.parent) for p in roots), key=lambda c: c["summary"]["first_fixed_step"])
        entries.append({"episode": member["identity"], "episode_result": result,
                        "captures": captures})
        print(f"[native-findings] {len(entries)}/{len(plan['members'])} episodes validated; {len(captures)} captures", flush=True)
    captures = [c for entry in entries for c in entry["captures"]]
    return {"schema": "issue_76_native_findings_v1", "entries": entries,
            "native_captures": len(captures),
            "complete_captures": sum(c["summary"]["complete"] for c in captures),
            "failed_captures": sum(not c["summary"]["complete"] for c in captures),
            "native_samples": sum(c["summary"]["sample_count"] for c in captures),
            "native_rgb_frames": sum(c["summary"]["frame_count"] for c in captures),
            "budget": files.read(run.OUTPUT / "budget.json"),
            "prior_C2_failure_retained": plan["prior_failed_result"],
            "source_text": Path(__file__).read_text(),
            "candidate_eligible": False, "fresh_access_allowed": False,
            "long_collection_ready": False, "long_refit_ready": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--publish", action="store_true")
    modes.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    value = publication()
    target = files.ROOT / "data/issue-76-native-findings/report.json"
    if args.publish:
        if target.exists() and files.read(target) != value:
            raise ValueError("do not overwrite a different published findings report")
        files.write(target, value)
    elif files.read(target) != value:
        raise ValueError("published native findings differ from retained evidence")
    print(f"[native-findings] {'published' if args.publish else 'validated'} {target}", flush=True)


if __name__ == "__main__":
    main()
