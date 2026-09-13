"""Separate engine-terminal outcome audit; never rewrite collection results."""
import argparse
import gzip
import json
from pathlib import Path

from scripts import issue_76_expansion as files
from scripts import run_issue_76_development as collection
from scripts.canonical_native_trace import NativeTrace


OUTPUT = files.ROOT / "data/issue-76-native-outcome-audit/report.json"


def terminal_evidence(segment):
    """Bind the terminal manifest to its recorded event without a corpus scan."""
    trace = NativeTrace(segment["native_root"])
    manifest = trace.manifest
    if segment["capture_id"] != manifest["capture_id"]:
        raise ValueError("outcome capture identity differs")
    terminal = manifest["terminal_evidence"]
    if terminal is None:
        return None
    matches = []
    for descriptor in manifest["chunks"]:
        if descriptor["first_fixed_step"] <= terminal["fixed_step"] <= descriptor["last_fixed_step"]:
            with gzip.open(trace.root / descriptor["path"], "rt") as stream:
                chunk = json.load(stream)
            matches.extend(e for e in chunk["events"] if e["event_id"] == terminal["event_id"])
    if (len(matches) != 1 or matches[0]["event_type"] != terminal["reason"]
            or matches[0]["fixed_step"] != terminal["fixed_step"]):
        raise ValueError("outcome terminal lacks its exact engine event")
    return {**terminal, "payload": matches[0]["payload"]}


def gameplay_outcome(result, terminals):
    """Use completed native terminals, not a later interface lifecycle state."""
    if len(terminals) != len(result["segments"]):
        raise ValueError("outcome terminal inventory differs from the episode")
    if result["failure"] is not None or not result["complete"]:
        return {"success": False, "reason": "collection_or_execution_failure", "clear_evidence": None}
    if any(s["summary"]["censored"] for s in result["segments"]):
        return {"success": False, "reason": "native_time_window_limit", "clear_evidence": None}
    clear = next((t for t in terminals if t and t["reason"] == "level_clear"), None)
    return {"success": clear is not None,
            "reason": "level_clear" if clear else "no_observed_level_clear", "clear_evidence": clear}


def report(root, plan):
    entries = []
    for member in plan["members"]:
        result = files.read(collection.c2.result_path(root, member))
        terminals = [terminal_evidence(s) for s in result["segments"]]
        outcome = gameplay_outcome(result, terminals)
        entries.append({"member_identity": member["identity"], "role": member["exposure_role"],
                        "original_recorded_success": result["gameplay_success"],
                        "original_game_state_after": result.get("game_state_after"),
                        "original_failure": result["failure"], "terminals": terminals, **outcome})
    return {"schema": "issue_76_native_outcome_audit_v1", "plan_identity": plan["identity"],
            "source_text": {"scripts/issue_76_native_outcomes.py": Path(__file__).read_text()},
            "assigned": len(entries), "original_recorded_successes": sum(e["original_recorded_success"] for e in entries),
            "native_terminal_successes": sum(e["success"] for e in entries),
            "disagreements": [e["member_identity"] for e in entries if e["success"] != e["original_recorded_success"]],
            "entries": entries, "original_results_rewritten": False,
            "model_comparison_performed": False, "issue_64_authorized": False, "fresh_access": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--publish", action="store_true")
    modes.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    plan = collection.load_plan("development")
    value = report(collection.DEVELOPMENT, plan)
    if args.publish:
        if OUTPUT.exists() and files.read(OUTPUT) != value:
            raise ValueError("do not overwrite a different outcome audit")
        files.write(OUTPUT, value)
    elif files.read(OUTPUT) != value:
        raise ValueError("published outcome audit differs from recorded terminal evidence")
    print(f"native outcome audit: {value['native_terminal_successes']}/{value['assigned']} engine-terminal successes; "
          f"{len(value['disagreements'])} original reporting discrepancies; raw results unchanged", flush=True)


if __name__ == "__main__":
    main()
