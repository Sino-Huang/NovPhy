"""Bounded replay supervisor; requires a separate completed execution freeze."""
import argparse
import multiprocessing
import os
from pathlib import Path
import time

import torch

from scripts import issue_76_live_episode as live
from scripts import prepare_issue_76_action_replay as metadata
from scripts.issue_76_fixed_replay_policy import FixedReplayPolicy
from scripts.run_issue_76_compatibility import process_rss, terminate_worker

ROOT = metadata.ROOT
files = live.files


def load_plan(root=ROOT):
    plan = files.read(root / "execution-plan.json")
    if (not plan["capture_execution_authorized"] or plan["fresh_access"]
            or not plan["comparability_source_frozen"] or plan["inventory"] != files.read(root / "inventory.json")):
        raise ValueError("complete source-bound replay execution authorization is required")
    sources = plan["source_text"]
    correction_path = root / "observation-lineage-correction.json"
    if correction_path.exists():
        correction = files.read(correction_path)
        original_paths = {"scripts/run_issue_76_action_replay.py", "scripts/validate_issue_76_action_replay.py",
                          "tests/test_issue_76_action_replay_validation.py"}
        if (correction["identity"] != "issue-76-replay-observation-lineage-correction-v1"
                or correction["execution_plan_identity"] != plan["identity"]
                or correction["original_source_text"] != {path: sources[path] for path in original_paths}
                or set(correction["corrected_source_text"]) != original_paths | {"tests/test_issue_76_observation_lineage_binding.py"}
                or correction["captures_repeated"] or correction["comparison_limits_changed"] or correction["assignments_changed"]):
            raise ValueError("observation-lineage correction does not bind the original execution source")
        sources = {**sources, **correction["corrected_source_text"]}
    for path, text in sources.items():
        if (files.ROOT / path).read_text() != text:
            raise ValueError(f"replay execution source changed: {path}")
    return plan


def worker(root, member, limits):
    torch.set_num_threads(1)
    if member["exposure_role"] != "training" or member["maximum_shots"] != 1 or len(member["actions"]) != 1:
        raise ValueError("fixed replay worker requires one assigned training shot")
    live.play_episode(root, member, limits, FixedReplayPolicy(member["actions"][0]))


def artifact_bytes(root):
    """Only this new experiment's artifacts, including retained runtime copies."""
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def stop_reason(limits, budget, attempt_seconds, *, smoke, captures=0, frames=0):
    if budget["active_seconds"] >= limits["collection_wall_seconds"]:
        return "collection_wall_limit"
    if smoke and budget["smoke_active_seconds"] >= limits["smoke_wall_seconds"]:
        return "smoke_wall_limit"
    if budget["peak_cpu_rss_mib"] > limits["cpu_rss_mib"]:
        return "aggregate_memory_limit"
    if budget["artifact_bytes"] > limits["artifact_bytes"]:
        return "artifact_limit"
    if captures > 1 or frames > limits["rgb_frames_per_shot_max"]:
        return "capture_or_frame_limit"
    if attempt_seconds >= limits["attempt_seconds"]:
        return "attempt_wall_limit"
    return None


def failed_result(member, attempt, reason):
    return {"member_identity": member["identity"], "base_cluster": member["base_cluster"],
            "exposure_role": member["exposure_role"], "engine_seed": member["engine_seed"],
            "complete": False, "failure": reason, "gameplay_success": False,
            "gameplay_failure_penalty": 1, "fresh_access": False, "raw_attempt_root": str(attempt),
            "partial_capture_preserved": True, "retry_permitted": False}


def selected_members(plan, root, smoke):
    inventory = plan["inventory"]
    members = inventory["members"]
    if smoke:
        selected = [member for member in members if member["identity"] in inventory["smoke_member_identities"]]
        if [member["identity"] for member in selected] != inventory["smoke_member_identities"]:
            raise ValueError("smoke member ordering differs")
        return selected
    validation = files.read(root / "smoke-validation.json")
    if (not validation["validated"] or validation["execution_plan_identity"] != plan["identity"]
            or validation["member_identities"] != inventory["smoke_member_identities"]):
        raise ValueError("the exact two-replay smoke must pass before remaining captures")
    return members


def run(plan, root=ROOT, *, smoke):
    members = selected_members(plan, root, smoke)
    limits = plan["inventory"]["limits"]
    if any(member["exposure_role"] != "training" for member in members):
        raise ValueError("non-training replay assignment is forbidden")
    budget_path = root / "capture-budget.json"
    budget = files.read(budget_path) if budget_path.exists() else {
        "active_seconds": 0., "smoke_active_seconds": 0., "peak_cpu_rss_mib": 0.,
        "artifact_bytes": artifact_bytes(root), "running": False, "stopped": False}
    if budget["running"]:
        raise ValueError("unclean replay supervisor exit: audit its live handle and unsaved costs before continuing")
    if budget["stopped"]:
        raise ValueError("recorded resource stop is terminal for this execution allowance")
    if not (root / "player/9001.x86_64").is_file():
        raise ValueError("frozen player copy must be prepared before capture")
    initial, initial_smoke, started = budget["active_seconds"], budget["smoke_active_seconds"], time.monotonic()
    budget.update(running=True, supervisor_pid=os.getpid(), active_member=None)
    live._replace_json(budget, budget_path)

    def update():
        elapsed = time.monotonic() - started
        budget["active_seconds"] = initial + elapsed
        budget["smoke_active_seconds"] = initial_smoke + (elapsed if smoke else 0.)
        budget["peak_cpu_rss_mib"] = max(budget["peak_cpu_rss_mib"], process_rss(os.getpid()))

    try:
        for member in members:
            target = root / "results" / (member["identity"] + ".json")
            attempt = root / "attempts" / member["identity"]
            if target.exists():
                if files.read(target)["member_identity"] != member["identity"]:
                    raise ValueError("existing replay result identity differs")
                continue
            if attempt.exists():
                files.write(target, failed_result(member, attempt, "interrupted_attempt_no_retry"))
                continue
            update()
            budget["artifact_bytes"] = artifact_bytes(root)
            reason = stop_reason(limits, budget, 0, smoke=smoke)
            if reason:
                budget.update(stopped=True, stop_reason=reason)
                break
            budget["active_member"] = member["identity"]
            live._replace_json(budget, budget_path)
            process = multiprocessing.get_context("spawn").Process(target=worker,
                args=(root, member, {**limits, "readiness_action": plan["readiness_action"]}))
            beginning = last_log = time.monotonic()
            reason = None
            try:
                process.start()
                while process.is_alive():
                    process.join(.5)
                    update()
                    folders = list((attempt / "aligned").glob("capture-v2:*"))
                    frames = max((sum(1 for _ in folder.glob("frame_*.json")) for folder in folders), default=0)
                    now = time.monotonic()
                    if now - last_log >= 5:
                        budget["artifact_bytes"] = artifact_bytes(root)
                        print(f"replay {member['ordinal']}/65 active={budget['active_seconds']:.1f}s smoke={budget['smoke_active_seconds']:.1f}s RSS={budget['peak_cpu_rss_mib']:.1f}MiB frames={frames}", flush=True)
                        last_log = now
                    reason = stop_reason(limits, budget, now - beginning, smoke=smoke, captures=len(folders), frames=frames)
                    live._replace_json(budget, budget_path)
                    if reason:
                        break
            except BaseException as error:
                reason = f"supervisor_{type(error).__name__}: {error}"
                raise
            finally:
                if process.pid is not None and process.is_alive():
                    terminate_worker(process)
                if not target.exists():
                    files.write(target, failed_result(member, attempt, reason or f"worker_exit_{process.exitcode}"))
                update()
                budget["artifact_bytes"] = artifact_bytes(root)
                final_reason = stop_reason(limits, budget, time.monotonic() - beginning, smoke=smoke)
                files.write(root / "supervision" / (member["identity"] + ".json"), {
                    "member_identity": member["identity"], "execution_plan_identity": plan["identity"],
                    "worker_exitcode": process.exitcode, "wall_seconds": time.monotonic() - beginning,
                    "stop_reason": reason or final_reason, "execution_within_limits": not (reason or final_reason),
                    "raw_result_not_rewritten": True})
                if final_reason in ("collection_wall_limit", "smoke_wall_limit", "aggregate_memory_limit", "artifact_limit"):
                    budget.update(stopped=True, stop_reason=final_reason)
                budget["active_member"] = None
                live._replace_json(budget, budget_path)
            print(f"Recorded {member['identity']}; retained without retries.", flush=True)
            if budget["stopped"]:
                break
    finally:
        update()
        budget["running"] = False
        live._replace_json(budget, budget_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--smoke", action="store_true")
    modes.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.smoke or args.run:
        run(load_plan(), smoke=args.smoke)
    else:
        print("No-write: replay capture requires the completed execution freeze; full collection additionally requires smoke validation.")


if __name__ == "__main__":
    main()
