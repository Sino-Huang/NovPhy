"""Source-bound eight-worker collection; no fitting or fresh-data access."""
import argparse
from collections import deque
import multiprocessing
import os
from pathlib import Path
import shutil
import time

from scripts import issue_76_event_resources as resources
from scripts import issue_76_event_validation as validation
from scripts import prepare_issue_76_event_development as metadata
from scripts import validate_issue_76_action_replay as previous
from scripts.issue_76_native_outcomes import terminal_evidence
from scripts.run_issue_76_compatibility import process_rss, start_isolated_worker, terminate_worker
from scripts.process_lifecycle import cleanup_actions

files = metadata.files
immutable = metadata.angular.previous.immutable
SOURCES = (*previous.SOURCES, "scripts/run_issue_76_event_development.py",
           "scripts/issue_76_event_resources.py", "scripts/issue_76_event_validation.py",
           "scripts/issue_76_live_episode.py", "scripts/issue_76_censored_episode.py",
           "scripts/issue_76_native_outcomes.py", "tests/test_issue_76_event_supervisor.py",
           "tests/test_issue_76_event_resources.py", "tests/test_issue_76_event_validation.py")


def prepare(root=metadata.ROOT, output=metadata.OUTPUT):
    inventory = files.read(root / "inventory.json")
    for name, source in inventory["source_text"].items():
        if (files.ROOT / name).read_text() != source:
            raise ValueError("frozen event inventory source changed")
    original = files.read(Path(inventory["source_collection_plan_path"]))
    if (metadata.select_sources(original["members"]) != inventory["source_members"]
            or metadata.assignments(inventory["source_members"]) != inventory["assignments"]):
        raise ValueError("event inventory no longer matches its outcome-independent source selection")
    plan = {"identity": "issue-76-event-development-execution-v1", "inventory": inventory,
            "capture_execution_authorized": True, "model_training_authorized": False,
            "new_optimizer_updates": 0, "fresh_access": False,
            "readiness_action": inventory["source_members"][0]["actions"][0],
            "comparability_limits": previous.comparison.LIMITS,
            "worker_ports": [list(resources.worker_ports(slot)) for slot in range(inventory["limits"]["workers"])],
            "source_text": {**inventory["source_text"],
                            **{name: (files.ROOT / name).read_text() for name in SOURCES}}}
    immutable(root / "execution-plan.json", plan)
    immutable(output / "execution-plan.json", plan)
    if not (root / "player").exists():
        started = time.monotonic()
        shutil.copytree(inventory["player_source"], root / "player")
        immutable(root / "player-preparation.json", {"source": inventory["player_source"],
                  "wall_seconds": time.monotonic() - started, "new_player_build": False})
    load_plan(root)
    print("Event execution frozen; player copied; no capture started.", flush=True)


def load_plan(root=metadata.ROOT):
    plan = files.read(root / "execution-plan.json")
    if (plan["identity"] != "issue-76-event-development-execution-v1"
            or plan["inventory"] != files.read(root / "inventory.json")
            or not plan["capture_execution_authorized"] or plan["model_training_authorized"]
            or plan["new_optimizer_updates"] or plan["fresh_access"]):
        raise ValueError("event collection requires its unchanged capture-only execution freeze")
    for name, source in plan["source_text"].items():
        if (files.ROOT / name).read_text() != source:
            raise ValueError(f"event execution source changed: {name}")
    return plan


def selected_assignments(plan, root, smoke):
    rows = plan["inventory"]["assignments"]
    expected = plan["inventory"]["smoke_member_identities"]
    selected = [row for row in rows if row["identity"] in expected]
    if [row["identity"] for row in selected] != expected:
        raise ValueError("smoke membership or ordering differs")
    if smoke:
        return selected
    report = files.read(root / "smoke-validation.json")
    if (not report["validated"] or not report["individual_integrity_passed"]
            or report["execution_plan_identity"] != plan["identity"]
            or report["member_identities"] != expected
            or report["maximum_overlapping_native_workers"] < plan["inventory"]["limits"]["workers"]):
        raise ValueError("the complete isolated smoke must validate before full continuation")
    return rows


def resume_inventory(root, plan, budget):
    if budget["running"] or budget["stopped"]:
        raise ValueError("live, unclean or stopped event budget requires an explicit audit, not a restart")
    completed = set(budget["completed_attempt_bytes"])
    assigned = {row["identity"] for row in plan["inventory"]["assignments"]}
    if not completed.issubset(assigned):
        raise ValueError("completed ledger contains an unassigned capture")
    for folder in ("results", "supervision"):
        if {path.stem for path in (root / folder).glob("*.json")} != completed:
            raise ValueError("recorded attempts and sealed accounting differ; audit interrupted work")
    if {path.name for path in (root / "attempts").glob("*")} != completed:
        raise ValueError("unsealed attempt exists; it must not be implicitly replayed")
    for identity in completed:
        receipt = files.read(root / "supervision" / (identity + ".json"))
        result = files.read(root / "results" / (identity + ".json"))
        if (result["member_identity"] != identity or receipt["member_identity"] != identity
                or receipt["execution_plan_identity"] != plan["identity"]
                or receipt["attempt_artifact_bytes"] != budget["completed_attempt_bytes"][identity]):
            raise ValueError("sealed result/receipt/accounting bindings differ")


def capture_extent(attempt):
    folders = list((attempt / "aligned").glob("capture-v2:*"))
    return len(folders), max((sum(1 for _ in folder.glob("frame_*.json")) for folder in folders), default=0)


def run(plan, root=metadata.ROOT, output=metadata.OUTPUT, *, smoke):
    rows = selected_assignments(plan, root, smoke)
    limits = plan["inventory"]["limits"]
    source = {member["identity"]: member for member in plan["inventory"]["source_members"]}
    budget_path = root / "capture-budget.json"
    budget = files.read(budget_path) if budget_path.exists() else {
        "running": False, "stopped": False, "active_seconds": 0., "smoke_active_seconds": 0.,
        "peak_cpu_rss_mib": 0., "worker_seconds": 0., "completed_attempt_bytes": {},
        "maximum_overlapping_native_workers": 0}
    resume_inventory(root, plan, budget)
    if not smoke and not set(plan["inventory"]["smoke_member_identities"]).issubset(budget["completed_attempt_bytes"]):
        raise ValueError("full continuation is missing the sealed smoke captures")
    if not (root / "player/9001.x86_64").is_file():
        raise ValueError("prepared native player is missing")
    ledger = resources.ArtifactLedger(root, output, completed=budget["completed_attempt_bytes"])
    queue = deque(row for row in rows if row["identity"] not in ledger.completed)
    active = {}
    initial, initial_smoke, started = budget["active_seconds"], budget["smoke_active_seconds"], time.monotonic()
    budget.update(running=True, supervisor_pid=os.getpid(), active_members=[], stage="smoke" if smoke else "collection")
    context = multiprocessing.get_context("spawn")
    cleanup_failures = []

    def update(scan=False):
        elapsed = time.monotonic() - started
        budget["active_seconds"] = initial + elapsed
        budget["smoke_active_seconds"] = initial_smoke + (elapsed if smoke else 0.)
        budget["peak_cpu_rss_mib"] = max(budget["peak_cpu_rss_mib"], process_rss(os.getpid()))
        budget["active_members"] = [job["row"]["identity"] for job in active.values()]
        budget["completed_attempt_bytes"] = ledger.completed
        if scan:
            budget["artifact_bytes"] = ledger.snapshot(budget["active_members"])
        resources.live._replace_json(budget, budget_path)

    def finish(slot, reason=None):
        job = active[slot]
        process, row = job["process"], job["row"]
        if process.pid is not None:
            failures = cleanup_actions(((row["identity"],lambda: terminate_worker(process)),))
            cleanup_failures.extend(failures)
            if failures:
                reason = reason or f"cleanup_{type(failures[0][1]).__name__}: {failures[0][1]}"
                budget.update(stopped=True,stop_reason=reason)
        elapsed = time.monotonic() - job["started"]
        attempt = root / "attempts" / row["identity"]
        captures, frames = capture_extent(attempt)
        reason = reason or resources.attempt_stop(limits, wall_seconds=elapsed,
                   rss_mib=job["peak_rss"], captures=captures, frames=frames)
        target = root / "results" / (row["identity"] + ".json")
        if not target.exists():
            member = metadata.materialize_assignment(source[row["source_member_identity"]], row)
            files.write(target, previous.run.failed_result(member, attempt, reason or f"worker_exit_{process.exitcode}"))
        result = files.read(target)
        # A worker can fail before creating its attempt directory. Retain an
        # explicit empty attempt so clean-resume membership remains exact.
        attempt.mkdir(parents=True, exist_ok=True)
        size = ledger.seal(row["identity"])
        receipt = {"member_identity": row["identity"], "execution_plan_identity": plan["identity"],
                   "worker_slot": slot, "worker_exitcode": process.exitcode,
                   "started_monotonic_seconds": job["started"], "wall_seconds": elapsed,
                   "peak_cpu_rss_mib": job["peak_rss"], "captured_frames": frames,
                   "attempt_artifact_bytes": size, "stop_reason": reason,
                   "execution_within_limits": reason is None and process.exitcode == 0,
                   "raw_result_not_rewritten": True, "capture_retries": 0}
        files.write(root / "supervision" / (row["identity"] + ".json"), receipt)
        budget["worker_seconds"] += elapsed
        if cleanup_failures:
            budget["cleanup_failures"] = [f"{name}: {type(error).__name__}: {error}" for name,error in cleanup_failures]
        del active[slot]
        update(scan=True)
        print(f"Recorded event assignment {len(ledger.completed)}/{len(plan['inventory']['assignments'])}: {row['identity']}", flush=True)
        return receipt["execution_within_limits"] and result["complete"] and result["failure"] is None

    last_log = started
    initially_completed = len(ledger.completed)
    try:
        update(scan=True)
        while queue or active:
            update()
            reason = resources.global_stop(limits, active_seconds=budget["active_seconds"],
                smoke_seconds=budget["smoke_active_seconds"], rss_mib=budget["peak_cpu_rss_mib"],
                artifact_bytes=budget["artifact_bytes"], smoke=smoke)
            if reason:
                budget.update(stopped=True, stop_reason=reason)
                break
            # Start one bounded wave. No slot is reused while another process
            # from that wave is still writing or shutting down.
            if not active:
                for slot in range(limits["workers"]):
                    if not queue:
                        break
                    if shutil.disk_usage(root).free < limits["minimum_free_bytes_before_attempt"]:
                        budget.update(stopped=True, stop_reason="disk_reserve_limit")
                        break
                    resources.check_ports_available(resources.worker_ports(slot))
                    row = queue.popleft()
                    process = start_isolated_worker(context,resources.worker,
                        (root, source[row["source_member_identity"]], row,
                         {**limits, "readiness_action": plan["readiness_action"]}, slot))
                    active[slot] = {"process": process, "row": row, "started": time.monotonic(), "peak_rss": 0.}
                    update()
            if budget["stopped"]:
                break
            overlap = 0
            for slot, job in list(active.items()):
                process = job["process"]
                if not process.is_alive():
                    valid = finish(slot)
                    if smoke and not valid:
                        budget.update(stopped=True, stop_reason="smoke_capture_failure")
                        break
                    continue
                job["peak_rss"] = max(job["peak_rss"], process_rss(process.pid))
                captures, frames = capture_extent(root / "attempts" / job["row"]["identity"])
                overlap += int(frames > 0)
                reason = resources.attempt_stop(limits, wall_seconds=time.monotonic() - job["started"],
                            rss_mib=job["peak_rss"], captures=captures, frames=frames)
                if reason:
                    finish(slot, reason)
                    if smoke:
                        budget.update(stopped=True, stop_reason="smoke_capture_failure")
                        break
            budget["maximum_overlapping_native_workers"] = max(budget["maximum_overlapping_native_workers"], overlap)
            if budget["stopped"]:
                break
            now = time.monotonic()
            if now - last_log >= 5:
                update(scan=True)
                newly_completed = len(ledger.completed) - initially_completed
                eta = ((now - started) * (len(queue) + len(active)) / newly_completed) if newly_completed else None
                print(f"event {budget['stage']} recorded={len(ledger.completed)} active={len(active)} pending={len(queue)} "
                      f"wall={budget['active_seconds']:.1f}s RSS={budget['peak_cpu_rss_mib']:.1f}MiB "
                      f"bytes={budget['artifact_bytes']} ETA_seconds={eta}", flush=True)
                last_log = now
            time.sleep(.5)
    except BaseException as error:
        budget.update(stopped=True, stop_reason=f"supervisor_{type(error).__name__}: {error}")
        raise
    finally:
        for slot in list(active):
            finish(slot, budget.get("stop_reason") or "supervisor_interrupted")
        update(scan=True)
        final_stop = resources.global_stop(limits, active_seconds=budget["active_seconds"],
            smoke_seconds=budget["smoke_active_seconds"], rss_mib=budget["peak_cpu_rss_mib"],
            artifact_bytes=budget["artifact_bytes"], smoke=smoke)
        if final_stop:
            budget.update(stopped=True, stop_reason=final_stop)
        budget.update(running=False, active_members=[],
                      unattempted_member_identities=[row["identity"] for row in rows if row["identity"] not in ledger.completed])
        resources.live._replace_json(budget, budget_path)
        if cleanup_failures: raise cleanup_failures[0][1]
    return budget


def validate_smoke(root=metadata.ROOT, output=metadata.OUTPUT):
    started = time.monotonic()
    plan = load_plan(root)
    rows = selected_assignments(plan, root, True)
    budget = files.read(root / "capture-budget.json")
    resume_inventory(root, plan, budget)
    if set(budget["completed_attempt_bytes"]) != {row["identity"] for row in rows}:
        raise ValueError("smoke validation requires exactly its completed assignment set")
    sources = {source["identity"]: source for source in plan["inventory"]["source_members"]}
    cases, terminals = {}, {}
    for row in rows:
        member = metadata.materialize_assignment(sources[row["source_member_identity"]], row)
        cases[row["identity"]] = validation.read_case(root, plan, member)
        result = files.read(root / "results" / (row["identity"] + ".json"))
        terminals[row["identity"]] = terminal_evidence(result["segments"][0])
        print(f"Validated event smoke {len(cases)}/{len(rows)}: {row['identity']}", flush=True)
        if time.monotonic() - started > plan["inventory"]["limits"]["offline_validation_wall_seconds"]:
            raise ValueError("smoke validation exceeded offline allowance")
    groups = []
    for identity in dict.fromkeys(row["source_member_identity"] for row in rows):
        assigned = [row for row in rows if row["source_member_identity"] == identity]
        reference, = [row for row in assigned if row["original_action_reference"]]
        comparisons = [{"member_identity": row["identity"], **previous.comparison.compare(
            cases[reference["identity"]], cases[row["identity"]], plan["comparability_limits"])} for row in assigned]
        groups.append({"source_member_identity": identity, "comparisons": comparisons,
                       "paired_ranking_admissible": all(row["comparable"] for row in comparisons)})
    overlap = budget["maximum_overlapping_native_workers"]
    report = {"execution_plan_identity": plan["identity"], "member_identities": [row["identity"] for row in rows],
              "validated": overlap >= plan["inventory"]["limits"]["workers"], "individual_integrity_passed": True,
              "maximum_overlapping_native_workers": overlap, "groups": groups, "native_terminals": terminals,
              "decision_image_paths": {key: case["decision_image_path"] for key, case in cases.items()},
              "capture_budget": budget, "validation_wall_seconds": time.monotonic() - started,
              "gameplay_success_used_as_continuation_criterion": False, "paired_ranking_comparability_relaxed": False,
              "model_training_authorized": False, "fresh_access": False, "advancement_authorized": False}
    immutable(root / "smoke-validation.json", report)
    immutable(output / "smoke-validation.json", report)
    print({"event_smoke_validated": report["validated"], "native_worker_overlap": overlap}, flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    for mode in ("prepare", "smoke", "validate-smoke", "run"):
        modes.add_argument("--" + mode, action="store_true")
    args = parser.parse_args()
    if args.prepare:
        prepare()
    elif args.validate_smoke:
        validate_smoke()
    elif args.smoke or args.run:
        run(load_plan(), smoke=args.smoke)
    else:
        print("No-write: prepare execution, then bounded smoke and its validation before full collection.")
