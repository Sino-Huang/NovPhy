"""Parallelize only deterministic native data derivation; keep the frozen fitter."""
import argparse
from contextlib import redirect_stdout, redirect_stderr
import multiprocessing
from pathlib import Path
import resource
import time

import torch

from scripts import run_issue_76_automated_refit as execution
from scripts.run_issue_76_compatibility import terminate_worker

fit = execution.fit
WORKERS = 6
RECEIPT = "parallel-preparation-v1.json"
SOURCES = ("scripts/run_issue_76_parallel_prepare.py", "docs/issue-76-parallel-preparation.md")


def prepare_one(root, collection_root, plan_identity, member):
    """Same derivation/failure record as the frozen sequential preparation body."""
    torch.set_num_threads(1)
    started = time.monotonic()
    target = root / "prepared" / (member["identity"] + ".pt")
    if target.exists():
        raise ValueError("parallel dispatcher must not reschedule an existing shard")
    path = fit.collection.c2.result_path(collection_root, member)
    result = fit.files.read(path) if path.exists() else None
    failure, shard = None, None
    log = root / "parallel-logs" / (member["identity"] + ".log")
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as stream, redirect_stdout(stream), redirect_stderr(stream):
        if result:
            try:
                shard = fit.prepare_episode(collection_root, member, result)
            except (ValueError, OSError) as error:
                failure = f"{type(error).__name__}: {error}"
                fit.log(f"preparation failure retained {member['identity']}: {failure}")
        if shard is None:
            shard = {"member_identity": member["identity"], "base_cluster": member["base_cluster"],
                     "exposure_role": member["exposure_role"], "tensors": {}, "references": [],
                     "source_result": result, "segment_ranges": [], "synthetic": False}
        shard["preparation_failure"] = failure
        shard["plan_identity"] = plan_identity
        fit.atomic_torch(target, shard)
    usage = resource.getrusage(resource.RUSAGE_SELF)
    fit.files.write(root / "parallel-costs" / (member["identity"] + ".json"), {
        "member_identity": member["identity"], "derivation_wall_seconds": time.monotonic() - started,
        "worker_process_cpu_seconds": usage.ru_utime + usage.ru_stime,
        "worker_peak_rss_mib": usage.ru_maxrss / 1024,
        "frames": len(shard["references"]), "preparation_failure": failure})


def prepare_receipt(root, plan):
    execution.load_amendment(root, plan)
    budget = fit.files.read(root / "budgets/data-preparation.json")
    if budget.get("running") or budget["stopped"]:
        raise ValueError("cleanly pause sequential preparation before the parallel amendment")
    target = root / RECEIPT
    if target.exists() or (root / "data-index.json").exists():
        raise ValueError("parallel receipt or complete data index already exists")
    completed = [m["identity"] for m in plan["members"]
                 if (root / "prepared" / (m["identity"] + ".pt")).exists()]
    value = dict(schema="issue_76_parallel_preparation_v1", base_plan_identity=plan["identity"],
        execution_amendment=execution.IDENTITY, workers=WORKERS, before_budget=budget,
        completed_shards=completed, source_text={p: (fit.files.ROOT / p).read_text() for p in SOURCES},
        source_derivation_changed=False, labels_or_membership_changed=False,
        fitting_updates_changed=False, fresh_access=False)
    fit.files.write(target, value)
    return value


def load_receipt(root, plan):
    value = fit.files.read(root / RECEIPT)
    if (value["base_plan_identity"] != plan["identity"] or value["workers"] != WORKERS
            or value["source_text"] != {p: (fit.files.ROOT / p).read_text() for p in SOURCES}):
        raise ValueError("parallel preparation source or base plan changed")
    current = fit.files.read(root / "budgets/data-preparation.json")
    if current["active_seconds"] < value["before_budget"]["active_seconds"]:
        raise ValueError("parallel preparation lost prior active cost")
    if any(not (root / "prepared" / (name + ".pt")).exists() for name in value["completed_shards"]):
        raise ValueError("parallel preparation lost an original completed shard")
    return value


def prepare_parallel(root, plan):
    if (root / "data-index.json").exists():
        fit.prepare_data(root, plan)
        return
    collection_root = Path(plan["collection_root"])
    pending = [m for m in plan["members"] if not (root / "prepared" / (m["identity"] + ".pt")).exists()]
    context = multiprocessing.get_context("spawn")
    active = []
    finished = len(plan["members"]) - len(pending)
    started = last_log = time.monotonic()
    with fit.fitting_budget(root, "data-preparation", execution.SECONDS, "cpu") as budget:
        try:
            while pending or active:
                budget.check()
                while pending and len(active) < WORKERS:
                    member = pending.pop(0)
                    process = context.Process(target=prepare_one, args=(root, collection_root, plan["identity"], member))
                    process.start()
                    active.append((member, process))
                time.sleep(.5)
                for member, process in list(active):
                    if process.is_alive():
                        continue
                    process.join()
                    active.remove((member, process))
                    if process.exitcode != 0:
                        budget.value["stopped"] = True
                        raise RuntimeError(f"preparation worker failed for {member['identity']}: exit {process.exitcode}; no automatic retry")
                    cost = fit.files.read(root / "parallel-costs" / (member["identity"] + ".json"))
                    finished += 1
                    fit.log(f"parallel prepared {member['identity']} frames={cost['frames']} "
                            f"worker_cpu={cost['worker_process_cpu_seconds']:.1f}s completed={finished}/{len(plan['members'])}")
                now = time.monotonic()
                if now - last_log >= 5:
                    size = fit.check_disk(root, plan)
                    budget.check()
                    budget.save()
                    fit.log(f"parallel preparation active={len(active)}/{WORKERS} pending={len(pending)} "
                            f"elapsed={now-started:.1f}s budget={budget.value['active_seconds']:.1f}s "
                            f"RSS={budget.value['peak_cpu_rss_mib']:.1f}MiB bytes={size}")
                    last_log = now
        finally:
            # A pause/resource stop cannot leave unaccounted writers behind.
            for _, process in active:
                terminate_worker(process)
            budget.value["parallel_workers"] = WORKERS
            usage = resource.getrusage(resource.RUSAGE_CHILDREN)
            budget.value["parallel_children_cpu_seconds"] = budget.value.get("parallel_children_cpu_seconds", 0.) + usage.ru_utime + usage.ru_stime
            budget.save()
        # Original source validates every retained result and builds all 15 role/family
        # gates in original member order. Workers never create the authoritative index.
        fit._prepare_data(root, plan, collection_root, budget)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    for name in ("prepare-amendment", "dry-run", "prepare-data"):
        modes.add_argument("--" + name, action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    root, plan = fit.ROOT, fit.load_plan()
    amendment = execution.load_amendment(root, plan)
    if args.prepare_amendment:
        value = prepare_receipt(root, plan)
        fit.log(f"parallel preparation frozen: {WORKERS} workers; retain {len(value['completed_shards'])} completed shards")
        return
    load_receipt(root, plan)
    if args.dry_run:
        fit.log("parallel preparation no-write check passed; original derivation and final data gate unchanged")
        return
    with execution.execution_policy(root, amendment):
        prepare_parallel(root, plan)


if __name__ == "__main__":
    main()
