"""Issue #104 phase 2: execute the frozen joint #104 + #110 capture campaign.

Membership, dispatch order, limits and stop rules come from
``.local-artifacts/issue-104-power-cohort-v1/plan.json``: 49 560 branches in five streams
(#109 evaluation, #110 evaluation, #110 fit, #109 fit, #109 policy), ordinal-major inside a
stream, captured once each on the frozen player (#113 window), renderer (Mesa 26.1.2 /
LLVM 22.1.6 llvmpipe) and pipeline (``scripts/capture_pipeline_v2.py``), 16 workers, no
retries. Capture renders on the CPU; the runner samples the GPUs to account that they stay
idle. Nothing is fitted or scored here, and no outcome is read while the campaign runs.

Stop rules (plan ``stop_rules``), all cumulative over the campaign:

* S1: once >= 200 branches have finished, a typed-failure share > 0.05 stops the campaign;
* S2: free bytes on the output filesystem < ``limits.minimum_free_bytes`` (pipeline check);
* S3: accounted unique attempt bytes > ``limits.artifact_bytes`` (pipeline check, given the
  remaining allowance per stream);
* S4: campaign wall > ``stop_rules.S4_wall_cap_hours``.

A stop is terminal: ``stop.json`` records the rule and every undispatched branch as
``campaign_stopped:<rule>``; ``--run`` refuses to continue after it.

Placement. On a mergerfs mount with an existing-path create policy every file lands on the
branch that already holds its parent directory, so an unpinned campaign directory would fill
the single branch holding ``.local-artifacts`` (the smoke's attempts all sit on one branch).
Each stream directory is therefore created on one branch before its first capture (largest
projected stream first, onto the branch with the most unreserved free bytes); the player copy
and every attempt of the stream share that branch, so the pipeline's hard links stay valid.

Modes: ``--dry-run`` (bindings, membership digest, per-stream records, placement; writes
nothing), ``--run`` (capture; resumable, no retries), ``--status`` (progress and typed
failures only; no outcome).
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time

from scripts import capture_pipeline_v2 as pipeline
from scripts import prepare_power_cohort as cohort

ROOT = cohort.ROOT
IDENTITY = "issue-104-power-cohort-campaign-v1"
OUTPUT = ROOT / ".local-artifacts" / IDENTITY
MANIFEST = OUTPUT / "campaign-manifest.json"
STOP = OUTPUT / "stop.json"
GPU_LOG = OUTPUT / "gpu-accounting.jsonl"
SMOKE_SUMMARY = ROOT / ".local-artifacts/issue-104-capture-smoke-v1/summary.json"
ASSEMBLY = "9001_Data/Managed/Assembly-CSharp.dll"
S1_MINIMUM_FINISHED = 200          # plan stop_rules.S1_failure_rate
S1_MAXIMUM_SHARE = 0.05
MONITOR_SECONDS = 30
GPU_SAMPLE_SECONDS = 300


def log(message):
    print(f"[{IDENTITY}] {message}", flush=True)


def utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def json_text(value):
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    pipeline.write_json(path, value)


# ---------------------------------------------------------------- bindings

def stream_identities(plan):
    """Frozen branch identities per stream, in dispatch order; re-derived and digest-checked."""
    rows = cohort.branch_rows(cohort.read(cohort.COHORT_109 / "plan.json"),
                              cohort.read(cohort.COHORT_110 / "plan.json"), plan["depth"])
    if cohort.digest(rows) != plan["membership"]["branch_digest"]:
        raise ValueError("re-derived campaign membership differs from the frozen branch digest")
    streams = {name: [] for name in plan["membership"]["order"]}
    for row in rows:
        streams[row["stream"]].append(row["identity"])
    for name, identities in streams.items():
        if len(identities) != plan["membership"]["streams"][name]["branches"]:
            raise ValueError(f"{name}: branch count differs from the frozen membership")
    return streams


def preflight(plan):
    """Frozen bindings the capture depends on; raises on any difference."""
    if cohort.file_sha256(cohort.PIPELINE) != plan["pipeline"]["sha256"]:
        raise ValueError("capture pipeline differs from the frozen pipeline")
    if cohort.player_identity()["assembly_sha256"] != plan["player"]["assembly_sha256"]:
        raise ValueError("campaign player differs from the frozen player")
    if cohort.renderer_block()["library_sha256"] != plan["renderer"]["library_sha256"]:
        raise ValueError("renderer libraries differ from the frozen renderer")
    smoke = read(SMOKE_SUMMARY)
    if smoke["cohort_plan_sha256"] != cohort.file_sha256(cohort.OUTPUT / "plan.json"):
        raise ValueError("the rendered smoke was not run against this plan")
    if smoke["campaign_launch"]["token"] != "supported":
        raise ValueError("the rendered smoke did not support the campaign launch")
    if smoke["cells"]["unsupported"]:
        # The frozen rule would withhold every branch of an unsupported cell; the smoke found none.
        raise ValueError("the smoke reports unsupported #110 cells; the runner has no exclusion path")
    return {"cohort_plan": cohort.relative(cohort.OUTPUT / "plan.json"),
            "cohort_plan_sha256": cohort.file_sha256(cohort.OUTPUT / "plan.json"),
            "pipeline_sha256": plan["pipeline"]["sha256"],
            "runner_sha256": cohort.file_sha256(__file__),
            "player_assembly_sha256": plan["player"]["assembly_sha256"],
            "renderer_library_sha256": plan["renderer"]["library_sha256"],
            "smoke_summary_sha256": cohort.file_sha256(SMOKE_SUMMARY),
            "branch_digest": plan["membership"]["branch_digest"],
            "order": plan["membership"]["order"]}


# --------------------------------------------------------------- placement

def mergerfs_branches(path):
    """(mount point, branch roots) of the mergerfs mount holding ``path``; None elsewhere."""
    mount = Path(path).resolve()
    while not os.path.ismount(mount):
        mount = mount.parent
    try:
        sources = os.getxattr(mount / ".mergerfs", "user.mergerfs.srcmounts").decode()
    except OSError:
        return None
    return mount, [Path(branch) for branch in sources.split(":")]


def plan_placement(plan):
    """Pin each stream directory to one mergerfs branch (largest projected stream first)."""
    found = mergerfs_branches(ROOT)
    if found is None:
        return None
    mount, branches = found
    per_branch = plan["budget"]["artifact_bytes_per_branch"]
    free = {str(branch): shutil.disk_usage(branch).free for branch in branches}
    projected = {name: value["branches"] * per_branch for name, value in plan["membership"]["streams"].items()}
    remaining, streams = dict(free), {}
    for name in sorted(projected, key=lambda key: (-projected[key], key)):
        branch = max(remaining, key=lambda key: (remaining[key], key))
        if remaining[branch] < projected[name]:
            raise RuntimeError(f"no mergerfs branch can hold the projected {name} stream")
        streams[name] = branch
        remaining[branch] -= projected[name]
    return {"mount": str(mount), "relative": str(OUTPUT.relative_to(mount)), "branches": streams,
            "free_bytes_at_placement": free, "projected_stream_bytes": projected,
            "unreserved_bytes_after_placement": remaining}


def pin_stream(placement, stream):
    if placement is None:
        (OUTPUT / stream).mkdir(parents=True, exist_ok=True)
        return
    target = Path(placement["branches"][stream]) / placement["relative"] / stream
    target.mkdir(parents=True, exist_ok=True)
    if not (OUTPUT / stream).is_dir():
        raise RuntimeError(f"{stream}: pinned branch directory is not visible through the mount")


# ----------------------------------------------------------------- monitor

def run_log_totals(plan):
    wall = artifact = 0.0
    for stream in plan["membership"]["order"]:
        for path in (OUTPUT / stream / "run-log").glob("*.json"):
            entry = read(path)
            wall += entry["wall_seconds"]
            artifact += entry["artifact_bytes_added"]
    return wall, int(artifact)


class Monitor(threading.Thread):
    """S1 / S4 over finished branches of every stream, and GPU accounting samples."""

    def __init__(self, plan):
        super().__init__(daemon=True)
        self.plan = plan
        self.order = plan["membership"]["order"]
        self.wall_cap = plan["stop_rules"]["S4_wall_cap_hours"] * 3600
        self.wall_base = run_log_totals(plan)[0]
        self.started = time.monotonic()
        self.seen = {stream: set() for stream in self.order}
        self.counts = Counter()
        self.halt = threading.Event()
        self.last_gpu = 0.0

    def scan(self):
        for stream in self.order:
            receipts = OUTPUT / stream / "receipts"
            if not receipts.is_dir():
                continue
            with os.scandir(receipts) as entries:
                names = [entry.name for entry in entries if entry.name.endswith(".json")]
            for name in names:
                identity = name[:-5]
                if identity in self.seen[stream]:
                    continue
                entry = pipeline.branch_entry(OUTPUT / stream, identity)
                if entry["status"] == "unattempted":
                    continue
                self.seen[stream].add(identity)
                self.counts["finished"] += 1
                if entry["status"] == "failed":
                    self.counts["failed"] += 1
                    self.counts[f"class:{entry['failure_class']}"] += 1

    def rule(self):
        finished, failed = self.counts["finished"], self.counts["failed"]
        if finished >= S1_MINIMUM_FINISHED and failed / finished > S1_MAXIMUM_SHARE:
            return "S1_failure_rate", {"finished": finished, "failed": failed,
                                       "share": round(failed / finished, 4)}
        wall = self.wall_base + time.monotonic() - self.started
        if wall > self.wall_cap:
            return "S4_wall_cap_hours", {"campaign_wall_seconds": round(wall, 1)}
        return None

    def gpu_sample(self):
        sample = {"at": utc_now()}
        try:
            gpus = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used",
                                   "--format=csv,noheader,nounits"], capture_output=True, text=True,
                                  timeout=60, check=True).stdout
            apps = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                                   "--format=csv,noheader,nounits"], capture_output=True, text=True,
                                  timeout=60, check=True).stdout
            sample["gpus"] = [dict(zip(("index", "utilization_percent", "memory_used_mib"),
                                       (int(v) for v in line.split(", "))))
                              for line in gpus.splitlines() if line.strip()]
            sample["compute_apps"] = [line.strip() for line in apps.splitlines() if line.strip()]
        except (OSError, subprocess.SubprocessError, ValueError) as error:
            sample["error"] = f"{type(error).__name__}: {error}"
        with GPU_LOG.open("a") as stream:
            stream.write(json.dumps(sample, sort_keys=True) + "\n")

    def run(self):
        while not self.halt.is_set():
            if time.monotonic() - self.last_gpu >= GPU_SAMPLE_SECONDS:
                self.last_gpu = time.monotonic()
                self.gpu_sample()
            self.scan()
            found = self.rule()
            if found is not None:
                name, evidence = found
                write(STOP, {"rule": name, "evidence": evidence, "triggered_at_utc": utc_now(),
                             "counts": dict(self.counts)})
                log(f"stop rule {name} triggered: {evidence}")
                os.kill(os.getpid(), signal.SIGTERM)
                return
            self.halt.wait(MONITOR_SECONDS)


def finalize_stop(plan):
    """Record every undispatched branch of a stopped campaign as campaign_stopped:<rule>."""
    stop = read(STOP)
    undispatched = {}
    for stream, identities in stream_identities(plan).items():
        missing = [identity for identity in identities
                   if not (OUTPUT / stream / "results" / f"{identity}.json").is_file()]
        if missing:
            undispatched[stream] = missing
    stop.update(token=f"campaign_stopped:{stop['rule']}", finalized_at_utc=utc_now(),
                undispatched=undispatched,
                undispatched_branches=sum(len(value) for value in undispatched.values()))
    write(STOP, stop)
    log(f"campaign stopped by {stop['rule']}: {stop['undispatched_branches']} branches undispatched")


# --------------------------------------------------------------------- run

def on_sigterm(signum, frame):
    raise KeyboardInterrupt("SIGTERM")


def run():
    plan = cohort.load_plan()
    if STOP.is_file():
        if "undispatched" not in read(STOP):
            finalize_stop(plan)
        raise SystemExit("the campaign was stopped by a frozen stop rule; a stop is terminal")
    binding = preflight(plan)
    identities = stream_identities(plan)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if MANIFEST.is_file():
        manifest = read(MANIFEST)
        if {key: manifest[key] for key in binding} != binding:
            raise ValueError("campaign manifest differs from the frozen bindings")
    else:
        manifest = {**binding, "placement": plan_placement(plan), "started_at_utc": utc_now()}
        write(MANIFEST, manifest)
    signal.signal(signal.SIGTERM, on_sigterm)
    monitor = Monitor(plan)
    monitor.scan()
    monitor.start()
    try:
        for stream in plan["membership"]["order"]:
            done = [identity for identity in identities[stream]
                    if (OUTPUT / stream / "results" / f"{identity}.json").is_file()]
            if len(done) == len(identities[stream]):
                continue
            pin_stream(manifest["placement"], stream)
            output = OUTPUT / stream
            if not (output / "player").exists():
                shutil.copytree(cohort.OUTPUT / "player", output / "player", symlinks=True)
            if cohort.file_sha256(output / "player" / ASSEMBLY) != "sha256:" + plan["player"]["assembly_sha256"]:
                raise ValueError(f"{stream}: player copy differs from the frozen player")
            records = list(cohort.campaign_records(plan, streams={stream}))
            if [record["identity"] for record in records] != identities[stream]:
                raise ValueError(f"{stream}: dispatch records differ from the frozen membership")
            used = run_log_totals(plan)[1]
            limits = {**plan["limits"], "artifact_bytes": plan["limits"]["artifact_bytes"] - used}
            log(f"{stream}: {len(records) - len(done)} of {len(records)} branches to capture "
                f"(artifact allowance {limits['artifact_bytes'] / 2**40:.2f} TiB)")
            pipeline.run_campaign(output, IDENTITY, records, limits, f"{IDENTITY}:{stream}")
    except RuntimeError as error:
        rules = {"minimum free storage reached": "S2_storage", "artifact byte cap reached": "S3_artifact_cap"}
        if str(error) not in rules:
            raise
        monitor.halt.set()
        write(STOP, {"rule": rules[str(error)], "evidence": {"pipeline": str(error)},
                     "triggered_at_utc": utc_now(), "counts": dict(monitor.counts)})
        finalize_stop(plan)
        sys.exit(2)
    except KeyboardInterrupt:
        monitor.halt.set()
        if STOP.is_file():
            finalize_stop(plan)
            sys.exit(2)
        log("interrupted without a stop rule; --run resumes (interrupted branches stay typed failures)")
        sys.exit(130)
    monitor.halt.set()
    monitor.join()
    monitor.scan()
    monitor.gpu_sample()
    write(OUTPUT / "complete.json", {"completed_at_utc": utc_now(), "counts": dict(monitor.counts),
                                     "branches": sum(len(value) for value in identities.values())})
    log(f"campaign complete: {dict(monitor.counts)}")


# --------------------------------------------------------- dry run / status

def dry_run():
    plan = cohort.load_plan()
    binding = preflight(plan)
    identities = stream_identities(plan)
    for stream in plan["membership"]["order"]:
        records = [record["identity"] for record in cohort.campaign_records(plan, streams={stream})]
        if records != identities[stream]:
            raise ValueError(f"{stream}: dispatch records differ from the frozen membership")
    placement = read(MANIFEST)["placement"] if MANIFEST.is_file() else plan_placement(plan)
    print(json_text({"binding": binding, "streams": {name: len(value) for name, value in identities.items()},
                     "placement": placement, "writes": "none"}), end="")


def status():
    plan = cohort.load_plan()
    rows, totals = {}, Counter()
    for stream in plan["membership"]["order"]:
        counts = Counter()
        receipts = OUTPUT / stream / "receipts"
        names = [path.stem for path in receipts.glob("*.json")] if receipts.is_dir() else []
        for identity in names:
            entry = pipeline.branch_entry(OUTPUT / stream, identity)
            counts[entry["status"]] += 1
            if entry["status"] == "failed":
                counts[f"class:{entry['failure_class']}"] += 1
        counts["scheduled"] = plan["membership"]["streams"][stream]["branches"]
        rows[stream] = dict(counts)
        totals.update(counts)
    wall, artifact = run_log_totals(plan)
    print(json_text({"streams": rows, "total": dict(totals), "closed_run_wall_hours": round(wall / 3600, 2),
                     "closed_run_artifact_tib": round(artifact / 2**40, 3),
                     "stop": read(STOP)["rule"] if STOP.is_file() else None,
                     "complete": (OUTPUT / "complete.json").is_file()}), end="")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--status", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        dry_run()
    elif args.run:
        run()
    else:
        status()


if __name__ == "__main__":
    main()
