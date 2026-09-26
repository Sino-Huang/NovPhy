"""Issue #109 capture-reliability smoke: publish and validate the frozen report.

Called from ``scripts/run_capture_reliability_smoke.py --publish / --validate``.
Every statistic, margin and verdict rule is read from the frozen
``plan.json['audit']``; nothing here chooses a threshold. ``validate``
recomputes every published file from the retained inputs, re-scans a
deterministic sample of smoke shots against ``outcomes.json`` and fails on any
byte difference.
"""
import csv
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
import html
import io
import json
from pathlib import Path
import re
import resource
import shutil
import time

import numpy as np

from scripts import capture_pipeline_v2 as pipeline
from scripts import run_capture_reliability_smoke as runner
from scripts import run_oracle_timeout_audit as audit90

ROOT = runner.ROOT
OUTPUT = runner.OUTPUT
MEDIA = runner.MEDIA
N1 = runner.N1
EIGHTY_SEVEN = runner.EIGHTY_SEVEN
EIGHTY_NINE = runner.EIGHTY_NINE
R3_SIZING = ROOT / ".local-artifacts" / "issue-76-r3-exploratory" / "d5_v2_sizing.json"
R3_REPORT = "docs/issue-76-r3-narrower-finding-report.md"
R3_WORKERS = 8  # R3 collection ran at 8 workers (docs/issue-76-r3-narrower-finding-report.md)
SUMMARY_SCHEMA = "issue_109_capture_smoke_report_v1"
MANIFEST_SCHEMA = "issue_109_capture_smoke_media_v1"
OUTCOMES = ("pig_removed", "native_clear", "right_censored")
MEMBER_PATTERN = re.compile(r"^(issue-77-n1-\d{3})-a\d{2}$")
PARTNER_WINDOW_SECONDS = 5.0
RESCAN_SAMPLE = 8
PUBLISHED = ("summary.json", "comparisons.csv", "findings.md")
TOKENS = ("supported", "not_supported_by_this_experiment", "readiness_or_precision_insufficient")


# ------------------------------------------------------------------ helpers

def log(message):
    print(f"[{runner.IDENTITY}:report] {message}", flush=True)


def read_json(path):
    return json.loads(Path(path).read_text())


def json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def digest(data):
    return "sha256:" + sha256(data).hexdigest()


def shown(path):
    path = Path(path)
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def member_of(branch):
    match = MEMBER_PATTERN.match(branch)
    if not match:
        raise ValueError(f"branch identity {branch!r} does not name an issue-77 N1 source member")
    return match.group(1)


def outcome_value(outcome, name):
    if name == "right_censored":
        return outcome["stop_kind"] == "right_censored"
    return bool(outcome[name])


def describe(values):
    values = [float(v) for v in values]
    if not values:
        return {"n": 0, "mean": None, "p50": None, "p90": None, "max": None}
    p50, p90 = np.percentile(values, [50, 90])
    return {"n": len(values), "mean": float(np.mean(values)), "p50": float(p50),
            "p90": float(p90), "max": float(max(values))}


def parse_stamp(text):
    return datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp()


def load_plan():
    """runner.load_plan against this module's (patchable) OUTPUT."""
    plan = read_json(OUTPUT / "plan.json")
    for item in plan["inputs"]:
        if runner.sha256_file(ROOT / item["artifact"]) != item["sha256"]:
            raise ValueError(f"frozen input changed: {item['artifact']}")
    return plan


# --------------------------------------------------------------- statistics

def clustered_contrast(rows, bootstrap):
    """Member-clustered percentile bootstrap of mean(target) - mean(reference).

    ``rows``: iterable of (cluster, is_target, value). Clusters are resampled
    with replacement; draws in which either side is empty are dropped and
    counted (the #90 convention).
    """
    rows = list(rows)
    clusters = sorted({cluster for cluster, _, _ in rows})
    index = {cluster: position for position, cluster in enumerate(clusters)}
    sums = {side: np.zeros(len(clusters)) for side in (True, False)}
    counts = {side: np.zeros(len(clusters)) for side in (True, False)}
    for cluster, side, value in rows:
        sums[bool(side)][index[cluster]] += float(value)
        counts[bool(side)][index[cluster]] += 1.0
    result = {"n_target": int(counts[True].sum()), "n_reference": int(counts[False].sum()),
              "members_target": int((counts[True] > 0).sum()),
              "members_reference": int((counts[False] > 0).sum()),
              "draws": int(bootstrap["draws"]), "seed": int(bootstrap["seed"]),
              "label": bootstrap["label"]}
    if result["n_target"] == 0 or result["n_reference"] == 0:
        return {**result, "computable": False, "reason": "a side is empty",
                "target_mean": None, "reference_mean": None, "difference": None,
                "usable_draws": 0, "ci_low": None, "ci_high": None}
    rng = np.random.Generator(np.random.PCG64(int(bootstrap["seed"])))
    picks = rng.integers(0, len(clusters), size=(int(bootstrap["draws"]), len(clusters)))
    drawn = {side: counts[side][picks].sum(axis=1) for side in (True, False)}
    usable = (drawn[True] > 0) & (drawn[False] > 0)
    target_mean = float(sums[True].sum() / counts[True].sum())
    reference_mean = float(sums[False].sum() / counts[False].sum())
    result.update(computable=True, target_mean=target_mean, reference_mean=reference_mean,
                  difference=target_mean - reference_mean, usable_draws=int(usable.sum()))
    if not usable.any():
        return {**result, "computable": False, "reason": "every bootstrap draw dropped a side",
                "ci_low": None, "ci_high": None}
    means = {side: sums[side][picks].sum(axis=1)[usable] / drawn[side][usable] for side in (True, False)}
    low, high = np.quantile(means[True] - means[False], bootstrap["quantiles"])
    result.update(ci_low=float(low), ci_high=float(high))
    return result


def excludes_zero(contrast):
    return contrast["ci_low"] > 0 or contrast["ci_high"] < 0


def correlated(contrast, margin):
    return (contrast["computable"] and abs(contrast["difference"]) >= margin
            and excludes_zero(contrast))


def guards_tripped(contrast, guards):
    if not contrast["computable"]:
        return ["contrast_unavailable"]
    tripped = []
    if contrast["n_target"] < guards["min_target_slots"]:
        tripped.append("min_target_slots")
    if contrast["members_target"] < guards["min_target_members"]:
        tripped.append("min_target_members")
    if contrast["usable_draws"] / contrast["draws"] < guards["min_usable_draws_fraction"]:
        tripped.append("min_usable_draws_fraction")
    return tripped


def failure_rate_token(failed, scheduled, gate):
    rate = failed / scheduled
    return rate, ("supported" if rate < gate else "not_supported_by_this_experiment")


def part_a_verdict(bound, contrasts, audit):
    """``contrasts`` is None when the direct contrast is not computable."""
    if contrasts is not None and any(correlated(c, audit["contrast_margin"]) for c in contrasts.values()):
        return "outcome_correlated"
    if bound <= audit["bias_margin"]:
        return "bias_bounded_below_margin"
    return "indeterminate"


def part_b_verdict(contrasts, audit):
    margin = audit["contrast_margin"]
    if any(c["guards_tripped"] for c in contrasts.values()):
        return "indeterminate"
    if any(correlated(c, margin) for c in contrasts.values()):
        return "outcome_correlated"
    if all(-margin < c["ci_low"] and c["ci_high"] < margin for c in contrasts.values()):
        return "outcome_blind"
    return "indeterminate"


def token_for(verdict, disposition):
    """Frozen disposition text is 'verdict <name> ...'; map the verdict to its token."""
    for token, text in disposition.items():
        words = text.split()
        if words[0] == "verdict" and words[1] == verdict:
            if token not in TOKENS:
                raise ValueError(f"frozen disposition token {token!r} is not a shared-rule token")
            return token
    raise ValueError(f"verdict {verdict!r} has no frozen disposition")


def outcome_contrasts(rows, audit, guards=None):
    """rows: (cluster, is_target, outcome dict). One contrast per frozen outcome."""
    rows = list(rows)
    contrasts = {}
    for name in OUTCOMES:
        contrast = clustered_contrast(((cluster, side, outcome_value(outcome, name))
                                       for cluster, side, outcome in rows), audit["bootstrap"])
        if guards is not None:
            contrast["guards_tripped"] = guards_tripped(contrast, guards)
        contrasts[name] = contrast
    return contrasts


# ------------------------------------------------------------------- inputs

def load_smoke(plan):
    scheduled = list(plan["membership"]["branches"])
    entries = {identity: pipeline.branch_entry(OUTPUT, identity) for identity in scheduled}
    results, receipts, markers = {}, {}, {}
    for identity in scheduled:
        for store, folder in ((results, "results"), (receipts, "receipts"), (markers, "markers")):
            path = OUTPUT / folder / f"{identity}.json"
            if path.is_file():
                store[identity] = read_json(path)
    unattempted = [identity for identity in scheduled if entries[identity]["status"] == "unattempted"]
    if unattempted:
        raise ValueError(f"coverage incomplete: {len(unattempted)} scheduled branches unattempted "
                         f"(first {unattempted[0]}); finish --run before publishing")
    outcomes_file = read_json(OUTPUT / "outcomes.json")
    reference_file = read_json(OUTPUT / "reference-outcomes.json")
    complete = [identity for identity in scheduled if entries[identity]["status"] == "complete"]
    missing = [identity for identity in complete if identity not in outcomes_file["outcomes"]]
    if missing:
        raise ValueError(f"outcomes.json lacks {len(missing)} complete branches (first {missing[0]}); "
                         "rerun --scan")
    run_logs = [read_json(path) for path in sorted((OUTPUT / "run-log").glob("*.json"))]
    return {"scheduled": scheduled, "entries": entries, "results": results, "receipts": receipts,
            "markers": markers, "outcomes": {i: outcomes_file["outcomes"][i] for i in complete},
            "outcomes_file": outcomes_file, "reference_file": reference_file,
            "reference": reference_file["outcomes"], "run_logs": run_logs}


def load_eighty_seven():
    records = {}
    for path in sorted((EIGHTY_SEVEN / "records").glob("oracle--*.json")):
        record = read_json(path)
        if record.get("cell_kind") == "oracle":
            records[record["cell"]["identity"]] = record
    return records


def load_eighty_nine():
    return [read_json(path) for path in sorted((EIGHTY_NINE / "records").glob("oracle--*.json"))]


# ------------------------------------------------------------------ tables

def coverage_table(smoke, audit):
    entries = smoke["entries"]
    failed = [i for i in smoke["scheduled"] if entries[i]["status"] == "failed"]
    rate, token = failure_rate_token(len(failed), len(smoke["scheduled"]), audit["failure_rate_gate"])
    return {"scheduled": len(smoke["scheduled"]),
            "attempted": sum(1 for e in entries.values() if e["status"] != "unattempted"),
            "complete": sum(1 for e in entries.values() if e["status"] == "complete"),
            "failed": len(failed),
            "failure_class_counts": dict(sorted(Counter(entries[i]["failure_class"] for i in failed).items())),
            "failures": [{"identity": i, "failure_class": entries[i]["failure_class"],
                          "failure": entries[i]["failure"]} for i in failed],
            "typed_failure_rate": rate, "gate": audit["failure_rate_gate"], "token": token}


def part_a_table(smoke, audit):
    spec = audit["part_a"]
    entries = smoke["entries"]
    bound = sum(1 for e in entries.values() if e["status"] == "failed") / len(smoke["scheduled"])
    rows = [(member_of(i), entries[i]["status"] == "failed", smoke["reference"][i])
            for i in smoke["scheduled"] if i in smoke["reference"]]
    failed_with_reference = sum(1 for _, side, _ in rows if side)
    computable = failed_with_reference >= 5
    contrasts = outcome_contrasts(rows, audit) if computable else None
    if contrasts is not None and not all(c["computable"] for c in contrasts.values()):
        computable, contrasts = False, None
    verdict = part_a_verdict(bound, contrasts, audit)
    return {"bound_B": bound, "bias_margin": audit["bias_margin"],
            "contrast_margin": audit["contrast_margin"],
            "smoke_failed_with_v1_reference": failed_with_reference,
            "direct_contrast_computable": computable, "direct_contrast": contrasts,
            "verdict": verdict, "token": token_for(verdict, spec["disposition"])}


def smoke_rows(slots, smoke):
    """(member, is_target, smoke outcome) over slots whose smoke branch executed."""
    return [(member_of(branch), side, smoke["outcomes"][branch])
            for branch, side in slots if branch in smoke["outcomes"]]


def part_b_stratum(slots, smoke, audit):
    spec = audit["part_b"]
    rows = smoke_rows(slots, smoke)
    contrasts = outcome_contrasts(rows, audit, audit["precision_guards"])
    verdict = part_b_verdict(contrasts, audit)
    return {"slots_target": sum(1 for _, side in slots if side),
            "slots_reference": sum(1 for _, side in slots if not side),
            "slots_target_without_executed_smoke_branch": sum(
                1 for b, side in slots if side and b not in smoke["outcomes"]),
            "slots_reference_without_executed_smoke_branch": sum(
                1 for b, side in slots if not side and b not in smoke["outcomes"]),
            "contrasts": contrasts, "verdict": verdict,
            "token": token_for(verdict, spec["disposition"])}


def part_b_tables(smoke, records87, audit):
    strata = {identity: audit90.stratum_of(record) for identity, record in records87.items()}
    b1 = [(records87[i]["cell"]["branch_identity"], strata[i] == "timeout")
          for i in sorted(records87) if strata[i] in ("timeout", "executed")]
    ledger = read_json(N1 / "ledger.json")["branches"]
    b2 = [(branch, entry["status"] == "failed") for branch, entry in sorted(ledger.items())
          if entry["status"] in ("failed", "complete")]
    stability = []
    for identity in sorted(records87):
        if strata[identity] != "stability":
            continue
        branch = records87[identity]["cell"]["branch_identity"]
        outcome = smoke["outcomes"].get(branch)
        stability.append({"slot": identity, "branch_identity": branch,
                          "issue87_failure": records87[identity]["failure"],
                          "smoke_status": smoke["entries"].get(branch, {}).get("status", "unscheduled"),
                          **{name: (None if outcome is None else outcome_value(outcome, name))
                             for name in OUTCOMES}})
    executed = [row for row in stability if row["smoke_status"] == "complete"]
    return {
        "b1_issue87": {"strata_counts": dict(sorted(Counter(strata.values()).items())),
                       **part_b_stratum(b1, smoke, audit)},
        "b2_issue77_n1": part_b_stratum(b2, smoke, audit),
        "b1_stability_stratum": {
            "slots": len(stability), "executed_on_smoke": len(executed),
            "interval": "interval-ineligible (separate typed stratum, never pooled)",
            "smoke_means": {name: (None if not executed else
                                   sum(row[name] for row in executed) / len(executed))
                            for name in OUTCOMES},
            "rows": stability},
    }


def _ports(values):
    return {int(values[key]) for key in ("agent", "game", "physics") if values and key in values}


def _address_in_use(root):
    return pipeline.engine_bind_conflict(Path(root) / "runtime") is not None


def _partners(cells):
    """cells: {identity: (stamp, ports)} -> identities with a same-port partner within the window."""
    found = set()
    ordered = sorted(cells.items(), key=lambda item: (item[1][0], item[0]))
    for position, (identity, (stamp, ports)) in enumerate(ordered):
        for other, (other_stamp, other_ports) in ordered[position + 1:]:
            if other_stamp - stamp > PARTNER_WINDOW_SECONDS:
                break
            if ports & other_ports:
                found.update((identity, other))
    return found


def mechanism_inventory(smoke, records87):
    """Descriptive only; never a verdict input."""
    cells87 = {}
    for identity, record in records87.items():
        marker = EIGHTY_SEVEN / "markers" / f"{identity}.json"
        runtime = EIGHTY_SEVEN / "attempts" / identity / "runtime.json"
        ports = read_json(runtime).get("ports") if runtime.is_file() else record.get("ports")
        if marker.is_file():
            cells87[identity] = (parse_stamp(read_json(marker)["dispatched_at_utc"]), _ports(ports))
    partners87 = _partners(cells87)
    by_stratum = {}
    for identity in sorted(records87):
        record = records87[identity]
        stratum = audit90.stratum_of(record)
        key = stratum if stratum != "timeout" else "timeout: " + record["failure"]
        row = by_stratum.setdefault(key, Counter())
        in_use = _address_in_use(EIGHTY_SEVEN / "attempts" / identity)
        partner = identity in partners87
        row["slots"] += 1
        row["address_already_in_use"] += in_use
        row["same_port_partner_within_5s"] += partner
        row["both"] += in_use and partner
    cells = {}
    for identity, marker in smoke["markers"].items():
        runtime = OUTPUT / "attempts" / identity / "runtime.json"
        ports = (smoke["results"].get(identity) or {}).get("ports")
        if ports is None and runtime.is_file():
            ports = read_json(runtime).get("ports")
        cells[identity] = (parse_stamp(marker["dispatched_at_utc"]), _ports(ports))
    partners = _partners(cells)
    attempts = [i for i in smoke["scheduled"] if (OUTPUT / "attempts" / i).is_dir()]
    return {
        "label": "DESCRIPTIVE mechanism inventory; never a verdict input",
        "partner_window_seconds": PARTNER_WINDOW_SECONDS,
        "issue87_by_stratum": {key: dict(value) for key, value in sorted(by_stratum.items())},
        "smoke": {"attempts_with_runtime": len(attempts),
                  "address_already_in_use": sum(_address_in_use(OUTPUT / "attempts" / i) for i in attempts),
                  "same_port_partner_within_5s": len(partners),
                  "dispatch_markers": len(cells)},
    }


def agreement(pairs):
    pairs = list(pairs)
    agree = sum(1 for left, right in pairs if left == right)
    return {"count": len(pairs), "agree": agree,
            "share": None if not pairs else agree / len(pairs),
            "confusion": dict(sorted(Counter(f"{fmt(left)}|{fmt(right)}" for left, right in pairs).items()))}


def determinism_table(smoke, records87, records89):
    ledger = read_json(N1 / "ledger.json")["branches"]
    coverage = read_json(N1 / "coverage.json")["branches"]
    stop = [(coverage[b]["stop_kind"], smoke["outcomes"][b]["stop_kind"]) for b in smoke["scheduled"]
            if b in smoke["outcomes"] and ledger.get(b, {}).get("status") == "complete"
            and coverage.get(b, {}).get("stop_kind") is not None]

    def pig_pairs(records):
        rows = [(r["cell"]["branch_identity"], bool(r["outcome"]["first_shot_success"]))
                for r in records if r.get("failure") is None and r.get("outcome") is not None]
        rows = [(b, value) for b, value in rows if b in smoke["outcomes"]]
        return rows, [(value, smoke["outcomes"][b]["pig_removed"]) for b, value in rows]
    rows87, pairs87 = pig_pairs(records87[i] for i in sorted(records87)
                                if audit90.stratum_of(records87[i]) == "executed")
    rows89, pairs89 = pig_pairs(records89)
    return {"stop_kind_vs_issue77_n1_v1": {**agreement(stop), "confusion_key": "v1|smoke"},
            "pig_removed_vs_issue87_executed_slots": {
                **agreement(pairs87), "unique_branches": len({b for b, _ in rows87}),
                "confusion_key": "issue87|smoke"},
            "pig_removed_vs_issue89_completion_slots": {
                **agreement(pairs89), "unique_branches": len({b for b, _ in rows89}),
                "confusion_key": "issue89|smoke"}}


def render_table(smoke):
    rows = []
    for identity in smoke["scheduled"]:
        result = smoke["results"].get(identity) or {}
        if smoke["entries"][identity]["status"] != "complete" or "endpoint_render_invariance" not in result:
            continue
        evidence = result["endpoint_render_invariance"]
        rows.append((evidence["before"], evidence["after"], result.get("shot_initial_render_invariance")))
    byte_equal = sum(1 for before, after, _ in rows if before["byte_equal"] and after["byte_equal"])
    animation = [max(before["differing_pixels"], after["differing_pixels"]) for before, after, _ in rows
                 if not (before["byte_equal"] and after["byte_equal"])
                 and before["outside_pixels"] == 0 and after["outside_pixels"] == 0]
    shot = [initial for _, _, initial in rows if initial is not None]
    return {"complete_with_evidence": len(rows),
            "endpoint_byte_equal": byte_equal,
            "endpoint_animation_only_difference": len(animation),
            "animation_only_differing_pixels": describe(animation),
            "shot_initial_byte_equal": sum(1 for s in shot if s["byte_equal"]),
            "shot_initial_animation_only_difference": sum(
                1 for s in shot if not s["byte_equal"] and s["outside_pixels"] == 0),
            "render_invariance_violation_failures": sum(
                1 for e in smoke["entries"].values() if e["failure_class"] == "render_invariance_violation")}


def artifact_bytes():
    seen = set()
    pipeline._attempt_bytes(OUTPUT / "player", seen)
    return sum(pipeline._attempt_bytes(path, seen)
               for path in sorted((OUTPUT / "attempts").iterdir()) if path.is_dir())


def throughput_table(smoke, plan, total_bytes):
    complete = [i for i in smoke["scheduled"] if smoke["entries"][i]["status"] == "complete"]
    stages = sorted({stage for i in complete for stage in smoke["results"][i].get("stage_seconds", {})})
    stage_seconds = {}
    for stage in stages:
        stats = describe([smoke["results"][i]["stage_seconds"][stage] for i in complete
                          if stage in smoke["results"][i].get("stage_seconds", {})])
        stage_seconds[stage] = {key: stats[key] for key in ("n", "mean", "p50", "p90")}
    attempted = [i for i in smoke["scheduled"] if smoke["entries"][i]["status"] != "unattempted"]
    walls = describe([smoke["receipts"][i]["wall_seconds"] for i in attempted])
    campaign = sum(float(entry["wall_seconds"]) for entry in smoke["run_logs"])
    workers = int(plan["limits"]["workers"])
    per_hour = (lambda n: None if campaign <= 0 else n / (campaign / 3600.0))
    smoke_scan, reference_scan = smoke["outcomes_file"], smoke["reference_file"]
    r3 = read_json(R3_SIZING)
    r3_totals, r3_cost = r3["corpus_totals"], r3["per_attempt_cost_distribution_seconds"]["overall"]
    n1_ledger = read_json(N1 / "ledger.json")
    n1_walls = describe([b["wall_seconds"] for b in n1_ledger["branches"].values()
                         if b.get("wall_seconds") is not None])
    n1_workers = int(read_json(N1 / "campaign.json")["limits"]["workers"])
    amortized = None if not attempted else campaign / len(attempted)
    value = {
        "workers": workers, "attempted": len(attempted), "complete": len(complete),
        "campaign_wall_seconds": campaign, "run_logs": len(smoke["run_logs"]),
        "amortized_seconds_per_branch": amortized,
        "worker_seconds_per_branch": None if amortized is None else amortized * workers,
        "branches_per_hour": per_hour(len(attempted)),
        "complete_branches_per_hour": per_hour(len(complete)),
        "per_branch_wall_mean": walls["mean"], "per_branch_wall_p50": walls["p50"],
        "per_branch_wall_p90": walls["p90"], "per_branch_wall_max": walls["max"],
        "stage_seconds": stage_seconds,
        "scan": {"smoke_cpu_seconds": smoke_scan["scan_cpu_seconds"],
                 "smoke_wall_seconds": smoke_scan["scan_wall_seconds"],
                 "smoke_shots": len(smoke_scan["outcomes"]),
                 "reference_cpu_seconds": reference_scan["scan_cpu_seconds"],
                 "reference_wall_seconds": reference_scan["scan_wall_seconds"],
                 "reference_shots": len(reference_scan["outcomes"])},
        "artifact_bytes_total": total_bytes,
        "artifact_bytes_per_branch": None if not attempted else total_bytes / len(attempted),
        "references": {
            "r3": {"source": [shown(R3_SIZING), R3_REPORT], "workers": R3_WORKERS,
                   "per_attempt_wall_mean": r3_cost["mean"], "per_attempt_wall_p50": r3_cost["p50"],
                   "per_attempt_wall_p90": r3_cost["p90"],
                   "attempted": r3_totals["attempted_branches"],
                   "collection_wall_seconds": r3_totals["collection_wall_seconds_elapsed"],
                   "amortized_seconds_per_branch": (r3_totals["collection_wall_seconds_elapsed"]
                                                    / r3_totals["attempted_branches"])},
            "issue77_n1_v1": {"source": shown(N1 / "ledger.json"), "workers": n1_workers,
                              "per_branch_wall_mean": n1_walls["mean"],
                              "per_branch_wall_p50": n1_walls["p50"],
                              "per_branch_wall_p90": n1_walls["p90"],
                              "attempted": n1_ledger["counts"]["attempted"],
                              "collection_wall_seconds": n1_ledger["collection_wall_seconds_elapsed"],
                              "amortized_seconds_per_branch": (n1_ledger["collection_wall_seconds_elapsed"]
                                                               / n1_ledger["counts"]["attempted"])}},
    }
    for name, reference in value["references"].items():
        mean_key = "per_attempt_wall_mean" if name == "r3" else "per_branch_wall_mean"
        reference["smoke_per_branch_wall_ratio"] = (None if walls["mean"] is None
                                                    else walls["mean"] / reference[mean_key])
        reference["smoke_amortized_ratio"] = (None if amortized is None
                                              else amortized / reference["amortized_seconds_per_branch"])
    return value


# -------------------------------------------------------------------- media

def media_sources(identity):
    """(decision frame, final frame) source paths with their provenance, or None."""
    root = OUTPUT / "attempts" / identity
    decision = None
    for history in sorted((root / "aligned").glob("decision-history-*")):
        frames = sorted(history.glob("frame_*.png"))
        preferred = history / "frame_000003.png"
        if preferred.is_file():
            decision = (preferred, "decision-history frame_000003 (sealed decision frame)")
        elif frames:
            decision = (frames[-1], f"decision-history {frames[-1].name} (partial history)")
        if decision:
            break
    final = None
    manifest = root / "shot-1" / "observation-trace" / "observation_trace_manifest.json"
    if manifest.is_file():
        record = read_json(manifest)["frame_records"][-1]["canonical_observation"]
        final = (manifest.parent / record["relative_path"], "last shot observation-trace frame")
    else:
        for capture in sorted((root / "aligned").glob("capture-v2*")):
            frames = sorted(capture.glob("frame_*.png"))
            if frames:
                final = (frames[-1], f"last partial native frame {frames[-1].name} (no observation trace)")
    return {"decision": decision, "final": final}


def media_manifest(smoke):
    branches, copies = [], []
    for identity in smoke["scheduled"]:
        entry = smoke["entries"][identity]
        outcome = smoke["outcomes"].get(identity)
        frames = {}
        for name, source in media_sources(identity).items():
            if source is None:
                continue
            path, provenance = source
            relative = f"{identity}/{name}.png"
            frames[name] = {"path": relative, "sha256": digest(path.read_bytes()),
                            "source": shown(path), "provenance": provenance}
            copies.append((path, relative))
        branches.append({"identity": identity, "member": member_of(identity), "status": entry["status"],
                         "failure": entry["failure"], "failure_class": entry["failure_class"],
                         "stop_kind": None if outcome is None else outcome["stop_kind"],
                         "pig_removed": None if outcome is None else outcome["pig_removed"],
                         "native_clear": None if outcome is None else outcome["native_clear"],
                         "frames": frames})
    manifest = {"schema": MANIFEST_SCHEMA, "identity": runner.IDENTITY,
                "label": "EXPLORATORY media; rendered frames only, no statistic",
                "frames": {"decision": "attempts/<id>/aligned/decision-history-*/frame_000003.png",
                           "final": "last frame of attempts/<id>/shot-1/observation-trace when present"},
                "branches": branches}
    return manifest, copies


def gallery_html(manifest):
    rows = []
    for branch in manifest["branches"]:
        cells = "".join(
            f'<td><img src="{html.escape(branch["frames"][name]["path"])}" width="320" '
            f'title="{html.escape(branch["frames"][name]["provenance"])}"></td>'
            if name in branch["frames"] else "<td>&mdash;</td>" for name in ("decision", "final"))
        status = branch["status"] if branch["status"] == "complete" else (
            f'{branch["status"]}: {branch["failure_class"]}<br><small>{html.escape(str(branch["failure"]))}</small>')
        rows.append(f'<tr><td>{html.escape(branch["identity"])}</td><td>{status}</td>'
                    f'<td>{branch["stop_kind"]}</td><td>{branch["pig_removed"]}</td>{cells}</tr>')
    return ("<!doctype html>\n<html><head><meta charset=\"utf-8\"><title>issue-109 capture smoke</title>"
            "<style>body{font-family:sans-serif}td{vertical-align:top;padding:4px;border-bottom:1px solid #ccc}"
            "</style></head><body>\n<h1>Issue #109 capture smoke: decision and final frames</h1>\n"
            "<table><tr><th>branch</th><th>status</th><th>stop kind</th><th>pig removed</th>"
            "<th>decision frame</th><th>final frame</th></tr>\n" + "\n".join(rows)
            + "\n</table></body></html>\n")


# ------------------------------------------------------------------ report

def build(plan):
    audit = plan["audit"]
    smoke = load_smoke(plan)
    records87 = load_eighty_seven()
    records89 = load_eighty_nine()
    coverage = coverage_table(smoke, audit)
    part_a = part_a_table(smoke, audit)
    part_b = part_b_tables(smoke, records87, audit)
    total_bytes = artifact_bytes()
    throughput = throughput_table(smoke, plan, total_bytes)
    manifest, copies = media_manifest(smoke)
    manifest_bytes = json_bytes(manifest)
    media_bytes = sum(path.stat().st_size for path, _ in copies)
    contrast_count = 3 * (int(part_a["direct_contrast_computable"]) + 2)
    compute = {"engine_worker_seconds": sum(float(smoke["receipts"][i]["wall_seconds"])
                                            for i in smoke["receipts"]),
               "campaign_wall_seconds": throughput["campaign_wall_seconds"],
               "workers": throughput["workers"],
               "outcome_scan_cpu_seconds": throughput["scan"]["smoke_cpu_seconds"]
               + throughput["scan"]["reference_cpu_seconds"],
               "outcome_scan_wall_seconds": throughput["scan"]["smoke_wall_seconds"]
               + throughput["scan"]["reference_wall_seconds"],
               "bootstrap_contrasts": contrast_count,
               "bootstrap_draws_total": contrast_count * int(audit["bootstrap"]["draws"]),
               "gpu": "no model trained or scored; GPU used only by the engine renderer",
               "artifact_bytes": total_bytes, "media_bytes": media_bytes,
               "publish_and_validate_cost": "compute.json (not part of the byte-compared report)"}
    dispositions = {
        "q_a1_failure_rate_gate": {"question": "typed failure rate < frozen gate on the rendered smoke",
                                   "token": coverage["token"],
                                   "typed_failure_rate": coverage["typed_failure_rate"],
                                   "gate": coverage["gate"]},
        "q_a2_part_a_bias_bound": {"question": audit["part_a"]["question"],
                                   "verdict": part_a["verdict"], "token": part_a["token"]},
        "q_b1_issue87_timeouts": {"question": audit["part_b"]["question"],
                                  "verdict": part_b["b1_issue87"]["verdict"],
                                  "token": part_b["b1_issue87"]["token"]},
        "q_b2_issue77_n1_failures": {"question": audit["part_b"]["question"],
                                     "verdict": part_b["b2_issue77_n1"]["verdict"],
                                     "token": part_b["b2_issue77_n1"]["token"]},
    }
    summary = {
        "schema": SUMMARY_SCHEMA, "identity": runner.IDENTITY, "plan_frozen_at": plan["frozen_at"],
        "pipeline": plan["pipeline"]["identity"], "validation_command": plan["validation_command"],
        "interval_label": audit["bootstrap"]["label"],
        "inputs": {name: digest((OUTPUT / name).read_bytes())
                   for name in ("plan.json", "outcomes.json", "reference-outcomes.json")},
        "dispositions": dispositions, "coverage": coverage, "part_a": part_a, "part_b": part_b,
        "mechanism_inventory": mechanism_inventory(smoke, records87),
        "determinism": determinism_table(smoke, records87, records89),
        "render_invariance": render_table(smoke), "throughput": throughput, "compute": compute,
        "media": {"manifest": shown(MEDIA / "manifest.json"), "manifest_sha256": digest(manifest_bytes),
                  "branches": len(manifest["branches"]), "frames": len(copies)},
        "claim_boundary": plan["claim_boundary"],
    }
    files = {"summary.json": json_bytes(summary), "comparisons.csv": comparisons_csv(summary),
             "findings.md": findings_md(summary, plan).encode()}
    return {"summary": summary, "files": files, "manifest": manifest, "manifest_bytes": manifest_bytes,
            "copies": copies, "smoke": smoke}


def flatten(value, prefix=""):
    if isinstance(value, dict):
        for key in value:
            yield from flatten(value[key], f"{prefix}.{key}" if prefix else str(key))
    elif isinstance(value, list):
        for position, item in enumerate(value):
            yield from flatten(item, f"{prefix}[{position}]")
    else:
        yield prefix, value


def comparisons_csv(summary):
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["section", "key", "value"])
    for section in ("dispositions", "coverage", "part_a", "part_b", "mechanism_inventory",
                    "determinism", "render_invariance", "throughput", "compute"):
        for key, value in flatten(summary[section]):
            writer.writerow([section, key, "" if value is None else value if isinstance(value, str)
                             else json.dumps(value) if isinstance(value, bool) else repr(value)])
    return buffer.getvalue().encode()


def fmt(value, digits=3):
    if value is None:
        return "—"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def contrast_lines(contrasts):
    lines = ["| outcome | n target | n ref | members target | target mean | ref mean | difference "
             "| DESCRIPTIVE 95% interval | usable draws | guards tripped |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for name, c in contrasts.items():
        interval = "—" if c["ci_low"] is None else f"[{fmt(c['ci_low'])}, {fmt(c['ci_high'])}]"
        guards = ", ".join(c.get("guards_tripped", [])) or "none"
        lines.append(f"| {name} | {c['n_target']} | {c['n_reference']} | {c['members_target']} | "
                     f"{fmt(c['target_mean'])} | {fmt(c['reference_mean'])} | {fmt(c['difference'])} | "
                     f"{interval} | {c['usable_draws']}/{c['draws']} | {guards} |")
    return lines


def agreement_line(name, value):
    return (f"| {name} | {value['count']} | {value['agree']} | {fmt(value['share'])} | "
            + ("; ".join(f"{k.replace('|', ' → ')}: {v}" for k, v in value["confusion"].items()) or "—")
            + " |")


def findings_md(summary, plan):
    s = summary
    audit = plan["audit"]
    cov, a, b, t = s["coverage"], s["part_a"], s["part_b"], s["throughput"]
    boot = audit["bootstrap"]
    out = [f"# Issue #109 capture-reliability smoke ({s['identity']})", "",
           f"Plan frozen {s['plan_frozen_at']} before any smoke outcome; pipeline `{s['pipeline']}`. "
           f"Intervals are {boot['label']}: member-clustered percentile bootstrap, {boot['draws']} draws, "
           f"PCG64 seed {boot['seed']}, quantiles {boot['quantiles']}.", "",
           "## Disposition table", "", "| question | verdict | token |", "|---|---|---|"]
    d = s["dispositions"]
    q = d["q_a1_failure_rate_gate"]
    out.append(f"| Q_A1: typed failure rate < gate (rate {fmt(q['typed_failure_rate'], 4)}, gate {q['gate']}) "
               f"| — | `{q['token']}` |")
    for key, label in (("q_a2_part_a_bias_bound", "Part A: bias bound on the fixed pipeline"),
                       ("q_b1_issue87_timeouts", "Part B1: #87 timeouts outcome-correlated"),
                       ("q_b2_issue77_n1_failures", "Part B2: #77 N1 v1 failures outcome-correlated")):
        out.append(f"| {label} | {d[key]['verdict']} | `{d[key]['token']}` |")
    out += ["", "## Coverage & failure taxonomy", "",
            f"Scheduled {cov['scheduled']}, attempted {cov['attempted']}, complete {cov['complete']}, "
            f"failed {cov['failed']} (typed failure rate {fmt(cov['typed_failure_rate'], 4)}; gate "
            f"{cov['gate']}; supported iff rate < gate).", "", "| failure class | count |", "|---|---|"]
    out += [f"| {name} | {count} |" for name, count in cov["failure_class_counts"].items()] or ["| none | 0 |"]
    if cov["failures"]:
        out += ["", "| branch | class | failure |", "|---|---|---|"]
        out += [f"| {f['identity']} | {f['failure_class']} | {str(f['failure']).replace('|', '/')} |"
                for f in cov["failures"]]
    out += ["", "## Part A", "", f"Question: {audit['part_a']['question']}", "",
            f"Bound B = {cov['failed']}/{cov['scheduled']} = {fmt(a['bound_B'], 4)} "
            f"(bias margin {a['bias_margin']}). Smoke-failed branches with a #77 N1 v1 reference "
            f"outcome: {a['smoke_failed_with_v1_reference']} (direct contrast computable iff >= 5: "
            f"{fmt(a['direct_contrast_computable'])}).", ""]
    if a["direct_contrast"]:
        out += contrast_lines(a["direct_contrast"]) + [""]
    out += [f"Verdict **{a['verdict']}** → `{a['token']}`.", "", "## Part B", "",
            f"Question: {audit['part_b']['question']} Statistic: {audit['part_b']['statistic']}. "
            f"Precision guards: {json.dumps(audit['precision_guards'], sort_keys=True)}; contrast margin "
            f"{audit['contrast_margin']}.", ""]
    for key, title in (("b1_issue87", "B1 — #87 oracle timeouts (target) vs executed slots"),
                       ("b2_issue77_n1", "B2 — #77 N1 v1 typed failures (target) vs complete branches")):
        row = b[key]
        out += [f"### {title}", "",
                f"Target slots {row['slots_target']} ({row['slots_target_without_executed_smoke_branch']} "
                f"without an executed smoke branch); reference slots {row['slots_reference']} "
                f"({row['slots_reference_without_executed_smoke_branch']} without).", ""]
        if key == "b1_issue87":
            out += [f"#87 strata: {json.dumps(row['strata_counts'], sort_keys=True)}.", ""]
        out += contrast_lines(row["contrasts"]) + ["", f"Verdict **{row['verdict']}** → `{row['token']}`.", ""]
    st = b["b1_stability_stratum"]
    out += ["### #87 stability stratum (separate, never pooled)", "",
            f"{st['slots']} slots, {st['executed_on_smoke']} executed on the smoke; {st['interval']}.", "",
            "| slot | branch | smoke status | " + " | ".join(OUTCOMES) + " |", "|---|---|---|---|---|---|"]
    out += [f"| {r['slot']} | {r['branch_identity']} | {r['smoke_status']} | "
            + " | ".join(fmt(r[name]) for name in OUTCOMES) + " |" for r in st["rows"]]
    m = s["mechanism_inventory"]
    out += ["", "## Mechanism inventory", "", m["label"] + ". A same-port partner is another cell sharing "
            f"any of its (agent, game, physics) ports dispatched within {m['partner_window_seconds']:g} s.", "",
            "| #87 stratum | slots | 'Address already in use' in engine log | same-port partner ≤5 s | both |",
            "|---|---|---|---|---|"]
    out += [f"| {k} | {v.get('slots', 0)} | {v.get('address_already_in_use', 0)} | "
            f"{v.get('same_port_partner_within_5s', 0)} | {v.get('both', 0)} |"
            for k, v in m["issue87_by_stratum"].items()]
    sm = m["smoke"]
    out += ["", f"Smoke: {sm['attempts_with_runtime']} attempts with retained runtime, "
            f"{sm['address_already_in_use']} with 'Address already in use', "
            f"{sm['same_port_partner_within_5s']} with a same-port partner ≤5 s "
            f"({sm['dispatch_markers']} dispatch markers; slot-private port blocks).", "",
            "## Determinism", "", "| comparison | pairs | agree | share | confusion |", "|---|---|---|---|---|"]
    det = s["determinism"]
    out += [agreement_line("stop kind: #77 N1 v1 vs smoke (complete in both)", det["stop_kind_vs_issue77_n1_v1"]),
            agreement_line("pig_removed: #87 executed slots vs smoke", det["pig_removed_vs_issue87_executed_slots"]),
            agreement_line("pig_removed: #89 completion slots vs smoke",
                           det["pig_removed_vs_issue89_completion_slots"])]
    r = s["render_invariance"]
    out += ["", "## Render invariance", "",
            f"Complete branches with endpoint evidence: {r['complete_with_evidence']}; byte-equal endpoint "
            f"renders (before and after hold): {r['endpoint_byte_equal']}; animation-only differences "
            f"(pixels differ only inside Bird/Pig boxes — the blink animation): "
            f"{r['endpoint_animation_only_difference']} (differing pixels "
            f"p50 {fmt(r['animation_only_differing_pixels']['p50'], 0)}, max "
            f"{fmt(r['animation_only_differing_pixels']['max'], 0)}). Shot-initial render byte-equal "
            f"{r['shot_initial_byte_equal']}, animation-only {r['shot_initial_animation_only_difference']}. "
            f"render_invariance_violation failures: {r['render_invariance_violation_failures']}.", "",
            "## Throughput profile", "", "| stage (complete branches) | n | mean s | p50 s | p90 s |",
            "|---|---|---|---|---|"]
    out += [f"| {k} | {v['n']} | {fmt(v['mean'], 1)} | {fmt(v['p50'], 1)} | {fmt(v['p90'], 1)} |"
            for k, v in t["stage_seconds"].items()]
    ref = t["references"]
    out += ["", "| figure | smoke | R3 (#76) | #77 N1 v1 |", "|---|---|---|---|",
            f"| workers | {t['workers']} | {ref['r3']['workers']} | {ref['issue77_n1_v1']['workers']} |",
            f"| per-branch wall mean s | {fmt(t['per_branch_wall_mean'], 1)} | "
            f"{fmt(ref['r3']['per_attempt_wall_mean'], 1)} | {fmt(ref['issue77_n1_v1']['per_branch_wall_mean'], 1)} |",
            f"| per-branch wall p50 s | {fmt(t['per_branch_wall_p50'], 1)} | {fmt(ref['r3']['per_attempt_wall_p50'], 1)} "
            f"| {fmt(ref['issue77_n1_v1']['per_branch_wall_p50'], 1)} |",
            f"| per-branch wall p90 s | {fmt(t['per_branch_wall_p90'], 1)} | {fmt(ref['r3']['per_attempt_wall_p90'], 1)} "
            f"| {fmt(ref['issue77_n1_v1']['per_branch_wall_p90'], 1)} |",
            f"| per-branch wall max s | {fmt(t['per_branch_wall_max'], 1)} | — | — |",
            f"| campaign wall s | {fmt(t['campaign_wall_seconds'], 0)} | {fmt(ref['r3']['collection_wall_seconds'], 0)} "
            f"| {fmt(ref['issue77_n1_v1']['collection_wall_seconds'], 0)} |",
            f"| attempted | {t['attempted']} | {ref['r3']['attempted']} | {ref['issue77_n1_v1']['attempted']} |",
            f"| amortized s/branch (campaign wall / attempted) | {fmt(t['amortized_seconds_per_branch'], 1)} | "
            f"{fmt(ref['r3']['amortized_seconds_per_branch'], 1)} | "
            f"{fmt(ref['issue77_n1_v1']['amortized_seconds_per_branch'], 1)} |",
            f"| smoke / reference per-branch wall | — | {fmt(ref['r3']['smoke_per_branch_wall_ratio'])} | "
            f"{fmt(ref['issue77_n1_v1']['smoke_per_branch_wall_ratio'])} |",
            f"| smoke / reference amortized | — | {fmt(ref['r3']['smoke_amortized_ratio'])} | "
            f"{fmt(ref['issue77_n1_v1']['smoke_amortized_ratio'])} |", "",
            f"Branches/hour {fmt(t['branches_per_hour'], 2)} (complete {fmt(t['complete_branches_per_hour'], 2)}); "
            f"worker-seconds/branch {fmt(t['worker_seconds_per_branch'], 1)}; artifact bytes/branch "
            f"{fmt(t['artifact_bytes_per_branch'], 0)}. Publish-time engine-outcome scan: smoke "
            f"{t['scan']['smoke_shots']} shots {fmt(t['scan']['smoke_cpu_seconds'], 1)} CPU s / "
            f"{fmt(t['scan']['smoke_wall_seconds'], 1)} wall s; references {t['scan']['reference_shots']} shots "
            f"{fmt(t['scan']['reference_cpu_seconds'], 1)} CPU s / {fmt(t['scan']['reference_wall_seconds'], 1)} "
            "wall s. R3 per-attempt figures are its combined per-attempt wall (≈563 s mean).", ""]
    c = s["compute"]
    out += ["## Compute accounting", "", "| item | value |", "|---|---|"]
    out += [f"| {k} | {fmt(v, 1)} |" for k, v in c.items()]
    out += ["", "## Claim boundary", "", s["claim_boundary"] + ". Intervals are DESCRIPTIVE; the mechanism "
            "inventory is never a verdict input.", "", "## Reproduction", "", "```",
            "python -u -m scripts.run_capture_reliability_smoke --publish",
            s["validation_command"], "```", "",
            f"Media gallery: `{shown(MEDIA / 'index.html')}` (manifest `{s['media']['manifest']}`, "
            f"{s['media']['frames']} frames over {s['media']['branches']} branches).", ""]
    return "\n".join(out)


# ------------------------------------------------------------ entry points

def publish():
    started, cpu = time.monotonic(), time.process_time()
    plan = load_plan()
    report = build(plan)
    for name, data in report["files"].items():
        (OUTPUT / name).write_bytes(data)
    if MEDIA.exists():
        shutil.rmtree(MEDIA)
    for source, relative in report["copies"]:
        target = MEDIA / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    (MEDIA / "manifest.json").write_bytes(report["manifest_bytes"])
    (MEDIA / "index.html").write_text(gallery_html(report["manifest"]))
    compute = {"accounting": report["summary"]["compute"],
               "publish": {"wall_seconds": time.monotonic() - started,
                           "cpu_seconds": time.process_time() - cpu,
                           "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}}
    (OUTPUT / "compute.json").write_bytes(json_bytes(compute))
    d = report["summary"]["dispositions"]
    log("published: " + ", ".join(f"{k}={v['token']}" for k, v in d.items()))


def rescan_sample(smoke):
    ordered = sorted(smoke["outcomes"], key=lambda i: sha256(i.encode()).hexdigest())
    mismatches = []
    for identity in ordered[:RESCAN_SAMPLE]:
        outcome = pipeline.branch_outcome(smoke["results"][identity]["segments"][0])
        if outcome != smoke["outcomes"][identity]:
            mismatches.append(f"engine outcome re-scan differs: {identity}")
    return ordered[:RESCAN_SAMPLE], mismatches


def validate():
    started = time.monotonic()
    plan = load_plan()
    report = build(plan)
    mismatches = []
    for name, data in report["files"].items():
        path = OUTPUT / name
        if not path.is_file() or path.read_bytes() != data:
            mismatches.append(f"{name} differs from recomputation")
    manifest = MEDIA / "manifest.json"
    if not manifest.is_file() or manifest.read_bytes() != report["manifest_bytes"]:
        mismatches.append("media manifest.json differs from recomputation")
    for branch in report["manifest"]["branches"]:
        for frame in branch["frames"].values():
            path = MEDIA / frame["path"]
            if not path.is_file() or digest(path.read_bytes()) != frame["sha256"]:
                mismatches.append(f"media frame differs: {frame['path']}")
    compute = OUTPUT / "compute.json"
    if not compute.is_file() or read_json(compute).get("accounting") != report["summary"]["compute"]:
        mismatches.append("compute.json accounting differs from recomputation")
    sample, rescan = rescan_sample(report["smoke"])
    mismatches += rescan
    if mismatches:
        raise ValueError("validation failed: " + "; ".join(mismatches))
    log(f"validated: {len(report['files'])} report files, media manifest, "
        f"{len(sample)} re-scanned shots in {time.monotonic() - started:.0f}s")
