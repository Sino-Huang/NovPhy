"""Issue-95 ADD-EXP: #94 follow-up controls, analysis + GPU seconds only (zero engine seconds).

Binding runner module: scripts/run_launch_power_followup.py
Exact validation command: python -u -m scripts.run_launch_power_followup --validate

Two objections to #94 (C24 supported, C25 not_supported, reading row 2) are
testable on the retained #94 engine outcomes without a new engine slot:

- Training-regime control (C26/C27).  The #94 candidates carry release 1000 ms,
  which the frozen rankers saw only on the 13 N1 radius-80 actions; their
  dominant training pool (#71, 4377 shots) is release 600 ms.  The release time
  is engine-inert: the interface jar builds Shot(x, y, t_shot=release,
  t_tap=tap) and forwards only (x, y, tap_time) to the engine
  (ProxyTapShootMessage).  Re-scoring the identical #94 inventory with release
  600 therefore changes only the ranker input, and every retained #94 engine
  verdict applies unchanged.  C26/C27 re-apply the frozen #94 C24/C25 rules.
- Power-only control (C28).  Every #94 choice sat in the saturated column, so
  the AUC could be the mechanical consequence of preferring maximum power.  The
  within-power AUC counts only success/failure pairs inside the same launch-
  power column; a pure power preference scores exactly 0.5 there.  A power-only
  ordering (cost = -expected speed) is reported beside the rankers.

Chronology: designed after the #94 results were published (a post-#94
follow-up); plan.json is frozen before any release-600 cost or any within-power
statistic is computed.  Every interval is DESCRIPTIVE.

Modes: --prepare (freeze), --score (GPU seconds), --publish, --validate.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import fcntl
from hashlib import sha256
import io
import json
from pathlib import Path
import subprocess
import tempfile
import time

from scripts import run_launch_power_probe as p94

ROOT = p94.ROOT
OUTPUT = ROOT / ".local-artifacts/issue-95-launch-power-followup-v1"
SOURCE = p94.OUTPUT
GPU_LOCK_PATH = p94.GPU_LOCK_PATH

IDENTITY = "issue-95-launch-power-followup-v1"
SCHEMA_PLAN = "issue_95_launch_power_followup_plan_v1"
SCHEMA_DECISION = "issue_95_decision_record_v1"
SCHEMA_INERT = "issue_95_release_inertness_v1"
SCHEMA_COMPUTE = "issue_95_compute_v1"
SCHEMA_REPORT = "issue_95_report_v1"
SCHEMA_LEDGER = "issue_95_ledger_v1"
VALIDATION_COMMAND = "python -u -m scripts.run_launch_power_followup --validate"

CONTROL_RELEASE_MS = 600
MODEL_SYSTEMS = p94.MODEL_SYSTEMS
SEEDS = p94.SEEDS
STOP_TOKEN = p94.STOP_TOKEN
DISPOSITION_TOKENS = p94.DISPOSITION_TOKENS
GPU_CAP_SECONDS = 3600.0
C28_RULE = {"members_with_pairs_min": 4, "supported_max_auc": 0.50, "not_supported_min_auc": 0.60}

CLAIMS = {
    "C26": ("With the engine-inert release input set to the rankers' dominant training value "
            "(600 ms), the three frozen rankers' pooled top-1 on the #94 angle x launch-power "
            "inventory is below inventory-matched chance."),
    "C27": ("With the engine-inert release input set to 600 ms, the frozen rankers' "
            "within-state engine-truth AUC on the #94 inventory is at or below chance."),
    "C28": ("Within a fixed launch-power column of the #94 inventory (release 1000, the #94 "
            "rankings), the frozen rankers' engine-truth ordering is at or below chance."),
}
RULES = {
    "C26": ("the frozen #94 C24 rule (cross-pool q2 criterion, margin 0.0) on the release-600 "
            "rankings: readiness_or_precision_insufficient if ceiling members < 4 or verdict "
            "coverage < 75%; supported if the pooled top-1 rate < pooled chance AND the "
            "member-clustered paired-difference interval lies entirely below 0; "
            "not_supported_by_this_experiment if the paired-difference point estimate >= 0; "
            "otherwise readiness_or_precision_insufficient"),
    "C27": ("the frozen #94 C25 rule on the release-600 rankings: same readiness gate; "
            "supported if member-clustered AUC_m <= 0.50; not_supported_by_this_experiment if "
            "AUC_m >= 0.60; otherwise readiness_or_precision_insufficient"),
    "C28": ("within-power AUC per cell = P(success has lower predicted cost than failure) over "
            "success/failure pairs inside the same radius column (5/9/13/17/80 px), ties 0.5; "
            "cells without such a pair are unmeasurable; readiness_or_precision_insufficient if "
            "fewer than 4 members carry a within-column pair or #94 verdict coverage < 75%; "
            "supported if member-clustered AUC_m <= 0.50; not_supported_by_this_experiment if "
            "AUC_m >= 0.60; otherwise readiness_or_precision_insufficient"),
}
READINGS = {
    "release_control": [
        {"C26": "supported", "C27": "not_supported_by_this_experiment",
         "reading": ("The #94 dissociation survives moving the release input into the rankers' "
                     "dominant training regime; the release-1000 extrapolation does not explain "
                     "it. The remaining training-range confound is the pull radius alone.")},
        {"C26": "supported", "C27": "not not_supported",
         "reading": ("Selection still fails at release 600 but the above-chance ordering does "
                     "not replicate; the #94 ordering result is tied to the release-1000 input "
                     "and must be scoped to it.")},
        {"C26": "not supported", "C27": "any",
         "reading": ("Selection is not below chance at release 600; the #94 top-1 failure is "
                     "tied to the release-1000 extrapolation and must be scoped to it.")},
    ],
    "power_control": [
        {"C28": "supported",
         "reading": ("Within a fixed launch power the rankers order outcomes at or below chance; "
                     "the #94 AUC is attributable to a launch-power preference, and 'ordering "
                     "tracks an engine-relevant coordinate' is scoped to launch power.")},
        {"C28": "not_supported_by_this_experiment",
         "reading": ("Within a fixed launch power the rankers still order outcomes above chance; "
                     "the #94 AUC is not only a launch-power preference.")},
        {"C28": STOP_TOKEN,
         "reading": "Report descriptively in the appendix only."},
    ],
}
SOURCE_FILES = ("scripts/run_launch_power_followup.py", "scripts/run_launch_power_probe.py",
                "scripts/run_engine_outcome_reactive_diagnostic.py", "scripts/run_lookahead_control.py")

read_json = p94.read_json
json_text = p94.json_text
write_json = p94.write_json
sha256_of = p94.sha256_of
bootstrap = p94.bootstrap
fmt = p94.fmt
fmt_interval = p94.fmt_interval


def log(message):
    print(f"[issue-95-launch-power-followup] {message}", flush=True)


def utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class GPULock:
    def __enter__(self):
        self._handle = open(GPU_LOCK_PATH, "a+")
        fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX)
        log("shared GPU flock acquired for this measured phase")
        return self

    def __exit__(self, *exception):
        fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        self._handle.close()
        log("shared GPU flock released")
        return False


# ---------------------------------------------------------------------------
# release-time engine inertness (code evidence)
# ---------------------------------------------------------------------------

def javap(scratch, member):
    return subprocess.run(["javap", "-c", "-p", member], cwd=scratch, check=True,
                          capture_output=True, text=True).stdout


def release_inertness():
    jar = p94.CAMPAIGN / "player" / "game_playing_interface.jar"
    members = ["server/util/Shot.class", "server/schema/ShootAndTapSchema.class",
               "proxy/message/ProxyTapShootMessage.class"]
    with tempfile.TemporaryDirectory() as scratch:
        subprocess.run(["unzip", "-q", "-o", str(jar), *members], cwd=scratch, check=True)
        shot = javap(scratch, members[0])
        schema = javap(scratch, members[1])
        message = javap(scratch, members[2])
    constructor = shot.split("public server.util.Shot(int, int, int, int);")[1].split("return")[0]
    fields = [line.split("Field ")[1].strip() for line in constructor.splitlines() if "putfield" in line]
    shoot = schema.split("public byte[] shoot(")[1].split("public org.json")[0]
    getters = sorted({line.split("Method server/util/Shot.")[1].split(":")[0]
                      for line in shoot.splitlines() if "Method server/util/Shot.get" in line})
    json_keys = [line.split("// String ")[1].strip() for line in message.splitlines()
                 if "ldc" in line and "// String " in line]
    inert = fields == ["x:I", "y:I", "t_shot:I", "t_tap:I"] and getters == ["getT_tap", "getX", "getY"]
    if not inert or "tap_time" not in json_keys:
        raise ValueError(f"release inertness not established: {fields} {getters} {json_keys}")
    return {
        "schema": SCHEMA_INERT, "identity": IDENTITY, "computed_at": utc_now(),
        "jar": str(jar.relative_to(ROOT)), "jar_sha256": sha256_of(jar),
        "bridge_transport": p94.repo_line("src/webui/bridge.py", r'self._send\(code, "iiii"'),
        "shot_constructor_fields_in_argument_order": fields,
        "shoot_and_tap_schema_shot_getters": getters,
        "proxy_tap_shoot_message_strings": json_keys,
        "engine_tapshoot_reads": p94.p93.source_line(
            "Scripts/Assembly-CSharp/AIBirdsConnection.cs", r'float asFloat3 = data\[2\]\["tap_time"\]'),
        "finding": "release_time_engine_inert",
        "reading": ("ScienceBirdsBridge.shoot sends (x, y, release_time, tap_time); the jar stores "
                    "them as Shot(x, y, t_shot, t_tap); ShootAndTapSchema.shoot reads only getX, "
                    "getY and getT_tap into ProxyTapShootMessage(x, y, tap_time), and "
                    "AIBirdsConnection.TapShoot reads only x, y and tap_time. The release time "
                    "never reaches the engine, so a #94 candidate at release 600 is the identical "
                    "engine action and its retained #94 verdict applies."),
    }


# ---------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------

def source_records_digest(plan94):
    digest = sha256()
    names = [cell["identity"] for cell in p94.scheduled_cells(plan94)] + [
        p94.decision_identity(system, seed, state["member"])
        for state in plan94["states"] for system in MODEL_SYSTEMS for seed in SEEDS]
    for name in sorted(names):
        digest.update(name.encode())
        digest.update(sha256_of(SOURCE / "records" / f"{name}.json").encode())
    return {"records": len(names), "sha256_of_sorted_name_and_sha256": "sha256:" + digest.hexdigest()}


def control_inventory(plan94, member):
    return [{**item, "action": {**item["action"], "release_time_ms": CONTROL_RELEASE_MS}}
            for item in plan94["inventory"]["members"][member]]


def frozen_plan(frozen_at, plan94, inert):
    summary94 = read_json(SOURCE / "summary.json")
    return {
        "schema": SCHEMA_PLAN, "identity": IDENTITY, "version": 1, "role": "terminal",
        "frozen_at": frozen_at, "frozen_before_scoring_run": True,
        "issue_64_authorized": False, "validation_command": VALIDATION_COMMAND,
        "runner": "scripts/run_launch_power_followup.py",
        "chronology": ("designed after the #94 publication (post-#94 follow-up, #94 dispositions "
                       f"{summary94['dispositions']} known); frozen before any release-600 cost "
                       "or within-power statistic was computed; the power-only baseline (engine "
                       "verdicts only, no ranker input) was printed by a synthetic-cost dry run "
                       "of this runner before the freeze"),
        "engine_seconds": 0,
        "source_experiment": {"identity": p94.IDENTITY, "plan_sha256": sha256_of(SOURCE / "plan.json"),
                              "summary_sha256": sha256_of(SOURCE / "summary.json"),
                              "compute_sha256": sha256_of(SOURCE / "compute.json"),
                              "records": source_records_digest(plan94)},
        "release_inertness": {"path": str(OUTPUT / "release_inertness.json"),
                              "sha256": sha256_of(OUTPUT / "release_inertness.json"),
                              "finding": inert["finding"]},
        "control_inventory_rule": ("the frozen #94 inventory per member with release_time_ms = "
                                   f"{CONTROL_RELEASE_MS} (drag and tap unchanged); ordinals and "
                                   "branch identities unchanged; verdicts from the #94 oracle records"),
        "training_regime": ("#71 predictor pool: 4377 shots, all release 600, radius 11.18-177.99 px, "
                            "26 shots below the 18.49 px clamp; N1 pool: 96 shots, release 1000, "
                            "radius 79.51-80.66 px (#94 training_action_support.json)"),
        "rankers": {"systems": list(MODEL_SYSTEMS), "seeds": list(SEEDS),
                    "anchors": "the frozen #94 plan anchors (states[].anchor), sha256-checked",
                    "dynamics_checkpoints": "the frozen #94 plan dynamics_checkpoints",
                    "selector": "ReactiveSelector.choose (argmin cost, ties lower ordinal)",
                    "adaptation": "none"},
        "estimands": {
            "release_control": ("all #94 estimands recomputed with the release-600 rankings by the "
                                "#94 tables() code: ceiling, pooled/per-system top-1 vs chance, "
                                "cell-unit and member-clustered AUC, top-3, structure, chosen "
                                "histogram, saturated-column share"),
            "rank_agreement": ("per cell Spearman rho between release-1000 and release-600 "
                               "predicted costs over the 20 candidates; DESCRIPTIVE"),
            "within_power_auc": ("per cell AUC over same-column success/failure pairs; cell-unit and "
                                 "member-clustered; for the #94 release-1000 rankings (C28) and the "
                                 "release-600 rankings (DESCRIPTIVE)"),
            "power_only_baseline": ("cost = -expected speed (ties lower ordinal for top-1, 0.5 for "
                                    "AUC), one deterministic ranking per member: top-1 vs chance, "
                                    "AUC_m, within-power AUC (0.5 by construction); DESCRIPTIVE"),
            "bootstrap": (f"{p94.BOOTSTRAP_DRAWS} draws, seed {p94.BOOTSTRAP_SEED}, quantiles "
                          f"{list(p94.INTERVAL_QUANTILES)}, member-clustered"),
        },
        "claims": CLAIMS, "disposition_rules": RULES | {"C28_numbers": C28_RULE,
                                                       "tokens": list(DISPOSITION_TOKENS)},
        "readings": READINGS,
        "stated_predictions": {
            "source": "runner author (the owner did not state predictions); recorded before scoring",
            "C26": {"expected_token": "supported",
                    "rationale": "top-1 was 0/72 at release 1000 and the release input is engine-inert, so nothing in the control adds success information."},
            "C27": {"expected_token": "no prediction",
                    "rationale": "the release input may move the power preference in either direction; no directional prediction."},
            "C28": {"expected_token": "no prediction",
                    "rationale": "within-column pairs are few (one success per ceiling member); no directional prediction."},
        },
        "caps": {"gpu_seconds": GPU_CAP_SECONDS, "engine_seconds": 0},
        "expected_structure": {"decision_cells": len(plan94["states"]) * len(MODEL_SYSTEMS) * len(SEEDS)},
        "source_text": {name: (ROOT / name).read_text() for name in SOURCE_FILES},
        "claim_boundary": ("controls on the #94 inventory and its retained engine verdicts only: the "
                           "release control moves one engine-inert ranker input into the training "
                           "regime and leaves the pull-radius extrapolation (5-17 px, mostly outside "
                           "training support) untouched; no retraining, no new engine slot, no other "
                           "ranker family; #94 dispositions are inputs and are not re-opened"),
    }


def prepare(output):
    output = Path(output)
    if (output / "plan.json").is_file():
        load_plan(output)
        log("existing frozen plan validated")
        return 0
    if (output / "records").exists():
        raise ValueError("records exist before the freeze; refusing")
    plan94 = p94.load_plan(SOURCE)
    inert = release_inertness()
    write_json(output / "release_inertness.json", inert)
    log(f"release inertness: {inert['finding']} (Shot fields {inert['shot_constructor_fields_in_argument_order']}, "
        f"forwarded getters {inert['shoot_and_tap_schema_shot_getters']})")
    plan = frozen_plan(utc_now(), plan94, inert)
    (output / "plan.json").write_text(json_text(plan))
    load_plan(output)
    log(f"frozen plan published: {plan['expected_structure']['decision_cells']} release-600 decision "
        "cells; no release-600 cost or within-power statistic computed")
    return 0


def load_plan(output):
    path = Path(output) / "plan.json"
    if not path.is_file():
        raise ValueError("plan.json missing; run --prepare first")
    plan = read_json(path)
    if plan.get("schema") != SCHEMA_PLAN or not plan.get("frozen_before_scoring_run"):
        raise ValueError("plan.json is not the frozen issue-95 protocol")
    if plan["source_experiment"]["plan_sha256"] != sha256_of(SOURCE / "plan.json"):
        raise ValueError("the #94 plan changed after the freeze")
    return plan


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------

def decision_path(output, system, seed, member):
    return Path(output) / "records" / f"decision--{system}--seed{seed}--{member}--power-release{CONTROL_RELEASE_MS}.json"


def score(output):
    output = Path(output)
    plan = load_plan(output)
    plan94 = p94.load_plan(SOURCE)
    wlc, adapter, objective = p94.load_scoring_stack()
    models, gpu, written = {}, 0.0, 0
    for state in plan94["states"]:
        anchor = state["anchor"]
        if sha256_of(anchor["frame_path"]) != anchor["sha256"]:
            raise ValueError(f"anchor of {state['member']} changed")
        inventory = control_inventory(plan94, state["member"])
        for system in MODEL_SYSTEMS:
            for seed in SEEDS:
                path = decision_path(output, system, seed, state["member"])
                if path.is_file():
                    continue
                decision, seconds = p94.score_inventory(wlc, adapter, objective, models, plan94,
                                                        anchor, system, seed, inventory)
                gpu += seconds
                write_json(path, {
                    "schema": SCHEMA_DECISION, "plan_identity": IDENTITY,
                    "cell": {"system": system, "seed": seed, "member": state["member"],
                             "state": state["canonical_state"], "release_time_ms": CONTROL_RELEASE_MS},
                    "anchor": anchor, "carrier_sha256": anchor["sha256"], "decision": decision,
                    "failure": (None if decision["failure"] is None
                                else f"decision_failure: {decision['failure']}"),
                    "failure_kind": None if decision["failure"] is None else "decision_failure",
                    "gpu_seconds": seconds, "issue_64_authorized": False})
                written += 1
    write_json(output / "ledger-score.json", {"schema": SCHEMA_LEDGER, "identity": IDENTITY,
                                              "gpu_seconds_elapsed": gpu, "cells": written})
    if gpu > GPU_CAP_SECONDS:
        raise ValueError(f"{STOP_TOKEN}: gpu_cap_exceeded")
    log(f"score: {written} release-600 decision records, gpu {gpu:.1f}s")
    return 0


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------

def control_decisions(output, plan94):
    decisions = {}
    for state in plan94["states"]:
        for system in MODEL_SYSTEMS:
            for seed in SEEDS:
                record = read_json(decision_path(output, system, seed, state["member"]))
                if record.get("schema") != SCHEMA_DECISION or record.get("plan_identity") != IDENTITY:
                    raise ValueError("control decision binding differs")
                decisions[p94.decision_identity(system, seed, state["member"])] = record
    return decisions


def within_power(rows, design):
    """Per-cell AUC over success/failure pairs inside one radius column."""
    column = {o: item["radius_index"] for o, item in design.items()}
    cells = []
    for row in rows:
        if not row["measurable"] or not row["ceiling"]:
            continue
        success = set(row["successes"])
        wins = pairs = 0.0
        for o1, c1 in row["admissible"]:
            if o1 not in success:
                continue
            for o2, c2 in row["admissible"]:
                if o2 in success or column[o2] != column[o1]:
                    continue
                pairs += 1
                wins += 1.0 if c1 < c2 else 0.5 if c1 == c2 else 0.0
        if pairs:
            cells.append({"member": row["member"], "system": row["system"], "seed": row["seed"],
                          "pairs": int(pairs), "auc": wins / pairs})
    return {"cells": len(cells), "members": sorted({c["member"] for c in cells}),
            "cell_unit": bootstrap([c["auc"] for c in cells]),
            "member_clustered": bootstrap([c["auc"] for c in cells], [c["member"] for c in cells]),
            "per_system": {s: bootstrap([c["auc"] for c in cells if c["system"] == s],
                                        [c["member"] for c in cells if c["system"] == s])
                           for s in MODEL_SYSTEMS},
            "rows": cells}


def power_only_baseline(plan94, oracle, design, realization):
    per_member = []
    for state in plan94["states"]:
        verdicts = {}
        for o in design:
            record = oracle.get(p94.oracle_identity(state["member"], o))
            if record is not None and record["outcome"] is not None:
                verdicts[o] = record["outcome"]["first_shot_success"]
        if not any(verdicts.values()):
            continue
        ranked = sorted(verdicts, key=lambda o: (-realization[o]["expected_speed"], o))
        pos = [-realization[o]["expected_speed"] for o in verdicts if verdicts[o]]
        neg = [-realization[o]["expected_speed"] for o in verdicts if not verdicts[o]]
        k, n = len(pos), len(verdicts)
        per_member.append({"member": state["member"], "chosen": ranked[0],
                           "top1_hit": bool(verdicts[ranked[0]]), "chance_top1": k / n,
                           "auc": p94.cell_auc(pos, neg) if neg else None})
    hits = sum(r["top1_hit"] for r in per_member)
    return {"members": len(per_member), "top1_hits": hits,
            "top1_rate": hits / len(per_member) if per_member else None,
            "chance_mean": sum(r["chance_top1"] for r in per_member) / len(per_member) if per_member else None,
            "auc_member_clustered": bootstrap([r["auc"] for r in per_member if r["auc"] is not None],
                                              [r["member"] for r in per_member if r["auc"] is not None]),
            "within_power_auc": 0.5, "rows": per_member}


def c28_disposition(block, coverage):
    members = len(block["members"])
    if members < C28_RULE["members_with_pairs_min"] or coverage < p94.RULE["verdict_coverage_min"]:
        return STOP_TOKEN
    auc_m = block["member_clustered"]["mean"]
    if auc_m <= C28_RULE["supported_max_auc"]:
        return "supported"
    if auc_m >= C28_RULE["not_supported_min_auc"]:
        return "not_supported_by_this_experiment"
    return STOP_TOKEN


def reading_rows(c26, c27, c28):
    release = READINGS["release_control"]
    if c26 == "supported":
        row = release[0] if c27 == "not_supported_by_this_experiment" else release[1]
    else:
        row = release[2]
    power = next(r for r in READINGS["power_control"] if r["C28"] == c28)
    return {"release_control": row, "power_control": power}


def compute_tables(output, plan):
    plan94 = p94.load_plan(SOURCE)
    oracle = p94.load_oracle_records(SOURCE, plan94)
    decisions94 = p94.load_decisions(SOURCE, plan94)
    control = control_decisions(output, plan94)
    if len(control) != plan["expected_structure"]["decision_cells"]:
        raise ValueError("control decision count differs; run --score")
    design = {item["ordinal"]: item for item in plan94["inventory"]["design"]}
    realization = {row["ordinal"]: row for row in plan94["inventory"]["realization"]}
    rows94 = p94.decision_rows(plan94, oracle, decisions94)
    rows600 = p94.decision_rows(plan94, oracle, control)
    table600 = p94.tables(plan94, oracle, rows600)
    agreement = []
    for key, record in control.items():
        base = {r["ordinal"]: r["predicted_cost"] for r in decisions94[key]["decision"]["ranking"]}
        new = {r["ordinal"]: r["predicted_cost"] for r in record["decision"]["ranking"]}
        ordinals = sorted(set(base) & set(new))
        agreement.append(p94.spearman([base[o] for o in ordinals], [new[o] for o in ordinals]))
    wp94 = within_power(rows94, design)
    wp600 = within_power(rows600, design)
    coverage = table600["verdict_coverage"]
    dispositions = {"C26": table600["dispositions"]["C24"], "C27": table600["dispositions"]["C25"],
                    "C28": c28_disposition(wp94, coverage)}
    summary94 = read_json(SOURCE / "summary.json")
    return {
        "schema": SCHEMA_COMPUTE, "identity": IDENTITY,
        "structure": {"decision_cells": len(control), "source_slots": len(oracle)},
        "release_control": table600,
        "release_1000_reference": {"top1": summary94["arm"]["top1"], "auc": summary94["arm"]["auc"],
                                   "dispositions": summary94["dispositions"]},
        "rank_agreement_release_1000_vs_600": {"median_rho": p94.median(agreement),
                                               "min_rho": min(a for a in agreement if a is not None),
                                               "cells": len(agreement)},
        "within_power": {"release_1000": wp94, "release_600": wp600},
        "power_only_baseline": power_only_baseline(plan94, oracle, design, realization),
        "dispositions": dispositions,
        "readings": reading_rows(dispositions["C26"], dispositions["C27"], dispositions["C28"]),
        "stated_predictions": plan["stated_predictions"],
        "compute": {"score_gpu_seconds": read_json(output / "ledger-score.json")["gpu_seconds_elapsed"],
                    "engine_seconds": 0},
    }


def slot_join_csv(compute):
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["member", "system", "seed", "within_power_pairs_release_1000",
                     "within_power_auc_release_1000"])
    for row in compute["within_power"]["release_1000"]["rows"]:
        writer.writerow([row["member"], row["system"], row["seed"], row["pairs"], row["auc"]])
    return buffer.getvalue()


def comparisons_csv(compute):
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["id", "statistic", "value", "interval_low", "interval_high", "interval_label", "detail"])

    def put(identity, statistic, value, block=None, detail=""):
        writer.writerow([identity, statistic, "" if value is None else value,
                         block["interval"][0] if block else "", block["interval"][1] if block else "",
                         "DESCRIPTIVE" if block else "", detail])

    t = compute["release_control"]
    put("r600_coverage", "verdict coverage (#94 slots)", t["verdict_coverage"])
    put("r600_ceiling", "member ceiling", t["ceiling"]["value"], t["ceiling"]["interval"],
        f"{t['ceiling']['numerator']} of {t['ceiling']['denominator']}")
    put("r600_top1", "release-600 pooled top-1", t["top1"]["rate"],
        detail=f"{t['top1']['hits']}/{t['top1']['cells']}; chance {t['top1']['chance_mean']}")
    d = t["top1"]["paired_difference_member_clustered"]
    put("r600_top1_minus_chance", "release-600 top-1 minus chance", d and d["mean"], d)
    for s in MODEL_SYSTEMS:
        b = t["top1"]["per_system"][s]
        put(f"r600_top1_{s}", "release-600 top-1 per system", b["rate"],
            detail=f"{b['hits']}/{b['cells']}; chance {b['chance_mean']}")
    for unit in ("cell_unit", "member_clustered"):
        b = t["auc"][unit]
        put(f"r600_auc_{unit}", "release-600 within-state AUC", b and b["mean"], b)
    for s in MODEL_SYSTEMS:
        b = t["auc"]["per_system"][s]["member_clustered"]
        put(f"r600_auc_{s}", "release-600 AUC per system", b and b["mean"], b)
    share = t["saturated_column_chosen_share"]
    put("r600_saturated_share", "release-600 selector choices at the saturated column",
        share["selector_full_inventory"], detail=json.dumps(t["chosen_ordinal_histogram"], sort_keys=True))
    agree = compute["rank_agreement_release_1000_vs_600"]
    put("rank_agreement", "median Spearman rho release-1000 vs release-600 costs", agree["median_rho"],
        detail=json.dumps(agree, sort_keys=True))
    for release in ("release_1000", "release_600"):
        w = compute["within_power"][release]
        for unit in ("cell_unit", "member_clustered"):
            b = w[unit]
            put(f"within_power_{release}_{unit}", f"within-power AUC ({release})", b and b["mean"], b,
                f"{w['cells']} cells, {len(w['members'])} members")
        for s in MODEL_SYSTEMS:
            b = w["per_system"][s]
            put(f"within_power_{release}_{s}", f"within-power AUC per system ({release})", b and b["mean"], b)
    base = compute["power_only_baseline"]
    put("power_only_top1", "power-only baseline top-1", base["top1_rate"],
        detail=f"{base['top1_hits']}/{base['members']}; chance {base['chance_mean']}")
    b = base["auc_member_clustered"]
    put("power_only_auc", "power-only baseline AUC (member-clustered)", b and b["mean"], b)
    for claim, token in compute["dispositions"].items():
        put(f"{claim}_disposition", f"{claim} disposition", token)
    return buffer.getvalue()


def findings_md(plan, compute):
    lines = []
    add = lines.append
    t = compute["release_control"]
    ref = compute["release_1000_reference"]
    add("# Issue-95: #94 follow-up controls (release-inert rescoring, within-power ordering) — findings")
    add("")
    add(f"- identity `{IDENTITY}`, plan v1 frozen {plan['frozen_at']}; {plan['chronology']}")
    add(f"- validation command: `{VALIDATION_COMMAND}`; zero engine seconds; every interval DESCRIPTIVE")
    add(f"- release inertness: {read_json(OUTPUT / 'release_inertness.json')['reading']}")
    add("")
    add("## Dispositions (frozen rules) vs stated predictions")
    add("")
    add("| claim | outcome | stated prediction |")
    add("|---|---|---|")
    for claim in ("C26", "C27", "C28"):
        add(f"| {claim} | **{compute['dispositions'][claim]}** | "
            f"{plan['stated_predictions'][claim]['expected_token']} |")
    add("")
    for key, row in compute["readings"].items():
        add(f"- {key} reading: \"{row['reading']}\"")
    add("")
    add("## Release control (release 600 vs #94 release 1000; identical engine actions and verdicts)")
    add("")
    add(f"- pooled top-1: release 600 **{t['top1']['hits']}/{t['top1']['cells']}** vs chance "
        f"{fmt(t['top1']['chance_mean'])}, paired difference "
        f"{fmt_interval(t['top1']['paired_difference_member_clustered'])} (release 1000: "
        f"{ref['top1']['hits']}/{ref['top1']['cells']})")
    for s in MODEL_SYSTEMS:
        b = t["top1"]["per_system"][s]
        add(f"  - {s}: {b['hits']}/{b['cells']}")
    add(f"- AUC_m: release 600 **{fmt_interval(t['auc']['member_clustered'])}** (release 1000: "
        f"{fmt_interval(ref['auc']['member_clustered'])}); cell-unit {fmt_interval(t['auc']['cell_unit'])}")
    for s in MODEL_SYSTEMS:
        add(f"  - {s}: {fmt_interval(t['auc']['per_system'][s]['member_clustered'])}")
    add(f"- chosen ordinals at release 600: {json.dumps(t['chosen_ordinal_histogram'])}; saturated-column "
        f"share {fmt(t['saturated_column_chosen_share']['selector_full_inventory'])}")
    add(f"- rank agreement release 1000 vs 600: {json.dumps(compute['rank_agreement_release_1000_vs_600'], sort_keys=True)}")
    add(f"- within-state structure at release 600, median rho cost vs realized speed within angle: "
        f"{fmt(t['structure']['pooled_within_angle_median_rho'])}")
    add("")
    add("## Power-only control")
    add("")
    for release in ("release_1000", "release_600"):
        w = compute["within_power"][release]
        add(f"- within-power AUC ({release}): member-clustered **{fmt_interval(w['member_clustered'])}** "
            f"over {len(w['members'])} members / {w['cells']} cells; cell-unit {fmt_interval(w['cell_unit'])}")
        for s in MODEL_SYSTEMS:
            add(f"  - {s}: {fmt_interval(w['per_system'][s])}")
    base = compute["power_only_baseline"]
    add(f"- power-only baseline (cost = -expected speed): top-1 {base['top1_hits']}/{base['members']} vs chance "
        f"{fmt(base['chance_mean'])}; AUC_m {fmt_interval(base['auc_member_clustered'])}; within-power AUC 0.5 "
        "by construction")
    add("")
    add("## Claim boundary")
    add("")
    add(plan["claim_boundary"] + ".")
    add("")
    return "\n".join(lines)


def rendered(plan, compute):
    report = {"schema": SCHEMA_REPORT, "identity": IDENTITY, "frozen_at": plan["frozen_at"],
              "validation_command": VALIDATION_COMMAND, "claims": CLAIMS,
              "dispositions": compute["dispositions"], "readings": compute["readings"],
              "stated_predictions": plan["stated_predictions"],
              "release_control": {k: compute["release_control"][k] for k in
                                  ("top1", "auc", "top3", "ceiling", "chosen_ordinal_histogram",
                                   "saturated_column_chosen_share", "verdict_coverage")},
              "within_power": {k: {kk: vv for kk, vv in v.items() if kk != "rows"}
                               for k, v in compute["within_power"].items()},
              "power_only_baseline": {k: v for k, v in compute["power_only_baseline"].items() if k != "rows"},
              "rank_agreement_release_1000_vs_600": compute["rank_agreement_release_1000_vs_600"],
              "compute": compute["compute"], "claim_boundary": plan["claim_boundary"],
              "issue_64_authorized": False}
    return {"summary.json": json_text(report), "findings.md": findings_md(plan, compute),
            "slot_join.csv": slot_join_csv(compute), "comparisons.csv": comparisons_csv(compute)}


def publish(output):
    output = Path(output)
    plan = load_plan(output)
    compute = compute_tables(output, plan)
    write_json(output / "compute.json", compute)
    for name, text in rendered(plan, compute).items():
        (output / name).write_bytes(text.encode())
    log(f"published: {compute['dispositions']}")
    return 0


def validate(output):
    output = Path(output)
    began = time.monotonic()
    plan = load_plan(output)
    fresh = compute_tables(output, plan)
    if read_json(output / "compute.json") != json.loads(json_text(fresh)):
        raise ValueError("compute.json differs from the fresh recomputation")
    for name, text in rendered(plan, fresh).items():
        if (output / name).read_bytes() != text.encode():
            raise ValueError(f"published {name} differs from the recomputation")
    log(f"exact recomputation validation passed: {fresh['structure']} re-derived from the retained "
        f"#94 records and the release-600 decision records; compute.json, summary.json, findings.md, "
        f"slot_join.csv and comparisons.csv byte-compared; {fresh['dispositions']} "
        f"({time.monotonic() - began:.1f}s)")
    return 0


MODES = {"prepare": (prepare, False), "score": (score, True),
         "publish": (publish, False), "validate": (validate, False)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    for mode in MODES:
        modes.add_argument("--" + mode, action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    mode = next(name for name in MODES if getattr(args, name))
    function, locked = MODES[mode]
    try:
        if locked:
            with GPULock():
                return function(args.output)
        return function(args.output)
    except (ValueError, OSError, KeyError) as error:
        log(f"error: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
