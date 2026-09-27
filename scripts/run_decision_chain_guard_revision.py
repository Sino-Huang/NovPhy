"""Issue-99 v2: guard revision of the two v1 publications (terminal version).

Binding runner module: scripts/run_decision_chain_guard_revision.py
Exact validation command: python -u -m scripts.run_decision_chain_guard_revision --validate

The v1 publications (issue-99-decision-chain-attribution-v1, issue-99-slot-encoder-
fix-v1) returned readiness_or_precision_insufficient for every question because one
guard in each failed for a reason unrelated to what the guard protects:

- attribution G2 demanded that each executed shot's position-0 frame be byte-identical
  to the member's sealed anchor image. It protects "every shot starts from the sealed
  physical state". 4/859 frames differ by about 200 rendered pixels while the engine
  state is identical (G3 spread 0.0). Rendering nondeterminism, not a state change.
- fix G3 demanded that R0's parsed-endpoint costs replicate the attribution records
  within 1e-3 on the count cost. It protects "R0 is the same parse as the attribution".
  The count cost weights pig presence by 1000, so a presence difference of 1.34e-4
  (batch-size float noise) read as 0.134.

v2 replaces exactly those two guard definitions, adds nothing else, and re-derives the
tokens from the retained v1 records under the unchanged v1 decision rules. No rollout,
no training, no parsing: zero GPU and engine seconds. v1 stays byte-identical and is
re-validated by its own --validate at every v2 publish and validate.

v2 was frozen AFTER every v1 outcome was known; that is disclosed in plan.json and in
every rendering. The v2 guards are defined from the failure diagnoses and the tolerance
the v1 smoke froze before any outcome (PARSE_BATCH_TOLERANCE = 1e-3), not from any AUC.
v2 is terminal: no further guard revision of #99.

Modes: --dry-run (no write, no statistic) --prepare --publish --validate.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import time

from scripts import run_decision_chain_attribution as m99
from scripts import run_launch_power_probe as p94
from scripts import run_slot_encoder_fix as fix
from scripts import run_tau_ad_within_checkpoint as t96

ROOT = t96.ROOT
OUTPUT = ROOT / ".local-artifacts/issue-99-guard-revision-v2"
IDENTITY = "issue-99-guard-revision-v2"
SCHEMA_PLAN = "issue_99_guard_revision_plan_v2"
SCHEMA_COMPUTE = "issue_99_guard_revision_compute_v2"
SCHEMA_REPORT = "issue_99_guard_revision_report_v2"
VALIDATION_COMMAND = "python -u -m scripts.run_decision_chain_guard_revision --validate"
RUNNER = "scripts/run_decision_chain_guard_revision.py"

STOP_TOKEN = m99.STOP_TOKEN
TOLERANCE = m99.PARSE_BATCH_TOLERANCE
ENGINE_TOLERANCE = m99.ENGINE_ANCHOR_TOLERANCE
PUBLISHED = ("plan.json", "smoke.json", "ledger.json", "compute.json", "summary.json", "findings.md",
             "comparisons.csv")
SOURCES = {"attribution": m99.OUTPUT, "fix": fix.OUTPUT}
RUNNERS = {"attribution": "scripts/run_decision_chain_attribution.py", "fix": "scripts/run_slot_encoder_fix.py",
           "model_module": fix.MODEL_MODULE}

read_json = t96.read_json
write_json = t96.write_json
json_text = t96.json_text
sha256_of = t96.sha256_of
fmt = t96.fmt
fmt_interval = t96.fmt_interval

GUARD_CHANGES = {
    "attribution.G2": {
        "v1": "every executed shot's position-0 agent frame is byte-identical to the member's sealed anchor",
        "v2": ("every member has at least one verdict-bearing shot whose position-0 agent frame is "
               "byte-identical to its sealed anchor, and (v1 G3, unchanged) every shot of the member has "
               f"the same engine-projected position-0 carrier within {ENGINE_TOLERANCE}; together every "
               "shot starts from the sealed anchor's physical state"),
        "reason": ("the guard protects the physical start state; v1 failed on 4/859 frames that differ by "
                   "196-207 rendered pixels while the engine state is identical (spread 0.0). Byte-"
                   "mismatched frames and their parsed position-0 deltas stay reported descriptively"),
    },
    "fix.G3": {
        "v1": "R0 parsed-endpoint costs (stage ap) replicate the attribution records within 1e-3 (count cost)",
        "v2": (f"R0 parsed presence of every vocabulary slot at positions 0 and 225, the pig displacement "
               f"and the tie-free cost replicate the attribution evidence within {TOLERANCE} "
               "(PARSE_BATCH_TOLERANCE, frozen in the v1 attribution smoke before any outcome)"),
        "reason": ("the guard protects 'R0 is the same parse'; the count cost multiplies pig presence by "
                   "1000, so v1 read a 1.34e-4 batch-size float difference as 0.134"),
    },
}


def log(message):
    print(f"[issue-99-guard-revision] {message}", flush=True)


def utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def bindings():
    rows = []
    for name, root in SOURCES.items():
        for file in PUBLISHED:
            path = Path(root) / file
            rows.append({"name": f"{name}/{file}", "artifact": str(path.relative_to(ROOT)),
                         "sha256": sha256_of(path)})
    for name, path in RUNNERS.items():
        rows.append({"name": f"runner/{name}", "artifact": path, "sha256": sha256_of(ROOT / path)})
    return rows


# ---------------------------------------------------------------------------
# revised guards (pure functions of retained v1 evidence)
# ---------------------------------------------------------------------------

def anchored_start_identity(frames, anchors, spread):
    """frames: member -> {shot: frame0 sha256}; anchors: member -> sealed sha256;
    spread: member -> max abs engine position-0 carrier difference over its shots."""
    members, mismatched = {}, []
    for member in sorted(frames):
        identical = sorted(s for s, sha in frames[member].items() if sha == anchors[member])
        mismatched.extend(f"{s}" for s, sha in sorted(frames[member].items()) if sha != anchors[member])
        members[member] = {"shots": len(frames[member]), "byte_identical": len(identical),
                           "engine_spread": spread[member],
                           "ok": bool(identical) and spread[member] <= ENGINE_TOLERANCE}
    return {"members": members, "byte_mismatched_shots": mismatched,
            "pass": all(row["ok"] for row in members.values())}


def parse_replication(pairs):
    """pairs: (R0 perception shot, R0 ap row, attribution evidence presences, attribution ap row)."""
    worst = {"presence": 0.0, "pig_displacement": 0.0, "tie_free": 0.0}
    for shot, row, reference, reference_row in pairs:
        for when in ("start", "end"):
            for mine, theirs in zip(shot[f"parsed_presence_{when}"], reference[when], strict=True):
                worst["presence"] = max(worst["presence"], abs(mine - theirs))
        for key in ("pig_displacement", "tie_free"):
            worst[key] = max(worst[key], abs(row[key] - reference_row[key]))
    return {"shots": len(pairs), "max_abs_delta": worst, "tolerance": TOLERANCE,
            "pass": all(value <= TOLERANCE for value in worst.values())}


def attribution_g2(universe):
    engine = read_json(m99.OUTPUT / "engine_anchors.json")["members"]
    frames, first_identical = {}, {}
    anchors = {}
    for inventory, member in m99.scheduled_members(universe):
        anchors.setdefault(member, p94.frozen_anchor(member)["sha256"])
        record = m99.read_record(m99.endpoint_path(m99.OUTPUT, inventory, member), m99.SCHEMA_ENDPOINT)
        for ordinal, shot in record["shots"].items():
            key = f"{inventory}/{member}/o{int(ordinal):02d}"
            frames.setdefault(member, {})[key] = shot["frame0_sha256"]
            if shot["frame0_sha256"] == anchors[member]:
                first_identical.setdefault(member, (inventory, int(ordinal)))
    result = anchored_start_identity(frames, anchors, {m: engine[m]["max_abs_spread"] for m in frames})
    # descriptive: parsed position-0 carrier of each mismatched shot against a byte-identical shot's
    deltas = []
    for key in result["byte_mismatched_shots"]:
        inventory, member, ordinal = key.split("/")
        mine = m99.load_evidence(m99.OUTPUT, inventory, member, int(ordinal[1:]))["parsed"][0]
        anchor_inventory, anchor_ordinal = first_identical[member]
        reference = m99.load_evidence(m99.OUTPUT, anchor_inventory, member, anchor_ordinal)["parsed"][0]
        pig = 2 + 13 * 8
        deltas.append({"shot": key, "max_abs_carrier_delta": float((mine - reference).abs().max()),
                       "pig_presence_delta": float(mine[pig] - reference[pig])})
    result["mismatched_parsed_position0"] = deltas
    return result


def fix_g3(sources, universe):
    pairs = []
    for inventory in m99.INVENTORIES:
        for member in sorted(m for m in universe[inventory] if m in m99.HELD_OUT):
            record = fix.read_record(fix.perception_path(fix.OUTPUT, "R0", inventory, member),
                                     fix.SCHEMA_PERCEPTION)
            reference_rows = m99.read_record(m99.endpoint_path(m99.OUTPUT, inventory, member),
                                             m99.SCHEMA_ENDPOINT)["stages"]["ap"]
            rows = {row["ordinal"]: row for row in record["ap_rows"]}
            for reference_row in reference_rows:
                ordinal = reference_row["ordinal"]
                evidence = m99.load_evidence(m99.OUTPUT, inventory, member, ordinal)
                reference = {"start": [float(v) for v in evidence["parsed"][0][2::13][:18]],
                             "end": [float(v) for v in evidence["parsed"][m99.ENDPOINT][2::13][:18]]}
                pairs.append((record["shots"][str(ordinal)], rows[ordinal], reference, reference_row))
    return parse_replication(pairs)


# ---------------------------------------------------------------------------
# tokens under the unchanged v1 decision rules
# ---------------------------------------------------------------------------

def attribution_tokens(v1, guards_ok):
    q1 = m99.token_q1(v1["Q1_attribution"]["gate_a"], guards_ok)
    shares = v1["Q2_tie_free_cost"]["tied_share_tie_free"]
    q2 = (STOP_TOKEN if not guards_ok else "supported"
          if all(v < m99.TIE_SHARE_TARGET for v in shares.values()) else "not_supported_by_this_experiment")
    return {"Q1_attribution": q1, "Q2_tie_free_cost": q2}


def fix_tokens(v1, guards_ok):
    met = v1["Q3_target"]["met_by"]
    q3 = STOP_TOKEN if not guards_ok else ("supported" if met else "not_supported_by_this_experiment")
    effect = v1["Q4_encoder_effect"]["estimate"]
    if not guards_ok or effect is None:
        q4 = STOP_TOKEN
    elif effect["mean"] >= fix.DELTA and effect["interval"][0] > 0:
        q4 = "supported"
    elif effect["interval"][1] < fix.DELTA:
        q4 = "not_supported_by_this_experiment"
    else:
        q4 = STOP_TOKEN
    return {"Q3_target": q3, "Q4_encoder_effect": q4}


def compute(plan):
    for entry in plan["inputs"]:
        if sha256_of(ROOT / entry["artifact"]) != entry["sha256"]:
            raise ValueError(f"v1 input {entry['name']} changed after the v2 freeze")
    if m99.validate(m99.OUTPUT) != 0 or fix.validate(fix.OUTPUT) != 0:
        raise ValueError("a v1 publication no longer validates")
    sources = t96.load_sources()
    _, universe, _ = t96.build_universe(sources)
    v1a = read_json(m99.OUTPUT / "compute.json")
    v1f = read_json(fix.OUTPUT / "compute.json")
    g2 = attribution_g2(universe)
    g3 = fix_g3(sources, universe)
    guards_a = {name: {"v1_pass": block["pass"], "v2_pass": block["pass"], "revised": False}
                for name, block in v1a["guards"].items()}
    guards_a["G2"] = {"v1_pass": v1a["guards"]["G2"]["pass"], "v2_pass": g2["pass"], "revised": True,
                      "v2_observed": g2}
    guards_f = {name: {"v1_pass": block["pass"], "v2_pass": block["pass"], "revised": False}
                for name, block in v1f["guards"].items()}
    guards_f["G3"] = {"v1_pass": v1f["guards"]["G3"]["pass"], "v2_pass": g3["pass"], "revised": True,
                      "v2_observed": g3}
    ok_a = all(g["v2_pass"] for g in guards_a.values())
    ok_f = all(g["v2_pass"] for g in guards_f.values())
    tokens_a, tokens_f = attribution_tokens(v1a, ok_a), fix_tokens(v1f, ok_f)
    gate = v1a["Q1_attribution"]["gate_a"]
    ledger = read_json(m99.OUTPUT / "ledger.json")
    return {
        "schema": SCHEMA_COMPUTE, "identity": IDENTITY,
        "attribution": {"guards": guards_a, "guards_pass": ok_a,
                        "tokens": {"v1": {"Q1_attribution": v1a["Q1_attribution"]["token"],
                                          "Q2_tie_free_cost": v1a["Q2_tie_free_cost"]["token"]},
                                   "v2": tokens_a},
                        "gate_a": gate,
                        "tied_share_tie_free": v1a["Q2_tie_free_cost"]["tied_share_tie_free"],
                        "extraction_wall_seconds": ledger["extraction_seconds_elapsed"]},
        "fix": {"guards": guards_f, "guards_pass": ok_f,
                "tokens": {"v1": {"Q3_target": v1f["Q3_target"]["token"],
                                  "Q4_encoder_effect": v1f["Q4_encoder_effect"]["token"]},
                           "v2": tokens_f},
                "target_requests": v1f["Q3_target"]["requests"], "met_by": v1f["Q3_target"]["met_by"],
                "reference_arms_meeting_target": v1f["Q3_target"]["reference_arms_meeting_target"],
                "encoder_effect": v1f["Q4_encoder_effect"]["estimate"],
                "e_minus_current_grid": v1f["inventories"][m99.PRIMARY]["contrasts"][m99.PRIMARY_COST][
                    "E-current"]["request_mean"]["estimate"],
                "e_count_cost_tied_share": {name.split(":", 1)[1]: block["tied_share"]
                                            for name, block in v1f["inventories"][m99.PRIMARY]["rankers"][
                                                "count"].items() if name.startswith("E:")}},
        "gpu_seconds": 0, "engine_seconds": 0,
    }


# ---------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------

def frozen_plan(frozen_at):
    a = read_json(m99.OUTPUT / "plan.json")
    f = read_json(fix.OUTPUT / "plan.json")
    return {
        "schema": SCHEMA_PLAN, "identity": IDENTITY, "version": 2, "role": "terminal", "issue": 99,
        "frozen_at": frozen_at, "frozen_after_v1_publication": True, "issue_64_authorized": False,
        "revision_of": ["issue-99-decision-chain-attribution-v1", "issue-99-slot-encoder-fix-v1"],
        "validation_command": VALIDATION_COMMAND, "runner": RUNNER,
        "runner_sha256_at_freeze": sha256_of(ROOT / RUNNER), "gpu_seconds": 0, "engine_seconds": 0,
        "disclosure": ("v2 was frozen after both v1 publications, with every v1 outcome known (all AUCs, "
                       "contrasts, the Gate A result and the counterfactual token readings written in the "
                       "v1 record and the #99 comment). The v2 guard definitions come from the diagnoses "
                       "of the two v1 guard failures and from the parse tolerance the v1 attribution smoke "
                       "froze before any outcome; no AUC or contrast entered their definition. v2 changes "
                       "no estimand, cost, threshold, cohort, record or decision rule, and v1 is retained "
                       "byte-identical (sha256-bound below) under the #87 versioned-protocol practice."),
        "changes": GUARD_CHANGES,
        "reporting_correction": ("v1 attribution compute.json compute.extraction.wall_seconds is an "
                                 "unpopulated field (0.0); v2 reports the extraction wall from the v1 "
                                 "ledger (extraction_seconds_elapsed)"),
        "unchanged": {"attribution_decision_rule": a["decision_rule"], "attribution_guards": a["guards"],
                      "fix_decision_rule": f["decision_rule"], "fix_guards": f["guards"],
                      "records": "every v1 record, evidence tensor and checkpoint, read-only"},
        "token_rule": ("each question's v2 token is its frozen v1 rule evaluated with the v2 guard set; a "
                       "failing v2 guard still forces readiness_or_precision_insufficient"),
        "terminal": "v2 is the final guard revision of #99; no v3",
        "inputs": bindings(),
        "claim_boundary": (a["claim_boundary"] + "; " + f["claim_boundary"]),
    }


def load_plan(output):
    path = Path(output) / "plan.json"
    if not path.is_file():
        raise ValueError("plan.json missing; run --prepare first")
    plan = read_json(path)
    if plan.get("schema") != SCHEMA_PLAN or plan.get("identity") != IDENTITY:
        raise ValueError("plan.json is not the issue-99 v2 protocol")
    blob = json_text(plan)
    for marker in m99.PLACEHOLDER_MARKERS:
        if marker in blob:
            raise ValueError(f"frozen plan contains a missing-value marker {marker!r}")
    return plan


def prepare(output):
    output = Path(output)
    if (output / "plan.json").is_file():
        load_plan(output)
        log("existing frozen v2 plan validated")
        return 0
    for name in ("compute.json", "summary.json", "findings.md"):
        if (output / name).exists():
            raise ValueError("v2 publications exist before the freeze; refusing")
    output.mkdir(parents=True, exist_ok=True)
    plan = frozen_plan(utc_now())
    (output / "plan.json").write_text(json_text(plan))
    load_plan(output)
    log(f"frozen v2 plan published: {len(plan['inputs'])} v1 artifacts sha256-bound; no statistic computed")
    return 0


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------

def guard_rows(block):
    return [(name, g["v1_pass"], g["v2_pass"], g["revised"]) for name, g in block["guards"].items()]


def findings_md(plan, c):
    a, f = c["attribution"], c["fix"]
    g2, g3 = a["guards"]["G2"]["v2_observed"], f["guards"]["G3"]["v2_observed"]
    lines = []
    add = lines.append
    add("# Issue-99 v2: guard revision of the attribution and slot-encoder publications — findings")
    add("")
    add(f"- identity `{IDENTITY}`, plan v2 frozen {plan['frozen_at']} **after** both v1 publications")
    add(f"- validation command: `{VALIDATION_COMMAND}`; 0 GPU and 0 engine seconds; every number below "
        "except the two revised guards is read from the sha256-bound v1 publications")
    add(f"- disclosure: {plan['disclosure']}")
    add("")
    add("## Revised guards")
    add("")
    add("| guard | v1 definition | v2 definition | reason |")
    add("|---|---|---|---|")
    for name, change in plan["changes"].items():
        add(f"| {name} | {change['v1']} | {change['v2']} | {change['reason']} |")
    add("")
    add("| publication | guard | v1 pass | v2 pass | revised |")
    add("|---|---|---|---|---|")
    for label, block in (("attribution", a), ("fix", f)):
        for name, v1, v2, revised in guard_rows(block):
            add(f"| {label} | {name} | {v1} | {v2} | {revised} |")
    add("")
    add(f"- attribution G2 v2: {sum(r['byte_identical'] > 0 for r in g2['members'].values())}/"
        f"{len(g2['members'])} members have a byte-identical shot; engine spread max "
        f"{max(r['engine_spread'] for r in g2['members'].values())}; byte-mismatched shots "
        f"{len(g2['byte_mismatched_shots'])}: "
        + "; ".join(f"{d['shot']} parsed position-0 max |delta| {d['max_abs_carrier_delta']:.2e}, pig presence "
                    f"delta {d['pig_presence_delta']:+.2e}" for d in g2["mismatched_parsed_position0"]))
    add(f"- fix G3 v2: {g3['shots']} held-out shots; max |delta| presence "
        f"{g3['max_abs_delta']['presence']:.2e}, pig displacement {g3['max_abs_delta']['pig_displacement']:.2e}, "
        f"tie-free {g3['max_abs_delta']['tie_free']:.2e} (tolerance {g3['tolerance']})")
    add("")
    add("## Dispositions (v1 -> v2, unchanged v1 decision rules)")
    add("")
    add("| question | v1 token | v2 token |")
    add("|---|---|---|")
    for block in (a, f):
        for q, v2 in block["tokens"]["v2"].items():
            add(f"| {q} | {block['tokens']['v1'][q]} | **{v2}** |")
    add("")
    gate = a["gate_a"]
    add(f"- Q1: Gate A dominant stage {gate['dominant']}; drops "
        + ", ".join(f"{k} {fmt(v['mean'])} [{fmt(v['interval'][0])}, {fmt(v['interval'][1])}]"
                    for k, v in gate["drops"].items()))
    add(f"- Q2: stage-c tied-cell share on mixed grid cells under the tie-free cost, max over 14 requests "
        f"{max(a['tied_share_tie_free'].values()):.3f} (target < {m99.TIE_SHARE_TARGET})")
    rows = {r["request"]: r for r in f["target_requests"]}
    add(f"- Q3: arm E requests meeting AUC >= {fix.TARGET_AUC} with lower > {fix.TARGET_LOWER} on held-out "
        f"grid: {', '.join(f['met_by']) or 'none'} ({len(f['met_by'])}/{len(rows)}): "
        + "; ".join(f"{r} {fmt_interval(rows[r]['auc'])}" for r in f["met_by"]))
    add(f"- Q4: request-mean E - R0 on held-out grid {fmt_interval(f['encoder_effect'])} (margin {fix.DELTA})")
    add("")
    add("## Reading limits (carried from v1, not rules)")
    add("")
    add(f"- Q3 rests on {len(f['met_by'])} of 12 requests over 3 held-out member clusters (10 distinct bootstrap "
        "resamples); no multiplicity adjustment was declared")
    for r in f["met_by"]:
        add(f"- under the count cost E:{r} is tied in {f['e_count_cost_tied_share'][r]:.3f} of held-out grid "
            "cells: its ranking comes only from the tie-free cost's continuous residue")
    add(f"- the target is not evidence that the fix helps: E - current (request-mean, held-out grid) "
        f"{fmt_interval(f['e_minus_current_grid'])}; the current #77 checkpoints meet the target on "
        f"{sum(x.startswith('current:') for x in f['reference_arms_meeting_target'])} requests without any fix")
    add(f"- attribution extraction wall {a['extraction_wall_seconds']:.1f} s ({plan['reporting_correction']})")
    add("")
    add("## Bound v1 artifacts")
    add("")
    add("| artifact | sha256 |")
    add("|---|---|")
    for entry in plan["inputs"]:
        add(f"| `{entry['artifact']}` | `{entry['sha256']}` |")
    add("")
    add("## Claim boundary")
    add("")
    add(plan["claim_boundary"] + "; v2 revises two guards only.")
    add("")
    return "\n".join(lines)


def comparisons_csv(c):
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["id", "publication", "statistic", "v1", "v2", "detail"])
    for label in ("attribution", "fix"):
        block = c[label]
        for name, v1, v2, revised in guard_rows(block):
            detail = json.dumps(block["guards"][name].get("v2_observed", {}), sort_keys=True) if revised else ""
            writer.writerow([f"{label}_guard_{name}", label, f"{name} pass", v1, v2, detail])
        for q, v2 in block["tokens"]["v2"].items():
            writer.writerow([f"{label}_{q}", label, f"{q} token", block["tokens"]["v1"][q], v2, ""])
    return buffer.getvalue()


def summary(plan, c):
    return {"schema": SCHEMA_REPORT, "identity": IDENTITY, "frozen_at": plan["frozen_at"],
            "frozen_after_v1_publication": True, "validation_command": VALIDATION_COMMAND,
            "disclosure": plan["disclosure"], "changes": plan["changes"],
            "attribution": c["attribution"], "fix": c["fix"], "gpu_seconds": 0, "engine_seconds": 0,
            "claim_boundary": plan["claim_boundary"], "issue_64_authorized": False}


def rendered(plan, c):
    return {"summary.json": json_text(summary(plan, c)), "findings.md": findings_md(plan, c),
            "comparisons.csv": comparisons_csv(c)}


def publish(output):
    output = Path(output)
    plan = load_plan(output)
    c = compute(plan)
    write_json(output / "compute.json", c)
    for name, text in rendered(plan, c).items():
        (output / name).write_bytes(text.encode())
    log(f"published v2 tokens: attribution {c['attribution']['tokens']['v2']}; fix {c['fix']['tokens']['v2']}")
    return 0


def validate(output):
    output = Path(output)
    began = time.monotonic()
    plan = load_plan(output)
    fresh = compute(plan)
    if read_json(output / "compute.json") != json.loads(json_text(fresh)):
        raise ValueError("compute.json differs from the fresh recomputation")
    for name, text in rendered(plan, fresh).items():
        if (output / name).read_bytes() != text.encode():
            raise ValueError(f"published {name} differs from the recomputation")
    log(f"v2 validation passed: v1 bindings, both v1 --validate, revised guards and tokens re-derived "
        f"and byte-compared ({time.monotonic() - began:.1f}s)")
    return 0


def dry_run(output):
    log(f"dry run (no write, no statistic); output root {Path(output)}")
    for entry in bindings():
        log(f"  bind {entry['artifact']} {entry['sha256'][:16]}")
    for name, change in GUARD_CHANGES.items():
        log(f"  revise {name}: {change['v2']}")
    log(f"  plan present: {(Path(output) / 'plan.json').is_file()}")
    return 0


MODES = {"dry-run": dry_run, "prepare": prepare, "publish": publish, "validate": validate}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    modes = parser.add_mutually_exclusive_group(required=True)
    for mode in MODES:
        modes.add_argument("--" + mode, action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    mode = next(name for name in MODES if getattr(args, name.replace("-", "_")))
    try:
        return MODES[mode](args.output)
    except (ValueError, OSError, KeyError) as error:
        log(f"error: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
