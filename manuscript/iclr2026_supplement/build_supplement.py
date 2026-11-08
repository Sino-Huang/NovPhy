"""Package existing per-unit outcome records into anonymized supplementary CSV files.

Reads only retained artifacts under the NovPhy repository's `.local-artifacts/`
(relative paths); runs no model, engine or new statistic. Every output row is a
transcription of a retained record; `checks.csv` re-derives each published
aggregate from the packaged rows and compares it with the published value.

Run from anywhere:  python build_supplement.py
"""
from __future__ import annotations

from collections import defaultdict
import csv
import io
import json
from pathlib import Path
import re

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
A96 = ".local-artifacts/issue-96-tau-ad-within-checkpoint-v1"
A93 = ".local-artifacts/issue-93-second-parameterization-v1"
A94 = ".local-artifacts/issue-94-launch-power-v1"
A87 = ".local-artifacts/issue-87-closed-loop-oracle-v1"
A89 = ".local-artifacts/issue-89-oracle-completion-v1"
A102 = ".local-artifacts/issue-102-closed-loop-v1"
A103 = ".local-artifacts/issue-103-engine-truth-frontier-v1"
A106 = ".local-artifacts/issue-106-pixel-benchmarks-v1"
A108 = ".local-artifacts/issue-108-gate-b-v1"
A101 = ".local-artifacts/issue-101-decision-aware-policy-v1"
A112 = ".local-artifacts/issue-112-v3-retrain-v1"
A113 = ".local-artifacts/issue-113-macro-predicates-v1"
A104P = ".local-artifacts/issue-104-power-cohort-v1"
SEEDS = (20260908, 20260909, 20260910)

INVENTORIES = ("grid", "offset", "angle", "power")
PREFIX93 = {"grid": "g", "offset": "o"}
PREFIX94 = "p"
BOOTSTRAP_SEED = 7201
BOOTSTRAP_DRAWS = 10000
INTERVAL_QUANTILES = (0.025, 0.975)
TOL = 1e-12


def read(rel):
    return json.loads((REPO / rel).read_text())


def exists(rel):
    return (REPO / rel).is_file()


# ---------------------------------------------------------------------------
# development verdict tables (#93 grid/offset, #94 power, #87+#89 angle)
# ---------------------------------------------------------------------------

def oracle_record(root, identity, schema):
    """Retained oracle record; a completion-pass record replaces a failed original."""
    rel = f"{root}/records/{identity}.json"
    if not exists(rel):
        return None, None
    record = read(rel)
    if record.get("schema") != schema:
        raise ValueError(f"{rel}: unexpected schema")
    source = rel
    completion = f"{root}/records-completion/{identity}.json"
    if record["outcome"] is None and exists(completion):
        replacement = read(completion)
        if replacement["outcome"] is not None:
            record, source = replacement, completion
    return record, source


def angle_union():
    """#87 oracle records, then #89 completion records override (keyed by state, ordinal)."""
    union = {}
    for path in sorted((REPO / A87 / "records").glob("*.json")):
        record = json.loads(path.read_text())
        if record.get("schema") != "issue_87_closed_loop_oracle_record_v1":
            continue
        key = (record["cell"]["state"], record["cell"]["ordinal"])
        rel = f"{A87}/records/{path.name}"
        if record.get("outcome") is not None:
            union[key] = {"verdict": bool(record["outcome"]["first_shot_success"]),
                          "failure_kind": None, "failure": None, "source": rel}
        else:
            union.setdefault(key, {"verdict": None})
            if union[key]["verdict"] is None:
                union[key] = {"verdict": None, "failure_kind": record.get("failure_kind"),
                              "failure": record.get("failure"), "source": rel}
    for path in sorted((REPO / A89 / "records").glob("*.json")):
        record = json.loads(path.read_text())
        if record.get("schema") != "issue_89_oracle_completion_record_v1":
            raise ValueError(f"{path.name}: unexpected schema")
        key = (record["cell"]["state"], record["cell"]["ordinal"])
        rel = f"{A89}/records/{path.name}"
        if record.get("outcome") is not None:
            union[key] = {"verdict": bool(record["outcome"]["first_shot_success"]),
                          "failure_kind": None, "failure": None, "source": rel}
        elif union.get(key, {}).get("verdict") is None:
            union[key] = {"verdict": None, "failure_kind": record.get("failure_kind"),
                          "failure": record.get("failure"), "source": rel}
    return union


def build_universe():
    plan93, plan94, plan87 = read(f"{A93}/plan.json"), read(f"{A94}/plan.json"), read(f"{A87}/plan.json")
    per89 = read(f"{A89}/summary.json")["ceiling_reestimate"]["per_state"]
    states87 = {state["identity"]: state for state in plan87["states"]}
    union = angle_union()
    universe = {inventory: {} for inventory in INVENTORIES}
    slots = []  # one row per (inventory, member, candidate)
    roles = {}

    def add(inventory, member, state, items, lookup, in_universe):
        verdicts = {}
        for item in items:
            info = lookup(item)
            verdict = info["verdict"] if info else None
            if verdict is not None:
                verdicts[item["ordinal"]] = verdict
            action = item["action"]
            slots.append({
                "inventory": inventory, "level": member, "state": state,
                "ordinal": item["ordinal"], "branch_identity": item["branch_identity"],
                "drag_x": action["drag_x"], "drag_y": action["drag_y"],
                "release_time_ms": action["release_time_ms"], "tap_time_ms": action["tap_time_ms"],
                "verdict": "" if verdict is None else int(verdict),
                "typed_failure": "" if verdict is not None else
                (info["failure_kind"] or "unresolved") if info else "no_record",
                "typed_failure_detail": "" if verdict is not None or not info else (info["failure"] or ""),
                "verdict_record": info["source"] if info else "",
                "in_issue96_universe": int(in_universe)})
        if in_universe:
            universe[inventory][member] = {"state": state, "verdicts": verdicts}

    for state in plan93["states"]:
        member, canonical = state["member"], state["canonical_state"]
        roles[member] = state["exposure_role"]
        for arm in ("grid", "offset"):
            def lookup93(item, arm=arm, member=member):
                record, source = oracle_record(
                    A93, f"oracle--{member}--{PREFIX93[arm]}{item['ordinal']:02d}", "issue_93_oracle_record_v1")
                if record is None:
                    return None
                ok = record["outcome"] is not None
                return {"verdict": bool(record["outcome"]["first_shot_success"]) if ok else None,
                        "failure_kind": record.get("failure_kind"), "failure": record.get("failure"),
                        "source": source}
            add(arm, member, canonical, plan93["inventories"][arm]["members"][member], lookup93, True)

        def lookup94(item, member=member):
            record, source = oracle_record(
                A94, f"oracle--{member}--{PREFIX94}{item['ordinal']:02d}", "issue_94_oracle_record_v1")
            if record is None:
                return None
            ok = record["outcome"] is not None
            return {"verdict": bool(record["outcome"]["first_shot_success"]) if ok else None,
                    "failure_kind": record.get("failure_kind"), "failure": record.get("failure"),
                    "source": source}
        add("power", member, canonical, plan94["inventory"]["members"][member], lookup94, True)

        unmeasurable = bool(per89[canonical]["typed_unmeasurable"])
        add("angle", member, canonical, states87[canonical]["inventory"],
            lambda item, canonical=canonical: union.get((canonical, item["ordinal"])), not unmeasurable)
    return universe, slots, roles


# ---------------------------------------------------------------------------
# #96 development cells
# ---------------------------------------------------------------------------

def cell_auc(pos, neg):
    """P(successful candidate has lower predicted cost than failed candidate); ties 0.5."""
    score = sum(1.0 if p < n else 0.5 if p == n else 0.0 for p in pos for n in neg)
    return score / (len(pos) * len(neg))


def cell_rows(universe, roles):
    plan = read(f"{A96}/plan.json")
    arms, seeds = plan["universe"]["arms"], plan["universe"]["seeds"]
    rows = []
    for inventory in INVENTORIES:
        for member in sorted(universe[inventory]):
            entry = universe[inventory][member]
            verdicts = entry["verdicts"]
            mixed = set(verdicts.values()) == {True, False}
            for seed in seeds:
                for arm in arms:
                    identity = f"decision--{inventory}--{member}--seed{seed}--{arm}"
                    rel = f"{A96}/records/{identity}.json"
                    record = read(rel)
                    if (record.get("schema") != "issue_96_tau_ad_within_checkpoint_decision_v1"
                            or record["cell"]["identity"] != identity):
                        raise ValueError(f"{rel}: binding differs")
                    decision = record["decision"]
                    costs = {r["ordinal"]: r["predicted_cost"] for r in decision["ranking"]}
                    admissible = sorted(((o, costs[o]) for o in verdicts if costs.get(o) is not None),
                                        key=lambda pair: (pair[1], pair[0]))
                    pos = [c for o, c in admissible if verdicts[o]]
                    neg = [c for o, c in admissible if not verdicts[o]]
                    chosen = decision["chosen_ordinal"]
                    rows.append({
                        "inventory": inventory, "level": member, "state": entry["state"],
                        "exposure_role": roles[member], "seed": seed, "arm": arm,
                        "checkpoint_family": record["cell"]["family"],
                        "candidates": decision["candidate_count"],
                        "verdict_slots": len(verdicts),
                        "verdict_successes": sum(verdicts.values()),
                        "verdict_failures": len(verdicts) - sum(verdicts.values()),
                        "nonfinite_predictions": sum(1 for o in verdicts if o in costs and costs[o] is None),
                        "admissible": len(admissible),
                        "mixed_level": int(mixed),
                        "cell_auc": cell_auc(pos, neg) if pos and neg else None,
                        "top1_hit": (int(bool(verdicts[admissible[0][0]])) if admissible else None),
                        "chance_top1": len(pos) / len(admissible) if admissible else None,
                        "all_tied": int(len({c for _, c in admissible}) <= 1),
                        "chosen_ordinal": chosen,
                        "chosen_verdict": (None if chosen is None or chosen not in verdicts
                                           else int(verdicts[chosen])),
                        "decision_failure": decision["failure"] or "",
                        "source_record": rel,
                    })
    return rows


# ---------------------------------------------------------------------------
# re-derivation of the published aggregates
# ---------------------------------------------------------------------------

def bootstrap(values, clusters):
    grouped = defaultdict(list)
    for value, cluster in zip(values, clusters, strict=True):
        grouped[cluster].append(value)
    means = np.array([np.mean(grouped[key]) for key in sorted(grouped)])
    generator = np.random.default_rng(np.random.PCG64(BOOTSTRAP_SEED))
    draws = generator.integers(0, len(means), size=(BOOTSTRAP_DRAWS, len(means)))
    low, high = np.quantile(means[draws].mean(axis=1), INTERVAL_QUANTILES)
    return {"mean": float(means.mean()), "interval": [float(low), float(high)],
            "clusters": len(means), "units": len(values)}


def checks_96(rows):
    summary = read(f"{A96}/summary.json")
    compute = read(f"{A96}/compute.json")
    out = []

    def check(quantity, population, published, recomputed, source):
        if isinstance(published, (int, float)) and isinstance(recomputed, (int, float)) \
                and not isinstance(published, bool):
            match = abs(float(published) - float(recomputed)) <= TOL
        else:
            match = published == recomputed
        out.append({"file": "dev_issue96_cells.csv", "quantity": quantity, "population": population,
                    "published_source": source, "published_value": published,
                    "recomputed_value": recomputed, "match": int(match)})

    by_cell = defaultdict(dict)
    for row in rows:
        if row["mixed_level"]:
            by_cell[(row["inventory"], row["level"], row["seed"])][row["arm"]] = row
    for inventory in INVENTORIES:
        block = summary["inventories"][inventory]
        cells = sorted(k for k in by_cell if k[0] == inventory)
        src = f"{A96}/summary.json .inventories.{inventory}"
        check(f"{inventory}: mixed cells", "mixed levels x 3 seeds", block["mixed_cells"], len(cells), src)
        check(f"{inventory}: mixed levels", "mixed levels", block["mixed_members"],
              sorted({k[1] for k in cells}), src)
        for arm, published in block["arm_auc"].items():
            vals = [(by_cell[k][arm]["cell_auc"], k[1]) for k in cells if by_cell[k][arm]["cell_auc"] is not None]
            boot = bootstrap([v for v, _ in vals], [m for _, m in vals])
            pop = f"{inventory} mixed cells, level-clustered mean"
            check(f"{inventory}/{arm}: AUC mean", pop, published["mean"], boot["mean"], src + f".arm_auc.{arm}")
            check(f"{inventory}/{arm}: AUC 95% interval low", pop, published["interval"][0], boot["interval"][0],
                  src + f".arm_auc.{arm}")
            check(f"{inventory}/{arm}: AUC 95% interval high", pop, published["interval"][1], boot["interval"][1],
                  src + f".arm_auc.{arm}")
            check(f"{inventory}/{arm}: AUC cells", pop, published["units"], boot["units"], src + f".arm_auc.{arm}")
            check(f"{inventory}/{arm}: AUC levels", pop, published["clusters"], boot["clusters"],
                  src + f".arm_auc.{arm}")
        for name, published in block["contrasts"].items():
            first, second = published["first"], published["second"]
            vals, clus, undefined = [], [], 0
            for k in cells:
                a, b = by_cell[k][first]["cell_auc"], by_cell[k][second]["cell_auc"]
                if a is None or b is None:
                    undefined += 1
                    continue
                vals.append(a - b)
                clus.append(k[1])
            boot = bootstrap(vals, clus)
            pop = f"{inventory} mixed cells, paired {first} - {second}"
            s = src + f".contrasts.{name}"
            check(f"{inventory}/{name}: mean", pop, published["estimate"]["mean"], boot["mean"], s)
            check(f"{inventory}/{name}: interval low", pop, published["estimate"]["interval"][0],
                  boot["interval"][0], s)
            check(f"{inventory}/{name}: interval high", pop, published["estimate"]["interval"][1],
                  boot["interval"][1], s)
            check(f"{inventory}/{name}: cells", pop, published["cells"], len(vals), s)
            check(f"{inventory}/{name}: undefined cells", pop, published["undefined_cells"], undefined, s)
        top1 = block["top1"]
        f_star = block["F_star"]
        s = src + ".top1"
        check(f"{inventory}: J top-1 hits", f"{inventory} mixed cells", top1["J_hits"],
              sum(by_cell[k]["J"]["top1_hit"] for k in cells), s)
        check(f"{inventory}: F* ({f_star}) top-1 hits", f"{inventory} mixed cells", top1["F_star_hits"],
              sum(by_cell[k][f_star]["top1_hit"] for k in cells), s)
        check(f"{inventory}: top-1 cells", f"{inventory} mixed cells", top1["cells"], len(cells), s)
        check(f"{inventory}: chance top-1 mean", f"{inventory} mixed cells", top1["chance_mean"],
              float(np.mean([by_cell[k]["J"]["chance_top1"] for k in cells])), s)
        # cell-by-cell comparison with the published per-cell table in compute.json
        published_rows = compute["inventories"][inventory]["rows"]
        mismatches = 0
        for prow in published_rows:
            cell = by_cell[(inventory, prow["member"], prow["seed"])]
            for arm in prow["auc"]:
                a, b = prow["auc"][arm], cell[arm]["cell_auc"]
                if (a is None) != (b is None) or (a is not None and abs(a - b) > TOL):
                    mismatches += 1
                hit = prow["top1_hit"][arm]
                if (None if hit is None else int(hit)) != cell[arm]["top1_hit"]:
                    mismatches += 1
            if prow["admissible"] != cell["J"]["admissible"]:
                mismatches += 1
        check(f"{inventory}: per-cell AUC/top-1/admissible mismatches vs published rows",
              f"{inventory} mixed cells x 18 arms", 0, mismatches, f"{A96}/compute.json .inventories.{inventory}.rows")
        check(f"{inventory}: published per-cell rows", f"{inventory} mixed cells",
              len(published_rows), len(cells), f"{A96}/compute.json .inventories.{inventory}.rows")
    check("scheduled cells", "all levels x 3 seeds", read(f"{A96}/plan.json")["universe"]["scheduled_cells"],
          len({(r["inventory"], r["level"], r["seed"]) for r in rows}), f"{A96}/plan.json .universe")
    check("decision records", "all levels x 3 seeds x 18 arms", read(f"{A96}/plan.json")["universe"]["records"],
          len(rows), f"{A96}/plan.json .universe")
    return out


def checks_verdicts(slots, universe):
    summary = read(f"{A96}/summary.json")
    out = []
    for inventory in INVENTORIES:
        published = summary["inventories"][inventory]["coverage"]["successful_slots"]
        recomputed = sum(1 for s in slots if s["inventory"] == inventory and s["in_issue96_universe"]
                         and s["verdict"] == 1
                         and set(universe[inventory][s["level"]]["verdicts"].values()) == {True, False})
        out.append({"file": "dev_issue96_verdicts.csv",
                    "quantity": f"{inventory}: successful verdict slots on mixed levels",
                    "population": f"{inventory} mixed levels in the #96 universe",
                    "published_source": f"{A96}/summary.json .inventories.{inventory}.coverage.successful_slots",
                    "published_value": published, "recomputed_value": recomputed,
                    "match": int(published == recomputed)})
        published = summary["inventories"][inventory]["members"]
        recomputed = len({s["level"] for s in slots if s["inventory"] == inventory and s["in_issue96_universe"]})
        out.append({"file": "dev_issue96_verdicts.csv", "quantity": f"{inventory}: levels in universe",
                    "population": f"{inventory} #96 universe",
                    "published_source": f"{A96}/summary.json .inventories.{inventory}.members",
                    "published_value": published, "recomputed_value": recomputed,
                    "match": int(published == recomputed)})
    return out


# ---------------------------------------------------------------------------
# generic comparison row
# ---------------------------------------------------------------------------

def check_row(file, quantity, population, source, published, recomputed, tol=TOL):
    if (isinstance(published, (int, float)) and isinstance(recomputed, (int, float))
            and not isinstance(published, bool) and not isinstance(recomputed, bool)):
        match = abs(float(published) - float(recomputed)) <= tol
    else:
        match = published == recomputed
    return {"file": file, "quantity": quantity, "population": population, "published_source": source,
            "published_value": published, "recomputed_value": recomputed, "match": int(match)}


# ---------------------------------------------------------------------------
# split membership: CLEVRER and Physion-Dominoes (#106), NovPhy sealed split (#109 via #102/#108/#112/#113)
# ---------------------------------------------------------------------------

CLEVRER_RANGES = {"train": (0, 9999), "validation": (10000, 14999)}
PHYSION_PREFIX = "data/physion-dominoes/extracted/"


def clevrer_rows():
    plan = read(f"{A106}/plan.json")
    membership, roles = plan["membership"]["clevrer"], plan["membership"]["roles"]["clevrer"]
    rows, checks = [], []
    src = f"{A106}/plan.json .membership.clevrer"
    for split, (low, high) in CLEVRER_RANGES.items():
        block = membership[split]
        included = set(block["scenes"])
        if len(included) != len(block["scenes"]) or not all(low <= i <= high for i in included):
            raise ValueError(f"CLEVRER {split}: recorded scenes outside {low}-{high} or repeated")
        for video in range(low, high + 1):
            rows.append({"video_id": video, "release_split": split, "role": roles[split],
                         "included": int(video in included),
                         "exclusion_reason": "" if video in included else "failed_membership_rule"})
        n = high - low + 1
        checks.append(check_row("clevrer_membership.csv", f"{split}: included videos", f"CLEVRER {split} {low}-{high}",
                                f"{src}.{split}.count", block["count"], len(included)))
        checks.append(check_row("clevrer_membership.csv", f"{split}: excluded videos", f"CLEVRER {split} {low}-{high}",
                                f"{src}.{split}.skipped_repeated_key_or_contract",
                                block["skipped_repeated_key_or_contract"], n - len(included)))
    return rows, checks


def physion_name(path):
    return Path(path).stem


def physion_rows():
    plan = read(f"{A106}/plan.json")
    membership = plan["membership"]["physion"]
    roles = plan["membership"]["roles"]["physion"]
    audit = read(f"{A106}/leakage-audit.json")
    summary = read(f"{A106}/summary.json")
    overlap = audit["physical_overlap"]
    shared = defaultdict(set)
    for pair in overlap["pairs"]:
        shared[pair["testing"]].add(pair["split"])
    near = set(overlap["near_identical"]["testing_trials"])
    trials, checks = [], []
    for split in ("dynamics", "readout", "testing"):
        block = membership[split]
        if block["excluded_names"]:
            raise ValueError(f"Physion {split}: excluded trials recorded; extend the packaging")
        for path in block["trials"]:
            if not path.startswith(PHYSION_PREFIX):
                raise ValueError(f"unexpected Physion path {path}")
            splits = shared.get(path, set())
            trials.append({"release_split": split, "trial": physion_name(path),
                           "release_path": path[len(PHYSION_PREFIX):], "role": roles[split], "included": 1,
                           "shares_physical_parameters_with_dynamics": int("dynamics" in splits) if split == "testing" else "",
                           "shares_physical_parameters_with_readout": int("readout" in splits) if split == "testing" else "",
                           "near_identical_trajectory_in_fit_split": int(path in near) if split == "testing" else ""})
        checks.append(check_row("physion_membership.csv", f"{split}: included trials", f"Physion-Dominoes {split}",
                                f"{A106}/plan.json .membership.physion.{split}.count", block["count"],
                                sum(1 for t in trials if t["release_split"] == split)))
        checks.append(check_row("physion_membership.csv", f"{split}: trials in leakage audit", f"Physion-Dominoes {split}",
                                f"{A106}/leakage-audit.json .trials.{split}", audit["trials"][split],
                                sum(1 for t in trials if t["release_split"] == split)))
    names = {s: {t["trial"] for t in trials if t["release_split"] == s} for s in ("dynamics", "readout", "testing")}
    for key, published in plan["membership"]["physion_split_overlap"].items():
        a, b = key.split("|")
        checks.append(check_row("physion_membership.csv", f"{key}: shared trial names", "trial-name intersection",
                                f"{A106}/plan.json .membership.physion_split_overlap.{key}.names",
                                published["names"], len(names[a] & names[b])))
    pairs = []
    for pair in overlap["pairs"]:
        pairs.append({"testing_trial": physion_name(pair["testing"]), "fit_split": pair["split"],
                      "fit_trial": physion_name(pair["trial"]),
                      "fit_release_path": pair["trial"][len(PHYSION_PREFIX):],
                      "testing_native_frames": pair["frames"][0], "fit_native_frames": pair["frames"][1],
                      "max_abs_position": pair["max_abs_position"], "max_abs_velocity": pair["max_abs_velocity"],
                      "contact_labels_identical": int(pair["contact_labels_identical"]),
                      "near_identical": int(pair["frames"][0] == pair["frames"][1] and pair["contact_labels_identical"]
                                            and pair["max_abs_position"] <= 1e-3)})
    testing = [t for t in trials if t["release_split"] == "testing"]
    sharing = audit["physical_overlap"]["testing_sharing"]
    src = f"{A106}/leakage-audit.json .physical_overlap"
    recomputed = {"dynamics": sum(t["shares_physical_parameters_with_dynamics"] for t in testing),
                  "readout": sum(t["shares_physical_parameters_with_readout"] for t in testing),
                  "both": sum(t["shares_physical_parameters_with_dynamics"] and t["shares_physical_parameters_with_readout"]
                              for t in testing),
                  "either": sum(t["shares_physical_parameters_with_dynamics"] or t["shares_physical_parameters_with_readout"]
                                for t in testing)}
    for key in ("dynamics", "readout", "both", "either"):
        checks.append(check_row("physion_membership.csv", f"testing trials sharing physical parameters: {key}",
                                "150 testing trials", f"{src}.testing_sharing.{key}", sharing[key], int(recomputed[key])))
        checks.append(check_row("physion_membership.csv", f"testing trials sharing physical parameters: {key}",
                                "150 testing trials", f"{A106}/summary.json .split_overlap.testing_sharing_physical_parameters.{key}",
                                summary["split_overlap"]["testing_sharing_physical_parameters"][key], int(recomputed[key])))
    checks.append(check_row("physion_overlap_pairs.csv", "testing trials with a near-identical fit-split trajectory",
                            "overlap pairs", f"{src}.near_identical.testing_trials",
                            sorted(physion_name(p) for p in near),
                            sorted({p["testing_trial"] for p in pairs if p["near_identical"]})))
    closest = min(pairs, key=lambda p: p["max_abs_position"])
    checks.append(check_row("physion_overlap_pairs.csv", "closest pair (min max_abs_position)", "overlap pairs",
                            f"{src}.closest_pair", [physion_name(overlap["closest_pair"]["testing"]),
                                                   physion_name(overlap["closest_pair"]["trial"]),
                                                   overlap["closest_pair"]["max_abs_position"]],
                            [closest["testing_trial"], closest["fit_trial"], closest["max_abs_position"]]))
    checks.append(check_row("physion_overlap_pairs.csv", "testing trials appearing in overlap pairs", "overlap pairs",
                            f"{src}.testing_trials_shared", sorted(physion_name(p) for p in overlap["testing_trials_shared"]),
                            sorted({p["testing_trial"] for p in pairs})))
    counts = []
    for key, block in audit["overlap"].items():
        a, b = key.split("|")
        counts.append({"split_a": a, "split_b": b, "shared_trial_names": block["names"],
                       "shared_layouts": block["layouts"], "shared_parameter_sets": block["parameter_sets"],
                       "shared_physical_parameter_sets": block["physical_sets"]})
    return trials, pairs, counts, checks


ROLE_OF = {"fit": "predictor_fit", "policy": "policy_fit", "evaluation": "evaluation"}
LEVEL_RE = re.compile(r"^issue-109-sc-(fit|policy|evaluation)-(type\d{6})-(\d{3})$")


def novphy_levels():
    checks = []
    fit = read(f"{A112}/fit-verdicts.json")
    fit_levels = defaultdict(lambda: {"grid": 0, "angle": 0, "family": None})
    for branch in fit["branches"]:
        entry = fit_levels[branch["level"]]
        entry["family"] = branch["family"]
        for item in branch["inventory_entries"]:
            entry[item["inventory"]] += 1
    policy_keys = read(f"{A113}/summaries-manifest-run.json")["summaries"]["policy"]
    policy_levels = defaultdict(int)
    for key in policy_keys:
        policy_levels[key.rsplit("-c", 1)[0]] += 1
    plan102 = read(f"{A102}/plan.json")
    evaluation = plan102["levels"]["N1"]
    n1_mixed = set(plan102["levels"]["N1-mixed"])
    gate_b = {}
    for name in ("unit", "c_star"):
        for seed in SEEDS:
            record = read(f"{A108}/records/{name}--seed{seed}.json")
            gate_b[(name, seed)] = {inv: sorted(record["inventories"][inv]) for inv in INVENTORIES}
    reference = gate_b[("unit", SEEDS[0])]
    if any(v != reference for v in gate_b.values()):
        raise ValueError("Gate-B records disagree on the mixed cells")
    mixed = {inv: set(levels) for inv, levels in reference.items()}
    rows = []

    def add(level, role_key, source, **extra):
        match = LEVEL_RE.match(level)
        if not match or match.group(1) != role_key:
            raise ValueError(f"unexpected level identity {level}")
        rows.append({"level": level, "role": ROLE_OF[role_key], "family": match.group(2),
                     "fit_grid_branches": "", "fit_angle_branches": "", "policy_branches": "",
                     "eval_grid_mixed": "", "eval_offset_mixed": "", "eval_angle_mixed": "", "eval_power_mixed": "",
                     "eval_grid_any_recorded_pass": "", "source": source, **extra})

    for level in sorted(fit_levels):
        entry = fit_levels[level]
        add(level, "fit", f"{A112}/fit-verdicts.json", fit_grid_branches=entry["grid"],
            fit_angle_branches=entry["angle"])
        if rows[-1]["family"] != entry["family"]:
            raise ValueError(f"{level}: family token differs from the recorded family")
    for level in sorted(policy_levels):
        add(level, "policy", f"{A113}/summaries-manifest-run.json", policy_branches=policy_levels[level])
    for level in evaluation:
        add(level, "evaluation", f"{A102}/plan.json .levels.N1; {A108}/records",
            eval_grid_mixed=int(level in mixed["grid"]), eval_offset_mixed=int(level in mixed["offset"]),
            eval_angle_mixed=int(level in mixed["angle"]), eval_power_mixed=int(level in mixed["power"]),
            eval_grid_any_recorded_pass=int(level in n1_mixed))
    if len({r["level"] for r in rows}) != len(rows):
        raise ValueError("a level appears in more than one role")
    if any(level not in set(evaluation) for inv in mixed for level in mixed[inv]):
        raise ValueError("a Gate-B mixed level is not an evaluation level")

    f = "novphy_sealed_levels.csv"
    streams = read(f"{A104P}/plan.json")["membership"]["streams"]
    depth = read(f"{A104P}/plan.json")["depth"]
    per_family = {"fit": depth["fit_levels_per_family"], "policy": depth["policy_levels_per_family"],
                  "evaluation": depth["grid"]["levels_per_family"]}
    for role_key in ("fit", "policy", "evaluation"):
        role = ROLE_OF[role_key]
        levels = [r for r in rows if r["role"] == role]
        checks.append(check_row(f, f"{role}: levels", f"{role} split", f"{A104P}/plan.json .membership.streams.issue109-{role_key}.levels",
                                streams[f"issue109-{role_key}"]["levels"], len(levels)))
        for family in sorted({r["family"] for r in rows}):
            checks.append(check_row(f, f"{role}: levels of family {family}", f"{role} split",
                                    f"{A104P}/plan.json .depth", per_family[role_key],
                                    sum(1 for r in levels if r["family"] == family)))
    checks.append(check_row(f, "predictor_fit: complete (verdict-bearing) branches", "fit split",
                            ".local-artifacts/issue-120-cost-alignment-v1/compute.json .fit.branches",
                            read(".local-artifacts/issue-120-cost-alignment-v1/compute.json")["fit"]["branches"],
                            len(fit["branches"])))
    checks.append(check_row(f, "policy_fit: captured branches", "policy split",
                            f"{A104P}/plan.json .membership.streams.issue109-policy.branches",
                            streams["issue109-policy"]["branches"], len(policy_keys)))
    pool = read(f"{A101}/compute.json")["pool"]
    checks.append(check_row(f, "policy_fit: levels", "policy split", f"{A101}/compute.json .pool.levels",
                            pool["levels"], len(policy_levels)))
    accounting = read(f"{A108}/plan.json")["evaluation"]["accounting"]
    for inv in INVENTORIES:
        checks.append(check_row(f, f"evaluation: {inv} mixed levels", "evaluation split",
                                f"{A108}/plan.json .evaluation.accounting.mixed_cells.{inv}",
                                accounting["mixed_cells"][inv], len(mixed[inv])))
    checks.append(check_row(f, "evaluation: levels with a recorded passing grid candidate", "evaluation split",
                            f"{A102}/plan.json .level_counts.N1-mixed", plan102["level_counts"]["N1-mixed"],
                            sum(r["eval_grid_any_recorded_pass"] == 1 for r in rows)))
    return rows, checks


# ---------------------------------------------------------------------------
# local aggregate tables (#102 live table, #103 frontier table) and #102 outcome counts
# ---------------------------------------------------------------------------

LIVE_SETS = ("N1", "N1-mixed", "N2-appearance")


def num(value, digits):
    return "n/a" if value is None else format(value, f".{digits}f")


def copy_tables():
    """Byte copies of the published tables plus a re-rendering of every row from compute.json."""
    checks = []
    live_rel, frontier_rel = f"{A102}/closed-loop-table.csv", f"{A103}/frontier-table.csv"
    live_text, frontier_text = (REPO / live_rel).read_text(), (REPO / frontier_rel).read_text()
    compute = read(f"{A102}/compute.json")
    plan102 = read(f"{A102}/plan.json")
    arms = list(plan102["arms"])
    published = list(csv.DictReader(live_text.splitlines()))
    mismatches, rendered = 0, 0
    for row in published:
        a = compute["tables"][row["set"]]["arms"][row["arm"]]
        t = compute["tables"][row["set"]]
        expect = {"group": a["group"], "episodes": str(a["episodes"])}
        for field in ("pass1", "pass2", "pass3", "shots_to_pass"):
            block = a[field]
            expect[field] = num(block and block["mean"], 4)
            expect[f"{field}_low"] = num(block and block["interval"][0], 4)
            expect[f"{field}_high"] = num(block and block["interval"][1], 4)
        expect["native_clear3"] = num(a["native_clear3"] and a["native_clear3"]["mean"], 4)
        expect["recorded_pass3"] = num(a["recorded_pass3"] and a["recorded_pass3"]["mean"], 4)
        c = a["compute"] or {}
        expect["mean_decision_seconds"] = num(c.get("mean_decision_seconds"), 5)
        expect["mean_rollout_seconds"] = num(c.get("mean_rollout_seconds"), 5)
        expect["mean_linear_macs_per_decision"] = num(c.get("mean_linear_macs_per_decision"), 0)
        expect["mean_linear_macs_per_candidate"] = num(c.get("mean_linear_macs_per_candidate"), 0)
        for key, name in (("vs_prior", "minus_prior"), ("vs_random", "minus_random")):
            con = t["contrasts"][key].get(row["arm"])
            est = con["estimate"] if con else None
            expect[name] = num(est and est["mean"], 4)
            expect[f"{name}_low"] = num(est and est["interval"][0], 4)
            expect[f"{name}_high"] = num(est and est["interval"][1], 4)
            expect[f"{name}_token"] = con["token"] if con else ""
        mismatches += sum(row[k] != v for k, v in expect.items())
        rendered += len(expect)
    f = "live_issue102_closed_loop_table.csv"
    checks.append(check_row(f, "rows (sets x arms)", "N1, N1-mixed, N2-appearance x all arms",
                            f"{A102}/plan.json .arms", len(LIVE_SETS) * len(arms), len(published)))
    checks.append(check_row(f, f"cell mismatches vs compute.json re-rendering ({rendered} cells)", "every row",
                            f"{A102}/compute.json .tables", 0, mismatches))

    plan103 = read(f"{A103}/plan.json")
    compute103 = read(f"{A103}/compute.json")
    sources = {p["id"]: p["source"] for p in plan103["policies"]}
    combos = ("auc_macs", "auc_wall", "top3_macs", "top3_wall")
    buffer = {}
    for set_name in ("primary", "secondary"):
        for inventory in INVENTORIES:
            t = compute103["sets"][set_name][inventory]
            for pid, p in t["policies"].items():
                auc, top3 = p["auc"] or {}, p["top3"] or {}
                values = [set_name, inventory, pid, p["group"], sources.get(pid, "gate_b rule"),
                          auc.get("mean"), (auc.get("interval") or [None, None])[0],
                          (auc.get("interval") or [None, None])[1], top3.get("mean"),
                          (top3.get("interval") or [None, None])[0], (top3.get("interval") or [None, None])[1],
                          p["chance_top3"], p["mean_macs_per_candidate"], p["mean_transitions_per_candidate"],
                          p["mean_policy_calls_per_candidate"], p["seconds_per_state"], p["recorded_seconds_per_state"],
                          p["cells"], p["tied_cells"], p["tied_pairs"], p["pairs"], p["nonfinite"], p["typed_failures"],
                          *(int(pid in t["frontiers"][c]["on"]) for c in combos),
                          *(t["frontiers"][c]["shares"][pid] for c in combos),
                          t["frontiers"]["auc_macs"]["verdicts"][pid]["text"]]
                out = io.StringIO()
                csv.writer(out, lineterminator="\n").writerow(values)
                buffer[(set_name, inventory, pid)] = out.getvalue().rstrip("\n")
    published = {}
    for line in frontier_text.splitlines()[1:]:
        key = tuple(next(csv.reader([line]))[:3])
        published[key] = line
    f = "frontier_issue103_table.csv"
    checks.append(check_row(f, "rows", "primary + secondary sets x 4 inventories x policies",
                            f"{A103}/compute.json .sets", len(buffer), len(published)))
    checks.append(check_row(f, "rows differing from a re-rendering of compute.json (keyed by set, inventory, policy)",
                            "every row", f"{A103}/compute.json .sets", 0,
                            sum(published.get(k) != v for k, v in buffer.items())
                            + sum(k not in buffer for k in published)))
    return live_text, frontier_text, checks


def live_outcome_counts():
    compute = read(f"{A102}/compute.json")
    arms = list(read(f"{A102}/plan.json")["arms"])
    table = {(r["set"], r["arm"]): r for r in
             csv.DictReader((REPO / f"{A102}/closed-loop-table.csv").read_text().splitlines())}
    rows, checks = [], []
    f = "live_issue102_outcome_counts.csv"
    for set_name in LIVE_SETS:
        t = compute["tables"][set_name]
        for arm in arms:
            a = t["arms"][arm]
            c = a["compute"] or {}
            per_shot = c.get("per_shot") or {}
            o = a["outcomes"]
            row = {"set": set_name, "arm": arm, "group": a["group"], "levels": t["levels"], "cells": t["cells"],
                   "episodes": a["episodes"], **{f"outcome_{k}": o[k] for k in
                                                 ("passed", "failed", "typed_failure", "no_candidate", "incomplete")},
                   **{f"passed_at_shot{k}": a["passed_at_shot"][str(k)] for k in (1, 2, 3)},
                   **{f"decisions_shot{k}": (per_shot.get(str(k)) or {}).get("decisions") for k in (1, 2, 3)},
                   "shots_executed_mean": a["shots_executed"]["mean"] if a["shots_executed"] else None,
                   "replay_differs_cells": a["replay_differs_cells"]}
            rows.append(row)
            q = f"{set_name}/{arm}"
            src = f"{A102}/compute.json .tables.{set_name}.arms.{arm}"
            checks.append(check_row(f, f"{q}: outcomes sum to episodes", "episodes", src + ".episodes",
                                    a["episodes"], sum(o.values())))
            checks.append(check_row(f, f"{q}: passed-at-shot sum to passed", "passed episodes", src + ".outcomes.passed",
                                    o["passed"], sum(a["passed_at_shot"].values())))
            published = table[(set_name, arm)]
            for k in (1, 3):
                passed = sum(a["passed_at_shot"][str(j)] for j in range(1, k + 1))
                checks.append(check_row(f, f"{q}: pass@{k} = passed by shot {k} / episodes (4 dp)", "episodes",
                                        f"{A102}/closed-loop-table.csv pass{k}", published[f"pass{k}"],
                                        num(passed / a["episodes"], 4)))
    typed = compute["cross_check"]["typed_failures"]
    failures = []
    for key, failure_class in typed:
        level, suffix = key[len("issue-102-"):].rsplit("-", 1)
        failures.append({"execution": key[len("issue-102-"):], "level": level, "candidate_suffix": suffix,
                         "set": "N1" if level.startswith("issue-109-sc-evaluation-") else "N2-appearance",
                         "failure_class": failure_class})
    by_set = compute["cross_check"]["by_set"]
    for set_name in ("N1", "N2-appearance"):
        checks.append(check_row("live_issue102_typed_failures.csv", f"{set_name}: typed-failure executions",
                                "live executions", f"{A102}/compute.json .cross_check.by_set.{set_name}.typed_failures",
                                by_set[set_name]["typed_failures"], sum(1 for x in failures if x["set"] == set_name)))
    return rows, failures, checks


# ---------------------------------------------------------------------------
# output
# ---------------------------------------------------------------------------

def fmt(value):
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(int(value))
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, list):
        return " ".join(str(v) for v in value)
    return str(value)


def write_csv(name, rows, columns):
    with open(HERE / name, "w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(columns)
        for row in rows:
            writer.writerow([fmt(row[c]) for c in columns])


FORBIDDEN = re.compile(r"(/p/Project|/home/|/mnt/|/Users/|github\.com|@[a-z0-9-]+\.)", re.IGNORECASE)


def main():
    universe, slots, roles = build_universe()
    rows = cell_rows(universe, roles)
    checks = checks_96(rows) + checks_verdicts(slots, universe)
    clevrer, c_checks = clevrer_rows()
    trials, pairs, overlap_counts, p_checks = physion_rows()
    levels, n_checks = novphy_levels()
    live_text, frontier_text, t_checks = copy_tables()
    outcome_rows, typed_rows, o_checks = live_outcome_counts()
    checks += c_checks + p_checks + n_checks + t_checks + o_checks
    write_csv("dev_issue96_cells.csv", rows, [
        "inventory", "level", "state", "exposure_role", "seed", "arm", "checkpoint_family",
        "candidates", "verdict_slots", "verdict_successes", "verdict_failures",
        "nonfinite_predictions", "admissible", "mixed_level", "cell_auc", "top1_hit",
        "chance_top1", "all_tied", "chosen_ordinal", "chosen_verdict", "decision_failure",
        "source_record"])
    write_csv("dev_issue96_verdicts.csv", slots, [
        "inventory", "level", "state", "ordinal", "branch_identity", "drag_x", "drag_y",
        "release_time_ms", "tap_time_ms", "verdict", "typed_failure", "typed_failure_detail",
        "verdict_record", "in_issue96_universe"])
    write_csv("clevrer_membership.csv", clevrer, ["video_id", "release_split", "role", "included", "exclusion_reason"])
    write_csv("physion_membership.csv", trials, [
        "release_split", "trial", "release_path", "role", "included", "shares_physical_parameters_with_dynamics",
        "shares_physical_parameters_with_readout", "near_identical_trajectory_in_fit_split"])
    write_csv("physion_overlap_pairs.csv", pairs, [
        "testing_trial", "fit_split", "fit_trial", "fit_release_path", "testing_native_frames", "fit_native_frames",
        "max_abs_position", "max_abs_velocity", "contact_labels_identical", "near_identical"])
    write_csv("physion_split_overlap_counts.csv", overlap_counts, [
        "split_a", "split_b", "shared_trial_names", "shared_layouts", "shared_parameter_sets",
        "shared_physical_parameter_sets"])
    write_csv("novphy_sealed_levels.csv", levels, [
        "level", "role", "family", "fit_grid_branches", "fit_angle_branches", "policy_branches",
        "eval_grid_mixed", "eval_offset_mixed", "eval_angle_mixed", "eval_power_mixed",
        "eval_grid_any_recorded_pass", "source"])
    (HERE / "live_issue102_closed_loop_table.csv").write_text(live_text)
    (HERE / "frontier_issue103_table.csv").write_text(frontier_text)
    write_csv("live_issue102_outcome_counts.csv", outcome_rows, [
        "set", "arm", "group", "levels", "cells", "episodes", "outcome_passed", "outcome_failed",
        "outcome_typed_failure", "outcome_no_candidate", "outcome_incomplete", "passed_at_shot1",
        "passed_at_shot2", "passed_at_shot3", "decisions_shot1", "decisions_shot2", "decisions_shot3",
        "shots_executed_mean", "replay_differs_cells"])
    write_csv("live_issue102_typed_failures.csv", typed_rows, ["execution", "level", "candidate_suffix", "set",
                                                               "failure_class"])
    write_csv("checks.csv", checks, ["file", "quantity", "population", "published_source",
                                     "published_value", "recomputed_value", "match"])
    outputs = sorted(p.name for p in HERE.glob("*.csv"))
    for name in outputs:
        hit = FORBIDDEN.search((HERE / name).read_text())
        if hit:
            raise ValueError(f"{name}: non-anonymous string {hit.group(0)!r}")
    for name in outputs:
        print(f"{name}: {len((HERE / name).read_text().splitlines()) - 1} rows")
    failed = [c for c in checks if not c["match"]]
    print(f"checks: {len(checks) - len(failed)}/{len(checks)} match")
    for c in failed:
        print("MISMATCH", c["file"], c["quantity"], c["published_value"], c["recomputed_value"])
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
