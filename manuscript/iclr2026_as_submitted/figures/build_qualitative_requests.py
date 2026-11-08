"""Draw the a-priori running scene's imagined requests and launch-cost ranks.

Run with the novphy environment:
    python -B figures/build_qualitative_requests.py

Evidence: read-only issue-96 decision records, its frozen plan, and the exact
run_tau_ad_within_checkpoint.load_sources/build_universe engine-verdict join.
No manuscript documents supply data. No model or engine is rerun.

Panel (a) shows candidate-dependent request schedules, not their quality.
Panel (b) checks those predictions against first-shot engine success; this
example was fixed by the running scene, not selected for a favorable outcome.
The aggregate alternative is reported as prediction-step-weighted fractions:
a step belongs to the bin containing its START frame, not its end or midpoint.
Cells are (inventory, member, seed), including non-mixed-outcome members.

Outputs: the PDF and 300-dpi PNG beside this script; a JSON report on stdout
contains decoded run-length schedules, all costs/ranks/outcomes, and aggregates.
All exports have fixed physical size; PDF text remains embedded TrueType text.
"""
from collections import Counter
import json
import math
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


ROOT = Path("/p/Project/NovPhy")
SOURCE = ROOT / ".local-artifacts/issue-96-tau-ad-within-checkpoint-v1"
OUT = Path(__file__).resolve().parent
SIZE = (5.5, 1.6)
PAIRS = tuple((d, a) for d in (1, 5, 15) for a in ("cont", "micro", "macro"))
PAIR_NAMES = tuple(f"{d}-{a}" for d, a in PAIRS)
BINS = ((0, 25), (25, 100), (100, 225))
INVENTORIES = ("angle", "offset", "grid", "power")
EXAMPLE = ("angle", "issue-77-n1-001", 20260908)
ARMS = ("J", "F-1-macro", "F-5-macro")
COLORS = {"cont": "#0072B2", "micro": "#CC79A7", "macro": "#D55E00"}
INK, GREY, REQUEST = "#18252d", "#637382", "#007C78"


def load_json(path):
    return json.loads(path.read_text())


def decode_schedule(text):
    """Return individual (delta, alpha) prediction steps, checking the endpoint."""
    steps = []
    for token in text.split():
        index, count = map(int, token.split("x"))
        assert 0 <= index < len(PAIRS) and count > 0, token
        steps.extend([PAIRS[index]] * count)
    assert sum(delta for delta, _ in steps) == 225, text
    return steps


def decoded_runs(text):
    return [
        {"delta": PAIRS[int(index)][0], "alpha": PAIRS[int(index)][1],
         "predictions": int(count)}
        for index, count in (token.split("x") for token in text.split())
    ]


def ranked_candidates(record):
    rows = record["decision"]["ranking"]
    eligible = [row for row in rows if not row["excluded"]]
    assert all(row["predicted_cost"] is not None
               and math.isfinite(row["predicted_cost"]) for row in eligible)
    ordered = sorted(eligible, key=lambda row: (row["predicted_cost"], row["ordinal"]))
    ranks = {row["ordinal"]: rank for rank, row in enumerate(ordered, 1)}
    assert ordered[0]["ordinal"] == record["decision"]["chosen_ordinal"]
    return ranks


def read_record(inventory, member, seed, arm, entry):
    path = SOURCE / "records" / f"decision--{inventory}--{member}--seed{seed}--{arm}.json"
    record = load_json(path)
    cell, decision = record["cell"], record["decision"]
    assert (cell["inventory"], cell["member"], cell["seed"], cell["arm"]) == (
        inventory, member, seed, arm)
    assert cell["state"] == entry["state"] and cell["endpoint"] == 225
    expected = {item["ordinal"]: item["branch_identity"] for item in entry["inventory"]}
    actual = {row["ordinal"]: row["branch_identity"] for row in decision["ranking"]}
    assert actual == expected, path
    assert decision["candidate_count"] == len(decision["ranking"]) == len(expected), path
    # Unresolved engine outcomes must never be silently relabeled as failures.
    assert set(entry["verdicts"]) | set(entry["oracle_failures"]) == set(expected), path
    assert not (set(entry["verdicts"]) & set(entry["oracle_failures"])), path
    if arm == "J":
        assert {int(o) for o in decision["schedules"]} == set(expected), path
        counts = Counter()
        for text in decision["schedules"].values():
            for delta, alpha in decode_schedule(text):
                counts[f"{delta}-{'continuous' if alpha == 'cont' else alpha}"] += 1
        assert dict(counts) == decision["pair_step_counts"], path
    return record


def shares(counts):
    total = sum(counts.values())
    assert total > 0
    by_delta = {str(d): sum(n for (delta, _), n in counts.items() if delta == d)
                for d in (1, 5, 15)}
    by_alpha = {a: sum(n for (_, alpha), n in counts.items() if alpha == a)
                for a in ("cont", "micro", "macro")}
    return {"prediction_steps": total,
            "request_counts": {name: counts[pair] for name, pair in zip(PAIR_NAMES, PAIRS)},
            "request_shares": {name: counts[pair] / total
                               for name, pair in zip(PAIR_NAMES, PAIRS)},
            "delta_counts": by_delta,
            "delta_shares": {d: n / total for d, n in by_delta.items()},
            "alpha_counts": by_alpha,
            "alpha_shares": {a: n / total for a, n in by_alpha.items()}}


def analyze():
    # Importing the original read-only join avoids duplicating completion-pass
    # precedence. Suppress bytecode so even imports cannot write to raw sources.
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(ROOT))
    from scripts import run_tau_ad_within_checkpoint as runner

    plan = load_json(SOURCE / "plan.json")
    sources = runner.load_sources()
    _, universe, excluded = runner.build_universe(sources)
    assert excluded == plan["universe"]["excluded"]
    assert {inv: sorted(members) for inv, members in universe.items()} == plan["universe"]["members"]
    assert tuple(name.replace("continuous", "cont") for name in plan["universe"]["pairs_order"]) == PAIR_NAMES
    seeds = plan["universe"]["seeds"]
    assert seeds == [20260908, 20260909, 20260910]
    pooled = [Counter() for _ in BINS]
    per_inventory = {}
    expected_paths = set()
    total_candidates = 0
    total_excluded_predictions = 0
    for inventory in INVENTORIES:
        counts = [Counter() for _ in BINS]
        differing, cells, candidates = 0, 0, 0
        for member, entry in sorted(universe[inventory].items()):
            for seed in seeds:
                record = read_record(inventory, member, seed, "J", entry)
                expected_paths.add(f"decision--{inventory}--{member}--seed{seed}--J.json")
                schedules = record["decision"]["schedules"]
                sequences = [tuple(decode_schedule(text)) for text in schedules.values()]
                differing += len(set(sequences)) > 1
                cells += 1
                candidates += len(sequences)
                total_excluded_predictions += sum(row["excluded"] for row in record["decision"]["ranking"])
                for sequence in sequences:
                    frame = 0
                    for delta, alpha in sequence:
                        bin_index = next(i for i, (start, end) in enumerate(BINS) if start <= frame < end)
                        counts[bin_index][delta, alpha] += 1
                        pooled[bin_index][delta, alpha] += 1
                        frame += delta
                    assert frame == 225
        total_candidates += candidates
        per_inventory[inventory] = {
            "cells": cells, "candidate_rollouts": candidates,
            "differing_schedule_cells": differing,
            "differing_schedule_fraction": differing / cells,
            "bins": {f"[{start},{end})": shares(counts[i])
                     for i, (start, end) in enumerate(BINS)}}
    assert expected_paths == {path.name for path in (SOURCE / "records").glob("decision--*--J.json")}
    total_cells = sum(v["cells"] for v in per_inventory.values())
    assert total_cells == plan["universe"]["scheduled_cells"] == 177
    differing = sum(v["differing_schedule_cells"] for v in per_inventory.values())

    inv, member, seed = EXAMPLE
    entry = universe[inv][member]
    records = {arm: read_record(inv, member, seed, arm, entry) for arm in ARMS}
    ordinals = sorted(entry["verdicts"])
    assert ordinals == list(range(11)) + [12]
    assert not entry["oracle_failures"]
    successes = [o for o in ordinals if entry["verdicts"][o]]
    assert successes == [6], successes
    state87 = next(row for row in sources["plan87"]["states"] if row["identity"] == entry["state"])
    assert state87["typed_dropped_branches"] == [f"{member}-a11"]
    ranks = {arm: ranked_candidates(record) for arm, record in records.items()}
    costs = {arm: {row["ordinal"]: row["predicted_cost"] for row in record["decision"]["ranking"]}
             for arm, record in records.items()}
    schedule_text = records["J"]["decision"]["schedules"]
    report = {
        "example": {"inventory": inv, "member": member, "seed": seed,
                    "selection": "a priori running scene, not selected by outcome",
                    "candidate_count": len(ordinals), "successful_ordinals": successes,
                    "absent_ordinal": 11, "absence_source": "plan87 states[].typed_dropped_branches",
                    "candidates": [
                        {"ordinal": o, "engine_first_shot_success": entry["verdicts"][o],
                         "policy_schedule": decoded_runs(schedule_text[str(o)]),
                         "policy_predictions": len(decode_schedule(schedule_text[str(o)])),
                         "costs": {arm: costs[arm][o] for arm in ARMS},
                         "ranks": {arm: ranks[arm].get(o) for arm in ARMS},
                         "excluded": {arm: next(row["excluded"] for row in records[arm]["decision"]["ranking"]
                                                if row["ordinal"] == o) for arm in ARMS}}
                        for o in ordinals],
                    "successful_candidate_ranks": {arm: ranks[arm][6] for arm in ARMS}},
        "aggregate": {"weighting": "pooled prediction steps, binned by start carrier frame",
                      "cell_key": ["inventory", "member", "seed"],
                      "cells": total_cells, "candidate_rollouts": total_candidates,
                      "excluded_prediction_candidates": total_excluded_predictions,
                      "differing_schedule_cells": differing,
                      "differing_schedule_fraction": differing / total_cells,
                      "bins": {f"[{start},{end})": shares(pooled[i])
                               for i, (start, end) in enumerate(BINS)},
                      "by_inventory": per_inventory},
        "figure": {"size_inches": list(SIZE), "png_dpi": 300,
                   "pdf": str(OUT / "fig_qualitative_requests.pdf"),
                   "png": str(OUT / "fig_qualitative_requests.png")}}
    return records, entry, ranks, report


def draw(records, entry, ranks):
    style = {"font.family": "DejaVu Sans", "font.size": 7,
             "mathtext.fontset": "dejavusans", "pdf.fonttype": 42,
             "ps.fonttype": 42, "text.color": INK, "axes.labelcolor": INK,
             "xtick.color": INK, "ytick.color": INK}
    with plt.rc_context(style):
        fig = plt.figure(figsize=SIZE, facecolor="white")

        def text(x, y, value, **kwargs):
            return fig.text(x / SIZE[0], y / SIZE[1], value, fontsize=7,
                            va="center", **kwargs)

        text(.04, 1.53, "(a)", weight="bold")
        text(.25, 1.53, "Requests chosen while imagining each launch")
        text(3.38, 1.53, "(b)", weight="bold")
        text(3.59, 1.53, "Rank of each launch by predicted cost")
        for x, alpha in ((.32, "cont"), (.88, "micro"), (1.48, "macro")):
            fig.add_artist(Rectangle((x / SIZE[0], 1.365 / SIZE[1]), .11 / SIZE[0], .045 / SIZE[1],
                                     transform=fig.transFigure, facecolor=COLORS[alpha], edgecolor="none"))
            text(x + .15, 1.39, alpha)
        text(2.16, 1.39, "width = Δ")

        ax = fig.add_axes([.32 / SIZE[0], .31 / SIZE[1], 2.45 / SIZE[0], 1.015 / SIZE[1]])
        ax.set(xlim=(0, 225), ylim=(12.65, -.65))
        ax.set_yticks(range(13), [f"a{o}" for o in range(13)])
        ax.set_xticks([0, 25, 100, 225])
        ax.tick_params(axis="y", length=0, labelsize=7, pad=2)
        ax.tick_params(axis="x", length=2, width=.5, labelsize=7, pad=1)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color(GREY)
        ax.spines["bottom"].set_linewidth(.5)
        for row in records["J"]["decision"]["ranking"]:
            ordinal, frame = row["ordinal"], 0
            for delta, alpha in decode_schedule(records["J"]["decision"]["schedules"][str(ordinal)]):
                ax.add_patch(Rectangle((frame, ordinal - .34), delta, .68,
                                       facecolor=GREY if row["excluded"] else COLORS[alpha],
                                       edgecolor="white", linewidth=.15))
                frame += delta
            if entry["verdicts"][ordinal]:
                ax.text(229, ordinal, "success", va="center", fontsize=7,
                        color=INK, clip_on=False, weight="bold")
        ax.text(112.5, 11, "not in frozen inventory", fontsize=7, ha="center", va="center", color=GREY)
        ax.get_yticklabels()[11].set_color(GREY)
        ax.get_yticklabels()[6].set_weight("bold")
        text(1.54, .055, "Carrier frame (segment width = prediction step)", ha="center")

        bx = fig.add_axes([3.61 / SIZE[0], .31 / SIZE[1], 1.72 / SIZE[0], 1.015 / SIZE[1]])
        bx.set(xlim=(-.5, 2.5), ylim=(12.65, .35))
        bx.set_yticks([1, 6, 12])
        bx.set_xticks([])
        bx.tick_params(axis="y", length=2, width=.5, labelsize=7, pad=2)
        bx.spines["left"].set_color(GREY)
        bx.spines["left"].set_linewidth(.5)
        for spine in ("top", "right", "bottom"):
            bx.spines[spine].set_visible(False)
        for column, arm in enumerate(ARMS):
            for ordinal, rank in ranks[arm].items():
                success = entry["verdicts"][ordinal]
                bx.plot(column, rank, "o", markersize=3.8 if success else 3.0,
                        markerfacecolor=REQUEST if success else "white",
                        markeredgecolor=INK if success else GREY,
                        markeredgewidth=.7, zorder=3 if success else 2)
            x = 3.61 + (column + .5) / 3 * 1.72
            label = ("Policy", r"$r^\dagger$", "Hindsight")[column]
            text(x, 1.39, label, ha="center")
            pair = ("adaptive", "(1,macro)", "(5,macro)")[column]
            text(x, .215, pair, ha="center")
            text(x, .07, f"{ranks[arm][6]}/12", ha="center", weight="bold")
        text(3.40, .07, "a6:", ha="center")

        fig.savefig(OUT / "fig_qualitative_requests.pdf", facecolor="white",
                    metadata={"CreationDate": None, "ModDate": None})
        fig.savefig(OUT / "fig_qualitative_requests.png", dpi=300, facecolor="white")
        plt.close(fig)


if __name__ == "__main__":
    records, entry, ranks, report = analyze()
    draw(records, entry, ranks)
    print(json.dumps(report, indent=2, sort_keys=True))
