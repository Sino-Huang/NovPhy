"""Build manuscript plots from retained artifacts; scene figures live in build_scene_figures.

Run from anywhere: python figures/build_figures.py. No engine, model or GPU is touched.
"""
import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("pdf")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from build_scene_figures import teaser_scene, qualitative_case  # noqa: E402

ART = Path("/p/Project/NovPhy/.local-artifacts")
OUT = Path(__file__).resolve().parent

# Okabe-Ito palette.
OI = {
    "black": "#000000", "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73",
    "yellow": "#F0E442", "blue": "#0072B2", "vermillion": "#D55E00", "purple": "#CC79A7",
    "grey": "#777777",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 9,
    "axes.labelsize": 8.5, "legend.fontsize": 8, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "pdf.fonttype": 42, "svg.fonttype": "none",
    "axes.spines.top": False, "axes.spines.right": False,
})


def load(p):
    return json.loads(Path(p).read_text())

def save_figure(fig, stem):
    """Editable vector artwork plus a final-size raster preview."""
    for ext in ("pdf", "svg", "png"):
        fig.savefig(OUT / f"{stem}.{ext}", dpi=240, bbox_inches="tight", pad_inches=0.035)
    plt.close(fig)


# --------------------------------------------------------------------------- hero
def hero():
    """Post-hoc N1 ordinal pattern; separate member and model-cell denominators."""
    s92 = load(ART / "issue-92-selection-validity-v1/summary.json")
    sv = s92["selection_validity"]
    chosen = {int(k): v for k, v in sv["chosen_support"].items()}
    assert chosen == {7: 90, 8: 12, 9: 45, 10: 10, 11: 6, 12: 44}
    assert sum(chosen.values()) == sv["scored_model_cells"] == 207
    assert sv["success_band"] == [2, 3, 4, 5, 6] and sv["chosen_in_band_cells"] == 0
    terms = s92["chance_references"]["member_terms"]
    assert len(terms) == 7 and all(t["k"] == 1 for t in terms.values())
    successes = {}
    for row in csv.DictReader((ART / "issue-92-selection-validity-v1/slot_join.csv").open()):
        if row["n_successes"] not in ("", "0"):
            assert row["n_successes"] == "1", row["cell_identity"]
            successes.setdefault(row["source_member"], set()).add(int(row["successful_ordinals"]))
    assert all(len(v) == 1 for v in successes.values())
    successes = {m.rsplit("-", 1)[1]: next(iter(v)) for m, v in successes.items()}
    assert successes.pop("005") == 7  # #89 typed-unmeasurable; excluded from the ceiling.
    assert successes == {"001": 6, "006": 5, "010": 2, "011": 4, "012": 4, "014": 4, "016": 3}
    assert set(successes) == {m.rsplit("-", 1)[1] for m in terms}
    member_counts = Counter(successes.values())
    assert member_counts == {2: 1, 3: 1, 4: 3, 5: 1, 6: 1}

    # Separate cross-split conditions are captioned as an OUTSIDE-N1 counterexample, not plotted
    # as if their cells were the seven N1 ceiling members. These are the corrected #98 cell AUCs.
    cross = load(ART / "issue-92-cross-pool-audit-v1/summary.json")["cross_split"]["per_condition"]
    for condition, hits, auc in (("zero-shot", 13, 0.5015), ("few-shot", 29, 0.5073)):
        record = cross[condition]
        cell_auc = record["auc"]["cell_unit"]
        assert record["cells"] == 612 and record["top1_hits"] == hits
        assert record["auc"]["scored_cells"] == cell_auc["units"] == 216
        assert cell_auc["clusters"] == 6 and cell_auc["label"] == "DESCRIPTIVE"
        assert round(cell_auc["mean"], 4) == auc

    fig, (top, bottom) = plt.subplots(2, 1, figsize=(7.0, 2.9), sharex=True,
                                      gridspec_kw={"hspace": 0.30}, layout="constrained")
    fig.get_layout_engine().set(rect=(0.01, 0.01, 0.99, 0.99))
    top.bar(range(13), [member_counts.get(o, 0) for o in range(13)], width=0.68,
            color=OI["blue"], label="measured success")
    for ordinal, count in member_counts.items():
        top.text(ordinal, count - 0.14 if count == 3 else count + 0.12, str(count),
                 ha="center", va="top" if count == 3 else "bottom", fontsize=8,
                 color="white" if count == 3 else OI["black"])
    top.bar([7], [1], width=0.68, facecolor="white", edgecolor=OI["blue"], hatch="////",
            lw=1.1, label="excluded typed-unmeasurable a07")
    top.text(7.5, 1.12, "a07 excluded", fontsize=8, va="bottom")
    top.set_ylim(0, 4.2)
    top.set_yticks([0, 1, 2, 3])
    top.set_ylabel("Members")
    top.text(0.01, 0.94, "a  Engine-successful candidates", transform=top.transAxes,
             fontsize=9, fontweight="bold", va="top")

    bottom.bar(range(13), [chosen.get(o, 0) for o in range(13)], width=0.68,
               color=OI["vermillion"], label="frozen model choice")
    for ordinal, count in chosen.items():
        bottom.text(ordinal, count + 2.5, str(count), ha="center", va="bottom", fontsize=8)
    bottom.set_ylim(0, 116)
    bottom.set_yticks([0, 40, 80])
    bottom.set_ylabel("Model cells")
    bottom.text(0.01, 0.94, "b  Frozen ranker choices", transform=bottom.transAxes,
                fontsize=9, fontweight="bold", va="top")
    bottom.set_xlim(-0.7, 12.7)
    bottom.set_xticks(range(13))
    bottom.set_xlabel("Candidate ordinal  (a0 flat  →  a12 steep launch)")
    save_figure(fig, "fig_hero_ordinal_masses")


# ------------------------------------------------------------ hero (r6, Fig. 1)
def hero_selection_ordering():
    """Four separate N1 inventories on the same 15 source members."""
    r4 = lambda x: round(float(x), 4)  # noqa: E731

    # N1, original 13-candidate sweep: issue-92-selection-validity-v1/summary.json.
    s92 = load(ART / "issue-92-selection-validity-v1/summary.json")
    mc = s92["unit_of_analysis"]["member_ceiling"]
    assert mc["label"] == "DESCRIPTIVE" and mc["units"] == 14 and round(mc["mean"] * mc["units"]) == 7
    assert [r4(mc["mean"]), *map(r4, mc["interval"])] == [0.5, 0.2143, 0.7857]
    t1 = s92["selection_validity"]["top1"]
    pd = t1["paired_difference"]
    assert (t1["hits"], t1["cells"], r4(t1["chance_rate"])) == (0, 108, 0.0780)
    assert pd["label"] == "DESCRIPTIVE" and pd["clusters"] == 7
    assert [r4(pd["mean"]), *map(r4, pd["interval"])] == [-0.0778, -0.0797, -0.0769]
    am = s92["selection_validity"]["auc_member_clustered"]
    assert am["label"] == "DESCRIPTIVE" and am["clusters"] == 7
    assert [r4(am["mean"]), *map(r4, am["interval"])] == [0.4180, 0.3135, 0.5252]

    # #93 offset sweep and drag grid: issue-93-second-parameterization-v1/comparisons.csv (+ plan, summary).
    root93 = ART / "issue-93-second-parameterization-v1"
    c93 = {r["id"]: r for r in csv.DictReader((root93 / "comparisons.csv").open())}
    tri = lambda r: [r4(r["value"]), r4(r["interval_low"]), r4(r["interval_high"])]  # noqa: E731
    assert c93["offset_ceiling"]["detail"] == "9 of 15" and c93["grid_ceiling"]["detail"] == "14 of 15"
    assert tri(c93["offset_ceiling"]) == [0.6, 0.3333, 0.8667]
    assert tri(c93["grid_ceiling"]) == [0.9333, 0.8, 1.0]
    top93 = {}
    for arm, kn in (("offset", "7/81"), ("grid", "8/126")):
        k_n, chance = c93[f"{arm}_top1"]["detail"].split("; chance ")
        assert k_n == kn, (arm, k_n)
        top93[arm] = (k_n, r4(chance))
        assert c93[f"{arm}_top1_minus_chance"]["unit"] == "member_clustered"
        for s in ("ceiling", "top1_minus_chance", "auc_member_clustered"):
            assert c93[f"{arm}_{s}"]["interval_label"] == "DESCRIPTIVE", (arm, s)
    assert top93 == {"offset": ("7/81", 0.1030), "grid": ("8/126", 0.0804)}
    assert tri(c93["offset_top1_minus_chance"]) == [-0.0166, -0.0781, 0.0575]
    assert tri(c93["grid_top1_minus_chance"]) == [-0.0169, -0.0938, 0.1190]
    assert tri(c93["offset_auc_member_clustered"]) == [0.5512, 0.4235, 0.6685]
    assert tri(c93["grid_auc_member_clustered"]) == [0.6178, 0.5126, 0.7139]
    tok = {"offset": c93["offset_disposition"]["value"], "grid": c93["grid_disposition"]["value"]}
    assert tok == {"offset": "readiness_or_precision_insufficient",
                   "grid": "not_supported_by_this_experiment"}, tok
    assert load(root93 / "summary.json")["dispositions"] == {"C22": tok["grid"], "C23": tok["offset"]}
    rule = load(root93 / "plan.json")["disposition_rule"]
    assert set(tok.values()) <= set(rule["tokens"])
    margins = (rule["supported_max_auc"], rule["not_supported_min_auc"])
    assert margins == (0.5, 0.6), margins

    # #94 launch-power sweep: issue-94-launch-power-v1/comparisons.csv (+ plan.json disposition rules).
    root94 = ART / "issue-94-launch-power-v1"
    c94 = {r["id"]: r for r in csv.DictReader((root94 / "comparisons.csv").open())}
    assert c94["ceiling"]["detail"] == "8 of 15" and tri(c94["ceiling"]) == [0.5333, 0.2667, 0.8]
    k_n94, ch94 = c94["top1"]["detail"].split("; chance ")
    assert (k_n94, r4(ch94), r4(c94["top1"]["value"])) == ("0/72", 0.0521, 0.0)
    assert c94["top1_minus_chance"]["unit"] == "member_clustered"
    assert tri(c94["top1_minus_chance"]) == [-0.0521, -0.0563, -0.05]
    assert c94["auc_member_clustered"]["unit"] == "member_clustered"
    assert tri(c94["auc_member_clustered"]) == [0.7441, 0.6923, 0.8026]
    for s in ("ceiling", "top1_minus_chance", "auc_member_clustered"):
        assert c94[s]["interval_label"] == "DESCRIPTIVE", s
    tok94 = c94["C25_disposition"]["value"]
    assert tok94 == "not_supported_by_this_experiment", tok94
    rule94 = load(root94 / "plan.json")["disposition_rules"]
    assert tok94 in rule94["tokens"]
    assert (rule94["c25_supported_max_auc"], rule94["c25_not_supported_min_auc"]) == margins
    # #95 speed-only (power-only) baseline, post hoc: issue-95-launch-power-followup-v1/comparisons.csv.
    c95 = {r["id"]: r for r in csv.DictReader(
        (ART / "issue-95-launch-power-followup-v1/comparisons.csv").open())}
    base95 = c95["power_only_auc"]
    assert base95["interval_label"] == "DESCRIPTIVE" and "member-clustered" in base95["statistic"]
    assert tri(base95) == [0.6889, 0.6424, 0.7599]

    # The type010101 breadth and held-out zero/few-shot pools are NOT N1 inventories;
    # report their distinct denominators in the manuscript tables, not these four paired rows.
    vi = lambda r: (float(r["value"]), [float(r["interval_low"]), float(r["interval_high"])])  # noqa: E731
    rows = [
        ("Original sweep", (mc["mean"], mc["interval"]), "7/14",
         (pd["mean"], pd["interval"]), "0/108", 0.0780,
         (am["mean"], am["interval"])),
        ("Offset sweep", vi(c93["offset_ceiling"]), "9/15",
         vi(c93["offset_top1_minus_chance"]), "7/81", top93["offset"][1],
         vi(c93["offset_auc_member_clustered"])),
        ("Drag grid", vi(c93["grid_ceiling"]), "14/15",
         vi(c93["grid_top1_minus_chance"]), "8/126", top93["grid"][1],
         vi(c93["grid_auc_member_clustered"])),
        ("Launch power", vi(c94["ceiling"]), "8/15",
         vi(c94["top1_minus_chance"]), "0/72", r4(ch94),
         vi(c94["auc_member_clustered"])),
    ]
    assert [r4(row[1][0]) for row in rows] == [0.5, 0.6, 0.9333, 0.5333]
    assert [r4(row[3][0]) for row in rows] == [-0.0778, -0.0166, -0.0169, -0.0521]
    assert [r4(row[6][0]) for row in rows] == [0.4180, 0.5512, 0.6178, 0.7441]
    assert t1["hits"] == 0 and k_n94 == "0/72"

    fig = plt.figure(figsize=(7.0, 3.1))
    a = fig.add_axes([0.19, 0.19, 0.205, 0.67])
    b = fig.add_axes([0.435, 0.19, 0.225, 0.67], sharey=a)
    c = fig.add_axes([0.72, 0.19, 0.255, 0.67], sharey=a)
    axes = (a, b, c)

    def dot(axis, x, interval, y, colour, marker="o", hollow=False):
        axis.errorbar(x, y, xerr=[[x - interval[0]], [interval[1] - x]], fmt=marker,
                      color=colour, mfc="white" if hollow else colour,
                      mec=colour, ms=5.5, capsize=2.5, lw=1.4, zorder=3)

    for i, (_name, ceiling, numerator, top1, hits, chance, auc) in enumerate(rows):
        y = float(i)
        dot(a, *ceiling, y, OI["blue"])
        a.text(0.99, y + 0.22, numerator, ha="right", va="center", fontsize=8,
               transform=a.get_yaxis_transform())
        dot(b, *top1, y, OI["vermillion"])
        b.text(0.5, y + 0.33, f"{hits} vs {chance:.4f}", ha="center", va="center",
               fontsize=8, transform=b.get_yaxis_transform(),
               bbox=dict(facecolor="white", edgecolor="none", pad=0.5))
        dot(c, *auc, y, OI["green"])
        c.text(0.98, y + 0.22, f"{auc[0]:.4f}", ha="right", va="center", fontsize=8,
               transform=c.get_yaxis_transform())
    # Post-hoc speed-only baseline on the same launch-power inventory, not a model arm.
    speed, interval = vi(base95)
    dot(c, speed, interval, 3.50, OI["grey"], marker="D", hollow=True)
    c.text(0.02, 3.78, "speed-only  0.6889", fontsize=8, va="center",
           transform=c.get_yaxis_transform(),
           bbox=dict(facecolor="white", edgecolor="none", pad=0.5))
    b.axvline(0, color=OI["black"], ls=":", lw=1, zorder=1)
    c.axvline(0.5, color=OI["black"], ls=":", lw=1, zorder=1)
    for axis in axes:
        axis.set_ylim(4.05, -0.65)
        axis.set_yticks(range(4))
        axis.tick_params(axis="y", length=0)
        axis.spines["left"].set_visible(False)
    a.set_yticklabels([row[0] for row in rows], fontsize=8.5)
    for axis in (b, c):
        axis.tick_params(axis="y", labelleft=False)
    a.set_xlim(0, 1.03)
    a.set_xticks([0, 0.5, 1])
    b.set_xlim(-0.11, 0.16)
    b.set_xticks([-0.1, 0, 0.1])
    c.set_xlim(0.28, 0.91)
    c.set_xticks([0.3, 0.5, 0.7, 0.9])
    a.set_title("a  Engine ceiling", loc="left", fontsize=9, weight="bold")
    b.set_title("b  Top-1 minus chance", loc="left", fontsize=9, weight="bold")
    c.set_title("c  Ordering AUC", loc="left", fontsize=9, weight="bold")
    a.set_xlabel("Measurable members")
    b.set_xlabel("Matched uniform draw = 0")
    c.set_xlabel("Chance = 0.5")
    save_figure(fig, "fig_hero_selection_ordering")


# ----------------------------------------------------------------------- taxonomy
def taxonomy():
    """Each contrast varies one factor; the other two remain matched."""
    fig, ax = plt.subplots(figsize=(7.0, 2.15))
    ax.set(xlim=(0, 7), ylim=(0, 3.3))
    ax.axis("off")
    ax.text(0.12, 3.02, "Estimand", fontsize=9, weight="bold")
    ax.text(2.85, 3.02, "What varies", fontsize=9, weight="bold")
    ax.text(4.90, 3.02, "What stays fixed", fontsize=9, weight="bold")
    rows = [
        ("Training  $\\tau_{\\mathrm{tr}}(\\Delta)$", "training recipe",
         "readout and request"),
        ("Execution  $\\tau_{\\mathrm{ex}}(\\Delta,\\alpha)$", "executed readout",
         "checkpoint and horizon"),
        ("Adaptation  $\\tau_{\\mathrm{ad}}$", "selection schedule",
         "training + readout pairs"),
    ]
    for i, (name, varied, fixed) in enumerate(rows):
        y = 2.42 - 0.82 * i
        ax.plot([0.12, 6.85], [y + 0.41, y + 0.41], color="#dddddd", lw=0.8)
        ax.text(0.12, y, name, fontsize=9, va="center")
        ax.text(2.85, y, varied, fontsize=9, va="center", weight="bold")
        ax.text(4.90, y, fixed, fontsize=8.5, va="center")
    save_figure(fig, "estimand_taxonomy")


# ------------------------------------------------------------------- error growth
CLEVRER_MSE = {  # issue-79-clevrer-boundary-v1/findings.md, recursive position MSE, per seed
    "continuous_h1": [[0.0133, 0.0350, 0.1420, 11.6907], [0.0377, 0.1747, 0.6826, 25.3873], [0.0095, 0.0634, 0.2877, 3.5200]],
    "continuous_h5": [[0.0016, 0.0026, 0.0050, 0.0370], [0.0031, 0.0069, 0.0244, 0.0634], [0.0009, 0.0027, 0.0049, 0.0410]],
    "continuous_h15": [[0.0013, 0.0014, 0.0009, 0.0095], [0.0015, 0.0019, 0.0018, 0.0120], [0.0011, 0.0016, 0.0012, 0.0131]],
    "hybrid_continuous_h1": [[0.0071, 0.0494, 0.1828, 2.8006], [0.0183, 0.0747, 0.2465, 2.5099], [0.0242, 0.1027, 0.4672, 3.6363]],
    "hybrid_continuous_h5": [[0.0013, 0.0020, 0.0046, 0.0474], [0.0019, 0.0077, 0.0053, 0.0537], [0.0026, 0.0052, 0.0093, 0.0321]],
    "hybrid_continuous_h15": [[0.0013, 0.0016, 0.0010, 0.0191], [0.0016, 0.0023, 0.0012, 0.0157], [0.0012, 0.0016, 0.0019, 0.0163]],
}


def error_growth():
    s = load(ART / "issue-77-n1-diagnostic-v1/summary.json")["per_seed"]
    seeds = ["20260908", "20260909", "20260910"]
    ends = [15, 30, 60, 150, 225, 600]
    assert round(s["20260908"]["continuous_h1"]["curves"]["150"]["carrier_mse"], 2) == 499.74
    colors = {1: OI["vermillion"], 5: OI["orange"], 15: OI["blue"]}
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 3.0))
    for h in (1, 5, 15):
        for arm, ls, mk in (("continuous", "-", "o"), ("hybrid_continuous", "--", "s")):
            sysname = f"{arm}_h{h}"
            ys = [sum(s[sd][sysname]["curves"][str(e)]["carrier_mse"] for sd in seeds) / 3 for e in ends]
            lab = f"{'hybrid' if arm != 'continuous' else 'continuous'} $h={h}$"
            a.plot(ends, ys, ls=ls, marker=mk, ms=4, lw=1.3, color=colors[h], label=lab)
            cm = [sum(CLEVRER_MSE[sysname][k][i] for k in range(3)) / 3 for i in range(4)]
            b.plot([15, 30, 60, 120], cm, ls=ls, marker=mk, ms=4, lw=1.3, color=colors[h])
    a.axvspan(150, 620, color=OI["grey"], alpha=0.13, lw=0)
    for ax, title, xl in ((a, "a  NovPhy N1 · 4 held-out states", "Elapsed endpoint $t$ (frames)"),
                          (b, "b  CLEVRER · 12 validation scenes", "Elapsed endpoint $e$ (frames)")):
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.set_title(title, loc="left", weight="bold")
        ax.set_xlabel(xl)
        ax.set_ylabel("Seed-mean MSE")
        ax.minorticks_off()
    a.set_xticks(ends)
    a.set_xticklabels([str(e) for e in ends])
    b.set_xticks([15, 30, 60, 120])
    b.set_xticklabels(["15", "30", "60", "120"])
    handles, labels = a.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.98), ncol=3,
               frameon=False, fontsize=8, columnspacing=1.5, handlelength=2.1)
    fig.subplots_adjust(left=0.095, right=0.98, top=0.70, bottom=0.20, wspace=0.36)
    save_figure(fig, "fig_error_growth")


# -------------------------------------------------------------- novelty boundary
def novelty():
    # issue-77-n2-eval-v1/findings.md; checked verbatim against the file text below.
    txt = (ART / "issue-77-n2-eval-v1/findings.md").read_text()
    cross = {  # (pair, condition) -> [(label, change, lo, hi)]
        ("type010101", "zero-shot"): [("hyb-cont $h{=}1$", 0.0662, -0.1008, 0.2288), ("hyb-cont $h{=}15$", -0.1928, -0.4525, 0.0845), ("hyb-macro $h{=}15$", -0.1189, -0.2569, 0.0199)],
        ("type010102", "zero-shot"): [("hyb-cont $h{=}1$", 0.0943, -0.0486, 0.2373), ("hyb-cont $h{=}15$", -0.2693, -0.5412, -0.0050), ("hyb-macro $h{=}15$", 0.4242, 0.0915, 0.7657)],
        ("type010101", "few-shot"): [("hyb-cont $h{=}1$", -0.0476, -0.1489, 0.0384), ("hyb-cont $h{=}15$", 0.0714, -0.1083, 0.2579), ("hyb-macro $h{=}15$", -0.0848, -0.2938, 0.1262)],
        ("type010102", "few-shot"): [("hyb-cont $h{=}1$", -0.0337, -0.2021, 0.1503), ("hyb-cont $h{=}15$", 0.1953, 0.0583, 0.3363), ("hyb-macro $h{=}15$", -0.1892, -0.2720, -0.0945)],
    }
    for vals in cross.values():
        for _, m, lo, hi in vals:
            assert f"{m:+.4f} [{lo:+.4f}, {hi:+.4f}]" in txt, (m, lo, hi)
    exposure = {  # Table tab:exp:adapt: few-shot minus zero-shot, continuous readout
        "type010102": [(1, -0.1970, -0.2688, -0.1253), (5, 0.1850, 0.1012, 0.2688), (15, 0.0672, 0.0432, 0.0912)],
        "type010101": [(1, -0.0866, -0.2422, 0.0690), (5, 0.1544, 0.0062, 0.3025), (15, 0.1080, 0.0339, 0.1820)],
    }
    for vals in exposure.values():
        for _, m, lo, hi in vals:
            assert f"{m:+.4f} [{lo:+.4f}, {hi:+.4f}]" in txt, (m, lo, hi)

    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.85))
    pc = {"type010101": OI["blue"], "type010102": OI["vermillion"]}
    for ax, cond, title in ((axs[0], "zero-shot", "a  Zero-shot (frozen)"),
                            (axs[1], "few-shot", "b  Few-shot (adapted)")):
        for k, pair in enumerate(("type010101", "type010102")):
            for i, (_lab, m, lo, hi) in enumerate(cross[(pair, cond)]):
                y = 2 - i + (0.17 if k == 0 else -0.17)
                ax.errorbar(m, y, xerr=[[m - lo], [hi - m]], fmt="o" if k == 0 else "s", ms=4,
                            color=pc[pair], capsize=2.5, lw=1.1, label=pair if i == 0 else None)
        ax.axvline(0, color=OI["black"], lw=0.8, ls=":")
        ax.set_yticks([2, 1, 0])
        ax.set_yticklabels(["continuous $h=1$", "continuous $h=15$", "macro $h=15$"])
        ax.set_ylim(-0.6, 2.6)
        ax.set_xlim(-0.6, 0.8)
        ax.set_title(title, loc="left", weight="bold")
        ax.set_xlabel("Novel − normal effect")
    axs[1].tick_params(axis="y", labelleft=False)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, frameon=False, fontsize=8,
               bbox_to_anchor=(0.52, 0.98))
    ax = axs[2]
    for k, pair in enumerate(("type010101", "type010102")):
        for h, m, lo, hi in exposure[pair]:
            y = {1: 2, 5: 1, 15: 0}[h] + (0.17 if k == 0 else -0.17)
            ax.errorbar(m, y, xerr=[[m - lo], [hi - m]], fmt="o" if k == 0 else "s", ms=4,
                        color=pc[pair], capsize=2.5, lw=1.1)
    ax.axvline(0, color=OI["black"], lw=0.8, ls=":")
    ax.set_yticks([2, 1, 0])
    ax.set_yticklabels(["continuous $h=1$", "continuous $h=5$", "continuous $h=15$"])
    ax.set_ylim(-0.6, 2.6)
    ax.set_xlim(-0.35, 0.35)
    ax.set_title("c  Exposure contrast", loc="left", weight="bold")
    ax.set_xlabel("Few − zero-shot regret")
    fig.subplots_adjust(left=0.18, right=0.985, top=0.75, bottom=0.24, wspace=0.75)
    save_figure(fig, "fig_novelty_boundary")


# ------------------------------------------------------------------ ceiling map
SYSTEMS = [  # (record system id, legend label, marker, colour); the four frozen #87 systems.
    ("continuous-fixed-h1", "continuous $h{=}1$", "o", OI["vermillion"]),
    ("continuous-fixed-h5", "continuous $h{=}5$", "s", OI["orange"]),
    ("hybrid-fixed-h1", "hybrid $h{=}1$", "^", OI["purple"]),
    ("no-model-ordinal-prior", "no-model prior", "D", OI["grey"]),
]


def ceiling_map():
    s92 = load(ART / "issue-92-selection-validity-v1/summary.json")
    unit = s92["unit_of_analysis"]
    members = unit["members"]
    per_state = load(ART / "issue-89-oracle-completion-v1/summary.json")["ceiling_reestimate"]["per_state"]
    plan = load(ART / "issue-87-closed-loop-oracle-v1/plan.json")

    # Membership: 15 source members / 24 states / 9 byte-identical duplicate pairs.
    states = {st: m for m, rec in members.items() for st in rec["states"]}
    assert len(members) == 15 and len(states) == 24, (len(members), len(states))
    assert set(states) == set(per_state) == {s["state"] for s in plan["decision_cells"]}
    plan_members = {p["execution_member"]["identity"]: p["execution_member"] for p in plan["states"]}
    assert set(plan_members) == set(members)
    dups = set(unit["duplicate_pairs"])
    assert len(dups) == 9 and all(len(members[m]["states"]) == 2 for m in dups)
    assert all(len(members[m]["states"]) == 1 for m in set(members) - dups)
    assert all(e["verdict_discordant_entries"] == 0 and e["inventory_byte_identical"]
               for e in unit["duplicate_evidence"])

    # Per-state successful ordinals: #92 member table == #89 (#87 slots + completion slots).
    succ = {}
    for m, rec in members.items():
        per = [sorted(v["successes"]) for v in rec["states"].values()]
        assert all(p == per[0] for p in per), m  # duplicates carry identical verdicts
        for st in rec["states"]:
            ps = per_state[st]
            assert ps["source_member"] == m
            assert sorted(ps["issue87_successful_ordinals"] + ps["completion_successful_ordinals"]) == per[0]
        succ[m] = per[0]

    # Ceiling indicator and the typed-unmeasurable exclusion (#89 frozen declaration).
    ceiling = sorted(m for m, r in members.items() if r["ceiling"])
    unmeasurable = sorted(m for m, r in members.items() if not r["measurable"])
    assert len(ceiling) == 7 and unit["measurable_members"] == 14 and unmeasurable == ["issue-77-n1-005"]
    assert s92["cited_cross_checks"]["issue_89_typed_unmeasurable"] == ["issue-77-n1-005-a07"]
    assert per_state["issue-77-n1-005-a07"]["typed_unmeasurable"] and succ["issue-77-n1-005"] == [7]
    mc = unit["member_ceiling"]
    assert mc["mean"] == 0.5 and [round(x, 4) for x in mc["interval"]] == [0.2143, 0.7857]
    assert all(set(succ[m]) <= set(range(2, 7)) and succ[m] for m in ceiling)
    assert all(not succ[m] for m in members if m not in ceiling and m not in unmeasurable)

    # Exposure split: only exposure_role == 'training' lineages joined fitting pools.
    unexposed = sorted(m for m, r in members.items() if r["exposure_role"] != "training")
    assert all(plan_members[m]["exposure_role"] == members[m]["exposure_role"] for m in members)
    assert [m[-3:] for m in unexposed] == ["007", "008", "014", "016"]
    split = s92["exposure_split"]
    train_c, calib_c = split["training"]["ceiling_members"], split["calibration"]["ceiling_members"]
    assert [m[-3:] for m in train_c] == ["001", "006", "010", "011", "012"]
    assert [m[-3:] for m in calib_c] == ["014", "016"] and sorted(train_c + calib_c) == ceiling
    assert split["training"]["top1_hits"] == 0 and split["calibration"]["top1_hits"] == 0

    # Chosen ordinals per (member, system), from the #87 decision records.
    chosen = {m: {sid: Counter() for sid, *_ in SYSTEMS} for m in members}
    model_support = Counter()
    for p in sorted((ART / "issue-87-closed-loop-oracle-v1/records").glob("decision--*.json")):
        rec = load(p)
        dec = rec.get("decision") or {}
        if rec.get("failure") is not None or dec.get("failure") is not None or not dec.get("chosen"):
            continue
        o = dec["chosen"]["ordinal"]
        chosen[states[rec["state_identity"]]][rec["system"]][o] += 1
        if rec["system"] != "no-model-ordinal-prior":
            model_support[o] += 1
    sv = s92["selection_validity"]
    assert dict(model_support) == {int(k): v for k, v in sv["chosen_support"].items()}, model_support
    assert sum(model_support.values()) == sv["scored_model_cells"] == 207
    assert sv["chosen_in_band_cells"] == 0
    all_chosen = {o for m in chosen for c in chosen[m].values() for o in c}
    assert all_chosen <= set(range(7, 13)), all_chosen
    # The prior picks ordinal 8 on every ceiling cell (0/36; it picks 7 on non-ceiling member 003).
    prior = Counter()
    for m in ceiling:
        prior.update(chosen[m]["no-model-ordinal-prior"])
    assert dict(prior) == {8: 36}, prior
    assert dict(chosen["issue-77-n1-003"]["no-model-ordinal-prior"]) == {7: 6}
    # 005-a07 carries no bound decision in any of its 12 cells (216 - 207 = its 9 model cells).
    assert not any(chosen["issue-77-n1-005"].values())

    order = sorted(members)
    n = len(order)
    fig, ax = plt.subplots(figsize=(7.0, 4.45))
    fig.subplots_adjust(left=0.15, right=0.82, top=0.79, bottom=0.15)
    ax.axvspan(1.5, 6.5, color=OI["sky"], alpha=0.16, lw=0)
    ax.axvspan(6.5, 12.5, color=OI["vermillion"], alpha=0.07, lw=0)
    for i, m in enumerate(order):
        y = n - 1 - i
        if m in unexposed:
            ax.add_patch(Rectangle((-0.6, y - 0.5), 13.2, 1.0, facecolor=OI["grey"], alpha=0.13, lw=0))
        for o in succ[m]:
            typed = m in unmeasurable
            ax.add_patch(Rectangle((o - 0.42, y - 0.42), 0.84, 0.84, zorder=2, lw=0.8,
                                   facecolor="white" if typed else OI["sky"], edgecolor=OI["blue"],
                                   hatch="////" if typed else None))
        for k, (sid, _, mk, col) in enumerate(SYSTEMS):
            dy = (k - 1.5) * 0.21
            for o, c in chosen[m][sid].items():
                ax.scatter(o, y + dy, s=14 + 5 * c, marker=mk, color=col, lw=0, zorder=3)
        status = "train" if m in train_c else "calib." if m in calib_c else "excluded" if m in unmeasurable else ""
        ax.text(12.85, y, status, va="center", fontsize=7.5, color="#333333")
    ax.set_yticks(range(n))
    ax.set_yticklabels([f"{m[-3:]}" + (" (2)" if m in dups else "") + (" *" if m in unexposed else "")
                        for m in reversed(order)], fontsize=8)
    ax.set_ylabel("N1 source member")
    ax.set_xlim(-0.6, 12.6)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xticks(range(13))
    ax.set_xlabel("Candidate ordinal  (a0 flat  →  a12 steep launch)")
    ax.text(4.0, n - 0.25, "engine success", ha="center", va="bottom", fontsize=8)
    ax.text(9.5, n - 0.25, "frozen choices", ha="center", va="bottom", fontsize=8)
    ax.text(1.01, 1.07, "Ceiling split", transform=ax.transAxes, fontsize=8, va="bottom")
    handles = [
        Rectangle((0, 0), 1, 1, facecolor=OI["sky"], edgecolor=OI["blue"], label="engine success"),
        Rectangle((0, 0), 1, 1, facecolor="white", edgecolor=OI["blue"], hatch="////",
                  label="excluded a07"),
        Rectangle((0, 0), 1, 1, facecolor=OI["grey"], alpha=0.13, label="* unexposed"),
    ] + [plt.Line2D([], [], ls="", marker=mk, color=col, ms=5, label=lab) for _, lab, mk, col in SYSTEMS]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, fontsize=8,
               handlelength=1.5, bbox_to_anchor=(0.50, 0.99), columnspacing=1.2)
    save_figure(fig, "fig_ceiling_map")






if __name__ == "__main__":
    hero()
    hero_selection_ordering()
    taxonomy()
    error_growth()
    novelty()
    ceiling_map()
    teaser_scene()
    qualitative_case()
    print("built:", sorted(p.name for p in OUT.glob("*.pdf")))
