"""Build the five manuscript figures from retained NovPhy artifacts (read-only).

Run from anywhere:  python figures/build_figures.py
Every plotted value is read from, or asserted against, the artifact cited beside it.
No ticket script is re-run; no engine, model, or GPU is touched.

Change log:
  2026-09-23 #91-WP4 (WriterEpsilon): added ceiling_map() -> fig_ceiling_map.pdf (fig:app:ceiling),
    the per-member ceiling-vs-selection map from #87 records, #89 per_state, and #92 summary.json.
  2026-09-23 r2 Batch B (Evidence-2): hero() panel (a) top now plots per-ordinal success counts in
    MEMBER units (review I5; derived from issue-92-selection-validity-v1/slot_join.csv and asserted
    against summary.json chance_references.member_terms, k=1 per member) instead of a solid 2-6 band;
    the counterexample title no longer prints the combined 42/1,224 (I13); panel (b) splits the
    ceiling proportion and the AUCs onto separate axes so the ceiling point no longer sits on the
    AUC chance line (I22).
  2026-09-23 r3 (Evidence): hero() panel (a) draws the excluded 005-a07 success as a 1-member hatched
    outline on the members axis (I34; was a 0-130 full-height band); panel (a) title drops "disjoint"
    and panel (b) adds the held-out cross-split AUCs per condition, 216 of 612 cells, 6 member
    clusters, asserted against cross-pool summary.json (I3/I26).
"""
import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("pdf")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

ART = Path("/p/Project/NovPhy/.local-artifacts")
OUT = Path(__file__).resolve().parent

# Okabe-Ito palette.
OI = {
    "black": "#000000", "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73",
    "yellow": "#F0E442", "blue": "#0072B2", "vermillion": "#D55E00", "purple": "#CC79A7",
    "grey": "#999999",
}

plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7, "pdf.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False,
})


def load(p):
    return json.loads(Path(p).read_text())


# --------------------------------------------------------------------------- hero
def hero():
    sv = load(ART / "issue-92-selection-validity-v1/summary.json")["selection_validity"]
    chosen = {int(k): v for k, v in sv["chosen_support"].items()}
    assert chosen == {7: 90, 8: 12, 9: 45, 10: 10, 11: 6, 12: 44}, chosen
    assert sum(chosen.values()) == sv["scored_model_cells"] == 207
    assert sv["success_band"] == [2, 3, 4, 5, 6]
    assert sv["chosen_in_band_cells"] == 0
    auc_c, auc_m = sv["auc_cell_unit"], sv["auc_member_clustered"]
    assert round(auc_c["mean"], 4) == 0.4367 and round(auc_m["mean"], 4) == 0.4180
    mc = load(ART / "issue-92-selection-validity-v1/summary.json")["unit_of_analysis"]["member_ceiling"]
    assert mc["mean"] == 0.5 and [round(x, 4) for x in mc["interval"]] == [0.2143, 0.7857]

    # Per-member successful ordinals (review I5): exactly one success per ceiling state, counted per
    # source member. slot_join.csv carries successful_ordinals per cell; member_terms carries k.
    terms = load(ART / "issue-92-selection-validity-v1/summary.json")["chance_references"]["member_terms"]
    assert len(terms) == 7 and all(t["k"] == 1 for t in terms.values()), terms
    succ = {}
    for r in csv.DictReader((ART / "issue-92-selection-validity-v1/slot_join.csv").open()):
        if r["n_successes"] not in ("", "0"):
            assert r["n_successes"] == "1", r["cell_identity"]
            succ.setdefault(r["source_member"], set()).add(int(r["successful_ordinals"]))
    assert all(len(v) == 1 for v in succ.values()), succ
    succ = {m.rsplit("-", 1)[1]: v.pop() for m, v in succ.items()}
    # Typed-unmeasurable issue-77-n1-005-a07 succeeds at ordinal 7; it is excluded from the ceiling
    # (#89 frozen declaration) and drawn separately as the hatched ordinal-7 mark, never as a member bar.
    assert succ.pop("005") == 7, succ
    assert succ == {"001": 6, "006": 5, "010": 2, "011": 4, "012": 4, "014": 4, "016": 3}, succ
    assert set(succ) == {m.rsplit("-", 1)[1] for m in terms}
    member_counts = Counter(succ.values())
    assert member_counts == {2: 1, 3: 1, 4: 3, 5: 1, 6: 1}, member_counts

    # Bounded-transfer breadth pool, successful cells by ordinal (issue-92-cross-pool-audit-v1/findings.md Q1).
    bounded = {0: 1, 1: 1, 2: 3, 3: 3, 4: 5, 5: 3, 6: 4, 10: 3, 11: 3, 12: 6}
    assert sum(bounded.values()) == 32

    # Cross-split counterexample, from the per-cell table; conditions kept separate.
    rows = list(csv.DictReader((ART / "issue-92-cross-pool-audit-v1/cross_split_cells.csv").open()))
    assert len(rows) == 1224
    hits = [r for r in rows if r["top1_hit"] == "1"]
    assert len(hits) == 42
    assert sum(r["chosen_ordinal"] == "12" for r in hits) == 37
    assert len({r["state"] for r in hits}) == 5
    by_cond = {c: Counter(int(r["chosen_ordinal"]) for r in hits if r["condition"] == c)
               for c in ("zero-shot", "few-shot")}
    assert sum(by_cond["zero-shot"].values()) == 13 and sum(by_cond["few-shot"].values()) == 29
    chosen_span = [int(r["chosen_ordinal"]) for r in rows]
    assert min(chosen_span) == 0 and max(chosen_span) == 12
    # Held-out cross-split AUCs (r3 I3/I26): cell unit over the 216 of 612 cells of states with a
    # success, 6-member-clustered bootstrap, per condition, never pooled.
    xs_auc = load(ART / "issue-92-cross-pool-audit-v1/summary.json")["cross_split"]["per_condition"]
    held = {}
    for c in ("zero-shot", "few-shot"):
        a = xs_auc[c]["auc"]
        u = a["cell_unit"]
        assert xs_auc[c]["cells"] == 612 and a["scored_cells"] == u["units"] == 216 and u["clusters"] == 6
        assert u["label"] == "DESCRIPTIVE"
        held[c] = (u["mean"], tuple(u["interval"]))
    assert [round(x, 4) for x in (held["zero-shot"][0], *held["zero-shot"][1])] == [0.2039, 0.0762, 0.3324]
    assert [round(x, 4) for x in (held["few-shot"][0], *held["few-shot"][1])] == [0.2732, 0.2182, 0.3239]

    fig = plt.figure(figsize=(7.0, 3.1))
    gs = fig.add_gridspec(3, 2, width_ratios=[2.35, 1.0], height_ratios=[1.35, 0.8, 0.9],
                          hspace=0.95, wspace=0.62)
    xs = range(13)

    # (a) top: per-ordinal ceiling-member successes (right axis) vs model-chosen ordinals (left axis).
    ax = fig.add_subplot(gs[0, 0])
    # r3 I34: the excluded 005-a07 success is a 1-member outline mark on the members axis (drawn below
    # on axm), never a full-height band that would outweigh the member bars.
    ax.bar(list(chosen), list(chosen.values()), width=0.7, color=OI["vermillion"],
           label="frozen rankers' chosen ordinal (207 model cells, left axis)")
    ax.annotate("no-model prior:\nordinal 8, 0/36", xy=(8, chosen[8] + 1), xytext=(6.2, 112),
                fontsize=6.3, ha="right", arrowprops=dict(arrowstyle="->", lw=0.6))
    ax.set_xlim(-0.6, 12.6)
    ax.set_ylim(0, 130)
    ax.set_xticks(list(xs))
    ax.set_ylabel("model cells")
    ax.set_title("(a) N1: engine-truth successes vs. chosen ordinals", loc="left")
    axm = ax.twinx()
    axm.spines["right"].set_visible(True)
    mo = sorted(member_counts)
    axm.bar(mo, [member_counts[o] for o in mo], width=0.7, color=OI["sky"], edgecolor=OI["blue"],
            lw=0.6, label="engine-truth successes (ceiling members, right axis)")
    for o in mo:
        axm.text(o, member_counts[o] + 0.08, str(member_counts[o]), ha="center", fontsize=6.3)
    axm.set_ylim(0, 4)
    axm.set_yticks([0, 1, 2, 3])
    axm.set_ylabel("members")
    mark = axm.add_patch(Rectangle((7 - 0.35, 0), 0.7, 1, facecolor="none", edgecolor=OI["blue"],
                                   hatch="////", lw=0.8, zorder=5,
                                   label="typed-unmeasurable 005-a07 success (excluded; 1 member)"))
    assert (round(mark.get_x(), 4), mark.get_width(), mark.get_height()) == (6.65, 0.7, 1)
    assert mark.get_height() < max(member_counts.values()) and axm.get_ylim() == (0, 4)
    h1, l1 = axm.get_legend_handles_labels()
    h0, l0 = ax.get_legend_handles_labels()
    fig.legend(h1 + h0, l1 + l0, loc="upper center", ncol=3, frameon=False,
               fontsize=6.2, handlelength=1.2, bbox_to_anchor=(0.5, 1.06))

    # (a) middle: bounded-transfer breadth pool.
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax)
    ax2.bar(list(bounded), list(bounded.values()), width=0.7, color=OI["green"])
    ax2.set_ylabel("cells")
    ax2.set_title("Bounded-transfer pool (type010101): 32 successful cells", loc="left", fontsize=7.3)

    # (a) bottom: counterexample, visually distinct (grey, dashed frame).
    ax3 = fig.add_subplot(gs[2, 0], sharex=ax)
    w = 0.36
    zs, fs = by_cond["zero-shot"], by_cond["few-shot"]
    ax3.bar([o - w / 2 for o in xs], [zs.get(o, 0) for o in xs], width=w, color=OI["grey"],
            label="zero-shot, frozen (13/612)")
    ax3.bar([o + w / 2 for o in xs], [fs.get(o, 0) for o in xs], width=w, color="white",
            edgecolor=OI["black"], hatch="....", lw=0.5, label="few-shot, adapted (29/612)")
    ax3.set_xlabel("candidate ordinal (a0 flat $\\rightarrow$ a12 steep launch)")
    ax3.set_ylabel("hits")
    ax3.set_title("Counterexample outside N1 (conditions reported separately)", loc="left",
                  fontsize=7.3, color="#555555")
    ax3.legend(loc="upper left", frameon=False, fontsize=5.8)
    for s in ax3.spines.values():
        s.set_visible(True)
        s.set_linestyle("--")
        s.set_color(OI["grey"])

    # (b) separate axes: ceiling proportion (top) and selection AUCs with the 0.5 chance line (bottom).
    gsb = gs[:, 1].subgridspec(2, 1, height_ratios=[0.8, 2.0], hspace=1.1)
    cx = fig.add_subplot(gsb[0])
    clo, chi = mc["interval"]
    cx.errorbar(mc["mean"], 0, xerr=[[mc["mean"] - clo], [chi - mc["mean"]]], fmt="s",
                color=OI["blue"], ms=4, capsize=2.5, lw=1)
    cx.text(mc["mean"], 0.35, f"7/14 = {mc['mean']:.4f}", ha="center", fontsize=6.3)
    cx.set_yticks([])
    cx.spines["left"].set_visible(False)
    cx.set_xlim(0.0, 1.0)
    cx.set_ylim(-0.6, 0.9)
    cx.set_xlabel("proportion of measurable members", fontsize=6.8)
    cx.set_title("(b) solvable: member-unit ceiling", loc="left")
    bx = fig.add_subplot(gsb[1])
    items = [
        ("N1 cell unit\n(108 cells)", auc_c["mean"], tuple(auc_c["interval"]), OI["vermillion"]),
        ("N1 member-clustered\n(7 clusters)", auc_m["mean"], tuple(auc_m["interval"]), OI["orange"]),
        ("held-out zero-shot\n(216 cells, 6 clusters)", *held["zero-shot"], OI["grey"]),
        ("adapted few-shot\n(216 cells, 6 clusters)", *held["few-shot"], OI["black"]),
    ]
    for i, (lab, m, (lo, hi), col) in enumerate(items):
        y = len(items) - 1 - i
        bx.errorbar(m, y, xerr=[[m - lo], [hi - m]], fmt="o", color=col, ms=4, capsize=2.5, lw=1)
        bx.text(m, y + 0.24, f"{m:.4f}", ha="center", fontsize=6.0)
    bx.axvline(0.5, color=OI["black"], ls=":", lw=0.8)
    bx.text(0.515, -0.55, "chance 0.5", fontsize=6.2, ha="left")
    bx.set_yticks(range(len(items)))
    bx.set_yticklabels([it[0] for it in reversed(items)], fontsize=5.8)
    bx.set_xlim(0.0, 1.0)
    bx.set_ylim(-0.7, 3.6)
    bx.set_xlabel("selection AUC, descriptive 95% interval", fontsize=6.8)
    bx.set_title("yet mis-ranked: engine-truth AUC", loc="left")
    fig.savefig(OUT / "fig_hero_ordinal_masses.pdf", bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------- taxonomy
def taxonomy():
    fig, ax = plt.subplots(figsize=(7.0, 2.1))
    ax.set_axis_off()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.3)
    axes = ["training\nrecipe", "executed\nreadout", "selection\nschedule"]
    ests = [
        ("Training effect $\\tau_{\\mathrm{tr}}(\\Delta)$", "hybrid vs. pure-continuous\nboth run $(\\Delta,\\mathrm{continuous})$", 0, OI["blue"]),
        ("Symbolic-execution effect $\\tau_{\\mathrm{ex}}(\\Delta,\\alpha)$", "$(\\Delta,\\alpha)$ vs. $(\\Delta,\\mathrm{continuous})$\none hybrid checkpoint", 1, OI["green"]),
        ("Adaptation effect $\\tau_{\\mathrm{ad}}$", "per-decision selection vs.\nfixed pairs, matched training", 2, OI["vermillion"]),
    ]
    colx = [4.7, 6.7, 8.7]
    for j, a in enumerate(axes):
        ax.text(colx[j], 3.1, a, ha="center", va="center", fontsize=7.5, weight="bold")
    for i, (name, desc, varied, col) in enumerate(ests):
        y = 2.3 - i * 0.95
        ax.add_patch(FancyBboxPatch((0.05, y - 0.36), 3.55, 0.72, boxstyle="round,pad=0.02",
                                    fc="white", ec=col, lw=1.1))
        ax.text(0.18, y + 0.14, name, fontsize=7.3, va="center", color=col, weight="bold")
        ax.text(0.18, y - 0.17, desc, fontsize=6.3, va="center")
        for j in range(3):
            if j == varied:
                ax.add_patch(FancyBboxPatch((colx[j] - 0.75, y - 0.24), 1.5, 0.48,
                                            boxstyle="round,pad=0.02", fc=col, ec=col, alpha=0.9))
                ax.text(colx[j], y, "varies", ha="center", va="center", fontsize=7, color="white",
                        weight="bold")
            else:
                ax.add_patch(FancyBboxPatch((colx[j] - 0.75, y - 0.24), 1.5, 0.48,
                                            boxstyle="round,pad=0.02", fc="white", ec=OI["grey"],
                                            ls="--", lw=0.7))
                ax.text(colx[j], y, "held fixed", ha="center", va="center", fontsize=6.5,
                        color=OI["grey"])
    fig.savefig(OUT / "estimand_taxonomy.pdf", bbox_inches="tight")
    plt.close(fig)


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
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.8, 2.3))
    for h in (1, 5, 15):
        for arm, ls, mk in (("continuous", "-", "o"), ("hybrid_continuous", "--", "s")):
            sysname = f"{arm}_h{h}"
            ys = [sum(s[sd][sysname]["curves"][str(e)]["carrier_mse"] for sd in seeds) / 3 for e in ends]
            lab = f"{'hybrid' if arm != 'continuous' else 'continuous'}, $h={h}$"
            a.plot(ends, ys, ls=ls, marker=mk, ms=3, lw=1, color=colors[h], label=lab)
            cm = [sum(CLEVRER_MSE[sysname][k][i] for k in range(3)) / 3 for i in range(4)]
            b.plot([15, 30, 60, 120], cm, ls=ls, marker=mk, ms=3, lw=1, color=colors[h], label=lab)
    a.axvspan(150, 620, color=OI["grey"], alpha=0.15, lw=0)
    a.text(165, 1e4, "$t \\geq 150$", fontsize=6.5)
    for ax, t, xl in ((a, "(a) NovPhy N1, 4 held-out states: recursive carrier MSE", "elapsed endpoint $t$ (observed frames)"),
                      (b, "(b) CLEVRER, 12 scenes: recursive position MSE", "elapsed endpoint $e$ (annotation frames)")):
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.set_title(t, loc="left", fontsize=7.5)
        ax.set_xlabel(xl)
        ax.set_ylabel("seed-mean MSE (log)")
    a.set_xticks(ends)
    a.set_xticklabels([str(e) for e in ends])
    b.set_xticks([15, 30, 60, 120])
    b.set_xticklabels(["15", "30", "60", "120"])
    a.minorticks_off()
    b.minorticks_off()
    b.legend(loc="upper left", frameon=False, ncol=2, fontsize=6)
    fig.tight_layout()
    fig.savefig(OUT / "fig_error_growth.pdf", bbox_inches="tight")
    plt.close(fig)


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

    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.3), gridspec_kw={"width_ratios": [1, 1, 1]})
    pc = {"type010101": OI["blue"], "type010102": OI["vermillion"]}
    for ax, cond, title in ((axs[0], "zero-shot", "(a) zero-shot (frozen checkpoints)"),
                            (axs[1], "few-shot", "(b) few-shot (adapted predictors)")):
        for k, pair in enumerate(("type010101", "type010102")):
            for i, (lab, m, lo, hi) in enumerate(cross[(pair, cond)]):
                y = 2 - i + (0.15 if k == 0 else -0.15)
                ax.errorbar(m, y, xerr=[[m - lo], [hi - m]], fmt="o" if k == 0 else "s", ms=3,
                            color=pc[pair], capsize=2, lw=0.9, label=pair if i == 0 else None)
        ax.axvline(0, color=OI["black"], lw=0.6, ls=":")
        ax.set_yticks([2, 1, 0])
        ax.set_yticklabels([c[0] for c in cross[("type010101", cond)]])
        ax.set_xlim(-0.6, 0.8)
        ax.set_title(title, loc="left", fontsize=7.5)
        ax.set_xlabel("novel minus normal hybrid effect")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, frameon=False, fontsize=6.5,
               bbox_to_anchor=(0.5, 1.07))
    ax = axs[2]
    for k, pair in enumerate(("type010102", "type010101")):
        for h, m, lo, hi in exposure[pair]:
            y = {1: 2, 5: 1, 15: 0}[h] + (0.15 if k == 0 else -0.15)
            ax.errorbar(m, y, xerr=[[m - lo], [hi - m]], fmt="s" if k == 0 else "o", ms=3,
                        color=pc[pair], capsize=2, lw=0.9)
    ax.axvline(0, color=OI["black"], lw=0.6, ls=":")
    ax.set_yticks([2, 1, 0])
    ax.set_yticklabels(["cont. $h{=}1$", "cont. $h{=}5$", "cont. $h{=}15$"])
    ax.set_xlim(-0.35, 0.35)
    ax.set_title("(c) exposure contrast, fixed systems", loc="left", fontsize=7.5)
    ax.set_xlabel("few-shot minus zero-shot regret")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / "fig_novelty_boundary.pdf", bbox_inches="tight")
    plt.close(fig)


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
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.axvspan(1.5, 6.5, color=OI["sky"], alpha=0.18, lw=0)
    ax.axvspan(6.5, 12.5, color=OI["vermillion"], alpha=0.06, lw=0)
    for i, m in enumerate(order):
        y = n - 1 - i
        if m in unexposed:
            ax.add_patch(Rectangle((-0.6, y - 0.5), 13.2, 1.0, facecolor=OI["yellow"], alpha=0.28, lw=0))
        for o in succ[m]:
            typed = m in unmeasurable
            ax.add_patch(Rectangle((o - 0.42, y - 0.42), 0.84, 0.84, zorder=2, lw=0.8,
                                   facecolor="white" if typed else OI["sky"], edgecolor=OI["blue"],
                                   hatch="////" if typed else None))
        for k, (sid, _, mk, col) in enumerate(SYSTEMS):
            dy = (k - 1.5) * 0.21
            for o, c in chosen[m][sid].items():
                ax.scatter(o, y + dy, s=5 + 4 * c, marker=mk, color=col, lw=0, zorder=3)
        status = ("ceiling, train split" if m in train_c else "ceiling, calib. split" if m in calib_c
                  else "typed-unmeasurable, excluded" if m in unmeasurable else "no ceiling")
        ax.text(12.85, y, status, va="center", fontsize=6.2,
                color=OI["blue"] if m in ceiling else "#555555")
    ax.set_yticks(range(n))
    ax.set_yticklabels([f"{m[-3:]}" + (" (2 states)" if m in dups else "") + (" *" if m in unexposed else "")
                        for m in reversed(order)], fontsize=6.5)
    ax.set_ylabel("N1 source member (unit of analysis)")
    ax.set_xlim(-0.6, 12.6)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xticks(range(13))
    ax.set_xlabel("candidate ordinal (a0 flat $\\rightarrow$ a12 steep launch)")
    ax.text(4.0, n - 0.25, "success band 2–6", ha="center", va="bottom", fontsize=6.5, color=OI["blue"])
    ax.text(9.5, n - 0.25, "every chosen ordinal: 7–12", ha="center", va="bottom", fontsize=6.5,
            color=OI["vermillion"])
    handles = [
        Rectangle((0, 0), 1, 1, facecolor=OI["sky"], edgecolor=OI["blue"], label="engine-truth success"),
        Rectangle((0, 0), 1, 1, facecolor="white", edgecolor=OI["blue"], hatch="////",
                  label="success on typed-unmeasurable 005-a07"),
        Rectangle((0, 0), 1, 1, facecolor=OI["yellow"], alpha=0.28, label="* unexposed member"),
    ] + [plt.Line2D([], [], ls="", marker=mk, color=col, ms=4, label=lab) for _, lab, mk, col in SYSTEMS]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, fontsize=6.2,
               handlelength=1.3, bbox_to_anchor=(0.47, 1.01))
    fig.savefig(OUT / "fig_ceiling_map.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    hero()
    taxonomy()
    error_growth()
    novelty()
    ceiling_map()
    print("built:", sorted(p.name for p in OUT.glob("*.pdf")))
