"""Observed AUC-MAC frontier on the sealed NovPhy grid (fig:res:frontier).

Run: python figures/build_frontier.py
Input (read-only): <NovPhy>/.local-artifacts/issue-103-engine-truth-frontier-v1/
frontier-table.csv, rows set == primary and inventory == grid. <NovPhy> defaults
to the repository root that contains manuscript/; override with NOVPHY_ROOT.
Every plotted value is a CSV field: x = macs_per_candidate / 1e6 (linear MACs per
candidate rollout to t=225, predictor + policy), y = auc with [auc_low, auc_high]
(95% level-clustered bootstrap), frontier = rows with on_auc_macs == 1.
Outputs: fig_frontier.{pdf,svg,png} beside this script; prints the plotted table.
"""
import csv
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUT = Path(__file__).resolve().parent
NOVPHY = Path(os.environ.get("NOVPHY_ROOT", OUT.parents[2]))
TABLE = NOVPHY / ".local-artifacts/issue-103-engine-truth-frontier-v1/frontier-table.csv"

INK, GREY, FRONTIER = "#18252d", "#637382", "#BBBBBB"
ALPHA_COLOUR = {"continuous": "#0072B2", "micro": "#CC79A7", "macro": "#D55E00"}
ALPHA_SHORT = {"continuous": "cont", "micro": "micro", "macro": "macro"}
# Learned request-policy sweeps: (record prefix, label, colour, line style, operating-point lambda).
SWEEPS = [
    ("J-", r"$J$", "#000000", "-", "1e-04"),
    ("Jcont-", r"$J_{\mathrm{cont}}$", "#56B4E9", (0, (4, 1.5)), "1e-04"),
    ("CJ-", r"$C\!-\!J$", "#E69F00", (0, (1, 1.2)), "1e-04"),
    ("J-dec-", r"$J_{\mathrm{dec}}$", "#009E73", (0, (4, 1.2, 1, 1.2)), "1e-01"),
]
ZOOM = (-1.6, 52.0)
YLIM = (0.608, 0.803)


def load():
    with TABLE.open(newline="") as handle:
        rows = [r for r in csv.DictReader(handle)
                if r["set"] == "primary" and r["inventory"] == "grid"]
    for r in rows:
        r["x"] = float(r["macs_per_candidate"]) / 1e6
        r["y"], r["lo"], r["hi"] = (float(r[k]) for k in ("auc", "auc_low", "auc_high"))
        r["front"] = r["on_auc_macs"] == "1"
    return {r["policy"]: r for r in rows}


def check_frontier(rows):
    """Recorded membership must equal the observed (weak-dominance) envelope over all rows."""
    envelope = {p for p, r in rows.items() if not any(
        o["x"] <= r["x"] and o["y"] >= r["y"] and (o["x"], o["y"]) != (r["x"], r["y"])
        for o in rows.values())}
    recorded = {p for p, r in rows.items() if r["front"]}
    assert envelope == recorded, (envelope, recorded)
    return sorted(recorded, key=lambda p: rows[p]["x"])


def sweep(rows, prefix):
    # "J-" must not capture the J-dec-/J-pg- families.
    pts = [r for p, r in rows.items() if p.startswith(prefix)
           and p[len(prefix):].replace("e-", "").isdigit()]
    return sorted(pts, key=lambda r: float(r["policy"][len(prefix):]))


def draw(rows):
    front = check_frontier(rows)
    fixed = [rows[f"F-{d}-{a}"] for d in (1, 5, 15) for a in ALPHA_COLOUR]
    cfixed = [rows[f"C-F-{d}"] for d in (1, 5, 15)]
    prior = rows["prior"]
    sweeps = [(spec, sweep(rows, spec[0])) for spec in SWEEPS]
    plotted = [prior] + fixed + cfixed + [r for _, pts in sweeps for r in pts]

    rc = {"font.family": "DejaVu Sans", "font.size": 7, "mathtext.fontset": "dejavusans",
          "axes.linewidth": .6, "xtick.major.width": .6, "ytick.major.width": .6,
          "xtick.major.size": 2.5, "ytick.major.size": 2.5, "axes.labelsize": 7,
          "xtick.labelsize": 7, "ytick.labelsize": 7, "svg.fonttype": "none",
          "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False}
    with plt.rc_context(rc):
        fig = plt.figure(figsize=(5.5, 2.75), facecolor="white")
        full = fig.add_axes([.093, .282, .355, .661])
        zoom = fig.add_axes([.505, .282, .485, .661], sharey=full)
        full.set_xlim(-12, 435)
        zoom.set_xlim(*ZOOM)
        full.set_ylim(*YLIM)
        full.set_ylabel("Ordering AUC", labelpad=3)
        fig.text(.54, .173, "Linear MACs per candidate (millions)", ha="center", va="center")
        plt.setp(zoom.get_yticklabels(), visible=False)
        fig.text(.008, .975, "a", fontsize=8, fontweight="bold", va="top")
        fig.text(.478, .975, "b", fontsize=8, fontweight="bold", va="top")
        full.axvspan(*ZOOM, color="#eef1f3", lw=0, zorder=0)
        full.text(ZOOM[1] + 5, .611, "b", fontsize=7, color=GREY, va="bottom")

        for ax in (full, zoom):
            xmax = ax.get_xlim()[1]
            ax.axhline(prior["y"], color=GREY, lw=.7, ls=(0, (4, 2.5)), zorder=1)
            fx = [rows[p]["x"] for p in front] + [xmax]
            fy = [rows[p]["y"] for p in front]
            ax.step(fx, fy + fy[-1:], where="post", color=FRONTIER, lw=3.2, zorder=.5,
                    solid_joinstyle="miter", solid_capstyle="butt")

            def ci(r, colour, lw=.6, alpha=.55):
                ax.plot([r["x"], r["x"]], [r["lo"], r["hi"]], color=colour, lw=lw,
                        alpha=alpha, zorder=3, solid_capstyle="butt")

            for (prefix, _, colour, ls, chosen), pts in sweeps:
                ax.plot([r["x"] for r in pts], [r["y"] for r in pts], color=colour, lw=.8,
                        ls=ls, zorder=4, alpha=.9)
                for r in pts:
                    ci(r, colour, alpha=.4)
                    ax.plot(r["x"], r["y"], "o", ms=2.6, mfc=colour, mew=0, zorder=5)
                    if r["policy"] == prefix + chosen:  # operating point: ring above fixed marks
                        ax.plot(r["x"], r["y"], "o", ms=7, mfc="none", mec=colour, mew=1.2,
                                zorder=8)
                        ax.plot(r["x"], r["y"], "o", ms=2.6, mfc=colour, mew=0, zorder=8)
            for r in fixed:
                colour = ALPHA_COLOUR[r["policy"].split("-")[2]]
                ci(r, colour, lw=.7, alpha=.75)
                ax.plot(r["x"], r["y"], "o", ms=4.4, mfc=colour, mec="white", mew=.5, zorder=7)
            for r in cfixed:
                ci(r, ALPHA_COLOUR["continuous"], lw=.7, alpha=.75)
                ax.plot(r["x"], r["y"], "D", ms=4.6, mfc="white", mec=ALPHA_COLOUR["continuous"],
                        mew=1.1, zorder=7)
            ci(prior, INK, lw=.8, alpha=.9)
            ax.plot(prior["x"], prior["y"], "s", ms=4, mfc=INK, mec="white", mew=.5, zorder=8,
                    clip_on=False)

        def label(ax, policy, text, xy, colour=INK, ha="left", leader=True):
            r = rows[policy]
            ax.annotate(text, (r["x"], r["y"]), xytext=xy, textcoords="data", fontsize=7,
                        color=colour, ha=ha, va="center", zorder=9,
                        arrowprops=dict(arrowstyle="-", color=GREY, lw=.5, shrinkA=1.5,
                                        shrinkB=2.5) if leader else None)

        # a: full MAC range; the highlighted operating points of the three lambda sweeps.
        label(full, "J-1e-04", r"$J$", (205, .735))
        label(full, "Jcont-1e-04", r"$J_{\mathrm{cont}}$", (128, .655))
        label(full, "CJ-1e-04", r"$C\!-\!J$", (250, .675))
        for d, xy in ((1, (362, .700)), (5, (90, .757))):
            full.text(*xy, rf"$\Delta$={d}", fontsize=7, color=GREY, ha="left" if d == 5 else "center")
        # b: frontier members and the two named fixed requests, once each.
        label(zoom, "C-F-15", r"$C^*$ = C-F(15)", (31.5, .792))
        label(zoom, "J-1e-01", r"$J$, $\lambda$=10$^{-1}$", (19.0, .7915), ha="right")
        label(zoom, "F-15-continuous", "F(15,cont)", (19.0, .7695), ha="right")
        label(zoom, "F-15-macro", r"$U$ = F(15,macro)", (19.0, .736), ha="right")
        label(zoom, "J-dec-1e-01", r"$J_{\mathrm{dec}}$", (27.5, .728))
        zoom.text(1.0, prior["y"] - .004, "no-model prior\n(0 model MACs)", fontsize=7,
                  color=GREY, ha="left", va="top", linespacing=1.1)

        handles = [Line2D([], [], ls="", marker="o", ms=4.4, mfc=c, mec="white", mew=.5,
                          label=f"F($\\Delta$,{ALPHA_SHORT[a]})") for a, c in ALPHA_COLOUR.items()]
        handles.append(Line2D([], [], ls="", marker="D", ms=4.6, mfc="white",
                              mec=ALPHA_COLOUR["continuous"], mew=1.1, label=r"C-F($\Delta$)"))
        handles.append(Line2D([], [], ls="", marker="o", ms=7, mfc="none", mec=GREY, mew=1.2,
                              label="operating point"))
        for _, name, colour, ls, chosen in SWEEPS:
            handles.append(Line2D([], [], color=colour, ls=ls, lw=.8, marker="o", ms=2.6,
                                  label=name))
        handles.append(Line2D([], [], color=FRONTIER, lw=3.2, label="observed frontier"))
        # Legend fills column-major: interleave so row 1 = fixed requests + prior, row 2 = sweeps.
        handles = [h for pair in zip(handles[:5], handles[5:]) for h in pair]
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.54, .005), ncol=5,
                   frameon=False, fontsize=7, handlelength=2.4, columnspacing=1.4,
                   handletextpad=.5, borderaxespad=0, labelspacing=.35)

        for ext in ("pdf", "svg"):
            fig.savefig(OUT / f"fig_frontier.{ext}", facecolor="white")
        fig.savefig(OUT / "fig_frontier.png", dpi=300, facecolor="white")
        plt.close(fig)

    print(f"source: {TABLE.relative_to(NOVPHY.parent)}")
    print(f"{'policy':16s} {'MACs (M)':>9s} {'AUC':>7s}  95% CI            frontier")
    for r in plotted:
        print(f"{r['policy']:16s} {r['x']:9.3f} {r['y']:.4f}  [{r['lo']:.4f}, {r['hi']:.4f}]"
              f"  {'yes' if r['front'] else ''}")
    print("frontier (on_auc_macs == 1):", ", ".join(front))


if __name__ == "__main__":
    draw(load())
