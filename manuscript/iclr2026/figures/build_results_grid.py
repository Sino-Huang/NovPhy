"""Draw fixed-request ordering AUC and cross-fitted references at print size.

Run: python figures/build_results_grid.py
Input: the read-only comparisons.csv below (no manuscript-derived data).
Output: fig_results_grid.pdf and fig_results_grid.png beside this script.

Each panel is one launch set; cells are member-clustered mean ordering AUCs.
The shared diverging scale marks 0.5 as no ordering. Descriptive intervals
are available in the CSV but are not encoded in this compact mean heatmap.
Solid outlines select the best fixed request on the displayed set; dashed
outlines select the best equally weighted mean on the OTHER three sets.
Selection uses unrounded values, with ties resolved in delta-major order,
then continuous, micro, macro. Policy values are not part of selection.
"""
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.cm import ScalarMappable
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle


SOURCE = Path(
    "/p/Project/NovPhy/.local-artifacts/"
    "issue-96-tau-ad-within-checkpoint-v1/comparisons.csv"
)
OUT = Path(__file__).resolve().parent
SETS = (
    ("angle", "Launch-angle sweep"),
    ("offset", "Offset sweep"),
    ("grid", "Drag grid"),
    ("power", "Launch-power sweep"),
)
REQUESTS = tuple(
    (delta, alpha)
    for delta in (1, 5, 15)
    for alpha in ("continuous", "micro", "macro")
)
EXPECTED_FSTAR = {
    "angle": (1, "macro"),
    "offset": (1, "macro"),
    "grid": (5, "macro"),
    "power": (5, "macro"),
}
INK, GREY, REQUEST = "#18252d", "#637382", "#007C78"
SIZE = (5.5, 1.75)


def load_results():
    """Read all required AUCs and assert the leave-one-set-out selections."""
    with SOURCE.open(newline="") as source:
        rows = {row["id"]: row for row in csv.DictReader(source)}
    required = {
        f"{inv}_auc_F-{delta}-{alpha}"
        for inv, _ in SETS
        for delta, alpha in REQUESTS
    } | {f"{inv}_auc_J" for inv, _ in SETS}
    missing = required - rows.keys()
    assert not missing, f"Missing required AUC ids: {sorted(missing)}"
    fixed = {
        inv: [float(rows[f"{inv}_auc_F-{d}-{a}"]["value"]) for d, a in REQUESTS]
        for inv, _ in SETS
    }
    policy = {inv: float(rows[f"{inv}_auc_J"]["value"]) for inv, _ in SETS}
    reference, hindsight = {}, {}
    for inv, _ in SETS:
        other_means = [
            sum(fixed[other][j] for other, _ in SETS if other != inv) / 3
            for j in range(len(REQUESTS))
        ]
        # max returns the first maximum, preserving the specified tie order.
        reference[inv] = max(range(len(REQUESTS)), key=other_means.__getitem__)
        hindsight[inv] = max(range(len(REQUESTS)), key=fixed[inv].__getitem__)
    actual = {inv: REQUESTS[reference[inv]] for inv, _ in SETS}
    assert actual == EXPECTED_FSTAR, (
        f"Cross-fitted F* mismatch: computed {actual}; expected {EXPECTED_FSTAR}"
    )
    return fixed, policy, reference, hindsight


def draw():
    fixed, policy, reference, hindsight = load_results()
    style = {
        "font.family": "DejaVu Sans",
        "font.size": 7,
        "mathtext.fontset": "dejavusans",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
    }
    with plt.rc_context(style):
        fig = plt.figure(figsize=SIZE, facecolor="white")
        norm = TwoSlopeNorm(vmin=0.38, vcenter=0.5, vmax=0.66)
        cmap = plt.get_cmap("PuOr")
        for panel, (inv, title) in enumerate(SETS):
            left, width = 0.46 + 1.25 * panel, 1.14
            ax = fig.add_axes([left / SIZE[0], 0.64 / SIZE[1],
                               width / SIZE[0], 0.82 / SIZE[1]])
            ax.set(xlim=(-0.5, 2.5), ylim=(2.5, -0.5))
            ax.set_xticks(range(3), ("cont", "micro", "macro"))
            ax.xaxis.tick_top()
            ax.tick_params(axis="both", length=0, labelsize=7, pad=3)
            if panel == 0:
                ax.set_yticks(range(3), ("Δ=1", "Δ=5", "Δ=15"))
            else:
                ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            for j, value in enumerate(fixed[inv]):
                row, col = divmod(j, 3)
                color = cmap(norm(value))
                ax.add_patch(Rectangle(
                    (col - 0.5, row - 0.5), 1, 1,
                    facecolor=color, edgecolor="white", linewidth=0.5,
                ))
                luminance = sum(c * w for c, w in zip(color[:3], (0.2126, 0.7152, 0.0722)))
                ax.text(col, row, f"{value:.2f}", ha="center", va="center",
                        fontsize=7, color="white" if luminance < 0.5 else INK)
            best_row, best_col = divmod(hindsight[inv], 3)
            best_box = Rectangle(
                (best_col - 0.475, best_row - 0.475), 0.95, 0.95,
                fill=False, edgecolor=INK, linewidth=1.05, zorder=3,
            )
            best_box.set_path_effects([pe.Stroke(linewidth=2, foreground="white"), pe.Normal()])
            ax.add_patch(best_box)
            ref_row, ref_col = divmod(reference[inv], 3)
            inset = 0.12 if reference[inv] == hindsight[inv] else 0.055
            ax.add_patch(Rectangle(
                (ref_col - 0.5 + inset, ref_row - 0.5 + inset),
                1 - 2 * inset, 1 - 2 * inset,
                fill=False, edgecolor=REQUEST, linewidth=1.15,
                linestyle=(0, (2.5, 1.5)), zorder=4,
            ))
            center = (left + width / 2) / SIZE[0]
            fig.text(center, 1.67 / SIZE[1], title,
                     fontsize=7, ha="center", va="center")
            fig.text(center, 0.53 / SIZE[1],
                     f"policy {policy[inv]:.2f} · $r^\\dagger$ {fixed[inv][reference[inv]]:.2f}",
                     fontsize=7, ha="center", va="center")

        handles = [
            Rectangle((0, 0), 1, 1, fill=False, edgecolor=INK, linewidth=1.05),
            Rectangle((0, 0), 1, 1, fill=False, edgecolor=REQUEST,
                      linewidth=1.15, linestyle=(0, (2.5, 1.5))),
        ]
        fig.legend(
            handles,
            ("best fixed request in hindsight",
             "fixed request tuned on the other three sets ($r^\\dagger$)"),
            loc="center", bbox_to_anchor=(0.5, 0.335 / SIZE[1]),
            ncol=2, frameon=False, fontsize=7,
            handlelength=1.4, handleheight=0.8, handletextpad=0.5,
            columnspacing=1.1, borderaxespad=0, borderpad=0,
        )
        cax = fig.add_axes([3.03 / SIZE[0], 0.14 / SIZE[1],
                            2.22 / SIZE[0], 0.055 / SIZE[1]])
        colorbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                               orientation="horizontal", ticks=(0.38, 0.5, 0.66))
        colorbar.ax.set_xticklabels(("0.38", "0.50", "0.66"))
        colorbar.ax.tick_params(labelsize=7, length=2, width=0.5, pad=1)
        colorbar.outline.set_linewidth(0.4)
        colorbar.outline.set_edgecolor(GREY)
        fig.text(2.90 / SIZE[0], 0.1675 / SIZE[1],
                 "ordering AUC (0.5 = no ordering)",
                 fontsize=7, ha="right", va="center")
        fig.savefig(OUT / "fig_results_grid.pdf", metadata={
            "CreationDate": None, "ModDate": None,
            "Creator": "build_results_grid.py",
        })
        fig.savefig(OUT / "fig_results_grid.png", dpi=300,
                    metadata={"Software": "build_results_grid.py"})
        plt.close(fig)

    for inv, title in SETS:
        print(f"{title} ({inv})")
        for row, delta in enumerate((1, 5, 15)):
            print(f"  Δ={delta}: " + " ".join(f"{v:.9f}" for v in fixed[inv][3 * row:3 * row + 3]))
        print(f"  policy={policy[inv]:.9f}; "
              f"F*={REQUESTS[reference[inv]]} AUC={fixed[inv][reference[inv]]:.9f}; "
              f"hindsight={REQUESTS[hindsight[inv]]} AUC={fixed[inv][hindsight[inv]]:.9f}")
    print(f"Size: {SIZE[0]} × {SIZE[1]} in; PNG: 300 dpi")
    print(OUT / "fig_results_grid.pdf")
    print(OUT / "fig_results_grid.png")


if __name__ == "__main__":
    draw()
