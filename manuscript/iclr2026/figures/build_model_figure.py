"""Draw the evaluated request-conditioned model, at manuscript print size.

Run: python figures/build_model_figure.py
Source evidence is recorded in iclr2026_conference_fig_model.tex's BRIEF.
Only the genuine agent-frame crop is raster; all labels remain vector text.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, FancyArrowPatch, FancyBboxPatch, Rectangle
from PIL import Image

OUT = Path(__file__).resolve().parent
# Description-level colours match build_scene_figures.py:286.
BLUE, ORANGE, VIOLET = "#0072B2", "#D55E00", "#CC79A7"
INK, GREY, REQUEST = "#18252d", "#637382", "#007C78"


def draw():
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 7,
                         "mathtext.fontset": "dejavusans", "svg.fonttype": "none",
                         "pdf.fonttype": 42}):
        fig, ax = plt.subplots(figsize=(5.5, 2.6), facecolor="white")
        fig.subplots_adjust(0, 0, 1, 1)
        ax.set(xlim=(0, 5.5), ylim=(0, 2.6), aspect="equal")
        ax.axis("off")

        def text(x, y, value, *, size=7, color=INK, **kwargs):
            return ax.text(x, y, value, fontsize=size, color=color,
                           va="center", **kwargs)

        def box(x, y, w, h, *, fill="white", edge=GREY, lw=.7, dashed=False):
            ax.add_patch(FancyBboxPatch(
                (x, y), w, h, boxstyle="round,pad=0,rounding_size=0.045",
                facecolor=fill, edgecolor=edge, lw=lw,
                linestyle=(0, (3, 2)) if dashed else "-"))

        def line(points, *, color=GREY, lw=.8, dashed=False):
            ax.plot(*zip(*points), color=color, lw=lw,
                    linestyle=(0, (3, 2)) if dashed else "-", solid_capstyle="round")

        def arrow(points, *, color=BLUE, lw=.9, dashed=False):
            if len(points) > 2:
                line(points[:-1], color=color, lw=lw, dashed=dashed)
            ax.add_patch(FancyArrowPatch(
                points[-2], points[-1], arrowstyle="-|>", mutation_scale=6,
                lw=lw, color=color, shrinkA=0, shrinkB=0,
                linestyle=(0, (3, 2)) if dashed else "-"))

        # One compact control lane. Named inputs avoid crossing the runtime lane.
        box(.08, 2.05, 1.57, .45, fill="#eff8f6", edge=REQUEST)
        text(.19, 2.38, r"Request policy $\pi_\psi$", size=8, weight="bold")
        text(.19, 2.23, r"$242\rightarrow128\rightarrow9$ on $[z_k;\,a;\,\kappa_k]$")
        text(.19, 2.09, r"mask $\Delta>\kappa_k$ (remaining)")
        # Drawn policy inputs: carrier from z_k, launch from its label.
        arrow(((1.82, 1.635), (1.82, 1.90), (1.50, 1.90), (1.50, 2.05)), color=BLUE, lw=.7)
        arrow(((1.00, 1.83), (1.00, 2.05)), color=INK, lw=.7)
        arrow(((1.65, 2.31), (2.12, 2.31)), color=REQUEST)

        # All nine requests; the filled cell is explicitly schematic.
        columns = (("cont", BLUE), ("micro", VIOLET), ("macro", ORANGE))
        text(2.70, 2.53, "schematic choice", color=GREY, ha="center")
        text(1.98, 2.39, r"$\Delta$", ha="center")
        for j, (name, color) in enumerate(columns):
            x = 2.16 + j * .39
            text(x + .16, 2.39, name, ha="center")
            for i, delta in enumerate((1, 5, 15)):
                y = 2.23 - .15 * i
                selected = i == 1 and j == 2
                ax.add_patch(Rectangle((x, y - .055), .32, .11,
                                      facecolor=color if selected else "white",
                                      edgecolor=color, lw=1 if selected else .6))
                if selected:
                    ax.plot(x + .16, y, marker="o", markersize=2, color="white")
                if j == 0:
                    text(1.98, y, str(delta), ha="center")
        arrow(((3.28, 2.08), (3.56, 2.08)), color=REQUEST)
        text(3.42, 2.23, r"$r$", ha="center", color=REQUEST)
        box(3.56, 2.02, 1.84, .48, fill="#eff8f6", edge=REQUEST)
        text(4.48, 2.36, r"Joint request code $c(r)$", size=8, ha="center", weight="bold")
        text(4.48, 2.17, r"sin/cos $\Delta$ fused with $\alpha$ embedding", ha="center")

        # Preserve the genuine observation crop's aspect ratio (520:202).
        with Image.open(OUT / "source/001_common_decision.png") as frame:
            crop = frame.crop((45, 105, 565, 307))
            ax.imshow(crop, extent=(.08, .76, 1.285, 1.549), interpolation="nearest")
        ax.add_patch(Rectangle((.08, 1.285), .68, .264, fill=False, edgecolor=GREY, lw=.6))
        text(.42, 1.15, r"Agent RGB $o_k$", ha="center")
        text(.08, 1.00, "(+ previous frame)", color=GREY)
        arrow(((.78, 1.42), (.92, 1.42)))
        box(.92, 1.20, .58, .44, fill="#f3f5f6")
        text(1.21, 1.47, r"CNN $E$", size=8, ha="center", weight="bold")
        text(1.25, 1.29, "frozen", ha="center")
        # Lock is a frozen-parameter cue, not a learned component.
        ax.add_patch(Arc((1.01, 1.31), .07, .09, theta1=0, theta2=180, color=GREY, lw=.7))
        ax.add_patch(Rectangle((.972, 1.245), .076, .065, facecolor=GREY, edgecolor="none"))
        arrow(((1.50, 1.42), (1.68, 1.42)))

        # The stack is a glyph for the fixed object-slot carrier, not extra layers.
        for offset in (.065, .032, 0):
            box(1.72 + offset, 1.26 + offset, .25, .31,
                fill="#eaf4fa", edge=BLUE, lw=.65)
        text(1.86, 1.43, r"$z_k$", size=8.5, ha="center")
        text(1.86, 1.11, r"$\mathbb{R}^{236}$", ha="center")
        text(1.86, .97, r"$18\!\times\!13+2$", ha="center")
        arrow(((2.05, 1.42), (2.46, 1.42)))
        text(1.16, 1.76, r"candidate launch $a$", ha="center")
        arrow(((1.90, 1.76), (2.25, 1.76), (2.25, 1.42)), color=INK)

        # The hero: one shared continuous transition, with three modulated blocks.
        box(2.46, 1.12, 2.15, .52, fill="#edf6fb", edge=BLUE, lw=1)
        for j in range(3):
            x = 2.70 + .58 * j
            box(x, 1.32, .42, .25, fill="white", edge=BLUE)
            text(x + .21, 1.445, f"FiLM {j + 1}", ha="center")
            if j < 2:
                arrow(((x + .42, 1.445), (x + .58, 1.445)))
        arrow(((4.48, 2.02), (4.48, 1.81)), color=REQUEST)
        line(((2.91, 1.81), (4.48, 1.81)), color=REQUEST)
        for x in (2.91, 3.49, 4.07):
            arrow(((x, 1.81), (x, 1.57)), color=REQUEST, lw=.8)
        text(4.91, 1.76, "AdaLN-Zero", ha="center", color=REQUEST)
        arrow(((4.61, 1.42), (4.94, 1.42)))
        text(5.18, 1.47, r"$\hat z_{k+\Delta}$", size=10, ha="center")
        text(5.18, 1.28, "same carrier", ha="center")

        text(3.535, 1.20, r"Shared residual $F_\theta$ · width 384", ha="center", size=7.5, weight="bold")
        box(2.30, .78, 1.42, .27, fill="#fcf6fa", edge=VIOLET)
        text(2.39, .975, "micro: contact / support")
        text(2.39, .845, "macro: steady / unstable")
        line(((2.34, .925), (2.34, 1.025)), color=VIOLET, lw=2)
        line(((2.34, .795), (2.34, .895)), color=ORANGE, lw=2)
        arrow(((2.20, 1.42), (2.20, .915), (2.30, .915)), color=VIOLET)
        box(3.90, .78, .68, .27, fill="#fcf6fa", edge=VIOLET)
        text(4.24, .975, "bias-free", ha="center")
        text(4.24, .845, "adapter", ha="center")
        arrow(((3.72, .915), (3.90, .915)), color=VIOLET)
        arrow(((4.58, .915), (4.69, .915), (4.69, 1.08),
               (2.39, 1.08), (2.39, 1.32), (2.46, 1.32)), color=VIOLET, lw=.8)
        text(3.49, .665, r"soft feedback only if $\alpha\ne\mathrm{cont}$", ha="center", color=GREY)

        # Rollout returns to the carrier, never to RGB or the frozen encoder.
        arrow(((4.80, 1.42), (4.80, .60), (1.55, .60), (1.55, 1.26), (1.72, 1.26)),
              color=BLUE, dashed=True, lw=.8)
        text(.08, .78, "Imagined rollout", color=BLUE, weight="bold")
        text(.08, .63, r"$k\leftarrow k+\Delta$; repeat to endpoint", color=BLUE)
        arrow(((5.18, 1.18), (5.18, 1.06)), color=BLUE)
        text(5.12, .98, "Endpoint cost", ha="center")
        text(5.12, .83, "min over", ha="center")
        text(5.12, .68, "launches", ha="center")

        # Training objectives are isolated from the runtime input graph.
        box(.06, .02, 5.38, .50, fill="#f5f6f7", edge="#a5afb7", lw=.65, dashed=True)
        for x in (2.76, 4.63):
            line(((x, .08), (x, .47)), color="#c4cbd0", lw=.55)
        text(.14, .435, "PREDICTOR TRAINING", weight="bold", color=GREY)
        text(.14, .315, "Local (teacher-forced) + recursive MSE")
        text(.14, .195, r"target $E(o_{k+\Delta})$ · 9-request round-robin")
        text(.14, .075, "0.1 BCE (engine labels) + 0.01 carrier bound")
        text(2.86, .435, "POLICY TRAINING", weight="bold", color=GREY)
        text(2.86, .315, "Cross-entropy on DP labels")
        text(2.86, .195, r"$\Delta\!\cdot\!\mathrm{MSE}+\lambda\!\cdot\!\mathrm{MAC\ ratio}$")
        text(2.86, .075, "DAgger-style second round")
        text(4.73, .435, "ENCODER", weight="bold", color=GREY)
        text(4.73, .265, "Frozen")
        text(4.73, .105, "No gradient")
        for extension in ("pdf", "svg", "png"):
            fig.savefig(OUT / f"fig_model_overview.{extension}", dpi=300)
        plt.close(fig)


if __name__ == "__main__":
    draw()
