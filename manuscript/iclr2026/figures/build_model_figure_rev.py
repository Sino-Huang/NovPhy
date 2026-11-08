"""Draw Figure 3 (fig:mth:arch): the primary post-submission model, at print size.

Run: python figures/build_model_figure_rev.py
Writes fig_model_overview_rev.{svg,pdf,png} next to this script (5.5 x 2.6 in).

Top row: runtime, one imagined rollout per candidate launch.  Bottom row: training
and labelling.  Every node and edge is traced to code/config in the comments below
(paths relative to the NovPhy repository).  The only raster element is the agent
frame crop reused from the submitted figure (source/001_common_decision.png); all
labels are vector text.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, FancyArrowPatch, FancyBboxPatch, Rectangle
from PIL import Image

OUT = Path(__file__).resolve().parent
# Description-level colours shared with build_description_levels.py / contract §9.
BLUE, VIOLET, ORANGE = "#0072B2", "#CC79A7", "#D55E00"
INK, GREY, TEAL = "#18252d", "#5b6b78", "#007C78"
PANEL = "#f4f6f7"


def draw():
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 7,
                         "mathtext.fontset": "dejavusans", "svg.fonttype": "none",
                         "pdf.fonttype": 42}):
        fig, ax = plt.subplots(figsize=(5.5, 2.6), facecolor="white")
        fig.subplots_adjust(0, 0, 1, 1)
        ax.set(xlim=(0, 5.5), ylim=(0, 2.6), aspect="equal")
        ax.axis("off")

        def text(x, y, value, *, size=7, color=INK, **kwargs):
            return ax.text(x, y, value, fontsize=size, color=color, va="center", **kwargs)

        def box(x, y, w, h, *, fill="white", edge=GREY, lw=.7, dashed=False):
            ax.add_patch(FancyBboxPatch(
                (x, y), w, h, boxstyle="round,pad=0,rounding_size=0.04",
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

        def lock(x, y):
            # Frozen-parameter cue, not a component.
            ax.add_patch(Arc((x + .038, y + .062), .07, .085, theta1=0, theta2=180,
                             color=GREY, lw=.7))
            ax.add_patch(Rectangle((x, y), .076, .062, facecolor=GREY, edgecolor="none"))

        # ------------------------------------------------------------------ runtime row
        text(.05, 2.50, "RUNTIME", weight="bold", color=GREY)
        text(.05, 2.36, r"one imagined rollout per launch $a$", color=GREY)

        # Decision frame o_0.  Crop keeps the source aspect (520:202); the source is a
        # 2x render of a 320x240 agent frame (encoder input: issue-99 plan.json
        # encoder.image_width/height = 320/240).
        with Image.open(OUT / "source/001_common_decision.png") as frame:
            crop = frame.crop((45, 105, 565, 307))
            ax.imshow(crop, extent=(.05, .70, 1.70, 1.9525), interpolation="nearest")
        ax.add_patch(Rectangle((.05, 1.70), .65, .2525, fill=False, edgecolor=GREY, lw=.6))
        text(.375, 1.60, "decision frame", ha="center")
        text(.375, 1.475, r"$o_0$, 320$\times$240", ha="center")
        arrow(((.72, 1.83), (.84, 1.83)))

        # Frozen slot encoder E (spatial_slot_parser.py:30-62: stride-8 CNN backbone,
        # one query per vocabulary slot, 2 cross-attention layers).  Runtime calls it
        # once, on the decision frame, with no prior frame
        # (scripts/run_lookahead_control.py:575-585 encode_carrier;
        # issue-108 plan.json evaluation.anchors).
        box(.82, 1.62, .54, .42, fill="#eef1f3")
        text(1.09, 1.94, r"$E$", size=9, ha="center", weight="bold")
        text(1.09, 1.80, "18 slots", ha="center")
        lock(.875, 1.655)
        text(.975, 1.69, "frozen", ha="left")
        text(1.09, 2.11, "once", ha="center", color=BLUE, weight="bold")
        arrow(((1.36, 1.83), (1.45, 1.83)))

        # Carrier z_0 in R^236 = 2 header + 18 x 13 (cnn_hybrid.py:20-21;
        # cohort_v2_feature_parser.py:44 ENTITY_KINEMATIC_WIDTH = 13).
        for offset in (.05, .025, 0):
            box(1.45 + offset, 1.70 + offset, .21, .23, fill="#eaf4fa", edge=BLUE, lw=.6)
        text(1.555, 1.815, r"$z_0$", size=8, ha="center")
        text(1.58, 1.60, r"$\mathbb{R}^{236}$", ha="center")
        text(1.58, 1.475, r"$2+18\!\times\!13$", ha="center")
        arrow(((1.72, 1.83), (1.88, 1.83)))

        # Current carrier: z_0 at k = 0, then the predicted carrier.
        box(1.88, 1.70, .32, .26, fill="#eaf4fa", edge=BLUE, lw=.9)
        text(2.04, 1.83, r"$\hat z_k$", size=9, ha="center")

        # Request policy pi_psi on [z_k; a; min(kappa,60)/60], Delta > kappa masked
        # (request_policy.py:65-84; kappa = 225 - k, :41 ENDPOINT).
        arrow(((2.04, 1.96), (2.04, 2.36), (2.62, 2.36)), lw=.75)
        box(2.62, 2.18, .98, .37, fill="#eef7f6", edge=TEAL)
        text(3.11, 2.43, r"$\pi_\psi(r\,|\,\hat z_k, a, \kappa_k)$", size=7.5, ha="center")
        text(3.11, 2.28, r"mask $\Delta>\kappa_k$", ha="center")
        # Request code c(r) = MLP[sin/cos(Delta); e_alpha] (predictor.py:218-256).
        arrow(((3.60, 2.36), (3.78, 2.36)), color=TEAL)
        text(3.69, 2.45, r"$r$", ha="center", color=TEAL)
        box(3.78, 2.18, .80, .37, fill="#eef7f6", edge=TEAL)
        text(4.18, 2.43, r"$c(r)$", size=7.5, ha="center")
        text(4.18, 2.28, r"MLP$[\phi(\Delta); e_\alpha]$", ha="center")

        # Candidate launch a enters the trunk input with z_k (cnn_hybrid.py:78).
        text(2.36, 2.07, r"$a$", size=8, ha="center")
        arrow(((2.36, 2.00), (2.36, 1.845)), color=INK, lw=.7)
        arrow(((2.20, 1.83), (2.50, 1.83)))

        # Hybrid predictor F_theta: Linear([z;a]) -> 3 FiLM blocks (width 384) ->
        # LayerNorm -> Linear, residual on z (cnn_hybrid.py:22,39-43,75-86;
        # predictor.py:259-274).
        box(2.50, 1.57, 1.50, .48, fill="#edf6fb", edge=BLUE, lw=1)
        for j in range(3):
            x = 2.60 + .47 * j
            box(x, 1.76, .36, .19, fill="white", edge=BLUE, lw=.6)
            text(x + .18, 1.855, "FiLM", ha="center")
            if j < 2:
                arrow(((x + .36, 1.855), (x + .47, 1.855)), lw=.7)
        text(3.25, 1.65, r"$F_\theta$: residual MLP, width 384", ha="center")
        # c(r) modulates every block.
        line(((4.18, 2.18), (4.18, 2.11), (2.78, 2.11)), color=TEAL, lw=.75)
        for j in range(3):
            arrow(((2.78 + .47 * j, 2.11), (2.78 + .47 * j, 1.95)), color=TEAL, lw=.75)

        # Predicate heads read z_k; their soft outputs enter the trunk through a
        # bias-free adapter only for alpha != cont (cnn_hybrid.py:44-48,62-82).
        arrow(((2.25, 1.83), (2.25, 1.13), (3.55, 1.13), (3.55, 1.19)), color=VIOLET, lw=.75)
        box(2.91, 1.19, 1.31, .29, fill="#fcf5f9", edge=VIOLET)
        line(((2.95, 1.355), (2.95, 1.455)), color=VIOLET, lw=2)
        line(((2.95, 1.215), (2.95, 1.315)), color=ORANGE, lw=2)
        text(2.985, 1.405, "micro: contact, support")
        text(2.985, 1.265, "macro: steady, unstable")
        arrow(((2.91, 1.335), (2.85, 1.335)), color=VIOLET, lw=.75)
        box(2.30, 1.19, .55, .29, fill="#fcf5f9", edge=VIOLET)
        text(2.575, 1.405, "adapter", ha="center")
        text(2.575, 1.265, r"if $\alpha\neq$cont", ha="center")
        arrow(((2.62, 1.48), (2.62, 1.57)), color=VIOLET, lw=.75)

        # Output carrier; recursion feeds the prediction back, never re-encodes
        # (request_policy.py:197-239 continue_rollout; cnn_hybrid.py:216-238 adaptive_rollout).
        arrow(((4.00, 1.83), (4.12, 1.83)))
        text(4.26, 1.83, r"$\hat z_{k+\Delta}$", size=9, ha="center")
        arrow(((4.28, 1.71), (4.28, 1.03), (2.04, 1.03), (2.04, 1.70)), dashed=True, lw=.85)
        text(3.16, .965, r"fed back: $k\leftarrow k+\Delta$, no re-encoding", ha="center",
             color=BLUE)

        # Endpoint and task cost: tie-free cost = pig presence - 0.1 x pig-centre
        # displacement from z_0 (scoring_harness.py:118-145), lower ranks first.
        arrow(((4.43, 1.83), (4.60, 1.83)))
        text(4.505, 1.935, r"$\kappa\!=\!0$", ha="center")
        box(4.60, 1.56, .86, .49, fill="white", edge=INK, lw=.7)
        text(5.03, 1.93, r"cost($\hat z_{225}$)", ha="center")
        text(5.03, 1.79, "pig presence", ha="center")
        text(5.03, 1.655, r"$-\,0.1\cdot$pig shift", ha="center")
        arrow(((5.03, 1.56), (5.03, 1.42)), color=INK, lw=.7)
        text(5.03, 1.33, r"rank launches $a$", ha="center")

        # ----------------------------------------------------------------- training row
        top, bottom = .86, .03

        def panel(x, w, title, lines, *, title_color=INK):
            box(x, bottom, w, top - bottom, fill=PANEL, edge="#b4bec6", lw=.6)
            text(x + .07, top - .1, title, weight="bold", color=title_color)
            for i, value in enumerate(lines):
                text(x + .07, top - .235 - .12 * i, value)

        # (1) Encoder: trained once on engine-projected targets, then frozen
        # (issue-99-slot-encoder-fix-v1/plan.json encoder.targets, epochs, selection).
        panel(.05, 1.05, r"1  Encoder $E$",
              ["engine-projected", "slot labels;", "trained once,", "then frozen"])
        lock(.93, .72)
        arrow(((1.10, .45), (1.20, .45)), color=GREY, lw=.7)

        # (2) Predictor on the 480-level fit split (issue-112-v3-retrain-v1/plan.json
        # data.levels, recipe.training.*; decision_dynamics.py:201-299;
        # run_decision_relevant_dynamics.py:480-496).
        panel(1.20, 2.34, r"2  Predictor $F_\theta$ · 480 fit levels",
              [r"local: teacher-forced + recursive, 4 $\Delta$-steps",
               r"long: recursive, horizon 60$\rightarrow$225 frames",
               r"rank: success $<$ failure cost (fit verdicts)",
               r"targets $E(o_t)$; predicates: engine labels",
               "9 requests round-robin"])
        arrow(((3.54, .45), (3.64, .45)), color=GREY, lw=.7)

        # (3) Request policy J on the 120-level policy split: prediction-error DP on
        # the fixed-context surrogate, cross-entropy distillation, 2 aggregation rounds
        # (issue-114 plan.json pool/policies; request_policy.py:41-45,111-171,333-357).
        # J_dec uses decision-valued labels instead (request_policy.py:263-320).
        panel(3.64, 1.82, r"3  Policy $\pi_\psi$ · 120 policy levels",
              ["DP labels, fixed context:",
               r"$\Delta\cdot$MSE $+\ \lambda\cdot$compute ratio",
               r"$\rightarrow$ cross-entropy distillation,",
               "2 aggregation rounds",
               r"($J_{\mathrm{dec}}$: decision-valued labels)"])

        for extension in ("pdf", "svg", "png"):
            fig.savefig(OUT / f"fig_model_overview_rev.{extension}", dpi=300)
        plt.close(fig)


if __name__ == "__main__":
    draw()
