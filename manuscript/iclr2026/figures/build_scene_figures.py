"""Build the two real-frame manuscript figures without running the model or engine.

The three packaged PNGs are unmodified gallery frames for state
issue-77-n1-001-a00. All text, circles, labels, and borders are vector artists;
only the original game screenshots are raster images.
"""

import csv
import filecmp
import json
from pathlib import Path
import subprocess
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle
from PIL import Image

OUT = Path(__file__).resolve().parent
ART = Path("/p/Project/NovPhy/.local-artifacts")
GALLERY = Path("/p/Project/NovPhy/data/issue-87-closed-loop-oracle")
STATE = "issue-77-n1-001-a00"
FRAME_CROP = (40, 120, 580, 330)  # identical crop, in source-image pixels
INK = "#18252d"
BLUE = "#075c84"
RUST = "#a33c27"


def _evidence():
    """Check the selected ordinal and the two actual engine-event verdicts."""
    with (ART / "issue-92-selection-validity-v1/slot_join.csv").open(newline="") as f:
        rows = [r for r in csv.DictReader(f)
                if (r["state"], r["system"], r["seed"]) ==
                (STATE, "continuous-fixed-h1", "20260908")]
    assert len(rows) == 1
    row = rows[0]
    assert (row["source_member"], row["n_candidates"], row["successful_ordinals"],
            row["chosen_ordinal"], row["top1_hit"]) == ("issue-77-n1-001", "12", "6", "9", "False")
    summary = json.loads((ART / "issue-92-selection-validity-v1/summary.json").read_text())
    assert summary["accounting"]["oracle_seed_label"]["engine_seeds_per_member"]["issue-77-n1-001"] == [764100001]
    manifest = json.loads((GALLERY / "manifest.json").read_text())
    cells = {c["ordinal"]: c for c in manifest["cells"]
             if c["state"] == STATE and c["ordinal"] in (6, 9)}
    assert set(cells) == {6, 9}
    for ordinal, succeeded in ((6, True), (9, False)):
        cell = cells[ordinal]
        assert cell["status"] == "executed" and cell["first_shot_success"] is succeeded
        assert cell["decision_frame"].endswith(f"--a{ordinal:02d}/decision.png")
        assert cell["final_frame"].endswith(f"--a{ordinal:02d}/final.png")
        assert cell["video"].endswith(f"--a{ordinal:02d}/shot.webm")
        _assert_matching_frame(f"001_a{ordinal:02d}_final.png", cell["final_frame"])
        if ordinal == 6:
            _assert_matching_frame("001_common_decision.png", cell["decision_frame"])


def _assert_matching_frame(packaged, source):
    assert filecmp.cmp(OUT / "source" / packaged, GALLERY / source, shallow=False)


def _frame(fig, rect, filename, border="#697783"):
    ax = fig.add_axes(rect)
    with Image.open(OUT / "source" / filename) as image:
        assert image.size == (640, 480)
        ax.imshow(image.crop(FRAME_CROP), interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(border)
        spine.set_linewidth(0.7)
    return ax


def _circle(ax, xy, radius, color, linestyle="-"):
    ax.add_patch(Circle(xy, radius, facecolor="none", edgecolor=color,
                        linewidth=1.5, linestyle=linestyle, zorder=4))


def _badge(ax, xy, success):
    """Vector verdict symbol, with text labels outside the screenshots."""
    x, y = xy
    color = BLUE if success else RUST
    ax.add_patch(Circle((x, y), 11, facecolor=color, edgecolor="white", lw=1.1, zorder=6))
    if success:
        ax.plot([x - 5, x - 1, x + 6], [y, y + 4, y - 5], color="white",
                lw=2, solid_capstyle="round", zorder=7)
    else:
        ax.plot([x - 4, x + 4], [y - 4, y + 4], color="white", lw=2, zorder=7)
        ax.plot([x - 4, x + 4], [y + 4, y - 4], color="white", lw=2, zorder=7)


def _save(fig, stem):
    with plt.rc_context({"svg.fonttype": "none", "pdf.fonttype": 42}):
        for extension in ("pdf", "svg", "png"):
            fig.savefig(OUT / f"{stem}.{extension}", dpi=240)
    plt.close(fig)


def teaser_scene():
    """One fixed continuous-latent ranking and two independent first-shot outcomes."""
    _evidence()
    fig = plt.figure(figsize=(5.5, 2.72), facecolor="white")

    def text(x, y, label, color=INK, size=7.1, **kwargs):
        fig.text(x, y, label, fontsize=size, color=color, **kwargs)

    def arrow(points, color=INK, width=0.85):
        for start, end in zip(points, points[1:-1]):
            fig.add_artist(FancyArrowPatch(start, end, arrowstyle="-", linewidth=width,
                                           color=color, transform=fig.transFigure))
        fig.add_artist(FancyArrowPatch(points[-2], points[-1], arrowstyle="-|>",
                                       mutation_scale=7, linewidth=width, color=color,
                                       transform=fig.transFigure))

    def card(x, y, w, h, edge, face="white", rounding=0.009):
        fig.add_artist(FancyBboxPatch(
            (x, y), w, h, boxstyle=f"round,pad=0.003,rounding_size={rounding}",
            transform=fig.transFigure, facecolor=face, edgecolor=edge, linewidth=0.85,
        ))

    def screenshot(x, y, filename, border, target_region=False):
        # The same 400 × 190 pixel crop is applied to all three original frames.
        # Keep FRAME_CROP for the separate qualitative figure unchanged.
        w = 0.217
        h = w * 5.5 * (190 / 400) / 2.72
        ax = fig.add_axes((x, y, w, h))
        with Image.open(OUT / "source" / filename) as image:
            assert image.size == (640, 480)
            ax.imshow(image.crop((70, 110, 470, 300)), interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(border)
            spine.set_linewidth(0.8)
        if target_region:
            # Original screenshot pixels: the green target survives in a09;
            # at a06 the grey ball occupies this area, not a target.
            ax.add_patch(Rectangle((236, 122), 43, 45, fill=False,
                                   edgecolor=border, linewidth=0.85))
            zoom = fig.add_axes((x + 0.154, y + 0.137, 0.059, 0.061))
            with Image.open(OUT / "source" / filename) as image:
                zoom.imshow(image.crop((306, 232, 349, 277)), interpolation="nearest")
            zoom.set_xticks([])
            zoom.set_yticks([])
            for spine in zoom.spines.values():
                spine.set_color(border)
                spine.set_linewidth(1.0)

    with plt.rc_context({"font.family": "DejaVu Sans"}):
        text(0.025, 0.954, "(a) Observed state", size=8, weight="bold")
        text(0.298, 0.954, "(b) World-model prediction", size=8, weight="bold")
        text(0.725, 0.954, "(c) Physics-engine outcome", size=7.0, weight="bold")

        screenshot(0.025, 0.671, "001_common_decision.png", "#697783", True)
        text(0.027, 0.616, "Frozen CNN object slots", size=7.2)
        for j, fill in enumerate(("#d5ebfa", "#ddefd0", "#ffdeb8",
                                  "#e7d9f7", "#f8d8db", "#fff0c5")):
            card(0.027 + j * 0.036, 0.538, 0.029, 0.050, "#586672", fill, 0.006)
        arrow(((0.127, 0.668), (0.127, 0.637)), width=0.75)
        text(0.027, 0.479, "12 admissible launches", size=7.2, weight="bold")
        for row in range(3):
            for column in range(4):
                fig.add_artist(Ellipse((0.047 + column * 0.056, 0.419 - row * 0.058),
                                       0.027, 0.052, transform=fig.transFigure,
                                       facecolor="#f0f3f5", edgecolor="#56636c", linewidth=0.85))
        arrow(((0.126, 0.535), (0.126, 0.498)), width=0.75)

        card(0.299, 0.819, 0.408, 0.108, "#a9b9c6", "#f3f8fc")
        text(0.310, 0.887, "Fixed request: Δ=1 model frame per step ·", size=7.0)
        text(0.310, 0.839, "continuous latent", size=7.0)

        # Offset ghost cards signify twelve separate candidate predictions;
        # they are NOT generated model images or alternate request types.
        card(0.347, 0.488, 0.345, 0.324, "#a5b5c4", "#f8fafc")
        card(0.334, 0.476, 0.345, 0.324, "#859db2", "#f5f9fd")
        card(0.320, 0.464, 0.345, 0.324, "#536e83", "#eaf2f9")
        text(0.335, 0.746, "One of 12 latent rollouts", size=7.2, weight="bold")
        text(0.336, 0.674, "launch aᵢ  →  continuous state", size=7.0)
        fig.add_artist(FancyArrowPatch((0.395, 0.649), (0.395, 0.596),
                                       arrowstyle="-|>", mutation_scale=7,
                                       linewidth=0.8, color="#607e96", transform=fig.transFigure))
        text(0.336, 0.542, "predicted task cost  Ĵ(aᵢ)", size=7.1)
        text(0.325, 0.403, "Rank 12 predicted costs", size=7.1)
        card(0.323, 0.294, 0.365, 0.070, RUST, "#fff4ef")
        text(0.333, 0.315, "Lowest predicted cost  →  a09", color=RUST,
             size=7.2, weight="bold")
        arrow(((0.245, 0.361), (0.313, 0.361)), width=1.0)
        text(0.253, 0.393, "aᵢ", size=7.0)

        # The selected candidate and the separate candidate-pool replay both
        # cross into the physics engine. Neither screenshot is a model frame.
        fig.add_artist(FancyArrowPatch((0.719, 0.207), (0.719, 0.910),
                                       arrowstyle="-", linestyle=(0, (3, 3)),
                                       linewidth=0.8, color="#97a4ae", transform=fig.transFigure))
        arrow(((0.690, 0.347), (0.740, 0.347), (0.740, 0.725), (0.752, 0.725)), RUST, 1.15)
        arrow(((0.123, 0.239), (0.123, 0.142), (0.703, 0.142),
               (0.703, 0.317), (0.752, 0.317)), BLUE, 1.0)
        text(0.300, 0.103, "Independent engine replays of all 12 launches", size=7.0,
             color=BLUE)

        text(0.735, 0.852, "a09 · model's pick · miss", RUST, 7.1, weight="bold")
        screenshot(0.755, 0.611, "001_a09_final.png", RUST, True)
        text(0.755, 0.493, "a06 · only success", BLUE, 7.0, weight="bold")
        text(0.755, 0.443, "of 12 · target removed", BLUE, 7.0, weight="bold")
        screenshot(0.755, 0.222, "001_a06_final.png", BLUE, True)

    _save(fig, "fig_teaser_scene")

def teaser_granularity():
    """Real cascade, actual imagined request trace, and compact raw-artifact evidence."""
    manifest = json.loads((GALLERY / "manifest.json").read_text())
    (cell,) = (entry for entry in manifest["cells"]
               if entry["state"] == STATE and entry["ordinal"] == 6)
    assert cell["status"] == "executed" and cell["first_shot_success"] is True
    assert cell["video_frames"] == 555
    _assert_matching_frame("001_common_decision.png", cell["decision_frame"])
    _assert_matching_frame("001_a06_final.png", cell["final_frame"])
    video = GALLERY / cell["video"]

    def video_frame(number):
        # Decode the original video frame; use one identical crop for all four tiles.
        result = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(video), "-vf",
             f"select=eq(n\\,{number})", "-frames:v", "1", "-f", "image2pipe",
             "-vcodec", "png", "-"], capture_output=True, check=True)
        with Image.open(BytesIO(result.stdout)) as image:
            assert image.size == (640, 480)
            return image.convert("RGB")

    # This is the actual imagined trajectory for the pictured launch, NOT a
    # hand-chosen phase schedule or an engine closed-loop controller trace.
    trace_path = (ART / "issue-92-lookahead-control-v1/records/"
                  "decision--hybrid-adaptive-e600--seed20260908--issue-77-n1-001-a00.json")
    trace = json.loads(trace_path.read_text())["decision"]["horizon_trace"]["6"]
    assert trace[0]["start_fixed_step"] == 0
    assert all(a["start_fixed_step"] + a["horizon"] == b["start_fixed_step"]
               for a, b in zip(trace, trace[1:]))
    assert trace[-1]["start_fixed_step"] + trace[-1]["horizon"] == 600

    # Gallery video order is exactly the raw observation-manifest order.
    observation_root = (ART / "issue-87-closed-loop-oracle-v1/attempts" /
                        cell["identity"] / "shot-1/observation-trace")
    frames = json.loads((observation_root / "observation_trace_manifest.json").read_text())["frame_records"]
    assert len(frames) == cell["video_frames"]
    origin = frames[0]["fixed_step"]
    photo_steps = [(frames[index]["fixed_step"] - origin) / 50
                   for index in (0, 80, 120, len(frames) - 1)]
    assert photo_steps == [0, 80, 120, 553.38]
    # Frame 211 (the imagined trace's first long request) is not yet settled
    # in the engine. Use the actual end-of-window image, not a fictitious 600.

    # Continuous-only recursive prediction, three seeds, no truth resets.
    benchmarks = (("NovPhy", "issue-77-n1-diagnostic-v1", (15, 30, 60, 150, 225)),
                  ("CLEVRER", "issue-79-clevrer-boundary-v1", (15, 30, 60, 120)),
                  ("Physion-Dominoes", "issue-84-third-family-v1", (15, 30, 60, 120)))
    errors = []
    for name, folder, endpoints in benchmarks:
        raw = json.loads((ART / folder / "summary.json").read_text())["per_seed"]
        curves = {}
        for horizon in (1, 15):
            values = np.array([[raw[seed][f"continuous_h{horizon}"]["curves"]
                                [str(endpoint)]["position_mse"] for endpoint in endpoints]
                               for seed in sorted(raw)])
            assert values.shape == (3, len(endpoints)) and np.all(values > 0)
            curves[horizon] = values
        errors.append((name, endpoints, curves))

    with (ART / "issue-96-tau-ad-within-checkpoint-v1/comparisons.csv").open(newline="") as f:
        evidence = {row["id"]: row for row in csv.DictReader(f)}
    inventories = (("angle", "Launch-angle sweep"), ("offset", "Offset sweep"),
                   ("grid", "Drag grid"), ("power", "Launch-power sweep"))
    ordering = []
    for inventory, name in inventories:
        auc = {f"{delta}-{level}": float(evidence[
            f"{inventory}_auc_F-{delta}-{level}"]["value"])
               for delta in (1, 5, 15) for level in ("continuous", "micro", "macro")}
        rank = 1 + sum(value > auc["5-macro"] for value in auc.values())
        contrast = evidence[f"{inventory}_J-F*"]
        ordering.append((name, rank, *(float(contrast[key]) for key in
                                      ("value", "interval_low", "interval_high"))))

    # 9 pt at 7 in becomes 7.07 pt at the manuscript's 5.5 in width.
    width, height = 7.0, 2.92
    fig = plt.figure(figsize=(width, height), facecolor="white")
    blue, orange, violet = "#0072B2", "#D55E00", "#CC79A7"
    grey = "#637382"
    xy = lambda x, y: (x / width, y / height)

    def text(x, y, message, **kwargs):
        fig.text(*xy(x, y), message, color=INK,
                 **{"fontsize": 9, **kwargs})

    def stroke(x0, y0, x1, y1, color=grey, width=.9, **kwargs):
        fig.add_artist(plt.Line2D((x0 / 7, x1 / 7), (y0 / height, y1 / height),
                                 transform=fig.transFigure, color=color, lw=width,
                                 **kwargs))

    with plt.rc_context({"font.family": "DejaVu Sans"}):
        text(.08, 2.80, "(a) One action, a cascade of events", weight="bold", fontsize=10)
        crop = (45, 105, 565, 307)
        sources = ((GALLERY / cell["decision_frame"], "Start"),
                   (video_frame(80), "Flight"),
                   (video_frame(120), "First contact"),
                   (GALLERY / cell["final_frame"], "Settled"))
        left, right = .14, 4.50
        timeline = lambda step: left + (right - left) * step / 600
        for i, ((source, label), step) in enumerate(zip(sources, photo_steps)):
            x = .08 + i * 1.14
            ax = fig.add_axes((*xy(x, 2.23), 1.03 / width, .40 / height))
            if isinstance(source, Image.Image):
                image = source
            else:
                with Image.open(source) as original:
                    assert original.size == (640, 480)
                    image = original.convert("RGB")
            ax.imshow(image.crop(crop), interpolation="nearest", aspect="equal")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color(grey)
                spine.set_linewidth(.7)
            text(x + .515, 2.65, label, ha="center")
            stroke(x + .515, 2.23, timeline(step), 2.08, grey, .6)
        stroke(left, 2.08, right, 2.08, grey, .6)
        # The settled photo's leader marks its true frame (553.38); no tick label,
        # so the axis reads 0/80/120/600 without an odd decimal.
        for step, label in ((0, "0"), (80, "80"), (120, "120"), (600, "600")):
            x = timeline(step)
            stroke(x, 2.06, x, 2.10, grey, .6)
            text(x, 1.93, label, ha="center")
        text(2.45, 1.93, "carrier frames", ha="center")

        text(.08, 1.76, "MDP world model: fixed r=(1, continuous) (schematic)")
        # Every carrier frame gets a uniform fine tick; the wedge is NOT data.
        fig.add_artist(Polygon([xy(left, 1.66), xy(right, 1.75),
                                xy(right, 1.57)], closed=True,
                               transform=fig.transFigure, facecolor=grey,
                               alpha=.32, edgecolor="none"))
        for step in range(601):
            x = timeline(step)
            stroke(x, 1.645, x, 1.675, grey, .15)
        text(.08, 1.43, "learned requests r=(Δ,α) (imagined rollout)")
        mode_colors = {"continuous": blue, "micro": violet, "macro": orange}
        for item in trace:
            x0 = timeline(item["start_fixed_step"])
            x1 = timeline(item["start_fixed_step"] + item["horizon"])
            stroke(x0, 1.33, x1, 1.33, mode_colors[item["mode"]], 4,
                   solid_capstyle="butt")
            stroke(x0, 1.30, x0, 1.36, "white", .25)
        text(.08, 1.15, "colour = level:")
        for x, mode in ((1.02, "continuous"), (2.07, "micro"), (2.81, "macro")):
            stroke(x, 1.19, x + .15, 1.19, mode_colors[mode], 3)
            text(x + .19, 1.15, mode)
        text(3.51, 1.15, "length = Δ")

        text(4.83, 2.77, "(b1) Longer jumps, less error", weight="bold", fontsize=9)
        text(4.83, 2.59, "Δ=1 dashed; Δ=15 solid")
        for i, (name, endpoints, curves) in enumerate(errors):
            y0 = 2.18 - .50 * i
            ax = fig.add_axes((*xy(5.37, y0), 1.50 / width, .19 / height))
            for horizon, color, style in ((1, orange, "--"), (15, blue, "-")):
                values = curves[horizon]
                ax.fill_between(endpoints, values.min(axis=0), values.max(axis=0),
                                color=color, alpha=.14, linewidth=0)
                ax.plot(endpoints, values.mean(axis=0), color=color,
                        linestyle=style, lw=1.25, marker="o" if horizon == 15 else "s",
                        markersize=2)
            ax.set_yscale("log")
            ax.set_xlim(0, max(endpoints) + 4)
            ax.set_xticks([15, max(endpoints)])
            ax.set_title(name, fontsize=9, pad=1, color=INK)
            ax.tick_params(axis="both", labelsize=9, length=2, pad=1)
            ax.tick_params(which="minor", left=False, bottom=False)
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_yticks(([.01, 1000], [.001, 10], [.0001, 100])[i])
            ax.set_yticklabels((("0.01", "1000"), ("0.001", "10"),
                                ("0.0001", "100"))[i])
        text(4.72, 1.67, "rollout MSE (log)", rotation=90, va="center")

        text(.08, .91, "(b2)", weight="bold", fontsize=10)
        text(2.74, .91, "Rank of fixed (5, macro)", ha="center")
        text(2.74, .76, "among 9 (3 Δ × 3 α)", ha="center")
        text(5.25, .90, "Per-prediction vs. fixed request", ha="center")
        text(5.25, .75, "tuned on other sets (difference in ordering AUC)", ha="center")
        rank_x = lambda rank: 2.00 + (rank - 1) * .185
        difference_x = lambda diff: 5.40 + diff * 8.5
        text(rank_x(1), .61, "1 best", ha="center")
        text(rank_x(9), .61, "9 worst", ha="center")
        for value, label in ((-.1, "−0.1"), (0, "0 (no difference)"), (.1, "+0.1")):
            text(difference_x(value), .61, label, ha="center")
        stroke(difference_x(0), .06, difference_x(0), .57, "#9EAAB0", .75)
        for i, (name, rank, value, low, high) in enumerate(ordering):
            y = .50 - .145 * i
            text(.08, y - .03, name)
            stroke(rank_x(1), y, rank_x(9), y, "#DFE4E7", .75)
            stroke(rank_x(rank), y - .027, rank_x(rank), y + .027, orange, 3)
            text(rank_x(rank) + .07, y - .03, str(rank))
            stroke(difference_x(low), y, difference_x(high), y, blue, 1.25)
            for bound in (low, high):
                stroke(difference_x(bound), y - .022, difference_x(bound), y + .022,
                       blue, .9)
            stroke(difference_x(value), y - .028, difference_x(value), y + .028,
                   blue, 3)
    _save(fig, "fig_teaser_granularity")


def _label(ax, xy, xytext, text, color):
    ax.annotate(text, xy=xy, xytext=xytext, fontsize=7.3, color=color,
                arrowprops={"arrowstyle": "-", "color": color, "lw": 0.9,
                            "shrinkA": 2, "shrinkB": 3}, zorder=5)


def qualitative_case():
    """Auditable matched three-frame case with object annotations and verdict badges."""
    _evidence()
    fig = plt.figure(figsize=(7.0, 2.75), facecolor="white")
    with plt.rc_context({"font.family": "DejaVu Sans"}):
        dec = _frame(fig, (0.020, 0.300, 0.363, 0.360), "001_common_decision.png")
        a06 = _frame(fig, (0.615, 0.555, 0.363, 0.360), "001_a06_final.png", BLUE)
        a09 = _frame(fig, (0.615, 0.045, 0.363, 0.360), "001_a09_final.png", RUST)
        fig.text(0.020, 0.685, "(a) Same starting frame", fontsize=8.2, weight="bold", color=INK)
        fig.text(0.615, 0.940, "(b) a06: target removed", fontsize=8.2, weight="bold", color=BLUE)
        fig.text(0.615, 0.430, "(c) model choice a09: miss", fontsize=8.2, weight="bold", color=RUST)
        # Coordinates are relative to the same 540 x 210 crop in all frames.
        _circle(dec, (59, 140), 15, BLUE)
        _label(dec, (59, 140), (76, 91), "slingshot", BLUE)
        _circle(dec, (178, 81), 13, INK)
        _label(dec, (178, 81), (143, 19), "ball", INK)
        _circle(dec, (284, 133), 15, BLUE)
        _label(dec, (284, 133), (327, 94), "target", BLUE)
        # The a06 target is absent; the round object at the ramp is the ball,
        # so it must not be highlighted or mislabeled as a knocked-out target.
        _circle(a06, (284, 129), 15, INK)
        _label(a06, (284, 129), (330, 67), "ball on ramp", INK)
        _badge(a06, (518, 20), True)
        _circle(a09, (284, 133), 15, RUST)
        _label(a09, (284, 133), (332, 101), "target intact", RUST)
        _circle(a09, (178, 81), 13, INK)
        _label(a09, (178, 81), (218, 40), "ball short", INK)
        _badge(a09, (518, 20), False)
        for endpoint, label, y in (((0.602, 0.705), "a06", 0.680),
                                   ((0.602, 0.205), "a09", 0.258)):
            fig.add_artist(FancyArrowPatch((0.397, 0.480), endpoint, arrowstyle="-|>",
                                           mutation_scale=11, color=INK, lw=1.0,
                                           transform=fig.transFigure))
            fig.text(0.455, y, label, fontsize=8, weight="bold", color=INK)
    _save(fig, "fig_qualitative_case")
