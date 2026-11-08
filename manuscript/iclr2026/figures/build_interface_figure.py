"""Build Figure 1 (`fig:interface`): the query interface only, no results.

Panels
  (a) One real NovPhy launch and its cascade on a carrier-frame timeline, with the
      scored rollout endpoint (carrier frame 225) marked.
  (b) What a request r = (Delta, alpha) is: horizon jumps and description levels.
  (c) Schematic, 45-frame segment: fixed fine r = (1, continuous), fixed coarse
      r = (15, continuous) and per-query requests, all served by one shared
      predictor F_theta(z_k, a, r). Panel (c) is NOT data or a policy trace.

Frame provenance (panel a)
  The four engine frames are the embedded PNG rasters of the submitted teaser
  `figures/fig_teaser_granularity.svg` (sha256 below), extracted verbatim (base64
  <image> elements, document order) and only flipped vertically to undo
  Matplotlib's SVG image transform `scale(1 -1)`. No other processing.
  They show NovPhy level state `issue-77-n1-001-a00`, candidate launch 6
  (gallery `data/issue-87-closed-loop-oracle/manifest.json`, first-shot success),
  cropped in the teaser build with one identical crop (45, 105, 565, 307) of the
  640x480 frames. Carrier frames 0 / 80 / 120 / 553.38 are the recorded
  `frame_records[].fixed_step` values 30000 / 34000 / 36000 / 57669 of
  `.local-artifacts/issue-87-closed-loop-oracle-v1/attempts/
  oracle--issue-77-n1-001-a00--a06/shot-1/observation-trace/
  observation_trace_manifest.json`, origin 30000, 50 engine steps per carrier
  frame (asserted in `build_scene_figures.py::teaser_granularity`; submitted
  appendix audit row for `fig:teaser`). The settled frame is the final engine
  frame (553.38), not the imagined rollout endpoint 600.
    start   -> source/interface_frame_start.png    (video frame 0)
    flight  -> source/interface_frame_flight.png   (video frame 80)
    contact -> source/interface_frame_contact.png  (video frame 120)
    settled -> source/interface_frame_settled.png  (final video frame 554)

Time base: one carrier frame = 50 engine steps = 0.02 s (0.0004 s per engine
step; contract H12, `scripts/derive_capture_window_spec.py:43`).
"""

import base64
import hashlib
import io
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Arc, FancyArrowPatch, FancyBboxPatch  # noqa: E402
from PIL import Image, ImageOps  # noqa: E402

OUT = Path(__file__).resolve().parent
SOURCE = OUT / "source"
TEASER = OUT / "fig_teaser_granularity.svg"
TEASER_SHA256 = "bcbc94423f3082c82a88f15b7d267ab681e13a36c77058ba7560aedcda0737dc"
FRAMES = (("start", 0.0), ("flight", 80.0), ("contact", 120.0), ("settled", 553.38))

INK = "#18252d"
GREY = "#637382"
LEVEL = {"cont": "#0072B2", "micro": "#CC79A7", "macro": "#D55E00"}
DELTA_GREY = {1: "#B0B0B0", 5: "#707070", 15: "#303030"}  # light -> dark = short -> long

W, H = 5.5, 2.6  # inches, final size (full text width)


def extract_frames():
    """Decode the teaser's four embedded rasters and store them as source PNGs."""
    data = TEASER.read_bytes()
    assert hashlib.sha256(data).hexdigest() == TEASER_SHA256, "teaser SVG changed"
    found = re.findall(rb'<image xlink:href="data:image/png;base64,\s*([^"]+)"([^>]*)>', data)
    assert len(found) == len(FRAMES)
    frames = {}
    for (payload, attrs), (name, _) in zip(found, FRAMES):
        assert b"scale(1 -1)" in attrs
        image = Image.open(io.BytesIO(base64.b64decode(re.sub(rb"\s", b"", payload))))
        image = ImageOps.flip(image.convert("RGB"))
        assert image.size[1] == 96 and image.size[0] in (247, 248)
        image.save(SOURCE / f"interface_frame_{name}.png")
        frames[name] = image
    return frames


def build():
    frames = extract_frames()
    fig = plt.figure(figsize=(W, H), facecolor="white")

    def P(x, y):
        return (x / W, y / H)

    def text(x, y, s, size=7, color=INK, **kw):
        fig.text(*P(x, y), s, fontsize=size, color=color, **kw)

    def line(xs, ys, color=GREY, lw=0.6, **kw):
        fig.add_artist(plt.Line2D([x / W for x in xs], [y / H for y in ys],
                                  transform=fig.transFigure, color=color, lw=lw,
                                  solid_capstyle="butt", **kw))

    def arrow(p0, p1, color=INK, lw=0.7, style="-|>", ms=5, **kw):
        fig.add_artist(FancyArrowPatch(P(*p0), P(*p1), transform=fig.transFigure,
                                       arrowstyle=style, mutation_scale=ms,
                                       color=color, lw=lw, shrinkA=0, shrinkB=0, **kw))

    def hop(x0, x1, y, height, color, lw=0.9, head=False):
        """Upper half-ellipse from (x0, y) to (x1, y): one query of horizon x1 - x0."""
        fig.add_artist(Arc(P((x0 + x1) / 2, y), (x1 - x0) / W, 2 * height / H,
                           theta1=0, theta2=180, transform=fig.transFigure,
                           color=color, lw=lw))
        if head:
            arrow((x1, y + 0.03), (x1, y), color=color, lw=lw, ms=4.5)

    def letter(x, y, s):
        text(x, y, s, size=8, weight="bold", va="top")

    # ---------------- (a) real launch on the carrier-frame timeline -------------
    letter(0.0, H - 0.01, "(a)")
    x_left, tile_w, gap = 0.30, 1.22, 0.07
    tile_h = tile_w * 96 / 247
    tile_y = H - 0.63
    axis_y, k_max = tile_y - 0.18, 560.0
    x_axis1 = x_left + 4 * tile_w + 3 * gap

    def kx(k):
        return x_left + (x_axis1 - x_left) * k / k_max

    labels = ("launch $a$", "flight", "first contact", "settled")
    for i, ((name, k), label) in enumerate(zip(FRAMES, labels)):
        x = x_left + i * (tile_w + gap)
        ax = fig.add_axes((*P(x, tile_y), tile_w / W, tile_h / H))
        ax.imshow(frames[name], interpolation="nearest", aspect="auto")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(GREY)
            spine.set_linewidth(0.5)
        text(x + tile_w / 2, tile_y + tile_h + 0.02, label, ha="center", va="bottom")
        line([x + tile_w / 2, kx(k)], [tile_y, axis_y + 0.03], lw=0.5)
    line([x_left, x_axis1], [axis_y, axis_y], color=INK, lw=0.7)
    for (_, k), tick in zip(FRAMES, ("0", "80", "120", "553")):
        line([kx(k), kx(k)], [axis_y - 0.035, axis_y + 0.035], color=INK, lw=0.7)
        text(kx(k), axis_y - 0.05, tick, ha="center", va="top")
    # Scored rollout endpoint (t = 225 carrier frames; contract H3/§5 tie-free cost).
    line([kx(225), kx(225)], [axis_y - 0.035, axis_y + 0.035], color=INK, lw=0.7)
    text(kx(225), axis_y - 0.05, "225: scored rollout endpoint", ha="center", va="top")
    # Axis title sits in the leader-free gap above the axis; time base below it.
    unit_x = (kx(225) + kx(553.38)) / 2 + 0.35
    text(unit_x, axis_y + 0.03, "carrier frame $k$", ha="center", va="bottom")
    text(unit_x, axis_y - 0.05, "1 frame = 50 engine steps = 0.02 s", ha="center", va="top")
    # One launch at k = 0, then no action until the scene settles.
    bracket_y = axis_y - 0.24
    line([kx(0), kx(0), kx(553.38), kx(553.38)],
         [bracket_y + 0.05, bracket_y, bracket_y, bracket_y + 0.05], color=GREY, lw=0.6)
    text((kx(0) + kx(553.38)) / 2, bracket_y, "no action between launch and settling",
         ha="center", va="center", color=GREY, style="italic",
         bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none"))

    # ---------------- (b) request r = (Delta, alpha) -------------------------------
    top_b = bracket_y - 0.13
    letter(0.0, top_b, "(b)")
    text(0.30, top_b, r"request $r=(\Delta,\alpha)$", size=7.5, va="top")
    # Horizon jumps on a 15-frame segment.
    hx0, scale = 0.30, 1.80 / 15
    hy = top_b - 0.33
    line([hx0, hx0 + 15 * scale], [hy, hy], color=INK, lw=0.6)
    for j in range(16):
        tall = j in (0, 1, 5, 15)
        line([hx0 + j * scale] * 2, [hy - (0.03 if tall else 0.015), hy], color=INK, lw=0.5)
    for j, tick in ((0, "$k$"), (1, "+1"), (5, "+5"), (15, "+15")):
        text(hx0 + j * scale, hy - 0.05, tick, ha="center", va="top")
    text(hx0 + 10 * scale, hy - 0.05, "carrier frames", ha="center", va="top", color=GREY)
    for delta, height in ((1, 0.06), (5, 0.12), (15, 0.20)):
        hop(hx0, hx0 + delta * scale, hy, height, DELTA_GREY[delta], lw=1.1, head=True)
    text(hx0 + 9.0 * scale, hy + 0.035, r"$\Delta\in\{1,5,15\}$", ha="center", va="bottom")
    # Description levels: cumulative content (wording fixed by contract R3-1).
    rows = (("cont", "continuous", ("object presence, kind,", "position, velocity")),
            ("micro", "micro", ("+ which objects touch", "or support others")),
            ("macro", "macro", ("+ scene predicates:", "steady-state (at rest),",
                                "structure unstable")))
    pitch, y = 0.105, hy - 0.19
    for key, name, gloss in rows:
        line([0.30, 0.30], [y + 0.01, y - pitch * len(gloss) + 0.02], color=LEVEL[key], lw=2.2)
        text(0.37, y, name, color=LEVEL[key], weight="bold", va="top")
        for n, g in enumerate(gloss):
            text(1.02, y - pitch * n, g, va="top")
        y -= pitch * len(gloss) + 0.05
    assert y > -0.02, "panel (b) overflows the canvas"

    # ---------------- (c) fixed vs per-query requests (schematic) ---------------
    cx = 2.58
    letter(cx, top_b, "(c)")
    text(W - 0.04, top_b, "schematic, 45-frame segment", va="top", ha="right",
         color=GREY, style="italic")
    tx0, tx1 = 3.56, W - 0.04
    total = 45
    unit = (tx1 - tx0) / total
    y_fine, y_coarse, y_query = top_b - 0.33, top_b - 0.68, top_b - 1.05

    def track(y, schedule, label):
        line([tx0, tx1], [y, y], color=GREY, lw=0.5)
        k = 0
        for delta, key in schedule:
            hop(tx0 + k * unit, tx0 + (k + delta) * unit, y,
                0.03 + 0.008 * delta, LEVEL[key], lw=0.9)
            k += delta
        assert k == total
        text(tx0, y + 0.03 + 0.008 * max(d for d, _ in schedule) + 0.03, label, va="bottom")

    track(y_fine, [(1, "cont")] * total, r"fixed: every query $r=(1,\mathsf{continuous})$")
    track(y_coarse, [(15, "cont")] * 3, r"fixed: every query $r=(15,\mathsf{continuous})$")
    track(y_query, [(1, "cont"), (5, "micro"), (15, "macro"), (1, "micro"), (5, "cont"),
                    (15, "cont"), (1, "cont"), (1, "macro"), (1, "cont")],
          r"per query: $r_k=(\Delta_k,\alpha_k)$")
    text(tx0, y_query - 0.05, "span = Δ carrier frames, colour = α", va="top", color=GREY)

    box_x0, box_x1 = cx + 0.02, tx0 - 0.20
    box_y0, box_y1 = y_coarse - 0.14, y_coarse + 0.14
    fig.add_artist(FancyBboxPatch(P(box_x0, box_y0), (box_x1 - box_x0) / W,
                                  (box_y1 - box_y0) / H, transform=fig.transFigure,
                                  boxstyle="round,pad=0,rounding_size=0.012",
                                  fc="#F2F4F5", ec=INK, lw=0.7))
    text((box_x0 + box_x1) / 2, (box_y0 + box_y1) / 2, r"$F_\theta(\hat z_k, a, r)$",
         ha="center", va="center")
    text((box_x0 + box_x1) / 2, box_y0 - 0.04, "one shared\npredictor",
         ha="center", va="top", linespacing=1.1)
    for y in (y_fine, y_coarse, y_query):
        arrow((box_x1, y_coarse), (tx0 - 0.03, y), color=INK, lw=0.6, ms=5)
    with plt.rc_context({"svg.fonttype": "none", "pdf.fonttype": 42}):
        for ext in ("svg", "pdf", "png"):
            fig.savefig(OUT / f"fig_interface.{ext}", dpi=600 if ext == "png" else 300)
    plt.close(fig)


if __name__ == "__main__":
    plt.rcParams.update({"font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans",
                         "svg.hashsalt": "fig_interface"})
    SOURCE.mkdir(exist_ok=True)
    build()
