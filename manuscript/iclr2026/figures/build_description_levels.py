"""Build a definitional illustration from real N1 engine states/training targets.

Run with the NovPhy Python environment:
    python figures/build_description_levels.py

Panel contract (all panels use the identical, unretouched RGB crop):
  a: cont / engine object centres and velocity / blue markers and arrows;
  b: micro / retained contact/supports training targets / violet edges;
  c: macro / retained scene-predicate targets / orange Boolean badges.
This is NOT a model-accuracy figure. The initially requested frozen-parser
readouts on the teaser's observed frame 80 miss the bird/pig and have no
presence/availability-gated relation above 0.5. The author-approved fallback
uses engine states and the actual labels consumed by N1 training, explicitly
identified in the caption. No model probability is substituted by a label.

Evidence: run_issue_77_n1_train.py:198-304 binds shard windows to observations;
run_issue_70_parser_repair.py:177-215 defines projection and label ordering.
The selected branch is training-role, predictor-partition, the same authored
scene/launch as the teaser, but its original N1 training capture. Frame 165 is
window 25, offset 0; settled frame 554 is window 27, offset 60. Relations/macros
come directly from the retained shard, not a new manual labelling pass.
Only the seven objects inside the common tower crop are displayed. Native
engine velocities are projected with the same camera matrices as positions.
Read-only input root: /p/Project/NovPhy. Outputs: only this script's PDF/PNG.
"""
from pathlib import Path
import gzip
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
import numpy as np
from PIL import Image
import torch

ROOT = Path("/p/Project/NovPhy")
OUT = Path(__file__).resolve().parent
DYNAMICS = ROOT / ".local-artifacts/issue-77-n1-dynamics-v1"
SHARD = DYNAMICS / "shards/lineage-n1-001.pt"
BRANCH = "issue-77-n1-001-a06"
SHOT = ROOT / ".local-artifacts/issue-77-n1-v1/attempts" / BRANCH / "shot-1"
FRAME, SETTLED_FRAME = 165, 554
CROP = (185, 160, 355, 280)
BLUE, VIOLET, ORANGE, INK = "#0072B2", "#CC79A7", "#D55E00", "#18252d"
VELOCITY_SECONDS = 0.5


def evidence():
    plan = json.loads((DYNAMICS / "plan.json").read_text())
    shard = torch.load(SHARD, map_location="cpu", weights_only=True)
    vocabulary = plan["contract"]["vocabulary"]
    assert len(vocabulary) == 18 and shard["contract"] == plan["contract"]
    assert shard["record"]["exposure_role"] == "training"
    assert shard["record"]["fit_partition"] == "predictor"
    assert shard["tensors"]["z"].shape[-1] == 236
    shot = shard["record"]["branches"].index(BRANCH)

    def target(frame):
        matches = [(i, frame - w["start"]) for i, w in enumerate(shard["windows"])
                   if w["shot"] == shot and 0 <= frame - w["start"] <= 60]
        assert len(matches) == 1
        i, offset = matches[0]
        t = shard["tensors"]
        assert bool(t["macros_mask"][i, offset].all())
        relations = t["relations"][i, offset].numpy()
        assert relations.shape == (18, 18, 2)
        assert not bool((t["relations"][i, offset] &
                         ~t["relations_mask"][i, offset]).any())
        return relations, t["macros"][i, offset].numpy().astype(int), (i, offset)

    relations, macros, source_index = target(FRAME)
    _, settled, settled_index = target(SETTLED_FRAME)
    observation_root = SHOT / "observation-trace"
    frames = json.loads((observation_root / "observation_trace_manifest.json")
                        .read_text())["frame_records"]
    record = frames[FRAME]
    assert record["fixed_step"] == 30000 + 50 * FRAME
    image_path = observation_root / record["agent_observation"]["relative_path"]
    with Image.open(image_path) as opened:
        assert opened.size == (640, 480)
        image = opened.convert("RGB")

    segment = json.loads((SHOT / "segment.json").read_text())
    native_root = Path(segment["native_root"])
    native = json.loads((native_root / "native-manifest.json").read_text())
    assert native["capture_id"] == segment["capture_id"]
    assert native["frame_records"][FRAME]["fixed_step"] == record["fixed_step"]
    chunk_info = next(c for c in native["chunks"]
                      if c["first_fixed_step"] <= record["fixed_step"] <= c["last_fixed_step"])
    with gzip.open(native_root / chunk_info["path"], "rt") as stream:
        chunk = json.load(stream)
    sample = next(s for s in chunk["fixed_step_samples"]
                  if s["fixed_step"] == record["fixed_step"])
    transform = record["capture_metadata"]["world_to_observation_transform"]
    camera = (np.asarray(transform["camera_to_clip_matrix"]).reshape(4, 4)
              @ np.asarray(transform["world_to_camera_matrix"]).reshape(4, 4))
    screen = np.asarray(transform["ndc_to_observation_matrix"]).reshape(3, 3)

    def project(position):
        clip = camera @ np.array([*position, 0, 1])
        return (screen @ np.array([*(clip[:2] / clip[3]), 1]))[:2]

    xy, velocity = {}, {}
    for entity in sample["entities"]:
        name = entity["scenario_object_id"]
        if name not in vocabulary or entity["lifecycle"] != "active" or not entity["body_present"]:
            continue
        i = vocabulary.index(name)
        body = entity["body"]
        xy[i] = project(body["position"])
        velocity[i] = project(np.asarray(body["position"]) + body["velocity"]) - xy[i]
    shown = [i for i, (x, y) in xy.items()
             if CROP[0] < x < CROP[2] and CROP[1] < y < CROP[3]]
    assert shown == [0, 3, 8, 9, 10, 11, 12]
    contacts = [(i, j) for i in shown for j in shown if i < j and relations[i, j, 0]]
    supports = [(i, j) for i in shown for j in shown if relations[i, j, 1]]
    assert contacts == [(0, 11), (3, 10), (8, 9)]
    assert supports == [(9, 8), (10, 3), (11, 0)]
    assert np.array_equal(relations[:, :, 0], relations[:, :, 0].T)
    report = {"image": str(image_path), "shard": str(SHARD), "frame": FRAME,
              "native_fixed_step": record["fixed_step"], "window_offset": source_index,
              "settled_frame": SETTLED_FRAME, "settled_window_offset": settled_index,
              "macros": macros.tolist(), "settled_macros": settled.tolist(),
              "contacts": [[vocabulary[i], vocabulary[j], 1] for i, j in contacts],
              "supports": [[vocabulary[i], vocabulary[j], 1] for i, j in supports],
              "positions_pixels": {vocabulary[i]: xy[i].tolist() for i in shown},
              "velocities_pixels_per_second": {vocabulary[i]: velocity[i].tolist() for i in shown}}
    print(json.dumps(report, indent=2))
    return image, xy, velocity, shown, contacts, supports, macros, settled


def draw():
    image, xy, velocity, shown, contacts, supports, macros, settled = evidence()
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 7,
                         "mathtext.fontset": "dejavusans", "pdf.fonttype": 42,
                         "svg.fonttype": "none"}):
        fig = plt.figure(figsize=(5.5, 1.6), facecolor="white")
        axes = []
        for left, title, color in zip((0.005, 0.342, 0.679),
                                      ("(a) cont: object state", "(b) micro: relations", "(c) macro: scene state"),
                                      (BLUE, VIOLET, ORANGE)):
            ax = fig.add_axes((left, 0.17, 0.316, 0.735))
            ax.imshow(image, interpolation="nearest")
            ax.set(xlim=CROP[::2], ylim=(CROP[3], CROP[1]))
            ax.set_axis_off()
            fig.text(left, 0.988, title, fontsize=7.5, fontweight="bold", color=INK, va="top")
            fig.add_artist(plt.Line2D((left, left + 0.316), (0.91, 0.91),
                                     transform=fig.transFigure, color=color, lw=1.1))
            axes.append(ax)

        cont, micro, macro = axes
        for i in shown:
            cont.plot(*xy[i], marker="+", ms=4, mew=0.8, color=BLUE)
            # All visible native velocities are rendered at one physical scale;
            # zero vectors have no arrow. Sub-pixel vectors are not exaggerated.
            end = xy[i] + VELOCITY_SECONDS * velocity[i]
            if np.linalg.norm(end - xy[i]) >= 1:
                cont.add_patch(FancyArrowPatch(xy[i], end, arrowstyle="-|>",
                                              mutation_scale=7, lw=0.9, color=BLUE))
        for i, label, text_xy in ((0, "bird", (190, 195)), (3, "block", (257, 207)),
                                  (8, "pig", (303, 241))):
            cont.annotate(label, xy=xy[i], xytext=text_xy, fontsize=7, color=INK,
                          arrowprops={"arrowstyle": "-", "color": BLUE, "lw": 0.6},
                          bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 0.4})

        for i, j in contacts:
            micro.plot(*np.array([xy[i], xy[j]]).T, color=VIOLET, lw=1.3)
        for i, j in supports:
            micro.add_patch(FancyArrowPatch(xy[i], xy[j], arrowstyle="-|>",
                                            connectionstyle="arc3,rad=0.42", shrinkA=1,
                                            shrinkB=1, mutation_scale=7, lw=1.0, color=INK))
        for i in shown:
            micro.plot(*xy[i], marker="o", ms=2.4, markerfacecolor="white",
                       markeredgecolor=VIOLET, markeredgewidth=0.7)

        # A uniform white veil discloses the macro panel's dimming and leaves
        # the same source crop visible behind the scene-level Boolean labels.
        macro.add_patch(Rectangle((CROP[0], CROP[1]), CROP[2] - CROP[0], CROP[3] - CROP[1],
                                  facecolor="white", alpha=0.83, edgecolor="none"))
        for name, value, y in zip(("steady-state", "structure-unstable"), macros, (0.67, 0.34)):
            macro.text(0.04, y + 0.1, name, transform=macro.transAxes, fontsize=7, color=INK)
            macro.text(0.94, y + 0.1, str(value), transform=macro.transAxes,
                       fontsize=8, fontweight="bold", color=INK, ha="right")
            macro.add_patch(Rectangle((0.04, y - 0.01), 0.90, 0.055,
                                      transform=macro.transAxes, facecolor="white", edgecolor=ORANGE, lw=0.8))
            if value:
                macro.add_patch(Rectangle((0.04, y - 0.01), 0.90, 0.055,
                                          transform=macro.transAxes, facecolor=ORANGE, edgecolor="none"))
        fig.text(0.005, 0.102, r"centres + velocity ($0.5\,$s)", fontsize=7, color=INK)
        fig.text(0.005, 0.025, f"observed frame {FRAME}", fontsize=7, color=INK)
        fig.add_artist(plt.Line2D((0.343, 0.368), (0.122, 0.122),
                                 transform=fig.transFigure, color=VIOLET, lw=1.3))
        fig.text(0.375, 0.102, "contact", fontsize=7, color=INK)
        fig.add_artist(FancyArrowPatch((0.494, 0.122), (0.525, 0.122), transform=fig.transFigure,
                                       arrowstyle="-|>", mutation_scale=7, color=INK, lw=1.0))
        fig.text(0.534, 0.102, "supports", fontsize=7, color=INK)
        fig.text(0.343, 0.025, "positive engine targets", fontsize=7, color=INK)
        fig.text(0.679, 0.102, "Boolean training targets", fontsize=7, color=INK)
        fig.text(0.679, 0.025, f"settled frame {SETTLED_FRAME}: ({settled[0]}, {settled[1]})", fontsize=7, color=INK)
        for suffix in ("pdf", "png"):
            fig.savefig(OUT / f"fig_description_levels.{suffix}", dpi=300,
                        metadata={"CreationDate": None} if suffix == "pdf" else None)
        plt.close(fig)


if __name__ == "__main__":
    draw()
