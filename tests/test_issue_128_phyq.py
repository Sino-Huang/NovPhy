"""CPU tests for the #128 A3 PhyQ-RelDQN adapter (world_model/baselines/phyq/novphy.py)."""
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from world_model.baselines.phyq import novphy as nv

ROOT = Path(__file__).resolve().parents[1]
FIT = ROOT / ".local-artifacts/issue-104-power-cohort-campaign-v1/issue109-fit/attempts"
ANCHOR = FIT / "issue-109-sc-fit-type010103-001-c00"
GRID = [(x, y) for x in (-40, -80, -120, -160) for y in (5, 30, 55, 80)]  # eb.grid_items ordinal order
CHANNEL = {"slingshot": 0, "redBird": 1, "pig": 6, "wood": 7, "ice": 8, "stone": 9, "platform": 11}

needs_fit = pytest.mark.skipif(not ANCHOR.is_dir(), reason="#109 fit captures not present")


def launch4(x, y):
    return [x / 480., y / 480., 1.0, 0.0]


# ---------------------------------------------------------------- state reconstruction (A3-sym)

def test_decision_step_is_frame_zero():
    from scripts import run_decision_chain_attribution as m99
    assert nv.DECISION_FIXED_STEP == m99.DECISION_FIXED_STEP


@needs_fit
def test_symbolic_state_of_one_fit_level_matches_the_engine_objects():
    physics = json.loads((ANCHOR / "decision-physics-v1.json").read_text())
    state = nv.symbolic_state(ANCHOR)
    assert state.dtype == torch.uint8 and tuple(state.shape) == (12, 120, 160)
    assert set(state.unique().tolist()) <= {0, 1}
    kinds = [n["object_type"] for n in physics["nodes"]]
    material = {"bird": "redBird", "pig": "pig", "wood": "wood", "ice": "ice", "stone": "stone",
                "Slingshot": "slingshot"}
    assert "Platform" in kinds
    present = {material[k.split("_")[0]] for k in kinds if k not in ("Ground", "Platform")}
    for name, channel in CHANNEL.items():
        assert bool(state[channel].any()) == (name in present), name
    # dev-mode pipeline: the reader drops "Platform" features, so channel 11 is empty (P-fid 2026-10-28, repair 1)
    assert not state[[2, 3, 4, 5, 10, 11]].any()  # no yellow/blue/white/black birds, no TNT, no platforms
    # the pig occupies the reader's cells of its contour's bounding box (official linspace binning)
    pig = next(n for n in physics["nodes"] if n["object_type"].startswith("pig"))
    xs = [p["x"] for p in pig["screen_polygon"]]
    ys = [p["y"] for p in pig["screen_polygon"]]
    x_range, y_range = np.linspace(0, 640, 160), np.linspace(0, 480, 120)

    def slot(edges, value):
        return max(i for i in range(len(edges) - 1) if edges[i] < value <= edges[i + 1])
    box = state[6, slot(y_range, min(ys)):slot(y_range, max(ys)) + 1, slot(x_range, min(xs)):slot(x_range, max(xs)) + 1]
    assert box.numel() > 0 and bool(box.all())


@needs_fit
def test_gt_json_uses_dev_labels_engine_colour_maps_and_the_official_classifier():
    physics = json.loads((ANCHOR / "decision-physics-v1.json").read_text())
    _, colormaps = nv.engine_colormaps(ANCHOR)
    gt = nv.gt_json(physics, colormaps)
    features = gt[0]["features"]
    assert [f["properties"]["label"] for f in features] == [n["object_type"] for n in physics["nodes"]]
    platform = next(f for f in features if f["properties"]["label"] == "Platform")
    assert [e["color"] for e in platform["properties"]["colormap"]] == [int(e["x"]) for e in colormaps["effects_21"]]
    # the reader's colour classifier recovers the material of every object it keeps (Platform is dropped)
    model, target_class = nv.reader_model()
    objects = [(f, n) for f, n in zip(features, physics["nodes"])
               if f["properties"]["label"] not in ("Slingshot", "Ground", "Platform")]
    matrix = np.zeros((256, len(objects)))
    for j, (feature, _) in enumerate(objects):
        for entry in feature["properties"]["colormap"]:
            matrix[entry["color"], j] = entry["percent"]
    predicted = [target_class[i] for i in (model[:, :1] + model[:, 1:] @ matrix).argmax(0)]
    truth = {"bird": "bird_red", "pig": "pig_basic_small"}
    for (_, node), label in zip(objects, predicted):
        kind = node["object_type"].split("_")[0]
        assert label == truth.get(kind, kind)


@needs_fit
def test_dev_labels_differ_from_non_dev_only_in_the_platform_channel():
    from world_model.baselines.phyq.vendor.StateReader.SymbolicStateReader import SymbolicStateReader
    physics = json.loads((ANCHOR / "decision-physics-v1.json").read_text())
    gt = nv.gt_json(physics, nv.engine_colormaps(ANCHOR)[1])
    non_dev = json.loads(json.dumps(gt))
    for feature in non_dev[0]["features"]:
        if feature["properties"]["label"] not in ("Slingshot", "Ground"):
            feature["properties"]["label"] = "Object"
    model, target_class = nv.reader_model()
    with nv.numpy_float_alias():
        full = SymbolicStateReader(non_dev, model, target_class).get_symbolic_image(h=120, w=160)
    dev = nv.symbolic_image(gt).numpy()
    assert full[11].any() and not dev[11].any()
    assert np.array_equal(full[:11], dev[:11])


def test_gt_json_rejects_a_non_decision_state():
    with pytest.raises(ValueError):
        nv.gt_json({"fixed_step": 30050, "coordinates": {"screen_origin": "top_left"}, "nodes": []}, {})


@needs_fit
def test_symbolic_store_matches_the_reader_when_built():
    path = nv.symbolic_store_path("fit")
    if not path.is_file():
        pytest.skip("symbolic fit store not built")
    store = nv.load_symbolic_store("fit")
    assert len(store["states"]) == 480
    assert torch.equal(store["states"]["issue-109-sc-fit-type010103-001"], nv.symbolic_state(ANCHOR))


# ---------------------------------------------------------------- action mapping

def test_grid_launches_map_to_the_nearest_native_action():
    indices = nv.action_indices(torch.tensor([launch4(x, y) for x, y in GRID])).tolist()
    assert indices == [97, 127, 144, 153, 94, 111, 125, 135, 92, 104, 115, 124, 92, 101, 109, 117]
    # 15 distinct actions: g08 (-120, 5) and g12 (-160, 5) are 0.6 deg apart and share Phy-Q action 92
    assert len(set(indices)) == 15 and indices[8] == indices[12]
    for (x, y), a in zip(GRID, indices):
        # Phy-Q's release offset for action a (DQNDiscreteAgent.__degToShot, screen y up) vs ours (screen y down)
        deg = a + 90
        dx, dy = 200 * math.cos(math.radians(deg)), 200 * math.sin(math.radians(deg))
        angle_ours = math.degrees(math.atan2(-y, x)) % 360
        assert abs((math.degrees(math.atan2(dy, dx)) % 360) - angle_ours) <= 0.5 + 1e-9


def test_angle_and_offset_style_launches_are_injective_and_rightward_pulls_are_rejected():
    angle = [(-80, 10), (-80, 7), (-78, 17), (-76, 26), (-72, 35), (-67, 44), (-61, 51), (-55, 59), (-47, 65),
             (-39, 70), (-30, 74), (-21, 77), (-11, 79)]
    assert len(set(nv.action_indices(torch.tensor([launch4(x, y) for x, y in angle])).tolist())) == 13
    with pytest.raises(ValueError):
        nv.action_index(40, -5)


def test_grid_ties_leave_g_rank_passing_by_construction():
    from scripts import external_baselines as eb
    torch.manual_seed(0)
    q = torch.randn(5, 180)
    costs = -q[:, nv.action_indices(torch.tensor([launch4(x, y) for x, y in GRID]))]
    guard = eb.g_rank(costs)
    assert guard["pass"] and guard["fully_tied_level_share"] == 0.0
    assert guard["mean_tied_pair_share"] == pytest.approx(1 / 120)


# ---------------------------------------------------------------- loss on a toy batch

class FixedQ(nn.Module):
    input_type = "symbolic"

    def __init__(self, q, scale=1.0):
        super().__init__()
        self.q = nn.Parameter(torch.tensor(q, dtype=torch.float32))
        self.scale = scale

    def forward(self, x):
        return self.scale * self.q[: x.shape[0]] * x.flatten(1)[:, :1]


def toy_data(rewards, actions):
    n = len(rewards)
    return nv.PhyqData(inputs=torch.ones(n, 12, 120, 160, dtype=torch.uint8), obs=torch.arange(n),
                       actions=torch.tensor(actions), rewards=torch.tensor(rewards, dtype=torch.float32),
                       probabilities=np.full(n, 1.0 / n), levels=())


def test_train_step_is_huber_on_terminal_targets_with_clamped_gradients():
    q = [[0.2, 9.0], [9.0, 1.5], [-0.5, 9.0]]
    model = FixedQ(q)
    data = toy_data([0.0, 1.0, 1.0], [0, 1, 0])
    optimizer, scheduler = nv.make_optimizer(model, 10, learning_rate=0.0)
    out = nv.runner_train_step(model, optimizer, scheduler, data, [0, 1, 2])
    # SmoothL1 (beta 1): 0.5 * 0.2^2, 0.5 * 0.5^2, |-1.5| - 0.5; target = reward (bootstrap zeroed: all terminal)
    assert float(out["loss"]) == pytest.approx((0.02 + 0.125 + 1.0) / 3, rel=1e-6)
    assert float(model.q.grad.abs().max()) <= nv.OFFICIAL["grad_clamp"]
    big = FixedQ([[50.0, 0.0]] * 3, scale=100.0)  # dL/dq = 100 / 3 per row before the clamp
    optimizer, scheduler = nv.make_optimizer(big, 10, learning_rate=0.0)
    out = nv.runner_train_step(big, optimizer, scheduler, toy_data([0.0] * 3, [0] * 3), [0, 1, 2])
    assert float(out["grad_norm"]) == pytest.approx(100 / 3 * math.sqrt(3), rel=1e-5)
    assert big.q.grad[:, 0].tolist() == [1.0, 1.0, 1.0]


def test_upstream_long_index_bug_zeroes_rows_zero_and_one_not_terminal_rows():
    next_values = torch.full((4, 1), 5.0)
    is_done = [True, True, False, True]
    bugged = next_values.clone()
    bugged[torch.Tensor(is_done).long()] = 0  # DQNBase.train_model_memory
    fixed = next_values.clone()
    fixed[torch.tensor(is_done)] = 0.0  # DQNBase.train_model_loader (bool tensor from the dataset)
    assert bugged.view(-1).tolist() == [0.0, 0.0, 5.0, 5.0]
    assert fixed.view(-1).tolist() == [0.0, 0.0, 5.0, 0.0]


def test_outer_steps_reset_adamw_state_and_round_trip():
    model = FixedQ([[0.3, 0.1]] * 2)
    data = toy_data([1.0, 0.0], [0, 1])
    optimizer, scheduler = nv.make_optimizer(model, 4, learning_rate=1e-3, resets=[2])
    steps = []
    for _ in range(4):
        nv.runner_train_step(model, optimizer, scheduler, data, [0, 1])
        steps.append(float(optimizer.state[model.q]["step"]))
    assert steps == [1.0, 2.0, 1.0, 2.0]
    clone = nv.OuterSteps(optimizer)
    clone.load_state_dict(scheduler.state_dict())
    assert clone.state_dict() == {"update": 4, "resets": [2]}
    with pytest.raises(ValueError):
        nv.make_optimizer(model, 4, resets=[4])


def test_native_recipe_follows_the_official_outer_loop():
    assert nv.outer_epochs(10) == [32, 31, 29, 26, 22, 17, 11, 4, 1, 1]
    assert nv.official_outer_steps(480) == 96 and nv.official_outer_steps(80) == 16
    code = nv.recipe("native-code", 13919, 480)
    assert code["batches_per_epoch"] == 434 and sum(code["epochs"]) == 260 and code["updates"] == 112840
    assert len(code["resets"]) == 95 and code["resets"][:2] == [32 * 434, 63 * 434]
    assert nv.recipe("native-paper", 13919, 480)["updates"] == 96 * 10 * 434
    assert nv.recipe("native-template", 13919, 480)["updates"] == 180 * 434
    assert {code["batch_size"], code["learning_rate"]} == {32, 3e-4}


# ---------------------------------------------------------------- shapes, data, scoring

@pytest.mark.parametrize("arm,params", [("A3-sym", 11_662_208), ("A3-img", 11_633_920)])
def test_networks_have_the_official_shapes(arm, params):
    torch.manual_seed(0)
    model = nv.build_model(arm)
    assert nv.trainable_parameters(model) == params and model.N == 20
    out = model(torch.rand(2, model.ch_in, 120, 160))
    assert tuple(out.shape) == (2, 180)


def test_image_input_is_the_official_crop_of_the_bgr_screenshot():
    frame = torch.zeros(3, 240, 320, dtype=torch.uint8)
    frame[0] = 255  # pure red
    shot = nv.phyq_screenshot(frame)
    assert shot.shape == (3, 480, 640) and shot[2].min() == 255 and shot[0].max() == 0  # BGR
    state = nv.image_state(frame)
    assert tuple(state.shape) == (3, 120, 160) and float(state.min()) >= -1 and float(state.max()) <= 1
    outside, inside = frame.clone(), frame.clone()
    outside[:, :, 40:60] = 0  # screen x 80..120: outside the official centre crop (x 140..500 of 640)
    inside[:, :, 150:170] = 0  # screen x 300..340
    assert torch.equal(nv.image_state(outside), state)
    assert not torch.equal(nv.image_state(inside), state)


def test_trainset_batches_and_scoring():
    torch.manual_seed(0)
    members = ["a"] * 4 + ["b"] * 4
    fit = SimpleNamespace(members=members, verdicts=torch.tensor([1, 0, 0, 0, 0, 0, 0, 1], dtype=torch.bool),
                          launch4=torch.tensor([launch4(*GRID[i]) for i in range(8)]))
    anchors = {m: (torch.rand(12, 120, 160) > 0.97).to(torch.uint8) for m in "ab"}
    data = nv.runner_trainset("A3-sym", fit, anchors)
    assert tuple(data.inputs.shape) == (2, 12, 120, 160) and data.obs.tolist() == [0, 0, 0, 0, 1, 1, 1, 1]
    assert data.actions.tolist() == nv.action_indices(fit.launch4).tolist()
    assert data.probabilities.sum() == pytest.approx(1.0)
    assert data.probabilities[fit.verdicts.numpy()].sum() == pytest.approx(0.5)
    from scripts import external_baselines as eb
    batch = nv.runner_batch(data, eb.batch_rng(20260908, 0))
    assert batch.shape == (32,) and np.array_equal(batch, nv.runner_batch(data, eb.batch_rng(20260908, 0)))
    model = nv.build_model("A3-sym")
    optimizer, scheduler = nv.make_optimizer(model, 2)
    out = nv.runner_train_step(model, optimizer, scheduler, data, batch)
    assert torch.isfinite(out["loss"]) and model.training
    scores = nv.score_actions(model, anchors["a"], fit.launch4)
    assert scores.dtype == torch.float32 and tuple(scores.shape) == (8,) and model.training
    assert torch.equal(scores, nv.score_actions(model, anchors["a"], fit.launch4))
