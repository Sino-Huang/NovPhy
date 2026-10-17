"""Issue #106: frozen description definitions, pixel carrier builder and QA program resolution."""
import numpy as np
import torch

from world_model.data import pixel_benchmarks as pb
from world_model.training import pixel_carrier as pc


def synthetic_clevrer():
    frames = pb.CLEVRER_FRAMES
    visible = np.zeros((frames, 2), bool)
    visible[:, 0] = True
    visible[40:, 1] = True  # object 1 enters at frame 40
    velocity = np.zeros((frames, 2, 3), np.float32)
    velocity[:60, 0, 0] = 1.0  # object 0 moves until frame 59, then rests
    return {"slots": np.array([pb.clevrer_slot("red", "metal"), pb.clevrer_slot("blue", "rubber")]),
            "shapes": np.array([0, 2]), "visible": visible, "velocity": velocity,
            "location": np.zeros((frames, 2, 3), np.float32), "collisions": [(50, 0, 1)]}


def test_clevrer_micro_is_the_collision_frame():
    scene = synthetic_clevrer()
    contact, mask = pb.clevrer_micro(scene)
    a, b = scene["slots"]
    assert contact[50, a, b] and contact[50, b, a] and contact.sum() == 2
    assert mask[a, b] and not mask[a, a] and mask.sum() == 2


def test_clevrer_macro_enter_exit_and_rest():
    values, available = pb.clevrer_macro(synthetic_clevrer())
    j1, j15 = pb.DELTAS.index(1), pb.DELTAS.index(15)
    # debounced in-view set changes at frame 41 (2 of frames 39..41); Delta = 1 sees it only there
    assert values[41, 0, j1] and not values[40, 0, j1] and not values[42, 0, j1]
    # Delta = 15 compares f with f - 15: positive on 41..55
    assert values[41:56, 0, j15].all() and not values[56, 0, j15]
    assert not available[1, 0, j1] and available[3, 0, j1]
    # at rest from frame 60 on: Delta = 15 needs 15 resting frames (f >= 74)
    assert values[60, 1, j1] and not values[59, 1, j1]
    assert values[74, 1, j15] and not values[73, 1, j15]


def synthetic_physion(native=60):
    roles = np.array([0, 1, 2, 3])
    contacts = np.zeros((native, 4, 4), bool)
    contacts[10:13, 2, 3] = contacts[10:13, 3, 2] = True
    velocity = np.full((native, 4), 0.5, np.float32)
    velocity[30:] = 0.0
    zone = np.zeros(native, bool)
    zone[20:] = True
    return {"roles": roles, "native_frames": native, "push_time": 0, "velocity": velocity,
            "angular": np.zeros((native, 4), np.float32), "contact_zone": zone, "contacts": contacts}


def test_physion_micro_window_and_absorption():
    trial = synthetic_physion()
    contact, mask = pb.physion_micro(trial, 30)
    # native 10..12 fall in carrier frames 4 (native 10..12) only
    assert contact[4, 2, 3] and not contact[3, 2, 3] and not contact[5, 2, 3]
    assert mask[:4, :4].sum() == 12 and not mask[4:].any()


def test_physion_macro_ocp_onset_and_settled():
    trial = synthetic_physion()
    values, available = pb.physion_macro(trial, 40)
    j1, j5 = pb.DELTAS.index(1), pb.DELTAS.index(5)
    # contact begins at native 20 = carrier frame 7 (native 19..21)
    assert values[7, 0, j1] and not values[6, 0, j1] and not values[8, 0, j1]
    assert values[7:12, 0, j5].all() and not values[12, 0, j5]
    # at rest from native 30: Delta = 1 settled at frame f iff native 3f-2..3f at rest -> f >= 11
    assert values[11, 1, j1] and not values[10, 1, j1]
    # past the end (native 59) the last frame is absorbed: still settled
    assert values[30, 1, j5] and available[30, 1, j5]


def test_carrier_values_layout():
    presence = torch.tensor([[0.9] + [0.0] * 17, [0.8] + [0.0] * 17])
    centers = torch.zeros(2, 18, 2)
    centers[0, 0] = torch.tensor([0.2, 0.4])
    centers[1, 0] = torch.tensor([0.3, 0.4])
    kinds = torch.zeros(2, 18, 3)
    kinds[..., 2] = 1.0
    z = pc.carrier_values(presence, centers, kinds, 0.5)
    assert z.shape == (2, 236)
    slot = z[1, 2:15]
    assert torch.allclose(slot[[0, 2, 5, 6, 7]], torch.tensor([0.8, 1.0, 0.3, 0.4, 1.0]))
    assert torch.allclose(slot[8:11], torch.tensor([0.2, 0.0, 1.0]))  # motion (0.1 / 0.5)
    assert z[0, 0] == 0 and z[1, 0] == 1 and abs(float(z[1, 1]) - 0.5) < 1e-7
    assert z[0, 2 + 13 + 7] == 0  # absent slot: no center


def test_resolve_objects():
    attributes = [{"slot": 2, "color": "red", "material": "rubber", "shape": "cube"},
                  {"slot": 5, "color": "blue", "material": "metal", "shape": "sphere"},
                  {"slot": 7, "color": "blue", "material": "metal", "shape": "cube"}]
    program = ["all_events", "objects", "red", "filter_color", "unique", "filter_collision", "objects", "blue",
               "filter_color", "sphere", "filter_shape", "unique", "filter_collision", "unique"]
    assert pc.resolve_objects(program, attributes) == [2, 5]
    ambiguous = ["objects", "blue", "filter_color", "unique"]
    assert pc.resolve_objects(ambiguous, attributes) == [None]


def test_grouped_bootstrap_point_is_pooled_mean():
    values = np.array([[1, 0, 1, 1], [0, 0, 1, 1]], float)
    stat = pc.clustered_bootstrap(values, groups=[0, 0, 1, 2], draws=200)
    assert abs(stat["mean"] - 0.625) < 1e-12 and stat["clusters"] == 3
